"""Distill the two-model ensemble into a single student model.

The teacher (runs/ensemble.pt) was trained only on the supplied training
text; its soft targets therefore leak no external information. The student
uses the same architecture/regularization as the final single model and the
same data pipeline as train.py. Loss = mean of soft cross-entropy against
teacher probabilities mixed with the ordinary hard-label cross-entropy:

    loss = (1 - alpha) * CE(hard) + alpha * KL(teacher || student)

Saves a checkpoint compatible with evaluate.py (implementation 'student').
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', nargs='+', type=Path,
                   default=[ROOT / 'runs/ensemble.pt', ROOT / 'runs/big-d15-20k/checkpoint.pt'],
                   help='One or more teacher checkpoints (geometric mixture of their log-probs).')
    p.add_argument('--teacher-weights', nargs='+', type=float, default=[1., .5],
                   help='Non-negative mixture weights, one per teacher.')
    p.add_argument('--input-noise', type=float, default=.1,
                   help='Noisy-student token replacement rate for the student inputs (teachers see clean inputs).')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/student.json')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/distill')
    p.add_argument('--device', default='cuda')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--alpha', type=float, default=0.7,
                   help='Weight of the soft-target KL term.')
    p.add_argument('--eval-every', type=int, default=4000)
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    student, implementation_sha = make_model('student', config, device)
    if len(args.teacher) != len(args.teacher_weights):
        p.error('Provide one weight per teacher.')
    weight_total = float(sum(args.teacher_weights))
    teacher_weights = [w / weight_total for w in args.teacher_weights]
    teachers = []
    for path in args.teacher:
        teacher_ckpt = torch.load(path, map_location='cpu', weights_only=True)
        teacher, _ = make_model(teacher_ckpt['implementation'], teacher_ckpt['config'], device)
        teacher.load_state_dict(teacher_ckpt['model'])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers.append(teacher)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history, validation_history = [], []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        learning_rate = args.lr * min(1., (step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        # Noisy student: the student sees token-replaced inputs, teachers see the clean ones.
        noisy = inputs
        if args.input_noise > 0.:
            replaced = torch.rand(inputs.shape, device=device) < args.input_noise
            random_tokens = torch.randint(0, config['vocab'], inputs.shape, device=device)
            noisy = torch.where(replaced, random_tokens, inputs)
        with torch.no_grad(), autocast(device, precision):
            mixed = None
            for weight, teacher in zip(teacher_weights, teachers):
                logp = teacher.predict_log_probs(inputs).float()
                mixed = weight * logp if mixed is None else mixed + weight * logp
            teacher_probs = F.log_softmax(mixed, dim=-1).exp()
        with autocast(device, precision):
            logits = student(noisy).flatten(0, 1).float()
            hard = F.cross_entropy(logits, targets.flatten())
            soft = F.kl_div(F.log_softmax(logits, dim=-1), teacher_probs.flatten(0, 1),
                            reduction='batchmean')
            loss = (1. - args.alpha) * hard + args.alpha * soft
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        if hasattr(student, 'update_ema'):
            student.update_ema()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(), 'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            intermediate = score(student, *data['validation'], device, 'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step': step + 1, **intermediate})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds
    validation = score(student, *data['validation'], device, 'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': 'student', 'config': config,
                'model': student.cpu().state_dict(), 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256,
                'teachers': [str(t) for t in args.teacher], 'teacher_weights': teacher_weights,
                'input_noise': args.input_noise, 'alpha': args.alpha}, checkpoint)
    result = {'protocol': PROTOCOL, 'implementation': 'student', 'config': config,
              'seed': args.seed, 'alpha': args.alpha, 'input_noise': args.input_noise,
              'teachers': [str(t) for t in args.teacher], 'teacher_weights': teacher_weights,
              'parameters': sum(p_.numel() for p_ in student.parameters()),
              'precision': precision, 'train_tokens': args.steps * args.batch_size * 256,
              'preparation_seconds': preparation_seconds, 'train_seconds': train_seconds,
              'validation': validation, 'history': history,
              'validation_history': validation_history,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint), 'implementation_sha256': implementation_sha,
              **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()
