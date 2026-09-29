"""Convert a trained checkpoint into a slim submission checkpoint.

The training checkpoint stores raw weights plus EMA shadow buffers (2x size).
The frozen predictor only ever evaluates the bias-corrected EMA weights, so a
submission checkpoint can bake them into the parameters and drop the buffers:
the slim config sets ema_decay=0, `predict_log_probs` then uses the raw (EMA)
weights directly. Prediction outputs are bit-identical to the original.

Usage: python make_submission.py --checkpoint runs/big-20k/checkpoint.pt --output runs/big-20k/submission.pt
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from common import PROTOCOL, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--temperature', type=float, default=1.,
                   help='Prediction temperature to store in the slim config (selected on validation).')
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = ckpt['model']
    if 'ema_step' not in state:
        raise SystemExit('Checkpoint has no EMA buffers; nothing to slim.')
    steps = int(state['ema_step'])
    if steps == 0:
        raise SystemExit('EMA was never updated; raw weights are the predictor.')
    decay = float(ckpt['config'].get('ema_decay', 0.))
    correction = 1. - decay ** steps
    # Buffer keys use underscores where parameter keys use dots; map through
    # the model's own named_parameters to get exact keys. The slim model is
    # built with ema_decay=0 (no buffers); its state_dict() re-creates tied
    # entries (e.g. head.weight sharing token.weight's storage) correctly.
    from common import make_model
    config = copy.deepcopy(ckpt['config'])
    config['ema_decay'] = 0.
    config['temperature'] = args.temperature
    model, _ = make_model(ckpt['implementation'], config, torch.device('cpu'))
    with torch.no_grad():
        for name, parameter in model.named_parameters():  # unique (tied) params
            parameter.copy_(state['ema_' + name.replace('.', '_')] / correction)
    slim = model.state_dict()
    out = {'protocol': PROTOCOL, 'implementation': ckpt['implementation'],
           'config': config, 'model': slim, 'seed': ckpt['seed'],
           'train_tokens': ckpt['train_tokens'],
           'slimmed_from': args.checkpoint.name,
           'source_sha256': sha(args.checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(json.dumps({'output': str(args.output),
                      'bytes': args.output.stat().st_size,
                      'mib': round(args.output.stat().st_size / 2**20, 2),
                      'ema_steps': steps}, indent=2))


if __name__ == '__main__':
    main()
