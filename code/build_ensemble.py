"""Compose a slim ensemble submission from existing single-model submissions.

Usage:
    python build_ensemble.py --members path1/submission.pt path2/submission.pt \
                             --output runs/E2/submission.pt

Each member must be a slim EMA-baked submission.pt (already produced by
make_submission.py). The ensemble checkpoint stores every member's parameters
under ``members.<i>.<param>`` keys, so the inference asset is the sum of the
member sizes and the inference time is the sum of the member times.

The ensemble uses geometric-mean averaging of per-token distributions
(arithmetic mean of log-probabilities in natural units), implemented in
student.EnsembleGPT.predict_log_probs.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--members', nargs='+', required=True,
                   help='Paths to slim submission.pt files.')
    p.add_argument('--output', required=True,
                   help='Output ensemble submission path.')
    p.add_argument('--temperature', type=float, default=1.,
                   help='Optional temperature applied to every member.')
    args = p.parse_args()

    # Load member sub-configs.
    sub_configs = []
    for path in args.members:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if ckpt['implementation'] != 'student':
            raise SystemExit(f'{path} is not a student submission.')
        sub_cfg = dict(ckpt['config'])
        sub_cfg.pop('temperature', None)
        sub_configs.append(sub_cfg)
    cfg = {'members': sub_configs, 'temperature': args.temperature,
           'context': 256, 'vocab': 2048}

    # Build model and load each member with the right prefix.
    from student import build_model
    model = build_model(cfg)
    new_state = {}
    for i, path in enumerate(args.members):
        sd = torch.load(path, map_location='cpu', weights_only=True)['model']
        for key, value in sd.items():
            new_state[f'members.{i}.{key}'] = value
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f'state_dict mismatch: missing={missing} unexpected={unexpected}')

    # Save the ensemble submission. Note: we store the state_dict as the
    # 'model' key so evaluate.py's torch.load(... weights_only=True)
    # ckpt['model'] = model.state_dict() loads cleanly.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out = {
        'protocol': '7506-mp1-wt2-v2',
        'model': model.state_dict(),
        'config': cfg,
        'implementation': 'student',
    }
    torch.save(ckpt_out, out_path)
    size_mib = out_path.stat().st_size / (1024 * 1024)

    # Report.
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        'members': [Path(m).name for m in args.members],
        'output': str(out_path),
        'bytes': out_path.stat().st_size,
        'mib': round(size_mib, 4),
        'total_params': n_params,
        'asset_budget_mib': 64,
        'time_budget_s': 5 * 7.95,
    }, indent=2))


if __name__ == '__main__':
    main()