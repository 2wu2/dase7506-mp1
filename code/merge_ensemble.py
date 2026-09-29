"""Merge two slim (EMA-baked) checkpoints into one ensemble checkpoint.

Usage:
  python merge_ensemble.py --checkpoints runs/main-20k/submission.pt runs/main-20k-s18/submission.pt \
      --temperature 1.15 --output runs/ensemble.pt
"""
import argparse
import json
from pathlib import Path
import torch
from common import PROTOCOL, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', required=True, nargs=2, type=Path)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    members = []
    state = {'mixture_weights': torch.tensor([1. / len(args.checkpoints)] * len(args.checkpoints))}
    train_tokens = 0
    for i, path in enumerate(args.checkpoints):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        cfg = dict(ckpt['config'])
        cfg.pop('ema_decay', None)
        cfg.setdefault('temperature', args.temperature)
        members.append(cfg)
        for key, value in ckpt['model'].items():
            state[f'members.{i}.{key}'] = value
        train_tokens += ckpt['train_tokens']
    config = {'members': members, 'weights': [1.] * len(members),
              'temperature': args.temperature, 'vocab': 2048}
    out = {'protocol': PROTOCOL, 'implementation': 'student_ensemble',
           'config': config, 'model': state,
           'seed': [int(torch.load(c, map_location='cpu', weights_only=True)['seed']) for c in args.checkpoints],
           'train_tokens': train_tokens,
           'merged_from': [c.name for c in args.checkpoints],
           'source_sha256': [sha(c) for c in args.checkpoints]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(json.dumps({'output': str(args.output),
                      'bytes': args.output.stat().st_size,
                      'mib': round(args.output.stat().st_size / 2**20, 2)}, indent=2))


if __name__ == '__main__':
    main()
