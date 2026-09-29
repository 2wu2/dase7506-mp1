"""Bake a temperature (or other config field) into a slim submission checkpoint."""
import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--temperature', type=float, required=True)
p.add_argument('--output', required=True)
args = p.parse_args()

ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
ckpt['config'] = dict(ckpt['config'])
ckpt['config']['temperature'] = args.temperature
torch.save(ckpt, args.output)
print(f'baked temperature={args.temperature} -> {args.output}')