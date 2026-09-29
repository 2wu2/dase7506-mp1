"""Validation-only temperature calibration for a slim submission checkpoint."""
import argparse
import torch
from common import load_data, make_model

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--temperatures', nargs='+', type=float,
              default=[0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1])
args = p.parse_args()

ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
data = load_data()
tokens, byte_count = data['validation']
# A fixed probe set of windows for speed: every 4th window batch.
probe_x, probe_y = [], []
for i, (x, y) in enumerate(
        __import__('common').windows(tokens, 32)):
    if i % 4 == 0:
        probe_x.append(x)
        probe_y.append(y)
x = torch.cat(probe_x)
y = torch.cat(probe_y)
print(f'probe windows: {x.shape[0]} (of full validation)')

import math
best = (None, 1e9)
for t in args.temperatures:
    cfg = dict(ckpt['config'])
    cfg['temperature'] = t
    cfg['ema_decay'] = 0.
    m, _ = make_model(ckpt['implementation'], cfg, torch.device('cpu'))
    m.load_state_dict(ckpt['model'])
    m.eval()
    with torch.no_grad():
        logp = m.predict_log_probs(x)
        losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        losses = losses.masked_fill(y == -100, 0)
        nll = losses.double().sum().item()
        n = (y != -100).sum().item()
    bpb = nll / math.log(2) / byte_count
    print(f'temperature={t:.2f} probe_val_bpb={bpb:.5f}')
    if bpb < best[1]:
        best = (t, bpb)
print(f'BEST temperature: {best[0]} (probe val bpb {best[1]:.5f})')