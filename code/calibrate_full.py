"""Full-validation temperature sweep for a slim submission checkpoint."""
import argparse
import torch
from common import load_data, make_model
from evaluate import score

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--temperatures', nargs='+', type=float,
              default=[1.0, 1.02, 1.04, 1.06, 1.08])
args = p.parse_args()

ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
data = load_data()
dev = torch.device('cpu')
best = (None, 1e9)
for t in args.temperatures:
    cfg = dict(ckpt['config'])
    cfg['temperature'] = t
    cfg['ema_decay'] = 0.
    m, _ = make_model(ckpt['implementation'], cfg, dev)
    m.load_state_dict(ckpt['model'])
    r = score(m, *data['validation'], dev, 'fp32')
    print(f'temperature={t:.2f} full_val_bpb={r["bpb"]:.5f}')
    if r['bpb'] < best[1]:
        best = (t, r['bpb'])
print(f'BEST temperature: {best[0]} (full val bpb {best[1]:.5f})')