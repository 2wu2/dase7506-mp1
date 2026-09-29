"""Measure E2 ensemble timing across threads."""
import time
import torch
from common import setup, load_data, make_model
from evaluate import score

ckpt = torch.load('runs/ens-E2-d3-s18/submission.pt', map_location='cpu', weights_only=True)
data = load_data()
for threads in (4, 8):
    dev, prec = setup('cpu', 'fp32', threads)
    m, _ = make_model(ckpt['implementation'], ckpt['config'], dev)
    m.load_state_dict(ckpt['model'])
    t0 = time.perf_counter()
    r = score(m, *data['test'], dev, prec)
    dt = time.perf_counter() - t0
    print(f'E2 threads={threads} test_bpb={r["bpb"]:.4f} seconds={dt:.2f}')