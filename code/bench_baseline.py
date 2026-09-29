"""Quick baseline timing comparison across thread counts."""
import time
import torch
from common import setup, load_data, make_model
from evaluate import score

ckpt = torch.load('runs/baseline-s17/checkpoint.pt', map_location='cpu', weights_only=True)
data = load_data()
for threads in (1, 2, 4, 8):
    dev, prec = setup('cpu', 'fp32', threads)
    m, _ = make_model(ckpt['implementation'], ckpt['config'], dev)
    m.load_state_dict(ckpt['model'])
    t0 = time.perf_counter()
    r = score(m, *data['test'], dev, prec)
    dt = time.perf_counter() - t0
    print(f'threads={threads} baseline_test_bpb={r["bpb"]:.4f} seconds={dt:.2f} budget_5x={dt*5:.2f}')