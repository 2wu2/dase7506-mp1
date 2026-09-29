"""Idle-machine timing ratio: baseline vs candidate, both 4 threads, official scorer."""
import time
import torch
from common import setup, load_data, make_model
from evaluate import score

data = load_data()
dev, prec = setup('cpu', 'fp32', 4)

for name, path in [('baseline', 'runs/baseline-s17/checkpoint.pt'),
                   ('E2-ensemble', 'runs/ens-E2-d3-s18/submission.pt'),
                   ('distill-e2', 'runs/distill-e2/submission.pt')]:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    m, _ = make_model(ckpt['implementation'], ckpt['config'], dev)
    m.load_state_dict(ckpt['model'])
    # warm-up (first batch pays lazy init costs)
    score(m, *data['validation'], dev, prec)
    t0 = time.perf_counter()
    r = score(m, *data['test'], dev, prec)
    dt = time.perf_counter() - t0
    print(f'{name:12s} test_bpb={r["bpb"]:.4f} seconds={dt:.2f}')