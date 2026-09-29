"""List all available checkpoints and their validation BPB / sizes."""
import json
import os
from pathlib import Path

print('=== checkpoint / submission files ===')
for f in sorted(Path('runs').glob('*/*.pt')):
    size_mib = f.stat().st_size / (1024 * 1024)
    metrics = f.parent / 'metrics.json'
    val = ''
    if metrics.exists():
        m = json.load(open(metrics))
        if 'validation' in m:
            val = f"val={m['validation']['bpb']:.4f}"
    print(f'  {f.parent.name + "/" + f.name:42s} {size_mib:7.2f} MiB  {val}')