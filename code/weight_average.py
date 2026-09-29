"""Build a single-model submission by averaging two existing slim submissions
element-wise on the parameter tensors. The resulting checkpoint is itself
a StudentGPT (no ensemble inference needed), so the evaluation cost is the
same as a single model.

This is sometimes called "model soup" and is a cheap form of ensembling
that works best when the two trajectories end in similar loss basins.

Usage:
    python weight_average.py --inputs a.pt b.pt [--weights 0.5 0.5] --output out.pt
"""
import argparse
import json
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--weights', nargs='+', type=float, default=None,
                   help='Per-input averaging weights (default: uniform).')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    if args.weights is None:
        weights = [1. / len(args.inputs)] * len(args.inputs)
    else:
        assert len(args.weights) == len(args.inputs)
        s = sum(args.weights)
        weights = [w / s for w in args.weights]

    states = []
    base_config = None
    for path in args.inputs:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        states.append(ckpt['model'])
        if base_config is None:
            base_config = dict(ckpt['config'])
        else:
            # Architectures must match (vocab, width, depth, heads, context,
            # dropout, tied/untied). Calibration fields like temperature are
            # allowed to differ -- they are picked from validation after the
            # soup is formed.
            for k in ('vocab', 'width', 'depth', 'heads', 'context', 'dropout', 'tied'):
                if base_config.get(k) != ckpt['config'].get(k):
                    raise SystemExit(f'Architecture field {k} differs: '
                                     f'{base_config.get(k)} vs {ckpt["config"].get(k)}')

    # Average tensors element-wise. Buffers (none, since inputs are slim) are
    # just copied from the first.
    avg = {}
    for key in states[0]:
        tensors = [s[key] for s in states]
        if tensors[0].dtype.is_floating_point:
            avg[key] = sum(w * t for w, t in zip(weights, tensors))
        else:
            avg[key] = tensors[0]  # integers (e.g., counters) — keep first

    ckpt_out = {
        'model': avg,
        'config': base_config,
        'implementation': 'student',
    }
    torch.save(ckpt_out, args.output)
    import os
    size_mib = os.path.getsize(args.output) / (1024 * 1024)
    print(json.dumps({
        'inputs': args.inputs,
        'weights': weights,
        'output': args.output,
        'mib': round(size_mib, 4),
    }, indent=2))


if __name__ == '__main__':
    main()