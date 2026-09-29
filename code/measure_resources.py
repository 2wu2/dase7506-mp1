"""Resource measurement for the frozen predictor (README section 4).

Measures, for one frozen checkpoint:
  * CPU FP32 full-test scoring time (limit: 5x baseline)
  * peak process working set during scoring (limit: 4 GiB)
  * uncompressed inference asset size = checkpoint file size (limit: 64 MiB)

Usage: python measure_resources.py --checkpoint runs/main-20k/checkpoint.pt
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
from pathlib import Path
import time
import torch
from common import ROOT, load_data, make_model, setup
from evaluate import score


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def peak_working_set_bytes():
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    kernel32 = ctypes.WinDLL('kernel32')
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return counters.PeakWorkingSetSize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    device, precision = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    data = load_data()
    asset_bytes = args.checkpoint.stat().st_size
    result = score(model, *data['test'], device, 'fp32')
    result.pop('window_nll_nats')
    report = {
        'checkpoint': str(args.checkpoint),
        'parameters': sum(p_.numel() for p_ in model.parameters()),
        'asset_bytes': asset_bytes,
        'asset_mib': round(asset_bytes / 2**20, 2),
        'score_seconds_cpu_fp32': result['seconds'],
        'peak_working_set_gib': round(peak_working_set_bytes() / 2**30, 2),
        'test_bpb': result['bpb'],
        'threads': args.threads,
        'torch_version': str(torch.__version__),
        'os': os.name,
    }
    print(json.dumps(report, indent=2))
    (args.checkpoint.parent / 'resources.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
