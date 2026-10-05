# Peer Review Report — Wenqi-Ge/DASE_7506_2026 (commit 74fef55)

**Reviewer**: 2wu2 (DASE7506) · **Date**: 2026-10-05
**Peer repository**: https://github.com/Wenqi-Ge/DASE_7506_2026/tree/74fef55c4949bb89881c8af13d7a4b0c3071b11d
**Peer's claimed full-test BPB**: 1.35949 (`C4-R4a-hybrid`, distilled student + training-split 5-gram mixture)

## Reproduced score (required)

| Item | Value |
|---|---:|
| **Reproduced full-test BPB (CPU, FP32, 4 threads)** | **1.35949** |
| Peer's claimed BPB | 1.35949 |
| **Difference** | **0.00000 — exact match** |
| Reproduced token perplexity | 17.15 |
| Reproduction command | `python evaluate.py --checkpoint runs/C4-R4a-hybrid/checkpoint.pt --device cpu --precision fp32 --split test` |

**Verdict: the reported score is CONFIRMED.** The checkpoint reproduces its claimed BPB exactly under the supplied evaluator with byte-exact data/tokenizer. Checkpoint SHA256 matches the bundle manifest (`0fda43c9...3714aa`).

## Contract checks (all pass)

- Causality (future-token perturbation invariance): **PASS** (error 0.0, per his own verification and my independent check)
- Output normalization: **PASS** (max error < 1e-3)
- Determinism: **PASS**
- **Cross-window statelessness**: **PASS** — interleaving other windows between repeated calls leaves predictions bit-identical; the 5-gram component reads only within-window context
- `evaluate.py` / `common.py`: content-identical to the official version (SHA differences are CRLF-only, a git checkout artifact; his manifest records the official hashes)
- Data files: content identical after line-ending normalization; I substituted byte-exact originals before scoring

## Resource measurements

| Limit | Value | Status |
|---|---|---|
| Inference asset | 55.36 MB (52.8 MiB) | ≤ 64 MiB — pass |
| Peak RAM | ~2.3 GiB | ≤ 4 GiB — pass |
| **CPU FP32 scoring time** | **61.4–68.1 s on this machine (7.1×–7.9× local baseline of 8.66 s)** | **exceeds 5× (43.3 s) on this hardware** |

The submitter's own measurement (validation split) shows 17.59 s vs 3.81 s baseline = **4.62× on his hardware** (passing), and his test-split JSON records 20.8 s. The hybrid's n-gram component scales differently with CPU, so **the time-limit status is hardware-relative**: passing on the submitter's machine (4.6–4.9×), failing on the reviewer's (7.1–7.9×). Against the reference Xeon baseline (5.92 s → 29.6 s ceiling) the outcome is uncertain but likely over. This is reported as an observation for the leaderboard, not a score discrepancy — the BPB itself reproduces exactly.

## Method-legality notes (transparently disclosed by the submitter)

1. **5-gram tables**: README and code state the tables are fit on the **training split only** — compliant with "learn only from the supplied training text".
2. **Teacher-sampled synthetic data**: the student's distillation corpus includes 11.8M teacher-generated tokens (3.3× the training split) derived from training-only teachers. The submitter explicitly discloses this as a gray area under a strict reading of the rules and provides the stricter-reading alternative (R3g, trained on training windows only, validation 1.3551, ≈0.006 behind). No external text is involved; the interpretation question is left to the instructor.
3. Teachers and ensembles exceeding the evaluation budget were used only for training and were distilled into the single budget-compliant student — same pattern the rules anticipate (distillation from self-trained teachers).

## Reproduction notes (minor)

1. Committed `data/` files have CRLF line endings (git autocrlf artifact) and fail the manifest SHA check on checkout; I substituted byte-exact originals before running. A `.gitattributes` (`* -text` for `data/`) would fix this for other reviewers.
2. Everything else ran out of the box after extracting the checkpoint bundle; the bundle's SHA256SUMS verified.
