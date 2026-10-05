# Peer Review Report — vectorBH6/HKU_DASE7506_MP1

**Reviewer**: 2wu2 (DASE7506) · **Date**: 2026-10-05
**Peer repository**: https://github.com/vectorBH6/HKU_DASE7506_MP1
**Peer's claimed full-test BPB**: 1.5107 (`student_final`, `checkpoint_best.pt`)

## Reproduced score (required)

| Item | Value |
|---|---:|
| **Reproduced full-test BPB (CPU, FP32, 4 threads)** | **1.5107** |
| Peer's claimed BPB | 1.5107 |
| **Difference** | **0.0000 — exact match** |
| Reproduced token perplexity | 23.53 |
| Reproduced scoring time (this machine, 4 threads) | 38.2 s |
| Reproduction command | `python evaluate.py --checkpoint runs/student_final/checkpoint_best.pt --device cpu --precision fp32 --split test` |

**Verdict: the reported score is CONFIRMED.** The checkpoint reproduces its claimed BPB exactly under the supplied evaluator with byte-exact data/tokenizer.

## Contract checks (additional)

- Causality (future-token perturbation invariance): **PASS**
- Output normalization (log-probs sum to 1 within 1e-3): **PASS**
- Determinism (repeat calls identical): **PASS**
- `evaluate.py` / `tokenizer.json`: identical to the official version apart from line endings (SHA differences are CRLF-only; content byte-comparison confirms equality). **No evaluator or tokenizer modification.**

## Reproduction notes (minor, none affecting the score)

1. The README says the checkpoint bundle is on the GitHub **Releases page, but no Releases exist**; the checkpoints are instead committed directly in the repository under `code/runs/`. Not a score issue, but the bundle link in the final submission should be checked.
2. `student_final.py` lives in `code/student_work/`, while the checkpoint's `implementation` field is `student_final`, so the module is not importable from `code/` as shipped — I copied it to `code/` root to run the evaluator. Adding a one-line note (or moving the file) would make the repo runnable out of the box.
3. The committed `data/tokenizer.json` has CRLF line endings (git autocrlf artifact), which fails the manifest SHA check; I substituted the byte-exact original tokenizer before running. Content is otherwise identical. A `.gitattributes` with `* -text` for `data/` would fix this for other reviewers.

## Resource-limit observation (for the instructor/leaderboard, not a discrepancy)

On this machine the peer's model scored 38.2 s ≈ 4.4× the locally measured baseline (8.66 s) — within the 5× rule on this hardware, consistent with the peer's own disclosure (34.18 s, 4.07× on their machine). However, against the reference Xeon Platinum 8457C baseline (5.92 s → 29.6 s ceiling) this timing would exceed the limit, as the peer's README already honestly notes ("limit status is hardware-relative"). Reproduced here for the record; no misconduct is implied.
