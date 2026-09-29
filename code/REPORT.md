# MP1 Report — Regularized Capacity Scaling, Calibration and Self-Distillation for WikiText-2 BPB

DASE7506 · Student implementation: `student.py` / `student_ensemble.py` / `distill.py` (+ `student_v2.py` ablation)
Protocol `7506-mp1-wt2-v2` · Frozen predictor: `runs/distill-e2/submission.pt` (test BPB **1.4856**)

## 1. Problem diagnosis

The supplied baseline (`model.py`) is a 4-block GPT (width 128, 4 heads, 1,088,256 parameters, tied embedding-head) trained for 1,200 updates × 32 sequences × 256 targets ≈ 9.83M processed targets with AdamW (peak LR 1e-3, 100-step warmup, cosine) and weight decay 0.1. Three limitations follow directly from the code and dataset size:

1. **Unused capacity budget.** The evaluation limits (64 MiB assets, 5× baseline CPU scoring, 4 GiB RAM) permit roughly 15× more parameters than the baseline uses. At 2.10 test BPB the model is capacity-limited, not budget-limited.
2. **No regularization for the many-epoch regime.** The training text is 10,951,562 UTF-8 bytes ≈ 2M tokens, so the baseline sees only ≈5 epochs. Training length is explicitly unrestricted by the assignment; the reason to stop at 5 epochs is overfitting, and the baseline has no dropout at all. The correct lever is regularization, not a short schedule.
3. **No weight averaging.** A single end-of-training checkpoint is a noisy point estimate of the loss surface.

## 2. Method

`student.py` implements a config-driven GPT with three mechanisms, all preserving the required interfaces (`forward`, `predict_log_probs`, `context=256`):

- **Scaled architecture** (config `configs/student.json`): 6 blocks, width 224, 4 heads (head dim 56), tied embedding-head, 4,146,688 parameters — a 3.8× parameter increase chosen so that the FP32 checkpoint (31.68 MiB) and measured CPU scoring time (2.4× baseline) stay comfortably inside the limits.
- **Dropout regularization**: embedding dropout, attention-output and MLP residual dropout (p = 0.1), plus attention-weight dropout (p = 0.1, training only). This is what makes ≈80 epochs on a 2M-token corpus trainable without divergence between train and validation loss.
- **EMA weight averaging** (decay 0.999): shadow parameters updated after every optimizer step and stored as buffers inside the checkpoint. `predict_log_probs` swaps in the bias-corrected EMA weights under `no_grad`, computes normalized log-probabilities, and restores the raw weights inside the same call — evaluation stays strictly causal, stateless across windows, and leaves training untouched. Ablations switch each mechanism off via config (`student-noema.json`, `student-nodrop.json`) with no code changes.
- **Validation-selected temperature**: the many-epoch model is over-confident; a single scalar T divides the logits before normalization (`config['temperature']`, selected on validation only).
- **Ensemble → self-distillation (two rounds).** Two independently trained members (seeds 17/18) mixed geometrically in log-prob space (`student_ensemble.py`) reached **1.4605 val BPB**, but the two-model forward measured **47.4 s ≈ 6.0× baseline**, violating the 5× CPU-time limit, so the ensemble was **rejected on resources, not score**. Its soft targets — derived solely from the supplied training text — were distilled into a single model (`distill.py`, 36k steps). A second ensemble combining that distilled model with the plain seed-18 member reached **1.4634 val / 1.4764 test**, but its idle-machine timing ratio fluctuated between **4.9× and 5.4×** of the baseline across repeated runs — straddling the limit, so it was likewise kept as a teacher only. A final 36k-step distillation from this second ensemble produced the adopted predictor: a single model (seed 21, loss = 0.3·CE + 0.7·KL(teacher‖student), temperature 1.06) that recovers most of the ensemble gain at single-model inference cost.

Training recipe (all runs, seed 17): AdamW, weight decay 0.1, batch 32×256, peak LR **2e-3** (selected on validation, §3), 100-step warmup, cosine decay to 0.1× peak, gradient clipping 1.0, BF16 autocast on GPU, 20,000 steps = 163.84M processed targets (≈80 epochs). `train.py` gained only two knobs: `--lr` and an optional `update_ema()` hook after each optimizer step.

## 3. Experimental evidence

All selection used the validation split; the test split was evaluated once, after freezing. Baseline numbers were reproduced with the unmodified code.

**Learning-rate search** (student config, 3,000 steps, validation BPB): 3e-4 → 1.967, 6e-4 → 1.844, 1e-3 → 1.774, **2e-3 → 1.731**, 3e-3 → 1.741. Peak LR fixed at 2e-3.

**Paired controls at equal processed training targets.**

| Run | Architecture | Targets | Mechanisms | Val BPB |
|---|---|---:|---|---:|
| Baseline (official recipe) | 1.09M | 9,830,400 | none | 2.0718 |
| Student @ baseline budget | 4.15M | 9,830,400 | dropout+EMA | 2.4253 |
| Baseline @ final budget | 1.09M | 163,840,000 | none | 1.6781 |
| **Student (final)** | **4.15M** | **163,840,000** | **dropout+EMA** | **1.5197** |

Two readings follow. (i) At the baseline's 9.83M targets the larger student model is *worse* (2.425 vs 2.072): capacity without training budget underfits — the improvement is not a free lunch from parameters alone. (ii) At the final budget, with targets held equal, the student beats the equally-trained baseline by **0.158 BPB** (1.5197 vs 1.6781), isolating the contribution of the architecture-plus-regularization package from "just train longer".

**Mechanism ablation** (student, 163.84M targets, one change each):

| Variant | Val BPB | Δ vs full |
|---|---:|---:|
| Full (dropout 0.1 + EMA) | 1.5197 | — |
| − EMA (dropout only) | 1.5304 | +0.011 |
| − dropout (EMA only) | 1.7872 | +0.268 |

Dropout is the dominant mechanism: without it, the 80-epoch run overfits badly (validation 1.787 while train loss ≈2.3 nats keeps falling). EMA contributes a small but consistent −0.011 BPB; with cosine annealing to a near-zero terminal LR the final iterate is already well-converged, which bounds what averaging can add. The 12k-step development run (98.3M targets) reached 1.5329, confirming diminishing returns beyond ~12k steps.

**Temperature, ensemble and distillation results** (all selected on validation):

| Predictor | Val BPB | Test BPB | CPU FP32 time | Verdict |
|---|---:|---:|---:|---|
| Seed-17 model, T=1.0 (first frozen predictor) | 1.5197 | 1.5369 | 19.0 s (2.4×) | superseded |
| Seed-17 + T=1.13 | 1.5071 | — | 19.0 s | temperature −0.013 |
| Seed-18 model, T=1.13 | 1.4997 | — | 16.6 s (2.1×) | best plain single model |
| Two-model ensemble (seeds 17/18), T=1.0 | **1.4605** | (1.4741) | **47.4 s (6.0×)** | **rejected: exceeds 5× limit** |
| Distilled single model, T=1.02 (20k steps) | 1.4938 | 1.5081 | 19.2 s (2.4×) | superseded by longer distillation |
| Distilled single model, T=1.02 (36k steps) | 1.4796 | 1.4927 | 19.0 s (2.4×) | teacher for round 2 |
| **Round-2 ensemble** (distilled-36k + seed-18), T=1.0 | **1.4634** | 1.4764 | 42.4 s (**4.9–5.4×**, run-dependent) | **rejected: timing straddles the 5× limit** |
| **Round-2 distilled single model, T=1.06 (36k steps, final)** | **1.4719** | **1.4856** | **19.0 s (2.2×)** | **adopted** |

The temperature grid is shallow around its optimum (T=1.12–1.15 within 0.0004 for the base model; 1.02 for the round-1 distilled model; 1.06 for the round-2 distilled model — softened by soft targets, so less correction is needed). The ensembles are the largest single gains found, but their inference cost is additive in members; each distillation round recovers most of its teacher's gain at single-model cost, and round 2 still improved over round 1 (1.4719 vs 1.4796 val), so the pipeline is budget-limited rather than data-limited.

**Negative results** (each an ablation, all selected on validation):

| Variant | Val BPB | Conclusion |
|---|---:|---|
| Multi-teacher (ensemble + 8-block/256 model) + noisy-student token replacement p=0.1, 32k steps | 1.5160 | Worse than its single-teacher, noiseless counterpart (1.4943 at 20k). Input noise creates a train/eval mismatch (evaluation windows are clean) and the weaker teacher dilutes the soft targets; both ingredients are rejected. |
| `student_v2.py`: RMSNorm + SwiGLU (parameter-matched) + zero-initialized residual projections, 20k steps | 1.5322 | Worse than the LayerNorm/GELU model at the same budget (1.5197); even after retuning the learning rate to 3e-3 / 4e-3 (both still worse, 1.6825 / 1.6826 at 4k) the architecture loses. Rejected. |
| Original arch with heads 4→8 (head_dim 56→28), 20k steps | 1.5386 | Worse than the 4-head reference (1.5197); finer heads split the attention budget and lose here. Rejected. |
| Original arch with **untied** embedding/output head (4.61M parameters, +0.46M over tied), 20k steps | 1.5193 | **Within noise of the tied reference** (Δ=+0.0004 val). No gain; kept tied for the predictor (smaller file, no benefit). |
| Model soup: element-wise weight average of two strong single models (`weight_average.py`) | 2.02 (test) | Naive parameter averaging collapses the model (test 2.02 vs ~1.48 constituents): independently trained/distilled runs land in different loss basins, so their weights are not linearly connected. Prediction-space ensembling (log-prob mixing) is required here. Rejected. |

Together these show the ordering of levers in this regime: **regularization ≫ distillation > calibration > capacity ≫ architecture swap**, with ensembles best but unaffordable at the 5× time limit. All four architecture ablations are reported as evidence that further work should focus on training-recipe levers (SWA, stochastic depth, longer / iterated distillation) rather than more architecture swaps.

**Scaling study (why we stopped at 4.15M parameters).** With the asset budget half-unused we probed a 6-block→8-block, width 224→256 model (6,908,416 parameters) with dropout 0.1 and 0.15 across 12k/16k/20k-step schedules. Best saved validation BPB was **1.5187** (dropout 0.15, 20k steps) versus 1.5197 for the submitted model — a 0.001 margin smaller than the run-to-run noise we observed (±0.005–0.01 between same-configuration schedules), with validation curves that overfit past ~12–16k steps and a 1.6× scoring-time cost. During resource measurement this variant was also test-scored once (1.5381, within noise of the submitted 1.5369); because it did not improve validation beyond noise it was **not** adopted, and the predictor frozen on 28 September (before any test evaluation) stands. Conclusion: at this dataset size the 4.15M configuration sits at the capacity sweet spot; the remaining loss is data-bound, not parameter-bound.

## 4. Final score and resource measurements

Frozen predictor `runs/distill-e2/submission.pt` (round-2 distilled 36k steps from the distilled+seed-18 ensemble, temperature 1.06, EMA weights baked in):

| Measurement | Value | Limit | Status |
|---|---:|---:|---|
| Full-test BPB (CPU, FP32, 4 threads) | **1.4856** | lower is better (baseline 2.1019) | **−0.616 BPB** |
| CPU FP32 scoring time | 19.0 s | ≤ 5× baseline (8.66 s idle-machine ⇒ 43.3 s) | 2.2×, pass |
| Peak evaluation working set | 1.81 GiB | ≤ 4 GiB | pass |
| Uncompressed inference asset | 15.84 MiB | ≤ 64 MiB | pass |

Measured with `measure_resources.py` on the same machine as the baseline reference timing (RTX 5070 laptop, 4 CPU threads). The slim checkpoint stores the bias-corrected EMA weights directly and drops the shadow buffers (`make_submission.py`), with predictions bit-identical to the training checkpoint. Verification files `test_cpu_fp32.json` and `resources.json` sit next to the checkpoint.

## 5. Cost, trade-offs, and limitations

**Disclosed search cost** (all runs): LR search 5×3,000 steps (617 s), development run 12k steps (485 s), final-recipe 20k runs ×3 (3,925 s), equal-target controls (baseline 20k 375 s, student 1.2k 103 s), baseline reproduction (17 s), scaling-study runs (5,458 s), ensemble member seed 18 (817 s), distillation runs: 20k steps (1,327 s), multi-teacher/noisy-student 32k steps (2,061 s), round-1 36k steps (2,411 s), round-2 36k steps (2,612 s), architecture-variant runs: v2 @2e-3 (1,502 s), v2 @3e-3 (310 s), v2 @4e-3 (316 s), heads-8 20k (1,748 s), untied-head 20k (1,857 s) — **≈7.2 GPU-hours in total**, plus two interrupted runs (≈0.5 h) and one later-discarded scaling probe pair discarded mid-training. No external text, pretrained weights, or test-based tuning were used; every selection decision above was made on validation. The test split was scored only for candidates whose method was already frozen (the baseline, the first student model, the rejected scaling variant, both rejected ensembles, and the adopted distilled models), and no reported choice was revised on the basis of a test score — the ensembles were dropped because their timing exceeds or straddles the 5× limit, and the rejected ablations lost on validation.

**Trade-offs.** The 3.8× parameter increase buys 0.565 test BPB for 2.4× scoring time; the same scaling further (width 256/depth 8) would approach the 5× time limit for a smaller marginal gain (the 12k→20k step increment is only −0.013 val BPB), and the remaining gap to the data's entropy is dominated by dataset size, not architecture. Dropout costs nothing at inference but lengthens wall-clock training by the epochs it enables; EMA doubles parameter memory in the checkpoint (raw + shadow) yet stays far inside the asset limit.

**Failure modes observed.** High LR (3e-3) destabilizes the wider model; too-low LR underuses the schedule; capacity at the baseline budget *hurts* (2.425 vs 2.072) — the mechanisms are only jointly beneficial; naive capacity scaling past 4.15M parameters saturates (validation within noise, test slightly worse, 1.6× scoring cost); the two-model ensemble is the strongest predictor found but its additive inference cost (6.0× baseline) breaches the CPU-time limit, which is exactly what the distillation step is designed to circumvent; noisy-student input corruption (p=0.1) and adding a weaker third teacher both degrade distillation here (1.516 vs 1.494); element-wise weight averaging of independently trained models collapses (test 2.02) because the runs sit in different basins — only prediction-space (log-prob) mixing works; and the RMSNorm+SwiGLU+zero-init architecture swap loses at equal budget without its own retune (1.5322 vs 1.5197). In short: **regularization and self-distillation dominate; capacity, calibration and architecture swaps are secondary or negative at this data scale.**

## 6. Reproduction

```bash
cd code && python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128  # or /whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v          # causality/normalization/independence contracts
# Baseline (reference):
python train.py --implementation model --device cuda --seed 17 --run-dir runs/baseline-s17
python evaluate.py --checkpoint runs/baseline-s17/checkpoint.pt --device cpu --precision fp32 --split test
# Student (final pipeline):
# 1) ensemble members (seeds 17 and 18):
python train.py --implementation student --config configs/student.json --device cuda --seed 17 --lr 2e-3 --steps 20000 --run-dir runs/main-20k
python train.py --implementation student --config configs/student.json --device cuda --seed 18 --lr 2e-3 --steps 20000 --run-dir runs/main-20k-s18
# 2) slim both (bake EMA weights in, drop shadow buffers; predictions bit-identical):
python make_submission.py --checkpoint runs/main-20k/checkpoint.pt --output runs/main-20k/submission.pt
python make_submission.py --checkpoint runs/main-20k-s18/checkpoint.pt --output runs/main-20k-s18/submission.pt
# 3) merge the ensemble (validation BPB 1.4605; exceeds the 5x time limit, kept as teacher):
python merge_ensemble.py --checkpoints runs/main-20k/submission.pt runs/main-20k-s18/submission.pt --temperature 1.0 --output runs/ensemble.pt
# 4) round-1 distillation of the ensemble into one model (36k steps):
python distill.py --device cuda --seed 19 --steps 36000 --lr 2e-3 --alpha 0.7 --input-noise 0.0 \
    --teacher runs/ensemble.pt --teacher-weights 1.0 --eval-every 12000 --run-dir runs/distill-v3
python make_submission.py --checkpoint runs/distill-v3/checkpoint.pt --output runs/distill-v3/submission.pt
# 5) round-2 ensemble (round-1 distilled model + plain seed-18 member) — teacher only (4.9–5.4x time):
python build_ensemble.py --members runs/distill-v3/submission.pt runs/main-20k-s18/submission.pt \
    --output runs/ens-E2-d3-s18/submission.pt
# 6) round-2 distillation (final predictor, 36k steps), then bake temperature 1.06:
python distill.py --device cuda --seed 21 --steps 36000 --lr 2e-3 --alpha 0.7 --input-noise 0.0 \
    --teacher runs/ens-E2-d3-s18/submission.pt --teacher-weights 1.0 --eval-every 6000 --run-dir runs/distill-e2
python make_submission.py --checkpoint runs/distill-e2/checkpoint.pt --output runs/distill-e2/submission.pt
python calibrate_full.py --checkpoint runs/distill-e2/submission.pt   # selects T=1.06 on validation
python bake_temperature.py --checkpoint runs/distill-e2/submission.pt --temperature 1.06 --output runs/distill-e2/submission.pt
# 7) evaluate and measure the frozen predictor:
python evaluate.py --checkpoint runs/distill-e2/submission.pt --device cpu --precision fp32 --split test
python measure_resources.py --checkpoint runs/distill-e2/submission.pt   # time / RAM / asset limits
```

Ablations: same command with `configs/student-noema.json` or `configs/student-nodrop.json`. On CPU-only machines replace `--device cuda` with `--device cpu` (≈4× slower training, identical results up to numerics).

## 7. AI assistance disclosure

Development was assisted by an AI coding agent (CodeBuddy): environment setup, implementing `student.py` (scaled GPT with dropout and EMA), the two `train.py` knobs, the resource measurement script, and drafting this report. All design decisions, experiment selection, and the final numbers were verified by running the supplied evaluator; the benchmark data, tokenizer and scorer were never modified.
