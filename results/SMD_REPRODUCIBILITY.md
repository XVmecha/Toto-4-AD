# SMD Anomaly-Detection Reproducibility

This document indexes every artifact behind the SMD results and explains how to regenerate
them. All analysis after the error-scoring step runs from cached scores and does **not** require
the model or a GPU.

## The three-stage detector

A forecasting model is turned into an anomaly detector in three stages: (1) **error scoring** —
per-variate negative log-likelihood (NLL) of each observation under the model's predicted
distribution; (2) **error aggregation** — combine the 38 per-variate errors into one anomaly
score per timestep; (3) **thresholding** — convert the score into a binary label. Only stage 1
requires the model; stages 2–3 operate on the cached stage-1 output.

## Configuration (identical across all SMD numbers)

- Model: `Datadog/toto-open-base-1.0`
- Context length: 512
- Calibration (fit) stride: 32 — used only to sample the normal-score distribution for thresholds
- Evaluation (detect) stride: 8 — a 1/8 temporal subsample, used for **every** SMD metric so all
  numbers are mutually comparable
- Label alignment: score index `i` corresponds to target timestep `t = 512 + i * detect_stride`
  (the model needs 512 steps of history before it can score a timestep, and the window advances by
  `detect_stride`). Earlier code aligned scores to raw indices `0..T_out`, which decoupled each
  score from its true timestep; this is fixed in `compute_auroc.py` and used consistently here.

## Cached scores (`toto/results/scores/`)

Raw per-variate NLL, shape `(28 machines, 38 variates, T)`. Produced once by the model; reused by
all downstream analysis.

| File | Stride | Purpose |
|---|---|---|
| `smd_nll_train_stride32.npy` | 32 | calibration (normal) errors — thresholds & per-variate standardization baseline |
| `smd_nll_test_stride8.npy` | 8 | evaluation errors — all detection/AUROC numbers |

## Scripts → outputs

| Script | Runs model? | Output(s) |
|---|---|---|
| `compute_per_machine_auroc.py` | yes (caches scores) | `toto/results/auroc/smd_{mean,max}_per_machine_auroc.json` — per-machine threshold, per-machine AUROC, **macro-average** (and pooled, for reference) |
| `analyze_smd_generalization.py` | no | `toto/results/auroc/smd_per_machine_detectability.json` — per-machine AUROC under raw vs per-variate-standardized aggregation, plus the label-informed best-single-variate upper bound |
| `analyze_smd_localization.py` | no | `toto/results/aggregation_analysis/smd_localization.json`, `toto/results/auroc/smd_event_detectability.json` — does error land on the genuinely-affected variates; are anomalies noticed |

## How to reproduce

```bash
# stage 1 (model) — produces cached scores + per-machine AUROC; slow, ~25 min on M-series MPS
python compute_per_machine_auroc.py

# stages 2–3 analysis (no model) — instant, read the cached scores
python analyze_smd_generalization.py
python analyze_smd_localization.py
```

## Headline numbers (macro-average over 28 machines)

| Aggregation | AUROC | Source |
|---|---|---|
| raw mean (current) | 0.48 | `smd_per_machine_detectability.json` |
| raw max (current) | 0.30 | " |
| robust per-variate normalized (median/IQR) mean | 0.69 | " |
| robust per-variate normalized (median/IQR) max | 0.73 | " |
| best single variate per machine | 0.93 | " (upper bound, label-informed, not zero-shot) |

Per-machine robust-max detectability: 16/28 > 0.7, 24/28 > 0.6, 2/28 < 0.45.
Per-event: ~79% of anomalies raise error on an affected variate (peak > 1σ); ~21% are unnoticed.

## Standardization baselines (documented design choice, not an inconsistency)

- **Detector** (`analyze_smd_generalization.py`): standardizes by **calibration-normal** per-variate
  statistics — the deployment-realistic, zero-shot-legal reference.
- **Diagnostic** (`analyze_smd_localization.py`): standardizes by **test-period normal** per-variate
  statistics — compares anomaly vs normal within the same period, isolating localization from any
  calibration/evaluation distribution shift.

## Known caveats

- The best-single-variate AUROC is an **upper bound**: the dimension is chosen using the labels, so
  it is not achievable in a true zero-shot setting. It only shows the error-scoring signal exists.
- The per-variate-standardized detector is a **post-hoc** method, devised after observing the
  failure; it is zero-shot-legal but was not pre-registered.
- Evaluation uses a 1/8 temporal subsample (stride 8); 77 short events contain no scored timestep
  and are excluded from the per-event "noticed" count.
- The cached NLL was spot-checked against fresh single-window forward passes at several timesteps
  (including the first and last): the cache matched to 0.0, confirming both faithfulness and the
  `t = 512 + i*detect_stride` alignment (`verify_scorer.py`).
