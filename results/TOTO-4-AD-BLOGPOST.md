# Zero-Shot Anomaly Detection with TOTO: Where Does It Fail?

## Introduction

TOTO (Time Series Optimized Transformer for Observability) is a multivariate time series (MVTS) forecasting foundation model trained on one trillion observability data points. Like other MVTS foundation models, it takes a window of recent multi-channel history and predicts a probability distribution over each channel's next value, and because it is pretrained at scale it does this zero-shot, on systems it has never seen and with no task-specific training.

Using such a model for anomaly detection rests on a simple idea: observations that are improbable under the model deviate from normal system behaviour, and are therefore anomalies. This is the central premise of forecasting-based multivariate time series anomaly detection. Acting on it requires computing the forecasting error for each variate and then, non-trivially, deciding whether that multi-dimensional error profile constitutes an anomaly. Colloquially this takes three steps: score the prediction error, aggregate the per-variate errors into a single anomaly score, and threshold that score into a label. The model owns only the first step; the rest are practitioner choices.

This blogpost does two things. First, it asks whether TOTO can model the normal behaviour of two MVTS benchmarks, SWaT (industrial control) and SMD (server monitoring), zero-shot. Second, it evaluates the go-to anomaly detection methodology for forecasting models when applied to TOTO. We treat this as a failure analysis: when end-to-end detection is weak the cause can lie in any of the three steps, so we remove steps from the evaluation to localise the one responsible. A threshold-free metric (AUROC) takes thresholding out of the picture and measures scoring plus aggregation; inspecting the per-variate errors against per-dimension labels takes aggregation out and isolates the model's scoring.

What we want to show is that this pipeline is anything but a formality. Seemingly minor choices (how the per-variate errors are aggregated, whether they are normalised first, where the threshold sits) swing performance from near chance to strong, and none is best across datasets. TOTO's learned sense of normal transfers well; whether that signal survives into a correct label is decided by this pipeline.

## Problem Formulation

**Zero-shot anomaly detection** requires detecting anomalies in new datasets without using any labelled anomaly examples for training or threshold tuning. This imposes a critical constraint: we can only use normal (non-anomalous) data to calibrate our detection system.

Given a multivariate timeseries forecasting model like TOTO, we convert it into an anomaly detector through three steps:

1. **Error scoring**: turn the model's forecast into a per-variate error score $e_t^{(j)}$: the Negative Log-Likelihood (NLL) of the observation under TOTO's predicted distribution.
2. **Error aggregation**: collapse the per-variate errors at each timestep into a single anomaly score $s_t$ (an $\mathbb{R}^N\!\to\!\mathbb{R}$ compression; e.g. mean or max).
3. **Thresholding**: flag a timestep when $s_t$ exceeds a threshold $\tau$ fit on normal data only, $a_t=\mathbb{1}[s_t>\tau]$.

Because thresholding (step 3) hinges on a practitioner choice, much of our evaluation uses **AUROC**, a threshold-free metric. It measures how well the anomaly score $s_t$ ranks anomalous timesteps above normal ones, so it captures steps 1 and 2 (scoring and aggregation) while leaving step 3 out.

## Methodology

### Datasets

**SWaT (Secure Water Treatment)**: Industrial control system with 51 sensors monitoring a water treatment plant.
- Calibration: 7 days normal operation (496,800 timesteps)
- Evaluation: 4 days with 36 cyber-physical attacks (449,919 timesteps, ~12% anomalous)

**SMD (Server Machine Dataset)**: Server monitoring metrics from 28 independent machines, each with 38 performance metrics.
- Each machine treated as separate time series (~23,687 timesteps per machine, ~4.3% anomalous)
- Anomalies include hardware failures, configuration errors, resource exhaustion

### Preprocessing

This is exactly what we ran, not the full range of options the scripts expose.

**SWaT** (`preprocess_swat.py`): parse the raw Excel logs (normal and attack), fill missing values by linear interpolation, downsample **10x** (median for the 51 continuous sensors, mode for the binary actuators), and z-score each variate using calibration-normal statistics only. Output: `(1, 51, 49500)` calibration and `(1, 51, 44991)` evaluation.

**SMD** (`preprocess_smd.py`): load the 28 machines separately with their per-timestep labels and per-event interpretation labels (which variates are anomalous, and when). **No downsampling and no normalisation**: the data ships in `[0, 1]`, and each machine is truncated to the common length of 23,687 timesteps. Output: `(28, 38, 23687)` for both calibration and evaluation.

Both use calibration (normal) statistics only, never evaluation data.

### Unsupervised Threshold Selection

The core challenge in zero-shot detection is setting a threshold $\tau$ without access to anomaly data. Our approach:

1. **Calibration phase**: Process normal data through TOTO, compute anomaly scores $S_{\text{calib}} = [s_1, s_2, ..., s_N]$
2. **Threshold estimation**: Set threshold as 95th percentile of calibration scores:
   $$\tau = \text{percentile}_{95}(S_{\text{calib}})$$
3. **Detection phase**: Flag evaluation timesteps where $s_t > \tau$ as anomalous
4. **Scoring**: report precision, recall, F1 and accuracy of these flags against the ground-truth labels (Table 2)

**Rationale**: If TOTO has learned normal behaviour, calibration scores should be consistently low. The 95th percentile captures the upper bound of "normal variation" while allowing for 5% noise/outliers. Any evaluation score exceeding this bound indicates deviation from normality.

**Critical constraint**: We never use evaluation data (including its anomalies) for threshold selection. This aims to ensure generalisation to unseen/unknown anomalies.

### Threshold-Agnostic Evaluation: AUROC

While threshold-based metrics (Precision, Recall, F1) depend heavily on the chosen threshold, **AUROC (Area Under the Receiver Operating Characteristic curve)** provides a threshold-independent assessment of the model's ability to *rank* anomalies above normal data. The ROC curve is traced by evaluating the threshold across every value and plotting, at each one, the **true positive rate** (the fraction of anomalies flagged, y-axis) against the **false positive rate** (the fraction of normal timesteps flagged, x-axis); AUROC is the area under that curve, summarising ranking quality across all thresholds at once. It also has a direct probabilistic reading: an AUROC of $p$ is the probability that the detector gives a randomly chosen anomalous timestep a higher score than a randomly chosen normal one. An AUROC of 0.74, for instance, means a 74% chance of correctly distinguishing a random anomalous case from a random normal one.

**AUROC interpretation**:
- **1.0**: Perfect ranking - all anomalies scored higher than all normal instances
- **0.5**: Random ranking - model cannot distinguish anomalies from normal data
- **< 0.5**: Inverted ranking - model scores anomalies lower than normal

AUROC helps localise where detection fails. A high AUROC with low F1 points to a thresholding failure (fixable). An aggregated AUROC at or below 0.5 is ambiguous on its own: it can be a genuine ranking failure *or* an artefact of the aggregation step, so before concluding the model is blind we inspect the per-variate errors directly (which, for SMD, is exactly what reverses the verdict below).

**SMD is evaluated per machine.** The 28 machines are independent multivariate timeseries, each with its own error scale, so we calibrate a separate threshold on each machine's own normal data and compute a separate AUROC per machine, then report the average over machines (macro-average). This is the methodologically correct metric for an independent-machines dataset: a single global threshold, or a single ROC pooled across all machines (micro-average), would mix error score distributions that are not comparable across machines. We report the pooled value alongside the macro-average for reference; the two are close here, and the conclusion does not depend on the choice.

### Aggregation and per-variate normalisation

At each timestep TOTO gives one error per variate; a detector needs one score per timestep, so we combine the per-variate errors by **mean** ($s_t = \frac{1}{M}\sum_i e_t^{(i)}$, for broad multi-sensor anomalies) or **max** ($s_t = \max_i e_t^{(i)}$, for a single-sensor spike).

We apply each rule two ways, and this contrast is the variable we study:

- **Unnormalised**: aggregate the errors directly.
- **Per-variate normalised**: first rescale each variate's error by its own median and IQR on calibration-normal data (a robust z-score, still zero-shot), then aggregate. This is the scoring step used by purpose-built detectors such as GDN (Deng and Hooi, AAAI 2021), and is usually filed under preprocessing.

### Implementation Details

- Context length 512; error metric is the Negative Log-Likelihood (NLL) under TOTO's Student-T mixture.
- Detect stride 8 for both datasets, so unnormalised and normalised aggregation are compared on the same grid; calibration stride 32 for threshold fitting.
- Threshold: 95th percentile of calibration scores (one for SWaT, one per machine for SMD). AUROC is threshold-free, so the main result does not depend on it.
- AUROC is computed on the single SWaT series and macro-averaged over the 28 SMD machines. It is stride-robust: SMD agrees within about 0.01 between stride 8 and stride 32, and SWaT's unnormalised mean is 0.86 at full stride-1 versus 0.83 at stride 8.

## Results

Table 1 is the whole result: ranking quality (AUROC) for each dataset and aggregation, with and without per-variate normalisation.

### Table 1: Ranking (AUROC), Unnormalised vs Per-Variate Normalised

| Aggregation | SWaT | SMD |
|---|---|---|
| mean, unnormalised | 0.83 | 0.48 |
| mean, per-variate normalised | 0.78 | 0.69 |
| max, unnormalised | 0.79 | 0.30 |
| max, per-variate normalised | 0.74 | 0.73 |

*AUROC (threshold-free ranking; 0.5 = chance). SWaT: single series. SMD: macro-average over 28 machines. Detect stride 8 for both. Sources: `toto/results/auroc/smd_per_machine_detectability.json`, `toto/results/auroc/swat_aggregation_comparison.json`.*

Normalisation raises SMD (0.48 to 0.69 for mean, 0.30 to 0.73 for max) and lowers SWaT (0.83 to 0.78 for mean, 0.79 to 0.74 for max). Additionally, on SMD, one variate (of 38) produces the largest error at 59% of timesteps, and its mean error sits 2.0 standard deviations above the median variate's. (Computed by `analyze_error_concentration.py` from the cached calibration NLL.)

### Table 2: Detection at the Threshold (Max Aggregation)

Table 1 measures ranking. Table 2 gives the labelled detection performance once the 95th-percentile threshold is applied to the unnormalised max score.

| Dataset | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| SWaT | 11.8% | 97.2% | 21.0% | 12.5% |
| SMD | 1.7% | 3.2% | 1.8% | 87.3% |

*SWaT: single series, single threshold, detect stride 1. SMD: one threshold per machine, macro-averaged over 28 machines, detect stride 8. Sources: `toto/results/swat_max/swat_detection_results.json`, `toto/results/auroc/smd_max_per_machine_auroc.json`.*

## Discussion

A weak end-to-end result does not by itself show that TOTO fails to model normal behaviour. Three components sit downstream of the forecast: the error metric, the aggregation and the threshold, and a loss of anomaly signal at any of them lowers the final F1. Because we evaluate the steps separately, we can attribute each dataset's failure to the step responsible.

First, TOTO's representation of normal behaviour transfers across domains. On SWaT it ranks anomalies at 0.83 AUROC (0.86 at full temporal resolution), even though SWaT is an industrial control system, a domain outside TOTO's observability training data of server and application telemetry. SMD is server monitoring and therefore closer to that training distribution, and ranks at 0.73. The out-of-domain result on SWaT is the stronger evidence that the learned notion of normal generalises.

Second, the steps after the model, not the model itself, determine whether detection succeeds, and the limiting step differs by dataset. On SWaT the ranking is strong but the operating point is not: the 95th-percentile threshold flags almost every timestep, giving 97% recall at 12% precision (Table 2). The limiting step is thresholding. On SMD the failure is one step earlier, in aggregation: the unnormalised score ranks at or below chance, 0.48 for mean and 0.30 for max.

Whether to normalise each variate's error before aggregating is the single choice that separates the two datasets, and it acts through one mechanism. Per-variate normalisation rescales each variate's error by its own calibration baseline, its median and IQR on normal data, which removes any persistent error offset TOTO has for one variate. On SMD this is decisive: TOTO forecasts one variate poorly, and its large but non-anomalous error dominates the unnormalised aggregate, producing the largest error at 59% of timesteps. Normalising removes that error offset, error signal from SMD's anomalies are propogated to the anomaly score, and AUROC rises to 0.69 for mean and 0.73 for max. SWaT results suggest no single variate dominates in this way, so the same normalisation mainly rescales real anomaly signal down toward the error level of fluctuation on quiet variates, and vice versa. AUROC falls to 0.78 and 0.74. The best choice is different between the two datasets.

More broadly, at each timestep the pipeline must decide whether the N-dimensional error vector indicates an anomaly, and it currently does so with a fixed aggregation function followed by a fixed threshold. Across these two datasets this fixed post-processing has no single best setting: the aggregation, normalisation and threshold that work on one are not the ones that work on the other. The error scores carry clear anomaly information that a fixed rule extracts only in part. A natural next step is to learn this reduction, replacing the fixed aggregation and threshold with a small trained model that maps the error vector to a decision. We are not aware of existing work that does this on top of a forecasting foundation model.

## Conclusion

TOTO's representation of normal behaviour generalises: it ranks anomalies at 0.83 AUROC on SWaT, an industrial control domain outside its observability training data, and at 0.73 on the in-domain SMD. The steps wrapped around the model, not the model itself, decide whether detection succeeds, and the limiting step is dataset-dependent: thresholding on SWaT and aggregation on SMD. Learning this reduction from error vector to decision, rather than fixing it by hand, is the natural next step, and a broader evaluation across multivariate benchmarks such as SMAP, MSL and NASA would show how far the pattern holds.

## Limitations

1. **Two datasets.** SWaT and SMD are one multivariate and one largely univariate example; the contrast may not generalise.
2. **Choosing the normalisation needs labels.** Because the right choice flips between datasets, a strict zero-shot user could not know which to apply without a small labelled validation set or a prior on the anomaly nature.
3. **Error metric.** We used NLL; on SWaT, MAE gives better normal-vs-anomaly separation (2.40x vs 1.56x for NLL and 1.69x for MSE), so the error metric is another dataset-dependent choice we did not vary.
4. **Detect stride 8.** A 1/8 temporal subsample; AUROC is stride-robust (SMD within about 0.01 of stride 32), but short events between sampled steps can be missed.

---

**Reproducibility**: Provenance and commands are in `SMD_REPRODUCIBILITY.md`. The AUROC numbers are in `toto/results/auroc/smd_per_machine_detectability.json` (SMD, unnormalised and normalised) and `toto/results/auroc/swat_aggregation_comparison.json` (SWaT, unnormalised and normalised); Table 2 is produced by `analyze_error_concentration.py`. Cached raw NLL arrays are in `toto/results/scores/`, so all aggregation analysis reruns without the model; the scorer was spot-checked against fresh forward passes (`verify_scorer.py`).
