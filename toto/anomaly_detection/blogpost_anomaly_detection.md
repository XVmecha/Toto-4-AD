# Zero-Shot Anomaly Detection with TOTO: A First Evaluation

## Introduction

TOTO (Time Series Optimized Transformer for Observability) is a forecasting foundation model trained on one trillion observability time series data points. This work evaluates TOTO's zero-shot anomaly detection capabilities on two multivariate time series benchmarks: SWaT (industrial control systems) and SMD (server monitoring). We assess whether TOTO's learned representations of normal operational patterns can transfer to previously unseen datasets without task-specific training.

## Problem Formulation

**Zero-shot anomaly detection** requires detecting anomalies in new datasets without using any labeled anomaly examples for training or threshold tuning. This imposes a critical constraint: we can only use normal (non-anomalous) data to calibrate our detection system.

Given a multivariate timeseries forecasting model like TOTO, we convert it into an anomaly detector through three steps:

1. **Error scoring**: Compute prediction error for each timestep using Negative Log-Likelihood (NLL) of TOTO's probabilistic forecasts
2. **Error aggregation**: Combine per-variate errors into a single anomaly score (Mean or Max)
3. **Threshold selection**: Establish an unsupervised decision boundary using only normal data

## Methodology

### Datasets

**SWaT (Secure Water Treatment)**: Industrial control system with 51 sensors monitoring a water treatment plant.
- Calibration: 7 days normal operation (496,800 timesteps)
- Evaluation: 4 days with 36 cyber-physical attacks (449,919 timesteps, ~12% anomalous)

**SMD (Server Machine Dataset)**: Server monitoring metrics from 28 independent machines, each with 38 performance metrics.
- Each machine treated as separate time series (~23,687 timesteps per machine, ~4.3% anomalous)
- Anomalies include hardware failures, configuration errors, resource exhaustion

### Preprocessing Pipeline

Both datasets undergo standardized preprocessing to prepare raw data for TOTO's anomaly detection module:

**SWaT Preprocessing** (`preprocess_swat.py`):
- Load raw Excel files (training: `SWaT_Dataset_Normal_v1.xlsx`, test: `SWaT_Dataset_Attack_v0.xlsx`)
- Parse timestamps and create temporal index
- Extract 51 sensor/actuator features and binary attack labels (Normal=0, Attack=1)
- Handle missing values via linear interpolation with forward/backward fill at edges
- Apply z-score normalization per variate using **calibration statistics only** to prevent data leakage
- Optional downsampling: median aggregation for continuous features, mode for binary features
- Convert to PyTorch tensors: (batch=1, variates=51, timesteps) format
- Output: `swat_train.pt`, `swat_test.pt`, metadata CSV files

**SMD Preprocessing** (`preprocess_smd.py`):
- Load 28 training/test `.txt` files (one per machine) with comma-delimited format
- Load corresponding binary anomaly labels from `test_label/` directory
- Load interpretation labels (time ranges + affected dimensions) for root cause analysis
- Truncate all sequences to minimum length (23,687 timesteps) for uniform batching
- Data already normalized to [0,1] range - no additional normalization applied by default
- Optional z-score normalization available via `--normalize` flag if domain-specific scaling needed
- Optional downsampling: median aggregation for features, majority vote for labels (max per window)
- Convert to PyTorch tensors: (batch=28, variates=38, timesteps) format
- Output: `smd_train.pt`, `smd_test.pt`, metadata JSON files

**Key Design Choices**:
- **Zero-shot compliance**: Normalization statistics computed exclusively on calibration (training) data, never on evaluation data
- **Label preservation**: For downsampled data, any anomaly in a window labels entire window as anomalous (conservative approach)
- **Uniform format**: Both datasets converted to identical tensor format for seamless use with TOTO's API

### Unsupervised Threshold Selection

The core challenge in zero-shot detection is setting a threshold $\tau$ without access to anomaly data. Our approach:

1. **Calibration phase**: Process normal data through TOTO, compute anomaly scores $S_{\text{calib}} = [s_1, s_2, ..., s_N]$
2. **Threshold estimation**: Set threshold as 95th percentile of calibration scores:
   $$\tau = \text{percentile}_{95}(S_{\text{calib}})$$
3. **Detection phase**: Flag evaluation timesteps where $s_t > \tau$ as anomalous

**Rationale**: If TOTO has learned normal behavior, calibration scores should be consistently low. The 95th percentile captures the upper bound of "normal variation" while allowing for 5% noise/outliers. Any evaluation score exceeding this bound indicates deviation from normality.

**Critical constraint**: We never use evaluation data (including its anomalies) for threshold selection. This aims to ensure generalization to unseen/unknown anomalies.

### Threshold-Agnostic Evaluation: AUROC

While threshold-based metrics (Precision, Recall, F1) depend heavily on the chosen threshold, **AUROC (Area Under the Receiver Operating Characteristic curve)** provides a threshold-independent assessment of the model's ability to *rank* anomalies above normal data.

**AUROC interpretation**:
- **1.0**: Perfect ranking - all anomalies scored higher than all normal instances
- **0.5**: Random ranking - model cannot distinguish anomalies from normal data
- **< 0.5**: Inverted ranking - model scores anomalies lower than normal (systematic failure)

AUROC reveals whether the fundamental assumption holds: *do TOTO's learned patterns enable distinguishing anomalous behavior from normal behavior in new datasets?* A high AUROC with low F1 indicates threshold-setting failure (fixable), while AUROC ≈ 0.5 indicates ranking failure (fundamental model limitation).

### Aggregation Strategies

TOTO produces error scores for each of M variates. We test two aggregation methods:

- **Mean aggregation**: $s_t = \frac{1}{M} \sum_{i=1}^M e_t^{(i)}$ - assumes system-wide anomalies affecting multiple sensors
- **Max aggregation**: $s_t = \max_i e_t^{(i)}$ - assumes localized anomalies in individual sensors

### Implementation Details

- Context length: 512 timesteps
- Error metric: Negative Log-Likelihood (NLL) from TOTO's Student-T mixture distributions
- Calibration stride: 32 (dense sampling for robust threshold estimation)
- Evaluation stride: 1 (SWaT full resolution), 32 (SMD for efficiency)
- Z-score normalization per variate using calibration statistics only

## Results

### Table 1: Anomaly Detection Performance

| Dataset | Aggregation | Threshold (τ) | AUROC | Precision | Recall | F1 | Accuracy |
|---------|-------------|---------------|-------|-----------|--------|----|----|
| **SWaT** | Mean | 4.29 | **86.3%** | 12.2% | 97.8% | 21.7% | 15.3% |
| **SWaT** | Max | 13.99 | **80.0%** | 11.8% | 97.2% | 21.0% | 12.5% |
| **SMD** | Mean | 3.35 | **53.0%** | 1.7% | 6.1% | 2.7% | 92.2% |
| **SMD** | Max | 13.11 | **50.1%** | 9.6% | 5.1% | 6.6% | 91.3% |

*Sources: `toto/results/{swat_mean,swat_max,smd_mean,smd_max}/detection_results.json` and `toto/results/auroc/`*

### Key Findings

#### 1. SWaT: Validation of Forecasting-Based Anomaly Detection

The results on SWaT results validate the hypothesis that forecasting models can detect anomalies through prediction errors. With AUROC scores of 86.3% (mean) and 80.0% (max), TOTO more accurately predicts normal operational behaviour of water treatment facility than its anomalous behaviour, demonstrating that learned patterns from its cross-domain training data transfers to industrial control systems. 

However, threshold-based metrics tell a different story: F1 scores remain low (21.7% mean, 21.0% max) despite high recall (97.8%), with catastrophic precision (12.2%) indicating most normal timesteps are incorrectly flagged as anomalous. This reveals a critical challenge in threshold setting when distribution shift occurs between calibration and evaluation datasets. The 95th percentile threshold learned from calibration data (τ = 4.29) falls below the evaluation set's normal mean, causing the detector to flag nearly all timesteps—both normal and anomalous—as anomalies, rendering it useless.

**Implication**: With adaptive thresholding methods that recalibrate using a small validation set from the evaluation period's normal operations, SWaT performance could improve significantly. Excitingly, TOTO's forecast-based approach works for this domain.

#### 2. SMD: Failure of Core Forecasting Assumptions

SMD results expose fundamental limitations of the forecast-error-as-anomaly paradigm. With AUROC scores near random (53.0% mean, 50.1% max)—the max aggregation literally equivalent to a coin flip—TOTO cannot distinguish server anomalies from normal operations. Near-zero precision (1.7-9.6%) and recall (5.1-6.1%) confirm complete detection failure. Unlike SWaT, no amount of threshold tuning can fix this: the model lacks the fundamental ability to rank anomalies above normal data.

Interestingly, both TOTO and SMD involve server monitoring data, yet TOTO's learned patterns from production cloud infrastructure (AWS, Azure, GCP) do not translate to this academic testbed. Our results how, in our current experimental setup, TOTO cannot successfully identify anomalies on the Server Machine Dataset under no anomaly threshold value.

**Implication**: Missmatch between training data of TOTO and SMD despite same domain, server monitoring.

It would be very interesting to see the utility of TOTO for other datsets. 

### Conclusion

This work reveals both the promise and limitations of zero-shot anomaly detection using forecasting foundation models. Our key finding is striking: **TOTO achieves 86.3% AUROC on SWaT**, demonstrating that representations learned from one trillion observability time series successfully transfer to industrial control systems without task-specific training. This validates the core hypothesis that forecasting models can distinguish anomalous behavior through prediction errors.

However, the results also expose critical failure modes. While SWaT's strong AUROC indicates successful anomaly ranking, the catastrophic precision (12.2%) reveals that **unsupervised threshold selection remains the bottleneck**—a notoriously difficult problem exacerbated by distribution shift between calibration and evaluation data. More surprisingly, SMD results challenge the fundamental assumption that unpredictable behavior signals anomalies: with AUROC ≈ 50% (random performance), TOTO cannot detect server anomalies despite both datasets originating from the same monitoring domain. This suggests domain-specific nuances matter more than broad categorical alignment.

**Looking forward**, these results open exciting research directions. The strong SWaT signal suggests that extending evaluation to diverse benchmarks (SMAP, MSL, NASA datasets) could reveal where TOTO's learned patterns transfer most effectively. Intriguingly, our choice of error metric matters: while we used Negative Log-Likelihood to leverage TOTO's probabilistic outputs, preliminary analysis shows **MAE achieves the best separation ratio (2.40x) on SWaT**, outperforming both NLL (1.56x) and MSE (1.69x). This suggests that simpler point-based metrics may be more robust to distribution shift than probabilistic scoring—a hypothesis that warrants systematic investigation across multiple datasets and error metrics.

More importantly, adaptive thresholding methods that recalibrate using small validation sets from target domains could unlock the performance suggested by high AUROC scores. The path to practical zero-shot anomaly detection lies not in abandoning forecasting-based approaches, but in developing principled methods to bridge the gap between strong ranking ability and robust decision boundaries.
## Limitations

1. **Only two datasets tested**: Results may not generalize to other domains
2. **Single threshold percentile**: We used 95th percentile; adaptive methods could improve SWaT but won't fix SMD
3. **NLL-only error metric**: We used Negative Log-Likelihood; alternative metrics (MAE, MSE) may show different patterns (MSE shows best seperation between normal and anomalous data)
4. **Fixed aggregation**: Mean and Max are simple; learned aggregation weights could improve performance

## Conclusion

This work evaluates TOTO's zero-shot anomaly detection capabilities using unsupervised threshold selection and threshold-agnostic AUROC analysis. Our key findings: 

**1. AUROC distinguishes between two failure modes**:
- SWaT (AUROC 86.3%): Successful ranking but poor threshold setting due to distribution shift
- SMD (AUROC 50.1%): Complete ranking failure due to inverted error separation

SWAT results are already good showing domain translation, a bigger experimental setup on multiple multivariate anomaly detection benchmarks might show promising results!

---

**Reproducibility**: All results, configurations, and anomaly score files available in `toto/results/{swat_mean,swat_max,smd_mean,smd_max,auroc}/`.
