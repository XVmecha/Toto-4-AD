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

The results on SWaT results validate the hypothesis that forecasting models can detect anomalies through prediction errors. With AUROC scores of 86.3% (mean) and 80.0% (max), TOTO more accurately predicts normal operational behaviour of water treatment facility than its anomalous behaviour, demonstrating that learned patterns from its cross-domain training data transfer to a new industrial control systems dataset. 

However, threshold-based metrics tell a different story: F1 scores remain low (21.7% mean, 21.0% max) with high recall (97.8%) accompanied by very poor precision (12.2%) indicating most normal timesteps are incorrectly flagged as anomalous. This reveals a critical challenge in threshold setting when distribution shift occurs between calibration and evaluation datasets. The 95th percentile threshold learned from calibration data (τ = 4.29) falls below the evaluation set's normal mean, causing the detector to flag nearly all timesteps—both normal and anomalous—as anomalies, rendering the anomaly system useless.

**Implication**: HERE Unsupervised anomaly detection does not work yet under our experimental setting but with adaptive thresholding methods that recalibrate using a small validation set from the evaluation period's normal operations, SWaT performance could improve significantly. Excitingly, TOTO's forecast-based approach works for this domain.

#### 2. SMD: Failure of Core Forecasting Assumptions

SMD results expose fundamental limitations of the forecast-error-as-anomaly paradigm. With AUROC scores near random (53.0% mean, 50.1% max)—the max aggregation literally equivalent to a coin flip. TOTO cannot distinguish server anomalies from normal operations. Near-zero precision (1.7-9.6%) and recall (5.1-6.1%) confirm complete detection failure. Unlike SWaT, no amount of threshold tuning can fix this: the model lacks the fundamental ability to rank anomalies above normal data.

Interestingly, both TOTO and SMD involve server monitoring data, yet TOTO's learned patterns from production cloud infrastructure (AWS, Azure, GCP) do not translate to this academic testbed. Our results how, in our current experimental setup, TOTO cannot successfully identify anomalies on the Server Machine Dataset under no anomaly threshold value.

**Implication**: Missmatch between training data of TOTO and SMD despite same domain, server monitoring.

It would be very interesting to see the utility of TOTO for other datsets. 

### Conclusion

This work evaluates whether forecasting foundation models can perform zero-shot anomaly detection by leveraging learned representations of normal operational patterns. Our investigation across two multivariate time series benchmarks—SWaT (industrial control systems) and SMD (server monitoring)—reveals both promising capabilities and fundamental limitations that clarify when and how forecasting-based anomaly detection can succeed.

#### Summary of Principal Findings

Our central finding is that **domain-specific characteristics, not just data modality, determine transferability**. TOTO achieves 86.3% AUROC on SWaT, demonstrating successful transfer from trillion-scale observability data to industrial control systems despite domain mismatch. This strong ranking ability validates the core hypothesis: forecasting models can distinguish anomalous behavior through prediction errors when normal operational patterns exhibit sufficient structure.

However, this success contrasts sharply with SMD's near-random performance (AUROC 50.1%), despite both TOTO's training data and SMD belonging to server monitoring domains. This failure exposes a critical insight: **categorical domain alignment does not guarantee transfer**. The academic testbed characteristics of SMD—with its controlled failure scenarios and synthetic anomalies—differ fundamentally from the production cloud infrastructure patterns TOTO learned, suggesting that operational context and anomaly characteristics matter more than superficial domain labels.

#### The Distribution Shift Challenge

Beyond ranking ability, our results expose **distribution shift as the critical bottleneck** in zero-shot detection. Even when models successfully rank anomalies (SWaT AUROC 86.3%), unsupervised threshold selection can fail catastrophically: SWaT's precision of 12.2% means 88% of alarms are false positives. The root cause is quantifiable: the calibration threshold (τ=4.29 at 95th percentile) falls below the evaluation set's normal mean (5.26), causing the detector to flag nearly all timesteps as anomalous.

This threshold failure reveals a fundamental tension in zero-shot anomaly detection: calibration data must be representative of evaluation data's *normal* behavior, yet we assume no access to evaluation data during setup. When operational regimes shift between calibration and evaluation periods—as commonly occurs in real-world deployments—purely unsupervised thresholds become unreliable regardless of ranking quality.

#### Critical Design Choices

Our analysis identifies three implementation decisions that profoundly impact performance:

**1. Error Metric Selection**: While we used Negative Log-Likelihood to leverage TOTO's probabilistic outputs, preliminary analysis reveals **MAE achieves superior separation** (2.40x anomaly-to-normal ratio) compared to NLL (1.56x) and MSE (1.69x) on SWaT. This suggests simpler point-based metrics may be more robust to distribution shift than probabilistic scoring, challenging the assumption that probabilistic models necessarily require probabilistic evaluation metrics.

**2. Aggregation Strategy**: Our comparison of mean versus max aggregation reveals their differential sensitivities: mean aggregation preserves score scale and suits system-wide anomalies, while max aggregation amplifies localized sensor failures but increases noise sensitivity. The choice fundamentally shapes threshold dynamics and detection characteristics.

**3. Normalization Robustness**: Post-hoc analysis of preprocessing pipelines reveals z-score normalization's fragility under distribution shift. When sensor operational ranges differ between calibration and evaluation periods—or when dormant sensors activate—division by calibration statistics can produce extreme outliers that overwhelm detection systems. This normalization brittleness compounds threshold selection challenges.

#### Implications for Practical Deployment

These findings carry direct implications for deploying forecasting-based anomaly detection:

**When It Works**: Zero-shot detection succeeds when (1) evaluation data's normal behavior resembles calibration data's distribution, (2) anomalies manifest as prediction errors rather than different-but-predictable patterns, and (3) the operational domain exhibits sufficient temporal structure for forecasting models to capture.

**When It Fails**: Detection fails when domain-specific patterns—sensor interactions, failure modes, operational constraints—differ fundamentally between training and target environments, even within the same nominal domain. The SMD results demonstrate that trillion-scale training data cannot compensate for fundamental distribution mismatch.

**Threshold Adaptation is Essential**: High AUROC with poor precision signals that threshold adaptation, not model retraining, is the solution. A small validation set from the target environment's normal operations—even dozens of samples—could recalibrate thresholds and unlock the ranking ability demonstrated by strong AUROC scores. This represents a middle ground between fully supervised methods and purely zero-shot approaches.

#### Limitations and Boundaries

Several limitations bound our conclusions:

**Limited Dataset Diversity**: Evaluation on two datasets—while revealing distinct failure modes—cannot fully characterize when forecasting-based detection succeeds. SWaT and SMD represent specific subsets of industrial and IT monitoring; performance on aerospace (SMAP/MSL), medical, or environmental monitoring domains remains unexplored.

**Single-Model Evaluation**: Results reflect TOTO's specific architecture and training data. Alternative forecasting foundations or different training objectives might exhibit different transfer characteristics. Our findings characterize this approach, not forecasting-based detection universally.

**Normalization Sensitivity**: Our preprocessing choices, particularly z-score normalization, introduce brittleness that may not reflect optimal practices. Robust preprocessing pipelines could mitigate some observed failures.

#### Future Research Directions

This work opens several promising research avenues:

**1. Systematic Error Metric Study**: The superiority of MAE over NLL on SWaT warrants comprehensive investigation across multiple datasets, anomaly types, and operational domains. Understanding which metrics best capture anomalousness under distribution shift could improve practical deployments.

**2. Adaptive Thresholding Methods**: Developing principled approaches to threshold recalibration using minimal validation data represents the most immediate path to practical impact. The gap between 86% AUROC and 12% precision on SWaT demonstrates significant room for improvement through better calibration strategies.

**3. Benchmark Expansion**: Evaluating on diverse benchmarks (SMAP, MSL, NASA datasets, UCR anomaly archive) would reveal where TOTO's learned patterns transfer most effectively and characterize the domain boundaries of zero-shot detection.

**4. Robust Preprocessing**: Investigating normalization alternatives (robust scaling, quantile transforms, domain-specific standardization) could reduce sensitivity to distribution shift and improve zero-shot generalization.

**5. Learned Aggregation**: Moving beyond fixed mean/max rules to learned, context-dependent aggregation could better capture domain-specific anomaly patterns and reduce false positives.

#### Closing Perspective

The path to practical zero-shot anomaly detection does not require abandoning forecasting-based approaches—SWaT's results prove their viability. Instead, progress demands recognizing that zero-shot detection is not truly zero-shot in deployment: minimal domain adaptation through threshold recalibration and robust preprocessing is essential. The challenge shifts from "can forecasting models detect anomalies without training?" to "what minimal adaptation enables forecasting models to detect anomalies in new domains?"

This reframing suggests a more nuanced deployment paradigm: use forecasting foundations for strong anomaly ranking (the hard part), then adapt decision boundaries using small validation sets from target environments (the practical part). By acknowledging this hybrid nature, we can leverage trillion-scale pretraining's benefits while remaining grounded in deployment realities. The 86% AUROC on SWaT demonstrates this approach's promise; addressing the precision gap through principled adaptation methods remains the key challenge for future work.

**Reproducibility**: All results, configurations, and anomaly score files available in `toto/results/{swat_mean,swat_max,smd_mean,smd_max,auroc}/`.
