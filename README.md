# TOTO Anomaly Detection

**Zero-shot anomaly detection for multivariate time series using TOTO**

---

## Overview

This module extends TOTO (Time Series Optimized Transformer for Observability) from a forecasting foundation model into a **zero-shot anomaly detection system**. The core insight: if TOTO has learned to accurately predict normal system behavior, then large prediction errors signal deviations from normality—i.e., anomalies.

### The Premise

TOTO was pretrained on **one trillion time series data points** from observability metrics (infrastructure monitoring, application performance, cloud services). Through this extensive training, TOTO learned rich representations of normal operational patterns. We leverage these learned patterns to detect previously unseen anomalies **without task-specific training**.

### The Approach

We transform TOTO's probabilistic forecasts into anomaly scores through three key steps:

1. **Error Scoring**: Compute Negative Log-Likelihood (NLL) of observations under TOTO's predicted distributions
2. **Error Aggregation**: Combine per-variate errors into scalar anomaly scores (Mean or Max)
3. **Threshold Selection**: Establish decision boundaries using only normal data (95th percentile)

**Key Advantage**: This is a true zero-shot approach—we never use anomaly data during threshold calibration, enabling detection of novel anomaly types.

---

## Quick Start

### Installation

```bash
# Install TOTO with anomaly detection support
pip install toto-ts

# Or install from source
git clone https://github.com/DataDog/Toto-4-AD.git
cd Toto-4-AD
pip install -e .
```

### Basic Usage

```python
import torch
from toto.model.toto import Toto
from toto.anomaly_detection import AnomalyDetector

# Load pretrained TOTO model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.to(device)
toto.eval()

# Create anomaly detector
detector = AnomalyDetector(
    model=toto.model,
    context_length=512,
    aggregation='mean',  # 'mean' or 'max'
    threshold_percentile=95.0,
)

# Fit on normal training data (no anomalies)
detector.fit(train_series, stride=32)  # train_series: (batch, variates, timesteps)
print(f"Threshold: {detector.threshold:.4f}")

# Detect anomalies in test data
is_anomaly, scores = detector.detect(
    test_series,
    stride=1,
    return_scores=True
)

# Results:
# - is_anomaly: (batch, timesteps) binary predictions
# - scores: (batch, timesteps) continuous anomaly scores
```

---

## Datasets & Preprocessing

We evaluate on two benchmark multivariate time series anomaly detection datasets:

### 1. SWaT (Secure Water Treatment)

**Domain**: Industrial control systems
**Sensors**: 51 variates (flow rates, tank levels, valve states)
**Training**: 7 days normal operation (496,800 timesteps)
**Test**: 4 days with 36 cyber-physical attacks (449,919 timesteps, ~12% anomalous)
**Anomalies**: Manipulated sensor readings, unauthorized valve controls

**Preprocess SWaT:**
```bash
# Download SWaT dataset and extract to toto/data/SWaT.A1 & A2_Dec 2015/
# Then preprocess:
python preprocess_swat.py \
    --data_dir "toto/data/SWaT.A1 & A2_Dec 2015/Physical" \
    --output_dir toto/data/preprocessed_datasets/swat \
    --downsample 1

# Output:
#   - swat_train.pt (normal operations)
#   - swat_test.pt (with attacks)
#   - swat_train_metadata.csv
#   - swat_test_metadata.csv
```

### 2. SMD (Server Machine Dataset)

**Domain**: Server monitoring
**Sensors**: 38 variates per machine (CPU, memory, disk I/O, network)
**Machines**: 28 independent servers
**Training**: ~23,687 timesteps per machine (all normal)
**Test**: ~23,687 timesteps per machine (~4.3% anomalous)
**Anomalies**: Hardware failures, configuration errors, resource exhaustion

**Preprocess SMD:**
```bash
# Download SMD dataset and extract to toto/data/ServerMachineDataset/
# Then preprocess:
python preprocess_smd.py \
    --data_dir toto/data/ServerMachineDataset \
    --output_dir toto/data/preprocessed_smd_1x \
    --downsample 1

# Output:
#   - smd_train.pt (28 machines, normal)
#   - smd_test.pt (28 machines, with anomalies)
#   - smd_train_metadata.json
#   - smd_test_metadata.json
```

---

## Running Anomaly Detection

### SWaT Detection

```bash
# Mean aggregation (system-wide anomalies)
python run_swat_anomaly_detection.py \
    --data_dir toto/data/preprocessed_datasets/swat \
    --output_dir toto/results/swat_mean \
    --aggregation mean \
    --context_length 512 \
    --threshold_percentile 95.0 \
    --fit_stride 32 \
    --detect_stride 1

# Max aggregation (localized anomalies)
python run_swat_anomaly_detection.py \
    --data_dir toto/data/preprocessed_datasets/swat \
    --output_dir toto/results/swat_max \
    --aggregation max \
    --detect_stride 1

# Results saved to:
#   - swat_detection_results.json (metrics: precision, recall, F1, AUROC)
#   - swat_anomaly_detection_results.png (visualization)
#   - swat_score_distribution.png (score histogram)
```

### SMD Detection

```bash
# Mean aggregation
python run_smd_anomaly_detection.py \
    --data_dir toto/data/preprocessed_smd_1x \
    --output_dir toto/results/smd_mean \
    --aggregation mean \
    --context_length 512 \
    --threshold_percentile 95.0 \
    --fit_stride 32 \
    --detect_stride 32 \
    --plot_machines 0 5 10

# Max aggregation
python run_smd_anomaly_detection.py \
    --data_dir toto/data/preprocessed_smd_1x \
    --output_dir toto/results/smd_max \
    --aggregation max \
    --detect_stride 32 \
    --plot_machines 0 5 10

# Results saved to:
#   - smd_detection_results.json (overall + per-machine metrics)
#   - smd_machine_{0,5,10}_anomaly_detection.png
#   - smd_machine_{0,5,10}_score_distribution.png
#   - Precision, Recall, F1 outputted in terminal
```

---

## Threshold-Agnostic Evaluation: AUROC

**Problem**: Threshold-based metrics (Precision, Recall, F1) depend heavily on the chosen threshold. Distribution shift between calibration and evaluation data can make fixed thresholds ineffective.

**Solution**: AUROC (Area Under ROC Curve) measures the model's ability to **rank** anomalies above normal data, independent of any threshold choice.

### Computing AUROC

```bash
# SWaT with mean aggregation
python compute_auroc.py swat mean

# SWaT with max aggregation
python compute_auroc.py swat max

# SMD with mean aggregation
python compute_auroc.py smd mean

# SMD with max aggregation
python compute_auroc.py smd max

# Results saved to:
#   - toto/results/auroc/{dataset}_{aggregation}_auroc.json
```

### Interpreting AUROC

- **AUROC = 1.0**: Perfect ranking—all anomalies scored higher than all normal points
- **AUROC = 0.5**: Random performance—model cannot distinguish anomalies from normal
- **AUROC < 0.5**: Inverted ranking—model scores anomalies lower than normal (systematic failure)

**Diagnostic Value**: AUROC reveals whether the model has the fundamental capacity to detect anomalies:
- **High AUROC, Low F1**: Threshold-setting failure → Try adaptive thresholding
- **AUROC ≈ 0.5**: Ranking failure → Requires domain fine-tuning or different approach

---

## Results Summary

### SWaT: Strong Ranking, Threshold Challenges

| Aggregation | AUROC | Precision | Recall | F1 | Interpretation |
|-------------|-------|-----------|--------|----|----|
| Mean | **86.3%** | 12.2% | 97.8% | 21.7% | ✓ Model ranks anomalies correctly |
| Max | **80.0%** | 11.8% | 97.2% | 21.0% | ⚠ Distribution shift causes threshold mismatch |

**Key Findings**:
- TOTO successfully learned transferable patterns from observability data to industrial control systems
- High AUROC demonstrates zero-shot transfer capability
- Low F1 due to distribution shift between calibration and evaluation sets (threshold too low)
- **Conclusion**: Threshold-setting failure, not ranking failure

### SMD: Fundamental Ranking Failure

| Aggregation | AUROC | Precision | Recall | F1 | Interpretation |
|-------------|-------|-----------|--------|----|----|
| Mean | **53.0%** | 1.7% | 6.1% | 2.7% | ✗ Random performance |
| Max | **50.1%** | 9.6% | 5.1% | 6.6% | ✗ Literally a coin flip |

**Key Findings**:
- AUROC ≈ 0.5 indicates model cannot distinguish anomalies from normal data
- Root cause: **Inverted separation**—anomalies are MORE predictable than normal operations
- Server anomalies (crashes, saturation) manifest as simple patterns (flatlines, zeros)
- Normal operations exhibit complex, multi-modal behavior
- **Conclusion**: Core assumption "anomalous = unpredictable" does not hold for SMD

For more information about our experimentation please read blogpost_anomaly_detection.md
---

## Design Choices & Trade-offs

### Error Aggregation Strategies

**Mean Aggregation**: $s_t = \frac{1}{M} \sum_{i=1}^M e_t^{(i)}$
- Assumes system-wide anomalies affecting multiple sensors
- Smooth, stable scores less sensitive to noise
- Can dilute localized anomalies
- **Best for**: Tightly coupled systems (SWaT: AUROC 86.3%)

**Max Aggregation**: $s_t = \max_i e_t^{(i)}$
- Detects sensor-specific anomalies
- High sensitivity to localized failures
- More prone to false positives from noise
- **Best for**: Independent components where single failures matter

### Threshold Selection Methods

**95th Percentile (Default)**:
- Rationale: Upper bound of "normal variation" observed in calibration data
- Conservative: Allows 5% of calibration scores to exceed threshold (accounts for noise)
- Limitation: Assumes calibration and evaluation distributions match

**Alternative Methods** (available in module):
- Mean + k*std: Statistical outlier detection
- Median Absolute Deviation (MAD): Robust to outliers
- Adaptive online thresholds: Update threshold as distribution shifts (future work)
