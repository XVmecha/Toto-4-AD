# Anomaly Detection Implementation Design Document

## Executive Summary

After thorough analysis of the Toto codebase, this document presents a comprehensive plan for implementing anomaly detection using **negative log-likelihood (NLL)** as the anomaly score with a **95th percentile threshold** computed on training data.

---

## Architecture Analysis

### 1. Distribution Output & Log Probability Support

**Location**: `toto/model/distribution.py`

#### Available Distributions:

1. **StudentTOutput** (lines 18-40)
   - Single Student-T distribution
   - Parameters: `df` (degrees of freedom), `loc` (location), `scale` (spread)
   - Returns: `torch.distributions.StudentT` or `AffineTransformed` wrapper

2. **MixtureOfStudentTsOutput** (lines 43-70)
   - Mixture of K Student-T components
   - Additional parameter: mixture weights (learned)
   - Returns: `torch.distributions.MixtureSameFamily`

#### Key Observation:
**Both distributions are PyTorch Distribution objects**, which means they **inherently support `.log_prob()`**:

```python
# From PyTorch distributions API (built-in):
distribution = model.output_distribution(embeddings)
log_prob = distribution.log_prob(actual_values)  # ✓ Works out of the box!
nll = -log_prob  # Anomaly score
```

**This is perfect for our use case!**

---

### 2. Scaling & Normalization **CRITICAL**

**Location**: `toto/model/scaler.py`

#### Available Scalers:

1. **StdMeanScaler** (lines 15-88)
   - Global mean/std normalization
   - Computes stats over entire sequence
   - Returns: `(scaled_data, loc, scale)`

2. **CausalStdMeanScaler** (lines 274-347)
   - **Causal** per-timestep normalization
   - At time `t`, uses only data up to `t` (no future leakage!)
   - Uses Welford's algorithm for numerical stability
   - Returns per-timestep: `(scaled_data, loc_per_t, scale_per_t)`

3. **CausalPatchStdMeanScaler** (lines 349-457)
   - **Causal** patch-level normalization
   - Each patch uses stats from all data up to that patch
   - More stable than per-timestep
   - **This is the default for Toto-Open-Base-1.0**

#### Critical Understanding: **Affine Transformation**

**Location**: `toto/inference/forecaster.py:389-422`

The model workflow is:
```
Raw Data → [Scaler] → Normalized → [Model] → Base Distribution (normalized space)
                ↓                                      ↓
              loc, scale                    [AffineTransform(loc, scale)]
                                                       ↓
                                            Final Distribution (original scale)
```

The distribution returned by the model is **already in the original data scale** via:
```python
distribution = AffineTransformed(base_distr, loc=loc, scale=scale)
```

**This means**: When we call `distribution.log_prob(actual_value)`, we should pass the **actual (unscaled) value**, and the distribution will handle the transformation internally!

---

### 3. Model Forward Pass

**Location**: `toto/model/backbone.py:211-227`

The model's forward pass returns:
```python
class TotoOutput(NamedTuple):
    distribution: torch.distributions.Distribution  # The full distribution
    loc: Float[torch.Tensor, "batch variate time_steps"]  # Scaling location
    scale: Float[torch.Tensor, "batch variate time_steps"]  # Scaling scale
```

**Key insight**: The `distribution` field is the **complete** distribution object that can be used for `log_prob()` calculations.

---

### 4. Autoregressive Forecasting Pipeline

**Location**: `toto/inference/forecaster.py`

#### Current Generate Loop (lines 353-375):
```python
for _ in range(rounded_steps // patch_size):
    # Get distribution for next patch
    base_distr, loc, scale = self.model(
        inputs=batch_inputs,
        input_padding_mask=batch_input_padding_mask,
        id_mask=batch_id_mask,
        kv_cache=kv_cache,
        scaling_prefix_length=scaling_prefix_length,
    )
    distr = self.create_affine_transformed(base_distr, loc, scale)

    # Currently: sample from distribution
    sample = distr.sample()

    # FOR ANOMALY DETECTION, WE NEED:
    # log_prob = distr.log_prob(actual_future_values)
    # nll = -log_prob

    # Append sample to input (autoregressive)
    batch_inputs = torch.cat([batch_inputs, samples], dim=-1)
```

**Critical observation**: The distribution is created at **each autoregressive step**. For anomaly detection, we need to:
1. Store distributions at each step (or compute log_prob immediately)
2. Compare against actual future values
3. Accumulate NLL scores

---

## Anomaly Detection Implementation Strategy

### Overview

```
┌─────────────────┐
│ Training Phase  │
│ (Normal Data)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 1. Forward pass on training data│
│ 2. Compute NLL for each timestep│
│ 3. Aggregate NLL scores          │
│ 4. Calculate 95th percentile     │
│    → THRESHOLD                   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Test Phase     │
│ (Unknown Data)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 1. Forward pass on test data    │
│ 2. Compute NLL for each timestep│
│ 3. Compare NLL > THRESHOLD       │
│    → ANOMALY!                    │
└──────────────────────────────────┘
```

---

### Key Design Decisions

#### 1. **One-Step-Ahead vs Multi-Step Prediction**

**Option A: One-Step-Ahead** (Recommended for initial implementation)
- At each timestep `t`, use context `[0:t]` to predict `t+1`
- Compute `NLL = -log P(x_{t+1} | x_{0:t})`
- **Pros**: Simple, no error propagation, each prediction is independent
- **Cons**: Doesn't capture multi-step dependencies

**Option B: Multi-Step Autoregressive**
- Predict multiple steps ahead: `[t+1, t+2, ..., t+H]`
- Use autoregressive sampling (feed predictions back)
- **Pros**: Captures longer-range anomalies, detects deviation trends
- **Cons**: Error propagation, more complex, need to decide on sampling strategy

**Recommendation**: Start with Option A, extend to Option B later.

#### 2. **Where to Compute Log Probability**

**Critical Question**: Should we compute log_prob in **normalized space** or **original space**?

**Answer**: **Original space** (already handled by AffineTransformed distribution!)

**Reasoning**:
```python
# The distribution returned by create_affine_transformed is:
distr = AffineTransformed(base_distr, loc=loc, scale=scale)

# When we call:
log_prob = distr.log_prob(actual_value)

# Internally, it does:
# 1. Transform actual_value to normalized space: (actual_value - loc) / scale
# 2. Compute base_distr.log_prob(normalized_value)
# 3. Apply Jacobian correction for the affine transformation
# 4. Return corrected log_prob

# So we just pass ORIGINAL VALUES, and it works correctly!
```

**Important**: The Jacobian correction is:
```
log_prob_affine(x) = log_prob_base((x - loc)/scale) - log(scale)
```
This accounts for the change of variables in the transformation.

#### 3. **Handling Multivariate Time Series**

**Shape Analysis**:
- Input shape: `[batch, variates, time_steps]`
- Distribution output: One distribution per variate per timestep
- Log prob shape: `[batch, variates, time_steps]`

**Aggregation Strategy**:

**Option A**: Per-variate anomaly scores
```python
nll_per_variate = -log_prob  # [batch, variates, time_steps]
anomaly_score_per_variate = nll_per_variate.mean(dim=2)  # Average over time
# Detect which variate is anomalous
```

**Option B**: Joint anomaly score
```python
nll_joint = -log_prob.sum(dim=1)  # [batch, time_steps]
# Sum over variates (assumes independence)
# Single anomaly score per timestep
```

**Option C**: Max pooling
```python
nll_max = -log_prob.max(dim=1)[0]  # [batch, time_steps]
# Most anomalous variate determines score
```

**Recommendation**: Implement all three, let user choose based on use case.

#### 4. **Threshold Calculation on Training Data**

**Implementation**:
```python
# Collect NLL scores from ALL training sequences
training_nlls = []

for batch in train_dataset:
    # Forward pass (one-step-ahead)
    nll_scores = compute_nll(batch)  # [batch, variates, time_steps]
    training_nlls.append(nll_scores.flatten())

# Concatenate all scores
all_training_nlls = torch.cat(training_nlls)

# Compute 95th percentile
threshold = torch.quantile(all_training_nlls, 0.95)
```

**Considerations**:
- **Memory**: For large datasets, use online/streaming percentile estimation
- **Per-variate thresholds**: Optionally compute separate thresholds for each variate
- **Burn-in period**: Ignore first K timesteps (model needs context)

---

## Implementation Architecture

### Proposed Module Structure

```
toto/
├── anomaly_detection/
│   ├── __init__.py
│   ├── detector.py              # Main AnomalyDetector class
│   ├── threshold.py             # Threshold estimation
│   ├── scoring.py               # NLL computation utilities
│   └── evaluation.py            # Metrics (precision, recall, F1)
```

### Core Classes

#### 1. **AnomalyDetector** (`detector.py`)

```python
class AnomalyDetector:
    """
    Anomaly detector using Toto model with NLL-based scoring.

    Attributes:
        model: TotoBackbone model
        threshold: Anomaly threshold (95th percentile from training)
        aggregation: How to aggregate multivariate scores
    """

    def fit(self, train_data: MaskedTimeseries) -> None:
        """
        Compute NLL scores on training data and set threshold.

        Args:
            train_data: Training time series (assumed clean/normal)
        """
        # 1. Forward pass on all training data
        # 2. Compute NLL at each timestep
        # 3. Calculate 95th percentile threshold
        pass

    def detect(
        self,
        test_data: MaskedTimeseries,
        return_scores: bool = False
    ) -> Union[BoolTensor, Tuple[BoolTensor, FloatTensor]]:
        """
        Detect anomalies in test data.

        Args:
            test_data: Test time series
            return_scores: Whether to return NLL scores

        Returns:
            anomalies: Boolean mask [batch, variates, time_steps]
            scores: (Optional) NLL scores
        """
        # 1. Forward pass
        # 2. Compute NLL
        # 3. Compare to threshold
        # 4. Return anomaly mask
        pass

    def score(
        self,
        data: MaskedTimeseries
    ) -> FloatTensor:
        """
        Compute NLL anomaly scores without thresholding.

        Returns:
            scores: NLL scores [batch, variates, time_steps]
        """
        pass
```

#### 2. **NLLScorer** (`scoring.py`)

```python
class NLLScorer:
    """
    Computes negative log-likelihood scores for anomaly detection.
    """

    @staticmethod
    def compute_one_step_ahead_nll(
        model: TotoBackbone,
        data: MaskedTimeseries,
        context_length: int,
    ) -> FloatTensor:
        """
        Compute one-step-ahead NLL.

        For each timestep t in [context_length, T]:
            - Use data[:, :, :t] as context
            - Predict distribution for t
            - Compute -log P(data[:, :, t] | context)

        Returns:
            nll_scores: [batch, variates, time_steps]
        """
        pass

    @staticmethod
    def compute_multi_step_nll(
        model: TotoBackbone,
        data: MaskedTimeseries,
        prediction_horizon: int,
        use_teacher_forcing: bool = False,
    ) -> FloatTensor:
        """
        Compute multi-step NLL with autoregressive prediction.

        Args:
            use_teacher_forcing: If True, use actual values for context
                                If False, use predicted values (error propagation)

        Returns:
            nll_scores: [batch, variates, time_steps]
        """
        pass
```

#### 3. **ThresholdEstimator** (`threshold.py`)

```python
class ThresholdEstimator:
    """
    Estimates anomaly thresholds from training data.
    """

    @staticmethod
    def estimate_percentile_threshold(
        training_scores: FloatTensor,
        percentile: float = 0.95,
        per_variate: bool = False,
    ) -> Union[float, FloatTensor]:
        """
        Estimate threshold as percentile of training scores.

        Args:
            training_scores: [N, variates, time_steps] or [N]
            percentile: Threshold percentile (default 0.95 = 95%)
            per_variate: Compute separate threshold per variate

        Returns:
            threshold: Scalar or [variates] tensor
        """
        pass

    @staticmethod
    def estimate_streaming_threshold(
        training_dataloader: DataLoader,
        scorer_fn: Callable,
        percentile: float = 0.95,
    ) -> float:
        """
        Memory-efficient streaming threshold estimation.

        Uses online quantile estimation for large datasets.
        """
        pass
```

---

## Critical Implementation Details

### 1. **Handling Patch-Based Predictions**

The model outputs distributions at **patch granularity** (e.g., 32 timesteps at once).

**From `forecaster.py:236-247`**:
```python
# Model outputs distribution for a PATCH of timesteps
base_distr, loc, scale = self.model(inputs, ...)
distr = self.create_affine_transformed(base_distr, loc, scale)

# distr.mean has shape: [batch, variates, patch_size]
# For NLL, we need to align this with actual values

# Get the actual values for this patch
actual_values = true_data[:, :, -patch_size:]

# Compute log prob
log_prob = distr.log_prob(actual_values)  # [batch, variates, patch_size]
nll = -log_prob
```

**Key**: The distribution outputs cover the **last patch** of the input. Need to:
1. Track which timesteps correspond to which patches
2. Align actual values correctly
3. Handle overlapping patches if stride < patch_size

### 2. **Scaling Prefix Length**

**From `backbone.py:172`**:
```python
scaled_inputs, loc, scale = self.scaler(
    inputs,
    weights=torch.ones_like(inputs, device=inputs.device),
    padding_mask=input_padding_mask,
    prefix_length=scaling_prefix_length,  # ← Controls which data is used for scaling
)
```

**For anomaly detection**:
- During training threshold estimation: `prefix_length = None` (use all training data for stats)
- During testing: `prefix_length = len(train_data)` (only use training data for scaling stats, avoid test data leakage)

**Important**: With causal scalers, this prevents using future test data in normalization statistics!

### 3. **Distribution Types and Log Prob**

**Student-T** (`distribution.py:32`):
```python
base_dist = torch.distributions.StudentT(df, base_loc, base_scale, validate_args=False)
```
- Heavy-tailed distribution
- More robust to outliers than Gaussian
- **Lower log_prob for outliers** → higher NLL → detected as anomalies ✓

**Mixture of Student-Ts** (`distribution.py:64-69`):
```python
components = StudentT(df, loc, scale)  # [batch, variates, time, k_components]
mixture_distribution = torch.distributions.Categorical(probs=probs)
return torch.distributions.MixtureSameFamily(mixture_distribution, components)
```
- Can model **multi-modal** distributions
- `.log_prob()` automatically handles mixture:
  ```
  log P(x) = log Σ_k π_k P_k(x)
  ```
- More expressive, can capture multiple normal behaviors

**Both support `.log_prob()`** out of the box! ✓

### 4. **One-Step vs Autoregressive: Detailed Comparison**

#### One-Step-Ahead (Independent Predictions):
```python
for t in range(context_length, T):
    # Use ACTUAL data up to t
    context = data[:, :, :t]

    # Predict distribution at t
    distr_t = model(context)

    # Compute NLL for actual value at t
    nll[t] = -distr_t.log_prob(data[:, :, t])
```

**Pros**:
- Each prediction is independent
- No error propagation
- Fair comparison (always uses ground truth context)
- Easier to implement

**Cons**:
- Doesn't test model's autoregressive ability
- May miss anomalies that appear over multiple steps

#### Autoregressive (Teacher Forcing = False):
```python
for t in range(context_length, T, patch_size):
    # Use predicted values in context (after first iteration)
    if t == context_length:
        context = data[:, :, :t]  # Initial: use actual data
    else:
        context = predicted_data[:, :, :t]  # Use predictions

    # Predict next patch
    distr_t = model(context)

    # Sample from distribution (feed back as input)
    predicted_patch = distr_t.sample()
    predicted_data = torch.cat([context, predicted_patch], dim=-1)

    # Compute NLL for actual values
    actual_patch = data[:, :, t:t+patch_size]
    nll[t:t+patch_size] = -distr_t.log_prob(actual_patch)
```

**Pros**:
- Tests full autoregressive capability
- Errors compound → anomalies may become more apparent
- More realistic for long-horizon detection

**Cons**:
- Error propagation can mask or amplify anomalies
- Harder to interpret (is high NLL due to anomaly or cascading errors?)
- More complex implementation

**Recommendation**:
1. Start with one-step-ahead for baseline
2. Implement autoregressive as advanced mode
3. Compare performance empirically

---

## Data Flow Diagram

```
                        TRAINING PHASE
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Training Data (Normal)                                 │
│  [batch, variates, time_steps]                         │
│              │                                           │
│              ▼                                           │
│  ┌───────────────────────┐                             │
│  │ For each sequence:    │                             │
│  │  - Sliding window     │                             │
│  │  - Context [:t]       │                             │
│  │  - Predict t+1        │                             │
│  └───────────┬───────────┘                             │
│              │                                           │
│              ▼                                           │
│  ┌───────────────────────┐                             │
│  │ Model Forward Pass    │                             │
│  │  1. Scale inputs      │                             │
│  │  2. Patch embed       │                             │
│  │  3. Transformer       │                             │
│  │  4. Distribution head │                             │
│  └───────────┬───────────┘                             │
│              │                                           │
│              ▼                                           │
│  ┌───────────────────────┐                             │
│  │ distribution.log_prob │                             │
│  │    (actual_values)    │                             │
│  └───────────┬───────────┘                             │
│              │                                           │
│              ▼                                           │
│  NLL = -log_prob                                        │
│  [batch, variates, time_steps]                         │
│              │                                           │
│              ▼                                           │
│  ┌───────────────────────┐                             │
│  │ Aggregate & Store:    │                             │
│  │ all_training_nlls.    │                             │
│  │   append(nll)         │                             │
│  └───────────┬───────────┘                             │
│              │                                           │
└──────────────┼───────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Compute 95th         │
    │ Percentile           │
    │ → THRESHOLD          │
    └──────────┬───────────┘
               │
               │
               │
               ▼
                        TEST PHASE
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Test Data (Unknown)                                    │
│  [batch, variates, time_steps]                         │
│              │                                           │
│              ▼                                           │
│  Same forward pass as training                          │
│              │                                           │
│              ▼                                           │
│  NLL scores [batch, variates, time_steps]              │
│              │                                           │
│              ▼                                           │
│  ┌───────────────────────┐                             │
│  │ Compare to threshold: │                             │
│  │ anomaly = NLL > THR   │                             │
│  └───────────┬───────────┘                             │
│              │                                           │
│              ▼                                           │
│  Anomaly Mask [batch, variates, time_steps]            │
│  (Boolean: True = anomaly)                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Considerations

### 1. **Memory Efficiency**

**Challenge**: Computing NLL for long sequences requires storing distributions.

**Solutions**:
- **Streaming processing**: Process in batches, accumulate NLL scores
- **KV Cache**: Use `use_kv_cache=True` for faster autoregressive decoding
- **Checkpoint gradients**: Not needed (inference only, no gradients)

### 2. **Speed Optimization**

**Bottlenecks**:
1. Model forward pass (transformer)
2. Log probability computation
3. Threshold comparison

**Optimizations**:
- **Batch processing**: Process multiple sequences in parallel
- **Compiled model**: Use `torch.compile()` (but not on MPS currently)
- **Half precision**: Use float16 where possible (careful with log operations!)
- **Vectorized operations**: Avoid Python loops

### 3. **Numerical Stability**

**Concerns**:
- Log probabilities can be very negative (→ numerical issues)
- Extreme values in data (→ NaN in scaling)

**Solutions**:
- Use `replace_extreme_values()` from `dataset.py:139`
- Clamp log probabilities: `log_prob.clamp(min=-100)`
- Use `torch.float32` for accumulation (even if model is float16)
- Monitor for NaN/Inf in pipeline

---

## Evaluation Metrics

### Anomaly Detection Metrics

Assuming we have **labeled anomalies** in test data:

```python
# Ground truth anomalies: [batch, variates, time_steps] boolean
# Predicted anomalies: [batch, variates, time_steps] boolean

# Binary classification metrics per timestep
TP = (predicted & ground_truth).sum()
FP = (predicted & ~ground_truth).sum()
FN = (~predicted & ground_truth).sum()
TN = (~predicted & ~ground_truth).sum()

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)

# ROC curve: vary threshold
thresholds = np.linspace(nll_scores.min(), nll_scores.max(), 100)
for threshold in thresholds:
    predicted = nll_scores > threshold
    compute_metrics(predicted, ground_truth)

# AUC-ROC, AUC-PR
```

### Additional Metrics

- **Point-adjusted metrics**: Credit detection if within window of true anomaly
- **Range-based F1**: Credit if any point in anomalous range is detected
- **Affiliation metrics**: Measure overlap of detected and true anomaly ranges

---

## Summary & Recommendations

### ✓ Key Findings

1. **Distribution API Perfect**: PyTorch distributions support `.log_prob()` natively ✓
2. **Scaling Handled**: AffineTransformed distributions work in original data space ✓
3. **Causal Scalers**: Prevent test data leakage in normalization ✓
4. **Patch-based Output**: Need to carefully align patches with actual values
5. **Autoregressive Support**: Can implement both one-step and multi-step

### Recommended Implementation Path

**Phase 1: Core Implementation**
1. Implement `NLLScorer.compute_one_step_ahead_nll()`
2. Implement `ThresholdEstimator.estimate_percentile_threshold()`
3. Implement `AnomalyDetector.fit()` and `.detect()`
4. Test on synthetic data with known anomalies

**Phase 2: Advanced Features**
1. Add multi-step autoregressive NLL
2. Add per-variate threshold estimation
3. Add streaming threshold estimation for large datasets
4. Implement evaluation metrics

**Phase 3: Optimization**
1. Batch processing for speed
2. Memory-efficient processing for long sequences
3. M4 GPU optimization (MPS fallback handling)

### Critical Implementation Notes

1. **Always use `.to(device)` for tensors** (M4 GPU support)
2. **Set `PYTORCH_ENABLE_MPS_FALLBACK=1`** before imports
3. **Use actual (unscaled) values** in `log_prob()` calls
4. **Set appropriate `prefix_length`** to prevent test data leakage
5. **Handle patch alignment** carefully
6. **Start with one-step-ahead**, extend to autoregressive later

### Next Steps

1. **Do NOT implement yet** - this is design phase ✓
2. Review this document for any issues or missing considerations
3. Get user approval on design decisions
4. Prepare synthetic test data for validation
5. Begin Phase 1 implementation

---

## Open Questions for Discussion

1. **Aggregation strategy**: Per-variate, joint, or max pooling for multivariate?
2. **Threshold granularity**: Global threshold vs per-variate thresholds?
3. **Evaluation data**: Do we have labeled anomaly datasets for validation?
4. **Prediction horizon**: Start with one-step-ahead or go straight to multi-step?
5. **Performance requirements**: Real-time detection needed, or batch processing OK?

---

**Document Status**: ✓ COMPLETE - Ready for Review
**Next Action**: User review and approval before implementation
