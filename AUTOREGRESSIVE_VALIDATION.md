# Toto Autoregressive Forecasting: Architecture Validation

## Executive Summary

**✓ VALIDATED**: Toto implements the exact autoregressive forecasting recipe you described for anomaly detection:

1. **Outputs probability distribution** - Student-T or Mixture of Student-T distributions
2. **Samples from distribution** - Generates multiple sample trajectories
3. **Autoregressive prediction** - Appends predictions to input and continues
4. **Multiple timesteps** - Generates full prediction horizons autoregressively

---

## Detailed Architecture Analysis

### 1. Distribution Output ✓

**Location**: `toto/model/distribution.py`

The model outputs **parametric probability distributions**, not just point estimates:

```python
class StudentTOutput(DistributionOutput):
    """
    Outputs Student-T distribution with learned parameters:
    - df (degrees of freedom)
    - loc (location/mean)
    - scale (spread/variance)
    """
```

**Available distributions**:
- **StudentT**: Single Student-T distribution (robust to outliers)
- **MixtureOfStudentTs**: Mixture of Student-T distributions (multi-modal)

The model returns a `torch.distributions.Distribution` object that supports:
- `.sample()` - Draw random samples
- `.mean` - Get mean prediction
- `.log_prob()` - Compute log probability (useful for anomaly detection!)

### 2. Autoregressive Sampling ✓

**Location**: `toto/inference/forecaster.py:353-375`

The critical autoregressive loop for **sample generation**:

```python
for _ in range(rounded_steps // patch_size):
    # Step 1: Model forward pass returns distribution
    base_distr, loc, scale = self.model(
        inputs=batch_inputs,
        input_padding_mask=batch_input_padding_mask,
        id_mask=batch_id_mask,
        kv_cache=kv_cache,
        scaling_prefix_length=scaling_prefix_length,
    )
    distr = self.create_affine_transformed(base_distr, loc, scale)

    # Step 2: Sample from the distribution
    sample = distr.sample()

    # Step 3: Append sample to input (AUTOREGRESSIVE)
    samples = replace_extreme_values(sample[:, :, -patch_size:])
    batch_inputs = torch.cat([batch_inputs, samples], dim=-1)

    # Update masks and timestamps
    batch_id_mask = torch.cat([batch_id_mask, dummy_id_mask], dim=-1)
    batch_input_padding_mask = torch.cat([batch_input_padding_mask, dummy_padding], dim=-1)
```

**Key observation**: Line 369 shows `batch_inputs = torch.cat([batch_inputs, samples], dim=-1)` - **this is exactly the autoregressive append you're looking for!**

### 3. Multiple Sample Trajectories ✓

**Location**: `toto/inference/forecaster.py:124-148`

The forecaster creates **multiple independent sample chains**:

```python
# Repeat inputs for each sample trajectory
inputs = repeat(
    inputs,
    "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
    sampling_batch_size=sampling_batch_size,
)

# Each trajectory evolves independently through autoregressive sampling
for _ in range(num_batches):
    batch_inputs = torch.clone(inputs)  # Fresh copy for each batch

    for _ in range(rounded_steps // patch_size):
        # Sample from distribution
        sample = distr.sample()
        # Append to trajectory
        batch_inputs = torch.cat([batch_inputs, samples], dim=-1)

    all_samples.append(batch_inputs)
```

**Result**: You get `num_samples` different future trajectories, each representing a plausible future based on the distribution.

### 4. Forecasting API ✓

**Location**: `toto/inference/forecaster.py:88-182`

```python
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,      # How many steps ahead
    num_samples=256,            # Number of sample trajectories
    samples_per_batch=256,      # Batch size for memory efficiency
)

# Access results
forecast.samples     # [batch, variate, time_steps, samples]
forecast.median      # Median across samples (robust point estimate)
forecast.mean        # Mean across samples
forecast.std         # Uncertainty estimate
forecast.quantile(q) # Any quantile for confidence intervals
```

---

## Perfect for Anomaly Detection!

This architecture is **ideal** for anomaly detection because:

### 1. **Probabilistic Predictions**
- Get full probability distribution, not just point estimates
- Can compute `log_prob(actual_value)` - low probability = anomaly!

### 2. **Uncertainty Quantification**
- Multiple samples show prediction uncertainty
- Wide distribution = high uncertainty (normal in volatile periods)
- Narrow distribution = high confidence (anomalies stand out more)

### 3. **Multi-Step Forecasting**
- Autoregressive means prediction uncertainty compounds naturally
- Short-term: tight distributions
- Long-term: wider distributions
- Anomalies detected at any horizon

### 4. **Multivariate Support**
- Handles correlated variables
- Detect anomalies in relationships between variables

---

## Key Implementation Details

### Patch-Based Processing
- Model processes time in "patches" (default patch_size from config)
- Generates `patch_size` timesteps per iteration
- Efficient for transformer processing

### KV Cache Optimization
- Caches transformer attention keys/values
- Avoids recomputing on already-seen context
- Speeds up autoregressive generation significantly

### Affine Transformation
**Location**: `toto/inference/forecaster.py:389-422`

```python
def create_affine_transformed(base_distr, loc, scale):
    """
    Transform base distribution to actual data scale:
    output = scale * base_distr + loc

    This allows model to output in normalized space
    while predictions are in original data scale
    """
    return AffineTransformed(base_distr, loc=loc, scale=scale)
```

---

## Architecture Flow for Anomaly Detection

```
Historical Data (context)
         ↓
    [Scaling] → normalize to mean=0, std=1
         ↓
  [Patch Embed] → convert to patches
         ↓
  [Transformer] → learn patterns with time/space attention
         ↓
   [Unembed] → project back to time series space
         ↓
[Distribution Head] → output Student-T parameters (df, loc, scale)
         ↓
         ├─→ .sample() → Draw N samples → Multiple futures
         ├─→ .mean → Point prediction
         └─→ .log_prob(actual) → ANOMALY SCORE! ← Perfect for AD
         ↓
[Autoregressive] → Append sample to context
         ↓
      Repeat for next timestep
```

---

## Code References for Anomaly Detection Implementation

### Getting Probability for Anomaly Scoring

You can modify `forecaster.py` to return distributions:

```python
# In generate_samples() or generate_mean(), before sampling:
base_distr, loc, scale = self.model(inputs, ...)
distr = self.create_affine_transformed(base_distr, loc, scale)

# For anomaly detection, you want:
log_prob = distr.log_prob(actual_future_value)
# Low log_prob → High anomaly score!

# Or use samples for threshold-based detection:
samples = distr.sample((num_samples,))
mean = samples.mean(dim=0)
std = samples.std(dim=0)
z_score = (actual_future_value - mean) / std
# High |z_score| → Anomaly
```

### Accessing Distributions During Forecasting

Current implementation samples and concatenates, but you can modify to:

1. **Store distributions at each step** for later anomaly scoring
2. **Compare actual vs predicted** at each timestep
3. **Use prediction uncertainty** (std) as confidence bounds

---

## Summary: Recipe Validation ✓

| Your Requirement | Toto Implementation | Status |
|-----------------|-------------------|--------|
| Outputs probability distribution | Student-T / Mixture-of-Student-T | ✓ |
| Samples from distribution | `distr.sample()` in line 363 | ✓ |
| Autoregressive appending | `torch.cat([batch_inputs, samples], dim=-1)` line 369 | ✓ |
| Multiple timesteps | Loop `for _ in range(rounded_steps // patch_size)` line 353 | ✓ |
| Each sample trajectory independent | Separate clones & loops per sample batch | ✓ |

**Conclusion**: Toto's architecture is **perfectly suited** for your timeseries anomaly detection use case. The model:
- ✓ Generates true probability distributions (not just point estimates)
- ✓ Samples from these distributions autoregressively
- ✓ Appends predictions to context for multi-step forecasting
- ✓ Provides uncertainty estimates crucial for AD
- ✓ Supports accessing distributions for custom anomaly scoring

---

## Next Steps for Anomaly Detection

1. **Extract distribution probabilities** - Modify forecaster to return distributions, not just samples
2. **Define anomaly score** - Use log probability, z-scores, or quantile violations
3. **Handle multi-variate anomalies** - Leverage Toto's native multivariate support
4. **Calibrate thresholds** - Use validation data to set anomaly detection thresholds
5. **Real-time inference** - Leverage KV cache for efficient streaming detection

The foundation is solid - Toto implements exactly the architecture you need!
