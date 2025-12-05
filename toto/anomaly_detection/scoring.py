#!/usr/bin/env python3
"""
NLL-based scoring for anomaly detection.

This module implements negative log-likelihood (NLL) computation for one-step-ahead
predictions on multivariate time series. The scoring works by:

1. Running the model to get the predictive distribution for t+1
2. Computing -log P(x_t+1 | x_1:t) for each variate
3. Returning per-variate NLL scores (B, V) where B=batch, V=variates

The scores can then be aggregated into a scalar anomaly score using various
aggregation functions (see detector.py).
"""

from typing import Optional

import torch
from torch import Tensor

from toto.data.util.dataset import MaskedTimeseries
from toto.model.backbone import TotoBackbone, TotoOutput


class NLLScorer:
    """
    Compute negative log-likelihood scores for one-step-ahead predictions.

    This scorer evaluates how "surprising" each timepoint is given the historical
    context. Higher NLL indicates the observation is less probable under the model,
    suggesting a potential anomaly.

    Args:
        model: The Toto backbone model (already loaded and on device)
        context_length: Number of historical timesteps to use for prediction

    Example:
        >>> scorer = NLLScorer(model, context_length=512)
        >>> nll = scorer.compute_nll(inputs, targets)
        >>> print(f"Per-variate NLL shape: {nll.shape}")  # (B, V)
    """

    def __init__(
        self,
        model: TotoBackbone,
        context_length: int = 512,
    ):
        self.model = model
        self.context_length = context_length
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def compute_nll(
        self,
        inputs: MaskedTimeseries,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute negative log-likelihood for one-step-ahead predictions.

        Args:
            inputs: Historical context as MaskedTimeseries
                    Shape: (B, V, T) where T >= context_length
            targets: Ground truth values at t+1 for each variate
                     Shape: (B, V, 1) or (B, V)

        Returns:
            nll: Negative log-likelihood for each variate
                 Shape: (B, V)

        Note:
            The model predicts a patch of size 64 timesteps, but we only extract
            the NLL for the first timestep (t+1) as per the single-timepoint
            anomaly detection strategy.
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Ensure targets have correct shape (B, V, 1)
        if targets.ndim == 2:
            targets = targets.unsqueeze(-1)  # (B, V) -> (B, V, 1)

        # Validate input shapes
        batch_size, num_variates, seq_len = inputs.series.shape
        assert seq_len >= self.context_length, \
            f"Input sequence length {seq_len} < context_length {self.context_length}"
        assert targets.shape[0] == batch_size, "Batch size mismatch"
        assert targets.shape[1] == num_variates, "Number of variates mismatch"

        # Truncate inputs to context_length if longer
        if seq_len > self.context_length:
            inputs = MaskedTimeseries(
                series=inputs.series[:, :, -self.context_length:],
                padding_mask=inputs.padding_mask[:, :, -self.context_length:],
                id_mask=inputs.id_mask[:, :, -self.context_length:],
                timestamp_seconds=inputs.timestamp_seconds[:, :, -self.context_length:],
                time_interval_seconds=inputs.time_interval_seconds,
            )

        # Forward pass: get predictive distribution for next patch
        output: TotoOutput = self.model(
            inputs=inputs.series,
            input_padding_mask=inputs.padding_mask,
            id_mask=inputs.id_mask,
        )

        # output.distribution predicts for the entire flattened sequence
        # For context_length=512 with patch_size=64, stride=64:
        #   - Input: 512 timesteps
        #   - Patches: 512/64 = 8 patches
        #   - Output distribution shape: (B, V, 512) - predicts for all timesteps
        # But the LAST patch (last 64 timesteps) is the forecast
        # We only need the FIRST timestep of this last patch (i.e., timestep t+1)

        # Get patch size from model
        patch_size = self.model.patch_embed.patch_size

        # The distribution is a MixtureSameFamily with shape (B, V, T_total, num_components)
        # We need to extract the marginal distribution for just one timestep
        # and evaluate log_prob for that single timestep

        # For MixtureSameFamily, we need to manually extract the components at the target position
        distr = output.distribution

        # Check if it's a MixtureSameFamily
        if hasattr(distr, 'component_distribution'):
            # MixtureSameFamily case
            # Extract the time dimension
            total_timesteps = distr.component_distribution.df.shape[2]  # (B, V, T, K)
            target_position = total_timesteps - patch_size  # First timestep of last patch

            # Extract components at target position
            # df, loc, scale shapes: (B, V, T, K) -> select timestep -> (B, V, K)
            df_t = distr.component_distribution.df[:, :, target_position, :]
            loc_t = distr.component_distribution.loc[:, :, target_position, :]
            scale_t = distr.component_distribution.scale[:, :, target_position, :]
            probs_t = distr.mixture_distribution.probs[:, :, target_position, :]

            # Create marginal distributions for target position
            from gluonts.torch.distributions.studentT import StudentT
            component_t = StudentT(df_t, loc_t, scale_t)
            mixture_t = torch.distributions.Categorical(probs=probs_t)
            distr_t = torch.distributions.MixtureSameFamily(mixture_t, component_t)

            # Evaluate log probability at target
            log_prob_first = distr_t.log_prob(targets.squeeze(-1))  # (B, V)

        else:
            # AffineTransformed case (StudentTOutput)
            total_timesteps = distr.base_dist.df.shape[-1]
            target_position = total_timesteps - patch_size

            # For AffineTransformed, we can slice the entire distribution
            # Extract parameters at target position
            df_t = distr.base_dist.df[:, :, target_position]
            loc_base_t = distr.base_dist.loc[:, :, target_position]
            scale_base_t = distr.base_dist.scale[:, :, target_position]
            loc_affine_t = distr.loc[:, :, target_position] if distr.loc.shape[-1] > 1 else distr.loc.squeeze(-1)
            scale_affine_t = distr.scale[:, :, target_position] if distr.scale.shape[-1] > 1 else distr.scale.squeeze(-1)

            # Create marginal distribution
            base_dist_t = torch.distributions.StudentT(df_t, loc_base_t, scale_base_t, validate_args=False)
            from gluonts.torch.distributions import AffineTransformed
            distr_t = AffineTransformed(base_dist_t, loc=loc_affine_t, scale=scale_affine_t)

            # Evaluate log probability
            log_prob_first = distr_t.log_prob(targets.squeeze(-1))  # (B, V)

        # Negative log-likelihood
        nll = -log_prob_first  # (B, V)

        return nll

    @torch.no_grad()
    def compute_nll_streaming(
        self,
        series: Tensor,
        padding_mask: Optional[Tensor] = None,
        id_mask: Optional[Tensor] = None,
        timestamp_seconds: Optional[Tensor] = None,
        time_interval_seconds: Optional[Tensor] = None,
        stride: int = 1,
    ) -> Tensor:
        """
        Compute NLL scores for a full time series in a streaming manner.

        This method slides a context window across the time series and computes
        NLL for each position t using context from t-context_length to t-1.

        Args:
            series: Time series data (B, V, T)
            padding_mask: Optional padding mask (B, V, T)
            id_mask: Optional ID mask (B, V, T)
            timestamp_seconds: Optional timestamps (B, V, T)
            time_interval_seconds: Optional intervals (B, V) or (V,)
            stride: How many timesteps to slide the window (default: 1 for dense coverage)

        Returns:
            nll_scores: NLL for each timestep that has sufficient context
                        Shape: (B, V, T_out) where T_out = (T - context_length) // stride

        Example:
            >>> # Score a full time series with stride=1 for dense coverage
            >>> nll = scorer.compute_nll_streaming(series, stride=1)
            >>> # Find anomalies above threshold
            >>> anomalies = nll > threshold  # (B, V, T_out)
        """
        batch_size, num_variates, seq_len = series.shape

        # Validate we have enough data
        assert seq_len > self.context_length, \
            f"Sequence length {seq_len} must be > context_length {self.context_length}"

        # Create default masks if not provided
        if padding_mask is None:
            padding_mask = torch.ones_like(series, dtype=torch.bool)
        if id_mask is None:
            id_mask = torch.zeros_like(series)
        if timestamp_seconds is None:
            timestamp_seconds = torch.zeros_like(series)
        if time_interval_seconds is None:
            time_interval_seconds = torch.ones(num_variates, device=series.device)

        # Ensure time_interval_seconds has correct shape
        if time_interval_seconds.ndim == 1:
            time_interval_seconds = time_interval_seconds.unsqueeze(0).expand(batch_size, -1)

        # Compute number of output timesteps
        num_outputs = (seq_len - self.context_length) // stride

        # Allocate output tensor
        nll_scores = torch.zeros(
            (batch_size, num_variates, num_outputs),
            device=series.device,
            dtype=series.dtype,
        )

        # Slide window across time series
        for i in range(num_outputs):
            t = self.context_length + i * stride

            # Extract context window
            context_start = t - self.context_length
            context_end = t

            inputs = MaskedTimeseries(
                series=series[:, :, context_start:context_end],
                padding_mask=padding_mask[:, :, context_start:context_end],
                id_mask=id_mask[:, :, context_start:context_end],
                timestamp_seconds=timestamp_seconds[:, :, context_start:context_end],
                time_interval_seconds=time_interval_seconds,
            )

            # Target is the value at position t
            targets = series[:, :, t]  # (B, V)

            # Compute NLL for this timestep
            nll_scores[:, :, i] = self.compute_nll(inputs, targets)

        return nll_scores
