#!/usr/bin/env python3
"""
Main anomaly detection interface integrating scoring, aggregation, and thresholding.

This module provides the AnomalyDetector class, which is the primary interface for
performing anomaly detection on multivariate time series using the Toto model.

The detector workflow:
1. **Fit**: Learn threshold from normal training data
   - Compute per-variate NLL scores
   - Aggregate to scalar scores (L2-norm by default)
   - Estimate 95th percentile threshold

2. **Detect**: Classify test data as normal/anomalous
   - Compute per-variate NLL scores
   - Aggregate to scalar scores
   - Compare to threshold
"""

from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from toto.data.util.dataset import MaskedTimeseries
from toto.model.backbone import TotoBackbone

from .scoring import NLLScorer
from .threshold import ThresholdEstimator


class AnomalyDetector:
    """
    Anomaly detector for multivariate time series using Toto foundation model.

    This detector uses negative log-likelihood (NLL) as the anomaly score. It learns
    a threshold from normal training data and classifies test points as anomalous
    if their aggregated NLL score exceeds the threshold.

    Args:
        model: Toto backbone model (already loaded and on device)
        context_length: Number of historical timesteps for prediction (default: 512)
        aggregation: Method to aggregate per-variate NLL to scalar score
        Options: 'l2' (default), 'mean', 'max', 'sum', 'topk'
        threshold_method: Method for threshold estimation (default: 'percentile')
        threshold_percentile: Percentile for threshold (default: 95.0)
        topk: Number of variates for 'topk' aggregation (default: 3)
        weights: Optional weights for 'weighted' aggregation (V,)

    Example:
        >>> # Load model
        >>> from toto.model.toto import Toto
        >>> toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
        >>> toto.to(device)
        >>> toto.eval()
        >>>
        >>> # Create detector
        >>> detector = AnomalyDetector(toto.model, context_length=512, aggregation='l2')
        >>>
        >>> # Fit on normal training data
        >>> detector.fit(train_data)
        >>>
        >>> # Detect anomalies in test data
        >>> is_anomaly = detector.detect(test_data)
        >>> anomaly_scores = detector.score(test_data)
    """

    def __init__(
        self,
        model: TotoBackbone,
        context_length: int = 512,
        aggregation: Literal['l2', 'mean', 'max', 'sum', 'topk', 'weighted'] = 'l2',
        threshold_method: Literal['percentile', 'mean_std', 'mad'] = 'percentile',
        threshold_percentile: float = 95.0,
        topk: int = 3,
        weights: Optional[Tensor] = None,
    ):
        self.model = model
        self.context_length = context_length
        self.aggregation = aggregation
        self.topk = topk
        self.weights = weights

        # Initialize scorer
        self.scorer = NLLScorer(model, context_length=context_length)

        # Initialize threshold estimator
        self.threshold_estimator = ThresholdEstimator(
            method=threshold_method,
            percentile=threshold_percentile,
        )

        self._fitted = False

    def fit(
        self,
        series: Tensor,
        padding_mask: Optional[Tensor] = None,
        id_mask: Optional[Tensor] = None,
        timestamp_seconds: Optional[Tensor] = None,
        time_interval_seconds: Optional[Tensor] = None,
        stride: int = 1,
    ) -> "AnomalyDetector":
        """
        Fit the anomaly detector on normal (non-anomalous) training data.

        This computes anomaly scores for the training data and estimates the
        threshold as the 95th percentile (or other configured percentile).

        Args:
            series: Training time series (B, V, T) - should contain only normal data
            padding_mask: Optional padding mask (B, V, T)
            id_mask: Optional ID mask (B, V, T)
            timestamp_seconds: Optional timestamps (B, V, T)
            time_interval_seconds: Optional intervals (B, V) or (V,)
            stride: Stride for sliding window (default: 1)

        Returns:
            self: For method chaining

        Raises:
            ValueError: If training data is too short
        """
        # Compute per-variate NLL scores for training data
        nll_scores = self.scorer.compute_nll_streaming(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
            stride=stride,
        )  # (B, V, T_out)

        # Aggregate per-variate NLL to scalar scores
        # nll_scores shape: (B, V, T_out)
        # We want to aggregate over V dimension to get (B, T_out)
        aggregated_scores = self._aggregate_nll(nll_scores)  # (B, T_out)

        # Flatten to (N,) for threshold estimation
        train_scores = aggregated_scores.flatten()

        # Fit threshold estimator
        self.threshold_estimator.fit(train_scores)

        self._fitted = True

        return self

    def detect(
        self,
        series: Tensor,
        padding_mask: Optional[Tensor] = None,
        id_mask: Optional[Tensor] = None,
        timestamp_seconds: Optional[Tensor] = None,
        time_interval_seconds: Optional[Tensor] = None,
        stride: int = 1,
        return_scores: bool = False,
    ) -> Tensor:
        """
        Detect anomalies in test time series.

        Args:
            series: Test time series (B, V, T)
            padding_mask: Optional padding mask (B, V, T)
            id_mask: Optional ID mask (B, V, T)
            timestamp_seconds: Optional timestamps (B, V, T)
            time_interval_seconds: Optional intervals (B, V) or (V,)
            stride: Stride for sliding window (default: 1)
            return_scores: If True, return (is_anomaly, scores) tuple

        Returns:
            is_anomaly: Boolean tensor indicating anomalies (B, T_out)
            scores: (Optional) Anomaly scores if return_scores=True

        Raises:
            RuntimeError: If detector has not been fitted
        """
        if not self._fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        # Compute anomaly scores
        scores = self.score(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
            stride=stride,
        )  # (B, T_out)

        # Classify using threshold
        is_anomaly = self.threshold_estimator.predict(scores)

        if return_scores:
            return is_anomaly, scores
        return is_anomaly

    def score(
        self,
        series: Tensor,
        padding_mask: Optional[Tensor] = None,
        id_mask: Optional[Tensor] = None,
        timestamp_seconds: Optional[Tensor] = None,
        time_interval_seconds: Optional[Tensor] = None,
        stride: int = 1,
    ) -> Tensor:
        """
        Compute anomaly scores for time series (without thresholding).

        Args:
            series: Time series data (B, V, T)
            padding_mask: Optional padding mask (B, V, T)
            id_mask: Optional ID mask (B, V, T)
            timestamp_seconds: Optional timestamps (B, V, T)
            time_interval_seconds: Optional intervals (B, V) or (V,)
            stride: Stride for sliding window (default: 1)

        Returns:
            scores: Aggregated anomaly scores (B, T_out)
        """
        # Compute per-variate NLL scores
        nll_scores = self.scorer.compute_nll_streaming(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
            stride=stride,
        )  # (B, V, T_out)

        # Aggregate to scalar scores
        aggregated_scores = self._aggregate_nll(nll_scores)  # (B, T_out)

        return aggregated_scores

    def _aggregate_nll(self, nll: Tensor) -> Tensor:
        """
        Aggregate per-variate NLL scores to scalar anomaly scores.

        Args:
            nll: Per-variate NLL scores (B, V, T_out)

        Returns:
            aggregated: Scalar scores (B, T_out)
        """
        # Aggregate over V dimension
        if self.aggregation == 'l2':
            # L2-norm (Euclidean distance) - RECOMMENDED DEFAULT
            # sqrt(sum of squares) - balanced between robust and sensitive
            aggregated = nll.norm(p=2, dim=1)  # (B, T_out)

        elif self.aggregation == 'mean':
            # Mean - scale-invariant, robust but may dilute signal
            aggregated = nll.mean(dim=1)

        elif self.aggregation == 'max':
            # Max - most sensitive, detects any single anomalous variate
            aggregated = nll.max(dim=1)[0]

        elif self.aggregation == 'sum':
            # Sum - theoretically principled if variates are independent
            # but sensitive to number of variates
            aggregated = nll.sum(dim=1)

        elif self.aggregation == 'topk':
            # Top-K mean - average of K most anomalous variates
            # Robust to noise in majority of variates
            if self.topk > nll.shape[1]:
                # Fall back to mean if K > V
                aggregated = nll.mean(dim=1)
            else:
                topk_values = nll.topk(k=self.topk, dim=1)[0]  # (B, K, T_out)
                aggregated = topk_values.mean(dim=1)  # (B, T_out)

        elif self.aggregation == 'weighted':
            # Weighted sum - requires domain knowledge to set weights
            if self.weights is None:
                raise ValueError("Weights must be provided for 'weighted' aggregation")
            if self.weights.shape[0] != nll.shape[1]:
                raise ValueError(
                    f"Weights dimension {self.weights.shape[0]} != "
                    f"number of variates {nll.shape[1]}"
                )
            # weights shape: (V,), nll shape: (B, V, T_out)
            weights_expanded = self.weights.unsqueeze(0).unsqueeze(-1)  # (1, V, 1)
            aggregated = (nll * weights_expanded).sum(dim=1)  # (B, T_out)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return aggregated
    
        #redundant remove
    # def fit_detect(
    #     self,
    #     train_series: Tensor,
    #     test_series: Tensor,
    #     train_kwargs: Optional[Dict] = None,
    #     test_kwargs: Optional[Dict] = None,
    #     return_scores: bool = False,
    # ) -> Tensor:
    #     """
    #     Fit on training data and detect anomalies in test data (convenience method).

    #     Args:
    #         train_series: Training time series (B_train, V, T_train)
    #         test_series: Test time series (B_test, V, T_test)
    #         train_kwargs: Optional kwargs for fit() (padding_mask, etc.)
    #         test_kwargs: Optional kwargs for detect() (padding_mask, etc.)
    #         return_scores: If True, return (is_anomaly, scores) tuple

    #     Returns:
    #         is_anomaly: Boolean tensor for test data
    #         scores: (Optional) If return_scores=True
    #     """
    #     train_kwargs = train_kwargs or {}
    #     test_kwargs = test_kwargs or {}

    #     self.fit(train_series, **train_kwargs)
    #     return self.detect(test_series, return_scores=return_scores, **test_kwargs)

    @property
    def threshold(self) -> Tensor:
        """Get the fitted threshold value."""
        if not self._fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        return self.threshold_estimator.threshold

    @property
    def is_fitted(self) -> bool:
        """Check if detector has been fitted."""
        return self._fitted
