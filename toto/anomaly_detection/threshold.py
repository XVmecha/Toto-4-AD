#!/usr/bin/env python3
"""
Threshold estimation for anomaly detection.

This module implements methods for estimating anomaly thresholds from training data
that contains only normal (non-anomalous) behavior. The threshold is used to classify
test data as normal or anomalous based on anomaly scores.

The primary method is percentile-based thresholding, where the threshold is set to
the Pth percentile of anomaly scores computed on normal training data (e.g., P=95).
"""

from typing import Literal, Optional

import torch
from torch import Tensor


class ThresholdEstimator:
    """
    Estimate anomaly detection thresholds from normal training data.

    The estimator computes a threshold from anomaly scores (e.g., aggregated NLL)
    obtained from training data that contains only normal behavior. Common approaches:

    - **Percentile**: Set threshold to Pth percentile (e.g., 95th)
    - **Mean + k*Std**: Set threshold to mean + k * std (e.g., k=3)
    - **MAD**: Median Absolute Deviation (robust to outliers)

    Args:
        method: Thresholding method ('percentile', 'mean_std', 'mad')
        percentile: Percentile value for 'percentile' method (0-100, default: 95)
        n_std: Number of standard deviations for 'mean_std' method (default: 3)

    Example:
        >>> estimator = ThresholdEstimator(method='percentile', percentile=95)
        >>> threshold = estimator.fit(train_scores)
        >>> is_anomaly = test_scores > threshold
    """

    def __init__(
        self,
        method: Literal['percentile', 'mean_std', 'mad'] = 'percentile',
        percentile: float = 95.0,
        n_std: float = 3.0,
    ):
        self.method = method
        self.percentile = percentile
        self.n_std = n_std
        self.threshold_: Optional[Tensor] = None
        self._fitted = False

    def fit(self, scores: Tensor) -> Tensor:
        """
        Fit the threshold estimator on training anomaly scores.

        Args:
            scores: Anomaly scores from normal training data
                    Shape: (N,) where N is number of training samples
                    These should be scalar scores (already aggregated across variates)

        Returns:
            threshold: The computed threshold value (scalar tensor)

        Raises:
            ValueError: If scores contain NaN or infinite values
        """
        # Validate input
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            raise ValueError("Scores contain NaN or infinite values")

        if scores.numel() == 0:
            raise ValueError("Cannot fit threshold on empty scores")

        # Flatten scores if multidimensional
        scores_flat = scores.flatten()

        # Compute threshold based on method
        if self.method == 'percentile':
            threshold = self._percentile_threshold(scores_flat)
        elif self.method == 'mean_std':
            threshold = self._mean_std_threshold(scores_flat)
        elif self.method == 'mad':
            threshold = self._mad_threshold(scores_flat)
        else:
            raise ValueError(f"Unknown threshold method: {self.method}")

        self.threshold_ = threshold
        self._fitted = True

        return threshold

    def predict(self, scores: Tensor) -> Tensor:
        """
        Predict anomalies using the fitted threshold.

        Args:
            scores: Anomaly scores to classify
                    Shape: (N,) or any shape - returns same shape

        Returns:
            is_anomaly: Boolean tensor indicating anomalies (True = anomaly)
                        Same shape as input scores

        Raises:
            RuntimeError: If threshold has not been fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Threshold has not been fitted. Call fit() first.")

        return scores > self.threshold_

    def fit_predict(self, train_scores: Tensor, test_scores: Tensor) -> Tensor:
        """
        Fit threshold on training scores and predict on test scores.

        Args:
            train_scores: Normal training scores (N_train,)
            test_scores: Test scores to classify (N_test,) or any shape

        Returns:
            is_anomaly: Boolean tensor for test_scores
        """
        self.fit(train_scores)
        return self.predict(test_scores)

    def _percentile_threshold(self, scores: Tensor) -> Tensor:
        """
        Compute threshold as the Pth percentile of scores.

        For P=95, approximately 5% of normal data will exceed the threshold,
        providing a natural false positive rate.
        """
        # torch.quantile requires percentile in [0, 1]
        quantile = self.percentile / 100.0
        threshold = torch.quantile(scores, quantile)
        return threshold

    def _mean_std_threshold(self, scores: Tensor) -> Tensor:
        """
        Compute threshold as mean + k * std.

        This assumes scores are approximately Gaussian. For k=3, approximately
        99.7% of normal data falls below threshold (3-sigma rule).
        """
        mean = scores.mean()
        std = scores.std()
        threshold = mean + self.n_std * std
        return threshold

    def _mad_threshold(self, scores: Tensor) -> Tensor:
        """
        Compute threshold using Median Absolute Deviation (MAD).

        MAD is robust to outliers and works well when training data may contain
        a small fraction of anomalies. Threshold = median + k * MAD, where
        k is derived from n_std for consistency.
        """
        median = scores.median()
        mad = torch.median(torch.abs(scores - median))

        # Scale MAD to match standard deviation for Gaussian
        # For normal distribution: std ≈ 1.4826 * MAD
        scaled_mad = 1.4826 * mad

        # Use n_std to compute threshold
        threshold = median + self.n_std * scaled_mad
        return threshold

    @property
    def threshold(self) -> Tensor:
        """Get the fitted threshold value."""
        if not self._fitted:
            raise RuntimeError("Threshold has not been fitted. Call fit() first.")
        return self.threshold_