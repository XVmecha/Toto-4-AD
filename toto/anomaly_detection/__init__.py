#!/usr/bin/env python3
"""
Anomaly Detection Module for Toto

This module provides NLL-based anomaly detection for multivariate time series
using the Toto foundation model.

Quick Start:
    >>> from toto.model.toto import Toto
    >>> from toto.anomaly_detection import AnomalyDetector
    >>>
    >>> # Load model
    >>> toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    >>> toto.to(device)
    >>> toto.eval()
    >>>
    >>> # Create detector with L2-norm aggregation (default)
    >>> detector = AnomalyDetector(toto.model, context_length=512)
    >>>
    >>> # Fit on normal training data
    >>> detector.fit(train_series)  # (B, V, T)
    >>>
    >>> # Detect anomalies in test data
    >>> is_anomaly = detector.detect(test_series)  # (B, T_out)

Main Classes:
    - AnomalyDetector: High-level interface for anomaly detection
    - NLLScorer: Compute negative log-likelihood scores
    - ThresholdEstimator: Estimate thresholds from normal data

Aggregation Methods:
    - 'l2': L2-norm (Euclidean) - RECOMMENDED DEFAULT
    - 'mean': Average across variates
    - 'max': Maximum across variates (most sensitive)
    - 'sum': Sum across variates
    - 'topk': Average of top-K most anomalous variates
    - 'weighted': Weighted sum (requires domain knowledge)

Threshold Methods:
    - 'percentile': Pth percentile (default: 95th)
    - 'mean_std': Mean + k * std (default: k=3)
    - 'mad': Median Absolute Deviation (robust)
"""

from .detector import AnomalyDetector
from .scoring import NLLScorer
from .threshold import ThresholdEstimator

__all__ = [
    "AnomalyDetector",
    "NLLScorer",
    "ThresholdEstimator",
]

__version__ = "1.0.0"
