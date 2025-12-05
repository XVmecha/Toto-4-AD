#!/usr/bin/env python3
"""
Run Toto Anomaly Detection on Preprocessed SWaT Dataset

This script performs anomaly detection on the full preprocessed SWaT dataset
using the Toto foundation model.

Steps:
1. Load preprocessed SWaT data
2. Load Toto-Open-Base-1.0 model
3. Fit anomaly detector on training data (normal operations)
4. Detect anomalies in test data (contains attacks)
5. Evaluate detection performance
6. Save results and visualizations

Usage:
    python run_swat_anomaly_detection.py [options]
"""

# CRITICAL: Enable MPS fallback BEFORE importing torch
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toto.model.toto import Toto
from toto.model.util import get_device
from toto.anomaly_detection import AnomalyDetector


def load_preprocessed_swat(data_dir: str):
    """Load preprocessed SWaT data."""
    data_path = Path(data_dir)

    print("Loading preprocessed SWaT data...")

    # Load training data
    train_data = torch.load(data_path / 'swat_train.pt', weights_only=False)
    train_series = train_data['series']
    train_labels = train_data['labels']

    # Load test data
    test_data = torch.load(data_path / 'swat_test.pt', weights_only=False)
    test_series = test_data['series']
    test_labels = test_data['labels']

    # Load metadata
    train_metadata = pd.read_csv(data_path / 'swat_train_metadata.csv')
    test_metadata = pd.read_csv(data_path / 'swat_test_metadata.csv')

    print(f"  Train: {train_series.shape} (all normal)")
    print(f"  Test: {test_series.shape}")
    print(f"    - Normal: {(test_labels == 0).sum().item():,} ({(test_labels == 0).sum().item() / test_labels.numel() * 100:.1f}%)")
    print(f"    - Attack: {(test_labels == 1).sum().item():,} ({(test_labels == 1).sum().item() / test_labels.numel() * 100:.1f}%)")

    return train_series, train_labels, test_series, test_labels, train_metadata, test_metadata


def evaluate_detection(predictions, ground_truth, context_length):
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted anomalies (1, T_pred)
        ground_truth: True anomalies (1, T_total)
        context_length: Offset for aligning predictions
    """
    # Align predictions with ground truth
    offset = context_length
    pred_length = predictions.shape[1]

    pred_flat = predictions[0].cpu().numpy()
    true_flat = ground_truth[0, offset:offset + pred_length].cpu().numpy()

    # Compute metrics
    tp = np.sum(pred_flat & true_flat)
    fp = np.sum(pred_flat & ~true_flat)
    fn = np.sum(~pred_flat & true_flat)
    tn = np.sum(~pred_flat & ~true_flat)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def plot_results(
    test_metadata,
    anomaly_scores,
    predictions,
    ground_truth,
    threshold,
    context_length,
    output_dir,
):
    """Plot anomaly detection results."""
    print("\nGenerating visualizations...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Align data
    offset = context_length
    pred_length = anomaly_scores.shape[1]

    scores_data = anomaly_scores[0].cpu().numpy()
    pred_data = predictions[0].cpu().numpy()
    true_data = ground_truth[0, offset:offset + pred_length].cpu().numpy()

    # Get timestamps
    timestamps = pd.to_datetime(test_metadata['timestamp'].iloc[offset:offset + pred_length])

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Anomaly scores with threshold
    axes[0].plot(timestamps, scores_data, 'b-', alpha=0.6, linewidth=0.5, label='Anomaly Score')
    axes[0].axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')

    # Highlight detected anomalies
    axes[0].fill_between(
        timestamps,
        threshold,
        scores_data,
        where=(scores_data > threshold),
        alpha=0.3,
        color='red',
        label='Detected Anomalies'
    )

    axes[0].set_ylabel('Anomaly Score (L2-norm of NLL)', fontsize=12)
    axes[0].set_title('SWaT Anomaly Detection: Anomaly Scores Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Detection comparison
    # Show ground truth as background
    for i, (ts, val) in enumerate(zip(timestamps, true_data)):
        if val:
            axes[1].axvspan(ts, ts, alpha=0.3, color='red', linewidth=0)

    # Overlay predictions
    pred_indices = np.where(pred_data)[0]
    true_indices = np.where(true_data)[0]

    if len(pred_indices) > 0:
        axes[1].scatter(timestamps[pred_indices], np.ones(len(pred_indices)),
                       c='blue', s=10, marker='|', alpha=0.7, label='Predicted Anomalies')
    if len(true_indices) > 0:
        axes[1].scatter(timestamps[true_indices], np.zeros(len(true_indices)),
                       c='red', s=10, marker='|', alpha=0.7, label='True Anomalies')

    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Ground Truth', 'Predictions'], fontsize=11)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_title('Anomaly Detection Comparison: Predicted vs Ground Truth', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_ylim(-0.5, 1.5)

    plt.tight_layout()

    output_file = output_path / 'swat_anomaly_detection_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Plot score distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    normal_scores = scores_data[~true_data]
    attack_scores = scores_data[true_data]

    bins = 50
    ax.hist(normal_scores, bins=bins, alpha=0.5, label='Normal', color='blue', density=True)
    ax.hist(attack_scores, bins=bins, alpha=0.5, label='Attack', color='red', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')

    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('SWaT Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_path / 'swat_score_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def save_results(metrics, threshold, config, output_dir):
    """Save detection results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': 'SWaT',
        'model': 'Toto-Open-Base-1.0',
        'configuration': config,
        'threshold': float(threshold),
        'metrics': metrics,
    }

    output_file = output_path / 'swat_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run Toto anomaly detection on SWaT dataset')
    parser.add_argument('--data_dir', type=str,
                        default='toto/data/preprocessed_datasets/swat',
                        help='Directory containing preprocessed SWaT data')
    parser.add_argument('--output_dir', type=str,
                        default='toto/results/swat',
                        help='Output directory for results')
    parser.add_argument('--context_length', type=int, default=512,
                        help='Context length for model (default: 512)')
    parser.add_argument('--aggregation', type=str, default='l2',
                        choices=['l2', 'mean', 'max', 'sum'],
                        help='Aggregation method for multivariate scores (default: l2)')
    parser.add_argument('--threshold_percentile', type=float, default=95.0,
                        help='Threshold percentile (default: 95.0)')
    parser.add_argument('--fit_stride', type=int, default=32,
                        help='Stride for fitting threshold on training data (default: 32)')
    parser.add_argument('--detect_stride', type=int, default=1,
                        help='Stride for detection on test data (default: 1)')

    args = parser.parse_args()

    print("=" * 70)
    print("Toto Anomaly Detection on SWaT Dataset")
    print("=" * 70)

    # Configuration
    config = {
        'context_length': args.context_length,
        'aggregation': args.aggregation,
        'threshold_percentile': args.threshold_percentile,
        'fit_stride': args.fit_stride,
        'detect_stride': args.detect_stride,
    }

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Device setup
    device = get_device()
    print(f"\nUsing device: {device}")

    # Load preprocessed data
    print(f"\n{'=' * 70}")
    print("Step 1: Loading Preprocessed SWaT Data")
    print(f"{'=' * 70}\n")

    train_series, train_labels, test_series, test_labels, train_metadata, test_metadata = \
        load_preprocessed_swat(args.data_dir)

    # Move to device
    train_series = train_series.to(device)
    test_series = test_series.to(device)

    # Load Toto model
    print(f"\n{'=' * 70}")
    print("Step 2: Loading Toto Model")
    print(f"{'=' * 70}\n")

    print("Loading Toto-Open-Base-1.0...")
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to(device)
    toto.eval()
    print(f"Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in toto.model.parameters()):,}")

    # Create anomaly detector
    print(f"\n{'=' * 70}")
    print("Step 3: Creating Anomaly Detector")
    print(f"{'=' * 70}\n")

    detector = AnomalyDetector(
        model=toto.model,
        context_length=args.context_length,
        aggregation=args.aggregation,
        threshold_method='percentile',
        threshold_percentile=args.threshold_percentile,
    )
    print(f"Detector created")

    # Fit on normal training data
    print(f"\n{'=' * 70}")
    print("Step 4: Fitting Detector on Training Data")
    print(f"{'=' * 70}\n")

    print(f"Computing NLL scores on training data...")
    print(f"  Training shape: {train_series.shape}")
    print(f"  Using stride={args.fit_stride} for threshold estimation...")

    detector.fit(train_series, stride=args.fit_stride)
    print(f"\n✓ Threshold fitted: {detector.threshold.item():.4f}")

    # Detect anomalies in test data
    print(f"\n{'=' * 70}")
    print("Step 5: Detecting Anomalies in Test Data")
    print(f"{'=' * 70}\n")

    print(f"Computing anomaly scores on test data...")
    print(f"  Test shape: {test_series.shape}")
    print(f"  Using stride={args.detect_stride} for detection...")

    is_anomaly, anomaly_scores = detector.detect(
        test_series,
        stride=args.detect_stride,
        return_scores=True,
    )

    num_detected = is_anomaly.sum().item()
    print(f"\n✓ Detected {num_detected:,} anomalies out of {is_anomaly.numel():,} timesteps")
    print(f"  Detection rate: {num_detected / is_anomaly.numel() * 100:.2f}%")

    # Evaluate performance
    print(f"\n{'=' * 70}")
    print("Step 6: Evaluating Detection Performance")
    print(f"{'=' * 70}\n")

    metrics = evaluate_detection(is_anomaly, test_labels, args.context_length)

    print(f"Confusion Matrix:")
    print(f"  True Positives:  {metrics['tp']:,}")
    print(f"  False Positives: {metrics['fp']:,}")
    print(f"  False Negatives: {metrics['fn']:,}")
    print(f"  True Negatives:  {metrics['tn']:,}")

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    # Save results
    print(f"\n{'=' * 70}")
    print("Step 7: Saving Results")
    print(f"{'=' * 70}\n")

    save_results(metrics, detector.threshold.item(), config, args.output_dir)
    plot_results(
        test_metadata,
        anomaly_scores,
        is_anomaly,
        test_labels,
        detector.threshold.item(),
        args.context_length,
        args.output_dir,
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("ANOMALY DETECTION COMPLETE!")
    print(f"{'=' * 70}\n")

    print(f"Dataset: SWaT (Secure Water Treatment)")
    print(f"Model: Toto-Open-Base-1.0 (151M parameters)")
    print(f"Configuration:")
    print(f"  - Context length: {args.context_length}")
    print(f"  - Aggregation: {args.aggregation}")
    print(f"  - Threshold: {detector.threshold.item():.4f} ({args.threshold_percentile}th percentile)")
    print(f"\nResults:")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"\nOutput directory: {args.output_dir}")

    if metrics['f1'] >= 0.7:
        print("\n✓ Excellent detection performance!")
    elif metrics['f1'] >= 0.5:
        print("\n✓ Good detection performance!")
    elif metrics['f1'] >= 0.3:
        print("\n⚠ Moderate detection performance")
    else:
        print("\n⚠ Low detection performance - consider tuning hyperparameters")


if __name__ == '__main__':
    main()
