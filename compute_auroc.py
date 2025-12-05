#!/usr/bin/env python3
"""
Compute AUROC for anomaly detection experiments.

AUROC measures ranking quality independent of threshold choice.
This addresses the threshold-setting failure in our zero-shot approach.

Usage:
    python compute_auroc.py swat
    python compute_auroc.py smd
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import json
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from toto.model.toto import Toto
from toto.model.util import get_device
from toto.anomaly_detection.detector import AnomalyDetector


def compute_auroc_for_dataset(dataset_name, aggregation='mean'):
    """Compute AUROC for a dataset by regenerating anomaly scores."""

    print("="*70)
    print(f"Computing AUROC: {dataset_name.upper()} (Aggregation: {aggregation})")
    print("="*70)

    # Load model
    device = get_device()
    print(f"Device: {device}")

    print("\nLoading Toto model...")
    toto = Toto.from_pretrained('Datadog/toto-open-base-1.0')
    toto.to(device)
    model = toto.model

    # Load data
    context_length = 512
    threshold_percentile = 95.0

    if dataset_name == 'swat':
        train_path = Path('toto/data/preprocessed_datasets/swat/swat_train.pt')
        test_path = Path('toto/data/preprocessed_datasets/swat/swat_test.pt')
        fit_stride = 32
        detect_stride = 1
    elif dataset_name == 'smd':
        train_path = Path('toto/data/preprocessed_smd_1x/smd_train.pt')
        test_path = Path('toto/data/preprocessed_smd_1x/smd_test.pt')
        fit_stride = 32
        detect_stride = 32
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nLoading {dataset_name.upper()} data...")
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    train_series = train_data['series']
    test_series = test_data['series']
    test_labels = test_data['labels']

    print(f"  Train: {train_series.shape}")
    print(f"  Test: {test_series.shape}")
    print(f"  Labels: {test_labels.shape}")

    # Initialize detector
    print(f"\nInitializing detector (aggregation={aggregation})...")
    detector = AnomalyDetector(
        model=model,
        context_length=context_length,
        aggregation=aggregation,
        threshold_percentile=threshold_percentile,
    )

    # Fit on training data
    print(f"\nFitting on training data (stride={fit_stride})...")
    detector.fit(train_series.to(device), stride=fit_stride)
    print(f"  Threshold (p{threshold_percentile}): {detector.threshold:.4f}")

    # Get anomaly scores on test data
    print(f"\nComputing anomaly scores on test data (stride={detect_stride})...")
    is_anomaly, anomaly_scores = detector.detect(
        test_series.to(device),
        stride=detect_stride,
        return_scores=True,
    )

    anomaly_scores_np = anomaly_scores.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    print(f"  Anomaly scores shape: {anomaly_scores_np.shape}")
    print(f"  Labels shape: {test_labels_np.shape}")

    # Align labels with scores (same logic as detection scripts)
    if dataset_name == 'swat':
        # SWaT: (1, timesteps) scores vs (1, timesteps) labels
        # Already aligned after sliding window
        scores_flat = anomaly_scores_np[0]  # (timesteps,)
        labels_flat = test_labels_np[0, :len(scores_flat)]  # Truncate to match
    else:
        # SMD: (28, timesteps) scores vs (28, timesteps) labels
        # Flatten across all machines
        scores_flat = anomaly_scores_np.flatten()
        labels_flat = test_labels_np[:, :anomaly_scores_np.shape[1]].flatten()

    # Ensure same length
    min_len = min(len(scores_flat), len(labels_flat))
    scores_flat = scores_flat[:min_len]
    labels_flat = labels_flat[:min_len]

    print(f"\n  Aligned data: {len(scores_flat)} points")
    print(f"  Normal: {(labels_flat == 0).sum()}")
    print(f"  Anomaly: {(labels_flat == 1).sum()}")

    # Compute AUROC
    if (labels_flat == 0).all() or (labels_flat == 1).all():
        print("\n  ERROR: All labels are the same class. Cannot compute AUROC.")
        auroc = None
    else:
        auroc = roc_auc_score(labels_flat, scores_flat)
        print(f"\n{'='*70}")
        print(f"AUROC: {auroc:.4f}")
        print(f"{'='*70}")

        # Also compute ROC curve for analysis
        fpr, tpr, thresholds = roc_curve(labels_flat, scores_flat)

        # Find threshold at 95% specificity (5% FPR)
        idx_95_spec = np.argmin(np.abs(fpr - 0.05))
        thresh_95_spec = thresholds[idx_95_spec]
        tpr_at_95_spec = tpr[idx_95_spec]

        print(f"\nAt 95% Specificity (5% FPR):")
        print(f"  Threshold: {thresh_95_spec:.4f}")
        print(f"  TPR (Recall): {tpr_at_95_spec:.2%}")

    # Save results
    output_dir = Path('toto/results/auroc')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': dataset_name,
        'aggregation': aggregation,
        'context_length': context_length,
        'fit_stride': fit_stride,
        'detect_stride': detect_stride,
        'threshold_percentile': threshold_percentile,
        'threshold': float(detector.threshold),
        'n_samples': int(min_len),
        'n_normal': int((labels_flat == 0).sum()),
        'n_anomaly': int((labels_flat == 1).sum()),
        'auroc': float(auroc) if auroc is not None else None,
    }

    if auroc is not None:
        results['threshold_at_95_specificity'] = float(thresh_95_spec)
        results['tpr_at_95_specificity'] = float(tpr_at_95_spec)

    output_file = output_dir / f'{dataset_name}_{aggregation}_auroc.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_auroc.py {swat|smd} [aggregation]")
        sys.exit(1)

    dataset = sys.argv[1].lower()
    aggregation = sys.argv[2] if len(sys.argv) > 2 else 'mean'

    if dataset not in ['swat', 'smd']:
        print(f"Error: Unknown dataset '{dataset}'. Use 'swat' or 'smd'.")
        sys.exit(1)

    compute_auroc_for_dataset(dataset, aggregation)


if __name__ == '__main__':
    main()
