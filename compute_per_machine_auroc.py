#!/usr/bin/env python3
"""
Per-machine anomaly detection evaluation for SMD.

Each SMD machine is an independent series with its own NLL scale, so a single
global threshold (and a pooled ROC across all machines) mixes incomparable
distributions. This script instead:

  * fits one threshold per machine (95th percentile of that machine's own
    normal/training scores), and
  * computes one AUROC per machine on that machine's own scores,

then reports the macro-average (mean over machines) alongside the per-machine
values. Pooled (micro-average) AUROC is also reported for comparison.

Usage:
    python compute_per_machine_auroc.py            # both mean and max
    python compute_per_machine_auroc.py mean
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from toto.model.toto import Toto
from toto.model.util import get_device
from toto.anomaly_detection.scoring import NLLScorer


CONTEXT_LENGTH = 512
THRESHOLD_PERCENTILE = 95.0
FIT_STRIDE = 32
DETECT_STRIDE = 8
TRAIN_PATH = Path('toto/data/preprocessed_smd_1x/smd_train.pt')
TEST_PATH = Path('toto/data/preprocessed_smd_1x/smd_test.pt')


def aggregate(nll, method):
    """Aggregate per-variate NLL (M, V, T) -> (M, T)."""
    if method == 'mean':
        return nll.mean(axis=1)
    if method == 'max':
        return nll.max(axis=1)
    raise ValueError(method)


def per_machine_metrics(train_scores, test_scores, aligned_labels):
    """Per-machine threshold, AUROC, and F1. Inputs are numpy (M, T*)."""
    n_machines = train_scores.shape[0]
    rows = []
    for m in range(n_machines):
        tr = train_scores[m]
        te = test_scores[m]
        labels = aligned_labels[m].astype(bool)

        threshold = float(np.percentile(tr, THRESHOLD_PERCENTILE))
        pred = te > threshold

        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        tn = int(np.sum(~pred & ~labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUROC only defined when both classes are present for this machine
        if labels.any() and not labels.all():
            auroc = float(roc_auc_score(labels, te))
        else:
            auroc = None

        rows.append({
            'machine_id': m,
            'threshold': threshold,
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_anomaly': int(labels.sum()),
        })
    return rows


def main():
    aggregations = [sys.argv[1]] if len(sys.argv) > 1 else ['mean', 'max']

    # Raw per-variate NLL cache. Computing it requires the model; once saved we
    # never need the model again (re-aggregation, re-thresholding, resume after
    # an interrupted run). This also satisfies the "use saved score files" goal.
    scores_dir = Path('toto/results/scores')
    scores_dir.mkdir(parents=True, exist_ok=True)
    train_cache = scores_dir / f'smd_nll_train_stride{FIT_STRIDE}.npy'
    test_cache = scores_dir / f'smd_nll_test_stride{DETECT_STRIDE}.npy'

    # Test labels are always needed (cheap, no model)
    test_data = torch.load(TEST_PATH, weights_only=False)
    test_labels = test_data['labels'].cpu().numpy()

    if train_cache.exists() and test_cache.exists():
        print(f"Loading cached NLL scores (skipping model):\n  {train_cache}\n  {test_cache}")
        nll_train = np.load(train_cache)
        nll_test = np.load(test_cache)
    else:
        device = get_device()
        print(f"Device: {device}")
        print("Loading Toto model...")
        toto = Toto.from_pretrained('Datadog/toto-open-base-1.0')
        toto.to(device)
        toto.eval()
        scorer = NLLScorer(toto.model, context_length=CONTEXT_LENGTH)

        print("Loading SMD data...")
        train_series = torch.load(TRAIN_PATH, weights_only=False)['series'].to(device)
        test_series = test_data['series'].to(device)
        print(f"  Train: {tuple(train_series.shape)}  Test: {tuple(test_series.shape)}")

        # Compute per-variate NLL once for train and test (reused across aggregations)
        print(f"\nComputing train NLL (stride={FIT_STRIDE})...", flush=True)
        nll_train = scorer.compute_nll_streaming(
            train_series, stride=FIT_STRIDE, progress_every=200).cpu().numpy()
        np.save(train_cache, nll_train)
        print(f"  Saved {train_cache}", flush=True)
        print(f"Computing test NLL (stride={DETECT_STRIDE})...", flush=True)
        nll_test = scorer.compute_nll_streaming(
            test_series, stride=DETECT_STRIDE, progress_every=200).cpu().numpy()
        np.save(test_cache, nll_test)
        print(f"  Saved {test_cache}", flush=True)

    # Align labels: output index i -> label position context_length + i*stride
    n_out = nll_test.shape[2]
    positions = CONTEXT_LENGTH + np.arange(n_out) * DETECT_STRIDE
    aligned_labels = test_labels[:, positions]  # (M, T_out)

    output_dir = Path('toto/results/auroc')
    output_dir.mkdir(parents=True, exist_ok=True)

    for agg in aggregations:
        train_scores = aggregate(nll_train, agg)
        test_scores = aggregate(nll_test, agg)

        rows = per_machine_metrics(train_scores, test_scores, aligned_labels)
        machine_aurocs = [r['auroc'] for r in rows if r['auroc'] is not None]
        macro_auroc = float(np.mean(machine_aurocs)) if machine_aurocs else None
        macro_f1 = float(np.mean([r['f1'] for r in rows]))

        # Pooled (micro-average) AUROC for comparison
        flat_scores = test_scores.flatten()
        flat_labels = aligned_labels.flatten().astype(bool)
        pooled_auroc = float(roc_auc_score(flat_labels, flat_scores))

        print(f"\n{'='*60}\nAggregation: {agg}")
        print(f"  Per-machine macro-avg AUROC: {macro_auroc:.4f} "
              f"(over {len(machine_aurocs)} machines)")
        print(f"  Pooled (global) AUROC:       {pooled_auroc:.4f}")
        print(f"  Macro-avg F1 (per-machine thresholds): {macro_f1:.4f}")

        result = {
            'dataset': 'smd',
            'aggregation': agg,
            'context_length': CONTEXT_LENGTH,
            'fit_stride': FIT_STRIDE,
            'detect_stride': DETECT_STRIDE,
            'threshold_percentile': THRESHOLD_PERCENTILE,
            'threshold_strategy': 'per_machine',
            'auroc_macro_avg': macro_auroc,
            'auroc_pooled': pooled_auroc,
            'f1_macro_avg': macro_f1,
            'n_machines': len(rows),
            'n_machines_with_anomalies': len(machine_aurocs),
            'per_machine': rows,
        }
        out = output_dir / f'smd_{agg}_per_machine_auroc.json'
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
