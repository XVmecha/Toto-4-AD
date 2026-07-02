#!/usr/bin/env python3
"""
SWaT: does the robust per-variate normalization that rescued SMD also change SWaT's AUROC?

SWaT is a single series (51 variates), so there is no per-machine macro-average: we compute
one AUROC over the whole evaluation series for each aggregation rule. Per-variate normalization
statistics (median/IQR for robust; mean/std for the plain z-score) come from the calibration
(normal) series only, so everything stays zero-shot.

Caches per-variate NLL to toto/results/scores/ so re-analysis needs no model.
Output: toto/results/auroc/swat_aggregation_comparison.json
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CONTEXT_LENGTH = 512
FIT_STRIDE = 32
DETECT_STRIDE = 8
SCORES_DIR = Path('toto/results/scores')
TRAIN_CACHE = SCORES_DIR / f'swat_nll_train_stride{FIT_STRIDE}.npy'
TEST_CACHE = SCORES_DIR / f'swat_nll_test_stride{DETECT_STRIDE}.npy'
TRAIN_PT = Path('toto/data/preprocessed_datasets/swat/swat_train.pt')
TEST_PT = Path('toto/data/preprocessed_datasets/swat/swat_test.pt')
OUT = Path('toto/results/auroc/swat_aggregation_comparison.json')


def main():
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    test_data = torch.load(TEST_PT, weights_only=False)
    test_labels = test_data['labels'].cpu().numpy()

    if TRAIN_CACHE.exists() and TEST_CACHE.exists():
        print(f"Loading cached SWaT NLL (skipping model):\n  {TRAIN_CACHE}\n  {TEST_CACHE}")
        nll_tr = np.load(TRAIN_CACHE)
        nll = np.load(TEST_CACHE)
    else:
        from toto.model.toto import Toto
        from toto.model.util import get_device
        from toto.anomaly_detection.scoring import NLLScorer
        device = get_device()
        print(f"Device: {device}\nLoading Toto model...")
        toto = Toto.from_pretrained('Datadog/toto-open-base-1.0')
        toto.to(device); toto.eval()
        scorer = NLLScorer(toto.model, context_length=CONTEXT_LENGTH)
        train_series = torch.load(TRAIN_PT, weights_only=False)['series'].to(device)
        test_series = test_data['series'].to(device)
        print(f"  Train {tuple(train_series.shape)}  Test {tuple(test_series.shape)}", flush=True)
        print(f"Computing train NLL (stride={FIT_STRIDE})...", flush=True)
        nll_tr = scorer.compute_nll_streaming(train_series, stride=FIT_STRIDE, progress_every=200).cpu().numpy()
        np.save(TRAIN_CACHE, nll_tr); print(f"  saved {TRAIN_CACHE}", flush=True)
        print(f"Computing test NLL (stride={DETECT_STRIDE})...", flush=True)
        nll = scorer.compute_nll_streaming(test_series, stride=DETECT_STRIDE, progress_every=200).cpu().numpy()
        np.save(TEST_CACHE, nll); print(f"  saved {TEST_CACHE}", flush=True)

    # single series: index 0
    e_tr, e = nll_tr[0], nll[0]                       # (V, T*)
    n_out = e.shape[1]
    pos = CONTEXT_LENGTH + np.arange(n_out) * DETECT_STRIDE
    y = test_labels[0, pos].astype(bool)

    # per-variate calibration-normal statistics
    mean = e_tr.mean(1, keepdims=True); std = e_tr.std(1, keepdims=True) + 1e-6
    med = np.median(e_tr, 1, keepdims=True)
    iqr = (np.percentile(e_tr, 75, 1, keepdims=True) - np.percentile(e_tr, 25, 1, keepdims=True)) + 1e-6

    z_ms = (e - mean) / std
    z_rob = (e - med) / iqr

    def auroc(score):
        return float(roc_auc_score(y, score))

    result = {
        'dataset': 'swat',
        'context_length': CONTEXT_LENGTH, 'fit_stride': FIT_STRIDE, 'detect_stride': DETECT_STRIDE,
        'n_samples': int(y.size), 'n_anomaly': int(y.sum()),
        'auroc': {
            'raw_mean': auroc(e.mean(0)), 'raw_max': auroc(e.max(0)),
            'zscore_meanstd_mean': auroc(z_ms.mean(0)), 'zscore_meanstd_max': auroc(z_ms.max(0)),
            'robust_median_iqr_mean': auroc(z_rob.mean(0)), 'robust_median_iqr_max': auroc(z_rob.max(0)),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    a = result['auroc']
    print(f"\nSaved {OUT}")
    print(f"SWaT AUROC (single series, detect stride {DETECT_STRIDE}):")
    print(f"  raw            mean={a['raw_mean']:.3f}  max={a['raw_max']:.3f}")
    print(f"  z mean/std     mean={a['zscore_meanstd_mean']:.3f}  max={a['zscore_meanstd_max']:.3f}")
    print(f"  robust med/IQR mean={a['robust_median_iqr_mean']:.3f}  max={a['robust_median_iqr_max']:.3f}")


if __name__ == '__main__':
    main()
