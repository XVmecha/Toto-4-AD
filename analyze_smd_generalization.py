#!/usr/bin/env python3
"""
SMD per-machine detectability: can TOTO's forecast error detect anomalies, and does
the answer depend on how per-variate errors are aggregated?

This script does NOT run the model. It reads the cached per-variate NLL arrays produced
by compute_per_machine_auroc.py and, for each of the 28 machines, computes the
threshold-independent ranking quality (AUROC) of several aggregation rules:

  - raw mean / raw max     : current method (combine raw per-variate NLL directly)
  - z-scored mean / max    : standardize each variate by its OWN calibration-normal NLL
                             distribution, then aggregate (zero-shot legal: uses only
                             calibration/normal statistics, never the labels)
  - best-single-variate    : per machine, the single variate whose NLL best ranks that
                             machine's anomalies. UPPER BOUND ONLY -- it is selected using
                             the labels, so it is not achievable in a zero-shot setting.

Output: toto/results/auroc/smd_per_machine_detectability.json

Standardization baseline: calibration (train/normal) per-variate mean & std -- the
deployment-realistic reference for a real detector.
"""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CONTEXT_LENGTH = 512
FIT_STRIDE = 32
DETECT_STRIDE = 8
SCORES_DIR = Path('toto/results/scores')
TRAIN_CACHE = SCORES_DIR / f'smd_nll_train_stride{FIT_STRIDE}.npy'
TEST_CACHE = SCORES_DIR / f'smd_nll_test_stride{DETECT_STRIDE}.npy'
TEST_PT = Path('toto/data/preprocessed_smd_1x/smd_test.pt')
OUT = Path('toto/results/auroc/smd_per_machine_detectability.json')


def build_pervariate_label(meta_m, V, T, scored_pos, pos_of):
    """Per-variate anomaly mask at scored positions, from interpretation labels."""
    P = np.zeros((V, len(scored_pos)), dtype=bool)
    for ev in meta_m['interpretation']:
        s, e = ev['time_range'].split('-')
        s, e = int(s), int(e)
        idx = [pos_of[t] for t in range(s, e + 1) if t in pos_of]
        if not idx:
            continue
        for d in ev['affected_dimensions']:
            if 1 <= d <= V:
                P[d - 1, idx] = True
    return P


def main():
    if not (TRAIN_CACHE.exists() and TEST_CACHE.exists()):
        raise SystemExit(f"Missing cached NLL. Run compute_per_machine_auroc.py first "
                         f"(expected {TRAIN_CACHE} and {TEST_CACHE}).")

    nll_tr = np.load(TRAIN_CACHE)            # (M, V, T_train)
    nll = np.load(TEST_CACHE)                # (M, V, T_test)
    data = torch.load(TEST_PT, weights_only=False)
    labels = data['labels'].cpu().numpy()
    meta = data['metadata']
    M, V, Tn = nll.shape
    pos = CONTEXT_LENGTH + np.arange(Tn) * DETECT_STRIDE
    pos_of = {int(t): i for i, t in enumerate(pos)}

    # Robust per-variate normalization (median / IQR), GDN-style (Deng & Hooi, AAAI 2021).
    # Statistics from calibration-normal data only, so the detector stays zero-shot.
    med = np.median(nll_tr, axis=2, keepdims=True)
    q75 = np.percentile(nll_tr, 75, axis=2, keepdims=True)
    q25 = np.percentile(nll_tr, 25, axis=2, keepdims=True)
    iqr = (q75 - q25) + 1e-6
    z = (nll - med) / iqr

    rows = []
    for m in range(M):
        y = labels[m, pos].astype(bool)
        if not y.any() or y.all():
            continue
        nm = nll[m]
        zm = z[m]
        P = build_pervariate_label(meta[m], V, max(pos) + 1, pos, pos_of)
        oracle = max((roc_auc_score(P[v], nm[v]) for v in range(V)
                      if P[v].any() and not P[v].all()), default=None)
        rows.append({
            'machine_id': m,
            'machine_name': meta[m]['file_name'],
            'n_anomaly': int(y.sum()),
            'auroc_raw_mean': float(roc_auc_score(y, nm.mean(0))),
            'auroc_raw_max': float(roc_auc_score(y, nm.max(0))),
            'auroc_robust_mean': float(roc_auc_score(y, zm.mean(0))),
            'auroc_robust_max': float(roc_auc_score(y, zm.max(0))),
            'auroc_best_variate_oracle': float(oracle) if oracle is not None else None,
        })

    def macro(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals))

    zmax = np.array([r['auroc_robust_max'] for r in rows])
    result = {
        'dataset': 'smd',
        'context_length': CONTEXT_LENGTH,
        'fit_stride': FIT_STRIDE,
        'detect_stride': DETECT_STRIDE,
        'normalization': 'robust per-variate (median/IQR) on calibration-normal data, GDN-style',
        'label_alignment': 'target timestep = context_length + i*detect_stride',
        'macro': {k: macro(k) for k in
                  ['auroc_raw_mean', 'auroc_raw_max', 'auroc_robust_mean',
                   'auroc_robust_max', 'auroc_best_variate_oracle']},
        'robust_max_machine_counts': {
            'gt_0.7': int((zmax > 0.7).sum()),
            'gt_0.6': int((zmax > 0.6).sum()),
            'chance_0.45_0.55': int(((zmax >= 0.45) & (zmax <= 0.55)).sum()),
            'lt_0.45': int((zmax < 0.45).sum()),
            'n_machines': len(zmax),
        },
        'per_machine': sorted(rows, key=lambda r: -r['auroc_robust_max']),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    m = result['macro']
    print(f"Saved {OUT}")
    print(f"Macro AUROC:  raw mean={m['auroc_raw_mean']:.3f} raw max={m['auroc_raw_max']:.3f} "
          f"| robust mean={m['auroc_robust_mean']:.3f} robust max={m['auroc_robust_max']:.3f} "
          f"| oracle best-dim={m['auroc_best_variate_oracle']:.3f}")
    c = result['robust_max_machine_counts']
    print(f"robust-max per machine: >0.7={c['gt_0.7']}  >0.6={c['gt_0.6']}  "
          f"~chance={c['chance_0.45_0.55']}  <0.45={c['lt_0.45']}  (of {c['n_machines']})")


if __name__ == '__main__':
    main()
