#!/usr/bin/env python3
"""
SMD localization diagnostic: when an anomaly occurs, does TOTO's forecast error rise on
the *specific variates that are actually anomalous*, and are anomalies "noticed" at all?

Uses SMD's per-dimension interpretation labels (each event lists exactly which variates are
anomalous). Does NOT run the model -- reads the cached per-variate NLL.

Computes:
  1. Alignment check (IoU): reconstruct a machine-level mask by OR-ing the per-dimension
     events, compare to the stored machine label. ~1.0 confirms the per-dimension labels are
     correctly time-aligned to the score tensor (a prerequisite, not a result).
  2. Localization: standardize each variate's error by its own normal distribution, then for
     each event compare affected variates (v in the event's dimension list) vs unaffected.
     If affected >> unaffected, the error lands on the right dimensions.
  3. Sparsity: average fraction of the 38 variates actually affected during anomaly timesteps.
  4. Per-machine tail concentration: where the large error responses come from.
  5. Per-event detectability ("noticed?"): a spike-preserving peak = max over affected variates
     AND over in-window timesteps. Events below 1 sigma are effectively invisible to the error
     metric.

Standardization baseline: test-period normal timesteps (a DIAGNOSTIC choice -- it compares
anomaly vs normal within the same period, controlling for any calibration/evaluation shift).

Outputs:
  toto/results/aggregation_analysis/smd_localization.json
  toto/results/auroc/smd_event_detectability.json
"""
import json
from pathlib import Path
import numpy as np
import torch

CONTEXT_LENGTH = 512
DETECT_STRIDE = 8
SCORES_DIR = Path('toto/results/scores')
TEST_CACHE = SCORES_DIR / f'smd_nll_test_stride{DETECT_STRIDE}.npy'
TEST_PT = Path('toto/data/preprocessed_smd_1x/smd_test.pt')
OUT_LOC = Path('toto/results/aggregation_analysis/smd_localization.json')
OUT_EVT = Path('toto/results/auroc/smd_event_detectability.json')


def build_P(meta, M, V, T):
    """Per-(machine,variate,timestep) anomaly mask from interpretation labels."""
    P = np.zeros((M, V, T), dtype=bool)
    for m in range(M):
        for ev in meta[m]['interpretation']:
            s, e = ev['time_range'].split('-')
            s, e = max(int(s), 0), min(int(e) + 1, T)
            for d in ev['affected_dimensions']:
                if 1 <= d <= V:
                    P[m, d - 1, s:e] = True
    return P


def main():
    if not TEST_CACHE.exists():
        raise SystemExit(f"Missing {TEST_CACHE}. Run compute_per_machine_auroc.py first.")

    nll = np.load(TEST_CACHE)                 # (M, V, T_test)
    data = torch.load(TEST_PT, weights_only=False)
    labels = data['labels'].cpu().numpy()
    meta = data['metadata']
    M, V, Tn = nll.shape
    T = labels.shape[1]
    pos = CONTEXT_LENGTH + np.arange(Tn) * DETECT_STRIDE
    pos_of = {int(t): i for i, t in enumerate(pos)}

    P_full = build_P(meta, M, V, T)
    recon = P_full.any(axis=1)                # (M,T) reconstructed machine-level mask
    inter = int((recon & (labels == 1)).sum())
    union = int((recon | (labels == 1)).sum())
    iou = inter / union if union else 0.0

    Pp = P_full[:, :, pos]                     # (M,V,Tn) per-variate anomaly at scored positions
    labp = labels[:, pos].astype(bool)

    aff_z, unaff_z = [], []
    ev_aff, ev_unaff = [], []
    frac_affected = []
    peaks = []
    per_machine_tail = []
    events_used = events_skipped = 0

    for m in range(M):
        normal = labp[m] == False
        if normal.sum() < 50:
            continue
        nm = nll[m]
        mu = nm[:, normal].mean(axis=1)
        sd = nm[:, normal].std(axis=1) + 1e-6
        zfull = ((nm.T - mu) / sd).T           # (V,Tn) standardized error

        # sparsity (fraction of variates affected during this machine's anomaly timesteps)
        if labp[m].any():
            frac_affected.append(float(Pp[m][:, labp[m]].mean()))

        machine_tail_cells = machine_aff_cells = 0
        machine_z_sum = 0.0
        for ev in meta[m]['interpretation']:
            s, e = ev['time_range'].split('-')
            s, e = int(s), int(e)
            idx = [pos_of[t] for t in range(s, e + 1) if t in pos_of]
            if not idx:
                events_skipped += 1
                continue
            events_used += 1
            idx = np.array(idx)
            D = set(d - 1 for d in ev['affected_dimensions'] if 1 <= d <= V)
            zwin = (nm[:, idx].mean(axis=1) - mu) / sd     # mean-over-window z per variate
            a = [zwin[v] for v in range(V) if v in D]
            u = [zwin[v] for v in range(V) if v not in D]
            aff_z += a
            unaff_z += u
            machine_aff_cells += len(a)
            machine_tail_cells += sum(1 for x in a if x > 2)
            machine_z_sum += sum(a)
            if a and u:
                ev_aff.append(float(np.mean(a)))
                ev_unaff.append(float(np.mean(u)))
            # spike-preserving peak over affected variates AND in-window timesteps
            if D:
                peaks.append(float(zfull[np.ix_(sorted(D), idx)].max()))
        if machine_aff_cells:
            per_machine_tail.append({
                'machine_id': m, 'machine_name': meta[m]['file_name'],
                'n_affected_cells': machine_aff_cells,
                'n_tail_z_gt2': machine_tail_cells,
                'mean_affected_z': machine_z_sum / machine_aff_cells,
            })

    aff_z, unaff_z = np.array(aff_z), np.array(unaff_z)
    ev_aff, ev_unaff = np.array(ev_aff), np.array(ev_unaff)
    peaks = np.array(peaks)

    tail_sorted = sorted(per_machine_tail, key=lambda r: -r['n_tail_z_gt2'])
    total_tail = sum(r['n_tail_z_gt2'] for r in per_machine_tail)
    cum3 = sum(r['n_tail_z_gt2'] for r in tail_sorted[:3])
    cum6 = sum(r['n_tail_z_gt2'] for r in tail_sorted[:6])

    loc = {
        'dataset': 'smd',
        'detect_stride': DETECT_STRIDE,
        'standardization_baseline': 'test_period_normal_per_variate',
        'alignment_iou': iou,
        'events_used': events_used,
        'events_skipped_no_scored_point': events_skipped,
        'anomaly_sparsity_frac_variates_affected': {
            'mean': float(np.mean(frac_affected)),
            'median': float(np.median(frac_affected)),
        },
        'standardized_error_sigma': {
            'affected': {'mean': float(aff_z.mean()), 'median': float(np.median(aff_z)),
                         'frac_gt1': float((aff_z > 1).mean()), 'frac_gt2': float((aff_z > 2).mean()),
                         'n': int(aff_z.size)},
            'unaffected': {'mean': float(unaff_z.mean()), 'median': float(np.median(unaff_z)),
                           'frac_gt1': float((unaff_z > 1).mean()), 'frac_gt2': float((unaff_z > 2).mean()),
                           'n': int(unaff_z.size)},
        },
        'paired_per_event': {
            'frac_events_affected_gt_unaffected': float((ev_aff > ev_unaff).mean()),
            'mean_gap_sigma': float((ev_aff - ev_unaff).mean()),
            'n_events': int(ev_aff.size),
        },
        'tail_concentration': {
            'total_affected_cells_z_gt2': total_tail,
            'top3_machines_share': cum3 / total_tail if total_tail else None,
            'top6_machines_share': cum6 / total_tail if total_tail else None,
            'per_machine': tail_sorted,
        },
    }
    OUT_LOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOC.write_text(json.dumps(loc, indent=2))

    evt = {
        'dataset': 'smd',
        'detect_stride': DETECT_STRIDE,
        'measure': 'peak standardized error = max over affected variates AND in-window timesteps',
        'standardization_baseline': 'test_period_normal_per_variate',
        'events_used': int(peaks.size),
        'events_skipped_no_scored_point': events_skipped,
        'frac_seen_at_threshold': {f'peak_gt_{thr}': float((peaks > thr).mean()) for thr in (1, 2, 3, 5)},
        'frac_unnoticed_peak_lt_1': float((peaks < 1).mean()),
        'peak_distribution': {'median': float(np.median(peaks)),
                              'pct75': float(np.percentile(peaks, 75)),
                              'pct90': float(np.percentile(peaks, 90))},
    }
    OUT_EVT.write_text(json.dumps(evt, indent=2))

    print(f"Saved {OUT_LOC}\nSaved {OUT_EVT}")
    print(f"alignment IoU={iou:.3f} (sanity gate)")
    print(f"localization: affected mean z={aff_z.mean():+.3f} vs unaffected {unaff_z.mean():+.3f}; "
          f"paired affected>unaffected in {(ev_aff>ev_unaff).mean()*100:.0f}% of events")
    print(f"tail: top3 machines hold {cum3}/{total_tail} ({cum3/total_tail*100:.0f}%) of z>2 cells")
    print(f"noticed: peak>2 in {(peaks>2).mean()*100:.0f}% of events; "
          f"UNNOTICED (peak<1) {(peaks<1).mean()*100:.0f}%")


if __name__ == '__main__':
    main()
