#!/usr/bin/env python3
"""How concentrated are TOTO's per-variate errors across variates?

Reads the cached calibration (normal) NLL and reports, per dataset:
  - the fraction of timesteps at which a single variate gives the largest error
    (how often raw max-aggregation is decided by one dominant variate), and
  - how far the highest per-variate mean error sits above the cross-variate
    median, in cross-variate standard deviations.

No model required.
"""
import numpy as np

def report(path, name):
    nll = np.load(path)                 # (B, V, T)
    B, V, T = nll.shape
    conc, stick = [], []
    for b in range(B):
        e = nll[b]                      # (V, T)
        counts = np.bincount(e.argmax(0), minlength=V)
        conc.append(counts.max() / T)
        pv = e.mean(1)                  # per-variate mean error
        stick.append((pv.max() - np.median(pv)) / (pv.std() + 1e-9))
    print(f"{name} (V={V}): single variate gives the max on "
          f"{np.mean(conc)*100:.0f}% of timesteps; top variate is "
          f"{np.mean(stick):.1f} cross-variate std devs above the median")

report('toto/results/scores/smd_nll_train_stride32.npy', 'SMD')
report('toto/results/scores/swat_nll_train_stride32.npy', 'SWaT')
