"""Spot-check: does the cached streaming NLL match a fresh single-window forward pass
at the timesteps we assume (t = 512 + i*detect_stride)?"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np, torch
from toto.model.toto import Toto
from toto.model.util import get_device
from toto.anomaly_detection.scoring import NLLScorer
from toto.data.util.dataset import MaskedTimeseries

CTX, DET = 512, 8
cache = np.load('toto/results/scores/smd_nll_test_stride8.npy')   # (28,38,2896)
series = torch.load('toto/data/preprocessed_smd_1x/smd_test.pt', weights_only=False)['series']
device = get_device()
toto = Toto.from_pretrained('Datadog/toto-open-base-1.0'); toto.to(device); toto.eval()
scorer = NLLScorer(toto.model, context_length=CTX)
series = series.to(device)
B, V, T = series.shape

for i in [0, 100, 1000, 2895]:
    t = CTX + i * DET
    ctx = series[:, :, t - CTX:t]
    inputs = MaskedTimeseries(
        series=ctx,
        padding_mask=torch.ones_like(ctx, dtype=torch.bool),
        id_mask=torch.zeros_like(ctx),
        timestamp_seconds=torch.zeros_like(ctx),
        time_interval_seconds=torch.ones(V, device=device).unsqueeze(0).expand(B, -1),
    )
    fresh = scorer.compute_nll(inputs, series[:, :, t]).detach().cpu().numpy()  # (28,38)
    cached = cache[:, :, i]
    diff = np.abs(fresh - cached)
    print(f"i={i:>4} t={t:>5}  max|fresh-cached|={diff.max():.3e}  "
          f"mean|.|={diff.mean():.3e}  cached[0,:3]={cached[0,:3].round(3)}")
