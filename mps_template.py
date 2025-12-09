#!/usr/bin/env python3
"""
Template for using Toto with M4 GPU (MPS backend).
Copy this file and modify for your use case.
"""

# ============================================================
# STEP 1: Enable MPS fallback BEFORE importing torch
# This must be the first thing in your script!
# ============================================================
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# ============================================================
# STEP 2: Now import torch and other libraries
# ============================================================
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from toto.model.util import get_device


def main():
    # Detect and use M4 GPU automatically
    device = get_device()
    print(f"Using device: {device}")

    # Load pre-trained model
    print("Loading model...")
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to(device)
    toto.eval()

    # Create forecaster
    forecaster = TotoForecaster(toto.model)

    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    # Example: Prepare your time series data
    num_variates = 7
    context_length = 2048
    prediction_length = 336

    # Load your actual data here
    input_series = torch.randn(num_variates, context_length).to(device)
    timestamp_seconds = torch.zeros(num_variates, context_length).to(device)
    time_interval_seconds = torch.full((num_variates,), 60).to(device)  # 1-minute intervals

    # Create input object
    inputs = MaskedTimeseries(
        series=input_series,
        padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
        id_mask=torch.zeros_like(input_series),
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )

    # Generate forecast
    print(f"Generating {prediction_length}-step forecast...")
    with torch.no_grad():
        forecast = forecaster.forecast(
            inputs,
            prediction_length=prediction_length,
            num_samples=256,
            samples_per_batch=256,
        )

    # Access results
    print(f"\nForecast generated!")
    print(f"Samples shape: {forecast.samples.shape}")
    print(f"Median forecast shape: {forecast.median.shape}")
    print(f"Prediction std shape: {forecast.std.shape}")

    # Use the forecasts for your application
    # ...

    print("\nDone!")


if __name__ == "__main__":
    main()
