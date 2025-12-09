#!/usr/bin/env python3
"""
Quick start example for Toto with M4 GPU support.
This demonstrates basic forecasting with automatic device detection.
"""

# IMPORTANT: Set MPS fallback BEFORE importing torch
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from toto.model.util import get_device

def main():
    print("=" * 60)
    print("Toto Quick Start Example")
    print("=" * 60)

    # Detect and set device
    device = get_device()
    print(f"\n✓ Using device: {device}")

    # Load the pre-trained model
    print("\nLoading pre-trained model...")
    print("(This may take a moment on first run as the model downloads)")
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to(device)
    toto.eval()
    print("✓ Model loaded successfully")

    # Create forecaster
    forecaster = TotoForecaster(toto.model)

    # Prepare sample input time series
    print("\nPreparing sample data...")
    num_variates = 7
    context_length = 1024  # Using shorter context for quick demo
    prediction_length = 96

    input_series = torch.randn(num_variates, context_length).to(device)
    timestamp_seconds = torch.zeros(num_variates, context_length).to(device)
    time_interval_seconds = torch.full((num_variates,), 60*15).to(device)  # 15-minute intervals

    # Create a MaskedTimeseries object
    inputs = MaskedTimeseries(
        series=input_series,
        padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
        id_mask=torch.zeros_like(input_series),
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )

    print(f"✓ Input prepared: {num_variates} variates, {context_length} timesteps")

    # Generate forecasts
    print(f"\nGenerating {prediction_length}-step forecast...")
    print("(Using 32 samples for quick demo - use 256 for production)")

    with torch.no_grad():
        forecast = forecaster.forecast(
            inputs,
            prediction_length=prediction_length,
            num_samples=32,  # Reduced for quick demo
            samples_per_batch=32,
        )

    print("✓ Forecast generated successfully!")

    # Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Forecast shape: {forecast.samples.shape}")
    print(f"  - Variates: {forecast.samples.shape[1]}")
    print(f"  - Time steps: {forecast.samples.shape[2]}")
    print(f"  - Samples: {forecast.samples.shape[3]}")

    print(f"\nMedian forecast (point estimate) shape: {forecast.median.shape}")
    print(f"Mean of samples: {forecast.mean.shape}")
    print(f"Standard deviation: {forecast.std.shape}")

    # Show some statistics
    print("\nSample statistics for first variate:")
    print(f"  Median forecast mean: {forecast.median[0, 0, :].mean().item():.4f}")
    print(f"  Median forecast std:  {forecast.median[0, 0, :].std().item():.4f}")
    print(f"  10th percentile: {forecast.quantile(0.1)[0, 0, :].mean().item():.4f}")
    print(f"  90th percentile: {forecast.quantile(0.9)[0, 0, :].mean().item():.4f}")

    print("\n" + "=" * 60)
    print("✓ Quick start example completed successfully!")
    print("=" * 60)
    print(f"\nYour M4 GPU ({device}) is working correctly with Toto!")
    print("\nNext steps:")
    print("  1. Check out toto/notebooks/inference_tutorial.ipynb for more details")
    print("  2. See M4_GPU_SETUP.md for full documentation")
    print("  3. Try running on your own time series data")

if __name__ == "__main__":
    main()
