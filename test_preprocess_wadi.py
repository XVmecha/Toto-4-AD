#!/usr/bin/env python3
"""
Test WADI preprocessing on a small subset of data for faster iteration.
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pandas as pd
from pathlib import Path

# Import the preprocessor
from preprocess_wadi import WADIPreprocessor


def create_subset_files(n_rows=10000):
    """Create smaller CSV files for testing."""
    data_dir = Path("toto/data/WADI.A1_9 Oct 2017")
    subset_dir = Path("toto/data/wadi_subset")
    subset_dir.mkdir(exist_ok=True, parents=True)

    print("Creating WADI subset files...")

    # Load and subset training data (skip metadata rows, use row 4 as header)
    print(f"Loading training data (taking first {n_rows} rows)...")
    df_train = pd.read_csv(data_dir / "WADI_14days.csv", skiprows=3, nrows=n_rows)
    output_train = subset_dir / "WADI_14days_subset.csv"
    print(f"Saving to {output_train}...")
    df_train.to_csv(output_train, index=False)
    print(f"  Saved {len(df_train)} rows")

    # Load and subset test data
    print(f"\nLoading test data (taking first {n_rows} rows)...")
    df_test = pd.read_csv(data_dir / "WADI_attackdata.csv", skiprows=3, nrows=n_rows)
    output_test = subset_dir / "WADI_attackdata_subset.csv"
    print(f"Saving to {output_test}...")
    df_test.to_csv(output_test, index=False)
    print(f"  Saved {len(df_test)} rows")

    return subset_dir


def test_preprocessing():
    """Test preprocessing on subset."""

    # Create subset files (or use existing)
    subset_dir = Path("toto/data/wadi_subset")
    if not (subset_dir / "WADI_14days_subset.csv").exists():
        print("Creating subset files (this will take a moment)...\n")
        subset_dir = create_subset_files(n_rows=10000)
    else:
        print(f"Using existing subset files in {subset_dir}\n")

    # Test preprocessing
    print("=" * 70)
    print("Testing WADI Preprocessing on Subset")
    print("=" * 70)

    preprocessor = WADIPreprocessor(
        data_dir=str(subset_dir),
        downsample_factor=10,
        handle_missing='interpolate',
        normalize=True,
    )

    # Preprocess train
    try:
        train_data = preprocessor.preprocess_split(
            file_name='WADI_14days_subset.csv',
            split='train',
        )
    except Exception as e:
        print(f"\n✗ ERROR in training preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Preprocess test
    try:
        test_data = preprocessor.preprocess_split(
            file_name='WADI_attackdata_subset.csv',
            split='test',
        )
    except Exception as e:
        print(f"\n✗ ERROR in test preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save
    output_dir = "toto/data/preprocessed_wadi_subset"
    preprocessor.save_preprocessed(
        output_dir=output_dir,
        train_data=train_data,
        test_data=test_data,
    )

    # Summary
    train_series, train_labels, _ = train_data
    test_series, test_labels, _ = test_data

    print("\n" + "=" * 70)
    print("SUBSET TEST SUCCESSFUL!")
    print("=" * 70)
    print(f"\nTrain: {train_series.shape}")
    print(f"Test: {test_series.shape}")
    print(f"\nBoth have same number of variates: {train_series.shape[1] == test_series.shape[1]}")

    if train_series.shape[1] != test_series.shape[1]:
        print(f"✗ ERROR: Mismatch! Train has {train_series.shape[1]} variates, test has {test_series.shape[1]}")
    else:
        print(f"✓ SUCCESS: Both have {train_series.shape[1]} variates")

    print(f"\nTest labels: {(test_labels == 0).sum().item()} normal, {(test_labels == 1).sum().item()} attacks")


if __name__ == '__main__':
    test_preprocessing()
