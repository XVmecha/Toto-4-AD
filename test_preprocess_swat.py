#!/usr/bin/env python3
"""
Test SWaT preprocessing on a small subset of data for faster iteration.
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import pandas as pd
from pathlib import Path

# Import the preprocessor
from preprocess_swat import SWaTPreprocessor

def create_subset_files(n_rows=10000):
    """Create smaller Excel files for testing."""
    data_dir = Path("toto/data/SWaT.A1 & A2_Dec 2015/Physical")
    subset_dir = Path("toto/data/swat_subset")
    subset_dir.mkdir(exist_ok=True, parents=True)

    print("Creating subset files...")

    # Load and subset training data
    print(f"Loading training data (taking first {n_rows} rows)...")
    # Load with original headers (row 0 = stage info, row 1 = column names)
    df_train_full = pd.read_excel(data_dir / "SWaT_Dataset_Normal_v1.xlsx", header=None, nrows=n_rows+2)
    output_train = subset_dir / "SWaT_train_subset.xlsx"
    print(f"Saving to {output_train}...")
    df_train_full.to_excel(output_train, index=False, header=False)
    print(f"  Saved {len(df_train_full)-2} data rows (plus 2 header rows)")

    # Load and subset test data
    print(f"\nLoading test data (taking first {n_rows} rows)...")
    df_test_full = pd.read_excel(data_dir / "SWaT_Dataset_Attack_v0.xlsx", header=None, nrows=n_rows+2)
    output_test = subset_dir / "SWaT_test_subset.xlsx"
    print(f"Saving to {output_test}...")
    df_test_full.to_excel(output_test, index=False, header=False)
    print(f"  Saved {len(df_test_full)-2} data rows (plus 2 header rows)")

    # Parse to check attack count
    df_test = pd.read_excel(output_test, header=1)

    # Check if test has attacks
    attack_count = (df_test['Normal/Attack'] != 'Normal').sum()
    print(f"  Test subset has {attack_count} attack samples ({attack_count/len(df_test)*100:.1f}%)")

    return subset_dir

def test_preprocessing():
    """Test preprocessing on subset."""

    # Create subset files (or use existing)
    subset_dir = Path("toto/data/swat_subset")
    if not (subset_dir / "SWaT_train_subset.xlsx").exists():
        print("Creating subset files (this will take a moment)...\n")
        subset_dir = create_subset_files(n_rows=10000)
    else:
        print(f"Using existing subset files in {subset_dir}\n")

    # Test preprocessing
    print("=" * 70)
    print("Testing SWaT Preprocessing on Subset")
    print("=" * 70)

    preprocessor = SWaTPreprocessor(
        data_dir=str(subset_dir),
        downsample_factor=10,
        handle_missing='interpolate',
        normalize=True,
    )

    # Preprocess train
    train_data = preprocessor.preprocess_split(
        file_name='SWaT_train_subset.xlsx',
        split='train',
    )

    # Preprocess test
    test_data = preprocessor.preprocess_split(
        file_name='SWaT_test_subset.xlsx',
        split='test',
    )

    # Save
    output_dir = "toto/data/preprocessed_swat_subset"
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
        print(f"ERROR: Mismatch! Train has {train_series.shape[1]} variates, test has {test_series.shape[1]}")
    else:
        print(f"✓ SUCCESS: Both have {train_series.shape[1]} variates")

if __name__ == '__main__':
    test_preprocessing()
