#!/usr/bin/env python3
"""
SWaT Dataset Preprocessing for Toto Anomaly Detection

This script preprocesses the SWaT (Secure Water Treatment) dataset for use with
the Toto time series foundation model for anomaly detection.

Dataset Information:
- Training: 495,000 timesteps (7 days, all normal operation)
- Test: 449,919 timesteps (4.7 days, with 41 attacks)
- Features: 51 sensor/actuator measurements
- Sampling rate: 1 second

The preprocessing pipeline:
1. Load Excel files (train and test)
2. Parse timestamps
3. Clean sensor data (handle missing values, outliers)
4. Normalize/standardize features
5. Create binary attack labels
6. Convert to PyTorch tensors
7. Save as .pt files for efficient loading

Usage:
    python preprocess_swat.py [--output_dir ./preprocessed_data] [--downsample 1]
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datetime import datetime

# CRITICAL: Enable MPS fallback before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class SWaTPreprocessor:
    """Preprocessor for SWaT dataset."""

    def __init__(
        self,
        data_dir: str = "toto/data/SWaT.A1 & A2_Dec 2015/Physical",
        downsample_factor: int = 1,
        handle_missing: str = 'interpolate',
        normalize: bool = True,
    ):
        """
        Initialize SWaT preprocessor.

        Args:
            data_dir: Directory containing SWaT Excel files
            downsample_factor: Downsample by this factor (1 = no downsampling)
            handle_missing: How to handle missing values ('interpolate', 'forward_fill', 'drop')
            normalize: Whether to normalize features (per-feature z-score)
        """
        self.data_dir = Path(data_dir)
        self.downsample_factor = downsample_factor
        self.handle_missing = handle_missing
        self.normalize = normalize

        # Stats computed on training data (for normalization)
        self.feature_mean = None
        self.feature_std = None
        self.feature_names = None

    def load_raw_data(self, file_name: str) -> pd.DataFrame:
        """
        Load raw SWaT Excel file.

        Args:
            file_name: Name of Excel file (e.g., 'SWaT_Dataset_Normal_v1.xlsx')

        Returns:
            DataFrame with timestamp and all sensor columns
        """
        file_path = self.data_dir / file_name
        print(f"Loading {file_path}...")

        # Load with row 1 as header (row 0 contains process stage info)
        df = pd.read_excel(file_path, header=1)

        print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamp column and set as index.

        Args:
            df: Raw dataframe with ' Timestamp' column

        Returns:
            DataFrame with parsed datetime index
        """
        print("Parsing timestamps...")

        # The timestamp column has a leading space
        timestamp_col = ' Timestamp'

        # Parse timestamps (format='mixed' handles variations in format)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed', dayfirst=True)

        # Set as index
        df = df.set_index(timestamp_col)
        df.index.name = 'timestamp'

        print(f"  Time range: {df.index[0]} to {df.index[-1]}")
        print(f"  Duration: {df.index[-1] - df.index[0]}")

        return df

    def extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract sensor features and attack labels.

        Args:
            df: DataFrame with all columns

        Returns:
            (features, labels) where features has 51 sensor columns,
            labels is binary (0=Normal, 1=Attack)
        """
        print("Extracting features and labels...")

        # Extract labels (last column)
        label_col = 'Normal/Attack'
        labels = df[label_col].copy()

        # Convert to binary: 'Normal' -> 0, anything else ('Attack', 'A ttack') -> 1
        labels_binary = (labels != 'Normal').astype(int)

        # Extract all sensor/actuator features (exclude label column)
        features = df.drop(columns=[label_col])

        # If feature names already set (from training), ensure same columns
        if self.feature_names is not None:
            # Ensure test has same columns as train
            missing_cols = set(self.feature_names) - set(features.columns)
            extra_cols = set(features.columns) - set(self.feature_names)

            if missing_cols:
                print(f"  WARNING: Test missing columns: {missing_cols}")
                # Add missing columns with zeros
                for col in missing_cols:
                    features[col] = 0.0

            if extra_cols:
                print(f"  WARNING: Test has extra columns (dropping): {extra_cols}")
                features = features.drop(columns=list(extra_cols))

            # Reorder to match training
            features = features[self.feature_names]
        else:
            # First time (training): save feature names
            self.feature_names = features.columns.tolist()

        print(f"  Features: {len(features.columns)} columns")
        print(f"  Label distribution:")
        print(f"    Normal: {(labels_binary == 0).sum():,} ({(labels_binary == 0).sum() / len(labels_binary) * 100:.1f}%)")
        print(f"    Attack: {(labels_binary == 1).sum():,} ({(labels_binary == 1).sum() / len(labels_binary) * 100:.1f}%)")

        return features, labels_binary

    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sensor features: handle missing values, outliers, etc.

        Args:
            features: Raw feature DataFrame

        Returns:
            Cleaned feature DataFrame
        """
        print("Cleaning features...")
        features = features.copy()

        # Check for missing values
        missing_count = features.isnull().sum().sum()
        if missing_count > 0:
            print(f"  Found {missing_count:,} missing values")

            if self.handle_missing == 'interpolate':
                print("  Interpolating missing values (linear)...")
                features = features.interpolate(method='linear', axis=0)
                # Fill any remaining NaNs at edges with forward/backward fill
                features = features.fillna(method='ffill').fillna(method='bfill')
            elif self.handle_missing == 'forward_fill':
                print("  Forward filling missing values...")
                features = features.fillna(method='ffill').fillna(method='bfill')
            elif self.handle_missing == 'drop':
                print("  WARNING: Dropping rows with missing values...")
                features = features.dropna()
        else:
            print("  No missing values found")

        # Report data quality
        print(f"  Final shape: {features.shape}")
        print(f"  Remaining NaNs: {features.isnull().sum().sum()}")

        return features

    def normalize_features(
        self,
        features: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.

        Args:
            features: Feature DataFrame
            fit: If True, compute mean/std from this data (training).
                 If False, use pre-computed stats (test).

        Returns:
            Normalized features
        """
        if not self.normalize:
            return features

        print("Normalizing features...")

        if fit:
            # Compute statistics on training data
            self.feature_mean = features.mean()
            self.feature_std = features.std()

            # Avoid division by zero (for constant features)
            self.feature_std = self.feature_std.replace(0, 1)

            print(f"  Computed mean/std statistics from training data")
        else:
            if self.feature_mean is None or self.feature_std is None:
                raise ValueError("Must fit normalization on training data first!")

        # Apply z-score normalization
        features_normalized = (features - self.feature_mean) / self.feature_std

        print(f"  Normalized to mean=0, std=1")
        return features_normalized

    def downsample(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Downsample time series using median aggregation for features and
        majority vote for labels.

        For continuous features: takes median of each window
        For binary features (0/1): takes mode (most frequent value)
        For labels: takes majority vote (if any attack in window, label as attack)

        Args:
            features: Feature DataFrame
            labels: Label Series

        Returns:
            (downsampled_features, downsampled_labels)
        """
        if self.downsample_factor == 1:
            return features, labels

        print(f"Downsampling by factor {self.downsample_factor} using median aggregation...")

        n = len(features)
        k = self.downsample_factor

        # Number of complete windows
        n_windows = n // k

        # Truncate to complete windows
        features_truncated = features.iloc[:n_windows * k]
        labels_truncated = labels.iloc[:n_windows * k]

        # Identify binary/boolean features (only values are 0 and 1)
        binary_features = []
        continuous_features = []

        for col in features_truncated.columns:
            unique_vals = features_truncated[col].unique()
            # Check if only contains 0, 1 (or subset)
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                binary_features.append(col)
            else:
                continuous_features.append(col)

        print(f"  Detected {len(binary_features)} binary features, {len(continuous_features)} continuous features")

        # Reshape into windows: (n_windows, k, n_features)
        features_reshaped = features_truncated.values.reshape(n_windows, k, -1)
        labels_reshaped = labels_truncated.values.reshape(n_windows, k)

        # For features: compute median along window dimension (axis=1)
        features_downsampled = np.median(features_reshaped, axis=1)  # (n_windows, n_features)

        # For binary features: use mode (most frequent) instead of median
        if binary_features:
            binary_indices = [features_truncated.columns.get_loc(col) for col in binary_features]
            for idx in binary_indices:
                # Mode along axis=1 for each window
                feature_windows = features_reshaped[:, :, idx]  # (n_windows, k)
                # Mode: use round after mean as approximation (>=0.5 -> 1, <0.5 -> 0)
                features_downsampled[:, idx] = (feature_windows.mean(axis=1) >= 0.5).astype(float)

        # For labels: majority vote (if ANY attack in window, label as attack)
        # This is conservative: we don't want to miss attacks
        labels_downsampled = (labels_reshaped.max(axis=1))  # (n_windows,)

        # Convert back to DataFrame/Series with appropriate timestamps
        # Use the first timestamp of each window
        timestamps_downsampled = features_truncated.index[::k][:n_windows]

        features_df = pd.DataFrame(
            features_downsampled,
            index=timestamps_downsampled,
            columns=features_truncated.columns,
        )

        labels_series = pd.Series(
            labels_downsampled,
            index=timestamps_downsampled,
        )

        print(f"  Original length: {len(features):,}")
        print(f"  Downsampled length: {len(features_df):,}")
        print(f"  Compression ratio: {len(features) / len(features_df):.1f}x")

        # Check label preservation
        original_attack_ratio = labels.sum() / len(labels)
        downsampled_attack_ratio = labels_series.sum() / len(labels_series)
        print(f"  Original attack ratio: {original_attack_ratio * 100:.2f}%")
        print(f"  Downsampled attack ratio: {downsampled_attack_ratio * 100:.2f}%")

        return features_df, labels_series

    def to_tensors(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to PyTorch tensors in Toto format.

        Toto expects:
            - series: (batch, variates, timesteps)
            - labels: (batch, timesteps)

        Args:
            features: Feature DataFrame (T, V)
            labels: Label Series (T,)

        Returns:
            (series_tensor, labels_tensor) with shapes (1, V, T) and (1, T)
        """
        print("Converting to PyTorch tensors...")

        # Convert to numpy
        features_np = features.values  # (T, V)
        labels_np = labels.values  # (T,)

        # Convert to tensors
        series_tensor = torch.from_numpy(features_np).float()  # (T, V)
        labels_tensor = torch.from_numpy(labels_np).long()  # (T,)

        # Transpose to (V, T) and add batch dimension -> (1, V, T)
        series_tensor = series_tensor.T.unsqueeze(0)  # (1, V, T)
        labels_tensor = labels_tensor.unsqueeze(0)  # (1, T)

        print(f"  Series shape: {series_tensor.shape} (batch, variates, timesteps)")
        print(f"  Labels shape: {labels_tensor.shape} (batch, timesteps)")

        return series_tensor, labels_tensor

    def preprocess_split(
        self,
        file_name: str,
        split: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Preprocess a single split (train or test).

        Args:
            file_name: Excel file name
            split: 'train' or 'test'

        Returns:
            (series_tensor, labels_tensor, metadata_df)
        """
        print(f"\n{'=' * 70}")
        print(f"Preprocessing {split.upper()} split")
        print(f"{'=' * 70}\n")

        # Load raw data
        df = self.load_raw_data(file_name)

        # Parse timestamps
        df = self.parse_timestamps(df)

        # Extract features and labels
        features, labels = self.extract_features_and_labels(df)

        # Clean features
        features = self.clean_features(features)

        # Normalize (fit on train, transform on test)
        fit_normalization = (split == 'train')
        features = self.normalize_features(features, fit=fit_normalization)

        # Downsample
        features, labels = self.downsample(features, labels)

        # Store metadata for reference
        metadata = pd.DataFrame({
            'timestamp': features.index,
            'label': labels.values,
        })

        # Convert to tensors
        series_tensor, labels_tensor = self.to_tensors(features, labels)

        return series_tensor, labels_tensor, metadata

    def save_preprocessed(
        self,
        output_dir: str,
        train_data: Tuple[torch.Tensor, torch.Tensor, pd.DataFrame],
        test_data: Tuple[torch.Tensor, torch.Tensor, pd.DataFrame],
    ):
        """
        Save preprocessed data to disk.

        Args:
            output_dir: Directory to save to
            train_data: (series, labels, metadata) for training
            test_data: (series, labels, metadata) for testing
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"Saving preprocessed data to {output_path}")
        print(f"{'=' * 70}\n")

        # Unpack
        train_series, train_labels, train_metadata = train_data
        test_series, test_labels, test_metadata = test_data

        # Save tensors
        torch.save({
            'series': train_series,
            'labels': train_labels,
            'feature_names': self.feature_names,
            'feature_mean': self.feature_mean.values if self.feature_mean is not None else None,
            'feature_std': self.feature_std.values if self.feature_std is not None else None,
        }, output_path / 'swat_train.pt')

        torch.save({
            'series': test_series,
            'labels': test_labels,
            'feature_names': self.feature_names,
        }, output_path / 'swat_test.pt')

        # Save metadata (timestamps and labels)
        train_metadata.to_csv(output_path / 'swat_train_metadata.csv', index=False)
        test_metadata.to_csv(output_path / 'swat_test_metadata.csv', index=False)

        # Save preprocessing config
        config = {
            'downsample_factor': self.downsample_factor,
            'handle_missing': self.handle_missing,
            'normalize': self.normalize,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'train_length': train_series.shape[2],
            'test_length': test_series.shape[2],
        }

        import json
        with open(output_path / 'swat_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved:")
        print(f"  - swat_train.pt ({train_series.shape})")
        print(f"  - swat_test.pt ({test_series.shape})")
        print(f"  - swat_train_metadata.csv")
        print(f"  - swat_test_metadata.csv")
        print(f"  - swat_config.json")

        # Print file sizes
        train_size = (output_path / 'swat_train.pt').stat().st_size / 1024 / 1024
        test_size = (output_path / 'swat_test.pt').stat().st_size / 1024 / 1024
        print(f"\nFile sizes:")
        print(f"  - swat_train.pt: {train_size:.1f} MB")
        print(f"  - swat_test.pt: {test_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Preprocess SWaT dataset for Toto anomaly detection')
    parser.add_argument('--data_dir', type=str,
                        default='toto/data/SWaT.A1 & A2_Dec 2015/Physical',
                        help='Directory containing SWaT Excel files')
    parser.add_argument('--output_dir', type=str,
                        default='toto/data/preprocessed_swat',
                        help='Output directory for preprocessed data')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsample factor (1 = no downsampling)')
    parser.add_argument('--handle_missing', type=str, default='interpolate',
                        choices=['interpolate', 'forward_fill', 'drop'],
                        help='How to handle missing values')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Disable feature normalization')

    args = parser.parse_args()

    print("=" * 70)
    print("SWaT Dataset Preprocessing for Toto Anomaly Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Downsample factor: {args.downsample}")
    print(f"  Missing value handling: {args.handle_missing}")
    print(f"  Normalize: {not args.no_normalize}")

    # Initialize preprocessor
    preprocessor = SWaTPreprocessor(
        data_dir=args.data_dir,
        downsample_factor=args.downsample,
        handle_missing=args.handle_missing,
        normalize=not args.no_normalize,
    )

    # Preprocess training data (normal)
    train_data = preprocessor.preprocess_split(
        file_name='SWaT_Dataset_Normal_v1.xlsx',
        split='train',
    )

    # Preprocess test data (with attacks)
    test_data = preprocessor.preprocess_split(
        file_name='SWaT_Dataset_Attack_v0.xlsx',
        split='test',
    )

    # Save preprocessed data
    preprocessor.save_preprocessed(
        output_dir=args.output_dir,
        train_data=train_data,
        test_data=test_data,
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("Preprocessing Complete!")
    print(f"{'=' * 70}\n")

    train_series, train_labels, _ = train_data
    test_series, test_labels, _ = test_data

    print("Dataset Summary:")
    print(f"  Training:")
    print(f"    - Shape: {train_series.shape} (batch, variates, timesteps)")
    print(f"    - Duration: {train_series.shape[2]} timesteps")
    print(f"    - All normal data (no attacks)")
    print(f"  Testing:")
    print(f"    - Shape: {test_series.shape}")
    print(f"    - Duration: {test_series.shape[2]} timesteps")
    print(f"    - Normal: {(test_labels == 0).sum().item():,} ({(test_labels == 0).sum().item() / test_labels.numel() * 100:.1f}%)")
    print(f"    - Attack: {(test_labels == 1).sum().item():,} ({(test_labels == 1).sum().item() / test_labels.numel() * 100:.1f}%)")

    print(f"\nPreprocessed data saved to: {args.output_dir}")
    print("Ready for Toto anomaly detection!")


if __name__ == '__main__':
    main()
