#!/usr/bin/env python3
"""
SMD (Server Machine Dataset) Preprocessing for Toto Anomaly Detection

This script preprocesses the Server Machine Dataset for use with
the Toto time series foundation model for anomaly detection.

Dataset Information:
- Training: 28 files (8 machine-1, 9 machine-2, 11 machine-3), all normal
- Test: 28 files with corresponding labels
- Features: 38 server metrics (already normalized to [0,1])
- Variable timesteps: ~23,700 to ~28,700 per file

The preprocessing pipeline:
1. Load all .txt files from train/test directories
2. Load corresponding labels from test_label directory
3. Load interpretation labels (time ranges + affected dimensions)
4. Truncate all sequences to minimum length for uniform batching
5. Optional: normalize/standardize features
6. Convert to PyTorch tensors
7. Save as .pt files for efficient loading

Usage:
    python preprocess_smd.py [--output_dir ./preprocessed_data] [--downsample 1]
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

import numpy as np
import pandas as pd
import torch

# CRITICAL: Enable MPS fallback before importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class SMDPreprocessor:
    """Preprocessor for Server Machine Dataset (SMD)."""

    def __init__(
        self,
        data_dir: str = "toto/data/ServerMachineDataset",
        downsample_factor: int = 1,
        normalize: bool = False,  # Data already normalized [0,1]
        truncate_to_min: bool = True,
    ):
        """
        Initialize SMD preprocessor.

        Args:
            data_dir: Directory containing SMD train/test/test_label subdirectories
            downsample_factor: Downsample by this factor (1 = no downsampling)
            normalize: Whether to apply additional z-score normalization (default: False, data already normalized)
            truncate_to_min: Whether to truncate all sequences to minimum length (default: True)
        """
        self.data_dir = Path(data_dir)
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        self.truncate_to_min = truncate_to_min

        # Subdirectories
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.test_label_dir = self.data_dir / "test_label"
        self.interp_label_dir = self.data_dir / "interpretation_label"

        # Stats computed on training data (for optional normalization)
        self.feature_mean = None
        self.feature_std = None
        self.num_features = 38

        # Minimum sequence length (computed from data)
        self.min_length = None

    def get_file_list(self, directory: Path) -> List[Path]:
        """
        Get sorted list of .txt files in directory.

        Args:
            directory: Directory path

        Returns:
            Sorted list of file paths
        """
        files = sorted(directory.glob("*.txt"))
        return files

    def load_data_file(self, file_path: Path) -> np.ndarray:
        """
        Load a single SMD data file.

        Args:
            file_path: Path to .txt file

        Returns:
            NumPy array of shape (timesteps, 38)
        """
        data = np.loadtxt(file_path, delimiter=',')
        return data

    def load_label_file(self, file_path: Path) -> np.ndarray:
        """
        Load a single SMD label file.

        Args:
            file_path: Path to label .txt file

        Returns:
            NumPy array of shape (timesteps,) with binary labels (0=normal, 1=anomaly)
        """
        labels = np.loadtxt(file_path, dtype=int)
        return labels

    def load_interpretation_labels(self, file_path: Path) -> List[Dict]:
        """
        Load and parse interpretation label file.

        Format: "start-end:dim1,dim2,dim3,..."
        Example: "15849-16368:1,9,10,12,13,14,15"

        Args:
            file_path: Path to interpretation label .txt file

        Returns:
            List of dicts with 'time_range' and 'affected_dimensions'
        """
        if not file_path.exists():
            return []

        with open(file_path, 'r') as f:
            content = f.read().strip()

        if not content:
            return []

        anomalies = []
        for line in content.split('\n'):
            if line.strip():
                parts = line.split(':')
                time_range = parts[0]
                affected_dims = [int(x) for x in parts[1].split(',')]
                anomalies.append({
                    'time_range': time_range,
                    'affected_dimensions': affected_dims
                })
        return anomalies

    def find_min_length(self, file_list: List[Path]) -> int:
        """
        Find minimum sequence length across all files.

        Args:
            file_list: List of file paths to check

        Returns:
            Minimum length
        """
        min_len = float('inf')
        for file_path in file_list:
            data = self.load_data_file(file_path)
            min_len = min(min_len, len(data))
        return int(min_len)

    def normalize_features(
        self,
        features: np.ndarray,
        fit: bool = False,
    ) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Args:
            features: Feature array of shape (timesteps, num_features)
            fit: If True, compute mean/std from this data (training).
                 If False, use pre-computed stats (test).

        Returns:
            Normalized features
        """
        if not self.normalize:
            return features

        if fit:
            # Compute statistics on training data
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0)

            # Avoid division by zero (for constant features)
            self.feature_std[self.feature_std == 0] = 1.0

            print(f"  Computed mean/std statistics from training data")
        else:
            if self.feature_mean is None or self.feature_std is None:
                raise ValueError("Must fit normalization on training data first!")

        # Apply z-score normalization
        features_normalized = (features - self.feature_mean) / self.feature_std

        return features_normalized

    def downsample(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Downsample time series using median aggregation for features and
        majority vote for labels.

        Args:
            features: Feature array (T, V)
            labels: Label array (T,) or None for training data

        Returns:
            (downsampled_features, downsampled_labels)
        """
        if self.downsample_factor == 1:
            return features, labels

        print(f"  Downsampling by factor {self.downsample_factor}...")

        n = len(features)
        k = self.downsample_factor

        # Number of complete windows
        n_windows = n // k

        # Truncate to complete windows
        features_truncated = features[:n_windows * k]

        # Reshape into windows: (n_windows, k, n_features)
        features_reshaped = features_truncated.reshape(n_windows, k, -1)

        # Compute median along window dimension (axis=1)
        features_downsampled = np.median(features_reshaped, axis=1)  # (n_windows, n_features)

        # For labels: majority vote (if ANY anomaly in window, label as anomaly)
        labels_downsampled = None
        if labels is not None:
            labels_truncated = labels[:n_windows * k]
            labels_reshaped = labels_truncated.reshape(n_windows, k)
            labels_downsampled = (labels_reshaped.max(axis=1))  # (n_windows,)

        return features_downsampled, labels_downsampled

    def truncate_to_length(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray],
        target_length: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Truncate sequences to target length.

        Args:
            features: Feature array (T, V)
            labels: Label array (T,) or None
            target_length: Target sequence length

        Returns:
            (truncated_features, truncated_labels)
        """
        features_truncated = features[:target_length]
        labels_truncated = labels[:target_length] if labels is not None else None

        return features_truncated, labels_truncated

    def process_file(
        self,
        data_file: Path,
        label_file: Optional[Path] = None,
        interp_file: Optional[Path] = None,
        target_length: Optional[int] = None,
        fit_normalization: bool = False,
    ) -> Dict:
        """
        Process a single SMD file.

        Args:
            data_file: Path to data .txt file
            label_file: Path to label .txt file (None for training)
            interp_file: Path to interpretation label file (None for training)
            target_length: Target length to truncate to (None = no truncation)
            fit_normalization: Whether to fit normalization stats on this file

        Returns:
            Dict with 'features', 'labels', 'interpretation', 'file_name', 'original_length'
        """
        # Load data
        features = self.load_data_file(data_file)
        original_length = len(features)

        # Load labels (if test set)
        labels = None
        if label_file is not None and label_file.exists():
            labels = self.load_label_file(label_file)

        # Load interpretation labels (if available)
        interpretation = []
        if interp_file is not None and interp_file.exists():
            interpretation = self.load_interpretation_labels(interp_file)

        # Truncate to target length if specified
        if target_length is not None:
            features, labels = self.truncate_to_length(features, labels, target_length)

        # Normalize features (optional)
        features = self.normalize_features(features, fit=fit_normalization)

        # Downsample
        features, labels = self.downsample(features, labels)

        return {
            'features': features,
            'labels': labels,
            'interpretation': interpretation,
            'file_name': data_file.stem,
            'original_length': original_length,
            'final_length': len(features),
        }

    def process_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Process all files for a split (train or test).

        Args:
            split: 'train' or 'test'

        Returns:
            (features_batch, labels_batch, metadata_list)
            - features_batch: (num_files, num_features, timesteps)
            - labels_batch: (num_files, timesteps)
            - metadata_list: List of metadata dicts for each file
        """
        print(f"\n{'=' * 70}")
        print(f"Processing {split.upper()} split")
        print(f"{'=' * 70}\n")

        # Get file lists
        if split == 'train':
            data_dir = self.train_dir
            label_dir = None
            interp_dir = None
        else:
            data_dir = self.test_dir
            label_dir = self.test_label_dir
            interp_dir = self.interp_label_dir

        data_files = self.get_file_list(data_dir)
        print(f"Found {len(data_files)} files in {data_dir}")

        # Find minimum length if truncating
        if self.truncate_to_min:
            if self.min_length is None:
                print(f"Finding minimum sequence length across all files...")
                all_files = self.get_file_list(self.train_dir) + self.get_file_list(self.test_dir)
                self.min_length = self.find_min_length(all_files)
                print(f"  Minimum length: {self.min_length}")
            target_length = self.min_length
        else:
            target_length = None

        # Process all files
        features_list = []
        labels_list = []
        metadata_list = []

        for i, data_file in enumerate(data_files):
            print(f"\nProcessing {data_file.name} ({i+1}/{len(data_files)})...")

            # Get corresponding label and interpretation files
            label_file = label_dir / data_file.name if label_dir else None
            interp_file = interp_dir / data_file.name if interp_dir else None

            # Process file (fit normalization only on first training file)
            fit_norm = (split == 'train' and i == 0 and self.normalize)
            result = self.process_file(
                data_file=data_file,
                label_file=label_file,
                interp_file=interp_file,
                target_length=target_length,
                fit_normalization=fit_norm,
            )

            features_list.append(result['features'])
            labels_list.append(result['labels'] if result['labels'] is not None else np.zeros(result['final_length'], dtype=int))

            # Store metadata
            metadata = {
                'file_name': result['file_name'],
                'machine_type': result['file_name'].split('-')[0] + '-' + result['file_name'].split('-')[1],  # e.g., 'machine-1'
                'instance': int(result['file_name'].split('-')[2]),  # e.g., 1, 2, 3
                'original_length': result['original_length'],
                'final_length': result['final_length'],
                'interpretation': result['interpretation'],
            }

            # Add label statistics for test set
            if result['labels'] is not None:
                metadata['num_anomalies'] = int(np.sum(result['labels']))
                metadata['anomaly_rate'] = float(np.mean(result['labels']))

            metadata_list.append(metadata)

            print(f"  Original length: {result['original_length']}")
            print(f"  Final length: {result['final_length']}")
            if result['labels'] is not None:
                print(f"  Anomalies: {np.sum(result['labels'])} ({np.mean(result['labels'])*100:.2f}%)")

        # Stack into batch tensors
        # Features: (num_files, timesteps, num_features) -> (num_files, num_features, timesteps)
        features_batch = np.stack(features_list, axis=0)  # (N, T, V)
        features_batch = np.transpose(features_batch, (0, 2, 1))  # (N, V, T)

        # Labels: (num_files, timesteps)
        labels_batch = np.stack(labels_list, axis=0)  # (N, T)

        print(f"\n{'=' * 70}")
        print(f"Batch Summary for {split.upper()}")
        print(f"{'=' * 70}")
        print(f"Features shape: {features_batch.shape} (batch, variates, timesteps)")
        print(f"Labels shape: {labels_batch.shape} (batch, timesteps)")

        if split == 'test':
            total_anomalies = np.sum(labels_batch)
            total_points = labels_batch.size
            print(f"Total anomalies: {total_anomalies:,} / {total_points:,} ({total_anomalies/total_points*100:.2f}%)")

        return features_batch, labels_batch, metadata_list

    def save_preprocessed(
        self,
        output_dir: str,
        train_data: Tuple[np.ndarray, np.ndarray, List[Dict]],
        test_data: Tuple[np.ndarray, np.ndarray, List[Dict]],
    ):
        """
        Save preprocessed data to disk.

        Args:
            output_dir: Directory to save to
            train_data: (features, labels, metadata) for training
            test_data: (features, labels, metadata) for testing
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"Saving preprocessed data to {output_path}")
        print(f"{'=' * 70}\n")

        # Unpack
        train_features, train_labels, train_metadata = train_data
        test_features, test_labels, test_metadata = test_data

        # Convert to PyTorch tensors
        train_series = torch.from_numpy(train_features).float()
        train_labels_tensor = torch.from_numpy(train_labels).long()
        test_series = torch.from_numpy(test_features).float()
        test_labels_tensor = torch.from_numpy(test_labels).long()

        # Save tensors
        torch.save({
            'series': train_series,
            'labels': train_labels_tensor,
            'num_features': self.num_features,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'metadata': train_metadata,
        }, output_path / 'smd_train.pt')

        torch.save({
            'series': test_series,
            'labels': test_labels_tensor,
            'num_features': self.num_features,
            'metadata': test_metadata,
        }, output_path / 'smd_test.pt')

        # Save metadata as JSON for easy inspection
        with open(output_path / 'smd_train_metadata.json', 'w') as f:
            json.dump(train_metadata, f, indent=2)

        with open(output_path / 'smd_test_metadata.json', 'w') as f:
            json.dump(test_metadata, f, indent=2)

        # Save preprocessing config
        config = {
            'dataset': 'Server Machine Dataset (SMD)',
            'downsample_factor': self.downsample_factor,
            'normalize': self.normalize,
            'truncate_to_min': self.truncate_to_min,
            'num_features': self.num_features,
            'min_length': self.min_length,
            'num_train_files': len(train_metadata),
            'num_test_files': len(test_metadata),
            'train_shape': list(train_series.shape),
            'test_shape': list(test_series.shape),
        }

        with open(output_path / 'smd_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved:")
        print(f"  - smd_train.pt {train_series.shape}")
        print(f"  - smd_test.pt {test_series.shape}")
        print(f"  - smd_train_metadata.json")
        print(f"  - smd_test_metadata.json")
        print(f"  - smd_config.json")

        # Print file sizes
        train_size = (output_path / 'smd_train.pt').stat().st_size / 1024 / 1024
        test_size = (output_path / 'smd_test.pt').stat().st_size / 1024 / 1024
        print(f"\nFile sizes:")
        print(f"  - smd_train.pt: {train_size:.1f} MB")
        print(f"  - smd_test.pt: {test_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Preprocess SMD dataset for Toto anomaly detection')
    parser.add_argument('--data_dir', type=str,
                        default='toto/data/ServerMachineDataset',
                        help='Directory containing SMD train/test subdirectories')
    parser.add_argument('--output_dir', type=str,
                        default='toto/data/preprocessed_smd',
                        help='Output directory for preprocessed data')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsample factor (1 = no downsampling)')
    parser.add_argument('--normalize', action='store_true',
                        help='Enable z-score normalization (data already normalized [0,1])')
    parser.add_argument('--no_truncate', action='store_true',
                        help='Disable truncation to minimum length (will pad instead)')

    args = parser.parse_args()

    print("=" * 70)
    print("SMD Dataset Preprocessing for Toto Anomaly Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Downsample factor: {args.downsample}")
    print(f"  Normalize: {args.normalize}")
    print(f"  Truncate to min length: {not args.no_truncate}")

    # Initialize preprocessor
    preprocessor = SMDPreprocessor(
        data_dir=args.data_dir,
        downsample_factor=args.downsample,
        normalize=args.normalize,
        truncate_to_min=not args.no_truncate,
    )

    # Preprocess training data
    train_data = preprocessor.process_split(split='train')

    # Preprocess test data
    test_data = preprocessor.process_split(split='test')

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

    train_features, train_labels, train_metadata = train_data
    test_features, test_labels, test_metadata = test_data

    print("Dataset Summary:")
    print(f"  Training:")
    print(f"    - Shape: {train_features.shape} (batch, variates, timesteps)")
    print(f"    - Files: {len(train_metadata)}")
    print(f"    - All normal data (no attacks)")
    print(f"  Testing:")
    print(f"    - Shape: {test_features.shape}")
    print(f"    - Files: {len(test_metadata)}")

    # Calculate anomaly statistics
    num_anomalies = sum(m.get('num_anomalies', 0) for m in test_metadata)
    total_points = test_features.shape[0] * test_features.shape[2]
    print(f"    - Normal: {total_points - num_anomalies:,} ({(total_points - num_anomalies)/total_points*100:.1f}%)")
    print(f"    - Anomaly: {num_anomalies:,} ({num_anomalies/total_points*100:.1f}%)")

    print(f"\nPreprocessed data saved to: {args.output_dir}")
    print("Ready for Toto anomaly detection!")


if __name__ == '__main__':
    main()
