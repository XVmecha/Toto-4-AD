#!/usr/bin/env python3
"""
Test script to verify MPS (Apple Silicon GPU) setup for Toto model.
This script tests basic functionality with the M4 GPU.
"""

# IMPORTANT: Set MPS fallback BEFORE importing torch
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import sys

def test_device_availability():
    """Test which devices are available"""
    print("=" * 60)
    print("Device Availability Test")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) is available!")
    else:
        print("✗ MPS is not available")
        return False

    return True

def test_device_detection():
    """Test the get_device utility function"""
    print("\n" + "=" * 60)
    print("Device Detection Test")
    print("=" * 60)

    try:
        from toto.model.util import get_device
        device = get_device()
        print(f"✓ Detected device: {device}")

        if device.type == "mps":
            print("✓ Successfully configured to use MPS (M4 GPU)")
            return True
        else:
            print(f"⚠ Using {device.type} instead of MPS")
            return True
    except Exception as e:
        print(f"✗ Error detecting device: {e}")
        return False

def test_basic_tensor_operations():
    """Test basic tensor operations on MPS"""
    print("\n" + "=" * 60)
    print("Basic Tensor Operations Test")
    print("=" * 60)

    try:
        device = torch.device("mps")

        # Create tensors on MPS
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)

        # Perform operations
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication successful on {device}")
        print(f"  Result shape: {z.shape}")

        return True
    except Exception as e:
        print(f"✗ Error with tensor operations: {e}")
        return False

def test_model_loading():
    """Test loading a small model on MPS"""
    print("\n" + "=" * 60)
    print("Model Loading Test")
    print("=" * 60)

    try:
        from toto.model.util import get_device

        device = get_device()
        print(f"Loading model to device: {device}")

        # Create a simple model to test
        model = torch.nn.Linear(10, 10)
        model.to(device)

        # Test forward pass
        x = torch.randn(5, 10, device=device)
        output = model(x)

        print(f"✓ Model successfully loaded and ran on {device}")
        print(f"  Input shape: {x.shape}, Output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Error loading/running model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Toto MPS (M4 GPU) Setup Verification")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Device Availability", test_device_availability()))
    results.append(("Device Detection", test_device_detection()))
    results.append(("Basic Tensor Operations", test_basic_tensor_operations()))
    results.append(("Model Loading", test_model_loading()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Your M4 GPU setup is ready.")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
