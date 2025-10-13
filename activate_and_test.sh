#!/bin/bash
# Quick activation and testing script for Toto with M4 GPU support

echo "============================================================"
echo "Activating Toto Virtual Environment"
echo "============================================================"

# Activate virtual environment
source .venv/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
    echo ""
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo ""

    # Check MPS availability
    echo "Checking M4 GPU (MPS) availability..."
    python -c "import torch; print('✓ MPS available!' if torch.backends.mps.is_available() else '✗ MPS not available')"
    echo ""

    echo "============================================================"
    echo "Available Commands:"
    echo "============================================================"
    echo "  python test_mps_setup.py       - Run setup verification tests"
    echo "  python quick_start_example.py  - Run a quick forecasting example"
    echo "  deactivate                     - Exit virtual environment"
    echo ""
    echo "Ready to use Toto with M4 GPU!"
    echo "============================================================"
else
    echo "✗ Failed to activate virtual environment"
    exit 1
fi
