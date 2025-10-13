# M4 GPU Setup Guide for Toto

This guide explains the changes made to enable Toto to run on Apple Silicon M4 GPU using the MPS (Metal Performance Shaders) backend.

## What Was Done

### 1. Virtual Environment Setup
A Python virtual environment was created in `.venv/` with all dependencies installed.

### 2. Code Adaptations for M4 GPU

#### Added Device Detection Utility (`toto/model/util.py`)
A new `get_device()` function was added that automatically detects and selects the best available device:
- **CUDA** (NVIDIA GPUs) - if available
- **MPS** (Apple Silicon GPUs) - if available
- **CPU** - as fallback

```python
from toto.model.util import get_device

device = get_device()
model.to(device)
```

#### Updated Evaluation Script (`toto/evaluation/run_lsf_eval.py`)
- Uses the new `get_device()` function instead of hardcoded CUDA check
- Skips `torch.compile` with `mode="max-autotune"` on MPS due to limited support
- Logs the device being used

#### Updated README
The Quick Start example now includes:
- Import of the `get_device()` utility
- Automatic device detection
- Conditional compilation to avoid issues on MPS

## How to Use

### Activate the Virtual Environment
```bash
source .venv/bin/activate
```

### Important: MPS Fallback for Student-T Sampling

The Toto model uses Student-T distributions for probabilistic forecasting. The Student-T sampling requires the `_standard_gamma` operation, which is **not yet implemented on MPS**.

**Solution**: We've automatically enabled CPU fallback for this operation. The scripts (`quick_start_example.py` and `test_mps_setup.py`) already include:

```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # BEFORE importing torch
```

This allows unsupported operations to fall back to CPU while keeping most computations on the M4 GPU. You'll see a warning like:
```
UserWarning: The operator 'aten::_standard_gamma' is not currently supported
on the MPS backend and will fall back to run on the CPU.
```

**This is normal and expected!** The vast majority of operations still run on your M4 GPU.

### Run the Test Script
```bash
python test_mps_setup.py
```

This will verify that:
- MPS is available on your system
- Device detection works correctly
- Basic tensor operations work on MPS
- Models can be loaded and run on MPS

### Use in Your Code

**Important**: Always set the MPS fallback environment variable **before** importing torch:

```python
# STEP 1: Enable MPS fallback BEFORE importing torch
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# STEP 2: Now import torch and other libraries
import torch
from toto.model.toto import Toto
from toto.model.util import get_device
from toto.inference.forecaster import TotoForecaster

# Load model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')

# Automatically use M4 GPU
device = get_device()
print(f"Using device: {device}")
toto.to(device)

# Create forecaster
forecaster = TotoForecaster(toto.model)

# Your code here...
```

**Why this order matters**: PyTorch initializes MPS on first import. Setting the environment variable after import has no effect.

### Run Evaluations
```bash
python toto/evaluation/run_lsf_eval.py --datasets ETTh1 --checkpoint-path Datadog/Toto-Open-Base-1.0
```

The script will automatically detect and use your M4 GPU.

## Known Limitations

### MPS Backend Limitations
- **torch.compile**: Some compilation modes (like "max-autotune") may not work on MPS. The code now skips compilation on MPS to avoid issues.
- **Some operations**: Not all PyTorch operations are fully optimized for MPS yet. Most common operations work well.
- **Deterministic algorithms**: Some deterministic operations might fall back to CPU.

### Performance Notes
- MPS performance is generally good for inference but may be slower than CUDA for some operations
- The M4 GPU is optimized for unified memory architecture, which works well for transformer models
- For very large batch sizes, you may need to reduce batch size compared to CUDA setups

## Troubleshooting

### If MPS is not detected:
1. Ensure you're running on an Apple Silicon Mac (M1/M2/M3/M4)
2. Update to the latest macOS version
3. Verify PyTorch installation: `python -c "import torch; print(torch.backends.mps.is_available())"`

### If you encounter errors:
1. Try running without `torch.compile`:
   ```python
   # Just skip the compile step
   model.to(device)
   model.eval()
   ```

2. For memory errors, reduce batch sizes:
   ```bash
   python toto/evaluation/run_lsf_eval.py --batch-size 1 --samples-per-batch 128
   ```

3. Check PyTorch MPS documentation: https://pytorch.org/docs/stable/notes/mps.html

## Verification

Run the test script to verify everything is working:

```bash
source .venv/bin/activate
python test_mps_setup.py
```

You should see all tests passing with "✓ PASSED" status.

## Summary

Your Toto installation is now configured to automatically use your M4 GPU for accelerated inference! The code will:
- Automatically detect MPS availability
- Use MPS when available (falling back to CPU if not)
- Skip incompatible compilation modes on MPS
- Log which device is being used

Happy forecasting! 🚀
