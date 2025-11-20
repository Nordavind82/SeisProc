#!/usr/bin/env python3
"""
Runtime GPU checkbox diagnostic - run this WHILE the app is running
"""
import sys

print("=" * 70)
print("GPU CHECKBOX RUNTIME DIAGNOSTIC")
print("=" * 70)

# Test 1: Check if GPU modules can be imported
print("\n[1] Testing GPU module imports...")
try:
    from processors.gpu.device_manager import get_device_manager
    from processors.tf_denoise_gpu import TFDenoiseGPU
    print("    ✓ GPU modules import successfully")
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"    ✗ GPU modules failed to import: {e}")
    GPU_AVAILABLE = False

# Test 2: Initialize device manager
if GPU_AVAILABLE:
    print("\n[2] Initializing GPU device manager...")
    try:
        dm = get_device_manager()
        print(f"    ✓ Device manager initialized")
        print(f"    - Device: {dm.get_device_name()}")
        print(f"    - Type: {dm.get_device_type()}")
        print(f"    - GPU available: {dm.is_gpu_available()}")
    except Exception as e:
        print(f"    ✗ Device manager failed: {e}")
else:
    print("\n[2] Skipping device manager test (GPU modules not available)")

# Test 3: Check control_panel module
print("\n[3] Checking control_panel module...")
try:
    from views.control_panel import GPU_AVAILABLE as CP_GPU_AVAILABLE
    print(f"    - GPU_AVAILABLE in control_panel: {CP_GPU_AVAILABLE}")
except Exception as e:
    print(f"    ✗ Failed to import control_panel: {e}")

# Test 4: PyTorch check
print("\n[4] PyTorch configuration...")
try:
    import torch
    print(f"    - PyTorch version: {torch.__version__}")
    print(f"    - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    - CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"    - MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
except Exception as e:
    print(f"    ✗ PyTorch check failed: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("\nIf all tests pass, the GPU checkbox should be visible in the UI.")
print("If you still don't see it, try:")
print("1. Restart the application")
print("2. Select 'TF-Denoise (S-Transform)' algorithm")
print("3. Clear Python cache: rm -rf __pycache__ views/__pycache__ processors/__pycache__")
print("=" * 70)
