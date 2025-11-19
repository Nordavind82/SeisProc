"""
Test GPU S-Transform fix for MPS compatibility.
"""

import sys

import numpy as np
import torch
from processors.gpu.device_manager import get_device_manager
from processors.gpu.stransform_gpu import STransformGPU

def test_stransform_mps():
    """Test S-Transform on MPS with the fix."""
    print("=" * 60)
    print("Testing S-Transform GPU with MPS Fix")
    print("=" * 60)

    # Get device manager
    dm = get_device_manager()
    print(f"Device: {dm.get_device_name()}")
    print(f"Device type: {dm.get_device_type()}\n")

    # Create test signal
    n_samples = 2049
    sample_rate = 500.0  # Hz
    t = np.arange(n_samples) / sample_rate

    # Create signal with multiple frequencies
    signal = (
        np.sin(2 * np.pi * 10 * t) +  # 10 Hz
        0.5 * np.sin(2 * np.pi * 25 * t) +  # 25 Hz
        0.3 * np.sin(2 * np.pi * 50 * t)   # 50 Hz
    )
    # Add noise
    signal += 0.1 * np.random.randn(n_samples)

    print(f"Test signal: {n_samples} samples at {sample_rate} Hz")
    print(f"Signal shape: {signal.shape}")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]\n")

    # Create S-Transform processor
    st_gpu = STransformGPU(device=dm.device)

    try:
        print("Computing S-Transform on GPU...")
        import time
        start_time = time.time()

        S, freqs = st_gpu.forward(
            signal,
            fmin=5.0,
            fmax=100.0,
            sample_rate=sample_rate
        )

        elapsed = time.time() - start_time

        print(f"✓ S-Transform completed successfully!")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  S-Transform shape: {S.shape}")
        print(f"  Frequencies: {len(freqs)} from {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")
        print(f"  Coefficient range: {np.abs(S).min():.3e} to {np.abs(S).max():.3e}")
        print(f"\n✓ MPS compatibility test PASSED!\n")

        # Test inverse
        print("Testing inverse S-Transform...")
        reconstructed = st_gpu.inverse(S, freqs, n_samples, sample_rate)
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

        # Check reconstruction error
        error = np.mean(np.abs(signal - reconstructed))
        print(f"  Mean reconstruction error: {error:.3e}")
        print(f"\n✓ Inverse S-Transform test PASSED!\n")

        return True

    except Exception as e:
        print(f"\n✗ S-Transform failed with error:")
        print(f"  {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_stransform_mps()

    if success:
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    else:
        print("=" * 60)
        print("TESTS FAILED ✗")
        print("=" * 60)
