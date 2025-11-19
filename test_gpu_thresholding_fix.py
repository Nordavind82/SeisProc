"""
Test GPU thresholding with MPS compatibility fix.
"""

import sys

import numpy as np
import torch
from processors.gpu.device_manager import get_device_manager
from processors.gpu.stransform_gpu import STransformGPU
from processors.gpu.thresholding_gpu import ThresholdingGPU

def test_mad_thresholding_mps():
    """Test MAD thresholding on MPS with the fix."""
    print("=" * 60)
    print("Testing MAD Thresholding GPU with MPS Fix")
    print("=" * 60)

    # Get device manager
    dm = get_device_manager()
    print(f"Device: {dm.get_device_name()}")
    print(f"Device type: {dm.get_device_type()}\n")

    # Create test signal ensemble (simulating 7 traces in aperture)
    n_samples = 2049
    n_traces = 7
    sample_rate = 500.0  # Hz

    # Create ensemble with signal + noise
    ensemble = []
    for i in range(n_traces):
        t = np.arange(n_samples) / sample_rate
        signal = (
            np.sin(2 * np.pi * 10 * t) +  # 10 Hz signal
            0.5 * np.sin(2 * np.pi * 25 * t)  # 25 Hz signal
        )
        # Add varying noise levels
        noise_level = 0.1 + 0.05 * i
        signal += noise_level * np.random.randn(n_samples)
        ensemble.append(signal)

    ensemble = np.array(ensemble).T  # (n_samples, n_traces)
    center_trace = ensemble[:, 3]  # Middle trace

    print(f"Test ensemble: {n_traces} traces × {n_samples} samples")
    print(f"Ensemble shape: {ensemble.shape}\n")

    # Compute S-Transforms for ensemble
    st_gpu = STransformGPU(device=dm.device)
    print("Computing S-Transforms for ensemble on GPU...")

    st_ensemble = []
    for i in range(n_traces):
        S, freqs = st_gpu.forward(
            ensemble[:, i],
            fmin=5.0,
            fmax=100.0,
            sample_rate=sample_rate
        )
        st_ensemble.append(S)

    st_ensemble = np.array(st_ensemble)  # (n_traces, n_freqs, n_times)
    st_center = st_ensemble[3]  # Center trace S-Transform

    print(f"✓ S-Transform ensemble computed")
    print(f"  Ensemble shape: {st_ensemble.shape}")
    print(f"  Center trace shape: {st_center.shape}\n")

    # Apply MAD thresholding
    print("Applying MAD thresholding on GPU...")
    thresholder = ThresholdingGPU(
        device=dm.device,
        threshold_k=3.0,
        threshold_type='soft'
    )

    try:
        import time
        start_time = time.time()

        st_denoised = thresholder.apply_mad_thresholding(
            st_center,
            st_ensemble,
            spatial_dim=0
        )

        elapsed = time.time() - start_time

        print(f"✓ MAD thresholding completed successfully!")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Denoised shape: {st_denoised.shape}")

        # Compare magnitudes
        orig_mag = np.abs(st_center)
        denoised_mag = np.abs(st_denoised)

        print(f"  Original magnitude: {orig_mag.min():.3e} to {orig_mag.max():.3e}")
        print(f"  Denoised magnitude: {denoised_mag.min():.3e} to {denoised_mag.max():.3e}")

        # Count zeros (thresholded coefficients)
        n_zeros = np.sum(denoised_mag < 1e-10)
        total = denoised_mag.size
        pct_zeros = 100 * n_zeros / total
        print(f"  Thresholded coefficients: {n_zeros}/{total} ({pct_zeros:.1f}%)")

        print(f"\n✓ MPS MAD thresholding test PASSED!\n")

        return True

    except Exception as e:
        print(f"\n✗ MAD thresholding failed with error:")
        print(f"  {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mad_thresholding_mps()

    if success:
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    else:
        print("=" * 60)
        print("TESTS FAILED ✗")
        print("=" * 60)
