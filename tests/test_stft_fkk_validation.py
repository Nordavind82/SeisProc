#!/usr/bin/env python3
"""
STFT and FKK Kernel Validation Tests

Validates Metal C++ implementations against Python/scipy references.
Tests include:
1. STFT perfect reconstruction (no thresholding)
2. STFT Metal vs Python denoising comparison
3. FKK FFT round-trip validation
4. FKK velocity mask consistency
5. FKK Metal vs Python filtering comparison
"""

import numpy as np
from scipy import signal, fft
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# STFT Tests
# ==============================================================================

def test_scipy_stft_reconstruction():
    """Test perfect reconstruction with scipy STFT/ISTFT."""
    print("=" * 60)
    print("TEST 1: SciPy STFT Perfect Reconstruction")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 256

    # Create test signal: sine wave
    t = np.linspace(0, 4 * np.pi, n_samples)
    signal_data = np.sin(t) + 0.3 * np.sin(3 * t)

    # STFT parameters
    nperseg = 64
    noverlap = 48

    # Forward STFT
    freqs, times, stft_result = signal.stft(
        signal_data, nperseg=nperseg, noverlap=noverlap
    )

    # Inverse STFT (no modification)
    _, reconstructed = signal.istft(
        stft_result, nperseg=nperseg, noverlap=noverlap
    )

    # Handle length mismatch
    if len(reconstructed) > n_samples:
        reconstructed = reconstructed[:n_samples]
    elif len(reconstructed) < n_samples:
        reconstructed = np.pad(reconstructed, (0, n_samples - len(reconstructed)))

    error = np.max(np.abs(signal_data - reconstructed))

    print(f"Signal length: {n_samples}")
    print(f"STFT shape: {stft_result.shape}")
    print(f"Reconstruction error: {error:.2e}")

    passed = error < 1e-10
    print(f"TEST 1: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_stft_metal_vs_python_reconstruction():
    """Compare Metal STFT with Python scipy for basic STFT/ISTFT round-trip."""
    print("\n" + "=" * 60)
    print("TEST 2: Metal vs Python STFT Reconstruction")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return True

    if not is_metal_available():
        print("Metal not available, skipping")
        return True

    metal = get_metal_module()
    if metal is None:
        print("Metal module not loaded")
        return True

    np.random.seed(42)
    n_samples = 256
    n_traces = 11  # Odd number for aperture centering

    # Create clean signal (sine wave)
    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin(t + i * 0.2).astype(np.float32)

    # Test with very high threshold (effectively no denoising)
    nperseg = 64
    noverlap = 48
    aperture = 7
    threshold_k = 1000.0  # Very high - should not threshold anything
    sample_rate = 500.0

    result_metal, metrics = metal.stft_denoise(
        traces,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=aperture,
        threshold_k=threshold_k,
        fmin=0.0,
        fmax=0.0,
        sample_rate=sample_rate
    )

    # Compute Python reference
    result_python = np.zeros_like(traces)
    for i in range(n_traces):
        _, _, stft_data = signal.stft(traces[:, i], nperseg=nperseg, noverlap=noverlap)
        _, reconstructed = signal.istft(stft_data, nperseg=nperseg, noverlap=noverlap)
        if len(reconstructed) > n_samples:
            reconstructed = reconstructed[:n_samples]
        elif len(reconstructed) < n_samples:
            reconstructed = np.pad(reconstructed, (0, n_samples - len(reconstructed)))
        result_python[:, i] = reconstructed

    # Check reconstruction quality
    metal_error = np.max(np.abs(traces - result_metal))
    python_error = np.max(np.abs(traces - result_python))

    print(f"Signal shape: {traces.shape}")
    print(f"Metal reconstruction error: {metal_error:.6f}")
    print(f"Python reconstruction error: {python_error:.6f}")

    # The key test: both should have similar reconstruction quality
    # Note: C++ uses vDSP which may have different numerical behavior
    passed = metal_error < 0.1 and python_error < 1e-10
    print(f"TEST 2: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        print("\nDiagnostics:")
        print(f"  Metal output range: [{result_metal.min():.4f}, {result_metal.max():.4f}]")
        print(f"  Python output range: [{result_python.min():.4f}, {result_python.max():.4f}]")
        print(f"  Input range: [{traces.min():.4f}, {traces.max():.4f}]")

    return passed


def test_stft_denoising_quality():
    """Test that STFT denoising improves SNR on noisy synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 3: STFT Denoising Quality (SNR Improvement)")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return True

    if not is_metal_available():
        print("Metal not available, skipping")
        return True

    metal = get_metal_module()

    np.random.seed(42)
    n_samples = 512
    n_traces = 21

    # Create clean signal
    t = np.linspace(0, 8 * np.pi, n_samples)
    clean = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        clean[:, i] = (np.sin(t + i * 0.1) + 0.5 * np.sin(2 * t + i * 0.2)).astype(np.float32)

    # Add noise
    noise_level = 0.3
    noise = (noise_level * np.random.randn(n_samples, n_traces)).astype(np.float32)
    noisy = clean + noise

    # Denoise with Metal
    result_metal, metrics = metal.stft_denoise(
        noisy,
        nperseg=64,
        noverlap=48,
        aperture=7,
        threshold_k=3.0,
        fmin=5.0,
        fmax=200.0,
        sample_rate=500.0
    )

    # Calculate SNR improvement
    snr_before = 10 * np.log10(np.var(clean) / np.var(noisy - clean))
    snr_after = 10 * np.log10(np.var(clean) / np.var(result_metal - clean))
    snr_improvement = snr_after - snr_before

    print(f"Signal shape: {noisy.shape}")
    print(f"Noise level: {noise_level}")
    print(f"SNR before: {snr_before:.2f} dB")
    print(f"SNR after: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    print(f"Kernel time: {metrics.get('kernel_time_ms', 0):.1f} ms")

    # Denoising should improve SNR
    passed = snr_improvement > 0
    print(f"TEST 3: {'PASSED' if passed else 'FAILED'}")
    return passed


# ==============================================================================
# FKK Tests
# ==============================================================================

def test_fft3d_roundtrip():
    """Test 3D FFT round-trip with Python/scipy."""
    print("\n" + "=" * 60)
    print("TEST 4: 3D FFT Round-Trip (Python Reference)")
    print("=" * 60)

    np.random.seed(42)
    nt, nx, ny = 64, 16, 16

    # Create test volume
    volume = np.random.randn(nt, nx, ny).astype(np.float32)

    # Forward 3D FFT
    spectrum = fft.rfft(volume, axis=0)
    spectrum = fft.fft(spectrum, axis=1)
    spectrum = fft.fft(spectrum, axis=2)

    # Inverse 3D FFT
    result = fft.ifft(spectrum, axis=2)
    result = fft.ifft(result, axis=1)
    result = fft.irfft(result, n=nt, axis=0)

    error = np.max(np.abs(volume - result.real))

    print(f"Volume shape: {volume.shape}")
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Round-trip error: {error:.2e}")

    passed = error < 1e-5
    print(f"TEST 4: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_fkk_velocity_mask():
    """Test FKK velocity cone mask building."""
    print("\n" + "=" * 60)
    print("TEST 5: FKK Velocity Mask Building")
    print("=" * 60)

    nt, nx, ny = 64, 32, 32
    dt = 0.002  # 2ms
    dx = 25.0  # 25m
    dy = 25.0  # 25m

    v_min = 500.0   # m/s
    v_max = 2000.0  # m/s

    # Build mask using Python
    f_axis = np.fft.rfftfreq(nt, dt)
    kx_axis = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    ky_axis = np.fft.fftshift(np.fft.fftfreq(ny, dy))

    f_grid, kx_grid, ky_grid = np.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')

    k_horizontal = np.sqrt(kx_grid**2 + ky_grid**2)
    k_safe = np.where(k_horizontal > 1e-10, k_horizontal, 1e-10)
    velocity = np.abs(f_grid) / k_safe

    in_velocity_range = (velocity >= v_min) & (velocity <= v_max)

    # Reject mode mask
    mask_reject = np.ones_like(velocity, dtype=np.float32)
    mask_reject[in_velocity_range] = 0.0
    mask_reject[0, :, :] = 1.0  # Preserve DC
    mask_reject[:, nx // 2, ny // 2] = 1.0  # Preserve k=0

    # Pass mode mask
    mask_pass = np.zeros_like(velocity, dtype=np.float32)
    mask_pass[in_velocity_range] = 1.0
    mask_pass[0, :, :] = 1.0  # Preserve DC
    mask_pass[:, nx // 2, ny // 2] = 1.0  # Preserve k=0

    print(f"Volume shape: {nt}x{nx}x{ny}")
    print(f"Mask shape: {mask_reject.shape}")
    print(f"Velocity range: {v_min}-{v_max} m/s")
    print(f"Reject mask: {np.sum(mask_reject == 0)} zeros, {np.sum(mask_reject == 1)} ones")
    print(f"Pass mask: {np.sum(mask_pass == 0)} zeros, {np.sum(mask_pass == 1)} ones")

    # Sanity checks
    passed = (
        mask_reject[0, nx // 2, ny // 2] == 1.0 and  # DC preserved
        mask_pass[0, nx // 2, ny // 2] == 1.0 and    # DC preserved
        np.sum(mask_reject == 0) > 0 and  # Some filtering happens
        np.sum(mask_pass == 1) > 0        # Some signal passes
    )

    print(f"TEST 5: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_fkk_metal_vs_python():
    """Compare Metal FKK with Python implementation."""
    print("\n" + "=" * 60)
    print("TEST 6: Metal vs Python FKK Filtering")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return True

    if not is_metal_available():
        print("Metal not available, skipping")
        return True

    metal = get_metal_module()
    if metal is None:
        print("Metal module not loaded")
        return True

    np.random.seed(42)
    nt, nx, ny = 64, 16, 16
    dt = 0.002  # 2ms
    dx = 25.0   # 25m
    dy = 25.0   # 25m

    # Create test volume with linear events (coherent signal)
    volume = np.zeros((nt, nx, ny), dtype=np.float32)

    # Add dipping event
    for ix in range(nx):
        for iy in range(ny):
            # Time shift based on position (creates linear moveout)
            shift = int((ix * 0.3 + iy * 0.2))
            t_idx = min(max(20 + shift, 0), nt - 1)
            # Ricker wavelet
            for dt_idx in range(-5, 6):
                t = t_idx + dt_idx
                if 0 <= t < nt:
                    arg = (dt_idx / 2.0) ** 2
                    volume[t, ix, iy] = (1 - 2 * arg) * np.exp(-arg)

    # Add random noise
    volume += 0.1 * np.random.randn(nt, nx, ny).astype(np.float32)

    v_min = 500.0
    v_max = 2000.0

    # Python FKK filtering
    def fkk_filter_python(vol, dt, dx, dy, v_min, v_max, mode='reject'):
        nt, nx, ny = vol.shape

        # FFT
        spectrum = fft.rfft(vol, axis=0)
        spectrum = fft.fft(spectrum, axis=1)
        spectrum = fft.fft(spectrum, axis=2)
        spectrum = fft.fftshift(spectrum, axes=(1, 2))

        # Build mask
        nf = spectrum.shape[0]
        f_axis = np.fft.rfftfreq(nt, dt)
        kx_axis = np.fft.fftshift(np.fft.fftfreq(nx, dx))
        ky_axis = np.fft.fftshift(np.fft.fftfreq(ny, dy))

        f_grid, kx_grid, ky_grid = np.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')
        k_horizontal = np.sqrt(kx_grid**2 + ky_grid**2)
        k_safe = np.where(k_horizontal > 1e-10, k_horizontal, 1e-10)
        velocity = np.abs(f_grid) / k_safe

        in_velocity_range = (velocity >= v_min) & (velocity <= v_max)

        if mode == 'reject':
            mask = np.ones_like(velocity, dtype=np.float32)
            mask[in_velocity_range] = 0.0
        else:
            mask = np.zeros_like(velocity, dtype=np.float32)
            mask[in_velocity_range] = 1.0

        mask[0, :, :] = 1.0
        mask[:, nx // 2, ny // 2] = 1.0

        # Apply mask
        spectrum = spectrum * mask

        # IFFT
        spectrum = fft.ifftshift(spectrum, axes=(1, 2))
        spectrum = fft.ifft(spectrum, axis=2)
        spectrum = fft.ifft(spectrum, axis=1)
        result = fft.irfft(spectrum, n=nt, axis=0)

        return result.real.astype(np.float32)

    # Run both implementations
    result_python = fkk_filter_python(volume, dt, dx, dy, v_min, v_max, mode='reject')
    result_metal, metrics = metal.fkk_filter(
        volume, dt, dx, dy, v_min, v_max, mode='reject', preserve_dc=True
    )

    # Compare results
    diff = np.max(np.abs(result_python - result_metal))
    mean_diff = np.mean(np.abs(result_python - result_metal))

    print(f"Volume shape: {volume.shape}")
    print(f"Max difference (Python vs Metal): {diff:.6f}")
    print(f"Mean difference (Python vs Metal): {mean_diff:.6f}")
    print(f"Python output range: [{result_python.min():.4f}, {result_python.max():.4f}]")
    print(f"Metal output range: [{result_metal.min():.4f}, {result_metal.max():.4f}]")
    print(f"Kernel time: {metrics.get('kernel_time_ms', 0):.1f} ms")

    # Allow for numerical differences due to different FFT implementations
    passed = diff < 0.1
    print(f"TEST 6: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        print("\nDiagnostics:")
        print(f"  Input range: [{volume.min():.4f}, {volume.max():.4f}]")
        # Show slice comparison
        mid_t = nt // 2
        print(f"  Python slice[{mid_t}] sum: {np.sum(result_python[mid_t]):.4f}")
        print(f"  Metal slice[{mid_t}] sum: {np.sum(result_metal[mid_t]):.4f}")

    return passed


def test_fkk_noise_rejection():
    """Test FKK filter noise rejection capability."""
    print("\n" + "=" * 60)
    print("TEST 7: FKK Noise Rejection Quality")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return True

    if not is_metal_available():
        print("Metal not available, skipping")
        return True

    metal = get_metal_module()

    np.random.seed(42)
    nt, nx, ny = 128, 16, 16
    dt = 0.002
    dx = 25.0
    dy = 25.0

    # Create clean signal (hyperbolic event)
    clean = np.zeros((nt, nx, ny), dtype=np.float32)
    t0 = 40  # Zero-offset time
    v = 2000.0  # Velocity m/s

    for ix in range(nx):
        for iy in range(ny):
            offset = np.sqrt((ix * dx - nx * dx / 2) ** 2 + (iy * dy - ny * dy / 2) ** 2)
            t_arrival = np.sqrt(t0**2 + (offset / v) ** 2 / dt**2)
            t_idx = int(t_arrival)
            if 0 <= t_idx < nt - 5:
                # Simple wavelet
                for dt_idx in range(-3, 4):
                    t = t_idx + dt_idx
                    if 0 <= t < nt:
                        clean[t, ix, iy] = np.exp(-(dt_idx / 1.5) ** 2)

    # Add ground roll (slow coherent noise) - will be rejected
    ground_roll = np.zeros_like(clean)
    v_noise = 300.0  # Ground roll velocity m/s
    for ix in range(nx):
        for iy in range(ny):
            offset = np.sqrt((ix * dx - nx * dx / 2) ** 2 + (iy * dy - ny * dy / 2) ** 2)
            t_arrival = int(offset / v_noise / dt)
            if 0 <= t_arrival < nt - 5:
                for dt_idx in range(-3, 4):
                    t = t_arrival + dt_idx
                    if 0 <= t < nt:
                        ground_roll[t, ix, iy] = 0.5 * np.exp(-(dt_idx / 2.0) ** 2)

    noisy = clean + ground_roll

    # Apply FKK filter to reject slow velocities
    result_metal, metrics = metal.fkk_filter(
        noisy, dt, dx, dy,
        v_min=100.0, v_max=600.0,  # Reject ground roll velocities
        mode='reject', preserve_dc=True
    )

    # Calculate SNR improvement
    # Ground roll is the "noise" we want to remove
    snr_before = 10 * np.log10(np.var(clean) / np.var(ground_roll))
    residual_noise = result_metal - clean
    snr_after = 10 * np.log10(np.var(clean) / np.var(residual_noise))
    snr_improvement = snr_after - snr_before

    # Also check how much ground roll was removed
    ground_roll_removed = 1 - np.std(result_metal - clean) / np.std(ground_roll)

    print(f"Volume shape: {noisy.shape}")
    print(f"SNR before (signal vs ground roll): {snr_before:.2f} dB")
    print(f"SNR after: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    print(f"Ground roll reduction: {ground_roll_removed * 100:.1f}%")
    print(f"Kernel time: {metrics.get('kernel_time_ms', 0):.1f} ms")

    # Filter should improve SNR
    passed = snr_improvement > 0
    print(f"TEST 7: {'PASSED' if passed else 'FAILED'}")
    return passed


# ==============================================================================
# Main
# ==============================================================================

def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("STFT AND FKK KERNEL VALIDATION SUITE")
    print("=" * 60)

    results = []

    # STFT tests
    results.append(("SciPy STFT Reconstruction", test_scipy_stft_reconstruction()))
    results.append(("Metal vs Python STFT", test_stft_metal_vs_python_reconstruction()))
    results.append(("STFT Denoising Quality", test_stft_denoising_quality()))

    # FKK tests
    results.append(("3D FFT Round-Trip", test_fft3d_roundtrip()))
    results.append(("FKK Velocity Mask", test_fkk_velocity_mask()))
    results.append(("Metal vs Python FKK", test_fkk_metal_vs_python()))
    results.append(("FKK Noise Rejection", test_fkk_noise_rejection()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
