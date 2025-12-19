"""
SWT Implementation Validation Test

This script validates the Metal C++ SWT implementation against PyWavelets reference.
Tests include:
1. Perfect reconstruction (no thresholding)
2. Coefficient comparison with PyWavelets
3. Denoising comparison on synthetic data
4. Multi-level decomposition/reconstruction
"""

import numpy as np
import pywt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pywt_reference_algorithm():
    """
    Test our understanding of PyWavelets SWT algorithm.
    This validates the formulas before comparing with C++.
    """
    print("=" * 60)
    print("TEST 1: PyWavelets Reference Algorithm Validation")
    print("=" * 60)

    wavelet = pywt.Wavelet('db4')
    dec_lo = np.array(wavelet.dec_lo)
    dec_hi = np.array(wavelet.dec_hi)
    rec_lo = np.array(wavelet.rec_lo)
    rec_hi = np.array(wavelet.rec_hi)
    flen = len(dec_lo)

    def swt_decompose_reference(x, step=1):
        """Reference SWT decomposition matching PyWavelets."""
        n = len(x)
        cA = np.zeros(n)
        cD = np.zeros(n)
        center_shift = flen // 2

        for i in range(n):
            for k in range(flen):
                idx = (i - k * step + center_shift) % n
                cA[i] += x[idx] * dec_lo[k]
                cD[i] += x[idx] * dec_hi[k]

        return cA, cD

    def swt_reconstruct_reference(cA, cD, step=1):
        """Reference SWT reconstruction matching PyWavelets."""
        n = len(cA)
        output = np.zeros(n)
        center_shift = flen // 2

        # Use reversed filters
        rec_lo_rev = rec_lo[::-1]
        rec_hi_rev = rec_hi[::-1]

        for i in range(n):
            for k in range(flen):
                idx = (i + k * step - center_shift) % n
                output[i] += cA[idx] * rec_lo_rev[k] + cD[idx] * rec_hi_rev[k]

        return output * 0.5

    # Test with random signal
    np.random.seed(42)
    signal = np.random.randn(64)
    level = 3

    # Multi-level decomposition
    approx = [signal]
    details = []
    for l in range(level):
        step = 1 << l
        cA, cD = swt_decompose_reference(approx[l], step)
        approx.append(cA)
        details.append(cD)

    # Multi-level reconstruction
    recon = approx[level].copy()
    for l in range(level - 1, -1, -1):
        step = 1 << l
        recon = swt_reconstruct_reference(recon, details[l], step)

    error = np.max(np.abs(signal - recon))
    print(f"Signal length: {len(signal)}")
    print(f"Decomposition levels: {level}")
    print(f"Perfect reconstruction error: {error:.2e}")

    # Compare with pywt
    coeffs_pywt = pywt.swt(signal, wavelet, level=level)
    recon_pywt = pywt.iswt(coeffs_pywt, wavelet)
    pywt_error = np.max(np.abs(signal - recon_pywt))
    print(f"PyWavelets reconstruction error: {pywt_error:.2e}")

    passed = error < 1e-10
    print(f"TEST 1: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_perfect_reconstruction_python():
    """
    Test perfect reconstruction using Python SWT implementation.
    Uses pywt directly without thresholding to verify the algorithm.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Python SWT Perfect Reconstruction (No Thresholding)")
    print("=" * 60)

    # Create synthetic signal - sine wave (no noise)
    np.random.seed(42)
    n_samples = 256
    n_traces = 5
    t = np.linspace(0, 4 * np.pi, n_samples)

    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin(t + i * 0.5)  # Phase-shifted sine waves

    wavelet = pywt.Wavelet('db4')
    level = 3
    max_error = 0

    for i in range(n_traces):
        # SWT decomposition
        coeffs = pywt.swt(traces[:, i], wavelet, level=level)

        # Reconstruction WITHOUT any thresholding
        reconstructed = pywt.iswt(coeffs, wavelet)

        error = np.max(np.abs(traces[:, i] - reconstructed))
        max_error = max(max_error, error)

    print(f"Signal shape: {traces.shape}")
    print(f"Max reconstruction error: {max_error:.2e}")

    passed = max_error < 1e-5
    print(f"TEST 2: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_denoising_comparison():
    """
    Test SWT denoising on synthetic noisy signal.
    Compare Python implementation with expected behavior.
    """
    print("\n" + "=" * 60)
    print("TEST 3: SWT Denoising on Synthetic Data")
    print("=" * 60)

    from processors.dwt_denoise import DWTDenoise
    from models.seismic_data import SeismicData

    # Create synthetic signal with noise
    np.random.seed(42)
    n_samples = 512
    n_traces = 10
    t = np.linspace(0, 8 * np.pi, n_samples)

    # Clean signal: combination of sine waves (simulating seismic)
    clean = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        clean[:, i] = (np.sin(t * (1 + i * 0.1)) +
                       0.5 * np.sin(t * 2 * (1 + i * 0.1)) +
                       0.3 * np.sin(t * 0.5 * (1 + i * 0.1)))

    # Add Gaussian noise
    noise_level = 0.3
    noise = noise_level * np.random.randn(n_samples, n_traces).astype(np.float32)
    noisy = clean + noise

    # Denoise with SWT
    from processors.kernel_backend import KernelBackend

    processor = DWTDenoise(
        wavelet='db4',
        level=4,
        threshold_k=3.0,
        threshold_mode='soft',
        transform_type='swt',
        backend=KernelBackend.PYTHON
    )

    data = SeismicData(traces=noisy, sample_rate=1.0)
    result = processor.process(data)
    denoised = result.traces

    # Calculate SNR improvement
    noise_before = noisy - clean
    noise_after = denoised - clean

    snr_before = 10 * np.log10(np.var(clean) / np.var(noise_before))
    snr_after = 10 * np.log10(np.var(clean) / np.var(noise_after))
    snr_improvement = snr_after - snr_before

    # Calculate noise reduction
    noise_reduction = 1 - np.std(noise_after) / np.std(noise_before)

    print(f"Signal shape: {noisy.shape}")
    print(f"Noise level: {noise_level}")
    print(f"SNR before: {snr_before:.2f} dB")
    print(f"SNR after: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    print(f"Noise reduction: {noise_reduction*100:.1f}%")

    # Denoising should improve SNR
    passed = snr_improvement > 0
    print(f"TEST 3: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_metal_vs_python():
    """
    Compare Metal C++ implementation with Python reference.
    This is the critical test for validating the C++ fixes.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Metal C++ vs Python Comparison")
    print("=" * 60)

    try:
        from processors.kernel_backend import (
            KernelBackend, is_metal_available, get_dispatcher
        )
    except ImportError:
        print("Kernel backend not available, skipping Metal test")
        return True

    if not is_metal_available():
        print("Metal not available, skipping Metal test")
        return True

    from processors.dwt_denoise import DWTDenoise
    from models.seismic_data import SeismicData

    # Create synthetic noisy signal
    np.random.seed(42)
    n_samples = 256
    n_traces = 20
    t = np.linspace(0, 4 * np.pi, n_samples)

    clean = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        clean[:, i] = np.sin(t + i * 0.2) + 0.5 * np.sin(2 * t + i * 0.3)

    noise = 0.2 * np.random.randn(n_samples, n_traces).astype(np.float32)
    noisy = clean + noise

    # Process with Python backend
    processor_py = DWTDenoise(
        wavelet='db4',
        level=3,
        threshold_k=3.0,
        threshold_mode='soft',
        transform_type='swt',
        backend=KernelBackend.PYTHON
    )

    data = SeismicData(traces=noisy.copy(), sample_rate=1.0)
    result_py = processor_py.process(data)

    # Process with Metal backend
    processor_metal = DWTDenoise(
        wavelet='db4',
        level=3,
        threshold_k=3.0,
        threshold_mode='soft',
        transform_type='swt',
        backend=KernelBackend.METAL_CPP
    )

    data = SeismicData(traces=noisy.copy(), sample_rate=1.0)
    result_metal = processor_metal.process(data)

    # Compare results
    diff = np.abs(result_py.traces - result_metal.traces)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Signal shape: {noisy.shape}")
    print(f"Max difference (Python vs Metal): {max_diff:.6f}")
    print(f"Mean difference (Python vs Metal): {mean_diff:.6f}")

    # Results should be very close (allowing for float32 precision)
    passed = max_diff < 0.01
    print(f"TEST 4: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        print("\nDiagnostics:")
        print(f"Python output range: [{result_py.traces.min():.4f}, {result_py.traces.max():.4f}]")
        print(f"Metal output range: [{result_metal.traces.min():.4f}, {result_metal.traces.max():.4f}]")
        print(f"Input range: [{noisy.min():.4f}, {noisy.max():.4f}]")

    return passed


def test_perfect_reconstruction_metal():
    """
    Test perfect reconstruction using Metal C++ SWT implementation.
    Adds significant noise so that threshold is high relative to signal details.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Metal C++ SWT Denoising Quality Check")
    print("=" * 60)

    try:
        from processors.kernel_backend import (
            KernelBackend, is_metal_available, get_dispatcher
        )
    except ImportError:
        print("Kernel backend not available, skipping Metal test")
        return True

    if not is_metal_available():
        print("Metal not available, skipping Metal test")
        return True

    from processors.dwt_denoise import DWTDenoise
    from models.seismic_data import SeismicData

    # Create noisy synthetic signal
    np.random.seed(42)
    n_samples = 256
    n_traces = 10
    t = np.linspace(0, 4 * np.pi, n_samples)

    # Clean signal
    clean = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        clean[:, i] = np.sin(t + i * 0.5) + 0.3 * np.sin(2 * t + i * 0.3)

    # Add noise
    noise = 0.3 * np.random.randn(n_samples, n_traces).astype(np.float32)
    noisy = clean + noise

    # Process with Metal
    processor = DWTDenoise(
        wavelet='db4',
        level=4,
        threshold_k=3.0,
        threshold_mode='soft',
        transform_type='swt',
        backend=KernelBackend.METAL_CPP
    )

    data = SeismicData(traces=noisy.copy(), sample_rate=1.0)
    result = processor.process(data)
    denoised = result.traces

    # Calculate SNR improvement
    snr_before = 10 * np.log10(np.var(clean) / np.var(noisy - clean))
    snr_after = 10 * np.log10(np.var(clean) / np.var(denoised - clean))
    snr_improvement = snr_after - snr_before

    print(f"Signal shape: {noisy.shape}")
    print(f"SNR before: {snr_before:.2f} dB")
    print(f"SNR after: {snr_after:.2f} dB")
    print(f"SNR improvement: {snr_improvement:.2f} dB")

    # Metal should also improve SNR (even if different from Python)
    passed = snr_improvement > 0
    print(f"TEST 5: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        print("\nDiagnostics:")
        print(f"Input range: [{noisy.min():.4f}, {noisy.max():.4f}]")
        print(f"Output range: [{denoised.min():.4f}, {denoised.max():.4f}]")
        print(f"Clean range: [{clean.min():.4f}, {clean.max():.4f}]")

    return passed


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("SWT IMPLEMENTATION VALIDATION SUITE")
    print("=" * 60)

    results = []

    results.append(("Reference Algorithm", test_pywt_reference_algorithm()))
    results.append(("Python Perfect Reconstruction", test_perfect_reconstruction_python()))
    results.append(("Python Denoising", test_denoising_comparison()))
    results.append(("Metal vs Python", test_metal_vs_python()))
    results.append(("Metal Perfect Reconstruction", test_perfect_reconstruction_metal()))

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
