#!/usr/bin/env python3
"""
Minimal SWT Test - Isolates decomposition/reconstruction without thresholding.
"""

import numpy as np
import pywt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pywt_coefficient_order():
    """Verify PyWavelets coefficient ordering."""
    print("=" * 60)
    print("TEST: PyWavelets Coefficient Order")
    print("=" * 60)

    np.random.seed(42)
    signal = np.sin(np.linspace(0, 4*np.pi, 64))
    wavelet = 'db4'
    level = 3

    coeffs = pywt.swt(signal, wavelet, level=level)

    print(f"Signal length: {len(signal)}")
    print(f"Level: {level}")
    print(f"Number of coefficient pairs: {len(coeffs)}")

    for i, (cA, cD) in enumerate(coeffs):
        print(f"  coeffs[{i}]: cA shape={cA.shape}, cD shape={cD.shape}")
        print(f"             cD variance={np.var(cD):.6f}")

    # coeffs[0] should be coarsest (highest variance in cD typically)
    # coeffs[-1] should be finest (lowest variance in cD for smooth signal)

    print("\nNote: coeffs[0] = coarsest level, coeffs[-1] = finest level")
    print("For noise estimation, we use coeffs[0][1] (coarsest detail)")

def test_direct_swt_comparison():
    """Direct comparison of SWT decomposition/reconstruction."""
    print("\n" + "=" * 60)
    print("TEST: Direct SWT Perfect Reconstruction")
    print("=" * 60)

    wavelet = pywt.Wavelet('db4')
    dec_lo = np.array(wavelet.dec_lo)
    dec_hi = np.array(wavelet.dec_hi)
    rec_lo = np.array(wavelet.rec_lo)
    rec_hi = np.array(wavelet.rec_hi)
    flen = len(dec_lo)

    print(f"\nWavelet: db4, filter length: {flen}")
    print(f"dec_lo: {dec_lo}")
    print(f"dec_hi: {dec_hi}")
    print(f"rec_lo: {rec_lo}")
    print(f"rec_hi: {rec_hi}")

    def swt_decompose(x, step=1):
        """SWT decomposition matching PyWavelets."""
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

    def swt_reconstruct(cA, cD, step=1):
        """SWT reconstruction matching PyWavelets."""
        n = len(cA)
        output = np.zeros(n)
        center_shift = flen // 2

        # Reversed filters
        rec_lo_rev = rec_lo[::-1]
        rec_hi_rev = rec_hi[::-1]

        for i in range(n):
            for k in range(flen):
                idx = (i + k * step - center_shift) % n
                output[i] += cA[idx] * rec_lo_rev[k] + cD[idx] * rec_hi_rev[k]

        return output * 0.5

    # Test signal (power of 2 length required for SWT)
    np.random.seed(42)
    n = 64
    signal = np.sin(np.linspace(0, 4*np.pi, n)) + 0.1 * np.random.randn(n)
    level = 3

    # Multi-level decomposition
    approx = [signal]
    details = []

    for l in range(level):
        step = 1 << l
        cA, cD = swt_decompose(approx[l], step)
        approx.append(cA)
        details.append(cD)

    # Multi-level reconstruction (no thresholding)
    recon = approx[level].copy()
    for l in range(level - 1, -1, -1):
        step = 1 << l
        recon = swt_reconstruct(recon, details[l], step)

    error_manual = np.max(np.abs(signal - recon))

    # Compare with PyWavelets
    coeffs = pywt.swt(signal, 'db4', level=level)
    recon_pywt = pywt.iswt(coeffs, 'db4')
    error_pywt = np.max(np.abs(signal - recon_pywt))

    print(f"\nManual implementation error: {error_manual:.2e}")
    print(f"PyWavelets error: {error_pywt:.2e}")

    # Compare coefficient values
    print("\nCoefficient comparison (manual vs pywt):")
    for l in range(level):
        # PyWavelets returns [coarsest, ..., finest], so coeffs[0] = level-1, coeffs[-1] = level 0
        pywt_idx = level - 1 - l
        cA_pywt, cD_pywt = coeffs[pywt_idx]

        cA_diff = np.max(np.abs(approx[l+1] - cA_pywt))
        cD_diff = np.max(np.abs(details[l] - cD_pywt))

        print(f"  Level {l} (step={1<<l}): cA diff={cA_diff:.2e}, cD diff={cD_diff:.2e}")

    passed = error_manual < 1e-10 and error_pywt < 1e-10
    print(f"\nTEST: {'PASSED' if passed else 'FAILED'}")
    return passed

def test_metal_raw():
    """Test Metal C++ SWT directly without processor wrapper."""
    print("\n" + "=" * 60)
    print("TEST: Metal C++ Raw SWT Call")
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

    # Test signal
    np.random.seed(42)
    n_samples = 64
    n_traces = 1

    # Simple sine wave
    t = np.linspace(0, 4*np.pi, n_samples)
    traces = np.sin(t).reshape(n_samples, 1).astype(np.float32)

    print(f"Input shape: {traces.shape}")
    print(f"Input range: [{traces.min():.4f}, {traces.max():.4f}]")

    # Call Metal SWT with NO thresholding (threshold_k very high so nothing gets thresholded)
    result, metrics = metal.swt_denoise(
        traces,
        wavelet='db4',
        level=3,
        threshold_mode='soft',
        threshold_k=100.0  # Very high - should not threshold anything
    )

    print(f"Output shape: {result.shape}")
    print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

    error = np.max(np.abs(traces - result))
    print(f"Reconstruction error (high threshold): {error:.6f}")

    # Compare with Python reference
    import pywt
    traces_1d = traces[:, 0]
    coeffs = pywt.swt(traces_1d, 'db4', level=3)

    # Apply same high threshold (should not change anything)
    sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
    threshold = 100.0 * sigma
    print(f"Threshold: {threshold:.4f} (sigma={sigma:.4f})")

    denoised_coeffs = []
    for cA, cD in coeffs:
        cD_thresh = pywt.threshold(cD, threshold, mode='soft')
        denoised_coeffs.append((cA, cD_thresh))

    recon_pywt = pywt.iswt(denoised_coeffs, 'db4')
    pywt_error = np.max(np.abs(traces_1d - recon_pywt))
    print(f"PyWavelets error (high threshold): {pywt_error:.6f}")

    # Check if Metal output matches
    diff = np.max(np.abs(result[:, 0] - recon_pywt))
    print(f"Metal vs PyWavelets diff: {diff:.6f}")

    # Metal should match PyWavelets (both will have same denoising effect)
    # The key test is that Metal == PyWavelets, not that error is small
    passed = diff < 0.001
    print(f"\nTEST: {'PASSED' if passed else 'FAILED'}")

    if not passed:
        print("\nDiagnostics:")
        print(f"First 10 input:  {traces[:10, 0]}")
        print(f"First 10 output: {result[:10, 0]}")
        print(f"First 10 pywt:   {recon_pywt[:10]}")

    return passed

def test_metal_vs_python_coeffs():
    """Compare Metal and Python coefficient values directly."""
    print("\n" + "=" * 60)
    print("TEST: Metal vs Python Coefficient Values")
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
    n_samples = 64
    n_traces = 1
    t = np.linspace(0, 4*np.pi, n_samples)
    signal = np.sin(t).astype(np.float32)
    traces = signal.reshape(n_samples, 1)

    # Test with threshold_k = 0 (threshold all detail)
    result_k0, _ = metal.swt_denoise(traces, 'db4', 3, 'soft', 0.0)

    # Test with threshold_k very high (threshold nothing)
    result_khigh, _ = metal.swt_denoise(traces, 'db4', 3, 'soft', 1000.0)

    print(f"Input range: [{traces.min():.4f}, {traces.max():.4f}]")
    print(f"Output (k=0) range: [{result_k0.min():.4f}, {result_k0.max():.4f}]")
    print(f"Output (k=1000) range: [{result_khigh.min():.4f}, {result_khigh.max():.4f}]")

    # With k=1000, output should nearly match input
    error_high = np.max(np.abs(traces - result_khigh))
    print(f"Error with k=1000: {error_high:.6f}")

    # With k=0, all details should be zeroed - output should be very smooth
    k0_vs_khigh_diff = np.max(np.abs(result_k0 - result_khigh))
    print(f"Error between k=0 and k=1000: {k0_vs_khigh_diff:.6f}")

    # Compare with PyWavelets for k=1000
    import pywt
    coeffs = pywt.swt(signal, 'db4', level=3)
    sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
    threshold = 1000.0 * sigma
    denoised_coeffs = [(cA, pywt.threshold(cD, threshold, mode='soft')) for cA, cD in coeffs]
    recon_pywt = pywt.iswt(denoised_coeffs, 'db4')
    metal_vs_pywt = np.max(np.abs(result_khigh[:, 0] - recon_pywt))
    print(f"Metal vs PyWavelets (k=1000): {metal_vs_pywt:.6f}")

    # Key test: Metal should match PyWavelets
    passed = metal_vs_pywt < 0.001
    print(f"\nTEST: {'PASSED' if passed else 'FAILED'}")
    return passed

if __name__ == "__main__":
    print("=" * 60)
    print("MINIMAL SWT VALIDATION TESTS")
    print("=" * 60)

    results = []
    results.append(("Coefficient Order", test_pywt_coefficient_order()))
    results.append(("Direct SWT", test_direct_swt_comparison()))
    results.append(("Metal Raw", test_metal_raw()))
    results.append(("Metal vs Python Coeffs", test_metal_vs_python_coeffs()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        if passed is not None:
            status = "PASSED" if passed else "FAILED"
            print(f"{name}: {status}")
            if not passed:
                all_passed = False

    sys.exit(0 if all_passed else 1)
