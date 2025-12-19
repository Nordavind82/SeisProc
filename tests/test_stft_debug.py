#!/usr/bin/env python3
"""
Debug test to trace STFT reconstruction issues in Metal C++ implementation.
"""

import numpy as np
from scipy import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def debug_metal_stft():
    """Debug Metal STFT step by step."""
    print("=" * 60)
    print("DEBUG: Metal STFT Implementation")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return

    if not is_metal_available():
        print("Metal not available")
        return

    metal = get_metal_module()

    # Simple test case: single trace sine wave
    np.random.seed(42)
    n_samples = 128
    n_traces = 1

    t = np.linspace(0, 2 * np.pi, n_samples)
    traces = np.sin(t).reshape(n_samples, 1).astype(np.float32)

    print(f"\nInput signal:")
    print(f"  Shape: {traces.shape}")
    print(f"  Range: [{traces.min():.4f}, {traces.max():.4f}]")
    print(f"  First 10 values: {traces[:10, 0]}")

    # Test 1: Very high threshold (should not modify)
    print("\n" + "-" * 60)
    print("Test with threshold_k=1000 (no thresholding):")
    print("-" * 60)

    result_high, _ = metal.stft_denoise(
        traces,
        nperseg=64,
        noverlap=48,
        aperture=1,  # Single trace
        threshold_k=1000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    print(f"  Output range: [{result_high.min():.4f}, {result_high.max():.4f}]")
    print(f"  First 10 values: {result_high[:10, 0]}")
    error_high = np.max(np.abs(traces - result_high))
    print(f"  Max error vs input: {error_high:.6f}")

    # Test 2: Different nperseg values
    print("\n" + "-" * 60)
    print("Test different nperseg values:")
    print("-" * 60)

    for nperseg in [32, 64, 128]:
        noverlap = nperseg * 3 // 4
        try:
            result, _ = metal.stft_denoise(
                traces,
                nperseg=nperseg,
                noverlap=noverlap,
                aperture=1,
                threshold_k=1000.0,
                fmin=0.0,
                fmax=0.0,
                sample_rate=500.0
            )
            error = np.max(np.abs(traces - result))
            print(f"  nperseg={nperseg}, noverlap={noverlap}: error={error:.6f}")
        except Exception as e:
            print(f"  nperseg={nperseg}: ERROR - {e}")

    # Test 3: Compare single-trace Python vs Metal STFT
    print("\n" + "-" * 60)
    print("Compare single-trace Python vs Metal:")
    print("-" * 60)

    nperseg = 64
    noverlap = 48

    # Python STFT -> ISTFT (no modification)
    _, _, stft_py = signal.stft(traces[:, 0], nperseg=nperseg, noverlap=noverlap)
    _, recon_py = signal.istft(stft_py, nperseg=nperseg, noverlap=noverlap)

    if len(recon_py) > n_samples:
        recon_py = recon_py[:n_samples]
    elif len(recon_py) < n_samples:
        recon_py = np.pad(recon_py, (0, n_samples - len(recon_py)))

    print(f"  Python STFT shape: {stft_py.shape}")
    print(f"  Python reconstruction error: {np.max(np.abs(traces[:, 0] - recon_py)):.6f}")

    # Metal STFT (single trace)
    result_metal, _ = metal.stft_denoise(
        traces,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=1,
        threshold_k=1000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    print(f"  Metal reconstruction error: {np.max(np.abs(traces - result_metal)):.6f}")
    print(f"  Metal vs Python recon diff: {np.max(np.abs(result_metal[:, 0] - recon_py)):.6f}")

    # Check if Metal is outputting something completely different
    correlation = np.corrcoef(traces[:, 0], result_metal[:, 0])[0, 1]
    print(f"  Correlation input vs Metal output: {correlation:.6f}")

    # Test 4: Check if output is same as input (no processing happened)
    print("\n" + "-" * 60)
    print("Check if Metal output equals input (no processing):")
    print("-" * 60)

    diff_input_output = np.max(np.abs(traces - result_metal))
    if diff_input_output < 1e-6:
        print("  WARNING: Output == Input, STFT processing may be bypassed!")
    else:
        print(f"  Output differs from input by {diff_input_output:.6f}")

    # Test 5: Check output pattern
    print("\n" + "-" * 60)
    print("Analyze output pattern:")
    print("-" * 60)

    print(f"  Output mean: {np.mean(result_metal):.6f}")
    print(f"  Output std: {np.std(result_metal):.6f}")
    print(f"  Input mean: {np.mean(traces):.6f}")
    print(f"  Input std: {np.std(traces):.6f}")

    # Check if output looks like scaled/shifted input
    if np.std(result_metal) > 1e-6:
        scale = np.std(traces) / np.std(result_metal)
        shift = np.mean(traces) - scale * np.mean(result_metal)
        rescaled = scale * result_metal + shift
        rescale_error = np.max(np.abs(traces - rescaled))
        print(f"  After rescaling: error = {rescale_error:.6f}")

    # Test 6: Check DC preservation
    print("\n" + "-" * 60)
    print("Check DC preservation:")
    print("-" * 60)

    # Add DC offset
    traces_dc = traces + 0.5
    result_dc, _ = metal.stft_denoise(
        traces_dc,
        nperseg=64,
        noverlap=48,
        aperture=1,
        threshold_k=1000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )
    print(f"  Input DC: mean={np.mean(traces_dc):.4f}")
    print(f"  Output DC: mean={np.mean(result_dc):.4f}")


def debug_vdsp_fft():
    """Debug vDSP FFT specifically by checking STFT window and normalization."""
    print("\n" + "=" * 60)
    print("DEBUG: Hann Window and FFT Normalization")
    print("=" * 60)

    # vDSP uses specific scaling conventions
    # Check what scipy uses

    nperseg = 64

    # Create Hann window (scipy default)
    window_scipy = signal.get_window('hann', nperseg)

    # Check C++ formula: 0.5 * (1 - cos(2*pi*i / (size-1)))
    window_cpp = np.array([
        0.5 * (1.0 - np.cos(2.0 * np.pi * i / (nperseg - 1)))
        for i in range(nperseg)
    ])

    print(f"Scipy Hann window: {window_scipy[:8]}")
    print(f"C++ formula window: {window_cpp[:8]}")
    print(f"Window difference: {np.max(np.abs(window_scipy - window_cpp)):.2e}")

    # Check COLA condition
    hop = nperseg - 48  # noverlap = 48
    cola_sum = np.zeros(nperseg)
    for i in range(5):
        start = i * hop
        cola_sum[:] += window_scipy ** 2  # Window squared sum for ISTFT
    print(f"COLA sum (should be constant): min={cola_sum.min():.4f}, max={cola_sum.max():.4f}")


if __name__ == "__main__":
    debug_metal_stft()
    debug_vdsp_fft()
