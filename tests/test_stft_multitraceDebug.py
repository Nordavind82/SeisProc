#!/usr/bin/env python3
"""
Debug test for multi-trace STFT denoising.
"""

import numpy as np
from scipy import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_multi_trace_stft():
    """Debug multi-trace STFT processing."""
    print("=" * 60)
    print("DEBUG: Multi-Trace STFT Processing")
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

    np.random.seed(42)
    n_samples = 256
    n_traces = 11

    # Create clean identical sine waves (all traces same)
    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin(t).astype(np.float32)

    print(f"\nInput (all traces identical):")
    print(f"  Shape: {traces.shape}")
    print(f"  Range: [{traces.min():.4f}, {traces.max():.4f}]")

    # Test with different aperture values
    print("\n" + "-" * 60)
    print("Test with different apertures (threshold_k=1000):")
    print("-" * 60)

    nperseg = 64
    noverlap = 48
    threshold_k = 1000.0

    for aperture in [1, 3, 7, 11]:
        result, metrics = metal.stft_denoise(
            traces,
            nperseg=nperseg,
            noverlap=noverlap,
            aperture=aperture,
            threshold_k=threshold_k,
            fmin=0.0,
            fmax=0.0,
            sample_rate=500.0
        )
        error = np.max(np.abs(traces - result))
        center_error = np.max(np.abs(traces[:, 5] - result[:, 5]))
        print(f"  aperture={aperture:2d}: max_error={error:.6f}, center_trace_error={center_error:.6f}")

    # Test with non-identical traces (phase shifted)
    print("\n" + "-" * 60)
    print("Test with phase-shifted traces:")
    print("-" * 60)

    for i in range(n_traces):
        traces[:, i] = np.sin(t + i * 0.2).astype(np.float32)

    print(f"  Input traces have different phases")

    for aperture in [1, 7, 11]:
        result, _ = metal.stft_denoise(
            traces,
            nperseg=nperseg,
            noverlap=noverlap,
            aperture=aperture,
            threshold_k=threshold_k,
            fmin=0.0,
            fmax=0.0,
            sample_rate=500.0
        )
        error = np.max(np.abs(traces - result))
        print(f"  aperture={aperture:2d}: max_error={error:.6f}")

    # Test thresholding effect
    print("\n" + "-" * 60)
    print("Test threshold effect with aperture=7:")
    print("-" * 60)

    for threshold_k in [0.0, 0.1, 1.0, 3.0, 10.0, 100.0, 1000.0]:
        result, _ = metal.stft_denoise(
            traces,
            nperseg=nperseg,
            noverlap=noverlap,
            aperture=7,
            threshold_k=threshold_k,
            fmin=0.0,
            fmax=0.0,
            sample_rate=500.0
        )
        error = np.max(np.abs(traces - result))
        output_range = f"[{result.min():.4f}, {result.max():.4f}]"
        print(f"  threshold_k={threshold_k:6.1f}: max_error={error:.6f}, output_range={output_range}")

    # Compare Python reference (using scipy stft for each trace)
    print("\n" + "-" * 60)
    print("Compare with Python trace-by-trace STFT:")
    print("-" * 60)

    result_python = np.zeros_like(traces)
    for i in range(n_traces):
        _, _, stft_data = signal.stft(traces[:, i], nperseg=nperseg, noverlap=noverlap)
        _, reconstructed = signal.istft(stft_data, nperseg=nperseg, noverlap=noverlap)
        if len(reconstructed) > n_samples:
            reconstructed = reconstructed[:n_samples]
        elif len(reconstructed) < n_samples:
            reconstructed = np.pad(reconstructed, (0, n_samples - len(reconstructed)))
        result_python[:, i] = reconstructed

    result_metal, _ = metal.stft_denoise(
        traces,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=1,  # No spatial processing
        threshold_k=1000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    python_recon_error = np.max(np.abs(traces - result_python))
    metal_recon_error = np.max(np.abs(traces - result_metal))
    metal_vs_python = np.max(np.abs(result_metal - result_python))

    print(f"  Python reconstruction error: {python_recon_error:.6f}")
    print(f"  Metal reconstruction error (aperture=1): {metal_recon_error:.6f}")
    print(f"  Metal vs Python difference: {metal_vs_python:.6f}")

    # Examine what the thresholding actually does
    print("\n" + "-" * 60)
    print("Examine thresholding behavior:")
    print("-" * 60)

    # Create signal with one outlier trace
    traces_outlier = traces.copy()
    traces_outlier[:, 5] = traces[:, 5] * 3.0  # Make center trace 3x amplitude

    result_no_thresh, _ = metal.stft_denoise(
        traces_outlier,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=7,
        threshold_k=1000.0,  # No thresholding
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    result_thresh, _ = metal.stft_denoise(
        traces_outlier,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=7,
        threshold_k=3.0,  # Normal thresholding
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    print(f"  Outlier trace (3x amplitude) input: [{traces_outlier[:, 5].min():.4f}, {traces_outlier[:, 5].max():.4f}]")
    print(f"  No threshold output: [{result_no_thresh[:, 5].min():.4f}, {result_no_thresh[:, 5].max():.4f}]")
    print(f"  With threshold output: [{result_thresh[:, 5].min():.4f}, {result_thresh[:, 5].max():.4f}]")

    # Check if outlier was reduced
    outlier_reduction = (np.max(np.abs(traces_outlier[:, 5])) -
                        np.max(np.abs(result_thresh[:, 5]))) / np.max(np.abs(traces_outlier[:, 5]))
    print(f"  Outlier reduction: {outlier_reduction*100:.1f}%")


if __name__ == "__main__":
    test_multi_trace_stft()
