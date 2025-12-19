#!/usr/bin/env python3
"""
Debug test specifically for phase-shifted traces - the case that fails.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_phase_shifted():
    """Test the exact case that fails: phase-shifted same-frequency traces."""
    print("=" * 60)
    print("DEBUG: Phase-Shifted Traces (Original Failure Case)")
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

    # Create phase-shifted traces (EXACT same as original failing test)
    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin(t + i * 0.2).astype(np.float32)  # Same freq, different phase

    print(f"\nInput (phase-shifted same-frequency sines):")
    print(f"  Shape: {traces.shape}")
    print(f"  Range: [{traces.min():.4f}, {traces.max():.4f}]")
    print(f"  Trace 0 first 5: {traces[:5, 0]}")
    print(f"  Trace 5 first 5: {traces[:5, 5]}")

    # Test 1: Very high threshold (original failing case)
    print("\n" + "-" * 60)
    print("Test 1: threshold_k=1000 (should be no modification)")
    print("-" * 60)

    result, _ = metal.stft_denoise(
        traces,
        nperseg=64,
        noverlap=48,
        aperture=7,
        threshold_k=1000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    error = np.max(np.abs(traces - result))
    print(f"  Max error: {error:.6f}")
    print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Check individual trace errors and correlations
    print(f"\n  Per-trace analysis:")
    for i in [0, 5, 10]:
        trace_error = np.max(np.abs(traces[:, i] - result[:, i]))
        trace_corr = np.corrcoef(traces[:, i], result[:, i])[0, 1]
        print(f"    Trace {i}: error={trace_error:.6f}, correlation={trace_corr:.6f}")

    # Test 2: Same but with aperture=1 (no spatial processing)
    print("\n" + "-" * 60)
    print("Test 2: aperture=1, threshold_k=1000")
    print("-" * 60)

    result_ap1, _ = metal.stft_denoise(
        traces,
        nperseg=64,
        noverlap=48,
        aperture=1,  # No spatial processing
        threshold_k=1000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    error_ap1 = np.max(np.abs(traces - result_ap1))
    print(f"  Max error: {error_ap1:.6f}")

    for i in [0, 5, 10]:
        trace_error = np.max(np.abs(traces[:, i] - result_ap1[:, i]))
        trace_corr = np.corrcoef(traces[:, i], result_ap1[:, i])[0, 1]
        print(f"    Trace {i}: error={trace_error:.6f}, correlation={trace_corr:.6f}")

    # Test 3: Single trace at a time
    print("\n" + "-" * 60)
    print("Test 3: Process each trace individually")
    print("-" * 60)

    result_single = np.zeros_like(traces)
    for i in range(n_traces):
        single_trace = traces[:, i:i+1].copy()
        result_i, _ = metal.stft_denoise(
            single_trace,
            nperseg=64,
            noverlap=48,
            aperture=1,
            threshold_k=1000.0,
            fmin=0.0,
            fmax=0.0,
            sample_rate=500.0
        )
        result_single[:, i] = result_i[:, 0]

    error_single = np.max(np.abs(traces - result_single))
    print(f"  Max error: {error_single:.6f}")

    for i in [0, 5, 10]:
        trace_error = np.max(np.abs(traces[:, i] - result_single[:, i]))
        trace_corr = np.corrcoef(traces[:, i], result_single[:, i])[0, 1]
        print(f"    Trace {i}: error={trace_error:.6f}, correlation={trace_corr:.6f}")

    # Compare batch ap=1 vs single
    print(f"\n  Batch(ap=1) vs Single max diff: {np.max(np.abs(result_ap1 - result_single)):.6f}")

    # Test 4: Check if output traces match WRONG input traces
    print("\n" + "-" * 60)
    print("Test 4: Check for trace misalignment")
    print("-" * 60)

    print(f"  Correlation of output traces with input traces:")
    for out_idx in [0, 5, 10]:
        best_corr = -1
        best_in = -1
        for in_idx in range(n_traces):
            corr = np.corrcoef(traces[:, in_idx], result[:, out_idx])[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_in = in_idx
        match = "CORRECT" if best_in == out_idx else f"WRONG (expected {out_idx})"
        print(f"    Output {out_idx} best matches Input {best_in} (corr={best_corr:.4f}) - {match}")

    # Test 5: Examine the actual values
    print("\n" + "-" * 60)
    print("Test 5: Compare actual values")
    print("-" * 60)

    center_trace = 5
    print(f"\n  Input trace {center_trace} (first 10):")
    print(f"    {traces[:10, center_trace]}")
    print(f"\n  Output trace {center_trace} (first 10):")
    print(f"    {result[:10, center_trace]}")
    print(f"\n  Difference (first 10):")
    print(f"    {result[:10, center_trace] - traces[:10, center_trace]}")


def test_threshold_effect_detailed():
    """Test how threshold affects phase-shifted traces."""
    print("\n" + "=" * 60)
    print("DEBUG: Threshold Effect on Phase-Shifted Traces")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        return

    if not is_metal_available():
        return

    metal = get_metal_module()

    np.random.seed(42)
    n_samples = 256
    n_traces = 11

    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin(t + i * 0.2).astype(np.float32)

    print(f"\nVarying threshold_k with aperture=7:")
    for k in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 10000.0]:
        result, _ = metal.stft_denoise(
            traces, nperseg=64, noverlap=48, aperture=7,
            threshold_k=k, fmin=0.0, fmax=0.0, sample_rate=500.0
        )
        error = np.max(np.abs(traces - result))
        out_range = f"[{result.min():.3f}, {result.max():.3f}]"
        print(f"  k={k:8.1f}: error={error:.6f}, range={out_range}")


if __name__ == "__main__":
    test_phase_shifted()
    test_threshold_effect_detailed()
