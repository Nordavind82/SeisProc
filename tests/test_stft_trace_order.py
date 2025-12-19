#!/usr/bin/env python3
"""
Debug test to check if STFT is mixing up trace ordering.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_trace_ordering():
    """Check if traces are being mixed up."""
    print("=" * 60)
    print("DEBUG: Check Trace Ordering in STFT")
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
    n_traces = 5

    # Create traces with UNIQUE signatures
    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)

    # Each trace has a unique frequency
    for i in range(n_traces):
        freq = 1 + i  # Frequencies 1, 2, 3, 4, 5
        traces[:, i] = np.sin(freq * t).astype(np.float32)

    print(f"\nInput traces (each has unique frequency):")
    print(f"  Shape: {traces.shape}")
    for i in range(n_traces):
        print(f"  Trace {i}: freq={1+i}, first 5: {traces[:5, i]}")

    # Process with high threshold (no modification expected)
    result, _ = metal.stft_denoise(
        traces,
        nperseg=64,
        noverlap=48,
        aperture=1,  # No spatial processing
        threshold_k=10000.0,  # Very high - no thresholding
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    print(f"\nOutput traces:")
    for i in range(n_traces):
        print(f"  Trace {i}: first 5: {result[:5, i]}")

    print(f"\nCorrelation matrix (should be diagonal):")
    corr_matrix = np.zeros((n_traces, n_traces))
    for i in range(n_traces):
        for j in range(n_traces):
            corr_matrix[i, j] = np.corrcoef(traces[:, i], result[:, j])[0, 1]

    print("  Input\\Output", end="")
    for j in range(n_traces):
        print(f"  T{j:d}", end="")
    print()

    for i in range(n_traces):
        print(f"  T{i:d}         ", end="")
        for j in range(n_traces):
            print(f" {corr_matrix[i, j]:5.2f}", end="")
        print()

    # Check if each output trace matches its corresponding input
    print(f"\nDiagonal correlations (should all be ~1.0):")
    for i in range(n_traces):
        corr = np.corrcoef(traces[:, i], result[:, i])[0, 1]
        match = "OK" if corr > 0.99 else "MISMATCH"
        print(f"  Trace {i}: correlation = {corr:.6f} {match}")

    # Check for trace swapping pattern
    print(f"\nFinding best match for each output trace:")
    for j in range(n_traces):
        best_i = np.argmax(corr_matrix[:, j])
        best_corr = corr_matrix[best_i, j]
        if best_i == j:
            print(f"  Output {j} matches Input {best_i} (corr={best_corr:.4f}) - CORRECT")
        else:
            print(f"  Output {j} matches Input {best_i} (corr={best_corr:.4f}) - SWAPPED!")


def test_memory_layout():
    """Verify NumPy memory layout matches C++ expectations."""
    print("\n" + "=" * 60)
    print("DEBUG: Check NumPy Memory Layout")
    print("=" * 60)

    n_samples = 4
    n_traces = 3

    # Create test array
    arr = np.zeros((n_samples, n_traces), dtype=np.float32)
    for s in range(n_samples):
        for t in range(n_traces):
            arr[s, t] = s * 10 + t  # Value = sample*10 + trace

    print(f"\nArray shape: {arr.shape}")
    print(f"Array (logical view):")
    print(arr)

    print(f"\nArray memory order: {'C (row-major)' if arr.flags['C_CONTIGUOUS'] else 'F (column-major)'}")

    flat = arr.flatten()
    print(f"\nFlattened (C-order): {flat}")

    # Verify C-order indexing: arr[s, t] = flat[s * n_traces + t]
    print(f"\nVerify indexing: arr[s, t] = flat[s * n_traces + t]")
    for s in range(n_samples):
        for t in range(n_traces):
            expected = flat[s * n_traces + t]
            actual = arr[s, t]
            match = "OK" if expected == actual else "MISMATCH"
            print(f"  arr[{s},{t}] = {actual}, flat[{s}*{n_traces}+{t}] = flat[{s*n_traces+t}] = {expected} {match}")


def test_single_vs_batch():
    """Compare single trace processing vs batch processing."""
    print("\n" + "=" * 60)
    print("DEBUG: Single vs Batch Processing")
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
    n_traces = 5

    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin((1 + i) * t).astype(np.float32)

    # Process all traces together
    result_batch, _ = metal.stft_denoise(
        traces,
        nperseg=64,
        noverlap=48,
        aperture=1,
        threshold_k=10000.0,
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    # Process each trace individually
    result_single = np.zeros_like(traces)
    for i in range(n_traces):
        single_trace = traces[:, i:i+1].copy()
        result_i, _ = metal.stft_denoise(
            single_trace,
            nperseg=64,
            noverlap=48,
            aperture=1,
            threshold_k=10000.0,
            fmin=0.0,
            fmax=0.0,
            sample_rate=500.0
        )
        result_single[:, i] = result_i[:, 0]

    print(f"\nCompare batch vs single trace processing:")
    for i in range(n_traces):
        batch_error = np.max(np.abs(traces[:, i] - result_batch[:, i]))
        single_error = np.max(np.abs(traces[:, i] - result_single[:, i]))
        batch_vs_single = np.max(np.abs(result_batch[:, i] - result_single[:, i]))
        print(f"  Trace {i}: batch_err={batch_error:.6f}, single_err={single_error:.6f}, diff={batch_vs_single:.6f}")


if __name__ == "__main__":
    test_trace_ordering()
    test_memory_layout()
    test_single_vs_batch()
