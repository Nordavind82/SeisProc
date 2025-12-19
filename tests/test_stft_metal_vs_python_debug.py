#!/usr/bin/env python3
"""
Systematic STFT Metal vs Python Comparison Test

This test compares each step of the STFT denoising pipeline between:
- Metal C++ implementation (vDSP-based)
- Python scipy implementation

Goal: Identify bugs in C++ implementation by finding where results diverge.
"""

import numpy as np
from scipy import signal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_hann_window_python(size):
    """Python Hann window matching scipy's default."""
    return signal.windows.hann(size, sym=False)


def create_hann_window_cpp_style(size):
    """C++ style Hann window: 0.5 * (1 - cos(2*pi*i/(N-1)))"""
    return np.array([0.5 * (1 - np.cos(2 * np.pi * i / (size - 1)))
                     for i in range(size)], dtype=np.float32)


def test_window_functions():
    """Compare window functions between implementations."""
    print("=" * 60)
    print("TEST: Window Function Comparison")
    print("=" * 60)

    for nperseg in [32, 64, 128]:
        # scipy default window (sym=True for istft)
        scipy_win = signal.windows.hann(nperseg, sym=True)

        # C++ style window
        cpp_win = create_hann_window_cpp_style(nperseg)

        # scipy with sym=False
        scipy_win_asym = signal.windows.hann(nperseg, sym=False)

        diff_sym = np.max(np.abs(scipy_win - cpp_win))
        diff_asym = np.max(np.abs(scipy_win_asym - cpp_win))

        print(f"\nnperseg={nperseg}:")
        print(f"  scipy(sym=True) vs C++:  max diff = {diff_sym:.2e}")
        print(f"  scipy(sym=False) vs C++: max diff = {diff_asym:.2e}")
        print(f"  scipy(sym=True)[0]={scipy_win[0]:.6f}, [-1]={scipy_win[-1]:.6f}")
        print(f"  scipy(sym=False)[0]={scipy_win_asym[0]:.6f}, [-1]={scipy_win_asym[-1]:.6f}")
        print(f"  C++ style [0]={cpp_win[0]:.6f}, [-1]={cpp_win[-1]:.6f}")


def test_stft_forward_comparison():
    """Compare STFT forward transform outputs."""
    print("\n" + "=" * 60)
    print("TEST: STFT Forward Transform Comparison")
    print("=" * 60)

    # Create simple test signal
    np.random.seed(42)
    n_samples = 256
    t = np.linspace(0, 4 * np.pi, n_samples)
    trace = (np.sin(t) + 0.3 * np.sin(3 * t)).astype(np.float32)

    nperseg = 64
    noverlap = 48
    hop = nperseg - noverlap

    # scipy STFT
    freqs, times, stft_scipy = signal.stft(
        trace, nperseg=nperseg, noverlap=noverlap,
        boundary='zeros', padded=True
    )

    print(f"\nSignal length: {n_samples}")
    print(f"nperseg: {nperseg}, noverlap: {noverlap}, hop: {hop}")
    print(f"scipy STFT shape: {stft_scipy.shape}")
    print(f"scipy n_freqs: {stft_scipy.shape[0]}, n_times: {stft_scipy.shape[1]}")

    # Calculate expected dimensions for C++ implementation
    pad_length = nperseg // 2
    padded_length = n_samples + 2 * pad_length
    n_times_cpp = (padded_length - nperseg) // hop + 1
    n_freqs_cpp = nperseg // 2 + 1

    print(f"\nExpected C++ dimensions:")
    print(f"  pad_length: {pad_length}")
    print(f"  padded_length: {padded_length}")
    print(f"  n_times: {n_times_cpp}")
    print(f"  n_freqs: {n_freqs_cpp}")

    # Check if dimensions match
    dims_match = (stft_scipy.shape[0] == n_freqs_cpp and
                  stft_scipy.shape[1] == n_times_cpp)
    print(f"\nDimensions match: {dims_match}")

    if not dims_match:
        print(f"  scipy: ({stft_scipy.shape[0]}, {stft_scipy.shape[1]})")
        print(f"  expected C++: ({n_freqs_cpp}, {n_times_cpp})")


def test_istft_reconstruction():
    """Compare ISTFT reconstruction."""
    print("\n" + "=" * 60)
    print("TEST: ISTFT Reconstruction Comparison")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 256
    t = np.linspace(0, 4 * np.pi, n_samples)
    trace = (np.sin(t) + 0.3 * np.sin(3 * t)).astype(np.float32)

    nperseg = 64
    noverlap = 48

    # scipy STFT -> ISTFT
    _, _, stft_data = signal.stft(trace, nperseg=nperseg, noverlap=noverlap)
    _, reconstructed = signal.istft(stft_data, nperseg=nperseg, noverlap=noverlap)

    if len(reconstructed) > n_samples:
        reconstructed = reconstructed[:n_samples]
    elif len(reconstructed) < n_samples:
        reconstructed = np.pad(reconstructed, (0, n_samples - len(reconstructed)))

    error = np.max(np.abs(trace - reconstructed))

    print(f"Original trace length: {n_samples}")
    print(f"Reconstructed length: {len(reconstructed)}")
    print(f"Max reconstruction error: {error:.2e}")
    print(f"Perfect reconstruction: {error < 1e-10}")

    return error < 1e-10


def test_metal_reconstruction():
    """Test Metal STFT perfect reconstruction (no thresholding)."""
    print("\n" + "=" * 60)
    print("TEST: Metal STFT Perfect Reconstruction")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return False

    if not is_metal_available():
        print("Metal not available")
        return False

    metal = get_metal_module()

    np.random.seed(42)
    n_samples = 256
    n_traces = 11

    # Create clean signal
    t = np.linspace(0, 4 * np.pi, n_samples)
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        traces[:, i] = np.sin(t + i * 0.2).astype(np.float32)

    nperseg = 64
    noverlap = 48
    aperture = 7
    threshold_k = 10000.0  # Very high - no thresholding
    sample_rate = 500.0

    result, metrics = metal.stft_denoise(
        traces,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=aperture,
        threshold_k=threshold_k,
        fmin=0.0,  # No frequency filtering
        fmax=0.0,
        sample_rate=sample_rate
    )

    # Compare
    max_error = np.max(np.abs(traces - result))
    mean_error = np.mean(np.abs(traces - result))

    print(f"Input shape: {traces.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Max reconstruction error: {max_error:.6f}")
    print(f"Mean reconstruction error: {mean_error:.6f}")
    print(f"Input range: [{traces.min():.4f}, {traces.max():.4f}]")
    print(f"Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Per-trace analysis
    print("\nPer-trace max errors:")
    for i in range(n_traces):
        trace_error = np.max(np.abs(traces[:, i] - result[:, i]))
        print(f"  Trace {i}: {trace_error:.6f}")

    passed = max_error < 0.01
    print(f"\nTest passed: {passed}")
    return passed


def test_metal_vs_python_single_trace():
    """Compare Metal vs Python for a single trace, step by step."""
    print("\n" + "=" * 60)
    print("TEST: Metal vs Python Single Trace Comparison")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
    except ImportError:
        print("Kernel backend not available")
        return False

    if not is_metal_available():
        print("Metal not available")
        return False

    metal = get_metal_module()

    np.random.seed(42)
    n_samples = 256
    n_traces = 7  # Minimum for aperture=7

    # Create clean signal (all same trace for consistent aperture behavior)
    t = np.linspace(0, 4 * np.pi, n_samples)
    single_trace = np.sin(t).astype(np.float32)
    traces = np.tile(single_trace.reshape(-1, 1), (1, n_traces))

    nperseg = 64
    noverlap = 48
    aperture = 7
    threshold_k = 10000.0  # Very high - no thresholding
    sample_rate = 500.0

    # Metal result
    result_metal, _ = metal.stft_denoise(
        traces,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=aperture,
        threshold_k=threshold_k,
        fmin=0.0,
        fmax=0.0,
        sample_rate=sample_rate
    )

    # Python STFT/ISTFT for center trace
    center_idx = n_traces // 2
    _, _, stft_data = signal.stft(
        traces[:, center_idx], nperseg=nperseg, noverlap=noverlap
    )
    _, result_python = signal.istft(stft_data, nperseg=nperseg, noverlap=noverlap)

    if len(result_python) > n_samples:
        result_python = result_python[:n_samples]
    elif len(result_python) < n_samples:
        result_python = np.pad(result_python, (0, n_samples - len(result_python)))

    # Compare
    metal_center = result_metal[:, center_idx]

    print(f"Input trace: sin(t)")
    print(f"Original: min={single_trace.min():.4f}, max={single_trace.max():.4f}")
    print(f"Metal:    min={metal_center.min():.4f}, max={metal_center.max():.4f}")
    print(f"Python:   min={result_python.min():.4f}, max={result_python.max():.4f}")

    error_metal_vs_orig = np.max(np.abs(single_trace - metal_center))
    error_python_vs_orig = np.max(np.abs(single_trace - result_python))
    error_metal_vs_python = np.max(np.abs(metal_center - result_python))

    print(f"\nErrors:")
    print(f"  Metal vs Original: {error_metal_vs_orig:.6f}")
    print(f"  Python vs Original: {error_python_vs_orig:.2e}")
    print(f"  Metal vs Python: {error_metal_vs_python:.6f}")

    # Find where max difference occurs
    diff = np.abs(single_trace - metal_center)
    max_idx = np.argmax(diff)
    print(f"\nMax error location: sample {max_idx}")
    print(f"  Original value: {single_trace[max_idx]:.6f}")
    print(f"  Metal value: {metal_center[max_idx]:.6f}")
    print(f"  Python value: {result_python[max_idx]:.6f}")

    # Show first/last samples (boundary effects)
    print(f"\nBoundary analysis (first 5 samples):")
    for i in range(5):
        print(f"  [{i}] orig={single_trace[i]:.6f}, metal={metal_center[i]:.6f}, python={result_python[i]:.6f}")

    print(f"\nBoundary analysis (last 5 samples):")
    for i in range(n_samples-5, n_samples):
        print(f"  [{i}] orig={single_trace[i]:.6f}, metal={metal_center[i]:.6f}, python={result_python[i]:.6f}")


def test_thresholding_comparison():
    """Compare thresholding behavior between Metal and Python."""
    print("\n" + "=" * 60)
    print("TEST: Thresholding Behavior Comparison")
    print("=" * 60)

    try:
        from processors.kernel_backend import get_metal_module, is_metal_available
        from processors.stft_denoise import STFTDenoise
        from models.seismic_data import SeismicData
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    if not is_metal_available():
        print("Metal not available")
        return False

    metal = get_metal_module()

    np.random.seed(42)
    n_samples = 512
    n_traces = 21

    # Create signal with some outliers (noise)
    t = np.linspace(0, 8 * np.pi, n_samples)
    clean = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        clean[:, i] = np.sin(t + i * 0.1).astype(np.float32)

    # Add noise to create outliers
    noise = 0.3 * np.random.randn(n_samples, n_traces).astype(np.float32)
    noisy = clean + noise

    nperseg = 64
    noverlap = 48
    aperture = 7
    threshold_k = 3.0
    sample_rate = 500.0

    # Metal result (soft thresholding)
    result_metal, _ = metal.stft_denoise(
        noisy,
        nperseg=nperseg,
        noverlap=noverlap,
        aperture=aperture,
        threshold_k=threshold_k,
        fmin=5.0,
        fmax=200.0,
        sample_rate=sample_rate
    )

    # Python soft thresholding
    processor = STFTDenoise(
        aperture=aperture,
        fmin=5.0,
        fmax=200.0,
        nperseg=nperseg,
        noverlap=noverlap,
        threshold_k=threshold_k,
        threshold_mode='soft'
    )

    data = SeismicData(traces=noisy, sample_rate=sample_rate)
    result_python = processor.process(data).traces

    # Compare
    print(f"Input noisy: min={noisy.min():.4f}, max={noisy.max():.4f}, std={noisy.std():.4f}")
    print(f"Metal output: min={result_metal.min():.4f}, max={result_metal.max():.4f}, std={result_metal.std():.4f}")
    print(f"Python output: min={result_python.min():.4f}, max={result_python.max():.4f}, std={result_python.std():.4f}")

    # Compare denoising effectiveness
    error_metal_vs_clean = np.sqrt(np.mean((clean - result_metal)**2))
    error_python_vs_clean = np.sqrt(np.mean((clean - result_python)**2))
    error_noisy_vs_clean = np.sqrt(np.mean((clean - noisy)**2))

    print(f"\nRMSE vs clean signal:")
    print(f"  Noisy: {error_noisy_vs_clean:.4f}")
    print(f"  Metal: {error_metal_vs_clean:.4f}")
    print(f"  Python: {error_python_vs_clean:.4f}")

    # Compare Metal vs Python directly
    diff = np.abs(result_metal - result_python)
    print(f"\nMetal vs Python difference:")
    print(f"  Max: {diff.max():.4f}")
    print(f"  Mean: {diff.mean():.4f}")

    # Per-trace comparison
    print("\nPer-trace RMSE (Metal vs Python):")
    for i in [0, n_traces//2, n_traces-1]:
        trace_diff = np.sqrt(np.mean((result_metal[:, i] - result_python[:, i])**2))
        print(f"  Trace {i}: {trace_diff:.4f}")


def test_detailed_stft_coefficients():
    """Compare actual STFT coefficients between implementations."""
    print("\n" + "=" * 60)
    print("TEST: Detailed STFT Coefficient Analysis")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 128

    # Simple test signal
    t = np.linspace(0, 2 * np.pi, n_samples)
    trace = np.sin(t).astype(np.float32)

    nperseg = 32
    noverlap = 24
    hop = nperseg - noverlap

    # scipy STFT
    _, _, stft_scipy = signal.stft(trace, nperseg=nperseg, noverlap=noverlap)

    print(f"Signal: sin(t), length={n_samples}")
    print(f"nperseg={nperseg}, noverlap={noverlap}, hop={hop}")
    print(f"STFT shape: {stft_scipy.shape}")

    # Analyze DC component (should be ~0 for sine wave)
    print(f"\nDC component (freq bin 0):")
    print(f"  Magnitude: {np.abs(stft_scipy[0]).mean():.6f}")

    # Find dominant frequency
    mag = np.abs(stft_scipy)
    dom_freq_idx = np.unravel_index(np.argmax(mag), mag.shape)
    print(f"\nDominant coefficient:")
    print(f"  Location: freq_bin={dom_freq_idx[0]}, time_bin={dom_freq_idx[1]}")
    print(f"  Magnitude: {mag[dom_freq_idx]:.6f}")

    # Show frequency content
    print(f"\nMean magnitude per frequency bin:")
    for f in range(min(5, stft_scipy.shape[0])):
        mean_mag = np.mean(np.abs(stft_scipy[f, :]))
        print(f"  Freq bin {f}: {mean_mag:.6f}")


def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "=" * 70)
    print("STFT METAL VS PYTHON DETAILED COMPARISON")
    print("=" * 70)

    # Basic comparisons
    test_window_functions()
    test_stft_forward_comparison()
    test_istft_reconstruction()
    test_detailed_stft_coefficients()

    # Metal specific tests
    test_metal_reconstruction()
    test_metal_vs_python_single_trace()
    test_thresholding_comparison()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
