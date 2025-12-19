#!/usr/bin/env python3
"""
Comprehensive Benchmark: Metal C++ vs Python/Numba

Compares seismic processing algorithms:
- DWT denoising
- SWT denoising
- STFT denoising
- Gabor denoising
- FKK filtering
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Callable, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Metal kernels
try:
    import seismic_metal
    METAL_AVAILABLE = seismic_metal.is_available()
except ImportError:
    METAL_AVAILABLE = False
    print("Warning: Metal kernels not available")

# Import Python/Numba implementations
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from scipy import signal
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================================
# Python/Numba Reference Implementations
# ============================================================================

def python_dwt_denoise(traces: np.ndarray, wavelet: str = 'db4',
                       level: int = 4, threshold_k: float = 3.0) -> np.ndarray:
    """Python DWT denoising using PyWavelets."""
    if not PYWT_AVAILABLE:
        return traces.copy()

    output = np.zeros_like(traces)
    n_samples, n_traces = traces.shape

    for i in range(n_traces):
        # Decompose
        coeffs = pywt.wavedec(traces[:, i], wavelet, level=level)

        # Estimate noise from finest detail coeffs
        detail_finest = coeffs[-1]
        sigma = np.median(np.abs(detail_finest)) / 0.6745
        threshold = threshold_k * sigma

        # Apply soft thresholding to detail coefficients
        new_coeffs = [coeffs[0]]
        for j in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[j], threshold, mode='soft'))

        # Reconstruct
        output[:, i] = pywt.waverec(new_coeffs, wavelet)[:n_samples]

    return output


def python_swt_denoise(traces: np.ndarray, wavelet: str = 'db4',
                       level: int = 4, threshold_k: float = 3.0) -> np.ndarray:
    """Python SWT denoising using PyWavelets."""
    if not PYWT_AVAILABLE:
        return traces.copy()

    output = np.zeros_like(traces)
    n_samples, n_traces = traces.shape

    # Pad to power of 2
    n_padded = 2 ** int(np.ceil(np.log2(n_samples)))

    for i in range(n_traces):
        trace = np.pad(traces[:, i], (0, n_padded - n_samples), mode='reflect')

        # Decompose
        coeffs = pywt.swt(trace, wavelet, level=level)

        # Estimate noise from finest detail coeffs
        detail_finest = coeffs[0][1]  # First tuple, detail
        sigma = np.median(np.abs(detail_finest)) / 0.6745
        threshold = threshold_k * sigma

        # Threshold
        new_coeffs = []
        for cA, cD in coeffs:
            new_coeffs.append((cA, pywt.threshold(cD, threshold, mode='soft')))

        # Reconstruct
        rec = pywt.iswt(new_coeffs, wavelet)
        output[:, i] = rec[:n_samples]

    return output


def python_stft_denoise(traces: np.ndarray, nperseg: int = 64,
                        noverlap: int = 48, threshold_k: float = 3.0) -> np.ndarray:
    """Python STFT denoising using SciPy."""
    if not SCIPY_AVAILABLE:
        return traces.copy()

    output = np.zeros_like(traces)
    n_samples, n_traces = traces.shape

    for i in range(n_traces):
        # STFT
        f, t, Zxx = signal.stft(traces[:, i], nperseg=nperseg, noverlap=noverlap)

        # Amplitude and phase
        amp = np.abs(Zxx)
        phase = np.angle(Zxx)

        # MAD threshold per frequency bin
        for f_idx in range(amp.shape[0]):
            median = np.median(amp[f_idx, :])
            mad = np.median(np.abs(amp[f_idx, :] - median))
            thresh = threshold_k * mad * 1.4826

            # Soft threshold
            deviation = np.abs(amp[f_idx, :] - median)
            mask = deviation > thresh
            new_dev = np.maximum(deviation - thresh, 0)
            sign = np.sign(amp[f_idx, :] - median)
            amp[f_idx, :] = np.where(mask, median + sign * new_dev, amp[f_idx, :])

        # Reconstruct
        Zxx_new = amp * np.exp(1j * phase)
        _, rec = signal.istft(Zxx_new, nperseg=nperseg, noverlap=noverlap)
        output[:len(rec), i] = rec[:n_samples]

    return output[:n_samples, :]


def python_fkk_filter(volume: np.ndarray, dt: float, dx: float, dy: float,
                      v_min: float, v_max: float, mode: str = 'reject') -> np.ndarray:
    """Python FKK filter using NumPy FFT."""
    nt, nx, ny = volume.shape

    # 3D FFT
    spectrum = np.fft.rfftn(volume, axes=(0, 1, 2))

    # Build velocity mask
    nf = spectrum.shape[0]
    nkx = spectrum.shape[1]
    nky = spectrum.shape[2]

    df = 1.0 / (nt * dt)
    dkx = 1.0 / (nx * dx)
    dky = 1.0 / (ny * dy)

    mask = np.ones_like(spectrum, dtype=np.float32)

    for f_idx in range(nf):
        freq = f_idx * df
        for kx_idx in range(nkx):
            kx = (kx_idx - nkx // 2) * dkx if kx_idx >= nkx // 2 else kx_idx * dkx
            for ky_idx in range(nky):
                ky = (ky_idx - nky // 2) * dky if ky_idx >= nky // 2 else ky_idx * dky

                k_horiz = np.sqrt(kx**2 + ky**2)
                velocity = abs(freq) / k_horiz if k_horiz > 1e-10 else 1e10

                in_cone = v_min <= velocity <= v_max
                if mode == 'reject':
                    mask[f_idx, kx_idx, ky_idx] = 0.0 if in_cone else 1.0
                else:
                    mask[f_idx, kx_idx, ky_idx] = 1.0 if in_cone else 0.0

    # Apply mask
    spectrum *= mask

    # Inverse FFT
    output = np.fft.irfftn(spectrum, s=(nt, nx, ny), axes=(0, 1, 2))

    return output.astype(np.float32)


# ============================================================================
# Benchmark Runner
# ============================================================================

class BenchmarkResult:
    def __init__(self, name: str, impl: str, time_ms: float, throughput: float):
        self.name = name
        self.impl = impl
        self.time_ms = time_ms
        self.throughput = throughput

    def __repr__(self):
        return f"{self.name} ({self.impl}): {self.time_ms:.2f}ms, {self.throughput:.0f} traces/s"


def run_benchmark(name: str, impl: str, func: Callable,
                  data: np.ndarray, n_runs: int = 3,
                  warmup: int = 1) -> BenchmarkResult:
    """Run benchmark with warmup and multiple iterations."""

    # Warmup
    for _ in range(warmup):
        result = func(data)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    avg_time = np.mean(times)
    n_traces = data.shape[1] if data.ndim == 2 else data.shape[1] * data.shape[2]
    throughput = n_traces / (avg_time / 1000)

    return BenchmarkResult(name, impl, avg_time, throughput)


def run_all_benchmarks(n_samples: int = 1000, n_traces: int = 100,
                       n_runs: int = 5) -> List[BenchmarkResult]:
    """Run all benchmarks."""
    results = []

    # Generate test data
    np.random.seed(42)
    data_2d = np.random.randn(n_samples, n_traces).astype(np.float32)

    # FKK test volume (smaller for speed)
    nt, nx, ny = 64, 32, 32
    data_3d = np.random.randn(nt, nx, ny).astype(np.float32)

    print(f"\n{'='*70}")
    print(f"Benchmark: Metal C++ vs Python/Numba")
    print(f"{'='*70}")
    print(f"Test size: {n_samples} samples x {n_traces} traces")
    print(f"FKK volume: {nt} x {nx} x {ny}")
    print(f"Runs: {n_runs}")
    print(f"{'='*70}\n")

    # ============== DWT ==============
    print("DWT Denoising:")
    print("-" * 40)

    if METAL_AVAILABLE:
        def metal_dwt(d):
            return seismic_metal.dwt_denoise(d, 'db4', 4, 'soft', 3.0)[0]
        result = run_benchmark("DWT", "Metal C++", metal_dwt, data_2d, n_runs)
        results.append(result)
        print(f"  Metal C++: {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    if PYWT_AVAILABLE:
        result = run_benchmark("DWT", "Python+PyWavelets",
                              lambda d: python_dwt_denoise(d), data_2d, n_runs)
        results.append(result)
        print(f"  Python:    {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    # ============== SWT ==============
    print("\nSWT Denoising:")
    print("-" * 40)

    if METAL_AVAILABLE:
        def metal_swt(d):
            return seismic_metal.swt_denoise(d, 'db4', 4, 'soft', 3.0)[0]
        result = run_benchmark("SWT", "Metal C++", metal_swt, data_2d, n_runs)
        results.append(result)
        print(f"  Metal C++: {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    if PYWT_AVAILABLE:
        result = run_benchmark("SWT", "Python+PyWavelets",
                              lambda d: python_swt_denoise(d), data_2d, n_runs)
        results.append(result)
        print(f"  Python:    {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    # ============== STFT ==============
    print("\nSTFT Denoising:")
    print("-" * 40)

    if METAL_AVAILABLE:
        def metal_stft(d):
            return seismic_metal.stft_denoise(d, nperseg=64, noverlap=48, threshold_k=3.0)[0]
        result = run_benchmark("STFT", "Metal C++", metal_stft, data_2d, n_runs)
        results.append(result)
        print(f"  Metal C++: {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    if SCIPY_AVAILABLE:
        result = run_benchmark("STFT", "Python+SciPy",
                              lambda d: python_stft_denoise(d), data_2d, n_runs)
        results.append(result)
        print(f"  Python:    {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    # ============== Gabor ==============
    print("\nGabor Denoising:")
    print("-" * 40)

    if METAL_AVAILABLE:
        def metal_gabor(d):
            return seismic_metal.gabor_denoise(d, window_size=64, overlap_pct=75, threshold_k=3.0)[0]
        result = run_benchmark("Gabor", "Metal C++", metal_gabor, data_2d, n_runs)
        results.append(result)
        print(f"  Metal C++: {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    # Gabor in Python uses same STFT
    if SCIPY_AVAILABLE:
        result = run_benchmark("Gabor", "Python+SciPy",
                              lambda d: python_stft_denoise(d, nperseg=64, noverlap=48), data_2d, n_runs)
        results.append(result)
        print(f"  Python:    {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    # ============== FKK ==============
    print("\nFKK Filter (3D):")
    print("-" * 40)

    if METAL_AVAILABLE:
        def metal_fkk(d):
            return seismic_metal.fkk_filter(d, 0.002, 12.5, 12.5, 1500, 4500, 'reject', True)[0]
        result = run_benchmark("FKK", "Metal C++", metal_fkk, data_3d, n_runs)
        result.throughput = (nx * ny) / (result.time_ms / 1000)  # Fix throughput calc
        results.append(result)
        print(f"  Metal C++: {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    result = run_benchmark("FKK", "Python+NumPy",
                          lambda d: python_fkk_filter(d, 0.002, 12.5, 12.5, 1500, 4500), data_3d, n_runs)
    result.throughput = (nx * ny) / (result.time_ms / 1000)
    results.append(result)
    print(f"  Python:    {result.time_ms:.2f} ms ({result.throughput:.0f} traces/s)")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print comparison summary."""
    print(f"\n{'='*70}")
    print("SUMMARY: Metal C++ Speedup vs Python")
    print(f"{'='*70}")

    # Group by algorithm
    algorithms = {}
    for r in results:
        if r.name not in algorithms:
            algorithms[r.name] = {}
        algorithms[r.name][r.impl] = r

    for alg_name, impls in algorithms.items():
        metal_result = impls.get("Metal C++")
        python_keys = [k for k in impls.keys() if k != "Metal C++"]

        if metal_result and python_keys:
            python_result = impls[python_keys[0]]
            speedup = python_result.time_ms / metal_result.time_ms

            print(f"\n{alg_name}:")
            print(f"  Metal C++: {metal_result.time_ms:.2f} ms ({metal_result.throughput:.0f} traces/s)")
            print(f"  Python:    {python_result.time_ms:.2f} ms ({python_result.throughput:.0f} traces/s)")
            print(f"  Speedup:   {speedup:.1f}x")


def main():
    """Main benchmark entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Metal C++ vs Python")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples per trace")
    parser.add_argument("--traces", type=int, default=100, help="Number of traces")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    args = parser.parse_args()

    # Check dependencies
    print("Checking dependencies...")
    print(f"  Metal C++:  {'Available' if METAL_AVAILABLE else 'Not available'}")
    print(f"  PyWavelets: {'Available' if PYWT_AVAILABLE else 'Not available'}")
    print(f"  SciPy:      {'Available' if SCIPY_AVAILABLE else 'Not available'}")
    print(f"  Numba:      {'Available' if NUMBA_AVAILABLE else 'Not available'}")

    if METAL_AVAILABLE:
        print(f"\nMetal Device: {seismic_metal.get_device_info()['device_name']}")

    # Run benchmarks
    results = run_all_benchmarks(args.samples, args.traces, args.runs)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
