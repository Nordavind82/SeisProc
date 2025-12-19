"""
Processor Profiling Script for SeisProc

Profiles the following processors to identify bottlenecks for C++ kernel acceleration:
- DWT-Denoise (including SWT mode)
- STFT-Denoise
- FKK-Filter
- Gabor-Denoise

Outputs detailed timing breakdown for each component.
"""

import numpy as np
import pandas as pd
import time
import sys
import json
import cProfile
import pstats
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.seismic_data import SeismicData
from benchmarks.synthetic_data_generator import (
    generate_benchmark_dataset,
    load_benchmark_gather,
    load_benchmark_metadata,
    CrossspreadGeometry
)


@dataclass
class TimingResult:
    """Timing result for a single component."""
    name: str
    total_ms: float
    calls: int = 1
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    percentage: float = 0.0

    def __post_init__(self):
        if self.calls > 0:
            self.avg_ms = self.total_ms / self.calls


@dataclass
class ProfileResult:
    """Complete profiling result for a processor."""
    processor_name: str
    total_time_ms: float
    n_traces: int
    n_samples: int
    traces_per_sec: float
    samples_per_sec: float
    components: List[TimingResult] = field(default_factory=list)
    cprofile_stats: Optional[str] = None


class ComponentTimer:
    """Context manager for timing code sections."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self._stack: List[tuple] = []

    @contextmanager
    def section(self, name: str):
        """Time a code section."""
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed)

    def get_results(self) -> List[TimingResult]:
        """Get timing results for all sections."""
        total = sum(sum(times) for times in self.timings.values())
        results = []

        for name, times in self.timings.items():
            result = TimingResult(
                name=name,
                total_ms=sum(times),
                calls=len(times),
                avg_ms=np.mean(times),
                min_ms=np.min(times),
                max_ms=np.max(times),
                percentage=100 * sum(times) / total if total > 0 else 0
            )
            results.append(result)

        return sorted(results, key=lambda x: -x.total_ms)

    def reset(self):
        """Reset all timings."""
        self.timings.clear()


def profile_dwt_denoise(
    traces: np.ndarray,
    sample_rate_ms: float,
    wavelet: str = 'db4',
    level: int = 5,
    transform_type: str = 'dwt'
) -> ProfileResult:
    """
    Profile DWT-Denoise processor with component-level timing.

    Args:
        traces: Trace data (n_samples, n_traces)
        sample_rate_ms: Sample rate in milliseconds
        wavelet: Wavelet to use
        level: Decomposition level
        transform_type: 'dwt' or 'swt'

    Returns:
        ProfileResult with detailed timing breakdown
    """
    import pywt

    n_samples, n_traces = traces.shape
    timer = ComponentTimer()

    # Convert to float32 if needed
    if traces.dtype != np.float32:
        traces = traces.astype(np.float32)

    # Pad for SWT if needed
    if transform_type == 'swt':
        target_len = 2**int(np.ceil(np.log2(n_samples)))
        pad_len = target_len - n_samples
        if pad_len > 0:
            traces = np.pad(traces, ((0, pad_len), (0, 0)), mode='reflect')

    denoised = np.zeros_like(traces)
    all_coeffs = []
    thresholds = []

    start_total = time.perf_counter()

    # Component 1: Wavelet Decomposition
    with timer.section("wavelet_decomposition"):
        for i in range(n_traces):
            if transform_type == 'swt':
                coeffs = pywt.swt(traces[:, i], wavelet, level=level)
            else:
                coeffs = pywt.wavedec(traces[:, i], wavelet, level=level)
            all_coeffs.append(coeffs)

    # Component 2: MAD Threshold Computation
    with timer.section("mad_computation"):
        for coeffs in all_coeffs:
            if transform_type == 'swt':
                sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
            else:
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresholds.append(3.0 * sigma)

    # Component 3: Thresholding
    with timer.section("thresholding"):
        thresholded_coeffs = []
        for coeffs, thresh in zip(all_coeffs, thresholds):
            if transform_type == 'swt':
                new_coeffs = []
                for cA, cD in coeffs:
                    cD_thresh = pywt.threshold(cD, thresh, mode='soft')
                    new_coeffs.append((cA, cD_thresh))
                thresholded_coeffs.append(new_coeffs)
            else:
                new_coeffs = [coeffs[0]]
                for c in coeffs[1:]:
                    new_coeffs.append(pywt.threshold(c, thresh, mode='soft'))
                thresholded_coeffs.append(new_coeffs)

    # Component 4: Reconstruction
    with timer.section("wavelet_reconstruction"):
        for i, coeffs in enumerate(thresholded_coeffs):
            if transform_type == 'swt':
                denoised[:, i] = pywt.iswt(coeffs, wavelet)
            else:
                denoised[:, i] = pywt.waverec(coeffs, wavelet)[:n_samples]

    total_ms = (time.perf_counter() - start_total) * 1000

    return ProfileResult(
        processor_name=f"DWT-Denoise ({transform_type.upper()})",
        total_time_ms=total_ms,
        n_traces=n_traces,
        n_samples=n_samples,
        traces_per_sec=n_traces / (total_ms / 1000),
        samples_per_sec=(n_traces * n_samples) / (total_ms / 1000),
        components=timer.get_results()
    )


def profile_stft_denoise(
    traces: np.ndarray,
    sample_rate_ms: float,
    nperseg: int = 64,
    noverlap: int = 32,
    aperture: int = 7
) -> ProfileResult:
    """
    Profile STFT-Denoise processor with component-level timing.

    Args:
        traces: Trace data (n_samples, n_traces)
        sample_rate_ms: Sample rate in milliseconds
        nperseg: STFT window size
        noverlap: STFT overlap
        aperture: Spatial aperture

    Returns:
        ProfileResult with detailed timing breakdown
    """
    from scipy import signal

    n_samples, n_traces = traces.shape
    timer = ComponentTimer()
    half_ap = aperture // 2

    denoised = np.zeros_like(traces)

    start_total = time.perf_counter()

    for trace_idx in range(n_traces):
        # Get spatial aperture
        start_idx = max(0, trace_idx - half_ap)
        end_idx = min(n_traces, trace_idx + half_ap + 1)
        ensemble = traces[:, start_idx:end_idx]
        center_idx = trace_idx - start_idx

        # Component 1: STFT Forward
        with timer.section("stft_forward"):
            freqs, times, stft_batch = signal.stft(
                ensemble.T, nperseg=nperseg, noverlap=noverlap, axis=-1
            )

        # Component 2: MAD Computation
        with timer.section("mad_computation"):
            all_amplitudes = np.abs(stft_batch)
            median_amp = np.median(all_amplitudes, axis=0)
            mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)
            mad_scaled = mad * 1.4826
            threshold = 3.0 * np.maximum(mad_scaled, 1e-10)

        # Component 3: Thresholding
        with timer.section("thresholding"):
            stft_center = stft_batch[center_idx]
            magnitudes = np.abs(stft_center)
            phases = np.angle(stft_center)
            deviations = np.abs(magnitudes - median_amp)
            new_magnitudes = np.where(deviations > threshold, median_amp, magnitudes)
            stft_denoised = new_magnitudes * np.exp(1j * phases)

        # Component 4: ISTFT Inverse
        with timer.section("istft_inverse"):
            _, denoised_trace = signal.istft(stft_denoised, nperseg=nperseg, noverlap=noverlap)

        # Store result (handle length mismatch)
        out_len = min(len(denoised_trace), n_samples)
        denoised[:out_len, trace_idx] = denoised_trace[:out_len]

    total_ms = (time.perf_counter() - start_total) * 1000

    return ProfileResult(
        processor_name="STFT-Denoise",
        total_time_ms=total_ms,
        n_traces=n_traces,
        n_samples=n_samples,
        traces_per_sec=n_traces / (total_ms / 1000),
        samples_per_sec=(n_traces * n_samples) / (total_ms / 1000),
        components=timer.get_results()
    )


def profile_gabor_denoise(
    traces: np.ndarray,
    sample_rate_ms: float,
    window_size: int = 64,
    overlap_pct: float = 75.0,
    aperture: int = 7
) -> ProfileResult:
    """
    Profile Gabor-Denoise processor with component-level timing.

    Args:
        traces: Trace data (n_samples, n_traces)
        sample_rate_ms: Sample rate in milliseconds
        window_size: Gabor window size
        overlap_pct: Overlap percentage
        aperture: Spatial aperture

    Returns:
        ProfileResult with detailed timing breakdown
    """
    from scipy import signal

    n_samples, n_traces = traces.shape
    timer = ComponentTimer()
    half_ap = aperture // 2

    nperseg = window_size
    noverlap = int(nperseg * overlap_pct / 100)

    # Create Gaussian window
    sigma = nperseg / 6.0
    gabor_window = signal.windows.gaussian(nperseg, std=sigma)

    denoised = np.zeros_like(traces)

    start_total = time.perf_counter()

    for trace_idx in range(n_traces):
        start_idx = max(0, trace_idx - half_ap)
        end_idx = min(n_traces, trace_idx + half_ap + 1)
        ensemble = traces[:, start_idx:end_idx]
        center_idx = trace_idx - start_idx

        # Component 1: Gabor Forward (STFT with Gaussian window)
        with timer.section("gabor_forward"):
            _, _, stft_batch = signal.stft(
                ensemble.T, nperseg=nperseg, noverlap=noverlap,
                window=gabor_window, axis=-1
            )

        # Component 2: MAD Computation
        with timer.section("mad_computation"):
            all_amplitudes = np.abs(stft_batch)
            median_amp = np.median(all_amplitudes, axis=0)
            mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)
            threshold = 3.0 * mad * 1.4826
            threshold = np.maximum(threshold, 1e-10)

        # Component 3: Soft Thresholding
        with timer.section("soft_thresholding"):
            stft_center = stft_batch[center_idx]
            magnitudes = np.abs(stft_center)
            phases = np.angle(stft_center)
            deviations = np.abs(magnitudes - median_amp)
            signs = np.where(magnitudes >= median_amp, 1, -1)
            new_deviations = np.maximum(deviations - threshold, 0)
            new_magnitudes = np.maximum(median_amp + signs * new_deviations, 0)
            stft_denoised = new_magnitudes * np.exp(1j * phases)

        # Component 4: Gabor Inverse
        with timer.section("gabor_inverse"):
            _, denoised_trace = signal.istft(
                stft_denoised, nperseg=nperseg, noverlap=noverlap,
                window=gabor_window
            )

        out_len = min(len(denoised_trace), n_samples)
        denoised[:out_len, trace_idx] = denoised_trace[:out_len]

    total_ms = (time.perf_counter() - start_total) * 1000

    return ProfileResult(
        processor_name="Gabor-Denoise",
        total_time_ms=total_ms,
        n_traces=n_traces,
        n_samples=n_samples,
        traces_per_sec=n_traces / (total_ms / 1000),
        samples_per_sec=(n_traces * n_samples) / (total_ms / 1000),
        components=timer.get_results()
    )


def profile_fkk_filter(
    traces: np.ndarray,
    sample_rate_ms: float,
    dx: float = 12.5,
    dy: float = 25.0,
    device: str = 'auto'
) -> ProfileResult:
    """
    Profile FKK-Filter processor with component-level timing.

    Args:
        traces: Trace data (n_samples, n_traces)
        sample_rate_ms: Sample rate in milliseconds
        dx: Inline spacing in meters
        dy: Crossline spacing in meters
        device: 'auto', 'cpu', 'mps', or 'cuda'

    Returns:
        ProfileResult with detailed timing breakdown
    """
    import torch

    n_samples, n_traces = traces.shape
    timer = ComponentTimer()

    # Reshape to approximate 3D volume
    n_side = int(np.sqrt(n_traces))
    nx, ny = n_side, n_traces // n_side
    nt = n_samples

    # Truncate traces to fit exact grid
    n_traces_used = nx * ny
    volume = traces[:, :n_traces_used].reshape(nt, nx, ny).astype(np.float32)

    dt = sample_rate_ms / 1000.0

    # Select device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    start_total = time.perf_counter()

    # Component 1: Padding
    with timer.section("padding"):
        nt_fft = 2**int(np.ceil(np.log2(nt)))
        nx_fft = 2**int(np.ceil(np.log2(nx)))
        ny_fft = 2**int(np.ceil(np.log2(ny)))

        padded = np.pad(volume, (
            (0, nt_fft - nt),
            (0, nx_fft - nx),
            (0, ny_fft - ny)
        ), mode='constant')

    # Component 2: CPU to GPU transfer
    with timer.section("cpu_to_gpu"):
        data_gpu = torch.from_numpy(padded).float().to(device)
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Component 3: 3D FFT
    with timer.section("fft_3d"):
        spectrum = torch.fft.rfft(data_gpu, dim=0)
        spectrum = torch.fft.fft(spectrum, dim=1)
        spectrum = torch.fft.fft(spectrum, dim=2)
        spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Component 4: Build velocity cone mask
    with timer.section("mask_building"):
        f_axis = torch.fft.rfftfreq(nt_fft, dt, device=device, dtype=torch.float32)
        kx_axis = torch.fft.fftshift(torch.fft.fftfreq(nx_fft, dx, device=device, dtype=torch.float32))
        ky_axis = torch.fft.fftshift(torch.fft.fftfreq(ny_fft, dy, device=device, dtype=torch.float32))

        f_grid, kx_grid, ky_grid = torch.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')
        k_horizontal = torch.sqrt(kx_grid**2 + ky_grid**2)
        k_safe = torch.where(k_horizontal > 1e-10, k_horizontal,
                             torch.tensor(1e-10, device=device))
        velocity = torch.abs(f_grid) / k_safe

        # Velocity filter (reject 200-1500 m/s)
        mask = torch.ones_like(velocity)
        mask[(velocity >= 200) & (velocity <= 1500)] = 0.0
        mask[0, :, :] = 1.0  # Preserve DC

        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Component 5: Apply mask
    with timer.section("mask_application"):
        spectrum_filtered = spectrum * mask
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Component 6: 3D IFFT
    with timer.section("ifft_3d"):
        spectrum_filtered = torch.fft.ifftshift(spectrum_filtered, dim=(1, 2))
        spectrum_filtered = torch.fft.ifft(spectrum_filtered, dim=2)
        spectrum_filtered = torch.fft.ifft(spectrum_filtered, dim=1)
        result = torch.fft.irfft(spectrum_filtered, n=nt_fft, dim=0)
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Component 7: GPU to CPU transfer
    with timer.section("gpu_to_cpu"):
        output = result.real.cpu().numpy()

    # Component 8: Unpadding
    with timer.section("unpadding"):
        output = output[:nt, :nx, :ny]

    total_ms = (time.perf_counter() - start_total) * 1000

    return ProfileResult(
        processor_name=f"FKK-Filter ({device})",
        total_time_ms=total_ms,
        n_traces=n_traces_used,
        n_samples=n_samples,
        traces_per_sec=n_traces_used / (total_ms / 1000),
        samples_per_sec=(n_traces_used * n_samples) / (total_ms / 1000),
        components=timer.get_results()
    )


def print_profile_result(result: ProfileResult):
    """Pretty print profiling result."""
    print(f"\n{'='*70}")
    print(f"PROFILE: {result.processor_name}")
    print(f"{'='*70}")
    print(f"Data: {result.n_traces:,} traces x {result.n_samples} samples")
    print(f"Total time: {result.total_time_ms:.2f} ms")
    print(f"Throughput: {result.traces_per_sec:.1f} traces/s")
    print(f"           {result.samples_per_sec / 1e6:.2f} Msamples/s")
    print(f"\nComponent Breakdown:")
    print(f"{'Component':<30} {'Time (ms)':<12} {'Calls':<8} {'Avg (ms)':<12} {'%':<8}")
    print("-" * 70)

    for comp in result.components:
        print(f"{comp.name:<30} {comp.total_ms:<12.2f} {comp.calls:<8} "
              f"{comp.avg_ms:<12.2f} {comp.percentage:<8.1f}")

    print("-" * 70)


def run_full_benchmark(
    data_dir: str,
    n_gathers: int = 5,
    output_json: Optional[str] = None
) -> Dict[str, ProfileResult]:
    """
    Run complete benchmark suite on dataset.

    Args:
        data_dir: Path to benchmark dataset
        n_gathers: Number of gathers to profile
        output_json: Optional path to save JSON results

    Returns:
        Dictionary of processor name -> ProfileResult
    """
    print(f"\n{'#'*70}")
    print(f"# SEISMIC PROCESSOR BENCHMARKING SUITE")
    print(f"# Dataset: {data_dir}")
    print(f"# Gathers: {n_gathers}")
    print(f"{'#'*70}")

    # Load metadata
    metadata = load_benchmark_metadata(data_dir)
    sample_rate = metadata['sample_rate']

    results = {}

    for gather_idx in range(n_gathers):
        print(f"\n>>> Processing gather {gather_idx + 1}/{n_gathers}")

        # Load gather
        traces, headers = load_benchmark_gather(data_dir, gather_idx)
        print(f"    Loaded: {traces.shape[1]} traces x {traces.shape[0]} samples")

        # Profile DWT
        print("    Profiling DWT-Denoise...")
        result_dwt = profile_dwt_denoise(traces, sample_rate, transform_type='dwt')
        print_profile_result(result_dwt)

        # Profile SWT
        print("    Profiling SWT-Denoise...")
        result_swt = profile_dwt_denoise(traces, sample_rate, transform_type='swt')
        print_profile_result(result_swt)

        # Profile STFT
        print("    Profiling STFT-Denoise...")
        result_stft = profile_stft_denoise(traces, sample_rate)
        print_profile_result(result_stft)

        # Profile Gabor
        print("    Profiling Gabor-Denoise...")
        result_gabor = profile_gabor_denoise(traces, sample_rate)
        print_profile_result(result_gabor)

        # Profile FKK
        print("    Profiling FKK-Filter...")
        result_fkk = profile_fkk_filter(traces, sample_rate)
        print_profile_result(result_fkk)

        # Store results (use last gather's results)
        results = {
            'DWT': result_dwt,
            'SWT': result_swt,
            'STFT': result_stft,
            'Gabor': result_gabor,
            'FKK': result_fkk
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Throughput Comparison")
    print(f"{'='*70}")
    print(f"{'Processor':<25} {'Traces/s':<15} {'Msamples/s':<15} {'Time (ms)':<12}")
    print("-" * 70)

    for name, result in results.items():
        print(f"{result.processor_name:<25} {result.traces_per_sec:<15.1f} "
              f"{result.samples_per_sec/1e6:<15.2f} {result.total_time_ms:<12.2f}")

    # Save to JSON if requested
    if output_json:
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(data_dir),
            'n_gathers': n_gathers,
            'results': {}
        }

        for name, result in results.items():
            json_data['results'][name] = {
                'processor_name': result.processor_name,
                'total_time_ms': result.total_time_ms,
                'n_traces': result.n_traces,
                'n_samples': result.n_samples,
                'traces_per_sec': result.traces_per_sec,
                'samples_per_sec': result.samples_per_sec,
                'components': [
                    {
                        'name': c.name,
                        'total_ms': c.total_ms,
                        'calls': c.calls,
                        'avg_ms': c.avg_ms,
                        'percentage': c.percentage
                    }
                    for c in result.components
                ]
            }

        with open(output_json, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"\nResults saved to: {output_json}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Profile SeisProc processors')
    parser.add_argument('--data-dir', '-d', type=str, default='benchmark_data',
                        help='Path to benchmark dataset')
    parser.add_argument('--gathers', '-g', type=int, default=3,
                        help='Number of gathers to profile')
    parser.add_argument('--generate', action='store_true',
                        help='Generate benchmark data if not exists')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with smaller dataset')

    args = parser.parse_args()

    data_path = Path(args.data_dir)

    # Generate data if needed
    if args.generate or not data_path.exists():
        if args.quick:
            geometry = CrossspreadGeometry(
                n_gathers=10,
                traces_per_gather=500,
                n_samples=800
            )
        else:
            geometry = CrossspreadGeometry(
                n_gathers=100,
                traces_per_gather=4000,
                n_samples=1600
            )
        generate_benchmark_dataset(args.data_dir, geometry)

    # Run benchmark
    run_full_benchmark(
        args.data_dir,
        n_gathers=args.gathers,
        output_json=args.output
    )
