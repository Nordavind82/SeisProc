"""
Performance benchmarks for S-Transform processing.

Compares CPU vs GPU implementations across various data sizes.

Usage:
    python -m tests.benchmarks.benchmark_stransform
"""
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    n_samples: int
    n_traces: int
    duration_seconds: float
    samples_per_second: float
    traces_per_second: float
    backend: str  # 'cpu', 'gpu-cuda', 'gpu-mps'
    memory_mb: Optional[float] = None
    error: Optional[str] = None


def generate_synthetic_data(n_samples: int, n_traces: int) -> np.ndarray:
    """Generate synthetic seismic data for benchmarking."""
    np.random.seed(42)
    t = np.arange(n_samples) * 0.002  # 2ms sample rate

    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        # Ricker wavelet
        freq = 30
        center = 0.2 + 0.01 * i
        wavelet_t = t - center
        wavelet = (1 - 2 * (np.pi * freq * wavelet_t) ** 2) * np.exp(-(np.pi * freq * wavelet_t) ** 2)
        traces[:, i] = wavelet + 0.1 * np.random.randn(n_samples).astype(np.float32)

    return traces


def benchmark_cpu_stransform(traces: np.ndarray, sample_rate_ms: float = 2.0) -> BenchmarkResult:
    """Benchmark CPU S-Transform implementation."""
    from processors.tf_denoise import TFDenoise
    from models.seismic_data import SeismicData

    n_samples, n_traces = traces.shape
    data = SeismicData(traces=traces, sample_rate=sample_rate_ms)

    processor = TFDenoise(
        aperture=5,
        fmin=5.0,
        fmax=80.0,
        threshold_k=3.0
    )

    # Warm-up run
    _ = processor.process(data)

    # Timed run
    start = time.perf_counter()
    result = processor.process(data)
    duration = time.perf_counter() - start

    total_samples = n_samples * n_traces

    return BenchmarkResult(
        name="CPU S-Transform",
        n_samples=n_samples,
        n_traces=n_traces,
        duration_seconds=duration,
        samples_per_second=total_samples / duration,
        traces_per_second=n_traces / duration,
        backend="cpu"
    )


def benchmark_gpu_stransform(traces: np.ndarray, sample_rate_ms: float = 2.0) -> BenchmarkResult:
    """Benchmark GPU S-Transform implementation."""
    try:
        from processors.tf_denoise_gpu import TFDenoiseGPU
        from models.seismic_data import SeismicData
        import torch
    except ImportError as e:
        return BenchmarkResult(
            name="GPU S-Transform",
            n_samples=traces.shape[0],
            n_traces=traces.shape[1],
            duration_seconds=0,
            samples_per_second=0,
            traces_per_second=0,
            backend="gpu",
            error=f"Import error: {e}"
        )

    # Determine backend
    if torch.cuda.is_available():
        backend = "gpu-cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        backend = "gpu-mps"
    else:
        return BenchmarkResult(
            name="GPU S-Transform",
            n_samples=traces.shape[0],
            n_traces=traces.shape[1],
            duration_seconds=0,
            samples_per_second=0,
            traces_per_second=0,
            backend="gpu",
            error="No GPU available"
        )

    n_samples, n_traces = traces.shape
    data = SeismicData(traces=traces, sample_rate=sample_rate_ms)

    processor = TFDenoiseGPU(
        aperture=5,
        fmin=5.0,
        fmax=80.0,
        threshold_k=3.0,
        use_gpu='auto'
    )

    # Warm-up run (important for GPU)
    _ = processor.process(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    result = processor.process(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration = time.perf_counter() - start

    total_samples = n_samples * n_traces

    return BenchmarkResult(
        name="GPU S-Transform",
        n_samples=n_samples,
        n_traces=n_traces,
        duration_seconds=duration,
        samples_per_second=total_samples / duration,
        traces_per_second=n_traces / duration,
        backend=backend
    )


def benchmark_bandpass_filter(traces: np.ndarray, sample_rate_ms: float = 2.0) -> BenchmarkResult:
    """Benchmark bandpass filter implementation."""
    from processors.bandpass_filter import BandpassFilter
    from models.seismic_data import SeismicData

    n_samples, n_traces = traces.shape
    data = SeismicData(traces=traces, sample_rate=sample_rate_ms)

    processor = BandpassFilter(low_freq=5.0, high_freq=80.0)

    # Warm-up
    _ = processor.process(data)

    # Timed run
    start = time.perf_counter()
    result = processor.process(data)
    duration = time.perf_counter() - start

    total_samples = n_samples * n_traces

    return BenchmarkResult(
        name="Bandpass Filter",
        n_samples=n_samples,
        n_traces=n_traces,
        duration_seconds=duration,
        samples_per_second=total_samples / duration,
        traces_per_second=n_traces / duration,
        backend="cpu"
    )


def run_benchmark_suite(sizes: List[tuple] = None) -> List[BenchmarkResult]:
    """Run complete benchmark suite across various data sizes."""
    if sizes is None:
        sizes = [
            (1000, 50),    # Small: 50K samples
            (2000, 100),   # Medium: 200K samples
            (4000, 200),   # Large: 800K samples
            (8000, 500),   # XL: 4M samples
        ]

    results = []

    for n_samples, n_traces in sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {n_samples} samples x {n_traces} traces")
        print(f"Total samples: {n_samples * n_traces:,}")
        print('='*60)

        traces = generate_synthetic_data(n_samples, n_traces)

        # Bandpass filter
        print("  Running Bandpass Filter...")
        bp_result = benchmark_bandpass_filter(traces)
        results.append(bp_result)
        print(f"    Duration: {bp_result.duration_seconds:.3f}s, "
              f"Throughput: {bp_result.samples_per_second/1e6:.2f} MS/s")

        # CPU S-Transform
        print("  Running CPU S-Transform...")
        cpu_result = benchmark_cpu_stransform(traces)
        results.append(cpu_result)
        print(f"    Duration: {cpu_result.duration_seconds:.3f}s, "
              f"Throughput: {cpu_result.samples_per_second/1e6:.2f} MS/s")

        # GPU S-Transform
        print("  Running GPU S-Transform...")
        gpu_result = benchmark_gpu_stransform(traces)
        results.append(gpu_result)
        if gpu_result.error:
            print(f"    Error: {gpu_result.error}")
        else:
            print(f"    Duration: {gpu_result.duration_seconds:.3f}s, "
                  f"Throughput: {gpu_result.samples_per_second/1e6:.2f} MS/s")
            if cpu_result.duration_seconds > 0:
                speedup = cpu_result.duration_seconds / gpu_result.duration_seconds
                print(f"    GPU Speedup: {speedup:.1f}x")

    return results


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save benchmark results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Test':<25} {'Size':<15} {'Backend':<12} {'Duration':<12} {'Throughput':<15}")
    print("-"*80)

    for r in results:
        if r.error:
            status = f"Error: {r.error[:20]}..."
            print(f"{r.name:<25} {r.n_samples}x{r.n_traces:<8} {r.backend:<12} {status}")
        else:
            size_str = f"{r.n_samples}x{r.n_traces}"
            throughput_str = f"{r.samples_per_second/1e6:.2f} MS/s"
            print(f"{r.name:<25} {size_str:<15} {r.backend:<12} {r.duration_seconds:.3f}s{'':<6} {throughput_str:<15}")

    print("="*80)


def check_regression(current: List[BenchmarkResult], baseline_path: Path,
                     threshold: float = 0.2) -> bool:
    """
    Check for performance regression against baseline.

    Returns True if no regression detected, False otherwise.
    """
    if not baseline_path.exists():
        print(f"No baseline found at {baseline_path}, skipping regression check")
        return True

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_results = {
        (r['name'], r['n_samples'], r['n_traces'], r['backend']): r
        for r in baseline_data['results']
    }

    regression_found = False

    for current_result in current:
        if current_result.error:
            continue

        key = (current_result.name, current_result.n_samples,
               current_result.n_traces, current_result.backend)

        if key in baseline_results:
            baseline = baseline_results[key]
            baseline_throughput = baseline['samples_per_second']
            current_throughput = current_result.samples_per_second

            # Check if current is more than threshold% slower
            if baseline_throughput > 0:
                change = (current_throughput - baseline_throughput) / baseline_throughput

                if change < -threshold:
                    print(f"REGRESSION: {current_result.name} ({current_result.n_samples}x{current_result.n_traces})")
                    print(f"  Baseline: {baseline_throughput/1e6:.2f} MS/s")
                    print(f"  Current:  {current_throughput/1e6:.2f} MS/s")
                    print(f"  Change:   {change*100:.1f}%")
                    regression_found = True

    return not regression_found


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SeisProc Performance Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with smaller sizes")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--baseline", type=str, help="Compare against baseline JSON file")
    args = parser.parse_args()

    if args.quick:
        sizes = [(1000, 50), (2000, 100)]
    else:
        sizes = None  # Use default sizes

    results = run_benchmark_suite(sizes)
    print_summary(results)

    if args.save:
        save_results(results, Path(args.save))

    if args.baseline:
        baseline_path = Path(args.baseline)
        if not check_regression(results, baseline_path):
            print("\nPERFORMANCE REGRESSION DETECTED!")
            sys.exit(1)
        else:
            print("\nNo performance regression detected.")
