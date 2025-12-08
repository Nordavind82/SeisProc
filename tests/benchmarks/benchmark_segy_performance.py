"""
Comprehensive SEGY performance benchmark suite.

Measures loading performance with various optimizations enabled.
Use this to validate performance improvements.

Usage:
    python -m tests.benchmarks.benchmark_segy_performance [--quick] [--create-test-file]
"""
import time
import gc
import tracemalloc
import tempfile
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    test_name: str
    n_traces: int
    n_samples: int
    file_size_mb: float
    duration_sec: float
    throughput_mb_per_sec: float
    throughput_traces_per_sec: float
    peak_memory_mb: float
    method: str
    success: bool
    error: Optional[str] = None


class SEGYBenchmarkSuite:
    """Comprehensive SEGY benchmark suite."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def create_test_segy(self, path: Path, n_samples: int, n_traces: int,
                         format_code: int = 5) -> float:
        """Create test SEGY file and return size in MB."""
        import segyio

        np.random.seed(42)

        spec = segyio.spec()
        spec.format = format_code  # 5 = IEEE float
        spec.samples = range(n_samples)
        spec.tracecount = n_traces

        with segyio.create(str(path), spec) as f:
            for i in range(n_traces):
                f.trace[i] = np.random.randn(n_samples).astype(np.float32)
                f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = i + 1
                f.header[i][segyio.TraceField.CDP] = i // 10
                f.header[i][segyio.TraceField.offset] = (i % 10) * 100

        return path.stat().st_size / (1024 * 1024)

    def benchmark_optimized_reader(self, segy_path: Path, n_samples: int,
                                    n_traces: int, file_size_mb: float) -> BenchmarkResult:
        """Benchmark optimized SEGYReader with mmap and batch headers."""
        from utils.segy_import.segy_reader import SEGYReader
        from utils.segy_import.header_mapping import HeaderMapping

        gc.collect()
        tracemalloc.start()

        try:
            mapping = HeaderMapping()
            mapping.add_standard_headers()  # Add standard headers for batch reading
            reader = SEGYReader(str(segy_path), mapping)

            start = time.perf_counter()
            traces, headers = reader.read_all_traces()
            duration = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Verify data
            assert traces.shape == (n_samples, n_traces), f"Shape mismatch: {traces.shape}"
            assert len(headers) == n_traces, f"Header count mismatch: {len(headers)}"

            return BenchmarkResult(
                test_name="optimized_reader",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=duration,
                throughput_mb_per_sec=file_size_mb / duration,
                throughput_traces_per_sec=n_traces / duration,
                peak_memory_mb=peak / 1024 / 1024,
                method="segyio+mmap+batch",
                success=True
            )

        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                test_name="optimized_reader",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=0,
                throughput_mb_per_sec=0,
                throughput_traces_per_sec=0,
                peak_memory_mb=0,
                method="segyio+mmap+batch",
                success=False,
                error=str(e)
            )

    def benchmark_fast_reader(self, segy_path: Path, n_samples: int,
                              n_traces: int, file_size_mb: float) -> BenchmarkResult:
        """Benchmark FastSEGYReader with segfast (if available)."""
        from utils.segy_import.segy_reader_fast import FastSEGYReader, SEGFAST_AVAILABLE
        from utils.segy_import.header_mapping import HeaderMapping

        if not SEGFAST_AVAILABLE:
            return BenchmarkResult(
                test_name="fast_reader",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=0,
                throughput_mb_per_sec=0,
                throughput_traces_per_sec=0,
                peak_memory_mb=0,
                method="segfast",
                success=False,
                error="segfast not installed"
            )

        gc.collect()
        tracemalloc.start()

        try:
            mapping = HeaderMapping()
            mapping.add_standard_headers()
            reader = FastSEGYReader(str(segy_path), mapping)

            start = time.perf_counter()
            traces, headers = reader.read_all_traces()
            duration = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return BenchmarkResult(
                test_name="fast_reader",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=duration,
                throughput_mb_per_sec=file_size_mb / duration,
                throughput_traces_per_sec=n_traces / duration,
                peak_memory_mb=peak / 1024 / 1024,
                method="segfast",
                success=True
            )

        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                test_name="fast_reader",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=0,
                throughput_mb_per_sec=0,
                throughput_traces_per_sec=0,
                peak_memory_mb=0,
                method="segfast",
                success=False,
                error=str(e)
            )

    def benchmark_chunked_reading(self, segy_path: Path, n_samples: int,
                                   n_traces: int, file_size_mb: float,
                                   chunk_size: int = 5000) -> BenchmarkResult:
        """Benchmark chunked reading with buffer pooling."""
        from utils.segy_import.segy_reader import SEGYReader
        from utils.segy_import.header_mapping import HeaderMapping

        gc.collect()
        tracemalloc.start()

        try:
            mapping = HeaderMapping()
            mapping.add_standard_headers()
            reader = SEGYReader(str(segy_path), mapping)

            start = time.perf_counter()
            total_traces = 0
            for traces, headers, start_idx, end_idx in reader.read_traces_in_chunks(chunk_size):
                total_traces = end_idx
                # Simulate processing
                _ = traces.mean()
            duration = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return BenchmarkResult(
                test_name=f"chunked_{chunk_size}",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=duration,
                throughput_mb_per_sec=file_size_mb / duration,
                throughput_traces_per_sec=n_traces / duration,
                peak_memory_mb=peak / 1024 / 1024,
                method=f"chunked-{chunk_size}",
                success=True
            )

        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                test_name=f"chunked_{chunk_size}",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=0,
                throughput_mb_per_sec=0,
                throughput_traces_per_sec=0,
                peak_memory_mb=0,
                method=f"chunked-{chunk_size}",
                success=False,
                error=str(e)
            )

    def benchmark_raw_segyio(self, segy_path: Path, n_samples: int,
                              n_traces: int, file_size_mb: float) -> BenchmarkResult:
        """Benchmark raw segyio without optimizations (baseline)."""
        import segyio

        gc.collect()
        tracemalloc.start()

        try:
            start = time.perf_counter()

            with segyio.open(str(segy_path), ignore_geometry=True) as f:
                # No mmap - baseline
                traces = np.zeros((n_samples, n_traces), dtype=np.float32)
                headers = []

                for i in range(n_traces):
                    traces[:, i] = f.trace[i]
                    headers.append(dict(f.header[i]))

            duration = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return BenchmarkResult(
                test_name="raw_segyio_baseline",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=duration,
                throughput_mb_per_sec=file_size_mb / duration,
                throughput_traces_per_sec=n_traces / duration,
                peak_memory_mb=peak / 1024 / 1024,
                method="segyio-baseline",
                success=True
            )

        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                test_name="raw_segyio_baseline",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=0,
                throughput_mb_per_sec=0,
                throughput_traces_per_sec=0,
                peak_memory_mb=0,
                method="segyio-baseline",
                success=False,
                error=str(e)
            )

    def benchmark_mmap_only(self, segy_path: Path, n_samples: int,
                            n_traces: int, file_size_mb: float) -> BenchmarkResult:
        """Benchmark segyio with mmap only."""
        import segyio

        gc.collect()
        tracemalloc.start()

        try:
            start = time.perf_counter()

            with segyio.open(str(segy_path), ignore_geometry=True) as f:
                f.mmap()  # Enable mmap
                traces = np.empty((n_samples, n_traces), dtype=np.float32)
                headers = []

                for i in range(n_traces):
                    traces[:, i] = f.trace[i]
                    headers.append(dict(f.header[i]))

            duration = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return BenchmarkResult(
                test_name="segyio_mmap",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=duration,
                throughput_mb_per_sec=file_size_mb / duration,
                throughput_traces_per_sec=n_traces / duration,
                peak_memory_mb=peak / 1024 / 1024,
                method="segyio+mmap",
                success=True
            )

        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                test_name="segyio_mmap",
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_mb=file_size_mb,
                duration_sec=0,
                throughput_mb_per_sec=0,
                throughput_traces_per_sec=0,
                peak_memory_mb=0,
                method="segyio+mmap",
                success=False,
                error=str(e)
            )

    def run_benchmark_suite(self, sizes: List[tuple] = None) -> List[BenchmarkResult]:
        """Run complete benchmark suite at various file sizes."""
        if sizes is None:
            sizes = [
                (1000, 1000),     # ~4 MB
                (2000, 5000),    # ~40 MB
                (2000, 10000),   # ~80 MB
                (2000, 25000),   # ~200 MB
            ]

        all_results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for n_samples, n_traces in sizes:
                self.log(f"\n{'='*70}")
                self.log(f"Benchmark: {n_samples} samples x {n_traces:,} traces")
                self.log('='*70)

                # Create test file
                segy_path = tmpdir / f"test_{n_samples}_{n_traces}.sgy"
                self.log("Creating test SEGY file...")
                file_size_mb = self.create_test_segy(segy_path, n_samples, n_traces)
                self.log(f"  File size: {file_size_mb:.1f} MB")

                # Run benchmarks
                benchmarks = [
                    ("Baseline (no opts)", self.benchmark_raw_segyio),
                    ("With mmap", self.benchmark_mmap_only),
                    ("Optimized Reader", self.benchmark_optimized_reader),
                    ("Chunked (5000)", lambda p, ns, nt, fs:
                        self.benchmark_chunked_reading(p, ns, nt, fs, 5000)),
                    ("Fast Reader", self.benchmark_fast_reader),
                ]

                for name, benchmark_fn in benchmarks:
                    self.log(f"\n  {name}...")
                    gc.collect()

                    result = benchmark_fn(segy_path, n_samples, n_traces, file_size_mb)
                    all_results.append(result)

                    if result.success:
                        self.log(f"    Duration: {result.duration_sec:.2f}s")
                        self.log(f"    Throughput: {result.throughput_mb_per_sec:.1f} MB/s, "
                                f"{result.throughput_traces_per_sec:.0f} traces/s")
                        self.log(f"    Peak memory: {result.peak_memory_mb:.1f} MB")
                    else:
                        self.log(f"    FAILED: {result.error}")

                # Cleanup
                segy_path.unlink()

        self.results = all_results
        return all_results

    def print_summary(self):
        """Print summary table of results."""
        print("\n" + "="*100)
        print("BENCHMARK SUMMARY")
        print("="*100)
        print(f"{'Test':<25} {'Traces':>10} {'Size MB':>10} {'Duration':>10} "
              f"{'MB/s':>10} {'Traces/s':>12} {'Memory MB':>12}")
        print("-"*100)

        for r in self.results:
            if r.success:
                print(f"{r.test_name:<25} {r.n_traces:>10,} {r.file_size_mb:>10.1f} "
                      f"{r.duration_sec:>10.2f} {r.throughput_mb_per_sec:>10.1f} "
                      f"{r.throughput_traces_per_sec:>12,.0f} {r.peak_memory_mb:>12.1f}")
            else:
                print(f"{r.test_name:<25} {r.n_traces:>10,} {'FAILED':>10} - {r.error}")

        print("="*100)

        # Calculate speedups
        self._print_speedup_analysis()

    def _print_speedup_analysis(self):
        """Print speedup analysis comparing optimizations."""
        print("\nSPEEDUP ANALYSIS")
        print("-"*60)

        # Group by trace count
        trace_counts = sorted(set(r.n_traces for r in self.results))

        for n_traces in trace_counts:
            subset = [r for r in self.results if r.n_traces == n_traces and r.success]
            if not subset:
                continue

            baseline = next((r for r in subset if 'baseline' in r.test_name), None)
            if not baseline:
                continue

            print(f"\n{n_traces:,} traces:")
            for r in subset:
                if r.test_name != baseline.test_name:
                    speedup = baseline.duration_sec / r.duration_sec if r.duration_sec > 0 else 0
                    print(f"  {r.method}: {speedup:.2f}x speedup")

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': [asdict(r) for r in self.results]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {filepath}")


def run_quick_benchmark():
    """Run quick benchmark with smaller file sizes."""
    suite = SEGYBenchmarkSuite()
    sizes = [
        (1000, 1000),   # ~4 MB
        (1000, 5000),   # ~20 MB
    ]
    suite.run_benchmark_suite(sizes)
    suite.print_summary()
    return suite.results


def run_full_benchmark():
    """Run full benchmark suite."""
    suite = SEGYBenchmarkSuite()
    suite.run_benchmark_suite()
    suite.print_summary()
    return suite.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEGY Performance Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.quick:
        results = run_quick_benchmark()
    else:
        results = run_full_benchmark()

    if args.save:
        suite = SEGYBenchmarkSuite(verbose=False)
        suite.results = results
        suite.save_results(args.save)
