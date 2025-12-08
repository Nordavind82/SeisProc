# SEGY Performance Benchmarks

## Overview

This document defines standardized benchmarks to measure SEGY loading performance before and after optimizations. Use these tests to track progress and validate improvements.

---

## Benchmark Categories

1. [Micro-Benchmarks](#1-micro-benchmarks) - Individual operation timing
2. [Integration Benchmarks](#2-integration-benchmarks) - End-to-end workflows
3. [Scalability Benchmarks](#3-scalability-benchmarks) - Performance at scale
4. [Memory Benchmarks](#4-memory-benchmarks) - Memory efficiency
5. [Comparison Benchmarks](#5-comparison-benchmarks) - Library comparisons

---

## 1. Micro-Benchmarks

### Benchmark 1.1: Single Trace Read Time

```python
# File: tests/benchmarks/bench_1_1_single_trace.py

import time
import statistics
import segyio
import numpy as np
from pathlib import Path

def benchmark_single_trace_read(segy_path: str, n_iterations: int = 1000) -> dict:
    """
    Measure time to read a single trace.

    This is the fundamental operation - all optimizations should improve this.
    """
    results = {'no_mmap': [], 'with_mmap': []}

    with segyio.open(segy_path, ignore_geometry=True) as f:
        n_traces = f.tracecount
        indices = np.random.randint(0, n_traces, n_iterations)

        # Without mmap
        for idx in indices:
            start = time.perf_counter_ns()
            _ = f.trace[idx]
            results['no_mmap'].append(time.perf_counter_ns() - start)

    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()

        # With mmap
        for idx in indices:
            start = time.perf_counter_ns()
            _ = f.trace[idx]
            results['with_mmap'].append(time.perf_counter_ns() - start)

    return {
        'no_mmap_mean_us': statistics.mean(results['no_mmap']) / 1000,
        'no_mmap_median_us': statistics.median(results['no_mmap']) / 1000,
        'no_mmap_p99_us': np.percentile(results['no_mmap'], 99) / 1000,
        'with_mmap_mean_us': statistics.mean(results['with_mmap']) / 1000,
        'with_mmap_median_us': statistics.median(results['with_mmap']) / 1000,
        'with_mmap_p99_us': np.percentile(results['with_mmap'], 99) / 1000,
    }

# Target metrics:
# - Mean single trace read: < 50 microseconds (with mmap on SSD)
# - P99 single trace read: < 200 microseconds
```

### Benchmark 1.2: Header Read Time

```python
# File: tests/benchmarks/bench_1_2_header_read.py

import time
import segyio
import numpy as np

def benchmark_header_read(segy_path: str, n_iterations: int = 1000) -> dict:
    """Measure header access time."""
    results = {'raw': [], 'parsed': []}

    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        indices = np.random.randint(0, n_traces, n_iterations)

        # Raw header access
        for idx in indices:
            start = time.perf_counter_ns()
            _ = f.header[idx]
            results['raw'].append(time.perf_counter_ns() - start)

        # Parsed header access (specific fields)
        fields = [
            segyio.TraceField.TRACE_SEQUENCE_LINE,
            segyio.TraceField.CDP,
            segyio.TraceField.offset,
            segyio.TraceField.SourceX,
            segyio.TraceField.SourceY,
        ]
        for idx in indices:
            start = time.perf_counter_ns()
            header = f.header[idx]
            parsed = {str(field): header[field] for field in fields}
            results['parsed'].append(time.perf_counter_ns() - start)

    return {
        'raw_mean_us': np.mean(results['raw']) / 1000,
        'parsed_mean_us': np.mean(results['parsed']) / 1000,
        'parsing_overhead_us': (np.mean(results['parsed']) - np.mean(results['raw'])) / 1000
    }

# Target metrics:
# - Raw header read: < 20 microseconds
# - Parsed header: < 50 microseconds
# - Parsing overhead: < 30 microseconds
```

### Benchmark 1.3: Batch Read Time

```python
# File: tests/benchmarks/bench_1_3_batch_read.py

import time
import segyio
import numpy as np

def benchmark_batch_read(segy_path: str) -> dict:
    """Measure batch read performance at different sizes."""
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    results = {}

    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        n_samples = len(f.samples)

        for batch_size in batch_sizes:
            if batch_size > n_traces:
                continue

            n_batches = min(10, n_traces // batch_size)
            times = []

            for batch_idx in range(n_batches):
                start_trace = batch_idx * batch_size
                buffer = np.empty((n_samples, batch_size), dtype=np.float32)

                start = time.perf_counter()
                for i in range(batch_size):
                    buffer[:, i] = f.trace[start_trace + i]
                times.append(time.perf_counter() - start)

            results[batch_size] = {
                'mean_time_ms': np.mean(times) * 1000,
                'throughput_traces_per_sec': batch_size / np.mean(times),
                'time_per_trace_us': np.mean(times) / batch_size * 1e6
            }

    return results

# Target metrics:
# - Batch 1000: > 20,000 traces/second
# - Per-trace time decreases with batch size (amortization)
```

---

## 2. Integration Benchmarks

### Benchmark 2.1: Full File Load

```python
# File: tests/benchmarks/bench_2_1_full_load.py

import time
import tempfile
import numpy as np
import segyio
from pathlib import Path
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping

def create_test_file(path: Path, n_samples: int, n_traces: int):
    """Create test SEGY file."""
    np.random.seed(42)
    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(str(path), spec) as f:
        for i in range(n_traces):
            f.trace[i] = np.random.randn(n_samples).astype(np.float32)
            f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = i + 1
            f.header[i][segyio.TraceField.CDP] = i // 10

def benchmark_full_file_load(n_samples: int, n_traces: int, n_runs: int = 3) -> dict:
    """
    Benchmark loading entire file through SEGYReader.

    This measures the real-world performance users experience.
    """
    with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
        path = Path(f.name)

    try:
        # Create test file
        create_test_file(path, n_samples, n_traces)
        file_size_mb = path.stat().st_size / (1024 * 1024)

        # Benchmark
        times = []
        for _ in range(n_runs):
            mapping = HeaderMapping()
            reader = SEGYReader(str(path), mapping)

            start = time.perf_counter()
            traces, headers = reader.read_all_traces()
            duration = time.perf_counter() - start
            times.append(duration)

            del traces, headers

        return {
            'n_samples': n_samples,
            'n_traces': n_traces,
            'file_size_mb': file_size_mb,
            'mean_time_sec': np.mean(times),
            'std_time_sec': np.std(times),
            'throughput_mb_per_sec': file_size_mb / np.mean(times),
            'throughput_traces_per_sec': n_traces / np.mean(times)
        }

    finally:
        path.unlink()

def run_full_load_benchmark_suite():
    """Run benchmark suite at multiple sizes."""
    sizes = [
        (1000, 1000),    # 4 MB
        (2000, 5000),    # 40 MB
        (2000, 10000),   # 80 MB
        (2000, 50000),   # 400 MB
        (2000, 100000),  # 800 MB
    ]

    results = []
    print("Full File Load Benchmark")
    print("=" * 70)
    print(f"{'Traces':>10} {'Samples':>10} {'Size MB':>10} {'Time':>10} {'MB/s':>10} {'Traces/s':>12}")
    print("-" * 70)

    for n_samples, n_traces in sizes:
        result = benchmark_full_file_load(n_samples, n_traces)
        results.append(result)
        print(f"{result['n_traces']:>10,} {result['n_samples']:>10} "
              f"{result['file_size_mb']:>10.1f} {result['mean_time_sec']:>10.2f} "
              f"{result['throughput_mb_per_sec']:>10.1f} {result['throughput_traces_per_sec']:>12,.0f}")

    print("=" * 70)
    return results

# Target metrics:
# - Throughput: > 100 MB/s on SSD
# - Linear scaling with file size
# - No significant degradation at 100K+ traces
```

### Benchmark 2.2: Chunked Load

```python
# File: tests/benchmarks/bench_2_2_chunked_load.py

import time
import gc
import tracemalloc
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping

def benchmark_chunked_load(segy_path: str, chunk_sizes: list = None) -> dict:
    """
    Benchmark chunked loading at different chunk sizes.

    Measures both throughput and memory efficiency.
    """
    if chunk_sizes is None:
        chunk_sizes = [1000, 5000, 10000, 50000]

    mapping = HeaderMapping()
    reader = SEGYReader(segy_path, mapping)
    info = reader.read_file_info()

    results = {}

    for chunk_size in chunk_sizes:
        gc.collect()
        tracemalloc.start()

        start = time.perf_counter()
        total_traces = 0

        for traces, headers, start_idx, end_idx in reader.read_traces_in_chunks(chunk_size):
            total_traces = end_idx
            # Simulate processing - just access the data
            _ = traces.mean()

        duration = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[chunk_size] = {
            'duration_sec': duration,
            'throughput_traces_per_sec': total_traces / duration,
            'peak_memory_mb': peak / 1024 / 1024,
            'memory_per_chunk_mb': peak / 1024 / 1024 / (total_traces / chunk_size)
        }

    return results

# Target metrics:
# - Memory usage proportional to chunk size
# - Peak memory < chunk_size * 2 (for double buffering)
# - Throughput within 20% of full load
```

### Benchmark 2.3: Ensemble Navigation

```python
# File: tests/benchmarks/bench_2_3_ensemble_navigation.py

import time
import numpy as np
from models.lazy_seismic_data import LazySeismicData

def benchmark_ensemble_navigation(zarr_dir: str, n_iterations: int = 100) -> dict:
    """
    Benchmark ensemble (gather) navigation performance.

    This simulates interactive gather review workflow.
    """
    lazy_data = LazySeismicData.from_storage_dir(zarr_dir)
    n_ensembles = lazy_data.get_ensemble_count()

    if n_ensembles == 0:
        return {'error': 'No ensembles in dataset'}

    # Random access pattern (simulates user navigation)
    indices = np.random.randint(0, n_ensembles, n_iterations)

    times = []
    for idx in indices:
        start = time.perf_counter()
        ensemble = lazy_data.get_ensemble(idx)
        _ = ensemble.shape  # Force load
        times.append(time.perf_counter() - start)

    return {
        'mean_time_ms': np.mean(times) * 1000,
        'median_time_ms': np.median(times) * 1000,
        'p95_time_ms': np.percentile(times, 95) * 1000,
        'p99_time_ms': np.percentile(times, 99) * 1000,
        'ensembles_per_sec': 1.0 / np.mean(times)
    }

# Target metrics:
# - Mean ensemble load: < 100ms
# - P99 ensemble load: < 500ms
# - Supports > 10 ensembles/second navigation
```

---

## 3. Scalability Benchmarks

### Benchmark 3.1: Trace Count Scaling

```python
# File: tests/benchmarks/bench_3_1_trace_scaling.py

import time
import tempfile
import numpy as np
import segyio
from pathlib import Path

def benchmark_trace_scaling():
    """
    Measure how performance scales with trace count.

    Ideal: O(n) - linear scaling
    Bad: O(n²) or worse - exponential slowdown
    """
    n_samples = 2000
    trace_counts = [1000, 5000, 10000, 50000, 100000, 500000]

    results = []

    for n_traces in trace_counts:
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            path = Path(f.name)

        try:
            # Create file
            create_test_segy(path, n_samples, n_traces)

            # Measure load time
            start = time.perf_counter()
            with segyio.open(str(path), ignore_geometry=True) as f:
                f.mmap()
                traces = np.array([f.trace[i] for i in range(n_traces)])
            duration = time.perf_counter() - start

            results.append({
                'n_traces': n_traces,
                'duration_sec': duration,
                'time_per_trace_us': duration / n_traces * 1e6,
                'normalized_time': duration / n_traces  # Should be constant
            })

        finally:
            path.unlink()

    # Analyze scaling
    first_normalized = results[0]['normalized_time']
    last_normalized = results[-1]['normalized_time']
    scaling_factor = last_normalized / first_normalized

    print(f"\nScaling Analysis:")
    print(f"  First (1K traces): {results[0]['time_per_trace_us']:.1f} us/trace")
    print(f"  Last ({results[-1]['n_traces']/1000:.0f}K traces): {results[-1]['time_per_trace_us']:.1f} us/trace")
    print(f"  Scaling factor: {scaling_factor:.2f}x")

    if scaling_factor < 1.5:
        print("  Result: EXCELLENT - Nearly linear scaling")
    elif scaling_factor < 2.0:
        print("  Result: GOOD - Acceptable scaling")
    elif scaling_factor < 3.0:
        print("  Result: WARNING - Sublinear scaling detected")
    else:
        print("  Result: CRITICAL - Severe performance degradation")

    return results, scaling_factor

# Target metrics:
# - Scaling factor < 2.0x from 1K to 500K traces
# - Per-trace time < 100 microseconds at all scales
```

### Benchmark 3.2: Concurrent Access

```python
# File: tests/benchmarks/bench_3_2_concurrent.py

import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from models.lazy_seismic_data import LazySeismicData

def benchmark_concurrent_access(zarr_dir: str, n_threads: int = 4, n_requests: int = 100) -> dict:
    """
    Benchmark concurrent access to lazy-loaded data.

    Simulates multiple UI components accessing data simultaneously.
    """
    lazy_data = LazySeismicData.from_storage_dir(zarr_dir)

    def access_window():
        """Random window access."""
        trace_start = np.random.randint(0, lazy_data.n_traces - 100)
        time_start = np.random.randint(0, int(lazy_data.duration) - 100)
        return lazy_data.get_window(
            time_start, time_start + 100,
            trace_start, trace_start + 100
        )

    # Warm-up
    for _ in range(10):
        access_window()

    # Benchmark
    times = []
    errors = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        start = time.perf_counter()
        futures = [executor.submit(access_window) for _ in range(n_requests)]

        for future in futures:
            try:
                req_start = time.perf_counter()
                future.result()
                times.append(time.perf_counter() - req_start)
            except Exception as e:
                errors.append(str(e))

        total_time = time.perf_counter() - start

    return {
        'n_threads': n_threads,
        'n_requests': n_requests,
        'total_time_sec': total_time,
        'mean_request_ms': np.mean(times) * 1000,
        'p99_request_ms': np.percentile(times, 99) * 1000,
        'requests_per_sec': n_requests / total_time,
        'error_count': len(errors)
    }

# Target metrics:
# - No errors under concurrent load
# - Mean request time < 50ms
# - > 100 requests/second with 4 threads
```

---

## 4. Memory Benchmarks

### Benchmark 4.1: Memory Efficiency

```python
# File: tests/benchmarks/bench_4_1_memory_efficiency.py

import gc
import tracemalloc
import numpy as np
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping

def benchmark_memory_efficiency(segy_path: str) -> dict:
    """
    Measure memory efficiency of loading.

    Compares actual memory usage to theoretical minimum.
    """
    gc.collect()
    tracemalloc.start()

    # Get file info first
    mapping = HeaderMapping()
    reader = SEGYReader(segy_path, mapping)
    info = reader.read_file_info()

    # Theoretical minimum: raw float32 data
    theoretical_min_mb = (info['n_samples'] * info['n_traces'] * 4) / 1024 / 1024

    # Load and measure
    snapshot_before = tracemalloc.take_snapshot()

    traces, headers = reader.read_all_traces()

    snapshot_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    actual_mb = current / 1024 / 1024
    peak_mb = peak / 1024 / 1024
    overhead_percent = ((actual_mb - theoretical_min_mb) / theoretical_min_mb) * 100

    # Analyze top allocations
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')[:10]

    return {
        'theoretical_min_mb': theoretical_min_mb,
        'actual_mb': actual_mb,
        'peak_mb': peak_mb,
        'overhead_percent': overhead_percent,
        'overhead_mb': actual_mb - theoretical_min_mb,
        'top_allocations': [(str(stat.traceback), stat.size_diff / 1024 / 1024)
                           for stat in top_stats]
    }

# Target metrics:
# - Overhead < 20% of theoretical minimum
# - Peak memory < 1.5x final memory
# - No large unexpected allocations
```

### Benchmark 4.2: Memory Stability

```python
# File: tests/benchmarks/bench_4_2_memory_stability.py

import gc
import time
import psutil
import numpy as np
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping

def benchmark_memory_stability(segy_path: str, n_iterations: int = 10) -> dict:
    """
    Test for memory leaks during repeated load/unload cycles.

    Memory should remain stable across iterations.
    """
    process = psutil.Process()
    gc.collect()
    baseline_memory = process.memory_info().rss

    memory_readings = []

    for i in range(n_iterations):
        # Load
        mapping = HeaderMapping()
        reader = SEGYReader(segy_path, mapping)
        traces, headers = reader.read_all_traces()

        # Simulate usage
        _ = traces.mean()

        # Unload
        del traces, headers, reader
        gc.collect()

        # Measure
        current_memory = process.memory_info().rss
        memory_readings.append(current_memory - baseline_memory)

        print(f"  Iteration {i+1}: {memory_readings[-1] / 1024 / 1024:.1f} MB above baseline")

    # Analyze trend
    if len(memory_readings) > 2:
        # Fit linear trend
        x = np.arange(len(memory_readings))
        slope, _ = np.polyfit(x, memory_readings, 1)
        leak_rate_mb_per_iteration = slope / 1024 / 1024

        is_stable = leak_rate_mb_per_iteration < 1.0  # Less than 1 MB/iteration

    return {
        'n_iterations': n_iterations,
        'final_overhead_mb': memory_readings[-1] / 1024 / 1024,
        'max_overhead_mb': max(memory_readings) / 1024 / 1024,
        'leak_rate_mb_per_iteration': leak_rate_mb_per_iteration,
        'is_stable': is_stable
    }

# Target metrics:
# - Leak rate < 1 MB per iteration
# - Final overhead < 10 MB
# - Memory returns to near-baseline after GC
```

---

## 5. Comparison Benchmarks

### Benchmark 5.1: Library Comparison

```python
# File: tests/benchmarks/bench_5_1_library_comparison.py

import time
import numpy as np
import tempfile
from pathlib import Path

def benchmark_library_comparison(n_samples: int = 2000, n_traces: int = 50000):
    """
    Compare performance across different SEGY libraries.

    Tests: segyio, segfast, obspy (if available)
    """
    # Create test file
    with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
        path = Path(f.name)
    create_test_segy(path, n_samples, n_traces)

    results = {}

    # segyio baseline
    import segyio
    start = time.perf_counter()
    with segyio.open(str(path), ignore_geometry=True) as f:
        f.mmap()
        traces = np.array([f.trace[i] for i in range(n_traces)])
    results['segyio'] = time.perf_counter() - start

    # segyio with collect
    start = time.perf_counter()
    with segyio.open(str(path), ignore_geometry=True) as f:
        f.mmap()
        traces = segyio.tools.collect(f.trace[:]).T
    results['segyio_collect'] = time.perf_counter() - start

    # segfast (if available)
    try:
        from segfast import MemmapLoader
        loader = MemmapLoader(str(path))
        start = time.perf_counter()
        traces = loader.load_traces(np.arange(n_traces))
        results['segfast'] = time.perf_counter() - start
    except ImportError:
        results['segfast'] = None

    # obspy (if available)
    try:
        from obspy.io.segy.core import _read_segy
        start = time.perf_counter()
        stream = _read_segy(str(path), unpack_trace_headers=False)
        results['obspy'] = time.perf_counter() - start
    except ImportError:
        results['obspy'] = None

    # Cleanup
    path.unlink()

    # Print comparison
    print(f"\nLibrary Comparison ({n_traces:,} traces, {n_samples} samples)")
    print("=" * 50)
    baseline = results['segyio']
    for lib, duration in results.items():
        if duration is not None:
            speedup = baseline / duration
            print(f"  {lib:20s}: {duration:.2f}s ({speedup:.1f}x)")
        else:
            print(f"  {lib:20s}: Not available")

    return results

# Target: segfast should be 2-10x faster than segyio
```

### Benchmark 5.2: Access Pattern Comparison

```python
# File: tests/benchmarks/bench_5_2_access_patterns.py

import time
import numpy as np
import segyio

def benchmark_access_patterns(segy_path: str, n_samples: int = 1000) -> dict:
    """
    Compare performance of different access patterns.

    Tests sequential, strided, random, and batch access.
    """
    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount

        results = {}

        # 1. Sequential access
        indices_seq = list(range(min(n_samples, n_traces)))
        start = time.perf_counter()
        for i in indices_seq:
            _ = f.trace[i]
        results['sequential'] = time.perf_counter() - start

        # 2. Strided access (every 10th trace)
        indices_strided = list(range(0, n_traces, max(1, n_traces // n_samples)))[:n_samples]
        start = time.perf_counter()
        for i in indices_strided:
            _ = f.trace[i]
        results['strided'] = time.perf_counter() - start

        # 3. Random access
        indices_random = np.random.randint(0, n_traces, n_samples).tolist()
        start = time.perf_counter()
        for i in indices_random:
            _ = f.trace[i]
        results['random'] = time.perf_counter() - start

        # 4. Batch access (using raw)
        start = time.perf_counter()
        batch = np.array([f.trace[i] for i in indices_seq])
        results['batch_list'] = time.perf_counter() - start

        # 5. Collect (most efficient for full read)
        start = time.perf_counter()
        batch = segyio.tools.collect(f.trace[0:n_samples])
        results['collect'] = time.perf_counter() - start

    # Normalize to sequential
    baseline = results['sequential']
    normalized = {k: v / baseline for k, v in results.items()}

    print(f"\nAccess Pattern Comparison ({n_samples} traces)")
    print("=" * 50)
    for pattern, norm_time in normalized.items():
        print(f"  {pattern:15s}: {norm_time:.2f}x (relative to sequential)")

    return {'times': results, 'normalized': normalized}

# Target metrics:
# - Random access < 10x slower than sequential (SSD)
# - Collect/batch should be fastest for bulk reads
```

---

## Benchmark Runner

```python
# File: tests/benchmarks/run_benchmarks.py

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

def run_all_benchmarks(segy_path: str, output_dir: str = 'benchmark_results'):
    """Run complete benchmark suite and save results."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"benchmark_{timestamp}.json"

    results = {
        'timestamp': timestamp,
        'segy_path': segy_path,
        'benchmarks': {}
    }

    # Run each benchmark
    from tests.benchmarks.bench_1_1_single_trace import benchmark_single_trace_read
    from tests.benchmarks.bench_1_2_header_read import benchmark_header_read
    from tests.benchmarks.bench_2_1_full_load import run_full_load_benchmark_suite
    from tests.benchmarks.bench_4_1_memory_efficiency import benchmark_memory_efficiency

    print("Running benchmark suite...")
    print("=" * 70)

    print("\n1. Single Trace Read")
    results['benchmarks']['single_trace'] = benchmark_single_trace_read(segy_path)

    print("\n2. Header Read")
    results['benchmarks']['header_read'] = benchmark_header_read(segy_path)

    print("\n3. Full Load Suite")
    results['benchmarks']['full_load'] = run_full_load_benchmark_suite()

    print("\n4. Memory Efficiency")
    results['benchmarks']['memory'] = benchmark_memory_efficiency(segy_path)

    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SEGY performance benchmarks")
    parser.add_argument("segy_path", help="Path to SEGY file for benchmarking")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    args = parser.parse_args()

    run_all_benchmarks(args.segy_path, args.output)
```

---

## Benchmark Results Template

```markdown
# Benchmark Results

**Date:** YYYY-MM-DD
**System:** [CPU, RAM, Storage type]
**File:** [filename, size, traces, samples]

## Summary

| Benchmark | Result | Target | Status |
|-----------|--------|--------|--------|
| Single trace read | ___ us | < 50 us | PASS/FAIL |
| Header read | ___ us | < 50 us | PASS/FAIL |
| Full load throughput | ___ MB/s | > 100 MB/s | PASS/FAIL |
| Memory overhead | ___% | < 20% | PASS/FAIL |
| Scaling factor | ___x | < 2x | PASS/FAIL |
| Random access penalty | ___x | < 10x | PASS/FAIL |

## Detailed Results

[Paste JSON results here]

## Comparison with Previous

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| ... | ... | ... | +/-% |

## Recommendations

Based on benchmark results:
1. ...
2. ...
```

---

## Running Benchmarks

```bash
# Run all benchmarks
python -m tests.benchmarks.run_benchmarks /path/to/test.sgy

# Run specific benchmark
python -m tests.benchmarks.bench_1_1_single_trace /path/to/test.sgy

# Run with profiling
python -m cProfile -s cumtime tests/benchmarks/run_benchmarks.py /path/to/test.sgy

# Compare before/after optimization
python -m tests.benchmarks.compare_results before.json after.json
```
