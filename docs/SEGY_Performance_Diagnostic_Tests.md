# SEGY Performance Diagnostic Tests

## Overview

This document provides a comprehensive set of diagnostic tests to identify performance bottlenecks in SEGY loading operations. Execute these tests systematically to pinpoint the specific causes of slowdown in your environment.

---

## Test Categories

1. [Baseline Performance Tests](#1-baseline-performance-tests)
2. [Memory Profiling Tests](#2-memory-profiling-tests)
3. [I/O Pattern Analysis](#3-io-pattern-analysis)
4. [Format Conversion Overhead Tests](#4-format-conversion-overhead-tests)
5. [Library Comparison Tests](#5-library-comparison-tests)
6. [Scalability Tests](#6-scalability-tests)

---

## 1. Baseline Performance Tests

### Test 1.1: Current Implementation Throughput

**Objective:** Measure current loading speed at various file sizes.

```python
# File: tests/diagnostics/test_1_1_baseline_throughput.py

import time
import numpy as np
import tempfile
import segyio
from pathlib import Path
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping

def test_baseline_throughput():
    """Measure baseline loading throughput."""

    sizes = [
        (1000, 1000),      # 1K traces
        (1000, 10000),     # 10K traces
        (1000, 100000),    # 100K traces
        (1000, 500000),    # 500K traces
        (1000, 1000000),   # 1M traces (if memory allows)
    ]

    results = []

    for n_samples, n_traces in sizes:
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            segy_path = Path(f.name)

        create_test_segy(segy_path, n_samples, n_traces)
        file_size_mb = segy_path.stat().st_size / (1024 * 1024)

        # Measure loading time
        mapping = HeaderMapping()
        reader = SEGYReader(str(segy_path), mapping)

        start = time.perf_counter()
        traces, headers = reader.read_all_traces()
        duration = time.perf_counter() - start

        throughput = file_size_mb / duration
        traces_per_sec = n_traces / duration

        results.append({
            'n_traces': n_traces,
            'file_size_mb': file_size_mb,
            'duration_sec': duration,
            'throughput_mb_s': throughput,
            'traces_per_sec': traces_per_sec
        })

        # Cleanup
        segy_path.unlink()
        del traces, headers

        print(f"Traces: {n_traces:,}, Size: {file_size_mb:.1f}MB, "
              f"Duration: {duration:.2f}s, Throughput: {throughput:.1f}MB/s")

    # Check for degradation
    if len(results) >= 3:
        first_throughput = results[0]['throughput_mb_s']
        last_throughput = results[-1]['throughput_mb_s']
        degradation = (first_throughput - last_throughput) / first_throughput * 100

        print(f"\nPerformance degradation: {degradation:.1f}%")
        if degradation > 20:
            print("WARNING: Significant performance degradation detected!")

    return results

def create_test_segy(path: Path, n_samples: int, n_traces: int):
    """Create test SEGY file."""
    np.random.seed(42)
    traces = np.random.randn(n_samples, n_traces).astype(np.float32)

    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(str(path), spec) as f:
        for i in range(n_traces):
            f.trace[i] = traces[:, i]
            f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = i + 1

if __name__ == "__main__":
    test_baseline_throughput()
```

**Expected Output:**
- Throughput should remain relatively constant across file sizes
- Degradation > 20% indicates performance issues

**Pass Criteria:**
- [ ] Throughput degradation < 20% from 1K to 100K traces
- [ ] Throughput > 50 MB/s on SSD storage
- [ ] No memory errors during test

---

### Test 1.2: Memory Mapping Effectiveness

**Objective:** Compare performance with and without memory mapping.

```python
# File: tests/diagnostics/test_1_2_mmap_effectiveness.py

import time
import segyio
import numpy as np
from pathlib import Path

def test_mmap_effectiveness(segy_path: str, n_iterations: int = 3):
    """Compare read performance with and without mmap."""

    results = {'without_mmap': [], 'with_mmap': []}

    for iteration in range(n_iterations):
        # Without mmap
        with segyio.open(segy_path, ignore_geometry=True) as f:
            start = time.perf_counter()
            traces = [f.trace[i] for i in range(f.tracecount)]
            duration_no_mmap = time.perf_counter() - start
            results['without_mmap'].append(duration_no_mmap)

        # With mmap
        with segyio.open(segy_path, ignore_geometry=True) as f:
            f.mmap()
            start = time.perf_counter()
            traces = [f.trace[i] for i in range(f.tracecount)]
            duration_mmap = time.perf_counter() - start
            results['with_mmap'].append(duration_mmap)

    avg_no_mmap = np.mean(results['without_mmap'])
    avg_mmap = np.mean(results['with_mmap'])
    improvement = (avg_no_mmap - avg_mmap) / avg_no_mmap * 100

    print(f"Without mmap: {avg_no_mmap:.3f}s (avg)")
    print(f"With mmap:    {avg_mmap:.3f}s (avg)")
    print(f"Improvement:  {improvement:.1f}%")

    return {
        'mmap_improvement_percent': improvement,
        'mmap_effective': improvement > 10
    }
```

**Pass Criteria:**
- [ ] mmap provides > 10% improvement
- [ ] No mmap-related errors on large files

---

## 2. Memory Profiling Tests

### Test 2.1: Memory Growth During Loading

**Objective:** Track memory allocation patterns during trace loading.

```python
# File: tests/diagnostics/test_2_1_memory_growth.py

import tracemalloc
import gc
import numpy as np
import segyio
from pathlib import Path

def test_memory_growth(segy_path: str, checkpoint_interval: int = 1000):
    """Track memory growth during loading."""

    tracemalloc.start()
    gc.collect()

    checkpoints = []
    baseline = tracemalloc.get_traced_memory()[0]

    with segyio.open(segy_path, ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)

        traces_list = []
        headers_list = []

        for i in range(n_traces):
            traces_list.append(f.trace[i].copy())
            headers_list.append(dict(f.header[i]))

            if (i + 1) % checkpoint_interval == 0:
                current, peak = tracemalloc.get_traced_memory()
                checkpoints.append({
                    'trace_num': i + 1,
                    'current_mb': (current - baseline) / 1024 / 1024,
                    'peak_mb': peak / 1024 / 1024
                })

    tracemalloc.stop()

    # Analyze growth pattern
    if len(checkpoints) >= 2:
        expected_linear = checkpoints[-1]['current_mb'] / len(checkpoints)
        actual_growth_rate = []
        for i in range(1, len(checkpoints)):
            rate = checkpoints[i]['current_mb'] - checkpoints[i-1]['current_mb']
            actual_growth_rate.append(rate)

        avg_rate = np.mean(actual_growth_rate)
        rate_variance = np.var(actual_growth_rate)

        print(f"Memory checkpoints: {len(checkpoints)}")
        print(f"Final memory: {checkpoints[-1]['current_mb']:.1f} MB")
        print(f"Peak memory: {checkpoints[-1]['peak_mb']:.1f} MB")
        print(f"Avg growth rate: {avg_rate:.2f} MB per {checkpoint_interval} traces")
        print(f"Growth variance: {rate_variance:.4f}")

        # Check for superlinear growth (fragmentation indicator)
        if rate_variance > (avg_rate * 0.5) ** 2:
            print("WARNING: Non-linear memory growth detected (possible fragmentation)")

    return checkpoints
```

**Pass Criteria:**
- [ ] Memory growth is approximately linear
- [ ] Peak memory < 2x final memory (low fragmentation)
- [ ] Growth rate variance < (avg_rate * 0.5)^2

---

### Test 2.2: Object Overhead Analysis

**Objective:** Measure Python object overhead per trace.

```python
# File: tests/diagnostics/test_2_2_object_overhead.py

import sys
import numpy as np

def test_object_overhead():
    """Measure Python object overhead for trace storage."""

    n_samples = 2000
    n_test_traces = 1000

    # Raw numpy array (expected efficient storage)
    raw_array = np.random.randn(n_samples, n_test_traces).astype(np.float32)
    raw_size = raw_array.nbytes

    # List of numpy arrays (common but inefficient)
    list_of_arrays = [np.random.randn(n_samples).astype(np.float32)
                      for _ in range(n_test_traces)]
    list_size = sum(arr.nbytes for arr in list_of_arrays)
    list_overhead = sum(sys.getsizeof(arr) - arr.nbytes for arr in list_of_arrays)
    list_overhead += sys.getsizeof(list_of_arrays)

    # List of tuples (trace + header dict)
    list_of_tuples = [
        (np.random.randn(n_samples).astype(np.float32), {'cdp': i, 'offset': i * 100})
        for i in range(n_test_traces)
    ]
    tuple_overhead = sum(sys.getsizeof(t) + sys.getsizeof(t[1]) for t in list_of_tuples)

    print(f"Data size per trace: {n_samples * 4} bytes")
    print(f"\nStorage patterns for {n_test_traces} traces:")
    print(f"  Raw 2D array:     {raw_size / 1024:.1f} KB (baseline)")
    print(f"  List of arrays:   {(list_size + list_overhead) / 1024:.1f} KB "
          f"(overhead: {list_overhead / 1024:.1f} KB, {list_overhead / list_size * 100:.1f}%)")
    print(f"  List of tuples:   overhead: {tuple_overhead / 1024:.1f} KB additional")

    overhead_percent = (list_overhead / raw_size) * 100

    return {
        'overhead_percent': overhead_percent,
        'acceptable': overhead_percent < 5
    }
```

**Pass Criteria:**
- [ ] Object overhead < 5% of raw data size
- [ ] Header dict overhead measured and documented

---

### Test 2.3: Garbage Collection Impact

**Objective:** Measure GC pause times during loading.

```python
# File: tests/diagnostics/test_2_3_gc_impact.py

import gc
import time
import segyio
from typing import List, Tuple

def test_gc_impact(segy_path: str) -> dict:
    """Measure garbage collection impact during loading."""

    gc.collect()
    gc.disable()  # Disable automatic GC

    gc_times = []
    load_times = []

    with segyio.open(segy_path, ignore_geometry=True) as f:
        n_traces = f.tracecount
        batch_size = 10000
        traces = []

        for batch_start in range(0, n_traces, batch_size):
            batch_end = min(batch_start + batch_size, n_traces)

            # Load batch
            load_start = time.perf_counter()
            for i in range(batch_start, batch_end):
                traces.append(f.trace[i].copy())
            load_time = time.perf_counter() - load_start
            load_times.append(load_time)

            # Manual GC and measure
            gc_start = time.perf_counter()
            gc.collect()
            gc_time = time.perf_counter() - gc_start
            gc_times.append(gc_time)

    gc.enable()

    total_load = sum(load_times)
    total_gc = sum(gc_times)
    gc_percent = total_gc / (total_load + total_gc) * 100

    print(f"Total load time: {total_load:.2f}s")
    print(f"Total GC time:   {total_gc:.2f}s")
    print(f"GC overhead:     {gc_percent:.1f}%")
    print(f"Max GC pause:    {max(gc_times) * 1000:.1f}ms")

    return {
        'gc_percent': gc_percent,
        'max_gc_pause_ms': max(gc_times) * 1000,
        'acceptable': gc_percent < 10 and max(gc_times) < 0.5
    }
```

**Pass Criteria:**
- [ ] GC overhead < 10% of total time
- [ ] Max GC pause < 500ms

---

## 3. I/O Pattern Analysis

### Test 3.1: Sequential vs Random Access

**Objective:** Compare sequential and random trace access patterns.

```python
# File: tests/diagnostics/test_3_1_access_patterns.py

import time
import numpy as np
import segyio

def test_access_patterns(segy_path: str, sample_size: int = 1000):
    """Compare sequential vs random access performance."""

    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount

        # Sequential access
        indices_seq = list(range(min(sample_size, n_traces)))
        start = time.perf_counter()
        for i in indices_seq:
            _ = f.trace[i]
        time_sequential = time.perf_counter() - start

        # Random access (shuffled indices)
        indices_random = indices_seq.copy()
        np.random.shuffle(indices_random)
        start = time.perf_counter()
        for i in indices_random:
            _ = f.trace[i]
        time_random = time.perf_counter() - start

        # Strided access (simulate crossline extraction)
        stride = max(1, n_traces // sample_size)
        indices_strided = list(range(0, n_traces, stride))[:sample_size]
        start = time.perf_counter()
        for i in indices_strided:
            _ = f.trace[i]
        time_strided = time.perf_counter() - start

    print(f"Access pattern comparison ({sample_size} traces):")
    print(f"  Sequential: {time_sequential:.3f}s ({sample_size/time_sequential:.0f} traces/s)")
    print(f"  Random:     {time_random:.3f}s ({sample_size/time_random:.0f} traces/s)")
    print(f"  Strided:    {time_strided:.3f}s ({sample_size/time_strided:.0f} traces/s)")
    print(f"  Random/Sequential ratio: {time_random/time_sequential:.1f}x slower")

    return {
        'sequential_traces_per_sec': sample_size / time_sequential,
        'random_traces_per_sec': sample_size / time_random,
        'random_penalty_factor': time_random / time_sequential
    }
```

**Pass Criteria:**
- [ ] Random access < 10x slower than sequential (on SSD)
- [ ] Random access < 100x slower than sequential (on HDD)

---

### Test 3.2: Buffer Size Optimization

**Objective:** Find optimal read buffer size for traces.

```python
# File: tests/diagnostics/test_3_2_buffer_optimization.py

import time
import numpy as np
import segyio

def test_buffer_sizes(segy_path: str):
    """Test different buffer sizes for batch reading."""

    batch_sizes = [1, 10, 50, 100, 500, 1000, 5000]
    results = []

    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        n_samples = len(f.samples)

        for batch_size in batch_sizes:
            if batch_size > n_traces:
                continue

            n_batches = min(100, n_traces // batch_size)

            start = time.perf_counter()
            for batch_idx in range(n_batches):
                start_trace = batch_idx * batch_size
                # Pre-allocate buffer
                buffer = np.zeros((n_samples, batch_size), dtype=np.float32)
                for i in range(batch_size):
                    buffer[:, i] = f.trace[start_trace + i]
            duration = time.perf_counter() - start

            traces_read = n_batches * batch_size
            throughput = traces_read / duration

            results.append({
                'batch_size': batch_size,
                'throughput': throughput
            })
            print(f"  Batch size {batch_size:5d}: {throughput:.0f} traces/s")

    # Find optimal
    optimal = max(results, key=lambda x: x['throughput'])
    print(f"\nOptimal batch size: {optimal['batch_size']} traces")

    return results
```

**Pass Criteria:**
- [ ] Optimal batch size identified
- [ ] Throughput > 10,000 traces/s with optimal batch

---

## 4. Format Conversion Overhead Tests

### Test 4.1: IBM Float vs IEEE Float

**Objective:** Measure IBM to IEEE float conversion overhead.

```python
# File: tests/diagnostics/test_4_1_float_conversion.py

import time
import tempfile
import numpy as np
import segyio
from pathlib import Path

def test_float_conversion():
    """Compare loading speed for IBM float vs IEEE float files."""

    n_samples, n_traces = 2000, 10000

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create IBM float file (format=1)
        ibm_path = tmpdir / "ibm_float.sgy"
        create_segy_with_format(ibm_path, n_samples, n_traces, format_code=1)

        # Create IEEE float file (format=5)
        ieee_path = tmpdir / "ieee_float.sgy"
        create_segy_with_format(ieee_path, n_samples, n_traces, format_code=5)

        # Benchmark IBM float
        start = time.perf_counter()
        with segyio.open(str(ibm_path), ignore_geometry=True) as f:
            f.mmap()
            for i in range(n_traces):
                _ = f.trace[i]
        ibm_time = time.perf_counter() - start

        # Benchmark IEEE float
        start = time.perf_counter()
        with segyio.open(str(ieee_path), ignore_geometry=True) as f:
            f.mmap()
            for i in range(n_traces):
                _ = f.trace[i]
        ieee_time = time.perf_counter() - start

    conversion_overhead = (ibm_time - ieee_time) / ieee_time * 100

    print(f"IBM float loading:  {ibm_time:.2f}s")
    print(f"IEEE float loading: {ieee_time:.2f}s")
    print(f"Conversion overhead: {conversion_overhead:.1f}%")

    return {
        'ibm_time': ibm_time,
        'ieee_time': ieee_time,
        'conversion_overhead_percent': conversion_overhead
    }

def create_segy_with_format(path: Path, n_samples: int, n_traces: int, format_code: int):
    """Create SEGY file with specified format."""
    np.random.seed(42)
    traces = np.random.randn(n_samples, n_traces).astype(np.float32)

    spec = segyio.spec()
    spec.format = format_code
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(str(path), spec) as f:
        for i in range(n_traces):
            f.trace[i] = traces[:, i]
```

**Pass Criteria:**
- [ ] IBM conversion overhead < 50%
- [ ] Both formats complete without errors

---

### Test 4.2: Header Parsing Overhead

**Objective:** Measure time spent parsing trace headers.

```python
# File: tests/diagnostics/test_4_2_header_parsing.py

import time
import segyio
from utils.segy_import.header_mapping import HeaderMapping

def test_header_parsing_overhead(segy_path: str):
    """Measure header parsing time separately from trace loading."""

    mapping = HeaderMapping()

    with segyio.open(segy_path, ignore_geometry=True) as f:
        n_traces = f.tracecount

        # Trace data only
        start = time.perf_counter()
        for i in range(n_traces):
            _ = f.trace[i]
        trace_only_time = time.perf_counter() - start

        # Headers only (raw access)
        start = time.perf_counter()
        for i in range(n_traces):
            _ = f.header[i]
        header_raw_time = time.perf_counter() - start

        # Headers with parsing
        start = time.perf_counter()
        for i in range(n_traces):
            header_dict = f.header[i]
            header_bytes = bytes(header_dict.buf)
            _ = mapping.read_headers(header_bytes, trace_idx=i)
        header_parsed_time = time.perf_counter() - start

    parsing_overhead = header_parsed_time - header_raw_time
    total_with_headers = trace_only_time + header_parsed_time
    header_percent = header_parsed_time / total_with_headers * 100

    print(f"Trace data loading:   {trace_only_time:.2f}s")
    print(f"Header raw access:    {header_raw_time:.2f}s")
    print(f"Header with parsing:  {header_parsed_time:.2f}s")
    print(f"Parsing overhead:     {parsing_overhead:.2f}s")
    print(f"Header % of total:    {header_percent:.1f}%")

    return {
        'header_percent': header_percent,
        'parsing_overhead_sec': parsing_overhead
    }
```

**Pass Criteria:**
- [ ] Header parsing < 30% of total time
- [ ] Per-header parsing < 10 microseconds

---

## 5. Library Comparison Tests

### Test 5.1: segyio vs segfast

**Objective:** Compare segyio with segfast library.

```python
# File: tests/diagnostics/test_5_1_library_comparison.py

import time
import numpy as np

def test_library_comparison(segy_path: str):
    """Compare segyio with segfast for trace loading."""

    import segyio

    results = {}

    # Test segyio
    with segyio.open(segy_path, ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount

        start = time.perf_counter()
        traces = np.array([f.trace[i] for i in range(n_traces)])
        results['segyio_time'] = time.perf_counter() - start
        results['segyio_shape'] = traces.shape

    # Test segfast (if available)
    try:
        from segfast import MemmapLoader

        loader = MemmapLoader(segy_path)
        indices = np.arange(n_traces)

        start = time.perf_counter()
        traces = loader.load_traces(indices)
        results['segfast_time'] = time.perf_counter() - start
        results['segfast_shape'] = traces.shape

        speedup = results['segyio_time'] / results['segfast_time']
        results['speedup'] = speedup

        print(f"segyio:  {results['segyio_time']:.2f}s")
        print(f"segfast: {results['segfast_time']:.2f}s")
        print(f"Speedup: {speedup:.1f}x")

    except ImportError:
        print("segfast not installed - install with: pip install segfast")
        results['segfast_available'] = False

    return results
```

**Pass Criteria:**
- [ ] segfast provides > 2x speedup
- [ ] Results are numerically equivalent

---

## 6. Scalability Tests

### Test 6.1: Trace Count Scaling

**Objective:** Measure performance scaling with trace count.

```python
# File: tests/diagnostics/test_6_1_scalability.py

import time
import numpy as np
import tempfile
import segyio
from pathlib import Path

def test_trace_count_scaling():
    """Measure how loading time scales with trace count."""

    n_samples = 2000
    trace_counts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    results = []

    for n_traces in trace_counts:
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            path = Path(f.name)

        try:
            # Create file
            create_test_segy(path, n_samples, n_traces)

            # Load and time
            with segyio.open(str(path), ignore_geometry=True) as f:
                f.mmap()

                start = time.perf_counter()
                traces = np.zeros((n_samples, n_traces), dtype=np.float32)
                for i in range(n_traces):
                    traces[:, i] = f.trace[i]
                duration = time.perf_counter() - start

            time_per_trace_us = duration / n_traces * 1e6

            results.append({
                'n_traces': n_traces,
                'duration': duration,
                'time_per_trace_us': time_per_trace_us
            })

            print(f"Traces: {n_traces:>8,}, Duration: {duration:>6.2f}s, "
                  f"Per trace: {time_per_trace_us:.1f}us")

        finally:
            path.unlink()

    # Analyze scaling
    if len(results) >= 3:
        # Linear scaling: time_per_trace should be constant
        times = [r['time_per_trace_us'] for r in results]
        scaling_factor = times[-1] / times[0]

        print(f"\nScaling factor (last/first per-trace time): {scaling_factor:.2f}x")
        if scaling_factor > 2.0:
            print("WARNING: Super-linear scaling detected!")
        elif scaling_factor < 0.5:
            print("INFO: Sub-linear scaling (good cache utilization)")

    return results
```

**Pass Criteria:**
- [ ] Per-trace time < 100 microseconds
- [ ] Scaling factor < 2.0x from 10K to 1M traces

---

### Test 6.2: Memory Stress Test

**Objective:** Test behavior under memory pressure.

```python
# File: tests/diagnostics/test_6_2_memory_stress.py

import psutil
import numpy as np
import segyio
import gc

def test_memory_stress(segy_path: str, target_memory_percent: float = 80):
    """Test loading behavior under memory pressure."""

    process = psutil.Process()
    initial_memory = process.memory_info().rss
    available_memory = psutil.virtual_memory().available

    print(f"Initial process memory: {initial_memory / 1024**3:.2f} GB")
    print(f"Available system memory: {available_memory / 1024**3:.2f} GB")

    # Create memory pressure
    target_usage = int(available_memory * target_memory_percent / 100)
    pressure_array = np.zeros(target_usage // 8, dtype=np.float64)

    after_pressure = psutil.virtual_memory().available
    print(f"Available after pressure: {after_pressure / 1024**3:.2f} GB")

    # Now try to load SEGY
    try:
        with segyio.open(segy_path, ignore_geometry=True) as f:
            f.mmap()
            n_traces = min(f.tracecount, 100000)

            traces = []
            for i in range(n_traces):
                traces.append(f.trace[i].copy())

                if i % 10000 == 0:
                    current_mem = process.memory_info().rss
                    print(f"  Loaded {i:,} traces, process memory: "
                          f"{current_mem / 1024**3:.2f} GB")

        result = {'success': True, 'traces_loaded': n_traces}

    except MemoryError as e:
        result = {'success': False, 'error': str(e)}
        print(f"MemoryError at trace load: {e}")

    finally:
        del pressure_array
        gc.collect()

    return result
```

**Pass Criteria:**
- [ ] Graceful degradation under memory pressure
- [ ] No crashes or data corruption

---

## Diagnostic Summary Template

After running all tests, fill in this summary:

```markdown
## Diagnostic Summary

**Date:** YYYY-MM-DD
**File Tested:** [filename]
**File Size:** [X] MB
**Trace Count:** [N] traces

### Key Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Baseline throughput | ___ MB/s | > 50 MB/s | PASS/FAIL |
| mmap improvement | ___% | > 10% | PASS/FAIL |
| Memory growth | ___ | Linear | PASS/FAIL |
| GC overhead | ___% | < 10% | PASS/FAIL |
| Random access penalty | ___x | < 10x | PASS/FAIL |
| IBM float overhead | ___% | < 50% | PASS/FAIL |
| Header parsing % | ___% | < 30% | PASS/FAIL |
| Scaling factor | ___x | < 2x | PASS/FAIL |

### Identified Bottlenecks

1. [ ] Memory fragmentation
2. [ ] I/O pattern (random access)
3. [ ] Format conversion (IBM float)
4. [ ] Header parsing overhead
5. [ ] GC pauses
6. [ ] Buffer size mismatch

### Recommended Optimizations

Based on diagnostics, prioritize:
1. ___
2. ___
3. ___
```

---

## Running All Tests

```bash
# Run all diagnostic tests
python -m pytest tests/diagnostics/ -v --tb=short

# Run with profiling
python -m cProfile -o profile.stats tests/diagnostics/run_all.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime').print_stats(20)"
```
