# SEGY Performance Implementation Tasks

## Overview

This document provides a prioritized list of implementation tasks to accelerate SEGY loading in the SeisProc application. Tasks are organized by complexity and expected impact.

---

## Task Categories

1. [Quick Wins (Low Effort, High Impact)](#1-quick-wins)
2. [Core Optimizations](#2-core-optimizations)
3. [Architecture Changes](#3-architecture-changes)
4. [Advanced Optimizations](#4-advanced-optimizations)

---

## 1. Quick Wins

### Task 1.1: Enable Memory Mapping in SEGYReader

**Priority:** HIGH | **Effort:** 1 hour | **Impact:** 10-50% speedup

**Current State:** `segy_reader.py` opens files without mmap

**File:** `utils/segy_import/segy_reader.py`

**Changes Required:**

```python
# In SEGYFileHandle.__enter__() - Line 79-81
def __enter__(self):
    """Open the SEG-Y file and return self."""
    self._file = segyio.open(str(self.filename), 'r', ignore_geometry=True)
    self._file.mmap()  # ADD THIS LINE
    return self
```

```python
# In read_all_traces() - Line 289
with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
    f.mmap()  # ADD THIS LINE
    n_traces = min(f.tracecount, max_traces) if max_traces else f.tracecount
```

```python
# In read_traces_in_chunks() - Line 449
with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
    f.mmap()  # ADD THIS LINE
    n_traces = f.tracecount
```

**Verification Test:**
```python
def test_mmap_enabled():
    """Verify mmap is being used."""
    from utils.segy_import.segy_reader import SEGYReader, SEGYFileHandle
    # Check that mmap is called after open
    pass
```

**Acceptance Criteria:**
- [ ] mmap() called in all file open operations
- [ ] Benchmark shows improvement
- [ ] No regressions in existing tests

---

### Task 1.2: Pre-allocate Trace Arrays

**Priority:** HIGH | **Effort:** 2 hours | **Impact:** 20-40% speedup

**Current State:** Traces appended to list, then potentially converted

**File:** `utils/segy_import/segy_reader.py`

**Changes Required:**

```python
# Replace in read_all_traces() - Lines 293-306
# BEFORE:
traces = np.zeros((n_samples, n_traces), dtype=np.float32)
for i in range(n_traces):
    traces[:, i] = f.trace[i]

# AFTER (optimized):
# Pre-allocate with exact dimensions
traces = np.empty((n_samples, n_traces), dtype=np.float32)

# Use raw trace access for speed
for i in range(n_traces):
    f.trace.raw[i, traces[:, i]]  # Direct write to pre-allocated buffer
```

**Alternative using segyio's built-in:**
```python
# Most efficient - single read operation
traces = segyio.tools.collect(f.trace[:]).T.astype(np.float32)
```

**Verification Test:**
```python
def test_preallocated_arrays():
    """Verify arrays are pre-allocated, not grown."""
    import tracemalloc
    tracemalloc.start()
    # Load traces
    current, peak = tracemalloc.get_traced_memory()
    # Peak should be close to current (no intermediate allocations)
    assert peak < current * 1.2
```

**Acceptance Criteria:**
- [ ] No list append operations during loading
- [ ] Peak memory < 1.2x final memory
- [ ] Benchmark shows improvement

---

### Task 1.3: Batch Header Reading

**Priority:** MEDIUM | **Effort:** 2 hours | **Impact:** 15-30% speedup

**Current State:** Headers read one at a time with dict conversion

**File:** `utils/segy_import/segy_reader.py`

**Changes Required:**

```python
# Add new method to SEGYFileHandle class
def read_headers_batch(self, start: int, end: int, fields: List[int]) -> np.ndarray:
    """
    Read specific header fields for a range of traces efficiently.

    Args:
        start: Start trace index
        end: End trace index
        fields: List of segyio.TraceField values to extract

    Returns:
        2D array of shape (end-start, len(fields))
    """
    n_traces = end - start
    result = np.empty((n_traces, len(fields)), dtype=np.int32)

    # Use segyio's attributes for efficient batch access
    for i, field in enumerate(fields):
        result[:, i] = self.file.attributes(field)[start:end]

    return result
```

```python
# Update read_headers_range to use batch method where possible
def read_headers_range(self, start: int, end: int) -> List[Dict]:
    """Read headers for a range of traces."""
    f = self.file

    # Get field mappings
    mappings = self.header_mapping.get_all_mappings()
    fields = list(mappings.values())
    names = list(mappings.keys())

    # Batch read all fields
    batch_data = np.empty((end - start, len(fields)), dtype=np.int32)
    for i, field in enumerate(fields):
        try:
            batch_data[:, i] = f.attributes(field)[start:end]
        except (KeyError, IndexError):
            batch_data[:, i] = 0

    # Convert to list of dicts
    headers = []
    for row in batch_data:
        header = dict(zip(names, row))
        headers.append(header)

    return headers
```

**Verification Test:**
```python
def test_batch_header_reading():
    """Verify batch header reading is faster than sequential."""
    import time
    # Compare batch vs sequential header reading
    pass
```

**Acceptance Criteria:**
- [ ] Batch method implemented
- [ ] Header reading 2x faster
- [ ] Results match sequential method

---

## 2. Core Optimizations

### Task 2.1: Implement segfast Integration

**Priority:** HIGH | **Effort:** 4 hours | **Impact:** 2-10x speedup

**Description:** Add optional segfast backend for trace loading.

**New File:** `utils/segy_import/segy_reader_fast.py`

```python
"""
Fast SEGY reader using segfast library.
Falls back to segyio if segfast not available.
"""
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path

try:
    from segfast import MemmapLoader
    SEGFAST_AVAILABLE = True
except ImportError:
    SEGFAST_AVAILABLE = False

from utils.segy_import.segy_reader import SEGYReader, CancellationToken
from utils.segy_import.header_mapping import HeaderMapping


class FastSEGYReader:
    """
    High-performance SEGY reader using segfast.

    Falls back to standard SEGYReader if segfast is not installed.
    """

    def __init__(self, filename: str, header_mapping: HeaderMapping):
        """Initialize reader with optional segfast backend."""
        self.filename = Path(filename)
        self.header_mapping = header_mapping
        self._use_segfast = SEGFAST_AVAILABLE

        if not self.filename.exists():
            raise FileNotFoundError(f"SEG-Y file not found: {filename}")

        if self._use_segfast:
            self._loader = MemmapLoader(str(self.filename))
        else:
            self._fallback = SEGYReader(str(filename), header_mapping)

    @property
    def using_segfast(self) -> bool:
        """Check if segfast backend is being used."""
        return self._use_segfast

    def read_file_info(self) -> Dict[str, any]:
        """Read basic file information."""
        if self._use_segfast:
            return {
                'filename': self.filename.name,
                'n_traces': self._loader.n_traces,
                'n_samples': self._loader.n_samples,
                'sample_interval': self._loader.sample_interval,
                'format': 'segfast'
            }
        return self._fallback.read_file_info()

    def read_all_traces(
        self,
        max_traces: Optional[int] = None,
        cancellation_token: Optional[CancellationToken] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Read all traces efficiently."""
        if not self._use_segfast:
            return self._fallback.read_all_traces(
                max_traces, cancellation_token, progress_callback
            )

        n_traces = self._loader.n_traces
        if max_traces:
            n_traces = min(n_traces, max_traces)

        indices = np.arange(n_traces)

        # Check for cancellation
        if cancellation_token and cancellation_token.is_cancelled:
            from utils.segy_import.segy_reader import OperationCancelledError
            raise OperationCancelledError("Operation cancelled")

        # Load traces in one batch
        traces = self._loader.load_traces(indices)

        # Transpose to (n_samples, n_traces) format
        if traces.shape[0] == n_traces:
            traces = traces.T

        # Load headers
        headers = self._load_headers_fast(indices)

        if progress_callback:
            progress_callback(n_traces, n_traces)

        return traces.astype(np.float32), headers

    def read_traces_chunked(
        self,
        chunk_size: int = 10000,
        cancellation_token: Optional[CancellationToken] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """Stream traces in chunks."""
        if not self._use_segfast:
            yield from self._fallback.read_traces_in_chunks(
                chunk_size, cancellation_token, progress_callback
            )
            return

        n_traces = self._loader.n_traces

        for start in range(0, n_traces, chunk_size):
            if cancellation_token and cancellation_token.is_cancelled:
                break

            end = min(start + chunk_size, n_traces)
            indices = np.arange(start, end)

            traces = self._loader.load_traces(indices)
            if traces.shape[0] == len(indices):
                traces = traces.T

            headers = self._load_headers_fast(indices)

            if progress_callback:
                progress_callback(end, n_traces)

            yield traces.astype(np.float32), headers, start, end

    def _load_headers_fast(self, indices: np.ndarray) -> List[Dict]:
        """Load headers for given trace indices."""
        headers_df = self._loader.load_headers(indices)

        # Convert to list of dicts matching expected format
        headers = []
        mapping = self.header_mapping.get_all_mappings()

        for idx in range(len(indices)):
            header = {}
            for name, byte_loc in mapping.items():
                col_name = self._byte_to_column_name(byte_loc)
                if col_name in headers_df.columns:
                    header[name] = int(headers_df.iloc[idx][col_name])
                else:
                    header[name] = 0
            headers.append(header)

        return headers

    def _byte_to_column_name(self, byte_loc: int) -> str:
        """Convert byte location to segfast column name."""
        # segfast uses standard segyio field names
        import segyio
        for field in segyio.TraceField.enums():
            if int(field) == byte_loc:
                return field.name
        return f"byte_{byte_loc}"
```

**Update factory function:**
```python
# In utils/segy_import/__init__.py or segy_reader.py
def create_segy_reader(filename: str, header_mapping: HeaderMapping,
                       prefer_fast: bool = True) -> 'SEGYReader':
    """
    Create appropriate SEGY reader based on availability.

    Args:
        filename: Path to SEGY file
        header_mapping: Header mapping configuration
        prefer_fast: If True, use FastSEGYReader when available

    Returns:
        SEGY reader instance
    """
    if prefer_fast:
        try:
            from utils.segy_import.segy_reader_fast import FastSEGYReader
            return FastSEGYReader(filename, header_mapping)
        except ImportError:
            pass

    return SEGYReader(filename, header_mapping)
```

**Acceptance Criteria:**
- [ ] FastSEGYReader implemented
- [ ] Automatic fallback to segyio
- [ ] 2-10x speedup when segfast available
- [ ] All existing tests pass

---

### Task 2.2: Optimize Chunked Processing

**Priority:** HIGH | **Effort:** 3 hours | **Impact:** 30-50% speedup

**File:** `utils/segy_import/segy_reader.py`

**Changes Required:**

```python
def read_traces_in_chunks_optimized(
    self,
    chunk_size: int = 10000,
    cancellation_token: Optional[CancellationToken] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    buffer_pool_size: int = 2
):
    """
    Optimized chunked reading with buffer pooling.

    Uses pre-allocated buffer pool to avoid repeated allocations.
    """
    with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        n_samples = len(f.samples)

        # Pre-allocate buffer pool
        buffers = [
            np.empty((n_samples, chunk_size), dtype=np.float32)
            for _ in range(buffer_pool_size)
        ]
        buffer_idx = 0

        for start_idx in range(0, n_traces, chunk_size):
            if cancellation_token and cancellation_token.is_cancelled:
                raise OperationCancelledError("Cancelled")

            end_idx = min(start_idx + chunk_size, n_traces)
            current_chunk_size = end_idx - start_idx

            # Use buffer from pool
            buffer = buffers[buffer_idx]
            buffer_idx = (buffer_idx + 1) % buffer_pool_size

            # Read directly into buffer
            for i in range(current_chunk_size):
                buffer[:, i] = f.trace[start_idx + i]

            # Slice to actual size (no copy if same size)
            traces_chunk = buffer[:, :current_chunk_size]

            # Batch header reading
            headers_chunk = self._read_headers_batch(f, start_idx, end_idx)

            if progress_callback:
                progress_callback(end_idx, n_traces)

            yield traces_chunk, headers_chunk, start_idx, end_idx

def _read_headers_batch(self, f, start: int, end: int) -> List[Dict]:
    """Read headers in batch using attributes."""
    mappings = self.header_mapping.get_all_mappings()
    n_traces = end - start
    headers = [{} for _ in range(n_traces)]

    for name, byte_loc in mappings.items():
        try:
            values = f.attributes(byte_loc)[start:end]
            for i, val in enumerate(values):
                headers[i][name] = int(val)
        except (KeyError, IndexError):
            for i in range(n_traces):
                headers[i][name] = 0

    return headers
```

**Acceptance Criteria:**
- [ ] Buffer pool eliminates allocations
- [ ] Batch header reading implemented
- [ ] 30-50% speedup vs current implementation

---

### Task 2.3: Add Progress Tracking with ETA

**Priority:** MEDIUM | **Effort:** 2 hours | **Impact:** UX improvement

**File:** `utils/segy_import/segy_reader.py`

```python
class LoadingProgress:
    """Track loading progress with ETA calculation."""

    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self._window_size = 10
        self._recent_times = []

    def update(self, current: int):
        """Update progress and calculate ETA."""
        now = time.time()
        self.current = current

        # Track recent progress for smoother ETA
        self._recent_times.append((current, now))
        if len(self._recent_times) > self._window_size:
            self._recent_times.pop(0)

    @property
    def percent(self) -> float:
        return (self.current / self.total) * 100 if self.total > 0 else 0

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining."""
        if len(self._recent_times) < 2:
            return float('inf')

        first = self._recent_times[0]
        last = self._recent_times[-1]

        traces_processed = last[0] - first[0]
        time_elapsed = last[1] - first[1]

        if traces_processed == 0 or time_elapsed == 0:
            return float('inf')

        rate = traces_processed / time_elapsed
        remaining = self.total - self.current
        return remaining / rate

    @property
    def throughput(self) -> float:
        """Current throughput in traces/second."""
        if len(self._recent_times) < 2:
            return 0

        first = self._recent_times[0]
        last = self._recent_times[-1]
        time_elapsed = last[1] - first[1]
        traces = last[0] - first[0]

        return traces / time_elapsed if time_elapsed > 0 else 0
```

**Acceptance Criteria:**
- [ ] Accurate ETA calculation
- [ ] Smooth throughput reporting
- [ ] Integration with existing progress_callback

---

## 3. Architecture Changes

### Task 3.1: Implement Zarr Conversion Pipeline

**Priority:** HIGH | **Effort:** 8 hours | **Impact:** 10-100x for repeated access

**Description:** Convert SEGY to Zarr format for efficient repeated access.

**New File:** `utils/segy_import/zarr_converter.py`

```python
"""
Convert SEGY files to Zarr format for optimized access.
"""
import numpy as np
import zarr
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping


class SEGYToZarrConverter:
    """
    Convert SEGY files to Zarr format for fast repeated access.

    Output structure:
        output_dir/
            ├── traces.zarr/     # Chunked trace data
            ├── headers.parquet  # Trace headers
            ├── ensembles.parquet # Ensemble index
            └── metadata.json    # File metadata
    """

    def __init__(
        self,
        chunk_traces: int = 1000,
        chunk_samples: Optional[int] = None,
        compression: str = 'blosc'
    ):
        """
        Initialize converter.

        Args:
            chunk_traces: Number of traces per Zarr chunk
            chunk_samples: Samples per chunk (None = all samples)
            compression: Zarr compression ('blosc', 'zlib', 'none')
        """
        self.chunk_traces = chunk_traces
        self.chunk_samples = chunk_samples
        self.compression = compression

    def convert(
        self,
        segy_path: str,
        output_dir: str,
        header_mapping: HeaderMapping,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Convert SEGY file to Zarr format.

        Args:
            segy_path: Path to input SEGY file
            output_dir: Output directory for Zarr data
            header_mapping: Header mapping configuration
            progress_callback: Optional callback(current, total, stage)

        Returns:
            Conversion statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        reader = SEGYReader(segy_path, header_mapping)
        info = reader.read_file_info()

        n_samples = info['n_samples']
        n_traces = info['n_traces']
        sample_rate = info['sample_interval']

        # Configure chunks
        chunk_samples = self.chunk_samples or n_samples
        chunks = (chunk_samples, self.chunk_traces)

        # Configure compression
        if self.compression == 'blosc':
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        elif self.compression == 'zlib':
            compressor = zarr.Zlib(level=3)
        else:
            compressor = None

        # Create Zarr array
        zarr_path = output_path / 'traces.zarr'
        zarr_array = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=chunks,
            dtype=np.float32,
            compressor=compressor
        )

        # Convert in chunks
        all_headers = []
        traces_written = 0

        for traces, headers, start, end in reader.read_traces_in_chunks(
            chunk_size=self.chunk_traces * 10  # Read larger chunks
        ):
            zarr_array[:, start:end] = traces
            all_headers.extend(headers)
            traces_written = end

            if progress_callback:
                progress_callback(traces_written, n_traces, 'Converting traces')

        # Save headers to Parquet
        if progress_callback:
            progress_callback(n_traces, n_traces, 'Saving headers')

        headers_df = pd.DataFrame(all_headers)
        headers_df['trace_index'] = range(len(all_headers))
        headers_df.to_parquet(output_path / 'headers.parquet', index=False)

        # Detect and save ensembles
        if header_mapping.ensemble_keys:
            ensembles = reader.detect_ensemble_boundaries(all_headers)
            ensemble_df = pd.DataFrame([
                {'ensemble_id': i, 'start_trace': s, 'end_trace': e}
                for i, (s, e) in enumerate(ensembles)
            ])
            ensemble_df.to_parquet(output_path / 'ensemble_index.parquet', index=False)

        # Save metadata
        metadata = {
            'source_file': str(segy_path),
            'n_samples': n_samples,
            'n_traces': n_traces,
            'sample_rate': sample_rate,
            'chunks': chunks,
            'compression': self.compression,
            'format_version': '1.0'
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            'output_dir': str(output_path),
            'n_traces': n_traces,
            'n_samples': n_samples,
            'zarr_size_mb': sum(f.stat().st_size for f in zarr_path.rglob('*')) / 1024 / 1024
        }
```

**Acceptance Criteria:**
- [ ] Conversion preserves all data
- [ ] Zarr access 10-100x faster than SEGY
- [ ] Headers and ensembles preserved

---

### Task 3.2: Implement Lazy Loading for UI

**Priority:** HIGH | **Effort:** 6 hours | **Impact:** Instant file open

**Description:** Load only visible traces for UI responsiveness.

**File:** `models/lazy_seismic_data.py` (enhance existing)

```python
# Add window caching to LazySeismicData

class CachedLazySeismicData(LazySeismicData):
    """LazySeismicData with LRU window caching."""

    def __init__(self, *args, cache_size_mb: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_size_mb = cache_size_mb
        self._cache = {}
        self._cache_order = []  # LRU tracking

    def get_window_cached(
        self,
        time_start: float,
        time_end: float,
        trace_start: int,
        trace_end: int
    ) -> np.ndarray:
        """Get window with LRU caching."""
        cache_key = (time_start, time_end, trace_start, trace_end)

        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._cache[cache_key]

        # Load window
        data = self.get_window(time_start, time_end, trace_start, trace_end)

        # Add to cache
        self._cache[cache_key] = data
        self._cache_order.append(cache_key)

        # Evict if needed
        self._evict_if_needed()

        return data

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds size limit."""
        current_size_mb = sum(
            arr.nbytes for arr in self._cache.values()
        ) / 1024 / 1024

        while current_size_mb > self._cache_size_mb and self._cache_order:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._cache:
                current_size_mb -= self._cache[oldest_key].nbytes / 1024 / 1024
                del self._cache[oldest_key]

    def prefetch_adjacent(self, current_trace: int, window_size: int):
        """Prefetch adjacent windows in background."""
        # Implementation for background prefetching
        pass
```

**Acceptance Criteria:**
- [ ] Instant file open (< 1 second)
- [ ] Smooth scrolling with prefetching
- [ ] Memory usage bounded by cache size

---

### Task 3.3: Add Parallel Loading with ThreadPoolExecutor

**Priority:** MEDIUM | **Effort:** 4 hours | **Impact:** 2-4x on multi-core

**File:** `utils/segy_import/parallel_reader.py`

```python
"""
Parallel SEGY reading for multi-core systems.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, List, Tuple
from pathlib import Path
import segyio
from utils.segy_import.header_mapping import HeaderMapping


class ParallelSEGYReader:
    """
    Read SEGY files using multiple threads.

    Note: Due to Python's GIL, this primarily helps with I/O-bound operations.
    Use with mmap for best results.
    """

    def __init__(
        self,
        filename: str,
        header_mapping: HeaderMapping,
        n_workers: int = 4
    ):
        self.filename = Path(filename)
        self.header_mapping = header_mapping
        self.n_workers = n_workers

    def read_all_parallel(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        """Read traces in parallel using thread pool."""

        with segyio.open(str(self.filename), ignore_geometry=True) as f:
            f.mmap()
            n_traces = f.tracecount
            n_samples = len(f.samples)

        # Divide into chunks for workers
        chunk_size = max(1000, n_traces // (self.n_workers * 4))
        chunks = [
            (start, min(start + chunk_size, n_traces))
            for start in range(0, n_traces, chunk_size)
        ]

        # Pre-allocate output
        traces = np.empty((n_samples, n_traces), dtype=np.float32)
        headers = [None] * n_traces

        completed = 0

        def read_chunk(chunk_range):
            start, end = chunk_range
            with segyio.open(str(self.filename), ignore_geometry=True) as f:
                f.mmap()
                chunk_traces = np.empty((n_samples, end - start), dtype=np.float32)
                chunk_headers = []

                for i, idx in enumerate(range(start, end)):
                    chunk_traces[:, i] = f.trace[idx]
                    chunk_headers.append(self._read_header(f, idx))

            return start, end, chunk_traces, chunk_headers

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(read_chunk, chunk): chunk for chunk in chunks}

            for future in as_completed(futures):
                start, end, chunk_traces, chunk_headers = future.result()
                traces[:, start:end] = chunk_traces
                headers[start:end] = chunk_headers

                completed += (end - start)
                if progress_callback:
                    progress_callback(completed, n_traces)

        return traces, headers

    def _read_header(self, f, idx: int) -> dict:
        """Read single header."""
        header = {}
        for name, byte_loc in self.header_mapping.get_all_mappings().items():
            try:
                header[name] = f.header[idx][byte_loc]
            except KeyError:
                header[name] = 0
        return header
```

**Acceptance Criteria:**
- [ ] 2-4x speedup on 4+ core systems
- [ ] Thread-safe operation
- [ ] Graceful degradation on single-core

---

## 4. Advanced Optimizations

### Task 4.1: GPU-Accelerated Float Conversion

**Priority:** LOW | **Effort:** 8 hours | **Impact:** 5-10x for IBM float

**Description:** Use CuPy for GPU-accelerated IBM float conversion.

**File:** `utils/segy_import/gpu_converter.py`

```python
"""
GPU-accelerated data conversion for SEGY files.
"""
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def ibm_to_ieee_gpu(ibm_data: np.ndarray) -> np.ndarray:
    """
    Convert IBM float to IEEE float using GPU.

    Args:
        ibm_data: Raw IBM float data as uint32 array

    Returns:
        IEEE float32 array
    """
    if not GPU_AVAILABLE:
        return ibm_to_ieee_cpu(ibm_data)

    # Move to GPU
    d_ibm = cp.asarray(ibm_data.view(np.uint32))

    # IBM float format:
    # [1 bit sign][7 bit exponent][24 bit mantissa]
    # IEEE float format:
    # [1 bit sign][8 bit exponent][23 bit mantissa]

    sign = (d_ibm >> 31).astype(cp.float32)
    exponent = ((d_ibm >> 24) & 0x7f).astype(cp.int32)
    mantissa = (d_ibm & 0x00ffffff).astype(cp.float32)

    # IBM uses base-16 exponent, IEEE uses base-2
    # IBM exponent is biased by 64
    ieee_exp = 4 * (exponent - 64)

    # Calculate value
    result = (-1.0) ** sign * mantissa * (2.0 ** (ieee_exp - 24))

    # Handle zeros
    result = cp.where(mantissa == 0, 0.0, result)

    return cp.asnumpy(result).astype(np.float32)


def ibm_to_ieee_cpu(ibm_data: np.ndarray) -> np.ndarray:
    """CPU fallback for IBM to IEEE conversion."""
    ibm = ibm_data.view(np.uint32)

    sign = (ibm >> 31).astype(np.float32)
    exponent = ((ibm >> 24) & 0x7f).astype(np.int32)
    mantissa = (ibm & 0x00ffffff).astype(np.float32)

    ieee_exp = 4 * (exponent - 64)
    result = (-1.0) ** sign * mantissa * (2.0 ** (ieee_exp - 24))
    result = np.where(mantissa == 0, 0.0, result)

    return result.astype(np.float32)
```

**Acceptance Criteria:**
- [ ] GPU conversion 5-10x faster
- [ ] Automatic CPU fallback
- [ ] Numerical accuracy verified

---

### Task 4.2: Implement Streaming Export

**Priority:** MEDIUM | **Effort:** 4 hours | **Impact:** Memory efficiency

**File:** `utils/segy_import/segy_export.py` (enhance existing)

```python
def export_chunked(
    self,
    source_segy: str,
    data_generator,  # Generator yielding (traces, headers) chunks
    output_path: str,
    chunk_size: int = 5000,
    progress_callback: Optional[Callable] = None
):
    """
    Export SEGY file from chunked data generator.

    Memory-efficient export for processed data.
    """
    import segyio

    # Get template from source
    with segyio.open(source_segy, ignore_geometry=True) as src:
        spec = segyio.spec()
        spec.samples = src.samples
        spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE

        # Count total traces from generator (first pass)
        # Or require it as parameter

    total_traces = 0
    trace_idx = 0

    with segyio.create(output_path, spec) as dst:
        for traces_chunk, headers_chunk in data_generator:
            n_chunk = traces_chunk.shape[1]

            for i in range(n_chunk):
                dst.trace[trace_idx] = traces_chunk[:, i]
                # Copy headers
                for key, val in headers_chunk[i].items():
                    if hasattr(segyio.TraceField, key.upper()):
                        field = getattr(segyio.TraceField, key.upper())
                        dst.header[trace_idx][field] = int(val)
                trace_idx += 1

            if progress_callback:
                progress_callback(trace_idx, total_traces)
```

**Acceptance Criteria:**
- [ ] Constant memory usage during export
- [ ] Handles arbitrarily large files
- [ ] Preserves header information

---

## Implementation Priority Matrix

| Task | Priority | Effort | Impact | Dependencies |
|------|----------|--------|--------|--------------|
| 1.1 Enable mmap | HIGH | 1h | 10-50% | None |
| 1.2 Pre-allocate arrays | HIGH | 2h | 20-40% | None |
| 1.3 Batch headers | MEDIUM | 2h | 15-30% | None |
| 2.1 segfast integration | HIGH | 4h | 2-10x | pip install segfast |
| 2.2 Optimized chunks | HIGH | 3h | 30-50% | Task 1.1, 1.2 |
| 2.3 Progress ETA | MEDIUM | 2h | UX | None |
| 3.1 Zarr conversion | HIGH | 8h | 10-100x | zarr, pandas |
| 3.2 Lazy loading | HIGH | 6h | Instant open | Task 3.1 |
| 3.3 Parallel loading | MEDIUM | 4h | 2-4x | Task 1.1 |
| 4.1 GPU conversion | LOW | 8h | 5-10x | cupy, CUDA |
| 4.2 Streaming export | MEDIUM | 4h | Memory | None |

---

## Suggested Implementation Order

### Phase 1: Quick Wins (Week 1)
1. Task 1.1: Enable mmap
2. Task 1.2: Pre-allocate arrays
3. Task 1.3: Batch header reading

### Phase 2: Core Optimizations (Week 2)
4. Task 2.1: segfast integration
5. Task 2.2: Optimized chunked processing
6. Task 2.3: Progress with ETA

### Phase 3: Architecture (Week 3-4)
7. Task 3.1: Zarr conversion pipeline
8. Task 3.2: Lazy loading enhancement
9. Task 3.3: Parallel loading

### Phase 4: Advanced (As Needed)
10. Task 4.1: GPU float conversion
11. Task 4.2: Streaming export

---

## Verification Checklist

After implementing each task:

- [ ] Run diagnostic tests to measure improvement
- [ ] Update benchmark suite with new measurements
- [ ] Verify no regressions in existing functionality
- [ ] Update documentation
- [ ] Add unit tests for new code
