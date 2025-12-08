# SEGY Acceleration Implementation Plan

## Overview

This document provides a detailed implementation plan for accelerating SEGY loading performance beyond the current 2x optimizations. Features are ordered by ROI (return on investment).

---

## Feature 1: Configurable Chunk Size (Easiest - 10-20% gain)

### Current State
- Hardcoded `chunk_size=5000` in multiple locations
- No user configuration option

### Files to Modify

#### 1.1 Add Configuration Constant
**File:** `utils/segy_import/__init__.py`

```python
# Add at top of file
DEFAULT_CHUNK_SIZE = 5000
LARGE_MEMORY_CHUNK_SIZE = 25000  # For systems with 32GB+ RAM
```

#### 1.2 Update SEGYReader Default
**File:** `utils/segy_import/segy_reader.py`
- Line 534: Change `chunk_size: int = 5000` to use imported constant
- Add docstring note about memory/performance tradeoff

#### 1.3 Update DataStorage Defaults
**File:** `utils/segy_import/data_storage.py`
- Line 94, 356: Update default chunk_size parameters
- Line 414: Update Zarr chunk size calculation

#### 1.4 Update Import Dialog
**File:** `views/segy_import_dialog.py`
- Line 1045, 1062: Replace hardcoded 5000 with configurable value
- Add chunk size dropdown/spinbox to Advanced Options panel (near line 200-300 in dialog setup)

#### 1.5 Update ChunkedProcessor
**File:** `processors/chunked_processor.py`
- Line 37: Update default parameter

### Cleanup
- Remove hardcoded `5000` values from:
  - `test_task_7_3_performance_benchmark.py`
  - `test_task_1_4_ensemble_streaming.py`
  - `test_task_1_3_header_streaming.py`
  - `main_window.py:1591`

### Integration
- Add to settings/preferences if app has settings dialog
- Or add as environment variable: `SEISPROC_CHUNK_SIZE`

---

## Feature 2: Parallel Trace Reading (Medium Effort - 2-4x gain)

### Current State
- Sequential trace reading in `segy_reader.py`
- Single-threaded I/O operations
- mmap already enabled (thread-safe for reads)

### Files to Modify

#### 2.1 Create Parallel Reader Module
**File (NEW):** `utils/segy_import/parallel_reader.py`

```python
"""
Parallel SEGY trace reader using ThreadPoolExecutor.
Provides 2-4x speedup on multi-core systems with SSD storage.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import segyio
from typing import List, Tuple, Optional, Callable
import threading

class ParallelSEGYReader:
    """
    Reads SEGY traces in parallel using multiple threads.

    Thread-safe because:
    1. segyio with mmap is safe for concurrent reads
    2. Each thread writes to different array slice
    3. No shared mutable state
    """

    def __init__(self, filename: str, header_mapping, n_workers: int = None):
        """
        Args:
            filename: Path to SEGY file
            header_mapping: HeaderMapping instance
            n_workers: Number of worker threads (default: CPU count)
        """
        pass

    def read_all_traces_parallel(self, max_traces=None, progress_callback=None):
        """
        Read all traces using parallel workers.

        Strategy:
        1. Divide trace range into N segments (one per worker)
        2. Each worker reads its segment into pre-allocated array slice
        3. Main thread coordinates and tracks progress
        """
        pass

    def read_traces_in_chunks_parallel(self, chunk_size=5000, n_prefetch=2):
        """
        Stream chunks with parallel prefetching.

        Strategy:
        1. Submit next chunk read while current chunk is being processed
        2. Overlap I/O with processing (pipelining)
        """
        pass
```

#### 2.2 Integrate with SEGYReader
**File:** `utils/segy_import/segy_reader.py`

Add method at line ~530 (after `read_all_traces`):

```python
def read_all_traces_parallel(
    self,
    max_traces: Optional[int] = None,
    n_workers: int = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Read traces using parallel I/O (2-4x faster on multi-core).

    Args:
        n_workers: Number of parallel readers (default: CPU cores - 1)
    """
    pass
```

#### 2.3 Update Factory Function
**File:** `utils/segy_import/segy_reader_fast.py`
- Line 300-322: Update `create_segy_reader()` to accept `parallel=True` option
- Return ParallelSEGYReader when requested

#### 2.4 Update Exports
**File:** `utils/segy_import/__init__.py`
- Add `ParallelSEGYReader` to imports and `__all__`

#### 2.5 Update Import Dialog (Optional)
**File:** `views/segy_import_dialog.py`
- Add checkbox "Use parallel loading" in Advanced Options
- Pass to reader creation

### Implementation Details

```python
# Key implementation pattern for parallel reading:

def _read_segment(self, f, start: int, end: int, output: np.ndarray):
    """Worker function - reads trace segment into pre-allocated array."""
    for i, idx in enumerate(range(start, end)):
        output[:, start + i] = f.trace[idx]

def read_all_traces_parallel(self, ...):
    with segyio.open(self.filename, 'r', ignore_geometry=True) as f:
        f.mmap()  # Enable memory mapping (thread-safe)

        n_traces = f.tracecount
        n_samples = len(f.samples)

        # Pre-allocate output array
        traces = np.empty((n_samples, n_traces), dtype=np.float32)

        # Calculate segments for each worker
        segment_size = n_traces // self.n_workers
        segments = [(i * segment_size, min((i+1) * segment_size, n_traces))
                    for i in range(self.n_workers)]

        # Submit parallel reads
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._read_segment, f, start, end, traces)
                for start, end in segments
            ]

            # Wait for completion with progress tracking
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Read headers (can also be parallelized)
        headers = self._read_headers_batch_with_mapping(f, 0, n_traces)

    return traces, headers
```

### Cleanup
- None required - this is additive

---

## Feature 3: Selective Header Loading (Easy - 5-15% gain)

### Current State
- `add_standard_headers()` adds 21 header fields
- All fields read even if unused
- Each field requires separate `segyio.attributes()` call

### Files to Modify

#### 3.1 Add Minimal Header Preset
**File:** `utils/segy_import/header_mapping.py`

Add after `add_standard_headers()` method (around line 150):

```python
def add_minimal_headers(self):
    """
    Add only essential headers for basic import.
    Faster than add_standard_headers() (7 fields vs 21).

    Includes: trace_sequence, cdp, offset, inline, crossline, cdp_x, cdp_y
    """
    essential_fields = [
        ('trace_sequence_line', 1),
        ('cdp', 21),
        ('offset', 37),
        ('inline', 189),
        ('crossline', 193),
        ('cdp_x', 181),
        ('cdp_y', 185),
    ]
    for name, byte_loc in essential_fields:
        self.add_field(name, byte_loc)

def add_geometry_only_headers(self):
    """
    Add only geometry headers (coordinates).
    Minimal for QC/visualization.
    """
    geometry_fields = [
        ('cdp_x', 181),
        ('cdp_y', 185),
        ('source_x', 73),
        ('source_y', 77),
    ]
    for name, byte_loc in geometry_fields:
        self.add_field(name, byte_loc)
```

#### 3.2 Update Import Dialog
**File:** `views/segy_import_dialog.py`

Add header preset dropdown:
- "Full Headers (21 fields)" - uses `add_standard_headers()`
- "Essential Headers (7 fields)" - uses `add_minimal_headers()`
- "Geometry Only (4 fields)" - uses `add_geometry_only_headers()`
- "Custom" - existing custom mapping

Location: Near header mapping configuration UI

### Cleanup
- None required

---

## Feature 4: segfast Backend Activation (Already Implemented)

### Current State
- `FastSEGYReader` exists in `segy_reader_fast.py`
- `create_segy_reader()` factory exists
- Just needs segfast installed: `pip install segfast`

### Files to Modify

#### 4.1 Update Import Dialog to Use Fast Reader
**File:** `views/segy_import_dialog.py`

Around line 1000-1010 where reader is created:

```python
# CHANGE FROM:
self.reader = SEGYReader(str(self.segy_file), self.header_mapping)

# CHANGE TO:
from utils.segy_import import create_segy_reader, is_fast_reader_available
self.reader = create_segy_reader(
    str(self.segy_file),
    self.header_mapping,
    prefer_fast=True
)
if is_fast_reader_available():
    print("Using segfast backend (2-10x faster)")
```

#### 4.2 Add Install Check to App Startup (Optional)
**File:** `main_window.py` or `__main__.py`

```python
from utils.segy_import import is_fast_reader_available
if not is_fast_reader_available():
    print("TIP: Install segfast for 2-10x faster SEGY loading: pip install segfast")
```

### Cleanup
- None required

---

## Feature 5: GPU-Accelerated IBM Float Conversion (Complex - Conditional Gain)

### When This Helps
- Only for SEGY files with IBM float format (format code 1)
- Most modern SEGY uses IEEE float (format code 5) - no benefit
- Requires NVIDIA GPU with CUDA

### Current State
- segyio handles IBM→IEEE conversion in C (already fast)
- No GPU path exists

### Files to Create

#### 5.1 GPU Conversion Module
**File (NEW):** `utils/segy_import/gpu_conversion.py`

```python
"""
GPU-accelerated IBM float to IEEE float conversion using CuPy.
Only beneficial for large IBM-format SEGY files.
"""
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return CUPY_AVAILABLE

def ibm_to_ieee_gpu(data: np.ndarray) -> np.ndarray:
    """
    Convert IBM float bytes to IEEE float on GPU.

    Note: This is only useful if reading raw bytes directly,
    bypassing segyio's automatic conversion.
    """
    # Implementation requires reading raw trace bytes
    # and performing bit manipulation on GPU
    pass
```

### Assessment
**Recommendation:** Skip this feature unless profiling shows IBM conversion is a bottleneck. The complexity is high and benefit is limited because:
1. segyio already does efficient C-level conversion
2. Most modern files use IEEE format
3. I/O is usually the bottleneck, not conversion

---

## Feature 6: Zarr Pre-conversion for Repeated Access (High Impact for Workflows)

### Current State
- Each session re-reads SEGY file
- Zarr storage exists but requires explicit import

### Concept
- First access: Convert SEGY → Zarr (one-time cost)
- Subsequent access: Load from Zarr (10-100x faster)
- Cache management for disk space

### Files to Modify

#### 6.1 Create Cache Manager
**File (NEW):** `utils/segy_import/segy_cache.py`

```python
"""
SEGY-to-Zarr cache manager for fast repeated access.
"""
import hashlib
from pathlib import Path

class SEGYCache:
    """
    Manages cached Zarr conversions of SEGY files.

    Cache key: hash of (file_path, file_size, mtime)
    """

    def __init__(self, cache_dir: str = "~/.seisproc/cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cached(self, segy_path: str) -> Optional[Path]:
        """Get cached Zarr path if valid cache exists."""
        pass

    def cache_segy(self, segy_path: str, zarr_path: Path):
        """Register a Zarr as cache for SEGY file."""
        pass

    def clear_cache(self, max_age_days: int = 30):
        """Remove old cache entries."""
        pass
```

### Integration Points
- `views/segy_import_dialog.py`: Check cache before import
- Settings: Cache location, size limit, auto-clear policy

---

## Implementation Priority Order

| Priority | Feature | Effort | Gain | Files Changed |
|----------|---------|--------|------|---------------|
| 1 | Configurable Chunk Size | Low | 10-20% | 5-6 files |
| 2 | segfast Activation | Very Low | 2-10x | 1-2 files |
| 3 | Selective Headers | Low | 5-15% | 2 files |
| 4 | Parallel Reading | Medium | 2-4x | 3-4 files |
| 5 | Zarr Caching | Medium | 10-100x* | 2-3 new files |
| 6 | GPU Conversion | High | Marginal | Skip |

*For repeated access only

---

## Recommended Execution Order

### Phase 1: Quick Wins (1-2 hours)
1. Update `segy_import_dialog.py` to use `create_segy_reader(prefer_fast=True)`
2. Add `add_minimal_headers()` to `header_mapping.py`
3. Extract chunk size to configuration constant

### Phase 2: Parallel Loading (4-6 hours)
1. Create `parallel_reader.py` with `ParallelSEGYReader`
2. Add parallel option to `SEGYReader`
3. Update factory and exports
4. Add UI option in import dialog

### Phase 3: Caching System (4-6 hours)
1. Create `segy_cache.py`
2. Integrate with import workflow
3. Add cache management UI

---

## Testing Requirements

### For Each Feature
1. Unit tests for new classes/methods
2. Integration test with sample SEGY file
3. Performance benchmark comparison
4. Memory usage validation

### Benchmark Script Updates
**File:** `tests/benchmarks/benchmark_segy_performance.py`

Add test cases for:
- Different chunk sizes
- Parallel vs sequential
- With/without segfast
- Different header presets

---

## Cleanup Checklist

### Hardcoded Values to Remove
- [ ] `chunk_size=5000` in 15+ locations (replace with constant)
- [ ] Magic numbers in tests

### Old Code to Remove
- [ ] None identified - all changes are additive

### Documentation Updates
- [ ] Update `SEGY_Performance_Implementation_Summary.md`
- [ ] Add docstrings to new classes
- [ ] Update README if exists

---

## Dependencies to Add

```
# Optional performance dependencies (requirements-perf.txt)
segfast>=0.1.0  # Fast SEGY reading
cupy>=12.0.0    # GPU acceleration (optional)
```
