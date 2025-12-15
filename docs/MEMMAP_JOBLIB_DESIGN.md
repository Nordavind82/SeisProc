# Memory-Mapped Arrays and Joblib Integration Design

## Executive Summary

This document analyzes the current parallel processing architecture across SeisProc's four major subsystems and provides recommendations for integrating **memory-mapped arrays (memmap)** and **Joblib** to improve memory efficiency, crash recovery, and performance.

### Key Findings

| Subsystem | Current State | Memmap Opportunity | Joblib Opportunity |
|-----------|---------------|--------------------|--------------------|
| SEGY Import | Spawn context, segment files | **HIGH** - Trace arrays | LOW - Complex workflow |
| Parallel Export | Spawn context, pickle headers | **HIGH** - Header arrays | MEDIUM - Simpler workflow |
| Batch Processing | Fork COW, shared data | **MEDIUM** - Already optimized | LOW - Fork COW preferred |
| CPU/GPU Processors | Numba JIT, conditionals | **LOW** - In-memory preferred | **HIGH** - Replace inner loops |

---

## Part 1: Current Architecture Analysis

### 1.1 Parallel SEGY Import (`utils/segy_import/multiprocess_import/`)

**Files**: coordinator.py (1024 lines), worker.py (310 lines), partitioner.py (231 lines)

**Current Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Partitioner │───▶│ Coordinator │───▶│  Workers    │
│ (File scan) │    │ (Spawn ctx) │    │ (Segments)  │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Zarr Output │
                   │ + Parquet   │
                   └─────────────┘
```

**Key Characteristics**:
- Uses **spawn** context (safe for SEGY library)
- Partitions file into segments by trace count
- Each worker writes segment headers to temp Parquet files
- Final merge combines segments into single output
- **Memory bottleneck**: Workers load full segment traces into memory

**Current Memory Pattern** (coordinator.py:890-920):
```python
# Workers receive segment bounds via pickle
segment = {
    'start_trace': start,
    'end_trace': end,
    'file_path': segy_path,
}
# Each worker loads traces independently - NO sharing
```

### 1.2 Parallel Export (`utils/parallel_export/`)

**Files**: coordinator.py (806 lines), worker.py (347 lines), merger.py (206 lines), header_vectorizer.py (346 lines)

**Current Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Header Vec  │───▶│ Coordinator │───▶│  Workers    │
│ (Vectorize) │    │ (Spawn ctx) │    │ (Chunks)    │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ SEGY Merger │
                   │ (Combine)   │
                   └─────────────┘
```

**Key Characteristics**:
- Uses **spawn** context with adaptive chunk sizing
- HeaderVectorizer creates O(1) header access from ensemble info
- Per-segment headers saved via **pickle** (memory inefficient!)
- Zarr source data accessed via memory mapping (good)

**Current Memory Pattern** (worker.py:180-220):
```python
# Headers pickled per segment - INEFFICIENT
with open(header_file, 'rb') as f:
    segment_headers = pickle.load(f)  # Copies into worker memory
```

### 1.3 Parallel Batch Processing (`utils/parallel_processing/`)

**Files**: coordinator.py (1353 lines), worker.py (1000 lines), shared_data.py (407 lines)

**Current Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Shared Data │───▶│ Coordinator │───▶│  Workers    │
│ (Pre-load)  │    │ (Fork COW)  │    │ (Gathers)   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Zarr Output │
                   │ + Streaming │
                   └─────────────┘
```

**Key Characteristics**:
- Uses **fork** context with Copy-on-Write (COW) optimization
- Pre-loads ensemble index and headers before fork (shared_data.py)
- Streaming sort mappings for memory efficiency
- Output modes: processed, noise, or both
- **Already well-optimized** for memory sharing

**Current Memory Pattern** (shared_data.py:50-80):
```python
# Pre-loaded data shared via fork COW - EFFICIENT
_SHARED_ENSEMBLE_INDEX = None
_SHARED_ENSEMBLE_ARRAYS = None  # NumPy arrays for O(1) access

def set_shared_ensemble_index(ensemble_df):
    global _SHARED_ENSEMBLE_INDEX, _SHARED_ENSEMBLE_ARRAYS
    _SHARED_ENSEMBLE_INDEX = ensemble_df
    _SHARED_ENSEMBLE_ARRAYS = {
        'start_trace': ensemble_df['start_trace'].to_numpy(),
        'end_trace': ensemble_df['end_trace'].to_numpy(),
    }
```

### 1.4 CPU/GPU Processors (`processors/`)

**Files**: 52 processor files including denoise_3d.py, nmo_processor.py, cdp_stacker.py

**Current Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Processor   │───▶│ Device Mgr  │───▶│ Compute     │
│ Interface   │    │ (CPU/GPU)   │    │ (Batch)     │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Key Characteristics**:
- Float32 optimization throughout
- **Numba JIT** for CPU-intensive loops
- **PyTorch** for GPU with automatic batching
- Conditional Joblib usage in some processors
- Device management with OOM recovery

**Current Joblib Pattern** (denoise_3d.py:145-170):
```python
# Conditional Joblib for embarrassingly parallel operations
if self.n_jobs > 1:
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=self.n_jobs)(
        delayed(process_gather)(gather) for gather in gathers
    )
else:
    results = [process_gather(g) for g in gathers]
```

---

## Part 2: Memory-Mapped Array Opportunities

### 2.1 What is Memory Mapping?

Memory mapping allows direct access to file data without loading into RAM:
```python
import numpy as np

# Create memmap - file on disk, accessed like array
arr = np.memmap('data.dat', dtype='float32', mode='r', shape=(10000, 1000))

# Slicing only reads required portions from disk
subset = arr[100:200, :]  # Only these rows loaded
```

**Benefits**:
- Constant memory footprint regardless of data size
- OS handles caching automatically
- Multiple processes can share same mapping
- Works with spawn context (unlike fork COW)

### 2.2 High-Priority: SEGY Import Trace Arrays

**Problem**: Each worker loads full segment traces into memory.

**Solution**: Write traces to memmap during import, share across workers.

**Implementation Location**: `utils/segy_import/multiprocess_import/worker.py`

```python
# CURRENT (memory inefficient)
traces = np.zeros((n_samples, n_traces), dtype=np.float32)
for i, trace in enumerate(segy_file.traces):
    traces[:, i] = trace.data

# PROPOSED (memory mapped)
import tempfile
import numpy as np

# Worker writes to memmap file
memmap_path = tempfile.mktemp(suffix='.dat')
traces = np.memmap(memmap_path, dtype='float32', mode='w+',
                   shape=(n_samples, n_traces))

for i, trace in enumerate(segy_file.traces):
    traces[:, i] = trace.data

traces.flush()  # Ensure written to disk
# Return memmap_path to coordinator for final merge
```

**Memory Savings**: ~50-80% reduction per worker for large segments.

### 2.3 High-Priority: Export Header Arrays

**Problem**: Headers pickled per segment, copied into worker memory.

**Solution**: Use structured memmap array for headers.

**Implementation Location**: `utils/parallel_export/header_vectorizer.py`

```python
# CURRENT (pickle-based)
headers_dict = {...}
with open(f'segment_{i}_headers.pkl', 'wb') as f:
    pickle.dump(headers_dict, f)

# PROPOSED (memmap-based)
import numpy as np

# Define structured dtype for headers
header_dtype = np.dtype([
    ('trace_seq', 'i4'),
    ('inline', 'i4'),
    ('crossline', 'i4'),
    ('cdp_x', 'f8'),
    ('cdp_y', 'f8'),
    ('offset', 'f4'),
    ('source_x', 'f8'),
    ('source_y', 'f8'),
])

# Create memmap for all headers
headers_memmap = np.memmap('headers.dat', dtype=header_dtype,
                           mode='w+', shape=(n_traces,))

# Fill from DataFrame
headers_memmap['trace_seq'] = df['TRACE_SEQUENCE_FILE'].values
headers_memmap['inline'] = df['INLINE_3D'].values
# ... etc

headers_memmap.flush()

# Workers open in read mode - shared mapping
worker_headers = np.memmap('headers.dat', dtype=header_dtype, mode='r')
```

**Memory Savings**: ~70% reduction (no pickle overhead, shared mapping).

### 2.4 Medium-Priority: Batch Processing Sort Mappings

**Problem**: Sort mappings loaded per worker (can be large for sorted output).

**Solution**: Memmap for sort index arrays.

**Implementation Location**: `utils/parallel_processing/coordinator.py`

```python
# CURRENT (streaming but still in-memory per worker)
sort_mapping = np.load(sort_mapping_path)

# PROPOSED (memmap)
sort_mapping = np.memmap(sort_mapping_path.replace('.npy', '.dat'),
                         dtype='i8', mode='r', shape=(n_traces,))
```

**Note**: This is lower priority as fork COW already provides sharing.

### 2.5 Low-Priority: Processor Intermediate Results

Most processors work on gather-sized data (fits in memory). Memmap adds overhead for small arrays. **Not recommended** for processor internals.

---

## Part 3: Joblib Integration Opportunities

### 3.1 What is Joblib?

Joblib provides parallel computing with:
- Automatic memmap for large arrays
- Crash recovery and timeout handling
- Simpler API than ProcessPoolExecutor
- Smart caching (optional)

```python
from joblib import Parallel, delayed, parallel_config

# Basic usage
results = Parallel(n_jobs=4)(
    delayed(process_func)(item) for item in items
)

# With memmap threshold (auto-memmap arrays > 1MB)
with parallel_config(max_nbytes='1M'):
    results = Parallel(n_jobs=4)(
        delayed(process_func)(large_array, item)
        for item in items
    )
```

### 3.2 High-Priority: Processor Inner Loops

**Problem**: Processors have conditional Joblib that's inconsistent.

**Solution**: Standardize Joblib pattern across all processors.

**Implementation Location**: Create `processors/parallel_mixin.py`

```python
"""Standardized parallel processing mixin for processors."""
from joblib import Parallel, delayed, parallel_config
import numpy as np

class ParallelProcessorMixin:
    """Mixin providing standardized parallel processing for processors."""

    def __init__(self, n_jobs: int = 1, memmap_threshold: str = '1M'):
        self.n_jobs = n_jobs
        self.memmap_threshold = memmap_threshold

    def parallel_map(self, func, items, desc: str = None):
        """Apply function to items in parallel with optional memmap."""
        if self.n_jobs == 1:
            return [func(item) for item in items]

        with parallel_config(max_nbytes=self.memmap_threshold):
            return Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(func)(item) for item in items
            )

    def parallel_map_with_shared(self, func, items, shared_data, desc: str = None):
        """Apply function with shared data (auto-memmaped if large)."""
        if self.n_jobs == 1:
            return [func(item, shared_data) for item in items]

        with parallel_config(max_nbytes=self.memmap_threshold):
            return Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(func)(item, shared_data) for item in items
            )
```

**Usage in Processors**:
```python
class Denoise3DProcessor(ParallelProcessorMixin):
    def __init__(self, n_jobs=1, **kwargs):
        ParallelProcessorMixin.__init__(self, n_jobs=n_jobs)
        # ... rest of init

    def process_gathers(self, gathers, velocity_model):
        # Automatically handles parallelism and memmap
        return self.parallel_map_with_shared(
            self._process_single_gather,
            gathers,
            velocity_model
        )
```

### 3.3 Medium-Priority: Export Worker Chunks

**Problem**: ProcessPoolExecutor requires manual crash handling.

**Solution**: Replace with Joblib for automatic retry.

**Implementation Location**: `utils/parallel_export/coordinator.py`

```python
# CURRENT
with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
    futures = [executor.submit(worker_func, chunk) for chunk in chunks]
    for future in as_completed(futures):
        try:
            result = future.result()
        except Exception as e:
            # Manual error handling
            pass

# PROPOSED (with Joblib)
from joblib import Parallel, delayed

results = Parallel(
    n_jobs=n_workers,
    backend='loky',  # Robust backend with crash recovery
    timeout=3600,    # 1 hour timeout per chunk
    verbose=10,      # Progress reporting
)(
    delayed(worker_func)(chunk) for chunk in chunks
)
```

**Benefits**: Automatic worker restart on crash, timeout handling.

### 3.4 Low-Priority: SEGY Import

**Not Recommended**: SEGY import has complex workflow with:
- File seeking requirements
- Segment-based file locks
- Custom progress reporting

ProcessPoolExecutor with spawn context is appropriate here.

### 3.5 Low-Priority: Batch Processing Main Loop

**Not Recommended**: Batch processing uses fork COW optimization which:
- Requires specific ProcessPoolExecutor configuration
- Uses pre-fork data sharing (incompatible with Joblib loky)
- Has streaming output requirements

Keep current architecture, but could use Joblib for inner processor loops.

---

## Part 4: GPU Considerations

### 4.1 Current GPU Architecture

GPU processors use PyTorch with:
```python
class GPUProcessor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def process(self, data):
        # Transfer to GPU
        tensor = torch.from_numpy(data).to(self.device)

        # Process
        result = self._gpu_operation(tensor)

        # Transfer back
        return result.cpu().numpy()
```

### 4.2 Memmap + GPU Integration

**Pattern**: Load from memmap, transfer subset to GPU.

```python
# Memmap source (disk-backed)
source = np.memmap('traces.dat', dtype='float32', mode='r', shape=(n_samples, n_traces))

# Process in GPU-sized batches
batch_size = 10000  # Traces per GPU batch
for i in range(0, n_traces, batch_size):
    batch = source[:, i:i+batch_size]  # Only loads this slice
    tensor = torch.from_numpy(batch).to(device)
    result = process_gpu(tensor)
    output[:, i:i+batch_size] = result.cpu().numpy()
```

### 4.3 Joblib + GPU

**Caution**: Joblib doesn't handle GPU memory well. For GPU:
- Use single-process GPU batching (current approach)
- Or manual ProcessPoolExecutor with GPU-per-worker assignment

---

## Part 5: Implementation Plan

### Phase 1: Quick Wins (Estimated Impact: High)

1. **Export Header Memmap** (`utils/parallel_export/header_vectorizer.py`)
   - Replace pickle with structured memmap
   - Files to modify: header_vectorizer.py, worker.py
   - Risk: Low (isolated subsystem)

2. **Processor Parallel Mixin** (`processors/parallel_mixin.py`)
   - Create standardized mixin
   - Update 5 most-used processors
   - Risk: Low (additive change)

### Phase 2: Medium Effort (Estimated Impact: Medium-High)

3. **SEGY Import Trace Memmap** (`utils/segy_import/multiprocess_import/worker.py`)
   - Add memmap option for large files
   - Files to modify: worker.py, coordinator.py
   - Risk: Medium (core import path)

4. **Export Joblib Conversion** (`utils/parallel_export/coordinator.py`)
   - Replace ProcessPoolExecutor with Joblib
   - Add crash recovery
   - Risk: Medium (changes parallel model)

### Phase 3: Optimization (Estimated Impact: Low-Medium)

5. **Batch Processing Sort Memmap** (optional)
   - Only if sort mappings become bottleneck
   - Fork COW already handles most cases

---

## Part 6: Code Location Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `processors/parallel_mixin.py` | Standardized Joblib parallel pattern |
| `utils/memmap_utils.py` | Memmap helper functions |

### Files to Modify

| File | Changes |
|------|---------|
| `utils/parallel_export/header_vectorizer.py` | Structured memmap for headers |
| `utils/parallel_export/worker.py` | Read memmap instead of pickle |
| `utils/parallel_export/coordinator.py` | Optional Joblib backend |
| `utils/segy_import/multiprocess_import/worker.py` | Memmap for large segments |
| `processors/denoise_3d.py` | Use ParallelProcessorMixin |
| `processors/nmo_processor.py` | Use ParallelProcessorMixin |
| `processors/cdp_stacker.py` | Use ParallelProcessorMixin |

---

## Part 7: Pros and Cons Analysis

### Memory-Mapped Arrays

**Pros**:
- Constant memory footprint
- OS-level caching optimization
- Works with spawn context
- Enables datasets larger than RAM

**Cons**:
- I/O bound for random access
- Requires disk space for temp files
- Slightly more complex error handling
- Cold cache performance penalty

### Joblib Integration

**Pros**:
- Automatic memmap for large arrays
- Built-in crash recovery
- Simpler API
- Progress reporting

**Cons**:
- Different backend than ProcessPoolExecutor
- Less control over process lifecycle
- Not suitable for fork COW patterns
- Adds dependency (though likely already installed)

---

## Part 8: Risk Assessment

| Change | Risk Level | Mitigation |
|--------|------------|------------|
| Export header memmap | LOW | Isolated subsystem, easy rollback |
| Processor mixin | LOW | Additive, opt-in per processor |
| SEGY import memmap | MEDIUM | Feature flag, fallback to current |
| Export Joblib | MEDIUM | Keep ProcessPoolExecutor as fallback |
| Batch processing changes | HIGH | Not recommended (already optimized) |

---

## Recommendations

### Immediate Actions (Low Risk, High Value)

1. **Create `processors/parallel_mixin.py`** - Standardize Joblib usage
2. **Create `utils/memmap_utils.py`** - Helper functions for memmap

### Short-term Actions (Medium Risk, High Value)

3. **Refactor export headers to memmap** - Major memory savings
4. **Add memmap option to SEGY import** - For very large files

### Future Considerations

5. **Joblib for export** - Only if crash recovery needed
6. **Leave batch processing as-is** - Fork COW is optimal

---

## Appendix A: Joblib Configuration Reference

```python
from joblib import Parallel, delayed, parallel_config

# Basic parallel execution
results = Parallel(n_jobs=4)(delayed(func)(x) for x in items)

# With memmap threshold (arrays > 1MB auto-memmaped)
with parallel_config(max_nbytes='1M'):
    results = Parallel(n_jobs=4)(delayed(func)(large_array, x) for x in items)

# With loky backend (crash recovery)
results = Parallel(n_jobs=4, backend='loky')(delayed(func)(x) for x in items)

# With timeout
results = Parallel(n_jobs=4, timeout=300)(delayed(func)(x) for x in items)
```

## Appendix B: Memmap Helper Functions

```python
"""utils/memmap_utils.py - Memory-mapped array utilities."""
import numpy as np
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import os

def create_temp_memmap(shape: Tuple[int, ...], dtype: str = 'float32') -> Tuple[np.memmap, Path]:
    """Create temporary memmap file.

    Returns:
        Tuple of (memmap array, path to temp file)
    """
    fd, path = tempfile.mkstemp(suffix='.dat')
    os.close(fd)

    arr = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    return arr, Path(path)

def open_memmap_readonly(path: Path, shape: Tuple[int, ...], dtype: str = 'float32') -> np.memmap:
    """Open existing memmap in read-only mode."""
    return np.memmap(str(path), dtype=dtype, mode='r', shape=shape)

def cleanup_memmap(arr: np.memmap, path: Path) -> None:
    """Safely cleanup memmap and temp file."""
    del arr
    if path.exists():
        path.unlink()
```

## Appendix C: Structured Header Dtype Example

```python
# Full SEGY header structured dtype
SEGY_HEADER_DTYPE = np.dtype([
    ('trace_sequence_line', 'i4'),
    ('trace_sequence_file', 'i4'),
    ('field_record', 'i4'),
    ('trace_number', 'i4'),
    ('energy_source_point', 'i4'),
    ('ensemble_number', 'i4'),
    ('trace_in_ensemble', 'i4'),
    ('trace_id', 'i2'),
    ('num_vert_stacked', 'i2'),
    ('num_horz_stacked', 'i2'),
    ('data_use', 'i2'),
    ('source_receiver_offset', 'i4'),
    ('receiver_elevation', 'i4'),
    ('source_elevation', 'i4'),
    ('source_depth', 'i4'),
    ('datum_receiver', 'i4'),
    ('datum_source', 'i4'),
    ('water_depth_source', 'i4'),
    ('water_depth_receiver', 'i4'),
    ('scalar_elevation', 'i2'),
    ('scalar_coordinates', 'i2'),
    ('source_x', 'i4'),
    ('source_y', 'i4'),
    ('receiver_x', 'i4'),
    ('receiver_y', 'i4'),
    ('coordinate_units', 'i2'),
    ('inline_3d', 'i4'),
    ('crossline_3d', 'i4'),
    ('cdp_x', 'i4'),
    ('cdp_y', 'i4'),
])
```

---

*Document generated: 2025-12-13*
*Based on analysis of SeisProc codebase parallel processing architecture*
