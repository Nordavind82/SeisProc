# Multiprocess SEGY Import Implementation Plan

## Overview

Replace threading with multiprocessing to bypass Python GIL. Use smart partitioning with boundary probing instead of full header scan.

## Problem

Current implementation shows ~40% CPU on one core, others idle. Python GIL prevents true parallelism with threading.

## Solution

1. Divide file into N segments at gather boundaries
2. Each worker process independently reads segment and writes output
3. Merge segment outputs into final dataset

---

## Architecture

### Pipeline

```
Calculate split points (instant)
        ↓
Probe split points for gather boundaries (<1 sec)
        ↓
Adjust segment sizes
        ↓
Launch N worker processes (parallel read/write)
        ↓
Merge outputs into final dataset
```

### Smart Partitioning (No Full Scan)

Instead of scanning all 12M headers:
- Calculate raw split points: 12M / 14 = 857,142 per worker
- At each split point, probe backward ~100 headers to find gather boundary
- Total probes: 14 × 100 = 1,400 headers (<1 second)

---

## File Structure

```
utils/segy_import/
├── __init__.py                    # Update exports
├── segy_reader.py                 # KEEP - used by workers
├── header_mapping.py              # KEEP - unchanged
├── data_storage.py                # KEEP - used for merge
├── parallel_reader.py             # DELETE - threading doesn't help
└── multiprocess_import/           # NEW PACKAGE
    ├── __init__.py
    ├── partitioner.py             # Smart segment partitioning
    ├── worker.py                  # Worker process logic
    ├── coordinator.py             # Main orchestrator
    └── merger.py                  # Merge segment outputs
```

---

## Implementation Tasks

### Task 1: Create Package Structure

**Files to create:**
- `utils/segy_import/multiprocess_import/__init__.py`

**Content:**
```python
from .partitioner import SmartPartitioner, Segment
from .worker import import_segment, WorkerTask, WorkerResult
from .coordinator import ParallelImportCoordinator, ImportConfig, ImportProgress
from .merger import OutputMerger

__all__ = [
    'SmartPartitioner',
    'Segment',
    'import_segment',
    'WorkerTask',
    'WorkerResult',
    'ParallelImportCoordinator',
    'ImportConfig',
    'ImportProgress',
    'OutputMerger',
]
```

---

### Task 2: Implement SmartPartitioner

**File:** `utils/segy_import/multiprocess_import/partitioner.py`

**Classes:**
```python
@dataclass
class Segment:
    segment_id: int
    start_trace: int      # Inclusive
    end_trace: int        # Exclusive
    n_traces: int

@dataclass
class PartitionConfig:
    segy_path: str
    n_segments: int
    total_traces: int
    ensemble_key: str = 'cdp'       # Header field for gather detection
    max_probe_distance: int = 10000  # Max traces to search for boundary

class SmartPartitioner:
    """Partitions SEGY file into segments at gather boundaries."""

    def __init__(self, config: PartitionConfig):
        pass

    def partition(self) -> List[Segment]:
        """
        1. Calculate raw split points
        2. Probe each for gather boundary
        3. Return adjusted segments
        """
        pass

    def _find_gather_boundary(self, approx_trace: int) -> int:
        """
        Probe backward from approx_trace to find gather start.
        Returns trace index where new gather begins.
        """
        pass

    def _read_header_value(self, trace_idx: int) -> int:
        """Read single header value at trace index."""
        pass
```

**Logic for `_find_gather_boundary`:**
1. Read header at `approx_trace`
2. Read header at `approx_trace - 1`
3. If different → boundary at `approx_trace`
4. If same → keep going back until different or max_probe_distance
5. Return first trace of current gather

---

### Task 3: Implement Worker

**File:** `utils/segy_import/multiprocess_import/worker.py`

**Data classes:**
```python
@dataclass
class WorkerTask:
    segment_id: int
    segy_path: str
    output_dir: str           # Worker writes to output_dir/segment_{id}/
    start_trace: int
    end_trace: int
    header_mapping_dict: dict # Serialized HeaderMapping
    chunk_size: int = 10000

@dataclass
class WorkerResult:
    segment_id: int
    traces_path: str          # Path to segment zarr
    headers_path: str         # Path to segment parquet
    n_traces: int
    elapsed_time: float
    success: bool
    error: Optional[str] = None
```

**Worker function (must be top-level for pickle):**
```python
def import_segment(task: WorkerTask, progress_queue: Queue = None) -> WorkerResult:
    """
    Import a single segment. Runs in separate process.

    1. Create output directory
    2. Open SEGY file (own handle)
    3. Read traces in chunks from start_trace to end_trace
    4. Write to segment-specific Zarr and Parquet
    5. Report progress via queue
    6. Return result
    """
    pass
```

**Key points:**
- Each worker opens its own SEGY file handle
- Writes to `{output_dir}/segment_{id}/traces.zarr`
- Writes to `{output_dir}/segment_{id}/headers.parquet`
- Progress reported via `multiprocessing.Queue`

---

### Task 4: Implement Coordinator

**File:** `utils/segy_import/multiprocess_import/coordinator.py`

**Data classes:**
```python
@dataclass
class ImportConfig:
    segy_path: str
    output_dir: str
    header_mapping: HeaderMapping
    ensemble_key: str = 'cdp'
    n_workers: int = None     # Auto-detect if None
    chunk_size: int = 10000

@dataclass
class ImportProgress:
    phase: str                # 'partitioning', 'importing', 'merging'
    current_traces: int
    total_traces: int
    active_workers: int
    elapsed_time: float
    eta_seconds: float

@dataclass
class ImportResult:
    success: bool
    output_dir: str
    n_traces: int
    n_gathers: int
    elapsed_time: float
    error: Optional[str] = None
```

**Main class:**
```python
class ParallelImportCoordinator:
    """Orchestrates parallel multiprocess import."""

    def __init__(self, config: ImportConfig):
        self.config = config
        self.n_workers = config.n_workers or self._detect_workers()
        self._cancel_event = multiprocessing.Event()

    def run(self, progress_callback=None) -> ImportResult:
        """
        Full import pipeline:
        1. Get file info
        2. Partition into segments
        3. Launch parallel workers
        4. Monitor progress
        5. Merge outputs
        6. Cleanup temp files
        """
        pass

    def _partition(self, n_traces: int) -> List[Segment]:
        pass

    def _run_workers(self, segments: List[Segment]) -> List[WorkerResult]:
        """
        Launch workers using ProcessPoolExecutor.
        Monitor progress via Queue.
        """
        pass

    def _monitor_progress(self, progress_queue, n_traces, callback):
        """Poll progress queue and call callback."""
        pass

    def cancel(self):
        """Signal workers to stop."""
        self._cancel_event.set()

    def _detect_workers(self) -> int:
        """Return optimal worker count (CPU cores - 2)."""
        pass
```

---

### Task 5: Implement Merger

**File:** `utils/segy_import/multiprocess_import/merger.py`

**Class:**
```python
class OutputMerger:
    """Merges segment outputs into final dataset."""

    def __init__(self, output_dir: str, results: List[WorkerResult]):
        self.output_dir = Path(output_dir)
        self.results = sorted(results, key=lambda r: r.segment_id)

    def merge(self, progress_callback=None) -> Tuple[str, str]:
        """
        Merge all segments into final output.
        Returns (traces_path, headers_path)
        """
        traces_path = self.merge_traces()
        headers_path = self.merge_headers()
        self.cleanup_segments()
        return traces_path, headers_path

    def merge_traces(self) -> str:
        """
        Concatenate segment Zarr arrays into final traces.zarr

        Strategy:
        - Create final array with total shape
        - Copy each segment to correct offset
        """
        pass

    def merge_headers(self) -> str:
        """
        Concatenate segment Parquet files.
        Adjust trace_index for global indexing.
        """
        pass

    def cleanup_segments(self):
        """Remove temporary segment directories."""
        pass
```

**Merge strategy for traces:**
1. Calculate total traces from all segments
2. Create final Zarr array with full shape
3. Copy each segment to correct offset: `final[:, offset:offset+n] = segment[:]`

**Merge strategy for headers:**
1. Read all segment parquets
2. Adjust `trace_index` column: `segment_df['trace_index'] += offset`
3. Concatenate all DataFrames
4. Write final parquet

---

### Task 6: Update Package Exports

**File:** `utils/segy_import/__init__.py`

**Changes:**
```python
# ADD import
from .multiprocess_import import (
    ParallelImportCoordinator,
    ImportConfig,
    ImportProgress,
)

# ADD to __all__
__all__ = [
    # ... existing exports ...

    # Multiprocess import
    'ParallelImportCoordinator',
    'ImportConfig',
    'ImportProgress',
]

# REMOVE (after integration complete)
# from .parallel_reader import ParallelSEGYReader, create_parallel_reader, get_optimal_workers
```

---

### Task 7: Integrate into Import Dialog

**File:** `views/segy_import_dialog.py`

**Changes to imports:**
```python
# ADD
from utils.segy_import.multiprocess_import import (
    ParallelImportCoordinator,
    ImportConfig,
    ImportProgress,
)

# REMOVE
from utils.segy_import.parallel_reader import ParallelSEGYReader, get_optimal_workers
```

**Revert `_update_reader` to simple SEGYReader:**
```python
def _update_reader(self):
    """Update SEG-Y reader with current header mapping."""
    if self.segy_file:
        self._update_mapping_from_table()
        # Simple reader for preview and small operations
        self.reader = SEGYReader(self.segy_file, self.header_mapping)
```

**Replace `_import_streaming` method:**
```python
def _import_streaming(self, output_dir: str, file_info: dict, ensemble_keys: List[str]):
    """Import using parallel multiprocessing."""

    n_traces = file_info['n_traces']

    # Create progress dialog
    progress = QProgressDialog(
        "Parallel import in progress...",
        "Cancel", 0, n_traces, self
    )
    progress.setWindowModality(Qt.WindowModality.WindowModal)

    # Configure parallel import
    config = ImportConfig(
        segy_path=str(self.segy_file),
        output_dir=output_dir,
        header_mapping=self.header_mapping,
        ensemble_key=ensemble_keys[0] if ensemble_keys else None,
        chunk_size=DEFAULT_CHUNK_SIZE
    )

    coordinator = ParallelImportCoordinator(config)

    # Progress callback
    def on_progress(prog: ImportProgress):
        if progress.wasCanceled():
            coordinator.cancel()
            return
        progress.setValue(prog.current_traces)
        progress.setLabelText(
            f"Phase: {prog.phase}\n"
            f"Progress: {prog.current_traces:,}/{prog.total_traces:,}\n"
            f"Workers: {prog.active_workers}"
        )

    # Run import
    try:
        result = coordinator.run(progress_callback=on_progress)

        if not result.success:
            raise RuntimeError(result.error)

        # Load and emit result...

    except Exception as e:
        QMessageBox.critical(self, "Import Error", str(e))
```

---

### Task 8: Delete Old Threading Code

**File to DELETE:** `utils/segy_import/parallel_reader.py`

**Clean up `__init__.py`:**
```python
# REMOVE these lines:
from .parallel_reader import ParallelSEGYReader, create_parallel_reader, get_optimal_workers

# REMOVE from __all__:
'ParallelSEGYReader',
'create_parallel_reader',
'get_optimal_workers',
```

---

### Task 9: Add Fallback for Small Files

**In `segy_import_dialog.py`:**

```python
PARALLEL_THRESHOLD = 100_000  # Use parallel for files > 100K traces

def _do_import(self, output_dir: str, file_info: dict, ensemble_keys: List[str]):
    """Choose import method based on file size."""

    if file_info['n_traces'] > PARALLEL_THRESHOLD:
        self._import_parallel(output_dir, file_info, ensemble_keys)
    else:
        self._import_simple(output_dir, file_info, ensemble_keys)
```

Keep simple streaming import for small files (less overhead than spawning processes).

---

### Task 10: Testing

**Create test file:** `tests/test_multiprocess_import.py`

**Test cases:**
1. `test_smart_partitioner` - verify boundary detection
2. `test_single_worker` - verify worker imports segment correctly
3. `test_parallel_coordinator` - full parallel import
4. `test_merger` - verify merge produces correct output
5. `test_cancellation` - verify cancel stops workers
6. `test_small_file_fallback` - verify small files use simple import

---

## Execution Order

| Order | Task | Estimated Time |
|-------|------|----------------|
| 1 | Create package structure | 5 min |
| 2 | Implement SmartPartitioner | 30 min |
| 3 | Implement Worker | 45 min |
| 4 | Implement Coordinator | 60 min |
| 5 | Implement Merger | 30 min |
| 6 | Update package exports | 5 min |
| 7 | Integrate into dialog | 30 min |
| 8 | Delete old threading code | 5 min |
| 9 | Add fallback for small files | 15 min |
| 10 | Testing | 60 min |

**Total: ~4-5 hours**

---

## Expected Performance

| Metric | Current | After Implementation |
|--------|---------|---------------------|
| CPU Usage | 40% one core | 80%+ all cores |
| Import Time (12M) | ~20 min | ~5-7 min |
| Speedup | 1x | 3-4x |

---

## Risk Mitigation

1. **Worker crashes** - Coordinator catches exceptions, reports failed segment
2. **Memory pressure** - Each worker uses ~150MB, total ~2GB for 14 workers
3. **Disk contention** - NVMe handles parallel writes well
4. **Uneven segments** - Smart partitioning balances within ~1% variance
5. **No ensemble key** - Falls back to even split (no boundary adjustment)

---

## Rollback Plan

If issues arise, revert to previous implementation:
1. Restore `parallel_reader.py` from git
2. Revert `segy_import_dialog.py` changes
3. Revert `__init__.py` changes
4. Delete `multiprocess_import/` package
