# Parallel Processing & Export Implementation Plan

## Overview

This plan implements multiprocess parallel processing and export for SeisProc, applying lessons learned from the successful SEGY import parallelization. The goal is to achieve 10-14x speedup on multi-core systems while maintaining memory efficiency.

**Key Principles:**
- Use `ProcessPoolExecutor` to bypass Python GIL
- Pre-create shared output arrays before workers start
- Workers write directly to their assigned regions
- Serialize processor config (not instances) for workers
- Proper cleanup of temporary files
- Seamless UI integration with progress tracking

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Control Panel   │  │ Progress Dialog  │  │ Status Bar        │  │
│  │ - Process btn   │  │ - Per-worker %   │  │ - Rate, ETA       │  │
│  │ - Export btn    │  │ - Cancel btn     │  │ - Phase info      │  │
│  └─────────────────┘  └──────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ParallelProcessingCoordinator                      │
│  1. Validate inputs (processor, data, paths)                        │
│  2. Partition gathers into N segments (one per CPU core)            │
│  3. Pre-create shared output Zarr array                             │
│  4. Serialize processor config to dict                              │
│  5. Launch N worker processes via ProcessPoolExecutor               │
│  6. Monitor progress via multiprocessing.Queue                      │
│  7. Handle cancellation and errors                                  │
│  8. Return ProcessingResult with output path                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   ProcessingWorker 0     │    │   ProcessingWorker N-1   │
│  - Reconstruct processor │    │  - Reconstruct processor │
│  - Process gathers 0-K   │    │  - Process gathers M-N   │
│  - Write to Zarr[0:K]    │    │  - Write to Zarr[M:N]    │
│  - Report progress       │    │  - Report progress       │
└──────────────────────────┘    └──────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ParallelExportCoordinator                         │
│  1. Partition traces into N segments                                │
│  2. Pre-convert headers DataFrame to dict of arrays (vectorized)    │
│  3. Launch N worker processes                                       │
│  4. Each worker writes segment to temp SEG-Y file                   │
│  5. Concatenate segment files into final output                     │
│  6. Cleanup temp files                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Task List

### Phase 1: Processor Serialization Infrastructure

#### Task 1.1: Add Serialization Methods to BaseProcessor
**File:** `processors/base_processor.py`

**Changes:**
```python
class BaseProcessor(ABC):
    # Existing code...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize processor configuration for multiprocess transfer."""
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'params': self.params.copy()
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BaseProcessor':
        """Reconstruct processor from serialized configuration."""
        import importlib
        module = importlib.import_module(config['module'])
        processor_class = getattr(module, config['class_name'])
        return processor_class(**config['params'])
```

**Validation:**
- Test serialization roundtrip for all processor types
- Ensure params dict contains only pickle-safe types

---

#### Task 1.2: Add Processor Registry
**File:** `processors/__init__.py`

**Changes:**
- Create `PROCESSOR_REGISTRY` dict mapping class names to classes
- Add `get_processor_class(name: str)` function
- Add `create_processor(config: dict)` factory function

**Purpose:** Enable workers to reconstruct processors without importing all modules.

---

### Phase 2: Parallel Processing Infrastructure

#### Task 2.1: Create Processing Package Structure
**Directory:** `utils/parallel_processing/`

**Files to create:**
```
utils/parallel_processing/
├── __init__.py           # Package exports
├── config.py             # ProcessingConfig, WorkerConfig dataclasses
├── partitioner.py        # GatherPartitioner - divide gathers by worker
├── worker.py             # process_gather_range() - worker function
├── coordinator.py        # ParallelProcessingCoordinator - orchestrator
└── result.py             # ProcessingResult dataclass
```

---

#### Task 2.2: Implement GatherPartitioner
**File:** `utils/parallel_processing/partitioner.py`

**Responsibilities:**
- Divide gathers into N roughly equal segments
- Balance by trace count (not gather count) for even workload
- Return list of `GatherSegment(segment_id, start_gather, end_gather, start_trace, end_trace)`

**Algorithm:**
```python
def partition_by_traces(ensembles_df: pd.DataFrame, n_segments: int) -> List[GatherSegment]:
    """Partition gathers so each segment has roughly equal trace count."""
    total_traces = ensembles_df['n_traces'].sum()
    target_per_segment = total_traces / n_segments

    segments = []
    current_segment_traces = 0
    segment_start = 0

    for i, row in ensembles_df.iterrows():
        current_segment_traces += row['n_traces']

        if current_segment_traces >= target_per_segment and len(segments) < n_segments - 1:
            segments.append(GatherSegment(
                segment_id=len(segments),
                start_gather=segment_start,
                end_gather=i,
                start_trace=ensembles_df.iloc[segment_start]['start_trace'],
                end_trace=row['end_trace']
            ))
            segment_start = i + 1
            current_segment_traces = 0

    # Last segment gets remainder
    # ...
```

---

#### Task 2.3: Implement Processing Worker Function
**File:** `utils/parallel_processing/worker.py`

**Key Design:**
- Top-level function (not method) for pickle compatibility
- Each worker opens its own data handles
- Writes directly to shared Zarr at assigned offset
- Reports progress via Queue

```python
@dataclass
class ProcessingTask:
    """Task definition for a worker process."""
    segment_id: int
    input_zarr_path: str      # Path to input traces.zarr
    output_zarr_path: str     # Path to shared output traces.zarr
    headers_parquet_path: str # Path to headers.parquet
    ensemble_index_path: str  # Path to ensemble_index.parquet
    processor_config: Dict    # Serialized processor config
    start_gather: int         # First gather index (inclusive)
    end_gather: int           # Last gather index (inclusive)
    start_trace: int          # Global start trace for output offset
    sample_rate: float        # For SeismicData construction

@dataclass
class ProcessingWorkerResult:
    """Result from a worker process."""
    segment_id: int
    n_gathers_processed: int
    n_traces_processed: int
    elapsed_time: float
    success: bool
    error: Optional[str] = None

def process_gather_range(
    task: ProcessingTask,
    progress_queue: Optional[Queue] = None
) -> ProcessingWorkerResult:
    """
    Process a range of gathers in a worker process.

    Each worker:
    1. Opens input Zarr (read-only)
    2. Opens output Zarr (read-write at assigned region)
    3. Loads ensemble index to find gather boundaries
    4. Reconstructs processor from config
    5. Processes each gather and writes to output
    6. Reports progress via queue
    """
    start_time = time.time()

    try:
        # Open data sources
        input_zarr = zarr.open(task.input_zarr_path, mode='r')
        output_zarr = zarr.open(task.output_zarr_path, mode='r+')
        ensemble_df = pd.read_parquet(task.ensemble_index_path)

        # Reconstruct processor
        processor = BaseProcessor.from_dict(task.processor_config)

        traces_done = 0

        for gather_idx in range(task.start_gather, task.end_gather + 1):
            # Get gather boundaries
            ensemble = ensemble_df.iloc[gather_idx]
            g_start = int(ensemble['start_trace'])
            g_end = int(ensemble['end_trace'])
            n_traces = g_end - g_start + 1

            # Load gather traces
            gather_traces = np.array(input_zarr[:, g_start:g_end+1])

            # Create SeismicData
            gather_data = SeismicData(
                traces=gather_traces,
                sample_rate=task.sample_rate
            )

            # Process
            processed = processor.process(gather_data)

            # Write to output at correct global offset
            output_zarr[:, g_start:g_end+1] = processed.traces

            traces_done += n_traces

            # Report progress
            if progress_queue is not None:
                progress_queue.put((task.segment_id, traces_done))

        return ProcessingWorkerResult(
            segment_id=task.segment_id,
            n_gathers_processed=task.end_gather - task.start_gather + 1,
            n_traces_processed=traces_done,
            elapsed_time=time.time() - start_time,
            success=True
        )

    except Exception as e:
        return ProcessingWorkerResult(
            segment_id=task.segment_id,
            n_gathers_processed=0,
            n_traces_processed=0,
            elapsed_time=time.time() - start_time,
            success=False,
            error=str(e)
        )
```

---

#### Task 2.4: Implement ParallelProcessingCoordinator
**File:** `utils/parallel_processing/coordinator.py`

**Responsibilities:**
- Validate inputs and configuration
- Pre-create shared output Zarr array
- Partition gathers across workers
- Launch and monitor workers
- Handle cancellation
- Return consolidated result

```python
@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    input_storage_dir: str      # Directory with traces.zarr, headers.parquet, etc.
    output_storage_dir: str     # Directory for processed output
    processor_config: Dict      # Serialized processor configuration
    n_workers: Optional[int] = None  # Auto-detect if None

@dataclass
class ProcessingResult:
    """Final result of processing operation."""
    success: bool
    output_dir: str
    output_zarr_path: str
    n_gathers: int
    n_traces: int
    elapsed_time: float
    throughput_traces_per_sec: float
    error: Optional[str] = None

class ParallelProcessingCoordinator:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.n_workers = config.n_workers or get_optimal_workers()
        self._cancel_requested = False

    def run(self, progress_callback=None) -> ProcessingResult:
        """Run parallel processing pipeline."""
        # 1. Load metadata and validate
        # 2. Partition gathers
        # 3. Pre-create output Zarr
        # 4. Launch workers with ProcessPoolExecutor
        # 5. Monitor progress
        # 6. Return result
```

---

### Phase 3: Parallel Export Infrastructure

#### Task 3.1: Create Export Package Structure
**Directory:** `utils/parallel_export/`

**Files to create:**
```
utils/parallel_export/
├── __init__.py           # Package exports
├── config.py             # ExportConfig dataclass
├── header_vectorizer.py  # Convert DataFrame to dict of arrays
├── worker.py             # export_trace_range() - worker function
├── coordinator.py        # ParallelExportCoordinator
└── merger.py             # Concatenate segment SEG-Y files
```

---

#### Task 3.2: Implement Header Vectorizer
**File:** `utils/parallel_export/header_vectorizer.py`

**Purpose:** Eliminate per-trace `iloc[i].to_dict()` overhead.

```python
def vectorize_headers(headers_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert DataFrame to dict of numpy arrays for fast indexed access.

    Instead of: headers_df.iloc[i].to_dict()  # O(n_columns) per trace
    Use: {col: arr[i] for col, arr in header_arrays.items()}  # O(1) per trace
    """
    # Pre-filter to only SEG-Y trace header fields
    segy_fields = {col for col in headers_df.columns
                   if hasattr(segyio.TraceField, col)}

    return {
        col: headers_df[col].values.astype(np.int32)
        for col in segy_fields
    }

def get_trace_headers(header_arrays: Dict[str, np.ndarray],
                      trace_idx: int) -> Dict[str, int]:
    """Get headers for a single trace from vectorized arrays."""
    return {col: int(arr[trace_idx]) for col, arr in header_arrays.items()}
```

---

#### Task 3.3: Implement Export Worker Function
**File:** `utils/parallel_export/worker.py`

**Strategy:**
- Each worker writes to its own segment SEG-Y file
- Uses vectorized header access
- Files are concatenated after all workers complete

```python
@dataclass
class ExportTask:
    """Task definition for export worker."""
    segment_id: int
    original_segy_path: str
    processed_zarr_path: str
    output_segment_path: str  # Temp file: segment_0.sgy, segment_1.sgy, etc.
    header_arrays: Dict[str, np.ndarray]  # Vectorized headers
    start_trace: int
    end_trace: int
    chunk_size: int = 10000

def export_trace_range(
    task: ExportTask,
    progress_queue: Optional[Queue] = None
) -> ExportWorkerResult:
    """Export a range of traces to a segment SEG-Y file."""
    # 1. Open original SEG-Y for spec/headers
    # 2. Open processed Zarr for data
    # 3. Create segment SEG-Y file
    # 4. Write traces with vectorized header access
    # 5. Report progress
```

---

#### Task 3.4: Implement SEG-Y Segment Merger
**File:** `utils/parallel_export/merger.py`

**Strategy:**
- SEG-Y files are binary format
- After first file (with text/binary headers), remaining files are trace data
- Can concatenate with header adjustment

```python
class SEGYSegmentMerger:
    """Merge segment SEG-Y files into single output file."""

    def merge(self, segment_paths: List[str], output_path: str):
        """
        Merge segment files into final output.

        Approach:
        1. Copy first segment (has text/bin headers) as base
        2. Append traces from remaining segments
        3. Update trace count in binary header
        """
```

---

#### Task 3.5: Implement ParallelExportCoordinator
**File:** `utils/parallel_export/coordinator.py`

```python
@dataclass
class ExportConfig:
    """Configuration for parallel export."""
    original_segy_path: str
    processed_zarr_path: str
    headers_parquet_path: str
    output_path: str
    n_workers: Optional[int] = None
    chunk_size: int = 10000

class ParallelExportCoordinator:
    def run(self, progress_callback=None) -> ExportResult:
        """Run parallel export pipeline."""
        # 1. Vectorize headers (once, in main process)
        # 2. Partition traces across workers
        # 3. Launch workers (each writes segment file)
        # 4. Merge segment files
        # 5. Cleanup temp files
        # 6. Return result
```

---

### Phase 4: UI Integration

#### Task 4.1: Update Main Window - Replace Batch Processing
**File:** `main_window.py`

**Changes:**
- Replace `_batch_process_all_gathers()` with parallel version
- Replace `_batch_process_and_export_streaming()` with parallel version
- Add worker progress display (per-worker progress bars or combined)
- Maintain cancellation support

**New Method:**
```python
def _batch_process_parallel(self):
    """Parallel batch processing using all CPU cores."""
    # 1. Validate processor and data
    # 2. Get output directory from settings or prompt
    # 3. Show confirmation with worker count
    # 4. Create progress dialog with per-worker info
    # 5. Run ParallelProcessingCoordinator
    # 6. Update UI with results
```

---

#### Task 4.2: Create Enhanced Progress Dialog
**File:** `views/parallel_progress_dialog.py`

**Features:**
- Overall progress bar
- Per-worker progress indicators
- Real-time throughput (traces/sec)
- ETA calculation
- Cancel button with graceful shutdown
- Phase indicator (Processing → Exporting → Cleanup)

---

#### Task 4.3: Update Control Panel
**File:** `views/control_panel.py`

**Changes:**
- Add "Parallel Processing" checkbox/option
- Show worker count setting
- Add "Process & Export" combined button

---

### Phase 5: Storage & Disk Management

#### Task 5.1: Implement Storage Manager
**File:** `utils/storage_manager.py`

**Responsibilities:**
- Manage temp directories for processing
- Track disk space usage
- Cleanup on completion or error
- Handle crash recovery (orphaned temp files)

```python
class ProcessingStorageManager:
    """Manages temporary and output storage for processing."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or AppSettings().get_effective_storage_directory()
        self.temp_dir = self.base_dir / 'temp'
        self.processing_dir = self.base_dir / 'processing'

    def create_processing_session(self, name: str) -> ProcessingSession:
        """Create isolated directory for a processing job."""
        session_id = f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        session_dir = self.processing_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return ProcessingSession(session_dir)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove orphaned processing sessions."""

    def get_disk_space_available(self) -> int:
        """Get available disk space in bytes."""

    def estimate_required_space(self, n_traces: int, n_samples: int) -> int:
        """Estimate disk space needed for processing."""
```

---

#### Task 5.2: Add Disk Space Validation
**File:** `utils/parallel_processing/coordinator.py`

**Changes:**
- Check available disk space before starting
- Estimate required space (input + output + temp)
- Warn user if space is tight
- Fail gracefully if insufficient space

---

### Phase 6: Old Code Cleanup

#### Task 6.1: Deprecate Memory-Intensive Batch Mode
**File:** `main_window.py`

**Changes:**
- Mark `_batch_process_all_gathers()` as deprecated
- Add warning if user has <16GB RAM and tries to use it
- Redirect to parallel streaming mode by default
- Keep for backwards compatibility but hide from menu

---

#### Task 6.2: Remove Redundant Export Paths
**File:** `utils/segy_import/segy_export.py`

**Changes:**
- Keep `export_from_zarr_chunked()` as fallback for single-threaded export
- Mark `AsyncSEGYExporter` as deprecated (threading doesn't help much)
- Add `export_parallel()` as new recommended entry point

---

#### Task 6.3: Consolidate Progress Callback Interfaces
**Files:** Multiple

**Changes:**
- Standardize progress callback signature across all modules:
  ```python
  ProgressCallback = Callable[[ProgressInfo], None]

  @dataclass
  class ProgressInfo:
      phase: str
      current: int
      total: int
      rate: float  # items/sec
      eta_seconds: float
      worker_progress: Optional[Dict[int, int]] = None
  ```

---

### Phase 7: Testing

#### Task 7.1: Unit Tests for Processor Serialization
**File:** `tests/test_processor_serialization.py`

**Tests:**
- Roundtrip serialization for each processor type
- Parameter preservation
- Error handling for invalid configs

---

#### Task 7.2: Integration Tests for Parallel Processing
**File:** `tests/test_parallel_processing.py`

**Tests:**
- Small dataset processing (verify correctness)
- Multi-worker coordination
- Cancellation handling
- Error recovery
- Disk cleanup on failure

---

#### Task 7.3: Integration Tests for Parallel Export
**File:** `tests/test_parallel_export.py`

**Tests:**
- Segment file creation
- Segment merging
- Header preservation
- Data integrity verification
- Compare output with sequential export

---

#### Task 7.4: Performance Benchmarks
**File:** `tests/benchmark_parallel.py`

**Benchmarks:**
- Processing speedup vs sequential (expect ~N workers speedup)
- Export speedup vs sequential
- Memory usage during processing
- Disk I/O patterns

---

## File Structure Summary

### New Files to Create:
```
utils/parallel_processing/
├── __init__.py
├── config.py
├── partitioner.py
├── worker.py
├── coordinator.py
└── result.py

utils/parallel_export/
├── __init__.py
├── config.py
├── header_vectorizer.py
├── worker.py
├── coordinator.py
└── merger.py

utils/storage_manager.py

views/parallel_progress_dialog.py

tests/test_processor_serialization.py
tests/test_parallel_processing.py
tests/test_parallel_export.py
tests/benchmark_parallel.py
```

### Files to Modify:
```
processors/base_processor.py      # Add serialization methods
processors/__init__.py            # Add processor registry
main_window.py                    # Replace batch processing methods
views/control_panel.py            # Add parallel processing options
utils/segy_import/segy_export.py  # Add parallel export entry point
models/app_settings.py            # Add parallel processing settings
```

### Files to Deprecate/Remove:
```
# Deprecate (keep but mark deprecated):
main_window.py::_batch_process_all_gathers()  # Memory-intensive mode
utils/segy_import/segy_export.py::AsyncSEGYExporter  # Threading not effective
```

---

## Implementation Order

1. **Phase 1** (Foundation): Processor serialization - Required for everything else
2. **Phase 2** (Core): Parallel processing infrastructure - Main speedup
3. **Phase 5** (Support): Storage management - Needed before UI integration
4. **Phase 4** (UI): Main window integration - User-facing changes
5. **Phase 3** (Enhancement): Parallel export - Additional speedup
6. **Phase 6** (Cleanup): Deprecate old code - After new code is stable
7. **Phase 7** (Quality): Testing - Throughout and after

---

## Expected Performance Improvements

| Operation | Current | Expected | Speedup |
|-----------|---------|----------|---------|
| Batch Processing (14 cores) | 1x | 10-12x | ~11x |
| Export (vectorized headers) | 1x | 3-5x | ~4x |
| Export (parallel + merge) | 1x | 6-10x | ~8x |
| Combined Process+Export | 1x | 8-12x | ~10x |

**For 12M trace dataset:**
- Current: ~4-6 hours processing + 2-3 hours export = 6-9 hours
- Expected: ~30-40 min processing + 20-30 min export = ~1 hour

---

## Risk Mitigation

1. **Data Corruption**: Workers write to non-overlapping regions; verify with checksums
2. **Memory Leaks**: Each worker is isolated process; cleanup automatic on exit
3. **Disk Full**: Pre-check available space; cleanup on failure
4. **Cancellation**: Graceful shutdown with Queue signaling; cleanup temp files
5. **Processor State**: Ensure all processors are stateless; add validation in serialization
