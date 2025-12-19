"""
Worker function for parallel gather processing.

Each worker runs in a separate process to bypass Python GIL.
Workers read from input Zarr and write directly to shared output Zarr.
Optionally sorts traces within gathers and saves sort mapping.

Memory-optimized version:
- Only loads required columns for sorting (not full headers)
- Streams sort mappings to disk instead of accumulating in memory
"""

import gc
import os
import time
import pickle
import struct
import traceback
import numpy as np
import zarr
import pandas as pd
import psutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, BinaryIO
from multiprocessing import Queue
from datetime import datetime

from .config import ProcessingTask, ProcessingWorkerResult, SortOptions

# Import shared data management from centralized module
from .shared_data import (
    get_shared_headers,
    get_shared_ensemble_index,
    get_shared_ensemble_arrays,
    set_shared_headers,  # Re-export for backward compatibility
    clear_shared_headers,  # Re-export for backward compatibility
)

# Import parquet_io for accelerated parquet reading
from utils.parquet_io import read_parquet


# =============================================================================
# WORKER DEBUG LOGGING - Each worker writes to its own log file
# =============================================================================
_WORKER_LOG_FILE = None
_WORKER_ID = None


def _init_worker_log(segment_id: int, output_dir: Optional[str] = None):
    """Initialize worker-specific debug log."""
    global _WORKER_LOG_FILE, _WORKER_ID
    _WORKER_ID = segment_id
    try:
        if output_dir:
            log_dir = Path(output_dir).parent
        else:
            log_dir = Path('/tmp')
        log_path = log_dir / f'worker_{segment_id}_debug.log'
        _WORKER_LOG_FILE = open(log_path, 'w')
        _worker_log(f"Worker {segment_id} debug log initialized at {datetime.now().isoformat()}")
        _worker_log(f"Log file: {log_path}")
        _worker_log(f"PID: {os.getpid()}")
    except Exception as e:
        pass  # Silent fail - don't crash worker


def _worker_log(msg: str, include_memory: bool = False):
    """Write debug message to worker log with immediate flush."""
    global _WORKER_LOG_FILE, _WORKER_ID
    if _WORKER_LOG_FILE is None:
        return
    try:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        mem_info = ""
        if include_memory:
            mem = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            proc_mem = process.memory_info().rss / (1024**3)
            mem_info = f" [PROC: {proc_mem:.2f}GB | SYS: {mem.used/(1024**3):.2f}GB/{mem.available/(1024**3):.2f}GB avail]"
        line = f"[{timestamp}] W{_WORKER_ID}: {msg}{mem_info}\n"
        _WORKER_LOG_FILE.write(line)
        _WORKER_LOG_FILE.flush()
        os.fsync(_WORKER_LOG_FILE.fileno())
    except Exception:
        pass


def _close_worker_log():
    """Close worker log file."""
    global _WORKER_LOG_FILE
    if _WORKER_LOG_FILE is not None:
        try:
            _worker_log("Worker log closing")
            _WORKER_LOG_FILE.close()
        except Exception:
            pass
        _WORKER_LOG_FILE = None


def apply_mute_to_trace(
    trace: np.ndarray,
    offset: float,
    sample_interval_ms: float,
    velocity: float,
    top_mute: bool,
    bottom_mute: bool,
    taper_samples: int
) -> np.ndarray:
    """
    Apply linear mute to a single trace.

    Mute formula: T = |offset| / velocity

    Args:
        trace: Trace data array
        offset: Offset in meters
        sample_interval_ms: Sample interval in milliseconds
        velocity: Mute velocity in m/s
        top_mute: Zero samples before mute time
        bottom_mute: Zero samples after mute time
        taper_samples: Number of samples for cosine taper

    Returns:
        Modified trace
    """
    n_samples = len(trace)
    result = trace.copy()

    # Calculate mute sample
    velocity_m_per_ms = velocity / 1000.0
    mute_time_ms = abs(offset) / velocity_m_per_ms
    mute_sample = int(mute_time_ms / sample_interval_ms)

    # Pre-compute taper if needed
    if taper_samples > 0:
        taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
    else:
        taper = np.array([])

    # Apply top mute
    if top_mute and mute_sample > 0:
        mute_end = min(mute_sample, n_samples)
        result[:mute_end] = 0

        # Apply taper after mute zone
        if taper_samples > 0 and mute_end < n_samples:
            taper_end = min(mute_end + taper_samples, n_samples)
            actual_taper_len = taper_end - mute_end
            if actual_taper_len > 0:
                result[mute_end:taper_end] *= taper[:actual_taper_len]

    # Apply bottom mute
    if bottom_mute and mute_sample < n_samples:
        mute_start = max(0, mute_sample)

        # Apply taper before mute zone
        if taper_samples > 0 and mute_start > 0:
            taper_start = max(0, mute_start - taper_samples)
            actual_taper_len = mute_start - taper_start
            if actual_taper_len > 0:
                result[taper_start:mute_start] *= taper[:actual_taper_len][::-1]

        # Zero after mute
        result[mute_start:] = 0

    return result


def apply_mute_to_gather(
    traces: np.ndarray,
    offsets: np.ndarray,
    sample_interval_ms: float,
    velocity: float,
    top_mute: bool,
    bottom_mute: bool,
    taper_samples: int
) -> np.ndarray:
    """
    Apply linear mute to all traces in a gather.

    Args:
        traces: Trace data array (n_samples, n_traces)
        offsets: Offset values for each trace
        sample_interval_ms: Sample interval in milliseconds
        velocity: Mute velocity in m/s
        top_mute: Zero samples before mute time
        bottom_mute: Zero samples after mute time
        taper_samples: Number of samples for cosine taper

    Returns:
        Muted traces
    """
    result = traces.copy()
    for i in range(traces.shape[1]):
        result[:, i] = apply_mute_to_trace(
            traces[:, i],
            offsets[i] if i < len(offsets) else 0.0,
            sample_interval_ms,
            velocity,
            top_mute,
            bottom_mute,
            taper_samples
        )
    return result


def apply_mute_to_trace_inplace(
    trace: np.ndarray,
    offset: float,
    sample_interval_ms: float,
    velocity: float,
    top_mute: bool,
    bottom_mute: bool,
    taper_samples: int
) -> None:
    """
    Apply linear mute to a single trace IN-PLACE (no copy).

    Memory-optimized version that modifies trace directly.

    Args:
        trace: Trace data array (modified in-place)
        offset: Offset in meters
        sample_interval_ms: Sample interval in milliseconds
        velocity: Mute velocity in m/s
        top_mute: Zero samples before mute time
        bottom_mute: Zero samples after mute time
        taper_samples: Number of samples for cosine taper
    """
    n_samples = len(trace)

    # Calculate mute sample
    velocity_m_per_ms = velocity / 1000.0
    mute_time_ms = abs(offset) / velocity_m_per_ms
    mute_sample = int(mute_time_ms / sample_interval_ms)

    # Pre-compute taper if needed
    if taper_samples > 0:
        taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
    else:
        taper = np.array([])

    # Apply top mute (in-place)
    if top_mute and mute_sample > 0:
        mute_end = min(mute_sample, n_samples)
        trace[:mute_end] = 0

        # Apply taper after mute zone
        if taper_samples > 0 and mute_end < n_samples:
            taper_end = min(mute_end + taper_samples, n_samples)
            actual_taper_len = taper_end - mute_end
            if actual_taper_len > 0:
                trace[mute_end:taper_end] *= taper[:actual_taper_len]

    # Apply bottom mute (in-place)
    if bottom_mute and mute_sample < n_samples:
        mute_start = max(0, mute_sample)

        # Apply taper before mute zone
        if taper_samples > 0 and mute_start > 0:
            taper_start = max(0, mute_start - taper_samples)
            actual_taper_len = mute_start - taper_start
            if actual_taper_len > 0:
                trace[taper_start:mute_start] *= taper[:actual_taper_len][::-1]

        # Zero after mute
        trace[mute_start:] = 0


def apply_mute_to_gather_inplace(
    traces: np.ndarray,
    offsets: np.ndarray,
    sample_interval_ms: float,
    velocity: float,
    top_mute: bool,
    bottom_mute: bool,
    taper_samples: int
) -> None:
    """
    Apply linear mute to all traces in a gather IN-PLACE.

    Memory-optimized version that modifies traces directly.

    Args:
        traces: Trace data array (n_samples, n_traces) - modified in-place
        offsets: Offset values for each trace
        sample_interval_ms: Sample interval in milliseconds
        velocity: Mute velocity in m/s
        top_mute: Zero samples before mute time
        bottom_mute: Zero samples after mute time
        taper_samples: Number of samples for cosine taper
    """
    for i in range(traces.shape[1]):
        apply_mute_to_trace_inplace(
            traces[:, i],
            offsets[i] if i < len(offsets) else 0.0,
            sample_interval_ms,
            velocity,
            top_mute,
            bottom_mute,
            taper_samples
        )


class StreamingSortWriter:
    """
    Streams sort mappings directly to disk to avoid memory accumulation.

    Instead of accumulating all sort mappings in a list and pickling at the end,
    this writes each mapping immediately to disk in a simple binary format.

    Format per entry:
        - gather_idx: int32 (4 bytes)
        - g_start: int64 (8 bytes)
        - g_end: int64 (8 bytes)
        - n_indices: int32 (4 bytes)
        - sort_indices: int64[] (n_indices * 8 bytes)
    """

    HEADER_MAGIC = b'SSRT'  # Streaming Sort
    VERSION = 1

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file: Optional[BinaryIO] = None
        self.n_entries = 0

    def open(self):
        """Open file for writing."""
        self.file = open(self.filepath, 'wb')
        # Write header
        self.file.write(self.HEADER_MAGIC)
        self.file.write(struct.pack('<I', self.VERSION))
        # Placeholder for entry count (will update on close)
        self._count_pos = self.file.tell()
        self.file.write(struct.pack('<I', 0))

    def write_mapping(self, gather_idx: int, g_start: int, g_end: int,
                      sort_indices: np.ndarray):
        """Write a single gather's sort mapping immediately to disk."""
        if self.file is None:
            raise RuntimeError("StreamingSortWriter not opened")

        # Ensure indices are int64
        indices = sort_indices.astype(np.int64)

        # Write entry
        self.file.write(struct.pack('<i', gather_idx))      # int32
        self.file.write(struct.pack('<q', g_start))         # int64
        self.file.write(struct.pack('<q', g_end))           # int64
        self.file.write(struct.pack('<I', len(indices)))    # uint32
        self.file.write(indices.tobytes())                  # int64[]

        self.n_entries += 1

        # Flush periodically to avoid buffer buildup
        if self.n_entries % 100 == 0:
            self.file.flush()

    def close(self):
        """Close file and update entry count."""
        if self.file is not None:
            # Update entry count in header
            self.file.seek(self._count_pos)
            self.file.write(struct.pack('<I', self.n_entries))
            self.file.close()
            self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def read_streaming_sort_file(filepath: str) -> List[Tuple[int, int, int, np.ndarray]]:
    """
    Read sort mappings from streaming format file.

    Args:
        filepath: Path to streaming sort file

    Returns:
        List of (gather_idx, g_start, g_end, sort_indices) tuples
    """
    mappings = []

    with open(filepath, 'rb') as f:
        # Read and verify header
        magic = f.read(4)
        if magic != StreamingSortWriter.HEADER_MAGIC:
            # Fall back to pickle format for backwards compatibility
            f.seek(0)
            return pickle.load(f)

        version = struct.unpack('<I', f.read(4))[0]
        n_entries = struct.unpack('<I', f.read(4))[0]

        # Read entries
        for _ in range(n_entries):
            gather_idx = struct.unpack('<i', f.read(4))[0]
            g_start = struct.unpack('<q', f.read(8))[0]
            g_end = struct.unpack('<q', f.read(8))[0]
            n_indices = struct.unpack('<I', f.read(4))[0]
            indices_bytes = f.read(n_indices * 8)
            sort_indices = np.frombuffer(indices_bytes, dtype=np.int64)

            mappings.append((gather_idx, g_start, g_end, sort_indices))

    return mappings


def compute_gather_sort_indices(
    headers_df: pd.DataFrame,
    start_trace: int,
    end_trace: int,
    sort_options: SortOptions
) -> np.ndarray:
    """
    Compute sort indices for traces within a gather.

    Args:
        headers_df: Full headers DataFrame
        start_trace: First trace index of gather (inclusive)
        end_trace: Last trace index of gather (inclusive)
        sort_options: Sort configuration

    Returns:
        Array of local indices (0 to n_traces-1) in sorted order
    """
    # Extract gather headers
    gather_headers = headers_df.iloc[start_trace:end_trace + 1]
    n_traces = len(gather_headers)

    if n_traces == 0:
        return np.array([], dtype=np.int64)

    # Get sort values
    sort_key = sort_options.sort_key
    if sort_key not in gather_headers.columns:
        # If sort key not found, return original order
        return np.arange(n_traces, dtype=np.int64)

    sort_values = gather_headers[sort_key].values

    if sort_options.secondary_key and sort_options.secondary_key in gather_headers.columns:
        # Multi-key sort using lexsort (sorts by last key first)
        secondary_values = gather_headers[sort_options.secondary_key].values

        # Adjust for ascending/descending
        if not sort_options.secondary_ascending:
            if np.issubdtype(secondary_values.dtype, np.number):
                secondary_values = -secondary_values
        if not sort_options.ascending:
            if np.issubdtype(sort_values.dtype, np.number):
                sort_values = -sort_values

        sort_indices = np.lexsort((secondary_values, sort_values))
    else:
        # Single key sort
        sort_indices = np.argsort(sort_values)
        if not sort_options.ascending:
            sort_indices = sort_indices[::-1]

    return sort_indices.astype(np.int64)


def process_gather_range(
    task: ProcessingTask,
    progress_queue: Optional[Queue] = None
) -> ProcessingWorkerResult:
    """
    Process a range of gathers in a worker process.

    This is a top-level function (not a method) for pickle compatibility
    with multiprocessing. Each worker:
    """
    # =========================================================================
    # SUPER EARLY CRASH LOGGING - Must be FIRST thing, before any heavy ops
    # This helps diagnose crashes that happen during worker startup
    # =========================================================================
    import os as _os
    from datetime import datetime as _dt
    _seg_id = getattr(task, 'segment_id', '?')
    _pid = _os.getpid()

    def _super_early_log(msg: str):
        """Write to shared crash log immediately - no dependencies."""
        try:
            import psutil as _ps
            with open('/tmp/seisproc_parallel_crash.log', 'a') as f:
                ts = _dt.now().strftime('%H:%M:%S.%f')[:-3]
                mem = _ps.virtual_memory()
                f.write(f"[{ts}] [MEM: {mem.used/(1024**3):.2f}GB/{mem.available/(1024**3):.2f}GB] "
                       f"WORKER-{_seg_id}(pid={_pid}): {msg}\n")
                f.flush()
                _os.fsync(f.fileno())
        except Exception:
            pass

    _super_early_log(">>> WORKER PROCESS STARTED - entering process_gather_range()")
    _super_early_log(f">>> Task segment_id={_seg_id}, gathers {task.start_gather}-{task.end_gather}")
    _super_early_log(f">>> Task attrs: input={task.input_zarr_path}")
    _super_early_log(f">>> Task attrs: output={task.output_zarr_path}")
    _super_early_log(f">>> Task attrs: noise={task.noise_zarr_path}")
    _super_early_log(f">>> Task attrs: processor_config keys={list(task.processor_config.keys()) if task.processor_config else 'None'}")

    """

    1. Opens input Zarr (read-only)
    2. Opens output Zarr (read-write at assigned region) - only if output_mode != 'noise'
    3. Loads ensemble index to find gather boundaries
    4. Reconstructs processor from serialized config
    5. Processes each gather and writes to output
    6. If sorting enabled, sorts traces within gather and STREAMS mapping to disk
    7. If noise output enabled, calculates and writes noise (input - processed)
    8. If mute enabled, applies mute to specified target
    9. Reports progress via queue

    MEMORY OPTIMIZATIONS:
    - Only loads required header columns for sorting (not full DataFrame)
    - Streams sort mappings to disk immediately (no accumulation in memory)
    - NOISE-ONLY MODE: Uses in-place subtraction to minimize memory (2 arrays instead of 3)
    - In-place mute operations when possible

    Output modes:
    - 'processed': Output processed traces only (default)
    - 'noise': Output ONLY noise (input - processed), memory-optimized
    - 'both': Output both processed and noise (legacy output_noise=True behavior)

    Args:
        task: ProcessingTask with all configuration
        progress_queue: Optional queue for progress updates (segment_id, traces_done)

    Returns:
        ProcessingWorkerResult with outcome details
    """
    # Initialize worker debug logging FIRST
    _super_early_log(">>> About to init worker log...")
    _init_worker_log(task.segment_id, task.noise_zarr_path or task.output_zarr_path)
    _super_early_log(">>> Worker log initialized")
    _worker_log("=" * 50)
    _worker_log("WORKER STARTING", include_memory=True)
    _super_early_log(">>> _worker_log WORKER STARTING done")
    _worker_log(f"Segment {task.segment_id}: gathers {task.start_gather}-{task.end_gather}")
    _worker_log(f"Input zarr: {task.input_zarr_path}")
    _worker_log(f"Output zarr: {task.output_zarr_path}")
    _worker_log(f"Noise zarr: {task.noise_zarr_path}")
    _worker_log(f"Output mode: {getattr(task, 'output_mode', 'processed')}")

    start_time = time.time()
    traces_done = 0
    gathers_done = 0

    sorting_enabled = task.sort_options is not None and task.sort_options.enabled
    sort_writer: Optional[StreamingSortWriter] = None

    # Check if mute is enabled
    mute_enabled = (
        task.mute_velocity > 0 and
        (task.mute_top or task.mute_bottom)
    )
    _worker_log(f"Mute enabled: {mute_enabled}")
    _worker_log(f"Sorting enabled: {sorting_enabled}")

    # Determine output mode - support both new output_mode and legacy output_noise
    output_mode = getattr(task, 'output_mode', 'processed')
    if output_mode == 'processed' and task.output_noise:
        output_mode = 'both'  # Legacy compatibility

    # Flags for what to output
    output_processed = output_mode in ('processed', 'both')
    output_noise = output_mode in ('noise', 'both')
    noise_only = output_mode == 'noise'
    _worker_log(f"Effective output_mode: {output_mode}, noise_only: {noise_only}")

    try:
        # Import here to avoid circular imports and ensure fresh import in worker
        _super_early_log(">>> TRY block entered - about to import processor modules")
        _worker_log("Importing processor modules...", include_memory=True)
        from processors.base_processor import BaseProcessor
        _super_early_log(">>> BaseProcessor imported")
        from models.seismic_data import SeismicData
        _super_early_log(">>> SeismicData imported")
        _worker_log("Imports complete", include_memory=True)

        # CRITICAL: Test and re-initialize CUDA in forked worker
        # The parent process may have initialized CUDA (e.g., for GPU status display),
        # which breaks CUDA in forked children. We need to test and potentially fix this.
        _super_early_log(">>> Testing CUDA availability in worker...")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            _super_early_log(f">>> torch.cuda.is_available() = {cuda_available}")
            _worker_log(f"CUDA available: {cuda_available}")

            if cuda_available:
                # Test if CUDA actually works by creating a small tensor
                try:
                    test_tensor = torch.zeros(1, device='cuda')
                    device_name = torch.cuda.get_device_name(0)
                    del test_tensor
                    torch.cuda.empty_cache()
                    _super_early_log(f">>> CUDA context test PASSED: {device_name}")
                    _worker_log(f"CUDA context verified: {device_name}")
                except Exception as cuda_e:
                    _super_early_log(f">>> CUDA context test FAILED: {cuda_e}")
                    _worker_log(f"CUDA context broken in worker: {cuda_e}, will use CPU")
            else:
                _super_early_log(">>> CUDA not available, worker will use CPU")
                _worker_log("CUDA not available, worker will use CPU fallback")
        except Exception as torch_e:
            _super_early_log(f">>> Error testing CUDA: {torch_e}")
            _worker_log(f"Error testing CUDA: {torch_e}")

        # Open data sources
        _super_early_log(">>> Opening input zarr...")
        _worker_log("Opening input zarr...", include_memory=True)
        input_zarr = zarr.open(task.input_zarr_path, mode='r')
        _super_early_log(f">>> Input zarr opened: shape={input_zarr.shape}")
        _worker_log(f"Input zarr opened: shape={input_zarr.shape}", include_memory=True)

        # Load ensemble index - try shared data first (fork COW), fall back to file
        _super_early_log(">>> Loading ensemble index...")
        _worker_log("Loading ensemble index...", include_memory=True)
        ensemble_df = get_shared_ensemble_index()
        if ensemble_df is not None:
            _super_early_log(f">>> Using pre-loaded ensemble index (fork COW - zero copy!): {len(ensemble_df)} rows")
            _worker_log(f"Using pre-loaded ensemble index: {len(ensemble_df)} rows (shared via fork COW)", include_memory=True)
        else:
            # Fall back to loading from file (spawn context or pre-load failed)
            _super_early_log(">>> Loading ensemble index from file...")
            ensemble_df = read_parquet(task.ensemble_index_path)
            _super_early_log(f">>> Ensemble index loaded from file: {len(ensemble_df)} rows")
            _worker_log(f"Ensemble index loaded from file: {len(ensemble_df)} rows", include_memory=True)

        # Open output zarr only if outputting processed
        output_zarr = None
        if output_processed and task.output_zarr_path:
            _worker_log("Opening output zarr...", include_memory=True)
            output_zarr = zarr.open(task.output_zarr_path, mode='r+')
            _worker_log(f"Output zarr opened: shape={output_zarr.shape}", include_memory=True)
        else:
            _worker_log("Skipping output zarr (not needed for this mode)")

        # Open noise zarr if noise output is enabled
        noise_zarr = None
        if output_noise and task.noise_zarr_path:
            _super_early_log(f">>> Opening noise zarr at {task.noise_zarr_path}...")
            _worker_log("Opening noise zarr...", include_memory=True)
            noise_zarr = zarr.open(task.noise_zarr_path, mode='r+')
            _super_early_log(f">>> Noise zarr opened: shape={noise_zarr.shape}")
            _worker_log(f"Noise zarr opened: shape={noise_zarr.shape}", include_memory=True)
        else:
            _super_early_log(">>> Skipping noise zarr (not needed)")
            _worker_log("Skipping noise zarr (not needed for this mode)")

        # Calculate sample interval in ms for mute calculation
        # NOTE: task.sample_rate is already in milliseconds (from SeismicData convention)
        _super_early_log(">>> Calculating sample_interval_ms...")
        sample_interval_ms = task.sample_rate  # sample_rate is already in milliseconds
        _super_early_log(f">>> sample_interval_ms = {sample_interval_ms}")
        _worker_log(f"Sample interval: {sample_interval_ms} ms")

        # Memory check right after zarr opens
        _super_early_log(">>> Memory check after zarr opens...")
        mem_after_zarr = psutil.virtual_memory()
        _super_early_log(f">>> POST-ZARR MEMORY: {mem_after_zarr.available/(1024**3):.2f}GB available")

        # Build list of required header columns
        _super_early_log(">>> Building header_columns list...")
        header_columns = []
        if sorting_enabled:
            header_columns.append(task.sort_options.sort_key)
            if task.sort_options.secondary_key:
                header_columns.append(task.sort_options.secondary_key)

        # Need offset for mute calculation
        if mute_enabled:
            # Add common offset column names
            for col in ['offset', 'OFFSET', 'Offset']:
                if col not in header_columns:
                    header_columns.append(col)

        # Check if FKK processor which needs all headers for volume building
        _super_early_log(">>> Checking processor class...")
        processor_class = task.processor_config.get('class_name', '') if task.processor_config else ''
        is_fkk_processor = processor_class == 'FKKProcessor'
        # Only load all headers for FKK - NOT just because header_columns is empty
        # Other processors (STFT, etc.) don't need headers if sorting/mute disabled
        load_all_headers = is_fkk_processor
        _super_early_log(f">>> processor_class={processor_class}, is_fkk={is_fkk_processor}, load_all={load_all_headers}")

        if is_fkk_processor:
            _super_early_log(">>> FKK processor detected - need all headers for volume building")
            _worker_log("FKK processor detected - will load all headers for volume building")

        # Load required header columns (or use pre-loaded shared headers from fork COW)
        headers_df = None
        _super_early_log(f">>> header_columns={header_columns}, load_all_headers={load_all_headers}")
        if header_columns or load_all_headers:
            # Check for pre-loaded shared headers (fork context optimization)
            # For FKK, coordinator now pre-loads ALL headers, so we can use shared headers
            _super_early_log(">>> Calling get_shared_headers()...")
            shared_headers, shared_cols = get_shared_headers()
            _super_early_log(f">>> shared_headers is {'NOT ' if shared_headers is None else ''}None, n_cols={len(shared_cols) if shared_cols else 0}")

            if shared_headers is not None:
                if is_fkk_processor:
                    # For FKK, coordinator should have pre-loaded ALL headers
                    # Check if we have enough columns (coordinator loads all for FKK)
                    if len(shared_cols) >= 10:  # Arbitrary threshold - real header files have many columns
                        _super_early_log(f">>> FKK: Using pre-loaded shared headers with {len(shared_cols)} columns (fork COW)")
                        _worker_log(f"FKK: Using pre-loaded shared headers: {len(shared_headers)} rows, "
                                   f"{len(shared_cols)} cols (shared via fork COW)", include_memory=True)
                        headers_df = shared_headers
                    else:
                        _super_early_log(f">>> FKK: Shared headers only have {len(shared_cols)} cols, need full headers")
                        _worker_log(f"FKK: Shared headers insufficient ({len(shared_cols)} cols), loading full headers...")
                else:
                    # For non-FKK, verify required columns are available
                    shared_cols_lower = {c.lower() for c in shared_cols}
                    missing_cols = []
                    for col in header_columns:
                        if col not in shared_cols and col.lower() not in shared_cols_lower:
                            missing_cols.append(col)

                    if not missing_cols:
                        _super_early_log(">>> Using pre-loaded shared headers (fork COW - zero copy!)")
                        _worker_log(f"Using pre-loaded shared headers: {len(shared_headers)} rows, "
                                   f"cols={shared_cols} (shared via fork COW)", include_memory=True)
                        headers_df = shared_headers
                    else:
                        _worker_log(f"Shared headers missing columns {missing_cols}, loading from file...")

            # Fall back to loading from file if shared headers not available
            _super_early_log(f">>> headers_df is {'NOT ' if headers_df is None else ''}None, checking if need to load...")
            if headers_df is None:
                if load_all_headers:
                    _super_early_log(f">>> About to load ALL headers from {task.headers_parquet_path}...")
                    _worker_log(f"Loading ALL headers from file (for FKK/volume processors)...", include_memory=True)
                    headers_df = read_parquet(task.headers_parquet_path)
                    _super_early_log(f">>> Headers loaded: {len(headers_df)} rows")
                    _worker_log(f"Full headers loaded: {len(headers_df)} rows, {len(headers_df.columns)} cols", include_memory=True)
                else:
                    _worker_log(f"Loading headers from file (columns: {header_columns})...", include_memory=True)
                    try:
                        # Use parquet_io for Polars acceleration
                        headers_df = read_parquet(
                            task.headers_parquet_path,
                            columns=header_columns
                        )
                        _worker_log(f"Headers loaded: {len(headers_df)} rows, {len(headers_df.columns)} cols", include_memory=True)
                    except Exception as e:
                        # Some columns may not exist, try loading all
                        _worker_log(f"Partial column load failed ({e}), loading full headers...", include_memory=True)
                        headers_df = read_parquet(task.headers_parquet_path)
                        _worker_log(f"Full headers loaded: {len(headers_df)} rows, {len(headers_df.columns)} cols", include_memory=True)

        _super_early_log(">>> Headers section complete")

        # Open streaming sort writer
        if sorting_enabled and task.sort_mapping_path:
            _worker_log(f"Opening streaming sort writer: {task.sort_mapping_path}")
            sort_writer = StreamingSortWriter(task.sort_mapping_path)
            sort_writer.open()

        # Reconstruct processor from config
        _super_early_log(">>> About to reconstruct processor...")
        _worker_log("Reconstructing processor from config...", include_memory=True)
        _worker_log(f"Processor config: {task.processor_config.get('class_name', 'unknown')}")

        # Memory check BEFORE processor reconstruction
        mem_before_proc = psutil.virtual_memory()
        _super_early_log(f">>> BEFORE processor: {mem_before_proc.available/(1024**3):.2f}GB available")

        processor = BaseProcessor.from_dict(task.processor_config)

        # Memory check AFTER processor reconstruction
        mem_after_proc = psutil.virtual_memory()
        _super_early_log(f">>> AFTER processor: {mem_after_proc.available/(1024**3):.2f}GB available")
        _super_early_log(f">>> Processor: {processor.get_description()}")
        _worker_log(f"Processor created: {processor.get_description()}", include_memory=True)

        # Memory check before processing starts
        mem_check = psutil.virtual_memory()
        _super_early_log(f">>> Memory check before processing: {mem_check.available/(1024**3):.2f}GB available")
        _worker_log(f"Memory before processing: {mem_check.available/(1024**3):.2f}GB available", include_memory=True)
        if mem_check.available / (1024**3) < 1.0:
            _super_early_log(f">>> WARNING: Very low memory ({mem_check.available/(1024**3):.2f}GB), may crash!")
            _worker_log(f"WARNING: Very low memory! Only {mem_check.available/(1024**3):.2f}GB available", include_memory=True)
            # Force garbage collection to try to free memory
            gc.collect()
            mem_after_gc = psutil.virtual_memory()
            _worker_log(f"After gc.collect(): {mem_after_gc.available/(1024**3):.2f}GB available", include_memory=True)

        # Process each gather in assigned range
        _worker_log(f"Starting gather processing loop: {task.start_gather} to {task.end_gather}", include_memory=True)
        for gather_idx in range(task.start_gather, task.end_gather + 1):
            # Get gather boundaries from ensemble index
            ensemble = ensemble_df.iloc[gather_idx]
            g_start = int(ensemble['start_trace'])
            g_end = int(ensemble['end_trace'])
            n_traces = g_end - g_start + 1

            # Log every 10 gathers or first gather for detailed memory tracking
            log_this_gather = (gathers_done == 0) or (gathers_done % 10 == 0)
            if log_this_gather:
                _worker_log(f"Processing gather {gather_idx} ({n_traces} traces, trace range {g_start}-{g_end})", include_memory=True)

            # Load gather traces from input
            if log_this_gather:
                _worker_log(f"  Loading {n_traces} traces from input zarr...", include_memory=True)
            gather_traces = np.array(input_zarr[:, g_start:g_end + 1])
            if log_this_gather:
                _worker_log(f"  Gather loaded: shape={gather_traces.shape}, dtype={gather_traces.dtype}, "
                           f"size={gather_traces.nbytes/(1024*1024):.2f}MB", include_memory=True)

            # Get gather headers (needed for FKKProcessor and mute)
            gather_headers = None
            if headers_df is not None:
                gather_headers = headers_df.iloc[g_start:g_end + 1]

            # Apply sorting BEFORE processing if enabled
            # This ensures processors with spatial aperture (STFT, etc.) see sorted neighbors
            sort_indices = None
            if sorting_enabled and headers_df is not None:
                sort_indices = compute_gather_sort_indices(
                    headers_df, g_start, g_end, task.sort_options
                )
                # Reorder traces and headers before processing
                gather_traces = gather_traces[:, sort_indices]
                if gather_headers is not None:
                    gather_headers = gather_headers.iloc[sort_indices].reset_index(drop=True)
                if log_this_gather:
                    _worker_log(f"  Applied pre-processing sort by {task.sort_options.sort_key}", include_memory=True)

            # Get offsets for mute if needed
            offsets = None
            if mute_enabled and gather_headers is not None:
                for col in ['offset', 'OFFSET', 'Offset']:
                    if col in gather_headers.columns:
                        offsets = gather_headers[col].values.astype(np.float32)
                        break
                if offsets is None:
                    offsets = np.zeros(n_traces, dtype=np.float32)

            # Create SeismicData for processing
            if log_this_gather:
                _worker_log(f"  Creating SeismicData object...", include_memory=True)
            gather_data = SeismicData(
                traces=gather_traces,
                sample_rate=task.sample_rate,
                metadata={
                    **task.metadata,
                    'gather_idx': gather_idx,
                    'n_traces': n_traces,
                    'headers_df': gather_headers  # Include headers for FKKProcessor (may be None)
                }
            )

            # Process the gather
            if log_this_gather:
                _worker_log(f"  Processing gather with {processor.__class__.__name__}...", include_memory=True)
            processed = processor.process(gather_data)
            if log_this_gather:
                _worker_log(f"  Processing complete", include_memory=True)

            # Get processed traces
            processed_traces = processed.traces
            if log_this_gather:
                _worker_log(f"  Processed traces: shape={processed_traces.shape}, "
                           f"size={processed_traces.nbytes/(1024*1024):.2f}MB", include_memory=True)

            # === MEMORY-OPTIMIZED OUTPUT HANDLING ===

            if noise_only:
                if log_this_gather:
                    _worker_log(f"  NOISE-ONLY mode: performing in-place subtraction...", include_memory=True)
                # NOISE-ONLY MODE: Memory-optimized path
                # Use in-place operations to minimize memory usage
                # Result will be stored in gather_traces (reused array)

                if mute_enabled and offsets is not None:
                    if task.mute_target == 'input':
                        # Mute input in-place, then subtract
                        apply_mute_to_gather_inplace(
                            gather_traces, offsets, sample_interval_ms,
                            task.mute_velocity, task.mute_top, task.mute_bottom,
                            task.mute_taper
                        )
                        # In-place subtraction: gather_traces -= processed_traces
                        np.subtract(gather_traces, processed_traces, out=gather_traces)
                    elif task.mute_target == 'processed':
                        # Mute processed in-place, then subtract from input
                        apply_mute_to_gather_inplace(
                            processed_traces, offsets, sample_interval_ms,
                            task.mute_velocity, task.mute_top, task.mute_bottom,
                            task.mute_taper
                        )
                        np.subtract(gather_traces, processed_traces, out=gather_traces)
                    else:  # 'output' - subtract first, then mute result
                        np.subtract(gather_traces, processed_traces, out=gather_traces)
                        apply_mute_to_gather_inplace(
                            gather_traces, offsets, sample_interval_ms,
                            task.mute_velocity, task.mute_top, task.mute_bottom,
                            task.mute_taper
                        )
                else:
                    # No mute - in-place subtraction
                    np.subtract(gather_traces, processed_traces, out=gather_traces)

                # Now gather_traces contains the noise - free processed immediately
                del processed_traces
                del processed
                del gather_data
                gc.collect()

                # Write sort mapping if enabled (traces already sorted before processing)
                if sorting_enabled and sort_indices is not None and sort_writer is not None:
                    sort_writer.write_mapping(gather_idx, g_start, g_end, sort_indices)

                # Write noise to output (noise goes to noise_zarr which is the primary output)
                if noise_zarr is not None:
                    noise_zarr[:, g_start:g_end + 1] = gather_traces

                # Cleanup
                del gather_traces

            else:
                # STANDARD MODE: processed only, or both processed and noise
                noise_traces = None

                if output_noise and noise_zarr is not None:
                    # Calculate noise with mute handling
                    if mute_enabled and offsets is not None:
                        if task.mute_target == 'input':
                            # Mute applied to input before subtraction
                            muted_input = apply_mute_to_gather(
                                gather_traces, offsets, sample_interval_ms,
                                task.mute_velocity, task.mute_top, task.mute_bottom,
                                task.mute_taper
                            )
                            noise_traces = muted_input - processed_traces
                            del muted_input  # Free immediately
                        elif task.mute_target == 'processed':
                            # Mute applied to processed before subtraction
                            muted_processed = apply_mute_to_gather(
                                processed_traces, offsets, sample_interval_ms,
                                task.mute_velocity, task.mute_top, task.mute_bottom,
                                task.mute_taper
                            )
                            noise_traces = gather_traces - muted_processed
                            del muted_processed  # Free immediately
                        else:  # 'output' - apply mute to noise result
                            noise_traces = gather_traces - processed_traces
                            apply_mute_to_gather_inplace(
                                noise_traces, offsets, sample_interval_ms,
                                task.mute_velocity, task.mute_top, task.mute_bottom,
                                task.mute_taper
                            )
                    else:
                        # No mute - simple subtraction
                        noise_traces = gather_traces - processed_traces

                # Free input traces early if not needed for further operations
                del gather_traces
                del gather_data

                # Write sort mapping if enabled (traces already sorted before processing)
                if sorting_enabled and sort_indices is not None and sort_writer is not None:
                    sort_writer.write_mapping(gather_idx, g_start, g_end, sort_indices)

                # Write processed traces to output
                if output_zarr is not None:
                    output_zarr[:, g_start:g_end + 1] = processed_traces

                # Write noise traces if enabled
                if noise_traces is not None and noise_zarr is not None:
                    noise_zarr[:, g_start:g_end + 1] = noise_traces

                # Cleanup
                del processed_traces
                del processed
                if noise_traces is not None:
                    del noise_traces

            # Update progress
            traces_done += n_traces
            gathers_done += 1

            # Report progress periodically
            if progress_queue is not None:
                progress_queue.put((task.segment_id, traces_done, gathers_done))

            # Periodic garbage collection
            if gathers_done % 10 == 0:
                gc.collect()

        # Close sort writer
        sort_mapping_path = None
        if sort_writer is not None:
            sort_writer.close()
            sort_mapping_path = task.sort_mapping_path

        elapsed = time.time() - start_time

        # Log successful completion
        _worker_log("=" * 50)
        _worker_log("WORKER COMPLETED SUCCESSFULLY", include_memory=True)
        _worker_log(f"Gathers processed: {gathers_done}")
        _worker_log(f"Traces processed: {traces_done}")
        _worker_log(f"Elapsed time: {elapsed:.2f}s")
        _worker_log(f"Throughput: {traces_done/elapsed:.0f} traces/sec" if elapsed > 0 else "N/A")
        _worker_log("=" * 50)
        _close_worker_log()

        return ProcessingWorkerResult(
            segment_id=task.segment_id,
            n_gathers_processed=gathers_done,
            n_traces_processed=traces_done,
            elapsed_time=elapsed,
            success=True,
            error=None,
            sort_mapping_path=sort_mapping_path
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_trace = traceback.format_exc()

        # Log the error
        _worker_log("=" * 50)
        _worker_log("EXCEPTION IN WORKER", include_memory=True)
        _worker_log(f"Exception type: {type(e).__name__}")
        _worker_log(f"Exception message: {str(e)}")
        _worker_log(f"Gathers completed before error: {gathers_done}")
        _worker_log(f"Traces completed before error: {traces_done}")
        _worker_log(f"Full traceback:\n{error_trace}")
        _worker_log("=" * 50)
        _close_worker_log()

        # Ensure sort writer is closed on error
        if sort_writer is not None:
            try:
                sort_writer.close()
            except Exception:
                pass

        return ProcessingWorkerResult(
            segment_id=task.segment_id,
            n_gathers_processed=gathers_done,
            n_traces_processed=traces_done,
            elapsed_time=elapsed,
            success=False,
            error=f"{str(e)}\n{error_trace}",
            sort_mapping_path=None
        )
