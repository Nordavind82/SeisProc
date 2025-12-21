"""
CPU Worker Actor for Seismic Processing

Ray actor that processes seismic gathers using CPU-based processors.
Integrates with existing BaseProcessor hierarchy.

Supports:
- Multiple output modes (processed, noise, both)
- In-gather trace sorting
- Linear velocity mute
"""

import gc
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple, BinaryIO
from uuid import UUID

import numpy as np
import pandas as pd

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .base_worker import BaseWorkerActor, WorkerState, WorkerProgress

logger = logging.getLogger(__name__)


# =============================================================================
# Mute Functions
# =============================================================================

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

    Mute formula: T = |offset| / velocity

    Parameters
    ----------
    trace : np.ndarray
        Trace data array (modified in-place)
    offset : float
        Offset in meters
    sample_interval_ms : float
        Sample interval in milliseconds
    velocity : float
        Mute velocity in m/s
    top_mute : bool
        Zero samples before mute time
    bottom_mute : bool
        Zero samples after mute time
    taper_samples : int
        Number of samples for cosine taper
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

    Parameters
    ----------
    traces : np.ndarray
        Trace data array (n_samples, n_traces) - modified in-place
    offsets : np.ndarray
        Offset values for each trace
    sample_interval_ms : float
        Sample interval in milliseconds
    velocity : float
        Mute velocity in m/s
    top_mute : bool
        Zero samples before mute time
    bottom_mute : bool
        Zero samples after mute time
    taper_samples : int
        Number of samples for cosine taper
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


# =============================================================================
# Sort Functions
# =============================================================================

def compute_gather_sort_indices(
    headers_df: pd.DataFrame,
    start_trace: int,
    end_trace: int,
    sort_key: str,
    ascending: bool = True,
    secondary_key: Optional[str] = None,
    secondary_ascending: bool = True,
) -> np.ndarray:
    """
    Compute sort indices for traces within a gather.

    Parameters
    ----------
    headers_df : pd.DataFrame
        Full headers DataFrame
    start_trace : int
        First trace index of gather (inclusive)
    end_trace : int
        Last trace index of gather (inclusive)
    sort_key : str
        Primary sort key column name
    ascending : bool
        Sort direction for primary key
    secondary_key : str, optional
        Secondary sort key column name
    secondary_ascending : bool
        Sort direction for secondary key

    Returns
    -------
    np.ndarray
        Array of local indices (0 to n_traces-1) in sorted order
    """
    # Extract gather headers
    gather_headers = headers_df.iloc[start_trace:end_trace + 1]
    n_traces = len(gather_headers)

    if n_traces == 0:
        return np.array([], dtype=np.int64)

    # Get sort values
    if sort_key not in gather_headers.columns:
        # If sort key not found, return original order
        return np.arange(n_traces, dtype=np.int64)

    sort_values = gather_headers[sort_key].values.copy()

    if secondary_key and secondary_key in gather_headers.columns:
        # Multi-key sort using lexsort (sorts by last key first)
        secondary_values = gather_headers[secondary_key].values.copy()

        # Adjust for ascending/descending
        if not secondary_ascending:
            if np.issubdtype(secondary_values.dtype, np.number):
                secondary_values = -secondary_values
        if not ascending:
            if np.issubdtype(sort_values.dtype, np.number):
                sort_values = -sort_values

        sort_indices = np.lexsort((secondary_values, sort_values))
    else:
        # Single key sort
        sort_indices = np.argsort(sort_values)
        if not ascending:
            sort_indices = sort_indices[::-1]

    return sort_indices.astype(np.int64)


class StreamingSortWriter:
    """
    Streams sort mappings directly to disk to avoid memory accumulation.

    Binary format per entry:
        - gather_idx: int32 (4 bytes)
        - g_start: int64 (8 bytes)
        - g_end: int64 (8 bytes)
        - n_indices: uint32 (4 bytes)
        - sort_indices: int64[] (n_indices * 8 bytes)
    """

    HEADER_MAGIC = b'SSRT'  # Streaming Sort
    VERSION = 1

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file: Optional[BinaryIO] = None
        self.n_entries = 0
        self._count_pos = 0

    def open(self):
        """Open file for writing."""
        self.file = open(self.filepath, 'wb')
        # Write header
        self.file.write(self.HEADER_MAGIC)
        self.file.write(struct.pack('<I', self.VERSION))
        # Placeholder for entry count (will update on close)
        self._count_pos = self.file.tell()
        self.file.write(struct.pack('<I', 0))

    def write_mapping(
        self,
        gather_idx: int,
        g_start: int,
        g_end: int,
        sort_indices: np.ndarray
    ):
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

    Parameters
    ----------
    filepath : str
        Path to streaming sort file

    Returns
    -------
    list
        List of (gather_idx, g_start, g_end, sort_indices) tuples
    """
    import pickle

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


@dataclass
class MuteConfig:
    """Mute configuration."""
    velocity: float = 0.0       # Mute velocity in m/s (0 = disabled)
    top_mute: bool = False      # Apply top mute
    bottom_mute: bool = False   # Apply bottom mute
    taper_samples: int = 20     # Taper samples for transition
    target: str = 'output'      # 'output', 'input', or 'processed'


@dataclass
class SortConfig:
    """Sort configuration."""
    enabled: bool = False
    sort_key: str = 'offset'
    ascending: bool = True
    secondary_key: Optional[str] = None
    secondary_ascending: bool = True


@dataclass
class GatherTask:
    """Single gather processing task."""
    gather_idx: int
    start_trace: int
    end_trace: int
    n_traces: int


@dataclass
class GatherResult:
    """Result of processing a single gather."""
    gather_idx: int
    n_traces: int
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class WorkerResult:
    """Final result from worker processing."""
    worker_id: str
    n_gathers_processed: int
    n_traces_processed: int
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None
    gather_results: Optional[List[GatherResult]] = None


def create_cpu_worker_actor():
    """
    Create a Ray actor class for CPU processing.

    This function returns a Ray actor class that can be instantiated
    to process seismic gathers in parallel.
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available. Install with: pip install ray")

    @ray.remote
    class CPUWorkerActorImpl(BaseWorkerActor):
        """
        Ray actor for CPU-based seismic gather processing.

        Processes gathers using existing BaseProcessor subclasses,
        with proper cancellation checking and progress reporting.

        Supports:
        - Multiple output modes (processed, noise, both)
        - In-gather trace sorting (before processing)
        - Linear velocity mute (configurable target)
        """

        def __init__(
            self,
            job_id: UUID,
            worker_id: str,
            input_zarr_path: str,
            output_zarr_path: Optional[str],
            noise_zarr_path: Optional[str],
            processor_config: Dict[str, Any],
            sample_rate: float,
            metadata: Dict[str, Any],
            output_mode: str = 'processed',
            mute_config: Optional[Dict[str, Any]] = None,
            sort_config: Optional[Dict[str, Any]] = None,
            progress_callback: Optional[Callable] = None,
        ):
            """
            Initialize CPU worker.

            Parameters
            ----------
            job_id : UUID
                Parent job identifier
            worker_id : str
                Unique worker identifier
            input_zarr_path : str
                Path to input Zarr array
            output_zarr_path : str, optional
                Path to output Zarr array (for processed output)
            noise_zarr_path : str, optional
                Path to noise Zarr array (for noise output)
            processor_config : dict
                Processor configuration for reconstruction
            sample_rate : float
                Sample rate in milliseconds
            metadata : dict
                Seismic metadata
            output_mode : str
                'processed', 'noise', or 'both'
            mute_config : dict, optional
                Mute configuration with keys: velocity, top_mute, bottom_mute,
                taper_samples, target
            sort_config : dict, optional
                Sort configuration with keys: enabled, sort_key, ascending,
                secondary_key, secondary_ascending
            progress_callback : callable, optional
                Progress callback
            """
            super().__init__(job_id, worker_id, progress_callback)

            self._input_zarr_path = input_zarr_path
            self._output_zarr_path = output_zarr_path
            self._noise_zarr_path = noise_zarr_path
            self._processor_config = processor_config
            self._sample_rate = sample_rate
            self._metadata = metadata
            self._output_mode = output_mode

            # Parse mute config
            self._mute_config = MuteConfig(
                velocity=mute_config.get('velocity', 0.0) if mute_config else 0.0,
                top_mute=mute_config.get('top_mute', False) if mute_config else False,
                bottom_mute=mute_config.get('bottom_mute', False) if mute_config else False,
                taper_samples=mute_config.get('taper_samples', 20) if mute_config else 20,
                target=mute_config.get('target', 'output') if mute_config else 'output',
            )

            # Parse sort config
            self._sort_config = SortConfig(
                enabled=sort_config.get('enabled', False) if sort_config else False,
                sort_key=sort_config.get('sort_key', 'offset') if sort_config else 'offset',
                ascending=sort_config.get('ascending', True) if sort_config else True,
                secondary_key=sort_config.get('secondary_key') if sort_config else None,
                secondary_ascending=sort_config.get('secondary_ascending', True) if sort_config else True,
            )

            # Lazily initialized
            self._input_zarr = None
            self._output_zarr = None
            self._noise_zarr = None
            self._processor = None

            logger.info(
                f"CPUWorkerActor {worker_id} created for job {job_id} "
                f"(mute={'enabled' if self._mute_config.velocity > 0 else 'disabled'}, "
                f"sort={'enabled' if self._sort_config.enabled else 'disabled'})"
            )

        def _initialize(self):
            """Initialize Zarr arrays and processor."""
            import zarr
            from processors.base_processor import BaseProcessor

            logger.info(f"Worker {self.worker_id}: Initializing...")

            # Open Zarr arrays
            self._input_zarr = zarr.open(self._input_zarr_path, mode='r')

            if self._output_zarr_path and self._output_mode in ('processed', 'both'):
                self._output_zarr = zarr.open(self._output_zarr_path, mode='r+')

            if self._noise_zarr_path and self._output_mode in ('noise', 'both'):
                self._noise_zarr = zarr.open(self._noise_zarr_path, mode='r+')

            # Reconstruct processor
            self._processor = BaseProcessor.from_dict(self._processor_config)

            logger.info(
                f"Worker {self.worker_id}: Initialized with processor "
                f"{self._processor.__class__.__name__}"
            )

        def process(
            self,
            gather_tasks: List[Tuple[int, int, int]],
            headers_subset: Optional[Any] = None,
        ) -> WorkerResult:
            """
            Process a list of gathers.

            Parameters
            ----------
            gather_tasks : list
                List of (gather_idx, start_trace, end_trace) tuples
            headers_subset : DataFrame, optional
                Headers for this worker's gathers

            Returns
            -------
            WorkerResult
                Processing result
            """
            return self._run_with_lifecycle(
                self._process_gathers,
                gather_tasks,
                headers_subset,
            )

        def _process_gathers(
            self,
            gather_tasks: List[Tuple[int, int, int]],
            headers_subset: Optional[Any],
        ) -> WorkerResult:
            """Internal processing implementation with sort and mute support."""
            from models.seismic_data import SeismicData

            # Initialize if needed
            if self._processor is None:
                self._initialize()

            self._items_total = len(gather_tasks)
            self._items_processed = 0

            gather_results = []
            total_traces = 0

            output_processed = self._output_mode in ('processed', 'both')
            output_noise = self._output_mode in ('noise', 'both')
            noise_only = self._output_mode == 'noise'

            # Check if mute is enabled
            mute_enabled = self._mute_config.velocity > 0
            sort_enabled = self._sort_config.enabled

            for gather_idx, start_trace, end_trace in gather_tasks:
                gather_start = time.time()
                n_traces = end_trace - start_trace + 1

                try:
                    # Check cancellation before each gather
                    self._check_cancellation()

                    # Wait if paused
                    if not self._wait_if_paused():
                        raise InterruptedError("Processing interrupted by cancellation")

                    # Load gather traces - use view first, copy only when needed
                    # This avoids 2x memory overhead for read-only access
                    gather_traces_view = self._input_zarr[:, start_trace:end_trace + 1]

                    # Determine if we need a writable copy (for in-place mute/sort)
                    needs_copy = mute_enabled or sort_enabled
                    if needs_copy:
                        # Need contiguous array for in-place operations
                        gather_traces = np.ascontiguousarray(gather_traces_view, dtype=np.float32)
                    else:
                        # Read-only view is sufficient
                        gather_traces = np.asarray(gather_traces_view, dtype=np.float32)

                    # Get gather headers if available
                    gather_headers = None
                    if headers_subset is not None:
                        try:
                            # Headers are relative to segment start, need local indexing
                            local_start = start_trace - gather_tasks[0][1]
                            local_end = end_trace - gather_tasks[0][1]
                            gather_headers = headers_subset.iloc[local_start:local_end + 1]
                        except Exception as e:
                            logger.debug(f"Header extraction failed: {e}")

                    # Get offsets for mute (if enabled)
                    offsets = None
                    if mute_enabled and gather_headers is not None:
                        if 'offset' in gather_headers.columns:
                            offsets = gather_headers['offset'].values

                    # Apply input mute (before processing)
                    if mute_enabled and self._mute_config.target == 'input' and offsets is not None:
                        apply_mute_to_gather_inplace(
                            gather_traces, offsets, self._sample_rate,
                            self._mute_config.velocity, self._mute_config.top_mute,
                            self._mute_config.bottom_mute, self._mute_config.taper_samples
                        )

                    # Apply sorting before processing (important for spatial aperture)
                    sort_indices = None
                    if sort_enabled and gather_headers is not None:
                        sort_indices = compute_gather_sort_indices(
                            gather_headers, 0, len(gather_headers) - 1,
                            self._sort_config.sort_key, self._sort_config.ascending,
                            self._sort_config.secondary_key, self._sort_config.secondary_ascending
                        )
                        # Reorder traces
                        gather_traces = gather_traces[:, sort_indices]
                        # Also reorder offsets if we have them
                        if offsets is not None:
                            offsets = offsets[sort_indices]

                    # Create SeismicData
                    gather_data = SeismicData(
                        traces=gather_traces,
                        sample_rate=self._sample_rate,
                        metadata={
                            **self._metadata,
                            'gather_idx': gather_idx,
                            'n_traces': n_traces,
                            'headers_df': gather_headers,
                        }
                    )

                    # Process
                    processed = self._processor.process(gather_data)
                    processed_traces = processed.traces

                    # Apply mute to processed (before computing noise)
                    if mute_enabled and self._mute_config.target == 'processed' and offsets is not None:
                        apply_mute_to_gather_inplace(
                            processed_traces, offsets, self._sample_rate,
                            self._mute_config.velocity, self._mute_config.top_mute,
                            self._mute_config.bottom_mute, self._mute_config.taper_samples
                        )

                    # NOTE: We do NOT un-sort after processing. The sorted order is kept
                    # in the output, which is the intended behavior for spatial filtering.
                    # The original parallel_processing/worker.py also preserves sorted order.

                    # Apply output mute
                    if mute_enabled and self._mute_config.target == 'output' and offsets is not None:
                        apply_mute_to_gather_inplace(
                            processed_traces, offsets, self._sample_rate,
                            self._mute_config.velocity, self._mute_config.top_mute,
                            self._mute_config.bottom_mute, self._mute_config.taper_samples
                        )

                    # Write outputs based on mode
                    if noise_only:
                        # Memory-optimized: in-place subtraction
                        np.subtract(gather_traces, processed_traces, out=gather_traces)
                        if self._noise_zarr is not None:
                            self._noise_zarr[:, start_trace:end_trace + 1] = gather_traces
                    else:
                        # Standard mode
                        if output_processed and self._output_zarr is not None:
                            self._output_zarr[:, start_trace:end_trace + 1] = processed_traces

                        if output_noise and self._noise_zarr is not None:
                            noise_traces = gather_traces - processed_traces
                            self._noise_zarr[:, start_trace:end_trace + 1] = noise_traces
                            del noise_traces

                    # Cleanup
                    del gather_traces
                    del processed_traces
                    del processed
                    del gather_data

                    elapsed = time.time() - gather_start
                    gather_results.append(GatherResult(
                        gather_idx=gather_idx,
                        n_traces=n_traces,
                        elapsed_seconds=elapsed,
                        success=True,
                    ))

                    total_traces += n_traces
                    self._items_processed += 1

                    # Report progress
                    self._report_progress()

                    # Periodic GC
                    if self._items_processed % 10 == 0:
                        gc.collect()

                except Exception as e:
                    elapsed = time.time() - gather_start
                    gather_results.append(GatherResult(
                        gather_idx=gather_idx,
                        n_traces=n_traces,
                        elapsed_seconds=elapsed,
                        success=False,
                        error=str(e),
                    ))

                    # Check if it's a cancellation - propagate it
                    from ..cancellation import CancellationError
                    if isinstance(e, CancellationError):
                        raise

                    # Log other errors but continue
                    logger.warning(
                        f"Worker {self.worker_id}: Gather {gather_idx} failed: {e}"
                    )

            # Final result
            total_elapsed = time.time() - self._start_time if self._start_time else 0

            return WorkerResult(
                worker_id=self.worker_id,
                n_gathers_processed=self._items_processed,
                n_traces_processed=total_traces,
                elapsed_seconds=total_elapsed,
                success=True,
                gather_results=gather_results,
            )

    return CPUWorkerActorImpl
