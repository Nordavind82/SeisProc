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
import time
import pickle
import struct
import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, BinaryIO
from multiprocessing import Queue

from .config import ProcessingTask, ProcessingWorkerResult, SortOptions


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

    1. Opens input Zarr (read-only)
    2. Opens output Zarr (read-write at assigned region)
    3. Loads ensemble index to find gather boundaries
    4. Reconstructs processor from serialized config
    5. Processes each gather and writes to output
    6. If sorting enabled, sorts traces within gather and STREAMS mapping to disk
    7. Reports progress via queue

    MEMORY OPTIMIZATIONS:
    - Only loads required header columns for sorting (not full DataFrame)
    - Streams sort mappings to disk immediately (no accumulation in memory)

    Args:
        task: ProcessingTask with all configuration
        progress_queue: Optional queue for progress updates (segment_id, traces_done)

    Returns:
        ProcessingWorkerResult with outcome details
    """
    start_time = time.time()
    traces_done = 0
    gathers_done = 0

    sorting_enabled = task.sort_options is not None and task.sort_options.enabled
    sort_writer: Optional[StreamingSortWriter] = None

    try:
        # Import here to avoid circular imports and ensure fresh import in worker
        from processors.base_processor import BaseProcessor
        from models.seismic_data import SeismicData

        # Open data sources
        input_zarr = zarr.open(task.input_zarr_path, mode='r')
        output_zarr = zarr.open(task.output_zarr_path, mode='r+')
        ensemble_df = pd.read_parquet(task.ensemble_index_path)

        # MEMORY OPTIMIZATION: Only load columns needed for sorting
        headers_df = None
        if sorting_enabled:
            # Build list of required columns
            sort_columns = [task.sort_options.sort_key]
            if task.sort_options.secondary_key:
                sort_columns.append(task.sort_options.secondary_key)

            # Only load required columns - massive memory savings!
            headers_df = pd.read_parquet(
                task.headers_parquet_path,
                columns=sort_columns
            )

            # Open streaming sort writer
            if task.sort_mapping_path:
                sort_writer = StreamingSortWriter(task.sort_mapping_path)
                sort_writer.open()

        # Reconstruct processor from config
        processor = BaseProcessor.from_dict(task.processor_config)

        # Process each gather in assigned range
        for gather_idx in range(task.start_gather, task.end_gather + 1):
            # Get gather boundaries from ensemble index
            ensemble = ensemble_df.iloc[gather_idx]
            g_start = int(ensemble['start_trace'])
            g_end = int(ensemble['end_trace'])
            n_traces = g_end - g_start + 1

            # Load gather traces from input
            gather_traces = np.array(input_zarr[:, g_start:g_end + 1])

            # Create SeismicData for processing
            gather_data = SeismicData(
                traces=gather_traces,
                sample_rate=task.sample_rate,
                metadata={
                    **task.metadata,
                    'gather_idx': gather_idx,
                    'n_traces': n_traces
                }
            )

            # Process the gather
            processed = processor.process(gather_data)

            # Get processed traces
            processed_traces = processed.traces

            # Apply sorting if enabled
            if sorting_enabled and headers_df is not None:
                sort_indices = compute_gather_sort_indices(
                    headers_df, g_start, g_end, task.sort_options
                )

                # Reorder traces according to sort indices
                processed_traces = processed_traces[:, sort_indices]

                # STREAM mapping to disk immediately (no memory accumulation!)
                if sort_writer is not None:
                    sort_writer.write_mapping(gather_idx, g_start, g_end, sort_indices)

            # Write processed (and optionally sorted) traces to output
            output_zarr[:, g_start:g_end + 1] = processed_traces

            # Update progress
            traces_done += n_traces
            gathers_done += 1

            # Report progress periodically
            if progress_queue is not None:
                progress_queue.put((task.segment_id, traces_done, gathers_done))

            # Cleanup to manage memory
            del gather_traces
            del gather_data
            del processed
            if gathers_done % 10 == 0:
                gc.collect()

        # Close sort writer
        sort_mapping_path = None
        if sort_writer is not None:
            sort_writer.close()
            sort_mapping_path = task.sort_mapping_path

        elapsed = time.time() - start_time

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
        import traceback
        elapsed = time.time() - start_time

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
            error=f"{str(e)}\n{traceback.format_exc()}",
            sort_mapping_path=None
        )
