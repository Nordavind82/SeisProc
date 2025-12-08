"""
Worker process for importing a single SEGY segment.

Each worker runs in a separate process to bypass Python GIL.
Workers write directly to a shared Zarr array at their assigned offset.
"""

import gc
import numpy as np
import segyio
import zarr
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
from multiprocessing import Queue


@dataclass
class WorkerTask:
    """Task definition for a worker process."""
    segment_id: int
    segy_path: str
    output_dir: str           # Shared output directory
    start_trace: int          # Inclusive (global index)
    end_trace: int            # Exclusive (global index)
    n_samples: int            # Samples per trace
    header_mapping_dict: Dict[str, int]  # Serialized: {name: byte_location}
    chunk_size: int = 10000
    report_interval: int = 5000  # Report progress every N traces


@dataclass
class WorkerResult:
    """Result from a worker process."""
    segment_id: int
    headers_path: str         # Path to segment parquet (still per-segment)
    n_traces: int
    start_trace: int          # For header index adjustment
    elapsed_time: float
    success: bool
    error: Optional[str] = None


def import_segment(task: WorkerTask, progress_queue: Optional[Queue] = None) -> WorkerResult:
    """
    Import a single segment of SEGY file directly to shared output.

    This function runs in a separate process. Each worker:
    1. Opens its own SEGY file handle
    2. Reads only its assigned trace range
    3. Writes traces directly to shared Zarr at correct offset
    4. Writes headers to segment-specific parquet (merged later)
    5. Reports progress via queue

    Args:
        task: WorkerTask with segment details
        progress_queue: Optional queue for progress updates

    Returns:
        WorkerResult with outcome details
    """
    start_time = time.time()
    output_dir = Path(task.output_dir)

    try:
        n_traces = task.end_trace - task.start_trace
        n_samples = task.n_samples

        # Open shared Zarr array (pre-created by coordinator)
        traces_path = output_dir / "traces.zarr"
        z = zarr.open(str(traces_path), mode='r+')

        # Headers still go to segment-specific file (small, fast to merge)
        headers_path = output_dir / f"headers_segment_{task.segment_id}.parquet"

        # Open SEGY file (each process has its own handle)
        with segyio.open(task.segy_path, 'r', ignore_geometry=True) as f:
            f.mmap()

            # Process in chunks
            all_headers = []
            traces_done = 0

            for chunk_start in range(task.start_trace, task.end_trace, task.chunk_size):
                chunk_end = min(chunk_start + task.chunk_size, task.end_trace)
                chunk_size = chunk_end - chunk_start

                # Read traces into buffer
                traces_buffer = np.empty((n_samples, chunk_size), dtype=np.float32)
                for i, global_idx in enumerate(range(chunk_start, chunk_end)):
                    traces_buffer[:, i] = f.trace[global_idx]

                # Write directly to shared Zarr at global offset
                z[:, chunk_start:chunk_end] = traces_buffer

                # Read headers for this chunk
                chunk_headers = _read_headers_batch(
                    f, chunk_start, chunk_end, task.header_mapping_dict
                )
                all_headers.extend(chunk_headers)

                traces_done += chunk_size

                # Report progress
                if progress_queue is not None:
                    progress_queue.put((task.segment_id, traces_done))

                # Cleanup
                del traces_buffer
                if traces_done % 50000 == 0:
                    gc.collect()

        # Write headers to segment parquet with global trace indices
        _write_headers_parquet(all_headers, headers_path, task.start_trace)

        elapsed = time.time() - start_time

        return WorkerResult(
            segment_id=task.segment_id,
            headers_path=str(headers_path),
            n_traces=n_traces,
            start_trace=task.start_trace,
            elapsed_time=elapsed,
            success=True,
            error=None
        )

    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        return WorkerResult(
            segment_id=task.segment_id,
            headers_path="",
            n_traces=0,
            start_trace=task.start_trace,
            elapsed_time=elapsed,
            success=False,
            error=f"{str(e)}\n{traceback.format_exc()}"
        )


def _read_headers_batch(
    f,
    start: int,
    end: int,
    header_mapping: Dict[str, int]
) -> List[Dict[str, int]]:
    """
    Read headers in batch using segyio attributes.

    Args:
        f: Open segyio file handle
        start: Start trace index
        end: End trace index
        header_mapping: Dict of {name: byte_location}

    Returns:
        List of header dictionaries
    """
    n_traces = end - start
    headers = [{} for _ in range(n_traces)]

    for name, byte_loc in header_mapping.items():
        try:
            values = f.attributes(byte_loc)[start:end]
            for i, val in enumerate(values):
                headers[i][name] = int(val)
        except (KeyError, IndexError, TypeError):
            for i in range(n_traces):
                headers[i][name] = 0

    return headers


def _write_headers_parquet(headers: List[Dict], output_path: Path, start_trace: int):
    """
    Write headers to Parquet file with global trace indices.

    Args:
        headers: List of header dictionaries
        output_path: Path to output parquet file
        start_trace: Global start trace index for this segment
    """
    if not headers:
        return

    df = pd.DataFrame(headers)
    # Add global trace index directly
    df['trace_index'] = np.arange(start_trace, start_trace + len(headers))

    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
