"""
Worker process for importing a single SEGY segment.

Each worker runs in a separate process to bypass Python GIL.
Workers write directly to a shared Zarr array at their assigned offset.
"""

import gc
import struct
import numpy as np
import segyio
import zarr
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from multiprocessing import Queue


# Standard SEG-Y trace header field byte positions that segyio supports natively
# These are fields where segyio.TraceField enum value == byte position
SEGYIO_NATIVE_FIELDS = {
    1, 5, 9, 13, 17, 21, 25, 29, 31, 33, 35, 37, 41, 45, 49, 53, 57, 61, 65,
    69, 71, 73, 77, 81, 85, 89, 91, 95, 99, 103, 107, 109, 111, 113, 115, 117,
    119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
    149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177,
    179, 181, 183, 185, 187, 189, 193, 197, 201, 203, 205, 209, 211, 213, 217
}


@dataclass
class WorkerTask:
    """Task definition for a worker process."""
    segment_id: int
    segy_path: str
    output_dir: str           # Shared output directory
    start_trace: int          # Inclusive (global index)
    end_trace: int            # Exclusive (global index)
    n_samples: int            # Samples per trace
    header_mapping_dict: Dict[str, Tuple[int, str]]  # Serialized: {name: (byte_location, format)}
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
    header_mapping: Dict[str, Tuple[int, str]]
) -> List[Dict[str, int]]:
    """
    Read headers in batch by reading raw trace header bytes.

    This function reads raw header bytes directly instead of using segyio.attributes()
    because segyio.attributes() interprets byte positions as TraceField enum values,
    which doesn't work correctly for non-standard header positions.

    For example, byte position 201 in the raw header contains user-defined inline data,
    but segyio.attributes(201) returns ShotPointScalar (a 2-byte field at enum value 201).

    Args:
        f: Open segyio file handle
        start: Start trace index
        end: End trace index
        header_mapping: Dict of {name: (byte_location, format_code)}
            where format_code is 'i' for 4-byte int, 'h' for 2-byte int

    Returns:
        List of header dictionaries
    """
    n_traces = end - start
    headers = [{} for _ in range(n_traces)]

    # Format code sizes and struct format strings (big-endian as per SEG-Y)
    FORMAT_INFO = {
        'i': (4, '>i'),  # 4-byte signed int, big-endian
        'h': (2, '>h'),  # 2-byte signed int, big-endian
        'I': (4, '>I'),  # 4-byte unsigned int, big-endian
        'H': (2, '>H'),  # 2-byte unsigned int, big-endian
    }

    for trace_offset, local_idx in enumerate(range(start, end)):
        # Get raw 240-byte trace header via buf property
        header_obj = f.header[local_idx]
        raw_header = bytes(header_obj.buf)

        # Read each field from raw bytes
        for name, (byte_loc, fmt_code) in header_mapping.items():
            try:
                # Byte location is 1-based in SEG-Y, convert to 0-based
                offset = byte_loc - 1

                # Get format info
                size, struct_fmt = FORMAT_INFO.get(fmt_code, (4, '>i'))

                # Extract bytes and unpack
                raw_bytes = raw_header[offset:offset + size]

                if len(raw_bytes) == size:
                    value = struct.unpack(struct_fmt, raw_bytes)[0]
                    headers[trace_offset][name] = int(value)
                else:
                    headers[trace_offset][name] = 0

            except (KeyError, IndexError, TypeError, struct.error) as e:
                headers[trace_offset][name] = 0

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
