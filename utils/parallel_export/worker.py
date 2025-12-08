"""
Worker function for parallel SEG-Y export.

Each worker runs in a separate process to bypass Python GIL.
Workers write their segment of traces to individual temp SEG-Y files.
"""

import gc
import os
import time
import numpy as np
import zarr
import segyio
from pathlib import Path
from typing import Optional, Dict
from multiprocessing import Queue

from .config import ExportTask, ExportWorkerResult
from .header_vectorizer import HeaderVectorizer


def export_trace_range(
    task: ExportTask,
    progress_queue: Optional[Queue] = None
) -> ExportWorkerResult:
    """
    Export a range of traces to a segment SEG-Y file.

    This is a top-level function (not a method) for pickle compatibility
    with multiprocessing. Each worker:

    1. Opens processed Zarr (read-only)
    2. Loads vectorized headers from pickle
    3. Opens original SEG-Y for binary/text header info
    4. Creates segment SEG-Y file
    5. Writes traces with vectorized header access
    6. Reports progress via queue

    Args:
        task: ExportTask with all configuration
        progress_queue: Optional queue for progress updates (segment_id, traces_done)

    Returns:
        ExportWorkerResult with outcome details
    """
    start_time = time.time()
    traces_done = 0

    try:
        # Load vectorized headers
        header_arrays = HeaderVectorizer.load(Path(task.header_arrays_path))

        # Open processed Zarr for trace data
        processed_zarr = zarr.open(task.processed_zarr_path, mode='r')

        # Create segment spec
        n_traces = task.end_trace - task.start_trace + 1

        # Build specification for this segment
        spec = segyio.spec()
        spec.samples = list(range(task.n_samples))
        spec.format = task.data_format
        spec.tracecount = n_traces

        # Create segment file
        with segyio.create(task.output_segment_path, spec) as dst:
            # For first segment only: copy text/binary headers from original
            if task.is_first_segment:
                with segyio.open(task.original_segy_path, 'r', ignore_geometry=True) as src:
                    # Copy text header
                    dst.text[0] = src.text[0]
                    # Copy binary header
                    dst.bin = src.bin
                    # Update sample count in case it differs
                    dst.bin[segyio.BinField.Samples] = task.n_samples
                    dst.bin[segyio.BinField.Interval] = task.sample_interval
            else:
                # Non-first segments: set minimal binary header
                dst.bin[segyio.BinField.Samples] = task.n_samples
                dst.bin[segyio.BinField.Interval] = task.sample_interval
                dst.bin[segyio.BinField.Format] = task.data_format

            # Write traces in chunks for memory efficiency
            chunk_size = 1000
            local_idx = 0

            for chunk_start in range(task.start_trace, task.end_trace + 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, task.end_trace + 1)
                chunk_n = chunk_end - chunk_start

                # Load chunk of trace data
                chunk_data = np.array(processed_zarr[:, chunk_start:chunk_end])

                # Write each trace with vectorized headers
                for i in range(chunk_n):
                    global_trace_idx = chunk_start + i

                    # Write trace data
                    dst.trace[local_idx] = chunk_data[:, i]

                    # Write headers using vectorized access
                    for field_name, arr in header_arrays.items():
                        if hasattr(segyio.TraceField, field_name):
                            field = getattr(segyio.TraceField, field_name)
                            dst.header[local_idx][field] = int(arr[global_trace_idx])

                    local_idx += 1
                    traces_done += 1

                # Report progress periodically
                if progress_queue is not None:
                    progress_queue.put((task.segment_id, traces_done))

                # Cleanup chunk memory
                del chunk_data
                gc.collect()

        elapsed = time.time() - start_time

        # Get output file size
        file_size = Path(task.output_segment_path).stat().st_size

        return ExportWorkerResult(
            segment_id=task.segment_id,
            n_traces_exported=traces_done,
            output_path=task.output_segment_path,
            file_size_bytes=file_size,
            elapsed_time=elapsed,
            success=True,
            error=None
        )

    except Exception as e:
        import traceback
        elapsed = time.time() - start_time

        return ExportWorkerResult(
            segment_id=task.segment_id,
            n_traces_exported=traces_done,
            output_path=task.output_segment_path,
            file_size_bytes=0,
            elapsed_time=elapsed,
            success=False,
            error=f"{str(e)}\n{traceback.format_exc()}"
        )
