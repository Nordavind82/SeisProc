"""
Worker function for parallel SEG-Y export.

Each worker runs in a separate process to bypass Python GIL.
Workers write their segment of traces to individual temp SEG-Y files.

Supports:
- Standard processed data export
- Noise export (input - processed)
- On-the-fly mute application (top/bottom with taper)
"""

import gc
import os
import time
import numpy as np
import zarr
import segyio
import psutil
from pathlib import Path
from typing import Optional, Dict
from multiprocessing import Queue

from .config import ExportTask, ExportWorkerResult
from .header_vectorizer import HeaderVectorizer


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
    Apply linear mute to a single trace (in-place modification).

    Mute formula: T = |offset| / velocity

    Args:
        trace: Trace data array (modified in place)
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
        trace[:mute_end] = 0

        # Apply taper after mute zone
        if taper_samples > 0 and mute_end < n_samples:
            taper_end = min(mute_end + taper_samples, n_samples)
            actual_taper_len = taper_end - mute_end
            if actual_taper_len > 0:
                trace[mute_end:taper_end] *= taper[:actual_taper_len]

    # Apply bottom mute
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

    return trace


def get_adaptive_chunk_size(n_samples: int, n_workers: int = 4) -> int:
    """
    Calculate adaptive chunk size based on available memory and trace depth.

    Aims to keep per-worker memory usage reasonable while maintaining throughput.

    Args:
        n_samples: Number of samples per trace
        n_workers: Number of parallel workers

    Returns:
        Optimal chunk size in traces
    """
    try:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        available_mb = 4000  # Conservative default: 4GB available

    # Reserve 2GB for system + app, divide remaining among workers
    usable_mb = max(500, (available_mb - 2000) / n_workers)

    # Calculate bytes per trace chunk
    bytes_per_trace = n_samples * 4  # float32

    # Target chunk memory: 100-200MB per chunk for good I/O performance
    target_chunk_mb = min(200, usable_mb * 0.5)

    chunk_size = int((target_chunk_mb * 1024 * 1024) / bytes_per_trace)

    # Clamp to reasonable range
    return max(100, min(5000, chunk_size))


def _create_default_text_header(n_samples: int, sample_interval: int) -> bytes:
    """
    Create a default EBCDIC text header for SEG-Y export.

    Args:
        n_samples: Number of samples per trace
        sample_interval: Sample interval in microseconds

    Returns:
        3200-byte EBCDIC text header
    """
    from datetime import datetime

    sample_rate_ms = sample_interval / 1000.0
    record_length_ms = (n_samples - 1) * sample_rate_ms

    lines = [
        "C 1 CLIENT: SEISPROC PROCESSED DATA",
        "C 2 LINE:",
        "C 3 AREA:",
        f"C 4 PROCESSED: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "C 5",
        f"C 6 RECORD LENGTH: {record_length_ms:.0f} MS",
        f"C 7 SAMPLE INTERVAL: {sample_rate_ms:.2f} MS",
        f"C 8 SAMPLES PER TRACE: {n_samples}",
        "C 9 DATA FORMAT: IEEE FLOAT",
        "C10",
        "C11 PROCESSING HISTORY:",
        "C12 - EXPORTED FROM SEISPROC INTERNAL FORMAT",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
        "C28",
        "C29",
        "C30",
        "C31",
        "C32",
        "C33",
        "C34",
        "C35",
        "C36",
        "C37",
        "C38",
        "C39",
        "C40 END EBCDIC HEADER",
    ]

    # Pad each line to 80 characters
    text = ""
    for line in lines:
        text += line.ljust(80)[:80]

    # Convert to EBCDIC
    # segyio expects bytes in EBCDIC encoding
    try:
        ebcdic_text = text.encode('cp500')  # EBCDIC encoding
    except Exception:
        # Fallback: just use ASCII bytes (some readers handle it)
        ebcdic_text = text.encode('ascii')

    return ebcdic_text


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

    Supports:
    - 'processed': Export processed data as-is
    - 'noise': Export input - processed (noise estimation)
    - Mute: Apply top/bottom mute based on offset and velocity

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

        # Open input Zarr if noise export
        input_zarr = None
        if task.export_type == 'noise' and task.input_zarr_path:
            input_zarr = zarr.open(task.input_zarr_path, mode='r')

        # Check if mute is enabled
        apply_mute = (
            task.mute_velocity is not None and
            task.mute_velocity > 0 and
            (task.mute_top or task.mute_bottom)
        )

        # Get sample interval in milliseconds for mute calculation
        sample_interval_ms = task.sample_interval / 1000.0  # microseconds to ms

        # Create segment spec
        n_traces = task.end_trace - task.start_trace + 1

        # Build specification for this segment
        spec = segyio.spec()
        spec.samples = list(range(task.n_samples))
        spec.format = task.data_format
        spec.tracecount = n_traces

        # Build byte position to TraceField lookup table (once per worker)
        # segyio.TraceField is a class with attributes, not an iterable enum
        byte_pos_to_field = {}
        for attr_name in dir(segyio.TraceField):
            if not attr_name.startswith('_'):
                try:
                    byte_pos = getattr(segyio.TraceField, attr_name)
                    if isinstance(byte_pos, int):
                        byte_pos_to_field[byte_pos] = attr_name
                except Exception:
                    pass

        # Create segment file
        with segyio.create(task.output_segment_path, spec) as dst:
            # For first segment only: set up text/binary headers
            if task.is_first_segment:
                if task.original_segy_path and Path(task.original_segy_path).exists():
                    # Copy headers from original SEG-Y if available
                    with segyio.open(task.original_segy_path, 'r', ignore_geometry=True) as src:
                        dst.text[0] = src.text[0]
                        dst.bin = src.bin
                else:
                    # Generate default EBCDIC text header
                    text_header = _create_default_text_header(task.n_samples, task.sample_interval)
                    dst.text[0] = text_header
                    # Set binary header fields
                    dst.bin[segyio.BinField.JobID] = 1
                    dst.bin[segyio.BinField.LineNumber] = 1
                    dst.bin[segyio.BinField.ReelNumber] = 1
                    dst.bin[segyio.BinField.Traces] = 1
                    dst.bin[segyio.BinField.AuxTraces] = 0
                    dst.bin[segyio.BinField.Format] = task.data_format
                    dst.bin[segyio.BinField.SortingCode] = 1
                    dst.bin[segyio.BinField.MeasurementSystem] = 1  # Meters

                # Update sample count and interval in binary header
                dst.bin[segyio.BinField.Samples] = task.n_samples
                dst.bin[segyio.BinField.Interval] = task.sample_interval
            else:
                # Non-first segments: set minimal binary header
                dst.bin[segyio.BinField.Samples] = task.n_samples
                dst.bin[segyio.BinField.Interval] = task.sample_interval
                dst.bin[segyio.BinField.Format] = task.data_format

            # Write traces in chunks for memory efficiency
            # Use adaptive chunk size based on available memory and trace depth
            chunk_size = get_adaptive_chunk_size(task.n_samples)
            local_idx = 0

            for chunk_start in range(task.start_trace, task.end_trace + 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, task.end_trace + 1)
                chunk_n = chunk_end - chunk_start

                # Helper function to apply mute to a chunk
                def apply_mute_to_chunk(chunk, chunk_start_idx):
                    """Apply mute to all traces in chunk."""
                    if not apply_mute:
                        return chunk
                    for i in range(chunk.shape[1]):
                        local_header_idx = (chunk_start_idx - task.start_trace) + i
                        offset = 0.0
                        for offset_field in ['offset', 'OFFSET', 'Offset']:
                            if offset_field in header_arrays:
                                offset = float(header_arrays[offset_field][local_header_idx])
                                break
                        if offset != 0.0:
                            chunk[:, i] = apply_mute_to_trace(
                                chunk[:, i].copy(),
                                offset,
                                sample_interval_ms,
                                task.mute_velocity,
                                task.mute_top,
                                task.mute_bottom,
                                task.mute_taper
                            )
                    return chunk

                # Load chunk of trace data based on export type and mute target
                if task.export_type == 'noise' and input_zarr is not None:
                    # Noise = Input - Processed
                    input_chunk = np.array(input_zarr[:, chunk_start:chunk_end])
                    processed_chunk = np.array(processed_zarr[:, chunk_start:chunk_end])

                    # Apply mute based on target
                    if apply_mute and task.mute_target == 'input':
                        input_chunk = apply_mute_to_chunk(input_chunk, chunk_start)
                    elif apply_mute and task.mute_target == 'processed':
                        processed_chunk = apply_mute_to_chunk(processed_chunk, chunk_start)

                    chunk_data = input_chunk - processed_chunk

                    # Apply mute to output if target is 'output'
                    if apply_mute and task.mute_target == 'output':
                        chunk_data = apply_mute_to_chunk(chunk_data, chunk_start)

                    del input_chunk, processed_chunk
                else:
                    # Standard processed export
                    chunk_data = np.array(processed_zarr[:, chunk_start:chunk_end])

                    # Apply mute to output (only option for processed-only export)
                    if apply_mute:
                        chunk_data = apply_mute_to_chunk(chunk_data, chunk_start)

                # Write each trace with vectorized headers
                # Note: header_arrays is now a SLICE for this segment only,
                # indexed 0 to (end_trace - start_trace), not global indices
                for i in range(chunk_n):
                    # Local index within this segment's header slice
                    local_header_idx = (chunk_start - task.start_trace) + i

                    # Get trace data for this trace
                    trace_data = chunk_data[:, i]

                    # Write trace data
                    dst.trace[local_idx] = trace_data

                    # Write headers using custom mapping if provided, otherwise use auto-detection
                    if task.header_mapping_list:
                        # Custom header mapping: parquet_column -> byte_position
                        # Simple and direct: read value, write to byte position
                        for mapping in task.header_mapping_list:
                            parquet_col = mapping['parquet_column']
                            byte_pos = mapping['segy_byte_pos']

                            # Get value from header arrays (now uses original parquet names)
                            if parquet_col in header_arrays:
                                header_value = int(header_arrays[parquet_col][local_header_idx])

                                # Find segyio TraceField for this byte position
                                field_name = byte_pos_to_field.get(byte_pos)
                                if field_name:
                                    field = getattr(segyio.TraceField, field_name)
                                    dst.header[local_idx][field] = header_value
                                # If byte position not in segyio's standard fields, skip
                                # (segyio doesn't support arbitrary byte positions directly)
                    else:
                        # Auto-detect: header_arrays uses segyio field names
                        # Write headers using standard field name mapping
                        for field_name, arr in header_arrays.items():
                            # Try exact match first
                            if hasattr(segyio.TraceField, field_name):
                                field = getattr(segyio.TraceField, field_name)
                                dst.header[local_idx][field] = int(arr[local_header_idx])
                            # Try common variations (lowercase, uppercase)
                            elif hasattr(segyio.TraceField, field_name.lower()):
                                field = getattr(segyio.TraceField, field_name.lower())
                                dst.header[local_idx][field] = int(arr[local_header_idx])
                            elif hasattr(segyio.TraceField, field_name.upper()):
                                field = getattr(segyio.TraceField, field_name.upper())
                                dst.header[local_idx][field] = int(arr[local_header_idx])

                    local_idx += 1
                    traces_done += 1

                # Report progress periodically (non-blocking to avoid deadlock)
                if progress_queue is not None:
                    try:
                        progress_queue.put_nowait((task.segment_id, traces_done))
                    except Exception:
                        pass  # Queue full, skip this update

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
