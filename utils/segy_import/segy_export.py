"""
SEG-Y export module - writes processed seismic data to SEG-Y format.

Preserves all original headers while replacing trace data with processed data.
Supports both in-memory and chunked export for large files.
Includes async/parallel export for improved throughput.
"""
import segyio
import numpy as np
import pandas as pd
import zarr
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
import queue
from models.seismic_data import SeismicData

logger = logging.getLogger(__name__)


class AsyncSEGYExporter:
    """
    Async/parallel SEG-Y exporter with double-buffered I/O.

    Uses ThreadPoolExecutor to overlap reading and writing operations
    for improved throughput on large files.
    """

    def __init__(self, output_path: str, num_workers: int = 2):
        """
        Initialize async SEG-Y exporter.

        Args:
            output_path: Path to output SEG-Y file
            num_workers: Number of worker threads (default 2 for double-buffering)
        """
        self.output_path = Path(output_path)
        self.num_workers = num_workers
        self._write_lock = Lock()
        self._executor: Optional[ThreadPoolExecutor] = None

    def export_from_zarr_async(
        self,
        original_segy_path: str,
        processed_zarr_path: str,
        chunk_size: int = 5000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        headers_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Export processed Zarr data to SEG-Y with async double-buffered I/O.

        Reads next chunk while writing current chunk for ~2x throughput
        on I/O bound operations.

        Args:
            original_segy_path: Path to original SEG-Y file (for headers)
            processed_zarr_path: Path to processed Zarr array
            chunk_size: Number of traces per chunk (default 5000)
            progress_callback: Optional callback(current, total, time_remaining)
            headers_df: Optional sorted headers DataFrame

        Raises:
            ValueError: If data dimensions don't match
            FileNotFoundError: If files not found
        """
        # Validate inputs
        original_path = Path(original_segy_path)
        zarr_path = Path(processed_zarr_path)

        if not original_path.exists():
            raise FileNotFoundError(f"Original SEG-Y file not found: {original_segy_path}")

        if not zarr_path.exists():
            raise FileNotFoundError(f"Processed Zarr not found: {processed_zarr_path}")

        # Open Zarr array
        processed_zarr = zarr.open(str(zarr_path), mode='r')
        n_samples_zarr, n_traces_zarr = processed_zarr.shape

        # Open original SEG-Y for reading
        with segyio.open(str(original_path), 'r', ignore_geometry=True) as src:
            n_traces_src = src.tracecount
            n_samples_src = len(src.samples)

            # Validate dimensions
            if n_samples_src != n_samples_zarr:
                raise ValueError(
                    f"Sample count mismatch: original={n_samples_src}, "
                    f"processed={n_samples_zarr}"
                )

            if n_traces_src != n_traces_zarr:
                raise ValueError(
                    f"Trace count mismatch: original={n_traces_src}, "
                    f"processed={n_traces_zarr}"
                )

            # Validate headers_df if provided
            if headers_df is not None and len(headers_df) != n_traces_zarr:
                raise ValueError(
                    f"Header count mismatch: headers_df={len(headers_df)}, "
                    f"traces={n_traces_zarr}"
                )

            # Create output file specification
            spec = segyio.tools.metadata(src)

            # Create output file
            with segyio.create(str(self.output_path), spec) as dst:
                # Copy binary and text headers
                dst.bin = src.bin
                dst.text[0] = src.text[0]

                # Use ThreadPoolExecutor for async I/O
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    self._executor = executor

                    start_time = time.time()
                    traces_exported = 0

                    # Pre-fetch first chunk
                    chunk_ranges = list(self._generate_chunk_ranges(n_traces_zarr, chunk_size))

                    # Double-buffered async processing
                    pending_read: Optional[Future] = None
                    current_chunk: Optional[Tuple[int, int, np.ndarray]] = None

                    for i, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                        # Start reading next chunk asynchronously
                        next_read: Optional[Future] = None
                        if i + 1 < len(chunk_ranges):
                            next_start, next_end = chunk_ranges[i + 1]
                            next_read = executor.submit(
                                self._read_chunk, processed_zarr, next_start, next_end
                            )

                        # Get current chunk data
                        if pending_read is not None:
                            chunk_data = pending_read.result()
                        else:
                            # First chunk - read synchronously
                            chunk_data = self._read_chunk(processed_zarr, chunk_start, chunk_end)

                        # Write current chunk (thread-safe)
                        self._write_chunk(
                            dst, src, chunk_start, chunk_end,
                            chunk_data, headers_df
                        )

                        # Update for next iteration
                        pending_read = next_read

                        # Update progress
                        traces_exported = chunk_end
                        if progress_callback is not None:
                            elapsed = time.time() - start_time
                            traces_per_sec = traces_exported / elapsed if elapsed > 0 else 0
                            remaining_traces = n_traces_zarr - traces_exported
                            time_remaining = remaining_traces / traces_per_sec if traces_per_sec > 0 else 0
                            progress_callback(traces_exported, n_traces_zarr, time_remaining)

                    self._executor = None

    def _generate_chunk_ranges(self, n_traces: int, chunk_size: int):
        """Generate (start, end) tuples for each chunk."""
        for start in range(0, n_traces, chunk_size):
            end = min(start + chunk_size, n_traces)
            yield (start, end)

    def _read_chunk(self, zarr_array, start: int, end: int) -> np.ndarray:
        """Read a chunk from Zarr array."""
        return np.array(zarr_array[:, start:end])

    def _write_chunk(
        self,
        dst,
        src,
        chunk_start: int,
        chunk_end: int,
        chunk_data: np.ndarray,
        headers_df: Optional[pd.DataFrame]
    ) -> None:
        """Write a chunk to SEG-Y file (thread-safe)."""
        with self._write_lock:
            for i in range(chunk_start, chunk_end):
                local_idx = i - chunk_start

                # Copy trace header
                if headers_df is not None:
                    trace_headers = headers_df.iloc[i].to_dict()
                    for key, value in trace_headers.items():
                        if hasattr(segyio.TraceField, key):
                            dst.header[i][getattr(segyio.TraceField, key)] = int(value)
                else:
                    dst.header[i] = src.header[i]

                # Write trace data
                dst.trace[i] = chunk_data[:, local_idx]


class SEGYExporter:
    """
    Exports processed seismic data to SEG-Y format.

    Preserves all original trace headers and binary header while
    replacing trace data with processed data.
    """

    def __init__(self, output_path: str):
        """
        Initialize SEG-Y exporter.

        Args:
            output_path: Path to output SEG-Y file
        """
        self.output_path = Path(output_path)

    def export(self,
               original_segy_path: str,
               processed_data: SeismicData,
               headers_df: Optional[pd.DataFrame] = None) -> None:
        """
        Export processed data to SEG-Y file.

        Args:
            original_segy_path: Path to original SEG-Y file (for headers)
            processed_data: Processed seismic data to export
            headers_df: Optional trace headers DataFrame (if available)

        Raises:
            ValueError: If data dimensions don't match
            FileNotFoundError: If original SEG-Y file not found
        """
        # Validate inputs
        if not Path(original_segy_path).exists():
            raise FileNotFoundError(f"Original SEG-Y file not found: {original_segy_path}")

        # Open original file for reading
        with segyio.open(original_segy_path, 'r', ignore_geometry=True) as src:
            # Get dimensions
            n_traces_src = src.tracecount
            n_samples_src = len(src.samples)

            # Validate processed data dimensions
            n_samples_proc, n_traces_proc = processed_data.traces.shape

            if n_samples_src != n_samples_proc:
                raise ValueError(
                    f"Sample count mismatch: original={n_samples_src}, "
                    f"processed={n_samples_proc}"
                )

            if n_traces_src != n_traces_proc:
                raise ValueError(
                    f"Trace count mismatch: original={n_traces_src}, "
                    f"processed={n_traces_proc}"
                )

            # Create output file specification
            spec = segyio.tools.metadata(src)

            # Validate headers_df if provided
            if headers_df is not None and len(headers_df) != n_traces_proc:
                raise ValueError(
                    f"Header count mismatch: headers_df={len(headers_df)}, "
                    f"traces={n_traces_proc}"
                )

            # Create output file
            with segyio.create(str(self.output_path), spec) as dst:
                # Copy binary header
                dst.bin = src.bin

                # Copy text header
                dst.text[0] = src.text[0]

                # Write processed traces with headers
                for i in range(n_traces_proc):
                    # Copy trace header (from sorted headers_df or original file)
                    if headers_df is not None:
                        # Use sorted headers from DataFrame
                        trace_headers = headers_df.iloc[i].to_dict()
                        for key, value in trace_headers.items():
                            if hasattr(segyio.TraceField, key):
                                dst.header[i][getattr(segyio.TraceField, key)] = int(value)
                    else:
                        # Use original file headers
                        dst.header[i] = src.header[i]

                    # Write processed trace data
                    dst.trace[i] = processed_data.traces[:, i]

    def export_with_custom_headers(self,
                                   processed_data: SeismicData,
                                   headers_df: pd.DataFrame,
                                   binary_header: dict,
                                   text_header: str) -> None:
        """
        Export processed data with custom headers.

        Args:
            processed_data: Processed seismic data to export
            headers_df: Trace headers as DataFrame
            binary_header: Binary header as dictionary
            text_header: Text header as string

        Raises:
            ValueError: If data dimensions don't match
        """
        n_samples, n_traces = processed_data.traces.shape

        if len(headers_df) != n_traces:
            raise ValueError(
                f"Header count mismatch: headers={len(headers_df)}, "
                f"traces={n_traces}"
            )

        # Create SEG-Y spec
        spec = segyio.spec()
        spec.format = binary_header.get('format', 1)  # 1 = 4-byte IBM float
        spec.samples = range(n_samples)
        spec.tracecount = n_traces

        # Create output file
        with segyio.create(str(self.output_path), spec) as dst:
            # Write text header
            dst.text[0] = text_header.encode('ascii', errors='replace')

            # Write binary header
            for key, value in binary_header.items():
                if hasattr(segyio.BinField, key):
                    setattr(dst.bin, getattr(segyio.BinField, key), value)

            # Write traces with headers
            for i in range(n_traces):
                # Write trace header
                trace_headers = headers_df.iloc[i].to_dict()
                for key, value in trace_headers.items():
                    if hasattr(segyio.TraceField, key):
                        dst.header[i][getattr(segyio.TraceField, key)] = int(value)

                # Write processed trace data
                dst.trace[i] = processed_data.traces[:, i]

    def export_from_zarr_chunked(
        self,
        original_segy_path: str,
        processed_zarr_path: str,
        chunk_size: int = 5000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        headers_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Export processed Zarr data to SEG-Y in memory-efficient chunks.

        Reads processed trace data from Zarr array in chunks while preserving
        all headers from the original SEG-Y file. Memory usage is O(chunk_size).

        Args:
            original_segy_path: Path to original SEG-Y file (for headers)
            processed_zarr_path: Path to processed Zarr array
            chunk_size: Number of traces per chunk (default 5000)
            progress_callback: Optional callback(current, total, time_remaining)
            headers_df: Optional sorted headers DataFrame. If provided, uses these
                       headers instead of original file headers (for sorted export)

        Raises:
            ValueError: If data dimensions don't match
            FileNotFoundError: If original SEG-Y or Zarr file not found
        """
        # Validate inputs
        original_path = Path(original_segy_path)
        zarr_path = Path(processed_zarr_path)

        if not original_path.exists():
            raise FileNotFoundError(f"Original SEG-Y file not found: {original_segy_path}")

        if not zarr_path.exists():
            raise FileNotFoundError(f"Processed Zarr not found: {processed_zarr_path}")

        # Open Zarr array
        processed_zarr = zarr.open(str(zarr_path), mode='r')
        n_samples_zarr, n_traces_zarr = processed_zarr.shape

        # Open original SEG-Y for reading
        with segyio.open(str(original_path), 'r', ignore_geometry=True) as src:
            # Get dimensions
            n_traces_src = src.tracecount
            n_samples_src = len(src.samples)

            # Validate dimensions
            if n_samples_src != n_samples_zarr:
                raise ValueError(
                    f"Sample count mismatch: original={n_samples_src}, "
                    f"processed={n_samples_zarr}"
                )

            if n_traces_src != n_traces_zarr:
                raise ValueError(
                    f"Trace count mismatch: original={n_traces_src}, "
                    f"processed={n_traces_zarr}"
                )

            # Create output file specification
            spec = segyio.tools.metadata(src)

            # Create output file
            with segyio.create(str(self.output_path), spec) as dst:
                # Copy binary header
                dst.bin = src.bin

                # Copy text header
                dst.text[0] = src.text[0]

                # Validate headers_df if provided
                if headers_df is not None and len(headers_df) != n_traces_zarr:
                    raise ValueError(
                        f"Header count mismatch: headers_df={len(headers_df)}, "
                        f"traces={n_traces_zarr}"
                    )

                # Process in chunks
                start_time = time.time()
                traces_exported = 0

                for chunk_start in range(0, n_traces_zarr, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_traces_zarr)

                    # Load chunk from Zarr
                    chunk_data = np.array(processed_zarr[:, chunk_start:chunk_end])

                    # Write traces with headers
                    for i in range(chunk_start, chunk_end):
                        local_idx = i - chunk_start

                        # Copy trace header (from sorted headers_df or original file)
                        if headers_df is not None:
                            # Use sorted headers from DataFrame
                            trace_headers = headers_df.iloc[i].to_dict()
                            for key, value in trace_headers.items():
                                if hasattr(segyio.TraceField, key):
                                    dst.header[i][getattr(segyio.TraceField, key)] = int(value)
                        else:
                            # Use original file headers
                            dst.header[i] = src.header[i]

                        # Write processed trace data
                        dst.trace[i] = chunk_data[:, local_idx]

                    # Update progress
                    traces_exported = chunk_end
                    if progress_callback is not None:
                        elapsed = time.time() - start_time
                        traces_per_sec = traces_exported / elapsed if elapsed > 0 else 0
                        remaining_traces = n_traces_zarr - traces_exported
                        time_remaining = remaining_traces / traces_per_sec if traces_per_sec > 0 else 0

                        progress_callback(traces_exported, n_traces_zarr, time_remaining)


def export_processed_segy(output_path: str,
                         original_segy_path: str,
                         processed_data: SeismicData,
                         headers_df: Optional[pd.DataFrame] = None) -> None:
    """
    Convenience function to export processed SEG-Y data.

    Args:
        output_path: Path to output SEG-Y file
        original_segy_path: Path to original SEG-Y file (for headers)
        processed_data: Processed seismic data
        headers_df: Optional trace headers DataFrame

    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If original file not found
    """
    exporter = SEGYExporter(output_path)
    exporter.export(original_segy_path, processed_data, headers_df)


def export_from_zarr_chunked(
    output_path: str,
    original_segy_path: str,
    processed_zarr_path: str,
    chunk_size: int = 5000,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    headers_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Convenience function to export processed Zarr data to SEG-Y in chunks.

    Memory-efficient export that reads Zarr data in chunks. Memory usage
    is O(chunk_size), not O(total_size).

    Args:
        output_path: Path to output SEG-Y file
        original_segy_path: Path to original SEG-Y file (for headers)
        processed_zarr_path: Path to processed Zarr array
        chunk_size: Number of traces per chunk (default 5000)
        progress_callback: Optional callback(current_trace, total_traces, time_remaining)
        headers_df: Optional sorted headers DataFrame. If provided, uses these
                   headers instead of original file headers (for sorted export)

    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If files not found

    Example:
        >>> def progress(current, total, time_rem):
        ...     print(f"Progress: {current}/{total}, {time_rem:.0f}s remaining")
        >>> export_from_zarr_chunked(
        ...     'output.sgy', 'input.sgy', 'processed.zarr',
        ...     chunk_size=5000, progress_callback=progress
        ... )
    """
    exporter = SEGYExporter(output_path)
    exporter.export_from_zarr_chunked(
        original_segy_path,
        processed_zarr_path,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        headers_df=headers_df
    )


def export_from_zarr_async(
    output_path: str,
    original_segy_path: str,
    processed_zarr_path: str,
    chunk_size: int = 5000,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    headers_df: Optional[pd.DataFrame] = None,
    num_workers: int = 2
) -> None:
    """
    Convenience function for async double-buffered Zarr to SEG-Y export.

    Uses ThreadPoolExecutor to overlap reading and writing operations,
    providing ~2x throughput improvement for I/O bound operations.

    Args:
        output_path: Path to output SEG-Y file
        original_segy_path: Path to original SEG-Y file (for headers)
        processed_zarr_path: Path to processed Zarr array
        chunk_size: Number of traces per chunk (default 5000)
        progress_callback: Optional callback(current_trace, total_traces, time_remaining)
        headers_df: Optional sorted headers DataFrame
        num_workers: Number of worker threads (default 2 for double-buffering)

    Raises:
        ValueError: If data validation fails
        FileNotFoundError: If files not found

    Example:
        >>> def progress(current, total, time_rem):
        ...     print(f"Progress: {current}/{total}, {time_rem:.0f}s remaining")
        >>> export_from_zarr_async(
        ...     'output.sgy', 'input.sgy', 'processed.zarr',
        ...     chunk_size=5000, progress_callback=progress
        ... )
    """
    exporter = AsyncSEGYExporter(output_path, num_workers=num_workers)
    exporter.export_from_zarr_async(
        original_segy_path,
        processed_zarr_path,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        headers_df=headers_df
    )
