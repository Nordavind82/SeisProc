"""
SEG-Y reader with custom header mapping support.
Reads SEG-Y files and extracts traces and headers according to mapping.

Performance optimizations:
- Memory mapping (mmap) for faster I/O
- Pre-allocated arrays to avoid fragmentation
- Batch header reading using segyio attributes
- Buffer pooling for chunked operations
- Explicit GC for large datasets (>1M traces)
"""
import gc
import numpy as np
import segyio
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import time
import threading
from models.seismic_data import SeismicData
from utils.segy_import.header_mapping import HeaderMapping

# Threshold for triggering explicit GC (in traces)
GC_THRESHOLD = 100000


class LoadingProgress:
    """Track loading progress with ETA calculation."""

    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self._window_size = 10
        self._recent_times: List[Tuple[int, float]] = []

    def update(self, current: int):
        """Update progress and calculate ETA."""
        now = time.time()
        self.current = current
        self._recent_times.append((current, now))
        if len(self._recent_times) > self._window_size:
            self._recent_times.pop(0)

    @property
    def percent(self) -> float:
        return (self.current / self.total) * 100 if self.total > 0 else 0

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining."""
        if len(self._recent_times) < 2:
            return float('inf')
        first = self._recent_times[0]
        last = self._recent_times[-1]
        traces_processed = last[0] - first[0]
        time_elapsed = last[1] - first[1]
        if traces_processed == 0 or time_elapsed == 0:
            return float('inf')
        rate = traces_processed / time_elapsed
        remaining = self.total - self.current
        return remaining / rate

    @property
    def throughput(self) -> float:
        """Current throughput in traces/second."""
        if len(self._recent_times) < 2:
            return 0
        first = self._recent_times[0]
        last = self._recent_times[-1]
        time_elapsed = last[1] - first[1]
        traces = last[0] - first[0]
        return traces / time_elapsed if time_elapsed > 0 else 0


class CancellationToken:
    """
    Thread-safe cancellation token for long-running operations.

    Usage:
        token = CancellationToken()
        # In worker thread:
        for item in items:
            if token.is_cancelled:
                raise OperationCancelledError("Operation cancelled by user")
            process(item)
        # In UI thread:
        token.cancel()
    """

    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        """Request cancellation of the operation."""
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled.is_set()

    def reset(self):
        """Reset the cancellation token for reuse."""
        self._cancelled.clear()


class OperationCancelledError(Exception):
    """Raised when a long-running operation is cancelled."""
    pass


class SEGYFileHandle:
    """
    Context manager for SEG-Y file access that keeps the file open.

    Use this for batch operations to avoid repeated open/close overhead.

    Usage:
        reader = SEGYReader(filename, mapping)
        with reader.open() as handle:
            info = handle.read_file_info()
            traces = handle.read_traces_range(0, 100)
            headers = handle.read_headers_range(0, 100)
    """

    def __init__(self, filename: str, header_mapping: 'HeaderMapping'):
        """
        Initialize the file handle.

        Args:
            filename: Path to SEG-Y file
            header_mapping: Header mapping configuration
        """
        self.filename = Path(filename)
        self.header_mapping = header_mapping
        self._file = None

    def __enter__(self):
        """Open the SEG-Y file with memory mapping for faster I/O."""
        self._file = segyio.open(str(self.filename), 'r', ignore_geometry=True)
        self._file.mmap()  # Enable memory mapping for performance
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the SEG-Y file."""
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    @property
    def file(self):
        """Get the underlying segyio file object."""
        if self._file is None:
            raise RuntimeError("File not open. Use 'with reader.open() as handle:'")
        return self._file

    def read_file_info(self) -> Dict[str, any]:
        """Read basic file information from open file."""
        f = self.file
        info = {
            'filename': self.filename.name,
            'n_traces': f.tracecount,
            'n_samples': len(f.samples),
            'sample_interval': segyio.tools.dt(f) / 1000.0,
            'trace_length_ms': (len(f.samples) - 1) * segyio.tools.dt(f) / 1000.0,
            'text_header': f.text[0].decode('ascii', errors='ignore'),
            'binary_header': {
                'job_id': f.bin[segyio.BinField.JobID],
                'line_number': f.bin[segyio.BinField.LineNumber],
                'reel_number': f.bin[segyio.BinField.ReelNumber],
            },
            'format': self._get_format_name(f.bin[segyio.BinField.Format]),
        }
        return info

    def _get_format_name(self, format_code: int) -> str:
        """Convert format code to human-readable name."""
        format_names = {
            1: 'IBM Float (4-byte)',
            2: 'Integer (4-byte)',
            3: 'Integer (2-byte)',
            4: 'Fixed Point with Gain (4-byte)',
            5: 'IEEE Float (4-byte)',
            8: 'Integer (1-byte)',
        }
        return format_names.get(format_code, f'Unknown ({format_code})')

    def read_traces_range(self, start: int, end: int) -> np.ndarray:
        """
        Read a range of traces from the open file.

        Args:
            start: Starting trace index (inclusive)
            end: Ending trace index (exclusive)

        Returns:
            Trace data as (n_samples, n_traces) array
        """
        f = self.file
        n_samples = len(f.samples)
        n_traces = end - start

        # Use pre-allocated array with np.empty (faster than zeros)
        traces = np.empty((n_samples, n_traces), dtype=np.float32)
        for i, idx in enumerate(range(start, end)):
            traces[:, i] = f.trace[idx]

        return traces

    def read_headers_range(self, start: int, end: int) -> List[Dict]:
        """
        Read headers for a range of traces using batch attribute access.

        Args:
            start: Starting trace index (inclusive)
            end: Ending trace index (exclusive)

        Returns:
            List of header dictionaries
        """
        return self._read_headers_batch(self.file, start, end)

    def _read_headers_batch(self, f, start: int, end: int) -> List[Dict]:
        """
        Read headers in batch using raw byte extraction.

        Note: We read raw bytes instead of using segyio.attributes() because
        segyio interprets byte positions as TraceField enum values, which
        doesn't work for non-standard header positions (e.g., byte 201 is
        interpreted as ShotPointScalar enum, not raw byte 201).
        """
        import struct

        mappings = self.header_mapping.get_all_mappings()
        n_traces = end - start

        # Pre-create header dictionaries
        headers = [{} for _ in range(n_traces)]

        # Format code sizes and struct format strings (big-endian as per SEG-Y)
        FORMAT_INFO = {
            'i': (4, '>i'),  # 4-byte signed int, big-endian
            'h': (2, '>h'),  # 2-byte signed int, big-endian
            'I': (4, '>I'),  # 4-byte unsigned int, big-endian
            'H': (2, '>H'),  # 2-byte unsigned int, big-endian
        }

        # Read each trace header and extract fields from raw bytes
        for trace_offset, local_idx in enumerate(range(start, end)):
            header_obj = f.header[local_idx]
            raw_header = bytes(header_obj.buf)

            for name, (byte_loc, fmt_code) in mappings.items():
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

                except (KeyError, IndexError, struct.error):
                    headers[trace_offset][name] = 0

        return headers

    @property
    def tracecount(self) -> int:
        """Get total number of traces."""
        return self.file.tracecount

    @property
    def n_samples(self) -> int:
        """Get number of samples per trace."""
        return len(self.file.samples)

    @property
    def sample_interval(self) -> float:
        """Get sample interval in milliseconds."""
        return segyio.tools.dt(self.file) / 1000.0


class SEGYReader:
    """
    SEG-Y file reader with custom header mapping.

    Reads SEG-Y files and extracts:
    - Trace data
    - Configured headers
    - Ensemble boundaries
    - File metadata
    """

    def __init__(self, filename: str, header_mapping: HeaderMapping):
        """
        Initialize SEG-Y reader.

        Args:
            filename: Path to SEG-Y file
            header_mapping: Header mapping configuration
        """
        self.filename = Path(filename)
        self.header_mapping = header_mapping

        if not self.filename.exists():
            raise FileNotFoundError(f"SEG-Y file not found: {filename}")

    def open(self) -> SEGYFileHandle:
        """
        Open the file for batch operations.

        Returns a context manager that keeps the file open for efficient
        multiple read operations.

        Usage:
            with reader.open() as handle:
                info = handle.read_file_info()
                for start in range(0, n_traces, chunk_size):
                    traces = handle.read_traces_range(start, start + chunk_size)
                    # process traces

        Returns:
            SEGYFileHandle context manager
        """
        return SEGYFileHandle(str(self.filename), self.header_mapping)

    def read_file_info(self) -> Dict[str, any]:
        """
        Read basic file information without loading all data.

        Returns:
            Dictionary with file metadata
        """
        with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            info = {
                'filename': self.filename.name,
                'n_traces': f.tracecount,
                'n_samples': len(f.samples),
                'sample_interval': segyio.tools.dt(f) / 1000.0,  # Convert to ms
                'trace_length_ms': (len(f.samples) - 1) * segyio.tools.dt(f) / 1000.0,
                'text_header': f.text[0].decode('ascii', errors='ignore'),
                'binary_header': {
                    'job_id': f.bin[segyio.BinField.JobID],
                    'line_number': f.bin[segyio.BinField.LineNumber],
                    'reel_number': f.bin[segyio.BinField.ReelNumber],
                },
                'format': self._get_format_name(f.bin[segyio.BinField.Format]),
            }
        return info

    def _get_format_name(self, format_code: int) -> str:
        """Get human-readable format name."""
        formats = {
            1: 'IBM float',
            2: '4-byte integer',
            3: '2-byte integer',
            5: 'IEEE float',
            8: '1-byte integer'
        }
        return formats.get(format_code, f'Unknown ({format_code})')

    def read_all_traces(
        self,
        max_traces: Optional[int] = None,
        cancellation_token: Optional[CancellationToken] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Read all traces and headers from SEG-Y file.

        Performance optimizations:
        - Memory mapping (mmap) for faster I/O
        - Pre-allocated arrays to avoid memory fragmentation
        - Batch header reading for reduced overhead

        Args:
            max_traces: Maximum number of traces to read (None = all)
            cancellation_token: Optional token to cancel the operation
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Tuple of:
                - traces: 2D array (n_samples, n_traces)
                - headers: List of header dictionaries

        Raises:
            OperationCancelledError: If cancellation is requested
        """
        with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            # Enable memory mapping for faster I/O
            f.mmap()

            n_traces = min(f.tracecount, max_traces) if max_traces else f.tracecount
            n_samples = len(f.samples)

            # Pre-allocate trace array with np.empty (faster than zeros)
            traces = np.empty((n_samples, n_traces), dtype=np.float32)

            # Initialize progress tracking
            progress = LoadingProgress(n_traces)
            batch_size = 5000  # Process headers in batches

            # Read traces with progress tracking
            for i in range(n_traces):
                # Check for cancellation
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    raise OperationCancelledError(
                        f"SEGY read cancelled at trace {i}/{n_traces}"
                    )

                # Read trace data directly into pre-allocated array
                traces[:, i] = f.trace[i]

                # Progress feedback every 1000 traces
                if (i + 1) % 1000 == 0:
                    progress.update(i + 1)
                    eta = progress.eta_seconds
                    eta_str = f"{eta:.1f}s" if eta < float('inf') else "calculating..."
                    print(f"  Read {i + 1}/{n_traces} traces... "
                          f"({progress.throughput:.0f} traces/s, ETA: {eta_str})")
                    if progress_callback is not None:
                        progress_callback(i + 1, n_traces)

            # Read headers in batch (much faster than per-trace)
            print(f"  Reading headers...")
            headers = self._read_headers_batch_with_mapping(f, 0, n_traces)

        print(f"  Completed reading {n_traces} traces")
        if progress_callback is not None:
            progress_callback(n_traces, n_traces)
        return traces, headers

    def _read_headers_batch_with_mapping(self, f, start: int, end: int) -> List[Dict]:
        """
        Read headers using raw byte extraction with header mapping applied.

        Note: We read raw bytes instead of using segyio.attributes() because
        segyio interprets byte positions as TraceField enum values, which
        doesn't work for non-standard header positions.
        """
        import struct

        mappings = self.header_mapping.get_all_mappings()
        n_traces = end - start

        # Pre-create header dictionaries
        headers = [{} for _ in range(n_traces)]

        # Format code sizes and struct format strings (big-endian as per SEG-Y)
        FORMAT_INFO = {
            'i': (4, '>i'),  # 4-byte signed int, big-endian
            'h': (2, '>h'),  # 2-byte signed int, big-endian
            'I': (4, '>I'),  # 4-byte unsigned int, big-endian
            'H': (2, '>H'),  # 2-byte unsigned int, big-endian
        }

        # Read each trace header and extract fields from raw bytes
        for trace_offset, local_idx in enumerate(range(start, end)):
            header_obj = f.header[local_idx]
            raw_header = bytes(header_obj.buf)

            for name, (byte_loc, fmt_code) in mappings.items():
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

                except (KeyError, IndexError, struct.error):
                    headers[trace_offset][name] = 0

        # Apply computed headers if configured
        if self.header_mapping.has_computed_headers():
            processor = self.header_mapping.get_computed_processor()
            if processor:
                for i, header in enumerate(headers):
                    computed = processor.compute(header, trace_idx=start + i)
                    header.update(computed)

        return headers

    def detect_ensemble_boundaries(self, headers: List[Dict]) -> List[Tuple[int, int]]:
        """
        Detect ensemble boundaries based on configured ensemble keys.

        Args:
            headers: List of header dictionaries

        Returns:
            List of (start_index, end_index) tuples for each ensemble
        """
        if not self.header_mapping.ensemble_keys:
            raise ValueError("No ensemble keys configured in header mapping")

        ensembles = []
        current_ensemble_start = 0
        current_ensemble_values = None

        for i, header in enumerate(headers):
            # Get current ensemble key values
            ensemble_values = tuple(
                header.get(key) for key in self.header_mapping.ensemble_keys
            )

            # Check if ensemble changed
            if current_ensemble_values is None:
                current_ensemble_values = ensemble_values
            elif ensemble_values != current_ensemble_values:
                # Ensemble boundary found
                ensembles.append((current_ensemble_start, i - 1))
                current_ensemble_start = i
                current_ensemble_values = ensemble_values

        # Add last ensemble
        ensembles.append((current_ensemble_start, len(headers) - 1))

        return ensembles

    def read_to_seismic_data(self, max_traces: Optional[int] = None) -> Tuple[SeismicData, List[Dict], List[Tuple[int, int]]]:
        """
        Read SEG-Y file and return SeismicData object with headers and ensembles.

        Args:
            max_traces: Maximum number of traces to read (None = all)

        Returns:
            Tuple of:
                - SeismicData object
                - List of header dictionaries
                - List of ensemble boundaries
        """
        # Get file info
        info = self.read_file_info()
        print(f"Reading SEG-Y file: {info['filename']}")
        print(f"  Traces: {info['n_traces']}, Samples: {info['n_samples']}")
        print(f"  Sample interval: {info['sample_interval']:.2f}ms")

        # Read traces and headers
        traces, headers = self.read_all_traces(max_traces)

        # Detect ensembles if configured
        ensembles = []
        if self.header_mapping.ensemble_keys:
            print(f"  Detecting ensembles using keys: {self.header_mapping.ensemble_keys}")
            ensembles = self.detect_ensemble_boundaries(headers)
            print(f"  Found {len(ensembles)} ensembles")

        # Create SeismicData object
        metadata = {
            'source_file': str(self.filename),
            'file_info': info,
            'header_mapping': self.header_mapping.to_dict(),
        }

        seismic_data = SeismicData(
            traces=traces,
            sample_rate=info['sample_interval'],
            metadata=metadata
        )

        return seismic_data, headers, ensembles

    def read_traces_in_chunks(
        self,
        chunk_size: int = 5000,
        cancellation_token: Optional[CancellationToken] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stream traces and headers in chunks without loading full file into memory.

        Performance optimizations:
        - Memory mapping (mmap) for faster I/O
        - Buffer pooling to avoid repeated allocations
        - Batch header reading for reduced overhead
        - Pre-allocated arrays

        Args:
            chunk_size: Number of traces per chunk (default: 5000)
            cancellation_token: Optional token to cancel the operation
            progress_callback: Optional callback(current, total) for progress updates

        Yields:
            Tuple containing:
                - traces_array: np.ndarray of shape (n_samples, chunk_traces) with trace data
                - headers_list: List of header dictionaries for traces in this chunk
                - start_index: Starting trace index for this chunk (0-based)
                - end_index: Ending trace index for this chunk (exclusive)

        Raises:
            OperationCancelledError: If cancellation is requested

        Memory Usage:
            O(chunk_size) - Only one chunk in memory at a time
        """
        with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            # Enable memory mapping
            f.mmap()

            n_traces = f.tracecount
            n_samples = len(f.samples)

            # Pre-allocate reusable buffer (buffer pooling)
            buffer = np.empty((n_samples, chunk_size), dtype=np.float32)

            # Progress tracking
            progress = LoadingProgress(n_traces)

            # Process file in chunks
            for start_idx in range(0, n_traces, chunk_size):
                # Check for cancellation at chunk boundaries
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    raise OperationCancelledError(
                        f"SEGY read cancelled at chunk starting at trace {start_idx}"
                    )

                end_idx = min(start_idx + chunk_size, n_traces)
                current_chunk_size = end_idx - start_idx

                # Read traces into buffer
                for i in range(current_chunk_size):
                    buffer[:, i] = f.trace[start_idx + i]

                # Slice buffer to actual size (view, not copy if same size)
                traces_chunk = buffer[:, :current_chunk_size].copy()

                # Read headers in batch (much faster than per-trace)
                headers_chunk = self._read_headers_batch_with_mapping(f, start_idx, end_idx)

                # Progress feedback
                progress.update(end_idx)
                if end_idx % 5000 == 0 or end_idx == n_traces:
                    eta = progress.eta_seconds
                    eta_str = f"{eta:.1f}s" if eta < float('inf') else "..."
                    print(f"  Streamed {end_idx}/{n_traces} traces "
                          f"({progress.throughput:.0f}/s, ETA: {eta_str})")
                    if progress_callback is not None:
                        progress_callback(end_idx, n_traces)

                # Yield chunk
                yield traces_chunk, headers_chunk, start_idx, end_idx

                # Explicit memory management for large datasets
                del traces_chunk
                del headers_chunk

                # Periodic GC to prevent memory fragmentation on large datasets
                if end_idx % GC_THRESHOLD == 0:
                    gc.collect()

    def read_sample_headers(self, n_traces: int = 10) -> List[Dict]:
        """
        Read a sample of headers for inspection.

        Args:
            n_traces: Number of traces to sample

        Returns:
            List of header dictionaries
        """
        with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            n_sample = min(n_traces, f.tracecount)
            headers = []

            for i in range(n_sample):
                header_dict = f.header[i]
                header_bytes = bytes(header_dict.buf)
                trace_headers = self.header_mapping.read_headers(header_bytes, trace_idx=i)
                headers.append(trace_headers)

        return headers

    def get_computed_header_errors(self) -> Optional[str]:
        """
        Get error summary from computed header processing.

        Returns:
            Error summary string or None if no computed headers or no errors
        """
        if not self.header_mapping.has_computed_headers():
            return None

        processor = self.header_mapping.get_computed_processor()
        if processor:
            return processor.get_error_summary()

        return None

    def reset_computed_header_stats(self):
        """Reset computed header error statistics."""
        if self.header_mapping.has_computed_headers():
            processor = self.header_mapping.get_computed_processor()
            if processor:
                processor.reset_stats()

    def __repr__(self) -> str:
        return f"SEGYReader('{self.filename.name}', {self.header_mapping})"
