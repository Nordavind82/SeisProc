"""
SEG-Y reader with custom header mapping support.
Reads SEG-Y files and extracts traces and headers according to mapping.
"""
import numpy as np
import segyio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from models.seismic_data import SeismicData
from utils.segy_import.header_mapping import HeaderMapping


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

    def read_all_traces(self, max_traces: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Read all traces and headers from SEG-Y file.

        Args:
            max_traces: Maximum number of traces to read (None = all)

        Returns:
            Tuple of:
                - traces: 2D array (n_samples, n_traces)
                - headers: List of header dictionaries
        """
        with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            n_traces = min(f.tracecount, max_traces) if max_traces else f.tracecount
            n_samples = len(f.samples)

            # Allocate arrays
            traces = np.zeros((n_samples, n_traces), dtype=np.float32)
            headers = []

            # Read traces and headers
            for i in range(n_traces):
                # Read trace data
                traces[:, i] = f.trace[i]

                # Read headers according to mapping
                # segyio returns headers as a dict-like object, convert to bytes
                header_dict = f.header[i]
                # Get raw header bytes (240 bytes)
                header_bytes = bytes(header_dict.buf)
                trace_headers = self.header_mapping.read_headers(header_bytes, trace_idx=i)
                headers.append(trace_headers)

                # Progress feedback
                if (i + 1) % 1000 == 0:
                    print(f"  Read {i + 1}/{n_traces} traces...")

        print(f"  Completed reading {n_traces} traces")
        return traces, headers

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

    def read_traces_in_chunks(self, chunk_size: int = 5000):
        """
        Stream traces and headers in chunks without loading full file into memory.

        This method yields chunks of trace data, enabling memory-efficient processing
        of large SEGY files that don't fit in RAM. Each chunk is yielded and then
        garbage collected before the next chunk is loaded.

        Args:
            chunk_size: Number of traces per chunk (default: 5000)

        Yields:
            Tuple containing:
                - traces_array: np.ndarray of shape (n_samples, chunk_traces) with trace data
                - headers_list: List of header dictionaries for traces in this chunk
                - start_index: Starting trace index for this chunk (0-based)
                - end_index: Ending trace index for this chunk (exclusive)

        Example:
            >>> reader = SEGYReader('large_file.sgy', header_mapping)
            >>> for traces, headers, start, end in reader.read_traces_in_chunks(chunk_size=1000):
            ...     print(f"Processing traces {start} to {end-1}")
            ...     # Process chunk here
            ...     # Memory is freed after this iteration

        Memory Usage:
            O(chunk_size) - Only one chunk in memory at a time
        """
        with segyio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = len(f.samples)

            # Process file in chunks
            for start_idx in range(0, n_traces, chunk_size):
                end_idx = min(start_idx + chunk_size, n_traces)
                current_chunk_size = end_idx - start_idx

                # Allocate memory for this chunk only
                traces_chunk = np.zeros((n_samples, current_chunk_size), dtype=np.float32)
                headers_chunk = []

                # Read traces and headers for this chunk
                for i in range(current_chunk_size):
                    trace_idx = start_idx + i

                    # Read trace data
                    traces_chunk[:, i] = f.trace[trace_idx]

                    # Read headers
                    header_dict = f.header[trace_idx]
                    header_bytes = bytes(header_dict.buf)
                    trace_headers = self.header_mapping.read_headers(header_bytes, trace_idx=trace_idx)
                    headers_chunk.append(trace_headers)

                # Progress feedback
                if end_idx % 1000 == 0 or end_idx == n_traces:
                    print(f"  Streamed {end_idx}/{n_traces} traces...")

                # Yield chunk - after this, Python can garbage collect it
                yield traces_chunk, headers_chunk, start_idx, end_idx

                # Explicitly delete references to help garbage collection
                # (Python will collect anyway, but this makes intent clear)
                del traces_chunk
                del headers_chunk

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
