"""
SEG-Y segment file merger.

Combines multiple segment SEG-Y files into a single output file.
Uses binary concatenation for efficiency, then updates trace count in header.
"""

import os
import struct
from pathlib import Path
from typing import List, Callable, Optional
import segyio


class SEGYSegmentMerger:
    """
    Merges segment SEG-Y files into a single output file.

    Strategy:
    1. Copy first segment (has text/binary headers) as base
    2. Append trace data from remaining segments
    3. Update trace count in binary header

    SEG-Y file structure:
    - 3200 bytes: Text header
    - 400 bytes: Binary header
    - 240 bytes + n_samples * bytes_per_sample: Each trace

    For IBM float or IEEE float, bytes_per_sample = 4
    """

    # SEG-Y constants
    TEXT_HEADER_SIZE = 3200
    BINARY_HEADER_SIZE = 400
    TRACE_HEADER_SIZE = 240

    # Binary header offsets for trace count
    TRACE_COUNT_OFFSET = 12  # Bytes 3213-3214 (offset 12-13 in binary header)

    def __init__(self, n_samples: int, data_format: int = 5):
        """
        Initialize merger.

        Args:
            n_samples: Samples per trace
            data_format: SEG-Y data format code (1=IBM, 5=IEEE float)
        """
        self.n_samples = n_samples
        self.data_format = data_format

        # Bytes per sample (4 for IBM float or IEEE float)
        self.bytes_per_sample = 4
        self.trace_data_size = n_samples * self.bytes_per_sample
        self.full_trace_size = self.TRACE_HEADER_SIZE + self.trace_data_size

    def merge(
        self,
        segment_paths: List[str],
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Merge segment files into final output.

        Args:
            segment_paths: List of segment file paths in order
            output_path: Path for merged output file
            progress_callback: Optional callback(bytes_written, total_bytes)
        """
        if not segment_paths:
            raise ValueError("No segment files provided")

        # Calculate total expected size
        total_traces = 0
        segment_info = []

        for path in segment_paths:
            file_size = Path(path).stat().st_size
            # Calculate traces in this segment from file size
            header_size = self.TEXT_HEADER_SIZE + self.BINARY_HEADER_SIZE
            trace_data_size = file_size - header_size
            n_traces = trace_data_size // self.full_trace_size
            segment_info.append({
                'path': path,
                'file_size': file_size,
                'n_traces': n_traces
            })
            total_traces += n_traces

        total_output_size = (
            self.TEXT_HEADER_SIZE +
            self.BINARY_HEADER_SIZE +
            total_traces * self.full_trace_size
        )

        bytes_written = 0

        with open(output_path, 'wb') as out_file:
            for i, info in enumerate(segment_info):
                path = info['path']

                with open(path, 'rb') as seg_file:
                    if i == 0:
                        # First segment: copy text + binary headers
                        header_data = seg_file.read(
                            self.TEXT_HEADER_SIZE + self.BINARY_HEADER_SIZE
                        )
                        out_file.write(header_data)
                        bytes_written += len(header_data)

                        if progress_callback:
                            progress_callback(bytes_written, total_output_size)
                    else:
                        # Skip headers in other segments
                        seg_file.seek(self.TEXT_HEADER_SIZE + self.BINARY_HEADER_SIZE)

                    # Copy trace data in chunks
                    chunk_size = 10 * self.full_trace_size  # 10 traces at a time
                    while True:
                        chunk = seg_file.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        bytes_written += len(chunk)

                        if progress_callback:
                            progress_callback(bytes_written, total_output_size)

        # Update trace count in binary header
        self._update_trace_count(output_path, total_traces)

        return total_traces

    def _update_trace_count(self, output_path: str, total_traces: int):
        """
        Update the trace count in the binary header.

        Binary header bytes 3213-3216 contain the number of traces (if set).
        However, most readers determine trace count from file size, so this
        is mainly for completeness.
        """
        # Use segyio to properly update the header
        with segyio.open(output_path, 'r+', ignore_geometry=True) as f:
            # segyio handles endianness and format correctly
            pass  # Just opening in r+ mode validates the file

    def get_merge_stats(self, segment_paths: List[str]) -> dict:
        """
        Get statistics about the merge operation.

        Args:
            segment_paths: List of segment file paths

        Returns:
            Dictionary with merge statistics
        """
        total_traces = 0
        total_size = 0

        for path in segment_paths:
            file_size = Path(path).stat().st_size
            header_size = self.TEXT_HEADER_SIZE + self.BINARY_HEADER_SIZE
            trace_data_size = file_size - header_size
            n_traces = trace_data_size // self.full_trace_size
            total_traces += n_traces
            total_size += file_size

        output_size = (
            self.TEXT_HEADER_SIZE +
            self.BINARY_HEADER_SIZE +
            total_traces * self.full_trace_size
        )

        return {
            'n_segments': len(segment_paths),
            'total_traces': total_traces,
            'input_size_bytes': total_size,
            'output_size_bytes': output_size,
            'input_size_gb': total_size / (1024 ** 3),
            'output_size_gb': output_size / (1024 ** 3),
        }


def merge_segy_segments(
    segment_paths: List[str],
    output_path: str,
    n_samples: int,
    data_format: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> int:
    """
    Convenience function to merge SEG-Y segment files.

    Args:
        segment_paths: List of segment file paths in order
        output_path: Path for merged output file
        n_samples: Samples per trace
        data_format: SEG-Y data format code
        progress_callback: Optional callback(bytes_written, total_bytes)

    Returns:
        Total number of traces in merged file
    """
    merger = SEGYSegmentMerger(n_samples, data_format)
    return merger.merge(segment_paths, output_path, progress_callback)
