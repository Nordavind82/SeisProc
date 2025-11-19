"""
Chunked processor - processes large Zarr datasets in memory-efficient chunks.

Enables processing of datasets that don't fit in RAM by loading and processing
one chunk at a time, with proper handling of chunk boundaries.
"""

import numpy as np
import zarr
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Tuple
import sys
from processors.base_processor import BaseProcessor
from models.seismic_data import SeismicData


class ChunkedProcessor:
    """
    Processes large Zarr seismic datasets in memory-efficient chunks.

    Loads chunks sequentially, processes them, and writes to output Zarr.
    Memory usage is O(chunk_size), not O(total_size).
    """

    def __init__(self):
        """Initialize chunked processor."""
        self._cancel_flag = threading.Event()
        self._processing = False

    def process(
        self,
        input_zarr_path: Path,
        output_zarr_path: Path,
        processor: BaseProcessor,
        chunk_size: int = 5000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        overlap_percent: float = 0.10
    ) -> bool:
        """
        Process Zarr dataset in chunks.

        Args:
            input_zarr_path: Path to input Zarr array
            output_zarr_path: Path to output Zarr array (will be created)
            processor: Processor to apply to each chunk
            chunk_size: Number of traces per chunk
            progress_callback: Optional callback(current_trace, total_traces, time_remaining)
            overlap_percent: Overlap for boundary handling (0.0-0.5)

        Returns:
            True if processing completed, False if cancelled

        Raises:
            ValueError: If paths are invalid or chunk_size is too small
        """
        self._cancel_flag.clear()
        self._processing = True

        try:
            # Validate inputs
            if chunk_size < 100:
                raise ValueError(f"chunk_size must be >= 100, got {chunk_size}")

            if not (0.0 <= overlap_percent <= 0.5):
                raise ValueError(f"overlap_percent must be in [0.0, 0.5], got {overlap_percent}")

            input_zarr_path = Path(input_zarr_path)
            output_zarr_path = Path(output_zarr_path)

            if not input_zarr_path.exists():
                raise ValueError(f"Input Zarr does not exist: {input_zarr_path}")

            # Open input Zarr
            input_zarr = zarr.open_array(str(input_zarr_path), mode='r')
            n_samples, n_traces = input_zarr.shape

            # Create output Zarr with same dimensions
            output_zarr = zarr.open_array(
                str(output_zarr_path),
                mode='w',
                shape=(n_samples, n_traces),
                chunks=(n_samples, min(chunk_size, n_traces)),
                dtype=input_zarr.dtype
            )

            # Calculate overlap size
            overlap_size = int(chunk_size * overlap_percent)

            # Process chunks
            start_time = time.time()
            traces_processed = 0

            for chunk_start in range(0, n_traces, chunk_size):
                # Check for cancellation
                if self._cancel_flag.is_set():
                    self._cleanup_partial_output(output_zarr_path)
                    return False

                # Calculate chunk boundaries
                chunk_end = min(chunk_start + chunk_size, n_traces)
                chunk_n_traces = chunk_end - chunk_start

                # Load chunk with overlap
                load_start, load_end, crop_start, crop_end = self._calculate_chunk_boundaries(
                    chunk_start, chunk_end, n_traces, overlap_size
                )

                # Load chunk data
                chunk_traces = np.array(input_zarr[:, load_start:load_end])

                # Create SeismicData for this chunk
                # Note: We need sample_rate from metadata, but for now use placeholder
                chunk_data = SeismicData(
                    traces=chunk_traces,
                    sample_rate=0.004,  # Will be overridden if metadata available
                    metadata={'chunk_start': load_start, 'chunk_end': load_end}
                )

                # Process chunk
                processed_chunk = processor.process(chunk_data)

                # Clear GPU cache after processing chunk (prevents slowdown over time)
                # Works for any GPU-accelerated processor
                if hasattr(processor, 'device_manager') and processor.device_manager is not None:
                    processor.device_manager.clear_cache()

                # Crop to remove overlap regions
                if overlap_size > 0:
                    cropped_traces = processed_chunk.traces[:, crop_start:crop_end]
                else:
                    cropped_traces = processed_chunk.traces

                # Write to output
                output_zarr[:, chunk_start:chunk_end] = cropped_traces

                # Update progress
                traces_processed = chunk_end
                if progress_callback is not None:
                    elapsed = time.time() - start_time
                    traces_per_sec = traces_processed / elapsed if elapsed > 0 else 0
                    remaining_traces = n_traces - traces_processed
                    time_remaining = remaining_traces / traces_per_sec if traces_per_sec > 0 else 0

                    progress_callback(traces_processed, n_traces, time_remaining)

            self._processing = False
            return True

        except Exception as e:
            self._processing = False
            # Clean up partial output on error
            if output_zarr_path.exists():
                self._cleanup_partial_output(output_zarr_path)
            raise

    def _calculate_chunk_boundaries(
        self,
        chunk_start: int,
        chunk_end: int,
        n_traces: int,
        overlap_size: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate chunk boundaries with overlap.

        Args:
            chunk_start: Start trace of chunk (without overlap)
            chunk_end: End trace of chunk (without overlap)
            n_traces: Total number of traces
            overlap_size: Number of traces to overlap on each side

        Returns:
            Tuple of (load_start, load_end, crop_start, crop_end)
            - load_start/load_end: Actual traces to load (with overlap)
            - crop_start/crop_end: Indices to crop after processing
        """
        # Extend boundaries for overlap
        load_start = max(0, chunk_start - overlap_size)
        load_end = min(n_traces, chunk_end + overlap_size)

        # Calculate crop indices (relative to loaded chunk)
        crop_start = chunk_start - load_start
        crop_end = crop_start + (chunk_end - chunk_start)

        return load_start, load_end, crop_start, crop_end

    def _cleanup_partial_output(self, output_path: Path):
        """
        Clean up partial output on cancellation or error.

        Args:
            output_path: Path to output Zarr to delete
        """
        try:
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)
        except Exception as e:
            print(f"Warning: Failed to clean up partial output {output_path}: {e}")

    def cancel(self):
        """Cancel currently running processing operation."""
        self._cancel_flag.set()

    def is_processing(self) -> bool:
        """Check if processing is currently running."""
        return self._processing

    def process_with_metadata(
        self,
        input_zarr_path: Path,
        output_zarr_path: Path,
        processor: BaseProcessor,
        sample_rate: float,
        chunk_size: int = 5000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        overlap_percent: float = 0.10
    ) -> bool:
        """
        Process Zarr dataset with proper sample rate metadata.

        This version accepts sample_rate explicitly for proper SeismicData creation.

        Args:
            input_zarr_path: Path to input Zarr array
            output_zarr_path: Path to output Zarr array (will be created)
            processor: Processor to apply to each chunk
            sample_rate: Sample rate in seconds
            chunk_size: Number of traces per chunk
            progress_callback: Optional callback(current_trace, total_traces, time_remaining)
            overlap_percent: Overlap for boundary handling (0.0-0.5)

        Returns:
            True if processing completed, False if cancelled
        """
        self._cancel_flag.clear()
        self._processing = True

        try:
            # Validate inputs
            if chunk_size < 100:
                raise ValueError(f"chunk_size must be >= 100, got {chunk_size}")

            if not (0.0 <= overlap_percent <= 0.5):
                raise ValueError(f"overlap_percent must be in [0.0, 0.5], got {overlap_percent}")

            input_zarr_path = Path(input_zarr_path)
            output_zarr_path = Path(output_zarr_path)

            if not input_zarr_path.exists():
                raise ValueError(f"Input Zarr does not exist: {input_zarr_path}")

            # Open input Zarr
            input_zarr = zarr.open_array(str(input_zarr_path), mode='r')
            n_samples, n_traces = input_zarr.shape

            # Create output Zarr with same dimensions
            output_zarr = zarr.open_array(
                str(output_zarr_path),
                mode='w',
                shape=(n_samples, n_traces),
                chunks=(n_samples, min(chunk_size, n_traces)),
                dtype=input_zarr.dtype
            )

            # Calculate overlap size
            overlap_size = int(chunk_size * overlap_percent)

            # Process chunks
            start_time = time.time()
            traces_processed = 0

            for chunk_start in range(0, n_traces, chunk_size):
                # Check for cancellation
                if self._cancel_flag.is_set():
                    self._cleanup_partial_output(output_zarr_path)
                    return False

                # Calculate chunk boundaries
                chunk_end = min(chunk_start + chunk_size, n_traces)

                # Load chunk with overlap
                load_start, load_end, crop_start, crop_end = self._calculate_chunk_boundaries(
                    chunk_start, chunk_end, n_traces, overlap_size
                )

                # Load chunk data
                chunk_traces = np.array(input_zarr[:, load_start:load_end])

                # Create SeismicData for this chunk
                chunk_data = SeismicData(
                    traces=chunk_traces,
                    sample_rate=sample_rate,
                    metadata={
                        'chunk_start': load_start,
                        'chunk_end': load_end,
                        'chunk_with_overlap': True
                    }
                )

                # Process chunk
                processed_chunk = processor.process(chunk_data)

                # Clear GPU cache after processing chunk (prevents slowdown over time)
                # Works for any GPU-accelerated processor
                if hasattr(processor, 'device_manager') and processor.device_manager is not None:
                    processor.device_manager.clear_cache()

                # Crop to remove overlap regions
                if overlap_size > 0:
                    cropped_traces = processed_chunk.traces[:, crop_start:crop_end]
                else:
                    cropped_traces = processed_chunk.traces

                # Verify dimensions
                if cropped_traces.shape[1] != (chunk_end - chunk_start):
                    raise RuntimeError(
                        f"Dimension mismatch after cropping: expected {chunk_end - chunk_start} traces, "
                        f"got {cropped_traces.shape[1]}"
                    )

                # Write to output
                output_zarr[:, chunk_start:chunk_end] = cropped_traces

                # Update progress
                traces_processed = chunk_end
                if progress_callback is not None:
                    elapsed = time.time() - start_time
                    traces_per_sec = traces_processed / elapsed if elapsed > 0 else 0
                    remaining_traces = n_traces - traces_processed
                    time_remaining = remaining_traces / traces_per_sec if traces_per_sec > 0 else 0

                    progress_callback(traces_processed, n_traces, time_remaining)

            self._processing = False
            return True

        except Exception as e:
            self._processing = False
            # Clean up partial output on error
            if output_zarr_path.exists():
                self._cleanup_partial_output(output_zarr_path)
            raise

    def __repr__(self) -> str:
        status = "processing" if self._processing else "idle"
        return f"ChunkedProcessor(status={status})"
