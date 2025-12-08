"""
Coordinator for parallel multiprocess gather processing.

Orchestrates the full processing pipeline:
1. Validate inputs and load metadata
2. Pre-create shared output Zarr array
3. Partition gathers across workers
4. Launch and monitor worker processes
5. If sorting enabled, create sorted headers from worker mappings
6. Handle progress updates and cancellation
"""

import os
import gc
import json
import time
import shutil
import pickle
import numpy as np
import zarr
import pandas as pd
import psutil
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .config import (
    ProcessingConfig,
    ProcessingTask,
    ProcessingWorkerResult,
    ProcessingProgress,
    ProcessingResult,
    GatherSegment,
    SortOptions
)
from .partitioner import GatherPartitioner
from .worker import process_gather_range, read_streaming_sort_file

logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Get optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    # Leave 2 cores free for system/UI, minimum 2 workers
    return max(2, cpu_count - 2)


def check_sorting_memory_budget(
    n_traces: int,
    n_header_columns: int = 50,
    safety_factor: float = 0.7
) -> Tuple[bool, float, float, str]:
    """
    Pre-flight memory check before sorting operation.

    Args:
        n_traces: Number of traces to process
        n_header_columns: Estimated number of header columns
        safety_factor: Fraction of available memory considered safe (default 70%)

    Returns:
        Tuple of (is_safe, available_mb, required_mb, message)
    """
    available_bytes = psutil.virtual_memory().available
    available_mb = available_bytes / (1024**2)

    # Estimate memory requirements for sorting
    # 1. Global mapping array: int64 per trace
    mapping_mb = (n_traces * 8) / (1024**2)

    # 2. Headers DataFrame (estimate ~8 bytes per value average)
    headers_mb = (n_header_columns * n_traces * 8) / (1024**2)

    # 3. Sorted headers copy during reorder
    sorted_copy_mb = headers_mb

    # Total peak estimate
    required_mb = mapping_mb + headers_mb + sorted_copy_mb

    safe_available = available_mb * safety_factor
    is_safe = required_mb < safe_available

    if is_safe:
        message = (
            f"Memory OK: ~{required_mb:.0f} MB required for sorting, "
            f"{available_mb:.0f} MB available"
        )
    else:
        message = (
            f"MEMORY WARNING: Sorting requires ~{required_mb:.0f} MB, "
            f"only {available_mb:.0f} MB available ({safety_factor*100:.0f}% threshold). "
            f"Consider disabling sorting or processing smaller batches."
        )

    return is_safe, available_mb, required_mb, message


class ParallelProcessingCoordinator:
    """
    Orchestrates parallel multiprocess gather processing.

    Workers write directly to a pre-created shared Zarr array,
    eliminating the need for output merging.

    When sorting is enabled, traces are sorted within each gather
    and a sorted headers.parquet is created for export.

    Usage:
        config = ProcessingConfig(
            input_storage_dir='/path/to/input',
            output_storage_dir='/path/to/output',
            processor_config=processor.to_dict(),
            sort_options=SortOptions(enabled=True, sort_key='offset')
        )
        coordinator = ParallelProcessingCoordinator(config)
        result = coordinator.run(progress_callback=update_ui)
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize coordinator.

        Args:
            config: Processing configuration
        """
        self.config = config
        self.n_workers = config.n_workers or get_optimal_workers()
        self._cancel_requested = False

    def run(
        self,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        Run the full parallel processing pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with outcome
        """
        start_time = time.time()
        input_dir = Path(self.config.input_storage_dir)
        output_dir = Path(self.config.output_storage_dir)
        sorting_enabled = (
            self.config.sort_options is not None and
            self.config.sort_options.enabled
        )

        try:
            # Phase 1: Validate and load metadata
            if progress_callback:
                progress_callback(ProcessingProgress(
                    phase='initializing',
                    current_traces=0,
                    total_traces=0,
                    current_gathers=0,
                    total_gathers=0,
                    active_workers=0
                ))

            metadata, ensemble_df = self._load_and_validate(input_dir)
            n_traces = metadata['n_traces']
            n_samples = metadata['n_samples']
            n_gathers = len(ensemble_df)
            sample_rate = metadata['sample_rate']

            sort_info = ""
            if sorting_enabled:
                sort_info = f" (sorting by {self.config.sort_options.sort_key})"

                # MEMORY GUARD: Check if we have enough memory for sorting
                is_safe, available_mb, required_mb, mem_message = check_sorting_memory_budget(
                    n_traces=n_traces,
                    safety_factor=0.7
                )
                logger.info(mem_message)

                if not is_safe:
                    # Log warning but allow to proceed (user can abort if needed)
                    logger.warning(
                        f"Low memory warning for sorting operation. "
                        f"Required: ~{required_mb:.0f}MB, Available: {available_mb:.0f}MB. "
                        f"Processing will continue but may fail."
                    )
                    print(f"  WARNING: {mem_message}")

            print(f"  Processing {n_gathers:,} gathers, {n_traces:,} traces{sort_info}")

            # Phase 2: Partition gathers across workers
            segments = self._partition_gathers(ensemble_df)
            print(f"  Partitioned into {len(segments)} segments")

            # Phase 3: Create output directory and shared Zarr
            output_dir.mkdir(parents=True, exist_ok=True)
            output_zarr_path = self._create_output_zarr(output_dir, n_samples, n_traces)
            print(f"  Created output Zarr: {n_samples} x {n_traces:,}")

            # Create temp directory for sort mappings if sorting enabled
            temp_dir = None
            if sorting_enabled:
                temp_dir = output_dir / 'temp_sort'
                temp_dir.mkdir(parents=True, exist_ok=True)

            # Phase 4: Run parallel workers
            if progress_callback:
                progress_callback(ProcessingProgress(
                    phase='processing',
                    current_traces=0,
                    total_traces=n_traces,
                    current_gathers=0,
                    total_gathers=n_gathers,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments=segments,
                input_dir=input_dir,
                output_zarr_path=output_zarr_path,
                n_samples=n_samples,
                sample_rate=sample_rate,
                metadata=metadata,
                n_traces=n_traces,
                n_gathers=n_gathers,
                temp_dir=temp_dir,
                progress_callback=progress_callback
            )

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            # Phase 5: Create sorted headers if sorting was enabled
            if progress_callback:
                progress_callback(ProcessingProgress(
                    phase='finalizing',
                    current_traces=n_traces,
                    total_traces=n_traces,
                    current_gathers=n_gathers,
                    total_gathers=n_gathers,
                    active_workers=0
                ))

            if sorting_enabled:
                print("  Creating sorted headers...")
                self._create_sorted_headers(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    worker_results=worker_results,
                    ensemble_df=ensemble_df,
                    n_traces=n_traces
                )
                # Cleanup temp directory
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            else:
                # Copy metadata files without sorting
                self._copy_metadata_files(input_dir, output_dir, metadata)

            # Update metadata with processing info
            self._save_processing_metadata(output_dir, metadata, len(segments), sorting_enabled)

            elapsed = time.time() - start_time
            throughput = n_traces / elapsed if elapsed > 0 else 0

            print(f"  Processing complete: {n_traces:,} traces in {elapsed:.1f}s "
                  f"({throughput:,.0f} traces/sec)")

            return ProcessingResult(
                success=True,
                output_dir=str(output_dir),
                output_zarr_path=output_zarr_path,
                n_gathers=n_gathers,
                n_traces=n_traces,
                n_samples=n_samples,
                elapsed_time=elapsed,
                throughput_traces_per_sec=throughput,
                n_workers_used=len(segments)
            )

        except Exception as e:
            import traceback
            elapsed = time.time() - start_time

            # Cleanup on failure
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                except Exception:
                    pass

            return ProcessingResult(
                success=False,
                output_dir=str(output_dir),
                output_zarr_path="",
                n_gathers=0,
                n_traces=0,
                n_samples=0,
                elapsed_time=elapsed,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    def _load_and_validate(self, input_dir: Path) -> tuple:
        """Load and validate input data."""
        # Check required files
        zarr_path = input_dir / 'traces.zarr'
        metadata_path = input_dir / 'metadata.json'
        ensemble_path = input_dir / 'ensemble_index.parquet'

        if not zarr_path.exists():
            raise FileNotFoundError(f"Input Zarr not found: {zarr_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble index not found: {ensemble_path}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load ensemble index
        ensemble_df = pd.read_parquet(ensemble_path)

        # Validate
        if 'n_traces' not in metadata or 'n_samples' not in metadata:
            raise ValueError("Metadata missing required fields (n_traces, n_samples)")

        required_cols = ['start_trace', 'end_trace', 'n_traces']
        missing = [c for c in required_cols if c not in ensemble_df.columns]
        if missing:
            raise ValueError(f"Ensemble index missing columns: {missing}")

        return metadata, ensemble_df

    def _partition_gathers(self, ensemble_df: pd.DataFrame) -> List[GatherSegment]:
        """Partition gathers across workers."""
        partitioner = GatherPartitioner(ensemble_df, self.n_workers)
        segments = partitioner.partition()

        # Log partition stats
        stats = partitioner.get_partition_stats(segments)
        print(f"  Partition stats: {stats}")

        return segments

    def _create_output_zarr(self, output_dir: Path, n_samples: int, n_traces: int) -> str:
        """Create pre-allocated output Zarr array."""
        output_path = output_dir / 'traces.zarr'

        zarr.open(
            str(output_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 1000),
            dtype=np.float32,
            compressor=None,  # No compression for speed
            zarr_format=2
        )

        return str(output_path)

    def _copy_metadata_files(self, input_dir: Path, output_dir: Path, metadata: dict):
        """Copy metadata and index files to output."""
        # Copy headers parquet
        headers_src = input_dir / 'headers.parquet'
        if headers_src.exists():
            shutil.copy2(headers_src, output_dir / 'headers.parquet')

        # Copy ensemble index
        ensemble_src = input_dir / 'ensemble_index.parquet'
        if ensemble_src.exists():
            shutil.copy2(ensemble_src, output_dir / 'ensemble_index.parquet')

        # Copy trace index if exists
        trace_idx_src = input_dir / 'trace_index.parquet'
        if trace_idx_src.exists():
            shutil.copy2(trace_idx_src, output_dir / 'trace_index.parquet')

    def _run_workers(
        self,
        segments: List[GatherSegment],
        input_dir: Path,
        output_zarr_path: str,
        n_samples: int,
        sample_rate: float,
        metadata: dict,
        n_traces: int,
        n_gathers: int,
        temp_dir: Optional[Path],
        progress_callback: Optional[Callable]
    ) -> List[ProcessingWorkerResult]:
        """Run worker processes in parallel."""
        sorting_enabled = (
            self.config.sort_options is not None and
            self.config.sort_options.enabled
        )

        # Create worker tasks
        tasks = []
        for segment in segments:
            # Set up sort mapping path if sorting enabled
            sort_mapping_path = None
            if sorting_enabled and temp_dir:
                sort_mapping_path = str(temp_dir / f'sort_mapping_{segment.segment_id}.pkl')

            task = ProcessingTask(
                segment_id=segment.segment_id,
                input_zarr_path=str(input_dir / 'traces.zarr'),
                output_zarr_path=output_zarr_path,
                headers_parquet_path=str(input_dir / 'headers.parquet'),
                ensemble_index_path=str(input_dir / 'ensemble_index.parquet'),
                processor_config=self.config.processor_config,
                start_gather=segment.start_gather,
                end_gather=segment.end_gather,
                n_samples=n_samples,
                sample_rate=sample_rate,
                metadata=metadata,
                sort_options=self.config.sort_options if sorting_enabled else None,
                sort_mapping_path=sort_mapping_path
            )
            tasks.append(task)

        # Use multiprocessing Manager for progress queue
        manager = mp.Manager()
        progress_queue = manager.Queue()

        # Track progress per worker
        worker_progress_traces = {s.segment_id: 0 for s in segments}
        worker_progress_gathers = {s.segment_id: 0 for s in segments}
        results = []
        processed_futures = set()

        print(f"  Launching {len(tasks)} worker processes...")

        # Submit all tasks
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(process_gather_range, task, progress_queue): task
                for task in tasks
            }

            # Monitor progress
            start_time = time.time()

            while len(processed_futures) < len(futures):
                # Check completed futures
                for future in futures:
                    if future.done() and future not in processed_futures:
                        processed_futures.add(future)
                        try:
                            result = future.result(timeout=0.1)
                            results.append(result)
                            worker_progress_traces[result.segment_id] = result.n_traces_processed
                            worker_progress_gathers[result.segment_id] = result.n_gathers_processed
                            print(f"    Worker {result.segment_id} completed: "
                                  f"{result.n_gathers_processed} gathers, "
                                  f"{result.n_traces_processed:,} traces in {result.elapsed_time:.1f}s")
                        except Exception as e:
                            task = futures[future]
                            results.append(ProcessingWorkerResult(
                                segment_id=task.segment_id,
                                n_gathers_processed=0,
                                n_traces_processed=0,
                                elapsed_time=0,
                                success=False,
                                error=str(e)
                            ))

                # Drain progress queue
                while not progress_queue.empty():
                    try:
                        segment_id, traces_done, gathers_done = progress_queue.get_nowait()
                        worker_progress_traces[segment_id] = traces_done
                        worker_progress_gathers[segment_id] = gathers_done
                    except:
                        break

                # Update progress callback
                if progress_callback:
                    total_traces_done = sum(worker_progress_traces.values())
                    total_gathers_done = sum(worker_progress_gathers.values())
                    elapsed = time.time() - start_time
                    rate = total_traces_done / elapsed if elapsed > 0 else 0
                    eta = (n_traces - total_traces_done) / rate if rate > 0 else 0

                    progress_callback(ProcessingProgress(
                        phase='processing',
                        current_traces=total_traces_done,
                        total_traces=n_traces,
                        current_gathers=total_gathers_done,
                        total_gathers=n_gathers,
                        active_workers=len(futures) - len(processed_futures),
                        worker_progress=worker_progress_traces.copy(),
                        elapsed_time=elapsed,
                        eta_seconds=eta
                    ))

                # Small sleep to avoid busy-waiting
                time.sleep(0.1)

        return results

    def _create_sorted_headers(
        self,
        input_dir: Path,
        output_dir: Path,
        worker_results: List[ProcessingWorkerResult],
        ensemble_df: pd.DataFrame,
        n_traces: int
    ):
        """
        Create sorted headers.parquet from worker sort mappings.

        MEMORY OPTIMIZATIONS:
        - Uses streaming sort file format (reads incrementally)
        - Vectorized mapping construction (no Python loops)
        - Chunked header reordering (avoids full DataFrame copy)
        - Explicit garbage collection between phases

        Each worker saved sort mappings in streaming format.
        We load these, build a global reorder index, and create sorted headers.
        """
        logger.info(f"Creating sorted headers for {n_traces:,} traces...")

        # Phase 1: Build global sort mapping using vectorized operations
        print("    Building global sort mapping (vectorized)...")
        global_mapping = np.arange(n_traces, dtype=np.int64)  # Identity by default

        # Load and apply each worker's sort mappings
        for result in worker_results:
            if result.sort_mapping_path and Path(result.sort_mapping_path).exists():
                # Use streaming reader (handles both new binary and legacy pickle format)
                sort_mappings = read_streaming_sort_file(result.sort_mapping_path)

                # VECTORIZED: Apply each gather's sort mapping without Python loops
                for gather_idx, g_start, g_end, local_sort_indices in sort_mappings:
                    n_gather_traces = len(local_sort_indices)
                    if n_gather_traces > 0:
                        # Vectorized index computation - no inner loop!
                        old_global_positions = g_start + local_sort_indices
                        global_mapping[g_start:g_start + n_gather_traces] = old_global_positions

                # Free mappings memory immediately
                del sort_mappings
                gc.collect()

        # Phase 2: Chunked header reordering to avoid full DataFrame copy
        print("    Reordering headers (chunked)...")
        CHUNK_SIZE = 100_000  # Process 100k rows at a time

        headers_path = input_dir / 'headers.parquet'
        output_headers_path = output_dir / 'headers.parquet'

        # Load original headers
        headers_df = pd.read_parquet(headers_path)
        n_columns = len(headers_df.columns)
        logger.info(f"Headers loaded: {len(headers_df):,} rows, {n_columns} columns")

        # Process in chunks to avoid memory spike from full copy
        if n_traces > CHUNK_SIZE:
            # Large dataset: chunked processing
            chunks = []
            n_chunks = (n_traces + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * CHUNK_SIZE
                chunk_end = min(chunk_start + CHUNK_SIZE, n_traces)
                chunk_mapping = global_mapping[chunk_start:chunk_end]

                # Extract only the rows we need for this chunk
                chunk_headers = headers_df.iloc[chunk_mapping].copy()
                chunk_headers['trace_index'] = np.arange(chunk_start, chunk_end)
                chunk_headers['original_trace_index'] = chunk_mapping

                chunks.append(chunk_headers)

                # Progress update for large datasets
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                    logger.debug(f"Header reorder: chunk {chunk_idx + 1}/{n_chunks}")

                # Cleanup
                del chunk_mapping
                if chunk_idx % 5 == 0:
                    gc.collect()

            # Concatenate chunks
            sorted_headers = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            # Small dataset: direct reorder is OK
            sorted_headers = headers_df.iloc[global_mapping].reset_index(drop=True)
            sorted_headers['trace_index'] = np.arange(len(sorted_headers))
            sorted_headers['original_trace_index'] = global_mapping

        # Free original headers memory
        del headers_df
        del global_mapping
        gc.collect()

        # Phase 3: Save sorted headers
        print("    Saving sorted headers...")
        sorted_headers.to_parquet(output_headers_path)

        n_saved = len(sorted_headers)
        del sorted_headers
        gc.collect()

        # Copy ensemble index (gather boundaries remain the same, just internal order changed)
        shutil.copy2(input_dir / 'ensemble_index.parquet', output_dir / 'ensemble_index.parquet')

        # Copy trace index if exists
        trace_idx_src = input_dir / 'trace_index.parquet'
        if trace_idx_src.exists():
            shutil.copy2(trace_idx_src, output_dir / 'trace_index.parquet')

        print(f"    Sorted headers saved: {n_saved:,} traces")
        logger.info(f"Sorted headers complete: {n_saved:,} traces")

    def _save_processing_metadata(
        self,
        output_dir: Path,
        original_metadata: dict,
        n_workers: int,
        sorting_enabled: bool
    ):
        """Save processing metadata, preserving original SEG-Y path for export."""
        metadata = original_metadata.copy()
        processing_info = {
            'method': 'parallel_multiprocess',
            'n_workers': n_workers,
            'processor_config': self.config.processor_config
        }

        if sorting_enabled and self.config.sort_options:
            processing_info['sorting'] = {
                'enabled': True,
                'sort_key': self.config.sort_options.sort_key,
                'ascending': self.config.sort_options.ascending,
                'secondary_key': self.config.sort_options.secondary_key,
                'secondary_ascending': self.config.sort_options.secondary_ascending
            }

        metadata['processing_info'] = processing_info

        # CRITICAL: Preserve original_segy_path for export functionality
        # The path may be at top level or nested in seismic_metadata
        original_segy_path = None

        # Check top-level first
        if 'original_segy_path' in original_metadata:
            original_segy_path = original_metadata['original_segy_path']

        # Fall back to seismic_metadata section
        if not original_segy_path:
            seismic_meta = original_metadata.get('seismic_metadata', {})
            original_segy_path = seismic_meta.get('original_segy_path') or seismic_meta.get('source_file')

        # Ensure path is at top level for export to find it
        if original_segy_path:
            metadata['original_segy_path'] = original_segy_path
            # Also preserve in seismic_metadata for consistency
            if 'seismic_metadata' not in metadata:
                metadata['seismic_metadata'] = {}
            metadata['seismic_metadata']['original_segy_path'] = original_segy_path
            logger.info(f"Preserved original SEG-Y path: {original_segy_path}")
        else:
            logger.warning(
                "No original_segy_path found in input metadata. "
                "Export to SEG-Y may not work. Re-import original SEG-Y to fix."
            )

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def cancel(self):
        """Request cancellation of processing."""
        self._cancel_requested = True
