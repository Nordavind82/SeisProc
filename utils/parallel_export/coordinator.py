"""
Coordinator for parallel SEG-Y export.

Orchestrates the full export pipeline:
1. Vectorize headers for fast access
2. Partition traces across workers
3. Launch worker processes
4. Merge segment files
5. Cleanup temporary files
"""

import os
import time
import shutil
import numpy as np
import zarr
import pandas as pd
import segyio
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import gc
import pickle

from .config import (
    ExportConfig,
    ExportTask,
    ExportWorkerResult,
    ExportProgress,
    ExportResult,
    TraceSegment
)
from .header_vectorizer import HeaderVectorizer
from .worker import export_trace_range
from .merger import SEGYSegmentMerger


@dataclass
class ExportStageResult:
    """
    Intermediate result from parallel export stage.

    Holds all data needed for subsequent merge and cleanup stages.
    """
    segment_paths: List[str]
    segments: List[TraceSegment]
    n_traces: int
    n_samples: int
    data_format: int
    temp_dir: Path
    start_time: float
    success: bool = True
    error: Optional[str] = None


def get_optimal_workers(n_traces: int = 0, n_header_fields: int = 50) -> int:
    """
    Get optimal number of worker processes based on CPU and memory.

    Args:
        n_traces: Number of traces (for memory calculation)
        n_header_fields: Estimated number of header fields

    Returns:
        Optimal worker count
    """
    import psutil

    cpu_count = os.cpu_count() or 4
    cpu_based = max(2, cpu_count - 2)

    # Memory-based limit: each worker loads header slice + chunk data
    # Estimate ~500MB base overhead per worker + header slice
    try:
        available_gb = psutil.virtual_memory().available / (1024**3)
        # Reserve 4GB for system/PyCharm, 500MB per worker base
        usable_gb = max(1, available_gb - 4)
        memory_based = max(2, int(usable_gb / 0.5))
    except Exception:
        memory_based = 4  # Conservative fallback

    return min(cpu_based, memory_based)


class ParallelExportCoordinator:
    """
    Orchestrates parallel SEG-Y export.

    Workers each write to separate segment files, which are then
    merged into the final output. This approach:
    - Bypasses Python GIL via multiprocessing
    - Uses vectorized headers for O(1) header access
    - Enables parallel I/O across multiple files

    Usage (single dialog - backwards compatible):
        coordinator = ParallelExportCoordinator(config)
        result = coordinator.run(progress_callback=update_ui)

    Usage (separate stage dialogs - recommended):
        coordinator = ParallelExportCoordinator(config)

        # Stage 1: Parallel export with traces progress
        stage_result = coordinator.run_parallel_export(progress_callback=on_export_progress)

        # Stage 2: Merge with bytes progress
        coordinator.run_merge(stage_result, progress_callback=on_merge_progress)

        # Stage 3: Cleanup with files progress
        coordinator.run_cleanup(stage_result, progress_callback=on_cleanup_progress)

        # Get final result
        result = coordinator.get_final_result(stage_result)
    """

    def __init__(self, config: ExportConfig):
        """
        Initialize coordinator.

        Args:
            config: Export configuration
        """
        self.config = config
        self.n_workers = config.n_workers or get_optimal_workers()
        self._cancel_requested = False
        self._was_cancelled = False

    def run(
        self,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None
    ) -> ExportResult:
        """
        Run the full parallel export pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            ExportResult with outcome
        """
        start_time = time.time()
        temp_dir = Path(self.config.temp_dir)
        segment_paths = []

        try:
            # Phase 1: Load and validate metadata
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='initializing',
                    current_traces=0,
                    total_traces=0,
                    active_workers=0
                ))

            # Get info from original SEG-Y
            with segyio.open(self.config.original_segy_path, 'r', ignore_geometry=True) as f:
                n_traces = f.tracecount
                n_samples = len(f.samples)
                sample_interval = int(segyio.tools.dt(f))  # microseconds
                data_format = int(f.bin[segyio.BinField.Format])

            print(f"  Exporting {n_traces:,} traces, {n_samples} samples")

            # Validate Zarr matches
            processed_zarr = zarr.open(self.config.processed_zarr_path, mode='r')
            if processed_zarr.shape != (n_samples, n_traces):
                raise ValueError(
                    f"Zarr shape {processed_zarr.shape} doesn't match "
                    f"SEG-Y ({n_samples}, {n_traces})"
                )

            # Phase 2: Vectorize headers
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='vectorizing',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=0
                ))

            print("  Vectorizing headers...")
            headers_df = pd.read_parquet(self.config.headers_parquet_path)
            vectorizer = HeaderVectorizer(headers_df)
            vectorizer.vectorize()

            stats = vectorizer.get_stats()
            print(f"    {stats['n_fields']} fields, {stats['memory_mb']:.1f} MB total")

            # Phase 3: Partition traces BEFORE saving headers
            # so we can save per-segment slices to reduce worker memory
            segments = self._partition_traces(n_traces)

            # Save per-segment header slices (not full arrays)
            # This dramatically reduces memory: each worker only loads its slice
            temp_dir.mkdir(parents=True, exist_ok=True)
            header_arrays = vectorizer._header_arrays

            for segment in segments:
                segment_header_path = temp_dir / f'headers_segment_{segment.segment_id}.pkl'
                # Extract only the slice this worker needs
                segment_headers = {
                    field: arr[segment.start_trace:segment.end_trace + 1].copy()
                    for field, arr in header_arrays.items()
                }
                with open(segment_header_path, 'wb') as f:
                    pickle.dump(segment_headers, f)

            # Release DataFrame and full header arrays to free memory
            del headers_df
            del vectorizer
            del header_arrays
            gc.collect()

            print(f"    Saved {len(segments)} header slices")
            print(f"  Partitioned into {len(segments)} segments")

            # Phase 4: Run parallel workers
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='exporting',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments=segments,
                temp_dir=temp_dir,
                n_samples=n_samples,
                sample_interval=sample_interval,
                data_format=data_format,
                n_traces=n_traces,
                progress_callback=progress_callback
            )

            # Collect segment paths in order
            segment_paths = [r.output_path for r in sorted(worker_results, key=lambda r: r.segment_id)]

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            # Phase 5: Merge segments
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='merging',
                    current_traces=n_traces,
                    total_traces=n_traces,
                    active_workers=0
                ))

            print("  Merging segments...")
            merger = SEGYSegmentMerger(n_samples, data_format)
            total_traces = merger.merge(segment_paths, self.config.output_path)

            # Phase 6: Cleanup
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='finalizing',
                    current_traces=n_traces,
                    total_traces=n_traces,
                    active_workers=0
                ))

            # Remove segment files
            for path in segment_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass

            # Remove header slice files
            for segment in segments:
                try:
                    os.remove(temp_dir / f'headers_segment_{segment.segment_id}.pkl')
                except Exception:
                    pass

            # Get output file size
            output_size = Path(self.config.output_path).stat().st_size

            elapsed = time.time() - start_time
            throughput = n_traces / elapsed if elapsed > 0 else 0

            print(f"  Export complete: {n_traces:,} traces in {elapsed:.1f}s "
                  f"({throughput:,.0f} traces/sec)")

            return ExportResult(
                success=True,
                output_path=self.config.output_path,
                n_traces=n_traces,
                n_samples=n_samples,
                file_size_bytes=output_size,
                elapsed_time=elapsed,
                throughput_traces_per_sec=throughput,
                n_workers_used=len(segments)
            )

        except Exception as e:
            import traceback
            elapsed = time.time() - start_time

            # Cleanup on failure
            for path in segment_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass

            return ExportResult(
                success=False,
                output_path=self.config.output_path,
                n_traces=0,
                n_samples=0,
                file_size_bytes=0,
                elapsed_time=elapsed,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE-BASED API - For separate progress dialogs per stage
    # ═══════════════════════════════════════════════════════════════════════════

    def run_parallel_export(
        self,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None
    ) -> ExportStageResult:
        """
        Stage 1: Run parallel export workers.

        This stage handles initialization, header vectorization, partitioning,
        and parallel worker execution. Returns intermediate result for subsequent stages.

        Args:
            progress_callback: Callback for ExportProgress updates (traces-based)

        Returns:
            ExportStageResult with segment paths and metadata for merge/cleanup stages
        """
        start_time = time.time()
        temp_dir = Path(self.config.temp_dir)
        segment_paths = []
        segments = []

        try:
            # Phase 1: Load and validate metadata
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='initializing',
                    current_traces=0,
                    total_traces=0,
                    active_workers=0
                ))

            # Get info from original SEG-Y
            with segyio.open(self.config.original_segy_path, 'r', ignore_geometry=True) as f:
                n_traces = f.tracecount
                n_samples = len(f.samples)
                sample_interval = int(segyio.tools.dt(f))
                data_format = int(f.bin[segyio.BinField.Format])

            print(f"  Exporting {n_traces:,} traces, {n_samples} samples")

            # Validate Zarr matches
            processed_zarr = zarr.open(self.config.processed_zarr_path, mode='r')
            if processed_zarr.shape != (n_samples, n_traces):
                raise ValueError(
                    f"Zarr shape {processed_zarr.shape} doesn't match "
                    f"SEG-Y ({n_samples}, {n_traces})"
                )

            # Phase 2: Vectorize headers
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='vectorizing',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=0
                ))

            print("  Vectorizing headers...")
            headers_df = pd.read_parquet(self.config.headers_parquet_path)
            vectorizer = HeaderVectorizer(headers_df)
            vectorizer.vectorize()

            stats = vectorizer.get_stats()
            print(f"    {stats['n_fields']} fields, {stats['memory_mb']:.1f} MB total")

            # Phase 3: Partition traces
            segments = self._partition_traces(n_traces)

            # Save per-segment header slices
            temp_dir.mkdir(parents=True, exist_ok=True)
            header_arrays = vectorizer._header_arrays

            for segment in segments:
                segment_header_path = temp_dir / f'headers_segment_{segment.segment_id}.pkl'
                segment_headers = {
                    field: arr[segment.start_trace:segment.end_trace + 1].copy()
                    for field, arr in header_arrays.items()
                }
                with open(segment_header_path, 'wb') as f:
                    pickle.dump(segment_headers, f)

            # Release memory
            del headers_df
            del vectorizer
            del header_arrays
            gc.collect()

            print(f"    Saved {len(segments)} header slices")
            print(f"  Partitioned into {len(segments)} segments")

            # Phase 4: Run parallel workers
            if progress_callback:
                progress_callback(ExportProgress(
                    phase='exporting',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments=segments,
                temp_dir=temp_dir,
                n_samples=n_samples,
                sample_interval=sample_interval,
                data_format=data_format,
                n_traces=n_traces,
                progress_callback=progress_callback
            )

            # Check for cancellation
            if self._cancel_requested:
                self._was_cancelled = True
                # Cleanup on cancel
                for r in worker_results:
                    try:
                        os.remove(r.output_path)
                    except Exception:
                        pass
                return ExportStageResult(
                    segment_paths=[],
                    segments=segments,
                    n_traces=n_traces,
                    n_samples=n_samples,
                    data_format=data_format,
                    temp_dir=temp_dir,
                    start_time=start_time,
                    success=False,
                    error="Export cancelled by user"
                )

            # Collect segment paths in order
            segment_paths = [r.output_path for r in sorted(worker_results, key=lambda r: r.segment_id)]

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            return ExportStageResult(
                segment_paths=segment_paths,
                segments=segments,
                n_traces=n_traces,
                n_samples=n_samples,
                data_format=data_format,
                temp_dir=temp_dir,
                start_time=start_time,
                success=True
            )

        except Exception as e:
            import traceback
            # Cleanup on failure
            for path in segment_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass

            return ExportStageResult(
                segment_paths=[],
                segments=segments,
                n_traces=0,
                n_samples=0,
                data_format=0,
                temp_dir=temp_dir,
                start_time=start_time,
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    def run_merge(
        self,
        stage_result: ExportStageResult,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Stage 2: Merge segment files into final output.

        Args:
            stage_result: Result from run_parallel_export()
            progress_callback: Callback(bytes_written, total_bytes) for progress

        Returns:
            Total number of traces in merged file
        """
        if not stage_result.success:
            raise RuntimeError(f"Cannot merge: previous stage failed - {stage_result.error}")

        print("  Merging segments...")
        merger = SEGYSegmentMerger(stage_result.n_samples, stage_result.data_format)
        total_traces = merger.merge(
            stage_result.segment_paths,
            self.config.output_path,
            progress_callback=progress_callback
        )
        return total_traces

    def run_cleanup(
        self,
        stage_result: ExportStageResult,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stage 3: Cleanup temporary segment and header files.

        Args:
            stage_result: Result from run_parallel_export()
            progress_callback: Callback(files_cleaned, total_files) for progress
        """
        total_files = len(stage_result.segment_paths) + len(stage_result.segments)
        cleaned = 0

        # Remove segment files
        for path in stage_result.segment_paths:
            try:
                os.remove(path)
            except Exception:
                pass
            cleaned += 1
            if progress_callback:
                progress_callback(cleaned, total_files)

        # Remove header slice files
        for segment in stage_result.segments:
            try:
                os.remove(stage_result.temp_dir / f'headers_segment_{segment.segment_id}.pkl')
            except Exception:
                pass
            cleaned += 1
            if progress_callback:
                progress_callback(cleaned, total_files)

    def get_merge_stats(self, stage_result: ExportStageResult) -> Dict[str, Any]:
        """
        Get statistics about the merge operation before running it.

        Useful for setting up the merge progress dialog with correct total.

        Args:
            stage_result: Result from run_parallel_export()

        Returns:
            Dictionary with merge statistics including total_bytes
        """
        merger = SEGYSegmentMerger(stage_result.n_samples, stage_result.data_format)
        return merger.get_merge_stats(stage_result.segment_paths)

    def get_final_result(
        self,
        stage_result: ExportStageResult
    ) -> ExportResult:
        """
        Build final ExportResult after all stages complete.

        Args:
            stage_result: Result from run_parallel_export()

        Returns:
            Final ExportResult with success status and statistics
        """
        elapsed = time.time() - stage_result.start_time
        throughput = stage_result.n_traces / elapsed if elapsed > 0 else 0

        if not stage_result.success:
            return ExportResult(
                success=False,
                output_path=self.config.output_path,
                n_traces=0,
                n_samples=0,
                file_size_bytes=0,
                elapsed_time=elapsed,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error=stage_result.error
            )

        # Get output file size
        output_size = Path(self.config.output_path).stat().st_size

        print(f"  Export complete: {stage_result.n_traces:,} traces in {elapsed:.1f}s "
              f"({throughput:,.0f} traces/sec)")

        return ExportResult(
            success=True,
            output_path=self.config.output_path,
            n_traces=stage_result.n_traces,
            n_samples=stage_result.n_samples,
            file_size_bytes=output_size,
            elapsed_time=elapsed,
            throughput_traces_per_sec=throughput,
            n_workers_used=len(stage_result.segments)
        )

    @property
    def was_cancelled(self) -> bool:
        """Check if export was cancelled."""
        return self._was_cancelled

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _partition_traces(self, n_traces: int) -> List[TraceSegment]:
        """Partition traces across workers."""
        segments = []
        traces_per_worker = n_traces // self.n_workers
        remainder = n_traces % self.n_workers

        start = 0
        for i in range(self.n_workers):
            # Distribute remainder across first workers
            n = traces_per_worker + (1 if i < remainder else 0)
            if n == 0:
                continue

            segments.append(TraceSegment(
                segment_id=i,
                start_trace=start,
                end_trace=start + n - 1,
                n_traces=n
            ))
            start += n

        return segments

    def _run_workers(
        self,
        segments: List[TraceSegment],
        temp_dir: Path,
        n_samples: int,
        sample_interval: int,
        data_format: int,
        n_traces: int,
        progress_callback: Optional[Callable]
    ) -> List[ExportWorkerResult]:
        """Run worker processes in parallel."""
        # Create tasks with segment-specific header paths
        tasks = []
        for segment in segments:
            segment_path = str(temp_dir / f'segment_{segment.segment_id}.sgy')
            # Each worker gets only its header slice, not the full arrays
            segment_header_path = str(temp_dir / f'headers_segment_{segment.segment_id}.pkl')
            # Convert header_mapping to serializable list for multiprocessing
            header_mapping_list = None
            if self.config.header_mapping:
                header_mapping_list = [
                    {
                        'parquet_column': hm.parquet_column,
                        'segy_byte_pos': hm.segy_byte_pos,
                        'format': hm.format
                    }
                    for hm in self.config.header_mapping.values()
                ]

            task = ExportTask(
                segment_id=segment.segment_id,
                original_segy_path=self.config.original_segy_path,
                processed_zarr_path=self.config.processed_zarr_path,
                output_segment_path=segment_path,
                header_arrays_path=segment_header_path,
                start_trace=segment.start_trace,
                end_trace=segment.end_trace,
                n_samples=n_samples,
                sample_interval=sample_interval,
                data_format=data_format,
                is_first_segment=(segment.segment_id == 0),
                # Export type and mute configuration
                export_type=self.config.export_type,
                input_zarr_path=self.config.input_zarr_path,
                mute_velocity=self.config.mute_velocity,
                mute_top=self.config.mute_top,
                mute_bottom=self.config.mute_bottom,
                mute_taper=self.config.mute_taper,
                mute_target=self.config.mute_target,
                header_mapping_list=header_mapping_list
            )
            tasks.append(task)

        # Use multiprocessing Manager for progress queue with bounded size
        # to prevent memory issues from queue overflow
        manager = mp.Manager()
        progress_queue = manager.Queue(maxsize=1000)

        # Track progress per worker
        worker_progress = {s.segment_id: 0 for s in segments}
        results = []
        processed_futures = set()

        print(f"  Launching {len(tasks)} export workers...")

        try:
            # Use SPAWN context instead of fork to avoid copying parent memory
            # Fork copies entire parent memory space which causes OOM with large datasets
            spawn_ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=spawn_ctx) as executor:
                futures = {
                    executor.submit(export_trace_range, task, progress_queue): task
                    for task in tasks
                }

                start_time = time.time()

                while len(processed_futures) < len(futures):
                    # Check completed futures
                    for future in futures:
                        if future.done() and future not in processed_futures:
                            processed_futures.add(future)
                            try:
                                result = future.result(timeout=0.1)
                                results.append(result)
                                worker_progress[result.segment_id] = result.n_traces_exported
                                print(f"    Worker {result.segment_id} completed: "
                                      f"{result.n_traces_exported:,} traces in {result.elapsed_time:.1f}s")
                            except Exception as e:
                                task = futures[future]
                                results.append(ExportWorkerResult(
                                    segment_id=task.segment_id,
                                    n_traces_exported=0,
                                    output_path=task.output_segment_path,
                                    file_size_bytes=0,
                                    elapsed_time=0,
                                    success=False,
                                    error=str(e)
                                ))

                    # Drain progress queue aggressively to prevent blocking workers
                    drain_count = 0
                    max_drain = 500
                    while drain_count < max_drain:
                        try:
                            segment_id, traces_done = progress_queue.get_nowait()
                            worker_progress[segment_id] = traces_done
                            drain_count += 1
                        except:
                            break

                    # Update progress callback
                    if progress_callback:
                        total_done = sum(worker_progress.values())
                        elapsed = time.time() - start_time
                        rate = total_done / elapsed if elapsed > 0 else 0
                        eta = (n_traces - total_done) / rate if rate > 0 else 0

                        progress_callback(ExportProgress(
                            phase='exporting',
                            current_traces=total_done,
                            total_traces=n_traces,
                            active_workers=len(futures) - len(processed_futures),
                            worker_progress=worker_progress.copy(),
                            elapsed_time=elapsed,
                            eta_seconds=eta
                        ))

                    # Small sleep to avoid busy-waiting
                    time.sleep(0.1)

            # Final drain of progress queue
            while True:
                try:
                    segment_id, traces_done = progress_queue.get_nowait()
                    worker_progress[segment_id] = traces_done
                except:
                    break

        finally:
            # CRITICAL: Always cleanup manager to prevent orphaned processes
            try:
                manager.shutdown()
            except Exception:
                pass

        return results

    def cancel(self):
        """Request cancellation of export."""
        self._cancel_requested = True
