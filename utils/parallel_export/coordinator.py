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
from typing import Optional, Callable, List
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

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


def get_optimal_workers() -> int:
    """Get optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    # Leave 2 cores free for system/UI, minimum 2 workers
    return max(2, cpu_count - 2)


class ParallelExportCoordinator:
    """
    Orchestrates parallel SEG-Y export.

    Workers each write to separate segment files, which are then
    merged into the final output. This approach:
    - Bypasses Python GIL via multiprocessing
    - Uses vectorized headers for O(1) header access
    - Enables parallel I/O across multiple files

    Usage:
        config = ExportConfig(
            original_segy_path='/path/to/original.sgy',
            processed_zarr_path='/path/to/processed/traces.zarr',
            headers_parquet_path='/path/to/headers.parquet',
            output_path='/path/to/output.sgy',
            temp_dir='/path/to/temp'
        )
        coordinator = ParallelExportCoordinator(config)
        result = coordinator.run(progress_callback=update_ui)
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

            # Save vectorized headers for workers
            temp_dir.mkdir(parents=True, exist_ok=True)
            header_arrays_path = temp_dir / 'header_arrays.pkl'
            vectorizer.save(header_arrays_path)

            stats = vectorizer.get_stats()
            print(f"    {stats['n_fields']} fields, {stats['memory_mb']:.1f} MB")

            # Phase 3: Partition traces
            segments = self._partition_traces(n_traces)
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
                header_arrays_path=str(header_arrays_path),
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

            # Remove header arrays file
            try:
                os.remove(header_arrays_path)
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
        header_arrays_path: str,
        n_samples: int,
        sample_interval: int,
        data_format: int,
        n_traces: int,
        progress_callback: Optional[Callable]
    ) -> List[ExportWorkerResult]:
        """Run worker processes in parallel."""
        temp_dir = Path(self.config.temp_dir)

        # Create tasks
        tasks = []
        for segment in segments:
            segment_path = str(temp_dir / f'segment_{segment.segment_id}.sgy')
            task = ExportTask(
                segment_id=segment.segment_id,
                original_segy_path=self.config.original_segy_path,
                processed_zarr_path=self.config.processed_zarr_path,
                output_segment_path=segment_path,
                header_arrays_path=header_arrays_path,
                start_trace=segment.start_trace,
                end_trace=segment.end_trace,
                n_samples=n_samples,
                sample_interval=sample_interval,
                data_format=data_format,
                is_first_segment=(segment.segment_id == 0)
            )
            tasks.append(task)

        # Use multiprocessing Manager for progress queue
        manager = mp.Manager()
        progress_queue = manager.Queue()

        # Track progress per worker
        worker_progress = {s.segment_id: 0 for s in segments}
        results = []
        processed_futures = set()

        print(f"  Launching {len(tasks)} export workers...")

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
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

                # Drain progress queue
                while not progress_queue.empty():
                    try:
                        segment_id, traces_done = progress_queue.get_nowait()
                        worker_progress[segment_id] = traces_done
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

        return results

    def cancel(self):
        """Request cancellation of export."""
        self._cancel_requested = True
