"""
Coordinator for parallel multiprocess SEGY import.

Orchestrates the full import pipeline:
1. Pre-create shared Zarr array
2. Partition file into segments
3. Launch parallel workers (write directly to shared Zarr)
4. Merge header parquet files
"""

import gc
import os
import time
import segyio
import zarr
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .partitioner import SmartPartitioner, PartitionConfig, Segment, get_ensemble_byte_location
from .worker import WorkerTask, WorkerResult, import_segment


@dataclass
class ImportConfig:
    """Configuration for parallel import."""
    segy_path: str
    output_dir: str
    header_mapping: Any  # HeaderMapping object
    ensemble_key: Optional[str] = 'cdp'
    n_workers: Optional[int] = None  # Auto-detect if None
    chunk_size: int = 10000


@dataclass
class ImportProgress:
    """Progress information for callbacks."""
    phase: str                # 'partitioning', 'importing', 'merging'
    current_traces: int
    total_traces: int
    active_workers: int
    worker_progress: Dict[int, int] = field(default_factory=dict)
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class ImportResult:
    """Final result of import operation."""
    success: bool
    output_dir: str
    traces_path: str
    headers_path: str
    n_traces: int
    n_segments: int
    elapsed_time: float
    error: Optional[str] = None


@dataclass
class ImportStageResult:
    """
    Intermediate result from parallel import stage.

    Holds all data needed for subsequent merge, index, and cleanup stages.
    """
    worker_results: List['WorkerResult']
    n_traces: int
    n_samples: int
    output_dir: Path
    traces_path: str
    file_info: Dict[str, Any]
    start_time: float
    success: bool = True
    error: Optional[str] = None


def get_optimal_workers() -> int:
    """Get optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    # Leave 2 cores free for system/UI, minimum 2 workers
    return max(2, cpu_count - 2)


class ParallelImportCoordinator:
    """
    Orchestrates parallel multiprocess SEGY import.

    Workers write directly to a pre-created shared Zarr array,
    eliminating the need for trace data merging.

    Usage (single dialog - backwards compatible):
        coordinator = ParallelImportCoordinator(config)
        result = coordinator.run(progress_callback=update_ui)

    Usage (separate stage dialogs - recommended):
        coordinator = ParallelImportCoordinator(config)

        # Stage 1: Parallel import with traces progress
        stage_result = coordinator.run_parallel_import(progress_callback=on_import_progress)

        # Stage 2: Merge headers with segments progress
        headers_path = coordinator.run_merge_headers(stage_result, progress_callback=on_merge_progress)

        # Stage 3: Build index with traces progress
        coordinator.run_build_index(stage_result, headers_path, progress_callback=on_index_progress)

        # Stage 4: Cleanup with files progress
        coordinator.run_cleanup(stage_result, progress_callback=on_cleanup_progress)

        # Get final result
        result = coordinator.get_final_result(stage_result, headers_path)
    """

    def __init__(self, config: ImportConfig):
        """
        Initialize coordinator.

        Args:
            config: Import configuration
        """
        self.config = config
        self.n_workers = config.n_workers or get_optimal_workers()
        self._cancel_requested = False
        self._was_cancelled = False
        self._manager = None
        self._progress_queue = None

    def run(self, progress_callback: Optional[Callable[[ImportProgress], None]] = None) -> ImportResult:
        """
        Run the full parallel import pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            ImportResult with outcome
        """
        start_time = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Get file info
            file_info = self._get_file_info()
            n_traces = file_info['n_traces']
            n_samples = file_info['n_samples']

            if progress_callback:
                progress_callback(ImportProgress(
                    phase='partitioning',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=0
                ))

            # Phase 2: Partition into segments
            segments = self._partition(n_traces)
            print(f"  Partitioned into {len(segments)} segments")

            # Phase 3: Pre-create shared Zarr array
            traces_path = self._create_shared_zarr(output_dir, n_samples, n_traces)
            print(f"  Created shared Zarr array: {n_samples} x {n_traces:,}")

            # Phase 4: Run parallel workers
            if progress_callback:
                progress_callback(ImportProgress(
                    phase='importing',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments, n_samples, n_traces, progress_callback
            )

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            # Phase 5: Merge header files only (traces already in shared Zarr)
            if progress_callback:
                progress_callback(ImportProgress(
                    phase='merging',
                    current_traces=n_traces,
                    total_traces=n_traces,
                    active_workers=0
                ))

            headers_path = self._merge_headers(output_dir, worker_results)
            self._create_indices(output_dir, n_traces)
            self._build_ensemble_index(output_dir, headers_path)
            self._cleanup_segment_files(output_dir, worker_results)

            # Save metadata
            self._save_metadata(output_dir, file_info, len(segments))

            elapsed = time.time() - start_time

            return ImportResult(
                success=True,
                output_dir=str(output_dir),
                traces_path=traces_path,
                headers_path=headers_path,
                n_traces=n_traces,
                n_segments=len(segments),
                elapsed_time=elapsed
            )

        except Exception as e:
            import traceback
            elapsed = time.time() - start_time
            return ImportResult(
                success=False,
                output_dir=str(output_dir),
                traces_path="",
                headers_path="",
                n_traces=0,
                n_segments=0,
                elapsed_time=elapsed,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE-BASED API - For separate progress dialogs per stage
    # ═══════════════════════════════════════════════════════════════════════════

    def run_parallel_import(
        self,
        progress_callback: Optional[Callable[[ImportProgress], None]] = None
    ) -> ImportStageResult:
        """
        Stage 1: Run parallel import workers.

        This stage handles file analysis, partitioning, Zarr creation,
        and parallel worker execution. Returns intermediate result for subsequent stages.

        Args:
            progress_callback: Callback for ImportProgress updates (traces-based)

        Returns:
            ImportStageResult with worker results and metadata for merge/index/cleanup stages
        """
        start_time = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Get file info
            file_info = self._get_file_info()
            n_traces = file_info['n_traces']
            n_samples = file_info['n_samples']

            if progress_callback:
                progress_callback(ImportProgress(
                    phase='partitioning',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=0
                ))

            # Phase 2: Partition into segments
            segments = self._partition(n_traces)
            print(f"  Partitioned into {len(segments)} segments")

            # Phase 3: Pre-create shared Zarr array
            traces_path = self._create_shared_zarr(output_dir, n_samples, n_traces)
            print(f"  Created shared Zarr array: {n_samples} x {n_traces:,}")

            # Phase 4: Run parallel workers
            if progress_callback:
                progress_callback(ImportProgress(
                    phase='importing',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments, n_samples, n_traces, progress_callback
            )

            # Check for cancellation
            if self._cancel_requested:
                self._was_cancelled = True
                return ImportStageResult(
                    worker_results=worker_results,
                    n_traces=n_traces,
                    n_samples=n_samples,
                    output_dir=output_dir,
                    traces_path=traces_path,
                    file_info=file_info,
                    start_time=start_time,
                    success=False,
                    error="Import cancelled by user"
                )

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            return ImportStageResult(
                worker_results=worker_results,
                n_traces=n_traces,
                n_samples=n_samples,
                output_dir=output_dir,
                traces_path=traces_path,
                file_info=file_info,
                start_time=start_time,
                success=True
            )

        except Exception as e:
            import traceback
            return ImportStageResult(
                worker_results=[],
                n_traces=0,
                n_samples=0,
                output_dir=output_dir,
                traces_path="",
                file_info={},
                start_time=start_time,
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    def run_merge_headers(
        self,
        stage_result: ImportStageResult,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Stage 2: Merge segment header parquet files.

        Args:
            stage_result: Result from run_parallel_import()
            progress_callback: Callback(segments_merged, total_segments) for progress

        Returns:
            Path to merged headers.parquet
        """
        if not stage_result.success:
            raise RuntimeError(f"Cannot merge: previous stage failed - {stage_result.error}")

        print(f"    Merging headers from {len(stage_result.worker_results)} segments...")

        final_path = stage_result.output_dir / "headers.parquet"
        sorted_results = sorted(stage_result.worker_results, key=lambda r: r.segment_id)
        total_segments = len(sorted_results)

        all_dfs = []
        for i, result in enumerate(sorted_results):
            if result.headers_path and Path(result.headers_path).exists():
                df = pd.read_parquet(result.headers_path)
                all_dfs.append(df)
                print(f"      Segment {result.segment_id}: {len(df):,} headers")

            if progress_callback:
                progress_callback(i + 1, total_segments)

        # Concatenate all DataFrames
        final_df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs

        # Sort by trace_index
        final_df = final_df.sort_values('trace_index').reset_index(drop=True)

        # Write final Parquet
        final_df.to_parquet(
            final_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        n_headers = len(final_df)
        print(f"      Total: {n_headers:,} headers")

        del final_df
        gc.collect()

        # Create trace indices (quick, no separate dialog needed)
        self._create_indices(stage_result.output_dir, stage_result.n_traces)

        return str(final_path)

    def run_build_index(
        self,
        stage_result: ImportStageResult,
        headers_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stage 3: Build ensemble index from headers.

        Args:
            stage_result: Result from run_parallel_import()
            headers_path: Path to merged headers from run_merge_headers()
            progress_callback: Callback(traces_scanned, total_traces) for progress
        """
        if not stage_result.success:
            raise RuntimeError(f"Cannot build index: previous stage failed - {stage_result.error}")

        output_dir = stage_result.output_dir
        n_traces = stage_result.n_traces

        # Only load the columns we need
        if self.config.ensemble_key:
            try:
                headers_df = pd.read_parquet(headers_path, columns=[self.config.ensemble_key])
                has_key = self.config.ensemble_key in headers_df.columns
            except Exception:
                headers_df = pd.read_parquet(headers_path)
                has_key = self.config.ensemble_key in headers_df.columns
        else:
            headers_df = pd.read_parquet(headers_path, columns=[])
            has_key = False

        # If no ensemble key, create single-ensemble index
        if not self.config.ensemble_key or not has_key:
            if not self.config.ensemble_key:
                print(f"    Building default ensemble index (single ensemble for all {n_traces:,} traces)...")
            else:
                print(f"      Warning: Ensemble key '{self.config.ensemble_key}' not found")
                print(f"      Creating default single-ensemble index instead...")

            del headers_df
            gc.collect()

            ensembles = [{
                'ensemble_id': 0,
                'ensemble_value': 0,
                'start_trace': 0,
                'end_trace': n_traces - 1,
                'n_traces': n_traces
            }]

            # Report completion
            if progress_callback:
                progress_callback(n_traces, n_traces)
        else:
            print(f"    Building ensemble index on '{self.config.ensemble_key}'...")

            ensemble_col = headers_df[self.config.ensemble_key].values
            del headers_df
            gc.collect()

            # Report every 1% or 10K traces
            report_interval = min(10000, max(1, n_traces // 100))

            ensembles = []
            current_value = ensemble_col[0]
            start_trace = 0
            ensemble_id = 0

            for i in range(1, len(ensemble_col)):
                if ensemble_col[i] != current_value:
                    ensembles.append({
                        'ensemble_id': ensemble_id,
                        'ensemble_value': int(current_value),
                        'start_trace': start_trace,
                        'end_trace': i - 1,
                        'n_traces': i - start_trace
                    })
                    ensemble_id += 1
                    current_value = ensemble_col[i]
                    start_trace = i

                # Progress callback
                if i % report_interval == 0 and progress_callback:
                    progress_callback(i, n_traces)

            # Last ensemble
            ensembles.append({
                'ensemble_id': ensemble_id,
                'ensemble_value': int(current_value),
                'start_trace': start_trace,
                'end_trace': len(ensemble_col) - 1,
                'n_traces': len(ensemble_col) - start_trace
            })

            # Final progress
            if progress_callback:
                progress_callback(n_traces, n_traces)

            del ensemble_col
            gc.collect()

        # Save to parquet
        ensemble_df = pd.DataFrame(ensembles)
        ensemble_path = output_dir / 'ensemble_index.parquet'
        ensemble_df.to_parquet(
            ensemble_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        print(f"      Created {len(ensembles):,} ensembles")

        del ensemble_df
        del ensembles
        gc.collect()

    def run_cleanup(
        self,
        stage_result: ImportStageResult,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stage 4: Cleanup temporary segment header files.

        Args:
            stage_result: Result from run_parallel_import()
            progress_callback: Callback(files_cleaned, total_files) for progress
        """
        print(f"    Cleaning up segment files...")

        total_files = len(stage_result.worker_results)
        cleaned = 0

        for result in stage_result.worker_results:
            if result.headers_path:
                header_path = Path(result.headers_path)
                try:
                    if header_path.exists():
                        header_path.unlink()
                except Exception as e:
                    print(f"      Warning: Could not remove {header_path}: {e}")

            cleaned += 1
            if progress_callback:
                progress_callback(cleaned, total_files)

    def get_final_result(
        self,
        stage_result: ImportStageResult,
        headers_path: str
    ) -> ImportResult:
        """
        Build final ImportResult after all stages complete.

        Args:
            stage_result: Result from run_parallel_import()
            headers_path: Path to headers from run_merge_headers()

        Returns:
            Final ImportResult with success status and statistics
        """
        elapsed = time.time() - stage_result.start_time

        if not stage_result.success:
            return ImportResult(
                success=False,
                output_dir=str(stage_result.output_dir),
                traces_path="",
                headers_path="",
                n_traces=0,
                n_segments=0,
                elapsed_time=elapsed,
                error=stage_result.error
            )

        # Save metadata
        self._save_metadata(
            stage_result.output_dir,
            stage_result.file_info,
            len(stage_result.worker_results)
        )

        return ImportResult(
            success=True,
            output_dir=str(stage_result.output_dir),
            traces_path=stage_result.traces_path,
            headers_path=headers_path,
            n_traces=stage_result.n_traces,
            n_segments=len(stage_result.worker_results),
            elapsed_time=elapsed
        )

    @property
    def was_cancelled(self) -> bool:
        """Check if import was cancelled."""
        return self._was_cancelled

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_file_info(self) -> Dict[str, Any]:
        """Get basic file information."""
        with segyio.open(self.config.segy_path, 'r', ignore_geometry=True) as f:
            return {
                'n_traces': f.tracecount,
                'n_samples': len(f.samples),
                'sample_interval': segyio.tools.dt(f) / 1000.0
            }

    def _partition(self, n_traces: int) -> List[Segment]:
        """Partition file into segments."""
        ensemble_byte = 21  # Default CDP
        if self.config.ensemble_key:
            ensemble_byte = get_ensemble_byte_location(self.config.ensemble_key)

        partition_config = PartitionConfig(
            segy_path=self.config.segy_path,
            n_segments=self.n_workers,
            total_traces=n_traces,
            ensemble_key=self.config.ensemble_key,
            ensemble_byte=ensemble_byte
        )

        partitioner = SmartPartitioner(partition_config)
        segments = partitioner.partition()

        # Log partition stats
        stats = partitioner.get_partition_stats(segments)
        print(f"  Partition stats: {stats}")

        return segments

    def _create_shared_zarr(self, output_dir: Path, n_samples: int, n_traces: int) -> str:
        """
        Pre-create shared Zarr array for all workers to write to.

        Args:
            output_dir: Output directory
            n_samples: Samples per trace
            n_traces: Total number of traces

        Returns:
            Path to traces.zarr
        """
        traces_path = output_dir / "traces.zarr"

        # Create Zarr array (no compression for speed)
        zarr.open(
            str(traces_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 1000),
            dtype=np.float32,
            compressor=None,
            zarr_format=2
        )

        return str(traces_path)

    def _run_workers(
        self,
        segments: List[Segment],
        n_samples: int,
        n_traces: int,
        progress_callback: Optional[Callable]
    ) -> List[WorkerResult]:
        """
        Run worker processes in parallel.

        Args:
            segments: List of segments to process
            n_samples: Samples per trace
            n_traces: Total traces (for progress)
            progress_callback: Progress callback

        Returns:
            List of worker results
        """
        # Serialize header mapping for workers
        header_mapping_dict = self.config.header_mapping.get_all_mappings()

        # Create worker tasks
        tasks = []
        for segment in segments:
            task = WorkerTask(
                segment_id=segment.segment_id,
                segy_path=self.config.segy_path,
                output_dir=str(self.config.output_dir),
                start_trace=segment.start_trace,
                end_trace=segment.end_trace,
                n_samples=n_samples,
                header_mapping_dict=header_mapping_dict,
                chunk_size=self.config.chunk_size
            )
            tasks.append(task)

        # Use multiprocessing Manager for progress queue with bounded size
        # to prevent memory issues from queue overflow
        manager = mp.Manager()
        progress_queue = manager.Queue(maxsize=1000)

        # Track progress per worker
        worker_progress = {s.segment_id: 0 for s in segments}
        results = []
        processed_futures = set()  # Track which futures we've already processed

        print(f"  Launching {len(tasks)} worker processes...")

        try:
            # Use SPAWN context instead of fork to avoid copying parent memory
            # Fork copies entire parent memory space which causes OOM with large datasets
            spawn_ctx = mp.get_context('spawn')
            # Submit all tasks
            with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=spawn_ctx) as executor:
                # Submit tasks
                futures = {
                    executor.submit(import_segment, task, progress_queue): task
                    for task in tasks
                }

                # Monitor progress while waiting
                start_time = time.time()

                while len(processed_futures) < len(futures):
                    # Check for completed futures
                    for future in futures:
                        if future.done() and future not in processed_futures:
                            processed_futures.add(future)
                            try:
                                result = future.result(timeout=0.1)
                                results.append(result)
                                worker_progress[result.segment_id] = result.n_traces
                                print(f"    Worker {result.segment_id} completed: "
                                      f"{result.n_traces:,} traces in {result.elapsed_time:.1f}s")
                            except Exception as e:
                                task = futures[future]
                                results.append(WorkerResult(
                                    segment_id=task.segment_id,
                                    headers_path="",
                                    n_traces=0,
                                    start_trace=task.start_trace,
                                    elapsed_time=0,
                                    success=False,
                                    error=str(e)
                                ))

                    # Drain progress queue aggressively to prevent blocking workers
                    drain_count = 0
                    max_drain = 500  # Limit per iteration to stay responsive
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

                        progress_callback(ImportProgress(
                            phase='importing',
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
            except Exception as e:
                print(f"    Warning: Manager shutdown encountered issue: {e}")

        return results

    def _merge_headers(self, output_dir: Path, results: List[WorkerResult]) -> str:
        """
        Merge segment header parquet files into single file.

        Headers already have correct global trace indices.

        Args:
            output_dir: Output directory
            results: Worker results with header file paths

        Returns:
            Path to merged headers.parquet
        """
        print(f"    Merging headers from {len(results)} segments...")

        final_path = output_dir / "headers.parquet"

        # Sort by segment_id to ensure correct order
        sorted_results = sorted(results, key=lambda r: r.segment_id)

        all_dfs = []
        for result in sorted_results:
            if result.headers_path and Path(result.headers_path).exists():
                df = pd.read_parquet(result.headers_path)
                all_dfs.append(df)
                print(f"      Segment {result.segment_id}: {len(df):,} headers")

        # Concatenate all DataFrames
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Release segment DataFrames immediately
        del all_dfs

        # Sort by trace_index to ensure correct order
        final_df = final_df.sort_values('trace_index').reset_index(drop=True)

        # Write final Parquet
        final_df.to_parquet(
            final_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        n_headers = len(final_df)
        print(f"      Total: {n_headers:,} headers")

        # Release memory before returning
        del final_df
        gc.collect()

        return str(final_path)

    def _create_indices(self, output_dir: Path, n_traces: int):
        """Create trace index file."""
        df_index = pd.DataFrame({
            'trace_index': np.arange(n_traces),
            'global_trace_id': np.arange(n_traces)
        })

        index_path = output_dir / "trace_index.parquet"
        df_index.to_parquet(
            index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Release memory
        del df_index
        gc.collect()

    def _build_ensemble_index(self, output_dir: Path, headers_path: str):
        """
        Build ensemble index from headers based on ensemble key.

        This creates the ensemble_index.parquet file needed for gather navigation
        and parallel processing. Always creates an index even if no key is specified
        (treats entire dataset as single ensemble).
        """
        # Only load the columns we need to minimize memory usage
        if self.config.ensemble_key:
            # Try to load only the ensemble key column
            try:
                headers_df = pd.read_parquet(headers_path, columns=[self.config.ensemble_key])
                n_traces = len(headers_df)
                has_key = self.config.ensemble_key in headers_df.columns
            except Exception:
                # Fall back to loading full file if column doesn't exist
                headers_df = pd.read_parquet(headers_path)
                n_traces = len(headers_df)
                has_key = self.config.ensemble_key in headers_df.columns
        else:
            # Just need count, load minimal data
            headers_df = pd.read_parquet(headers_path, columns=[])
            n_traces = len(headers_df)
            has_key = False

        # If no ensemble key, create single-ensemble index covering all traces
        if not self.config.ensemble_key:
            print(f"    Building default ensemble index (single ensemble for all {n_traces:,} traces)...")
            # Release headers_df early since we don't need it
            del headers_df
            gc.collect()

            ensembles = [{
                'ensemble_id': 0,
                'ensemble_value': 0,
                'start_trace': 0,
                'end_trace': n_traces - 1,
                'n_traces': n_traces
            }]
        else:
            print(f"    Building ensemble index on '{self.config.ensemble_key}'...")

            # Check if ensemble key column exists
            if not has_key:
                print(f"      Warning: Ensemble key '{self.config.ensemble_key}' not found in headers")
                print(f"      Creating default single-ensemble index instead...")
                # Release headers_df early
                del headers_df
                gc.collect()

                ensembles = [{
                    'ensemble_id': 0,
                    'ensemble_value': 0,
                    'start_trace': 0,
                    'end_trace': n_traces - 1,
                    'n_traces': n_traces
                }]
            else:
                # Build ensemble index from key
                ensembles = []
                ensemble_col = headers_df[self.config.ensemble_key].values

                # Release headers_df - we only need ensemble_col now
                del headers_df
                gc.collect()

                # Find ensemble boundaries
                current_value = ensemble_col[0]
                start_trace = 0
                ensemble_id = 0

                for i in range(1, len(ensemble_col)):
                    if ensemble_col[i] != current_value:
                        # End of ensemble
                        ensembles.append({
                            'ensemble_id': ensemble_id,
                            'ensemble_value': int(current_value),
                            'start_trace': start_trace,
                            'end_trace': i - 1,
                            'n_traces': i - start_trace
                        })
                        ensemble_id += 1
                        current_value = ensemble_col[i]
                        start_trace = i

                # Last ensemble
                ensembles.append({
                    'ensemble_id': ensemble_id,
                    'ensemble_value': int(current_value),
                    'start_trace': start_trace,
                    'end_trace': len(ensemble_col) - 1,
                    'n_traces': len(ensemble_col) - start_trace
                })

                # Release ensemble_col
                del ensemble_col
                gc.collect()

        # Save to parquet
        ensemble_df = pd.DataFrame(ensembles)
        ensemble_path = output_dir / 'ensemble_index.parquet'
        ensemble_df.to_parquet(
            ensemble_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        n_ensembles = len(ensembles)
        print(f"      Created {n_ensembles:,} ensembles")

        # Release memory
        del ensemble_df
        del ensembles
        gc.collect()

    def _cleanup_segment_files(self, output_dir: Path, results: List[WorkerResult]):
        """Remove temporary segment header files."""
        print(f"    Cleaning up segment files...")

        for result in results:
            if result.headers_path:
                header_path = Path(result.headers_path)
                try:
                    if header_path.exists():
                        header_path.unlink()
                except Exception as e:
                    print(f"      Warning: Could not remove {header_path}: {e}")

    def _save_metadata(self, output_dir: Path, file_info: dict, n_segments: int):
        """Save import metadata."""
        import json

        metadata = {
            'shape': [file_info['n_samples'], file_info['n_traces']],
            'sample_rate': file_info['sample_interval'],
            'n_samples': file_info['n_samples'],
            'n_traces': file_info['n_traces'],
            'duration_ms': (file_info['n_samples'] - 1) * file_info['sample_interval'],
            'nyquist_freq': 1000.0 / (2.0 * file_info['sample_interval']),
            'import_info': {
                'method': 'parallel_multiprocess',
                'n_workers': self.n_workers,
                'n_segments': n_segments
            },
            'storage_info': {
                'zarr_compression': 'none',
                'parquet_compression': 'snappy'
            }
        }

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def cancel(self):
        """Request cancellation of import."""
        self._cancel_requested = True
