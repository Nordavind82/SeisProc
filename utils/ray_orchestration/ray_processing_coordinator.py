"""
Ray Processing Coordinator

Orchestrates Ray actors for distributed seismic gather processing.
Replaces ProcessPoolExecutor-based ParallelProcessingCoordinator with
proper per-worker GPU initialization.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from uuid import UUID

import numpy as np
import pandas as pd

from .cluster import initialize_ray, is_ray_initialized, get_cluster_resources
from .cancellation import CancellationToken, CancellationError

logger = logging.getLogger(__name__)

# Lazy import Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class RayProcessingCoordinator:
    """
    Ray-based coordinator for parallel seismic processing.

    Replaces ParallelProcessingCoordinator (ProcessPoolExecutor) with Ray actors.
    Key advantages:
    - Per-worker GPU initialization (Metal works correctly)
    - Better fault tolerance
    - Proper cancellation propagation
    - Scalable to distributed clusters

    Usage
    -----
    >>> from utils.parallel_processing import ProcessingConfig
    >>> config = ProcessingConfig(...)
    >>> coordinator = RayProcessingCoordinator(config, job_id, token)
    >>> result = coordinator.run(progress_callback)
    """

    def __init__(
        self,
        config: Any,
        job_id: Optional[UUID] = None,
        cancellation_token: Optional[CancellationToken] = None,
        qt_bridge: Optional[Any] = None,
    ):
        """
        Initialize Ray processing coordinator.

        Parameters
        ----------
        config : ProcessingConfig
            Processing configuration
        job_id : UUID, optional
            Job identifier for tracking
        cancellation_token : CancellationToken, optional
            Token for cancellation signaling
        qt_bridge : JobManagerBridge, optional
            Qt bridge for UI updates
        """
        self._config = config
        self._job_id = job_id
        self._token = cancellation_token
        self._qt_bridge = qt_bridge

        # Runtime state
        self._actors = []
        self._actor_refs = []
        self._start_time = None
        self._cancel_requested = False

        # Metadata loaded during run
        self._n_samples = 0
        self._n_traces = 0
        self._n_gathers = 0
        self._sample_rate = 1.0
        self._metadata = {}

        # Track compute kernel for progress reporting
        self._compute_kernel = "detecting..."
        self._n_workers = 0

    def run(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Any:
        """
        Run parallel processing with Ray actors.

        Parameters
        ----------
        progress_callback : callable, optional
            Progress callback (receives ProcessingProgress)

        Returns
        -------
        ProcessingResult
            Result of processing operation
        """
        from utils.parallel_processing import (
            ProcessingProgress,
            ProcessingResult,
            GatherPartitioner,
        )

        self._start_time = time.time()

        # Phase 1: Load and validate metadata
        logger.info("Phase 1: Loading metadata...")
        self._report_progress(progress_callback, 'initializing', 0, 1, 0)

        if not self._load_and_validate():
            return ProcessingResult(
                success=False,
                output_dir=str(self._config.output_storage_dir),
                output_zarr_path="",
                n_gathers=0,
                n_traces=0,
                n_samples=0,
                elapsed_time=0,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error="Failed to load input data",
            )

        # Check cancellation
        if self._check_cancelled():
            return self._cancelled_result()

        # Phase 2: Partition gathers
        logger.info("Phase 2: Partitioning gathers...")
        n_workers = self._config.n_workers or self._get_optimal_workers()

        ensemble_df = self._load_ensemble_index()
        partitioner = GatherPartitioner(ensemble_df, n_workers)
        segments = partitioner.partition()

        if not segments:
            return ProcessingResult(
                success=False,
                output_dir=str(self._config.output_storage_dir),
                output_zarr_path="",
                n_gathers=0,
                n_traces=0,
                n_samples=0,
                elapsed_time=0,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error="No gathers to process",
            )

        stats = partitioner.get_partition_stats(segments)
        logger.info(f"Partitioned into {len(segments)} segments: {stats}")

        # Phase 3: Create output arrays
        logger.info("Phase 3: Creating output arrays...")
        output_zarr_path, noise_zarr_path = self._create_output_arrays()

        if self._check_cancelled():
            return self._cancelled_result()

        # Phase 4: Initialize Ray and run workers
        logger.info("Phase 4: Running Ray actors...")

        if not is_ray_initialized():
            if not initialize_ray():
                return ProcessingResult(
                    success=False,
                    output_dir=str(self._config.output_storage_dir),
                    output_zarr_path="",
                    n_gathers=0,
                    n_traces=0,
                    n_samples=0,
                    elapsed_time=0,
                    throughput_traces_per_sec=0,
                    n_workers_used=0,
                    error="Failed to initialize Ray cluster",
                )

        # Put shared data in Ray object store
        ensemble_ref = ray.put(ensemble_df)
        headers_ref = self._put_headers_if_needed()

        try:
            # Run distributed processing
            worker_results = self._run_actors(
                segments,
                ensemble_ref,
                headers_ref,
                output_zarr_path,
                noise_zarr_path,
                progress_callback,
            )

            # Aggregate results
            total_gathers = sum(r.n_gathers_processed for r in worker_results)
            total_traces = sum(r.n_traces_processed for r in worker_results)

            elapsed = time.time() - self._start_time
            throughput = total_traces / elapsed if elapsed > 0 else 0

            # Check for any failures
            errors = [r.error for r in worker_results if r.error]

            logger.info(
                f"Processing complete: {total_gathers} gathers, "
                f"{total_traces} traces in {elapsed:.1f}s "
                f"({throughput:.0f} traces/sec)"
            )

            # Finalize output: copy metadata files
            if len(errors) == 0:
                try:
                    self._finalize_output(total_traces, total_gathers, throughput)
                    logger.info("Output finalization complete")
                except Exception as e:
                    logger.error(f"Output finalization failed: {e}")
                    errors.append(f"Finalization failed: {e}")

            return ProcessingResult(
                success=len(errors) == 0,
                output_dir=str(self._config.output_storage_dir),
                output_zarr_path=output_zarr_path or "",
                n_gathers=total_gathers,
                n_traces=total_traces,
                n_samples=self._n_samples,
                elapsed_time=elapsed,
                throughput_traces_per_sec=throughput,
                n_workers_used=len(segments),
                error=errors[0] if errors else None,
                noise_zarr_path=noise_zarr_path,
            )

        except CancellationError:
            logger.info("Processing cancelled by user")
            self._cleanup_actors()
            return self._cancelled_result()

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self._cleanup_actors()
            return ProcessingResult(
                success=False,
                output_dir=str(self._config.output_storage_dir),
                output_zarr_path=output_zarr_path or "",
                n_gathers=0,
                n_traces=0,
                n_samples=self._n_samples,
                elapsed_time=time.time() - self._start_time,
                throughput_traces_per_sec=0,
                n_workers_used=len(segments),
                error=str(e),
            )

    def _load_and_validate(self) -> bool:
        """Load and validate input data."""
        input_dir = Path(self._config.input_storage_dir)

        # Check required files
        traces_path = input_dir / "traces.zarr"
        metadata_path = input_dir / "metadata.json"
        ensemble_path = input_dir / "ensemble_index.parquet"

        for path in [traces_path, metadata_path, ensemble_path]:
            if not path.exists():
                logger.error(f"Missing required file: {path}")
                return False

        # Load metadata
        with open(metadata_path) as f:
            self._metadata = json.load(f)

        self._n_samples = self._metadata.get('n_samples', 0)
        self._n_traces = self._metadata.get('n_traces', 0)
        self._sample_rate = self._metadata.get('sample_rate_ms', 1.0)

        # Count gathers
        ensemble_df = pd.read_parquet(ensemble_path)
        self._n_gathers = len(ensemble_df)

        logger.info(
            f"Input validated: {self._n_gathers} gathers, "
            f"{self._n_traces} traces, {self._n_samples} samples"
        )

        return True

    def _load_ensemble_index(self) -> pd.DataFrame:
        """Load ensemble index DataFrame."""
        path = Path(self._config.input_storage_dir) / "ensemble_index.parquet"
        return pd.read_parquet(path)

    def _get_optimal_workers(self) -> int:
        """Determine optimal number of workers."""
        try:
            resources = get_cluster_resources()
            # Use available CPUs minus 2 for OS
            n_workers = max(1, int(resources.num_cpus) - 2)
            # Don't exceed number of gathers
            n_workers = min(n_workers, self._n_gathers)
            return n_workers
        except Exception:
            return max(1, min(4, self._n_gathers))

    def _create_output_arrays(self) -> Tuple[Optional[str], Optional[str]]:
        """Create output Zarr arrays."""
        import zarr

        output_dir = Path(self._config.output_storage_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_zarr_path = None
        noise_zarr_path = None

        output_mode = self._config.output_mode
        output_noise = self._config.output_noise

        # Determine what outputs to create
        create_processed = output_mode in ('processed', 'both')
        create_noise = output_mode in ('noise', 'both') or output_noise

        if create_processed:
            output_zarr_path = str(output_dir / "traces.zarr")
            zarr.open(
                output_zarr_path,
                mode='w',
                shape=(self._n_samples, self._n_traces),
                chunks=(self._n_samples, 1000),
                dtype='float32',
            )
            logger.info(f"Created output zarr: {output_zarr_path}")

        if create_noise:
            noise_zarr_path = str(output_dir / "noise.zarr")
            zarr.open(
                noise_zarr_path,
                mode='w',
                shape=(self._n_samples, self._n_traces),
                chunks=(self._n_samples, 1000),
                dtype='float32',
            )
            logger.info(f"Created noise zarr: {noise_zarr_path}")

            # For noise-only mode, symlink traces.zarr -> noise.zarr
            if output_mode == 'noise' and not create_processed:
                traces_link = output_dir / "traces.zarr"
                if not traces_link.exists():
                    import os
                    os.symlink("noise.zarr", str(traces_link))
                    output_zarr_path = str(traces_link)

        return output_zarr_path, noise_zarr_path

    def _put_headers_if_needed(self) -> Optional[Any]:
        """Load headers and put in Ray object store if needed."""
        needs_headers = (
            self._config.mute_velocity > 0 or
            (self._config.sort_options and self._config.sort_options.enabled) or
            self._is_fkk_processor()
        )

        if not needs_headers:
            return None

        headers_path = Path(self._config.input_storage_dir) / "headers.parquet"
        if not headers_path.exists():
            logger.warning("Headers requested but not found")
            return None

        # Determine which columns to load
        columns = ['trace_index']

        if self._config.mute_velocity > 0:
            columns.append('offset')

        if self._config.sort_options and self._config.sort_options.enabled:
            columns.append(self._config.sort_options.sort_key)
            if self._config.sort_options.secondary_key:
                columns.append(self._config.sort_options.secondary_key)

        if self._is_fkk_processor():
            # FKK needs all headers for volume building
            headers_df = pd.read_parquet(headers_path)
        else:
            headers_df = pd.read_parquet(headers_path, columns=columns)

        logger.info(f"Loaded headers: {len(headers_df)} rows, columns={list(headers_df.columns)}")

        return ray.put(headers_df)

    def _is_fkk_processor(self) -> bool:
        """Check if processor is FKK (needs full headers)."""
        class_name = self._config.processor_config.get('class_name', '')
        return 'fkk' in class_name.lower()

    def _should_use_gpu(self) -> bool:
        """Check if GPU processing should be used."""
        proc_config = self._config.processor_config

        # Check explicit flags
        if proc_config.get('use_metal', False):
            return True
        if proc_config.get('use_gpu', False):
            return True

        # Check backend setting
        backend = proc_config.get('backend', '') or ''
        if backend.lower() in ('metal', 'metal_cpp', 'auto'):
            try:
                from processors.kernel_backend import is_metal_available
                return is_metal_available()
            except ImportError:
                return False

        # Also check if Metal is available globally (auto mode)
        # This catches cases where backend isn't explicitly set
        try:
            from processors.kernel_backend import get_backend_info
            info = get_backend_info()
            if info.get('effective_backend') == 'metal_cpp':
                return True
        except ImportError:
            pass

        return False

    def _run_actors(
        self,
        segments: List[Any],
        ensemble_ref: Any,
        headers_ref: Optional[Any],
        output_zarr_path: Optional[str],
        noise_zarr_path: Optional[str],
        progress_callback: Optional[Callable],
    ) -> List[Any]:
        """
        Create and run Ray actors for processing.

        Parameters
        ----------
        segments : list
            List of GatherSegment objects
        ensemble_ref : ObjectRef
            Ray reference to ensemble DataFrame
        headers_ref : ObjectRef, optional
            Ray reference to headers DataFrame
        output_zarr_path : str, optional
            Path to output zarr
        noise_zarr_path : str, optional
            Path to noise zarr
        progress_callback : callable, optional
            Progress callback

        Returns
        -------
        list
            List of WorkerResult from each actor
        """
        from utils.ray_orchestration.workers.cpu_worker import (
            create_cpu_worker_actor,
            WorkerResult,
        )

        use_gpu = self._should_use_gpu()

        if use_gpu:
            try:
                from utils.ray_orchestration.workers.metal_worker import (
                    create_metal_worker_actor,
                )
                WorkerActorClass = create_metal_worker_actor()
                logger.info("Using Metal GPU workers (dedicated Metal actors)")
                self._compute_kernel = "Metal GPU"
            except Exception as e:
                logger.warning(f"Metal worker unavailable, using CPU workers: {e}")
                WorkerActorClass = create_cpu_worker_actor()
                use_gpu = False
                self._compute_kernel = "CPU (Python)"
        else:
            WorkerActorClass = create_cpu_worker_actor()
            # Note: CPUWorkerActor still uses Metal GPU via processor's auto backend
            try:
                from processors.kernel_backend import get_backend_info
                info = get_backend_info()
                effective = info.get('effective_backend', 'python')
                if effective == 'metal_cpp':
                    logger.info("Using CPU workers with Metal GPU acceleration")
                    self._compute_kernel = "Metal GPU (Auto)"
                else:
                    logger.info(f"Using CPU workers (backend: {effective})")
                    self._compute_kernel = f"CPU ({effective})"
            except ImportError:
                logger.info("Using CPU workers")
                self._compute_kernel = "CPU (Python)"

        # Store worker count for progress tracking
        self._n_workers = len(segments)

        input_zarr_path = str(Path(self._config.input_storage_dir) / "traces.zarr")

        # Build mute config dict
        mute_config = None
        if self._config.mute_velocity > 0:
            mute_config = {
                'velocity': self._config.mute_velocity,
                'top_mute': self._config.mute_top,
                'bottom_mute': self._config.mute_bottom,
                'taper_samples': self._config.mute_taper,
                'target': self._config.mute_target,
            }

        # Build sort config dict
        sort_config = None
        if self._config.sort_options and self._config.sort_options.enabled:
            sort_config = {
                'enabled': True,
                'sort_key': self._config.sort_options.sort_key,
                'ascending': self._config.sort_options.ascending,
                'secondary_key': self._config.sort_options.secondary_key,
                'secondary_ascending': self._config.sort_options.secondary_ascending,
            }

        # Create actors
        self._actors = []
        futures = []

        # CRITICAL: Fetch shared data ONCE before the loop to avoid memory explosion
        # Each ray.get() copies data from object store - calling inside loop
        # would multiply memory usage by number of segments
        ensemble_df = ray.get(ensemble_ref)
        headers_df = ray.get(headers_ref) if headers_ref is not None else None

        for segment in segments:
            worker_id = f"worker-{segment.segment_id}"

            # Create actor
            actor = WorkerActorClass.remote(
                job_id=self._job_id or UUID(int=0),
                worker_id=worker_id,
                input_zarr_path=input_zarr_path,
                output_zarr_path=output_zarr_path,
                noise_zarr_path=noise_zarr_path,
                processor_config=self._config.processor_config,
                sample_rate=self._sample_rate,
                metadata=self._metadata,
                output_mode=self._config.output_mode,
                mute_config=mute_config,
                sort_config=sort_config,
            )

            self._actors.append(actor)

            # Build gather task list for this segment
            gather_tasks = []
            for g_idx in range(segment.start_gather, segment.end_gather + 1):
                row = ensemble_df.iloc[g_idx]
                gather_tasks.append((
                    g_idx,
                    int(row['start_trace']),
                    int(row['end_trace']),
                ))

            # Get headers subset for this segment if available
            headers_subset = None
            if headers_df is not None:
                headers_subset = headers_df.iloc[
                    segment.start_trace:segment.end_trace + 1
                ]

            # Submit processing task
            future = actor.process.remote(gather_tasks, headers_subset)
            futures.append((segment, future))

        # Monitor progress
        return self._monitor_futures(futures, progress_callback)

    def _monitor_futures(
        self,
        futures: List[Tuple[Any, Any]],
        progress_callback: Optional[Callable],
    ) -> List[Any]:
        """
        Monitor Ray futures and collect results.

        Parameters
        ----------
        futures : list
            List of (segment, future) tuples
        progress_callback : callable, optional
            Progress callback

        Returns
        -------
        list
            List of WorkerResult
        """
        from utils.ray_orchestration.workers.cpu_worker import WorkerResult

        pending = {f: seg for seg, f in futures}
        results = []
        completed_traces = 0
        completed_gathers = 0

        # Track maximum progress seen to prevent backwards jumps in UI
        # (When workers timeout, we get partial counts - never show lower than previous max)
        max_reported_traces = 0
        max_reported_gathers = 0

        # Build mapping from segment_id to (actor, segment) for progress polling
        segment_actor_map = {}
        for i, (seg, f) in enumerate(futures):
            if i < len(self._actors):
                # Calculate average traces per gather for this segment
                segment_traces = seg.end_trace - seg.start_trace + 1
                segment_gathers = seg.end_gather - seg.start_gather + 1
                traces_per_gather = segment_traces / segment_gathers if segment_gathers > 0 else 1
                segment_actor_map[id(f)] = (self._actors[i], seg, traces_per_gather)

        # Report initial state with totals so dashboard shows targets immediately
        if progress_callback:
            logger.info(f"[PROGRESS] Initial: 0/{self._n_traces} traces, {self._n_workers} workers, kernel={self._compute_kernel}")
            self._report_progress(
                progress_callback,
                phase='processing',
                current_traces=0,
                total_traces=self._n_traces,
                active_workers=self._n_workers,
                current_gathers=0,
            )

        while pending:
            # Check cancellation
            if self._check_cancelled():
                raise CancellationError("Cancelled by user")

            # Wait for any future to complete (short timeout for responsiveness)
            ready, _ = ray.wait(list(pending.keys()), timeout=0.5, num_returns=1)

            for future in ready:
                segment = pending.pop(future)

                try:
                    result = ray.get(future)
                    results.append(result)

                    completed_traces += result.n_traces_processed
                    completed_gathers += result.n_gathers_processed

                    logger.info(
                        f"Worker {result.worker_id} completed: "
                        f"{result.n_gathers_processed} gathers, "
                        f"{result.n_traces_processed} traces"
                    )

                except Exception as e:
                    logger.error(f"Worker for segment {segment.segment_id} failed: {e}")
                    results.append(WorkerResult(
                        worker_id=f"worker-{segment.segment_id}",
                        n_gathers_processed=0,
                        n_traces_processed=0,
                        elapsed_seconds=0,
                        success=False,
                        error=str(e),
                    ))

            # Poll in-progress workers for intermediate progress
            in_progress_traces = 0
            in_progress_gathers = 0
            active_workers = 0
            polled_count = 0
            timeout_count = 0

            for future in pending.keys():
                future_id = id(future)
                if future_id in segment_actor_map:
                    actor, seg, traces_per_gather = segment_actor_map[future_id]
                    try:
                        # Non-blocking progress poll with short timeout
                        progress_ref = actor.get_progress.remote()
                        progress = ray.get(progress_ref, timeout=0.2)  # Increased timeout

                        # Calculate traces from gathers processed
                        worker_gathers = progress.items_processed
                        worker_traces = int(worker_gathers * traces_per_gather)

                        in_progress_gathers += worker_gathers
                        in_progress_traces += worker_traces
                        polled_count += 1

                        # Count as active if processing
                        from utils.ray_orchestration.workers.base_worker import WorkerState
                        if progress.state == WorkerState.PROCESSING:
                            active_workers += 1
                    except ray.exceptions.GetTimeoutError:
                        # Actor is busy, assume still processing
                        active_workers += 1
                        timeout_count += 1
                    except Exception as e:
                        # Other error, still count as active
                        active_workers += 1
                        logger.debug(f"Progress poll error: {e}")

            # Log polling summary periodically
            if polled_count > 0 or timeout_count > 0:
                logger.debug(f"[PROGRESS POLL] polled={polled_count}, timeouts={timeout_count}, gathers={in_progress_gathers}, traces={in_progress_traces}")

            # Report combined progress (completed + in-progress)
            total_current_traces = completed_traces + in_progress_traces
            total_current_gathers = completed_gathers + in_progress_gathers

            # Ensure monotonic progress - never report lower than previous maximum
            # This prevents UI from showing backwards progress when workers timeout
            # (timeouts cause partial counts, making raw totals fluctuate)
            raw_traces = total_current_traces
            total_current_traces = max(total_current_traces, max_reported_traces)
            total_current_gathers = max(total_current_gathers, max_reported_gathers)

            # Log when monotonic correction is applied
            if total_current_traces > raw_traces:
                logger.debug(f"[MONOTONIC] Corrected traces {raw_traces} -> {total_current_traces} (prevented backwards jump)")

            # Update maximums for next iteration
            max_reported_traces = total_current_traces
            max_reported_gathers = total_current_gathers

            if progress_callback:
                self._report_progress(
                    progress_callback,
                    phase='processing',
                    current_traces=total_current_traces,
                    total_traces=self._n_traces,
                    active_workers=active_workers if active_workers > 0 else len(pending),
                    current_gathers=total_current_gathers,
                )

        return results

    def _report_progress(
        self,
        callback: Optional[Callable],
        phase: str,
        current_traces: int,
        total_traces: int,
        active_workers: int,
        current_gathers: int = 0,
    ):
        """Report progress to callback with full statistics."""
        if callback is None:
            return

        from utils.parallel_processing import ProcessingProgress

        elapsed = time.time() - self._start_time if self._start_time else 0

        if current_traces > 0 and elapsed > 0:
            rate = current_traces / elapsed
            remaining = total_traces - current_traces
            eta = remaining / rate if rate > 0 else 0
        else:
            eta = 0
            rate = 0

        # Calculate current_gathers estimate if not provided
        if current_gathers == 0 and current_traces > 0 and total_traces > 0:
            # Estimate gathers based on trace progress
            progress_ratio = current_traces / total_traces
            current_gathers = int(self._n_gathers * progress_ratio)

        progress = ProcessingProgress(
            phase=phase,
            current_traces=current_traces,
            total_traces=total_traces,
            current_gathers=current_gathers,
            total_gathers=self._n_gathers,
            active_workers=active_workers,
            elapsed_time=elapsed,
            eta_seconds=eta,
            traces_per_sec=rate,
            compute_kernel=self._compute_kernel,
        )

        try:
            callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")

    def _check_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        if self._cancel_requested:
            return True
        if self._token and self._token.is_cancelled:
            self._cancel_requested = True
            return True
        return False

    def _cancelled_result(self) -> Any:
        """Create result for cancelled processing."""
        from utils.parallel_processing import ProcessingResult

        elapsed = time.time() - self._start_time if self._start_time else 0

        return ProcessingResult(
            success=False,
            output_dir=str(self._config.output_storage_dir),
            output_zarr_path="",
            n_gathers=0,
            n_traces=0,
            n_samples=self._n_samples,
            elapsed_time=elapsed,
            throughput_traces_per_sec=0,
            n_workers_used=0,
            error="Processing cancelled by user",
        )

    def _cleanup_actors(self):
        """Cleanup Ray actors on cancellation or error."""
        for actor in self._actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self._actors = []

    def get_actors(self) -> List[Any]:
        """Get list of actor handles for external management."""
        return self._actors

    def cancel(self):
        """Request cancellation of processing."""
        self._cancel_requested = True
        self._cleanup_actors()

    def _finalize_output(
        self,
        total_traces: int,
        total_gathers: int,
        throughput: float,
    ):
        """
        Finalize output directory with metadata files.

        Copies headers.parquet and ensemble_index.parquet from input,
        and creates metadata.json with processing info.
        """
        import shutil

        input_dir = Path(self._config.input_storage_dir)
        output_dir = Path(self._config.output_storage_dir)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy required metadata files
        required_files = [
            ('headers.parquet', 'Trace headers'),
            ('ensemble_index.parquet', 'Ensemble/gather index'),
        ]

        for filename, description in required_files:
            src = input_dir / filename
            dst = output_dir / filename

            if not src.exists():
                raise FileNotFoundError(
                    f"Required metadata file missing: {filename} ({description}). "
                    f"Input dataset at {input_dir} may be corrupted."
                )

            try:
                shutil.copy2(src, dst)
                if not dst.exists():
                    raise RuntimeError(f"File copy failed: {filename}")
                logger.info(f"Copied {filename} to output")
            except Exception as e:
                raise RuntimeError(f"Failed to copy {filename}: {e}") from e

        # Copy optional files
        optional_files = ['trace_index.parquet']
        for filename in optional_files:
            src = input_dir / filename
            dst = output_dir / filename
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied optional file {filename}")
                except Exception as e:
                    logger.warning(f"Failed to copy optional file {filename}: {e}")

        # Create metadata.json
        metadata = self._metadata.copy()
        metadata.update({
            'n_traces': total_traces,
            'n_gathers': total_gathers,
            'n_samples': self._n_samples,
            'sample_rate_ms': self._sample_rate,
            'processing': {
                'processor': self._config.processor_config.get('class_name', 'Unknown'),
                'throughput_traces_per_sec': throughput,
                'elapsed_time': time.time() - self._start_time if self._start_time else 0,
                'n_workers': self._n_workers,
                'compute_kernel': self._compute_kernel,
            },
        })

        # Preserve original SEG-Y path for export
        if 'original_segy_path' not in metadata:
            # Try to find in seismic_metadata
            if 'seismic_metadata' in metadata:
                original_path = metadata['seismic_metadata'].get('original_segy_path')
                if original_path:
                    metadata['original_segy_path'] = original_path
                    logger.info(f"Preserved original SEG-Y path: {original_path}")

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Created metadata.json in output")

        # Validate output
        self._validate_output(output_dir)

    def _validate_output(self, output_dir: Path):
        """Validate all required output files exist."""
        required = [
            'headers.parquet',
            'ensemble_index.parquet',
            'metadata.json',
            'traces.zarr',
        ]

        missing = [f for f in required if not (output_dir / f).exists()]

        if missing:
            raise RuntimeError(
                f"Output validation failed. Missing: {', '.join(missing)}"
            )
