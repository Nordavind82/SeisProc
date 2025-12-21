"""
Processing Job Adapter

Integrates parallel processing with the job management system.
Provides proper cancellation, progress reporting, and checkpoint support
for batch seismic processing operations.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from uuid import UUID
from datetime import datetime

from models.job import Job, JobType, JobState

# Note: JobType.BATCH_PROCESS is the correct enum value
from models.job_progress import JobProgress, WorkerProgress, ProgressUpdate
from models.job_config import JobConfig
from .job_manager import get_job_manager
from .cancellation import (
    CancellationToken,
    get_cancellation_coordinator,
    CancellationError,
)
from .checkpoint import (
    get_checkpoint_manager,
    Checkpoint,
    save_checkpoint,
    load_latest_checkpoint,
)

logger = logging.getLogger(__name__)


class ProcessingJobAdapter:
    """
    Adapter for running seismic processing through the job management system.

    .. deprecated::
        This adapter uses ParallelProcessingCoordinator (ProcessPoolExecutor)
        which has issues with Metal GPU initialization. Use RayProcessingJobAdapter
        instead, which uses Ray actors for proper GPU context initialization.

    Wraps the existing ParallelProcessingCoordinator with:
    - Proper multi-level cancellation
    - Progress reporting to JobManager
    - Checkpoint support for pause/resume
    - Integration with Job Dashboard UI

    Usage (deprecated)
    ------------------
    >>> adapter = ProcessingJobAdapter(config)
    >>> result = adapter.run()

    New Usage (preferred)
    ---------------------
    >>> adapter = RayProcessingJobAdapter(config, use_ray=True)
    >>> result = adapter.run()
    """

    def __init__(
        self,
        processing_config: Any,
        job_name: Optional[str] = None,
        qt_bridge: Optional[Any] = None,
    ):
        """
        Initialize adapter.

        Parameters
        ----------
        processing_config : ProcessingConfig
            Processing configuration
        job_name : str, optional
            Human-readable job name (defaults to processor name)
        qt_bridge : JobManagerBridge, optional
            Qt bridge for UI signal integration. When provided, progress
            and state updates are emitted to the Qt Dashboard.
        """
        import warnings
        warnings.warn(
            "ProcessingJobAdapter is deprecated. Use RayProcessingJobAdapter "
            "with use_ray=True for proper Metal GPU support.",
            DeprecationWarning,
            stacklevel=2
        )
        self._processing_config = processing_config
        self._job_name = job_name or self._get_default_name()
        self._qt_bridge = qt_bridge
        self._job: Optional[Job] = None
        self._token: Optional[CancellationToken] = None
        self._manager = get_job_manager()
        self._coordinator = None
        self._running = False
        self._result = None
        self._checkpoint_interval = 60  # Checkpoint every 60 seconds
        self._last_checkpoint_time = 0.0

    def _get_default_name(self) -> str:
        """Generate default job name from processor config."""
        proc_config = self._processing_config.processor_config
        if proc_config:
            class_name = proc_config.get('class_name', 'Unknown')
            return f"Process: {class_name}"
        return "Seismic Processing"

    def _detect_compute_kernel(self) -> str:
        """Detect which compute kernel is being used from processor config."""
        proc_config = self._processing_config.processor_config
        if not proc_config:
            return "CPU (Python)"

        # Check for Metal/GPU flags in processor config
        use_metal = proc_config.get('use_metal', False)
        use_gpu = proc_config.get('use_gpu', False)
        backend = proc_config.get('backend', '') or ''
        backend_lower = backend.lower()

        # Check backend string - 'metal_cpp' is the actual value from KernelBackend enum
        if use_metal or backend_lower in ('metal', 'metal_cpp'):
            return "Metal GPU"
        elif use_gpu or backend_lower in ('cuda', 'gpu'):
            return "CUDA GPU"
        elif backend_lower == 'mlx':
            return "MLX (Apple Silicon)"
        elif backend_lower == 'auto':
            # Auto mode - check if Metal is available and being used
            try:
                from processors.kernel_backend import get_backend_info
                info = get_backend_info()
                effective = info.get('effective_backend', '')
                if effective == 'metal_cpp':
                    return "Metal GPU (Auto)"
            except ImportError:
                pass

        # Check class name for hints
        class_name = proc_config.get('class_name', '').lower()
        if 'metal' in class_name:
            return "Metal GPU"
        elif 'gpu' in class_name or 'cuda' in class_name:
            return "CUDA GPU"

        return "CPU (Python)"

    def _emit_job_queued(self):
        """Emit job queued event to Qt bridge."""
        if self._qt_bridge and self._job:
            try:
                self._qt_bridge.signals.emit_job_queued(self._job)
            except Exception as e:
                logger.debug(f"Failed to emit job_queued signal: {e}")

    def _emit_job_started(self):
        """Emit job started event to Qt bridge."""
        if self._qt_bridge and self._job:
            try:
                self._qt_bridge.signals.emit_job_started(self._job)
            except Exception as e:
                logger.debug(f"Failed to emit job_started signal: {e}")

    def _emit_progress(self, progress_info: Dict[str, Any]):
        """
        Emit progress update to both JobManager and Qt bridge.

        Parameters
        ----------
        progress_info : dict
            Progress information with keys:
            - percent: Overall progress percentage
            - message: Progress message
            - phase: Current processing phase
            - eta_seconds: Estimated time remaining
            - workers: Number of workers
            - active_workers: Number of active workers
        """
        if self._qt_bridge and self._job:
            try:
                self._qt_bridge.signals.job_progress.emit(
                    self._job.id,
                    progress_info
                )
            except Exception as e:
                logger.debug(f"Failed to emit progress signal: {e}")

    def _emit_job_completed(self):
        """Emit job completed event to Qt bridge."""
        if self._qt_bridge and self._job:
            try:
                self._qt_bridge.signals.emit_job_completed(self._job)
            except Exception as e:
                logger.debug(f"Failed to emit job_completed signal: {e}")

    def _emit_job_failed(self):
        """Emit job failed event to Qt bridge."""
        if self._qt_bridge and self._job:
            try:
                self._qt_bridge.signals.emit_job_failed(self._job)
            except Exception as e:
                logger.debug(f"Failed to emit job_failed signal: {e}")

    def _emit_state_changed(self):
        """Emit state changed event to Qt bridge."""
        if self._qt_bridge and self._job:
            try:
                self._qt_bridge.signals.emit_state_changed(self._job)
            except Exception as e:
                logger.debug(f"Failed to emit state_changed signal: {e}")

    @property
    def job(self) -> Optional[Job]:
        """Get the associated job."""
        return self._job

    @property
    def job_id(self) -> Optional[UUID]:
        """Get job ID."""
        return self._job.id if self._job else None

    def submit(self) -> Job:
        """
        Submit the processing job for execution.

        Returns
        -------
        Job
            The created job (in QUEUED state)
        """
        # Estimate resource requirements
        n_workers = self._processing_config.n_workers or 4

        # Create job (trace_count used for worker estimation)
        self._job = self._manager.submit_job(
            name=self._job_name,
            job_type=JobType.BATCH_PROCESS,
            config=JobConfig.for_batch_processing(trace_count=n_workers * 10000),
            custom_config={
                'input_dir': self._processing_config.input_storage_dir,
                'output_dir': self._processing_config.output_storage_dir,
                'processor': self._processing_config.processor_config,
                'n_workers': n_workers,
            },
        )

        # Get cancellation token
        self._token = self._manager.get_cancellation_token(self._job.id)

        # Emit queued signal to Qt bridge for Dashboard visibility
        self._emit_job_queued()

        logger.info(f"Processing job submitted: {self._job.id}")
        return self._job

    def run(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Any:
        """
        Run the processing synchronously.

        This is a blocking call that runs the full processing pipeline.

        Parameters
        ----------
        progress_callback : callable, optional
            Progress callback (legacy compatibility)

        Returns
        -------
        ProcessingResult
            Result of the processing operation
        """
        # Submit job if not already done
        if self._job is None:
            self.submit()

        # Start job
        self._manager.start_job(self._job.id)
        self._running = True

        # Emit started signal to Qt bridge for Dashboard
        self._emit_job_started()

        try:
            # Check for existing checkpoint
            checkpoint = load_latest_checkpoint(self._job.id)
            if checkpoint:
                logger.info(
                    f"Resuming from checkpoint: {checkpoint.items_completed}/"
                    f"{checkpoint.items_total} items"
                )

            # Run the processing
            result = self._run_processing(progress_callback, checkpoint)

            if result.success:
                self._manager.complete_job(
                    self._job.id,
                    result={
                        'n_gathers': result.n_gathers,
                        'n_traces': result.n_traces,
                        'output_dir': result.output_dir,
                        'elapsed_time': result.elapsed_time,
                        'throughput': result.throughput_traces_per_sec,
                    },
                )
                # Emit completion signals to Qt bridge
                self._emit_job_completed()
                self._emit_state_changed()
            else:
                self._manager.fail_job(
                    self._job.id,
                    error=result.error or "Unknown error"
                )
                # Emit failure signals to Qt bridge
                self._emit_job_failed()
                self._emit_state_changed()

            self._result = result
            return result

        except CancellationError:
            self._manager.finalize_cancellation(self._job.id)
            # Save final checkpoint for resume
            self._save_cancellation_checkpoint()
            # Emit cancellation state to Qt bridge
            self._emit_state_changed()

            from utils.parallel_processing import ProcessingResult
            return ProcessingResult(
                success=False,
                output_dir=str(self._processing_config.output_storage_dir),
                output_zarr_path="",
                n_gathers=0,
                n_traces=0,
                n_samples=0,
                elapsed_time=0,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error="Processing cancelled by user",
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self._manager.fail_job(self._job.id, error=str(e))
            # Emit failure signals to Qt bridge
            self._emit_job_failed()
            self._emit_state_changed()
            raise

        finally:
            self._running = False

    def _run_processing(
        self,
        progress_callback: Optional[Callable],
        checkpoint: Optional[Checkpoint],
    ) -> Any:
        """Run the actual processing using the coordinator."""
        from utils.parallel_processing.coordinator import ParallelProcessingCoordinator

        # Create coordinator
        self._coordinator = ParallelProcessingCoordinator(self._processing_config)

        # If resuming from checkpoint, adjust start position
        if checkpoint and checkpoint.state:
            start_gather = checkpoint.state.get('next_gather_idx', 0)
            if start_gather > 0:
                logger.info(f"Resuming from gather {start_gather}")
                # Note: The coordinator would need to support starting from
                # a specific gather index. For now, we log the intent.

        # Create wrapper progress callback
        def wrapped_progress(progress):
            # Check for cancellation
            if self._token and self._token.is_cancelled:
                self._coordinator._cancel_requested = True
                raise CancellationError(self._token.cancellation_request)

            # Check for pause
            if self._token:
                self._token.wait_if_paused()

            # Update job progress
            if self._job:
                total = progress.total_traces
                current = progress.current_traces

                update = ProgressUpdate(
                    job_id=self._job.id,
                    worker_id="coordinator",
                    items_processed=current,
                    items_total=total,
                    message=f"Phase: {progress.phase}",
                    metrics={
                        'active_workers': progress.active_workers,
                        'current_gathers': progress.current_gathers,
                        'total_gathers': progress.total_gathers,
                        'elapsed_time': progress.elapsed_time,
                        'eta_seconds': progress.eta_seconds,
                    },
                )
                self._manager.update_progress(update)

                # Emit progress to Qt bridge for Dashboard with full statistics
                percent = (current / total * 100) if total > 0 else 0
                elapsed = progress.elapsed_time if progress.elapsed_time > 0 else 0.001
                traces_per_sec = current / elapsed if elapsed > 0 else 0

                self._emit_progress({
                    'percent': percent,
                    'message': f"Phase: {progress.phase}",
                    'phase': progress.phase,
                    'eta_seconds': progress.eta_seconds,
                    'active_workers': progress.active_workers,
                    # Extended statistics for Dashboard display
                    'current_gathers': progress.current_gathers,
                    'total_gathers': progress.total_gathers,
                    'current_traces': current,
                    'total_traces': total,
                    'traces_per_sec': traces_per_sec,
                    # Compute kernel info
                    'compute_kernel': self._detect_compute_kernel(),
                })

            # Periodic checkpoint
            self._maybe_save_checkpoint(progress)

            # Call legacy callback
            if progress_callback:
                progress_callback(progress)

        # Run processing
        return self._coordinator.run(progress_callback=wrapped_progress)

    def _maybe_save_checkpoint(self, progress):
        """Save checkpoint if interval has elapsed."""
        now = time.time()
        if (now - self._last_checkpoint_time) < self._checkpoint_interval:
            return

        self._last_checkpoint_time = now

        try:
            save_checkpoint(
                job_id=self._job.id,
                phase=progress.phase,
                items_completed=progress.current_traces,
                items_total=progress.total_traces,
                state={
                    'current_gathers': progress.current_gathers,
                    'total_gathers': progress.total_gathers,
                    'worker_progress': progress.worker_progress,
                },
            )
            logger.debug(
                f"Checkpoint saved: {progress.current_traces}/{progress.total_traces}"
            )
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _save_cancellation_checkpoint(self):
        """Save checkpoint when cancelled for potential resume."""
        if not self._job:
            return

        try:
            # Get current progress from job
            job = self._manager.get_job(self._job.id)
            if job and job.progress:
                save_checkpoint(
                    job_id=self._job.id,
                    phase='cancelled',
                    items_completed=job.progress.items_completed,
                    items_total=job.progress.items_total,
                    state={'cancelled_at': datetime.now().isoformat()},
                )
        except Exception as e:
            logger.warning(f"Failed to save cancellation checkpoint: {e}")

    def cancel(self) -> bool:
        """
        Request cancellation of the processing.

        Returns
        -------
        bool
            True if cancellation was initiated
        """
        if self._job is None:
            return False

        return self._manager.cancel_job(self._job.id)

    def pause(self) -> bool:
        """
        Pause the processing (workers will finish current gather then wait).

        Returns
        -------
        bool
            True if pause was initiated
        """
        if self._job is None:
            return False

        return self._manager.pause_job(self._job.id)

    def resume(self) -> bool:
        """
        Resume a paused processing job.

        Returns
        -------
        bool
            True if resume was initiated
        """
        if self._job is None:
            return False

        return self._manager.resume_job(self._job.id)


class RayProcessingJobAdapter:
    """
    Ray-based processing adapter for distributed execution.

    Uses Ray actors instead of ProcessPoolExecutor for:
    - Better fault tolerance
    - Distributed execution across cluster
    - More efficient cancellation
    """

    def __init__(
        self,
        processing_config: Any,
        job_name: Optional[str] = None,
        use_ray: bool = True,
        qt_bridge: Optional[Any] = None,
    ):
        """
        Initialize Ray processing adapter.

        Parameters
        ----------
        processing_config : ProcessingConfig
            Processing configuration
        job_name : str, optional
            Human-readable job name
        use_ray : bool
            Whether to use Ray (falls back to local if False)
        qt_bridge : JobManagerBridge, optional
            Qt bridge for UI signal integration
        """
        self._processing_config = processing_config
        self._job_name = job_name or "Ray Processing"
        self._use_ray = use_ray
        self._qt_bridge = qt_bridge
        self._job: Optional[Job] = None
        self._token: Optional[CancellationToken] = None
        self._manager = get_job_manager()
        self._worker_actors = []
        self._running = False

    @property
    def job(self) -> Optional[Job]:
        """Get the associated job."""
        return self._job

    def _detect_compute_kernel(self) -> str:
        """Detect which compute kernel is being used from processor config."""
        proc_config = self._processing_config.processor_config
        if not proc_config:
            return "CPU (Python)"

        # Check for Metal/GPU flags in processor config
        use_metal = proc_config.get('use_metal', False)
        use_gpu = proc_config.get('use_gpu', False)
        backend = proc_config.get('backend', '') or ''
        backend_lower = backend.lower()

        if use_metal or backend_lower in ('metal', 'metal_cpp'):
            return "Metal GPU"
        elif use_gpu or backend_lower in ('cuda', 'gpu'):
            return "CUDA GPU"
        elif backend_lower == 'mlx':
            return "MLX (Apple Silicon)"
        elif backend_lower == 'auto':
            try:
                from processors.kernel_backend import get_backend_info
                info = get_backend_info()
                effective = info.get('effective_backend', '')
                if effective == 'metal_cpp':
                    return "Metal GPU (Auto)"
            except ImportError:
                pass

        class_name = proc_config.get('class_name', '').lower()
        if 'metal' in class_name:
            return "Metal GPU"
        elif 'gpu' in class_name or 'cuda' in class_name:
            return "CUDA GPU"

        return "CPU (Python)"

    def submit(self) -> Job:
        """Submit the processing job."""
        n_workers = self._processing_config.n_workers or 4

        self._job = self._manager.submit_job(
            name=self._job_name,
            job_type=JobType.BATCH_PROCESS,
            config=JobConfig.for_batch_processing(trace_count=n_workers * 10000),
            custom_config={
                'input_dir': self._processing_config.input_storage_dir,
                'output_dir': self._processing_config.output_storage_dir,
                'use_ray': self._use_ray,
            },
        )

        self._token = self._manager.get_cancellation_token(self._job.id)

        logger.info(f"Ray processing job submitted: {self._job.id}")
        return self._job

    def run(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Any:
        """
        Run processing with Ray actors.

        Parameters
        ----------
        progress_callback : callable, optional
            Progress callback

        Returns
        -------
        ProcessingResult
            Result of the processing operation
        """
        if self._job is None:
            self.submit()

        self._manager.start_job(self._job.id)
        self._running = True

        try:
            if self._use_ray:
                result = self._run_with_ray(progress_callback)
            else:
                result = self._run_locally(progress_callback)

            if result.success:
                self._manager.complete_job(self._job.id, result=result.__dict__)
            else:
                self._manager.fail_job(self._job.id, error=result.error)

            return result

        except CancellationError:
            self._manager.finalize_cancellation(self._job.id)
            self._cancel_workers()

            from utils.parallel_processing import ProcessingResult
            return ProcessingResult(
                success=False,
                output_dir=str(self._processing_config.output_storage_dir),
                output_zarr_path="",
                n_gathers=0,
                n_traces=0,
                n_samples=0,
                elapsed_time=0,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error="Processing cancelled",
            )

        except Exception as e:
            logger.error(f"Ray processing failed: {e}")
            self._manager.fail_job(self._job.id, error=str(e))
            self._cancel_workers()
            raise

        finally:
            self._running = False

    def _run_with_ray(self, progress_callback: Optional[Callable]) -> Any:
        """Run processing using Ray actors."""
        try:
            import ray
        except ImportError:
            logger.warning("Ray not available, falling back to local execution")
            return self._run_locally(progress_callback)

        from .ray_processing_coordinator import RayProcessingCoordinator
        from .cluster import initialize_ray, is_ray_initialized

        # Ensure Ray is initialized
        if not is_ray_initialized():
            if not initialize_ray():
                logger.warning("Ray init failed, falling back to local execution")
                return self._run_locally(progress_callback)

        logger.info("Using Ray-based parallel processing")

        # Create coordinator with job context
        coordinator = RayProcessingCoordinator(
            self._processing_config,
            job_id=self._job.id if self._job else None,
            cancellation_token=self._token,
            qt_bridge=self._qt_bridge,
        )

        # Store actors reference for cleanup on cancel
        self._coordinator = coordinator

        # Wrap progress callback to emit to Qt bridge
        def wrapped_progress(progress):
            # Check for cancellation
            if self._token and self._token.is_cancelled:
                raise CancellationError(self._token.cancellation_request)

            # Check for pause
            if self._token:
                self._token.wait_if_paused()

            # Emit progress to Qt bridge
            if self._qt_bridge and self._job:
                try:
                    percent = (progress.current_traces / progress.total_traces * 100
                               if progress.total_traces > 0 else 0)

                    # Use progress fields if available, else calculate
                    traces_per_sec = getattr(progress, 'traces_per_sec', 0)
                    if traces_per_sec == 0 and progress.elapsed_time > 0:
                        traces_per_sec = progress.current_traces / progress.elapsed_time

                    compute_kernel = getattr(progress, 'compute_kernel', '')
                    if not compute_kernel:
                        compute_kernel = self._detect_compute_kernel()

                    logger.info(f"[PROGRESS EMIT] {percent:.1f}% - {progress.current_traces}/{progress.total_traces} traces, {progress.active_workers} workers, kernel={compute_kernel}")

                    # Also update JobManager's internal progress so polling works
                    try:
                        from models.job_progress import ProgressUpdate as JobProgressUpdate
                        job_update = JobProgressUpdate(
                            job_id=self._job.id,
                            worker_id="coordinator",
                            items_processed=progress.current_traces,
                            items_total=progress.total_traces,
                            message=f"Phase: {progress.phase}",
                            metrics={
                                'current_gathers': progress.current_gathers,
                                'total_gathers': progress.total_gathers,
                                'traces_per_sec': traces_per_sec,
                                'compute_kernel': compute_kernel,
                                'active_workers': progress.active_workers,
                            },
                        )
                        self._manager.update_progress(job_update)
                    except Exception as e:
                        logger.debug(f"Failed to update JobManager progress: {e}")

                    self._qt_bridge.signals.job_progress.emit(
                        self._job.id,
                        {
                            'percent': percent,
                            'message': f"Phase: {progress.phase}",
                            'phase': progress.phase,
                            'eta_seconds': progress.eta_seconds,
                            'active_workers': progress.active_workers,
                            'current_gathers': progress.current_gathers,
                            'total_gathers': progress.total_gathers,
                            'current_traces': progress.current_traces,
                            'total_traces': progress.total_traces,
                            'traces_per_sec': traces_per_sec,
                            'compute_kernel': compute_kernel,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit progress signal: {e}")

            # Call user callback
            if progress_callback:
                progress_callback(progress)

        # Run the coordinator
        return coordinator.run(progress_callback=wrapped_progress)

    def _run_locally(self, progress_callback: Optional[Callable]) -> Any:
        """Fall back to local processing."""
        adapter = ProcessingJobAdapter(
            self._processing_config,
            job_name=self._job_name,
            qt_bridge=self._qt_bridge,
        )
        # Reuse our job instead of creating a new one
        adapter._job = self._job
        adapter._token = self._token
        return adapter._run_processing(progress_callback, None)

    def _cancel_workers(self):
        """Cancel all Ray worker actors."""
        # Cancel via coordinator if available
        if hasattr(self, '_coordinator') and self._coordinator:
            self._coordinator.cancel()
            return

        # Legacy fallback
        for actor in self._worker_actors:
            try:
                import ray
                ray.cancel(actor)
            except Exception as e:
                logger.warning(f"Failed to cancel worker: {e}")

        self._worker_actors.clear()

    def cancel(self) -> bool:
        """Request cancellation."""
        if self._job is None:
            return False

        self._cancel_workers()
        return self._manager.cancel_job(self._job.id)

    def pause(self) -> bool:
        """Pause processing."""
        if self._job is None:
            return False
        return self._manager.pause_job(self._job.id)

    def resume(self) -> bool:
        """Resume processing."""
        if self._job is None:
            return False
        return self._manager.resume_job(self._job.id)


def create_processing_job(
    input_storage_dir: str,
    output_storage_dir: str,
    processor_config: Dict[str, Any],
    job_name: Optional[str] = None,
    n_workers: Optional[int] = None,
    use_ray: bool = False,
    qt_bridge: Optional[Any] = None,
    **kwargs,
) -> ProcessingJobAdapter:
    """
    Convenience function to create a processing job.

    Parameters
    ----------
    input_storage_dir : str
        Path to input storage directory
    output_storage_dir : str
        Path for output storage directory
    processor_config : dict
        Processor configuration dictionary
    job_name : str, optional
        Human-readable job name
    n_workers : int, optional
        Number of worker processes
    use_ray : bool
        Whether to use Ray for distributed processing
    qt_bridge : JobManagerBridge, optional
        Qt bridge for UI signal integration
    **kwargs
        Additional ProcessingConfig options

    Returns
    -------
    ProcessingJobAdapter or RayProcessingJobAdapter
        Adapter ready to run
    """
    from utils.parallel_processing import ProcessingConfig

    config = ProcessingConfig(
        input_storage_dir=input_storage_dir,
        output_storage_dir=output_storage_dir,
        processor_config=processor_config,
        n_workers=n_workers,
        **kwargs,
    )

    if use_ray:
        return RayProcessingJobAdapter(config, job_name=job_name, qt_bridge=qt_bridge)
    else:
        return ProcessingJobAdapter(config, job_name=job_name, qt_bridge=qt_bridge)
