"""
High-level Processing API for SeisProc.

Provides simple functions that handle all job management internally.
These functions integrate with the Job Dashboard, toast notifications,
and job history automatically.

Usage
-----
>>> from utils.ray_orchestration.processing_api import run_parallel_processing
>>>
>>> result = run_parallel_processing(
...     input_dir='/path/to/input',
...     output_dir='/path/to/output',
...     processor_config=processor.to_dict(),
...     progress_callback=on_progress,
... )

This replaces direct usage of ParallelProcessingCoordinator and provides:
- Job visibility in Dashboard
- Toast notifications on complete/failure
- Job history storage
- Pause/resume support
- Multi-level cancellation
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    job_id: Optional[UUID] = None
    output_dir: Optional[str] = None
    n_gathers: int = 0
    n_traces: int = 0
    elapsed_time: float = 0.0
    throughput: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_coordinator_result(cls, result: Any, job_id: UUID = None) -> 'ProcessingResult':
        """Create from a ParallelProcessingCoordinator result."""
        return cls(
            success=result.success,
            job_id=job_id,
            output_dir=result.output_dir,
            n_gathers=result.n_gathers,
            n_traces=result.n_traces,
            elapsed_time=result.elapsed_time,
            throughput=result.throughput_traces_per_sec,
            error=result.error,
        )


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    job_id: Optional[UUID] = None
    output_file: Optional[str] = None
    n_traces: int = 0
    elapsed_time: float = 0.0
    error: Optional[str] = None


def get_optimal_workers() -> int:
    """Get optimal number of workers based on system resources."""
    import os
    import multiprocessing

    # Use environment variable if set
    env_workers = os.environ.get('SEISPROC_WORKERS')
    if env_workers:
        try:
            return int(env_workers)
        except ValueError:
            pass

    # Use 75% of CPUs, minimum 2
    cpu_count = multiprocessing.cpu_count()
    return max(2, int(cpu_count * 0.75))


def _get_qt_bridge() -> Optional[Any]:
    """Get the Qt job bridge if available."""
    try:
        from .qt_bridge import get_job_bridge
        return get_job_bridge()
    except Exception as e:
        logger.debug(f"Qt bridge not available: {e}")
        return None


def run_parallel_processing(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    processor_config: Dict[str, Any],
    job_name: Optional[str] = None,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    use_qt_bridge: bool = True,
    show_in_dashboard: bool = True,
) -> ProcessingResult:
    """
    Run parallel processing with full job management.

    This is the recommended way to run batch processing operations.
    Handles:
    - Job creation and tracking in JobManager
    - Progress updates to Dashboard (if UI running)
    - Toast notifications on complete/failure
    - Job history storage for analytics
    - Cancellation and pause/resume support

    Parameters
    ----------
    input_dir : str or Path
        Path to input storage directory (Zarr format)
    output_dir : str or Path
        Path for output storage directory
    processor_config : dict
        Processor configuration dictionary from processor.to_dict()
    job_name : str, optional
        Human-readable job name (defaults to processor class name)
    n_workers : int, optional
        Number of worker processes (defaults to 75% of CPUs)
    progress_callback : callable, optional
        Legacy progress callback for additional UI updates.
        Signature: callback(progress) where progress has attributes:
        - current_traces, total_traces, current_gathers, total_gathers
        - phase, elapsed_time, eta_seconds, active_workers
    use_qt_bridge : bool
        Whether to use Qt bridge for Dashboard updates (default True)
    show_in_dashboard : bool
        Whether to show job in Dashboard (default True)

    Returns
    -------
    ProcessingResult
        Result with success status, statistics, and any errors

    Examples
    --------
    >>> from processors.denoise_3d import Denoise3DProcessor
    >>>
    >>> processor = Denoise3DProcessor()
    >>> processor.configure(strength=0.5)
    >>>
    >>> result = run_parallel_processing(
    ...     input_dir='/data/project/storage',
    ...     output_dir='/data/project/processed',
    ...     processor_config=processor.to_dict(),
    ...     job_name='Denoise 3D',
    ...     progress_callback=lambda p: print(f"{p.current_traces}/{p.total_traces}"),
    ... )
    >>>
    >>> if result.success:
    ...     print(f"Processed {result.n_traces} traces in {result.elapsed_time:.1f}s")
    """
    from .processing_job_adapter import ProcessingJobAdapter
    from utils.parallel_processing import ProcessingConfig

    # Get Qt bridge if requested
    qt_bridge = _get_qt_bridge() if use_qt_bridge else None

    # Determine job name
    if job_name is None:
        class_name = processor_config.get('class_name', 'Unknown')
        job_name = f"Batch: {class_name}"

    # Create processing config
    config = ProcessingConfig(
        input_storage_dir=str(input_dir),
        output_storage_dir=str(output_dir),
        processor_config=processor_config,
        n_workers=n_workers or get_optimal_workers(),
    )

    # Create adapter with Qt integration
    adapter = ProcessingJobAdapter(
        config,
        job_name=job_name,
        qt_bridge=qt_bridge,
    )

    try:
        # Submit job (appears in Dashboard immediately)
        job = adapter.submit()
        logger.info(f"Started processing job: {job.id} - {job_name}")

        # Run processing (blocking)
        coordinator_result = adapter.run(progress_callback=progress_callback)

        # Convert to our result type
        return ProcessingResult.from_coordinator_result(
            coordinator_result,
            job_id=job.id,
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return ProcessingResult(
            success=False,
            job_id=adapter.job_id,
            error=str(e),
        )


def run_segy_export(
    input_dir: Union[str, Path],
    output_file: Union[str, Path],
    template_segy: Union[str, Path],
    job_name: Optional[str] = None,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    use_qt_bridge: bool = True,
) -> ExportResult:
    """
    Run parallel SEG-Y export with full job management.

    Parameters
    ----------
    input_dir : str or Path
        Path to input storage directory (Zarr format)
    output_file : str or Path
        Path for output SEG-Y file
    template_segy : str or Path
        Path to template SEG-Y file for header copying
    job_name : str, optional
        Human-readable job name
    n_workers : int, optional
        Number of worker processes
    progress_callback : callable, optional
        Progress callback
    use_qt_bridge : bool
        Whether to use Qt bridge for Dashboard updates

    Returns
    -------
    ExportResult
        Result with success status and any errors
    """
    from .job_manager import get_job_manager
    from models.job import JobType
    from models.job_config import JobConfig

    qt_bridge = _get_qt_bridge() if use_qt_bridge else None
    manager = get_job_manager()

    if job_name is None:
        job_name = f"Export: {Path(output_file).name}"

    # Create job for tracking
    job = manager.submit_job(
        name=job_name,
        job_type=JobType.SEGY_EXPORT,
        config=JobConfig.for_segy_export(file_size_mb=0),
        custom_config={
            'input_dir': str(input_dir),
            'output_file': str(output_file),
            'template': str(template_segy),
        },
    )

    # Emit queued signal
    if qt_bridge:
        try:
            qt_bridge.signals.emit_job_queued(job)
        except Exception:
            pass

    try:
        manager.start_job(job.id)

        # Emit started signal
        if qt_bridge:
            try:
                qt_bridge.signals.emit_job_started(job)
            except Exception:
                pass

        # Run the export
        from utils.segy_export import parallel_export_segy

        result = parallel_export_segy(
            input_storage_dir=str(input_dir),
            output_segy_path=str(output_file),
            template_segy_path=str(template_segy),
            n_workers=n_workers or get_optimal_workers(),
            progress_callback=progress_callback,
        )

        if result.get('success', False):
            manager.complete_job(job.id, result=result)
            if qt_bridge:
                try:
                    qt_bridge.signals.emit_job_completed(job)
                except Exception:
                    pass

            return ExportResult(
                success=True,
                job_id=job.id,
                output_file=str(output_file),
                n_traces=result.get('n_traces', 0),
                elapsed_time=result.get('elapsed_time', 0.0),
            )
        else:
            error = result.get('error', 'Unknown error')
            manager.fail_job(job.id, error=error)
            if qt_bridge:
                try:
                    qt_bridge.signals.emit_job_failed(job)
                except Exception:
                    pass

            return ExportResult(
                success=False,
                job_id=job.id,
                error=error,
            )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        manager.fail_job(job.id, error=str(e))
        if qt_bridge:
            try:
                qt_bridge.signals.emit_job_failed(job)
            except Exception:
                pass

        return ExportResult(
            success=False,
            job_id=job.id,
            error=str(e),
        )


def cancel_processing_job(job_id: UUID) -> bool:
    """
    Cancel a running processing job.

    Parameters
    ----------
    job_id : UUID
        ID of the job to cancel

    Returns
    -------
    bool
        True if cancellation was initiated
    """
    from .job_manager import get_job_manager
    return get_job_manager().cancel_job(job_id)


def pause_processing_job(job_id: UUID) -> bool:
    """
    Pause a running processing job.

    Workers will complete their current gather, then wait.

    Parameters
    ----------
    job_id : UUID
        ID of the job to pause

    Returns
    -------
    bool
        True if pause was initiated
    """
    from .job_manager import get_job_manager
    return get_job_manager().pause_job(job_id)


def resume_processing_job(job_id: UUID) -> bool:
    """
    Resume a paused processing job.

    Parameters
    ----------
    job_id : UUID
        ID of the job to resume

    Returns
    -------
    bool
        True if resume was initiated
    """
    from .job_manager import get_job_manager
    return get_job_manager().resume_job(job_id)


def get_job_status(job_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Get the current status of a job.

    Parameters
    ----------
    job_id : UUID
        ID of the job

    Returns
    -------
    dict or None
        Job status with state, progress, and timing info
    """
    from .job_manager import get_job_manager

    manager = get_job_manager()
    job = manager.get_job(job_id)

    if job is None:
        return None

    progress = manager.get_progress(job_id)

    return {
        'id': str(job.id),
        'name': job.name,
        'state': job.state.name,
        'progress_percent': progress.overall_percent if progress else 0,
        'message': progress.message if progress else '',
        'eta_seconds': progress.eta_seconds if progress else None,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'duration_seconds': job.duration_seconds,
    }
