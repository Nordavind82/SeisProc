"""
SEGY Import/Export Job Adapter

Integrates the existing SEGY multiprocess import with the new job management system.
Provides proper cancellation, progress reporting, and checkpoint support.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from uuid import UUID
from datetime import datetime

from models.job import Job, JobType, JobState
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


class SEGYImportJobAdapter:
    """
    Adapter for running SEGY imports through the job management system.

    Wraps the existing ParallelImportCoordinator with:
    - Proper multi-level cancellation
    - Progress reporting to JobManager
    - Checkpoint support for pause/resume
    - Integration with Job Dashboard UI

    Usage
    -----
    >>> from utils.segy_import.multiprocess_import.coordinator import ImportConfig
    >>>
    >>> config = ImportConfig(
    ...     segy_path="/path/to/file.sgy",
    ...     output_dir="/path/to/output",
    ...     header_mapping=mapping,
    ... )
    >>>
    >>> adapter = SEGYImportJobAdapter(config)
    >>> result = adapter.run()  # Blocking
    >>> # or
    >>> job = adapter.submit()  # Non-blocking, returns Job
    """

    def __init__(self, import_config: Any, job_name: Optional[str] = None):
        """
        Initialize adapter.

        Parameters
        ----------
        import_config : ImportConfig
            SEGY import configuration
        job_name : str, optional
            Human-readable job name (defaults to filename)
        """
        self._import_config = import_config
        self._job_name = job_name or Path(import_config.segy_path).name
        self._job: Optional[Job] = None
        self._token: Optional[CancellationToken] = None
        self._manager = get_job_manager()
        self._coordinator = None
        self._running = False
        self._result = None

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
        Submit the import job for execution.

        Returns
        -------
        Job
            The created job (in QUEUED state)
        """
        # Create job
        self._job = self._manager.submit_job(
            name=f"Import: {self._job_name}",
            job_type=JobType.SEGY_IMPORT,
            config=JobConfig.for_segy_import(
                file_size_mb=self._get_file_size_mb()
            ),
            custom_config={
                "segy_path": self._import_config.segy_path,
                "output_dir": self._import_config.output_dir,
            },
        )

        # Get cancellation token
        self._token = self._manager.get_cancellation_token(self._job.id)

        logger.info(f"SEGY import job submitted: {self._job.id}")
        return self._job

    def run(self, progress_callback: Optional[Callable] = None) -> Any:
        """
        Run the import synchronously.

        This is a blocking call that runs the full import pipeline.

        Parameters
        ----------
        progress_callback : callable, optional
            Progress callback (legacy compatibility)

        Returns
        -------
        ImportResult
            Result of the import operation
        """
        # Submit job if not already done
        if self._job is None:
            self.submit()

        # Start job
        self._manager.start_job(self._job.id)
        self._running = True

        try:
            # Run the import
            result = self._run_import(progress_callback)

            if result.success:
                self._manager.complete_job(
                    self._job.id,
                    result={
                        "n_traces": result.n_traces,
                        "traces_path": result.traces_path,
                        "headers_path": result.headers_path,
                        "elapsed_time": result.elapsed_time,
                    },
                )
            else:
                self._manager.fail_job(self._job.id, error=result.error or "Unknown error")

            self._result = result
            return result

        except CancellationError:
            self._manager.finalize_cancellation(self._job.id)
            # Return partial result if available
            from utils.segy_import.multiprocess_import.coordinator import ImportResult
            return ImportResult(
                success=False,
                output_dir=self._import_config.output_dir,
                traces_path="",
                headers_path="",
                n_traces=0,
                n_segments=0,
                elapsed_time=0,
                error="Import cancelled by user",
            )

        except Exception as e:
            logger.error(f"SEGY import failed: {e}")
            self._manager.fail_job(self._job.id, error=str(e))
            raise

        finally:
            self._running = False

    def _run_import(self, progress_callback: Optional[Callable]) -> Any:
        """Run the actual import using the coordinator."""
        from utils.segy_import.multiprocess_import.coordinator import (
            ParallelImportCoordinator,
            ImportProgress,
        )

        # Create coordinator
        self._coordinator = ParallelImportCoordinator(self._import_config)

        # Create wrapper progress callback that:
        # 1. Checks for cancellation
        # 2. Updates job progress
        # 3. Calls legacy callback if provided
        def wrapped_progress(import_progress: ImportProgress):
            # Check for cancellation
            if self._token and self._token.is_cancelled:
                # Signal cancellation to coordinator
                self._coordinator._cancel_requested = True
                raise CancellationError(self._token.cancellation_request)

            # Check for pause
            if self._token:
                self._token.wait_if_paused()

            # Update job progress
            if self._job:
                total = import_progress.total_traces
                current = import_progress.current_traces
                percent = (current / total * 100) if total > 0 else 0

                update = ProgressUpdate(
                    job_id=self._job.id,
                    worker_id="coordinator",
                    items_processed=current,
                    items_total=total,
                    message=f"Phase: {import_progress.phase}",
                    metrics={
                        "active_workers": import_progress.active_workers,
                        "elapsed_time": import_progress.elapsed_time,
                    },
                )
                self._manager.update_progress(update)

            # Call legacy callback
            if progress_callback:
                progress_callback(import_progress)

        # Run import
        return self._coordinator.run(progress_callback=wrapped_progress)

    def cancel(self) -> bool:
        """
        Request cancellation of the import.

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
        Pause the import (workers will finish current batch then wait).

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
        Resume a paused import.

        Returns
        -------
        bool
            True if resume was initiated
        """
        if self._job is None:
            return False

        return self._manager.resume_job(self._job.id)

    def _get_file_size_mb(self) -> int:
        """Get SEGY file size in MB."""
        try:
            path = Path(self._import_config.segy_path)
            return int(path.stat().st_size / (1024 * 1024))
        except Exception:
            return 100  # Default estimate


class SEGYExportJobAdapter:
    """
    Adapter for running SEGY exports through the job management system.

    Similar to SEGYImportJobAdapter but for export operations.
    """

    def __init__(
        self,
        traces_path: str,
        headers_path: str,
        output_path: str,
        job_name: Optional[str] = None,
    ):
        """
        Initialize export adapter.

        Parameters
        ----------
        traces_path : str
            Path to traces Zarr array
        headers_path : str
            Path to headers parquet file
        output_path : str
            Path for output SEGY file
        job_name : str, optional
            Human-readable job name
        """
        self._traces_path = traces_path
        self._headers_path = headers_path
        self._output_path = output_path
        self._job_name = job_name or Path(output_path).name
        self._job: Optional[Job] = None
        self._token: Optional[CancellationToken] = None
        self._manager = get_job_manager()

    @property
    def job(self) -> Optional[Job]:
        """Get the associated job."""
        return self._job

    def submit(self) -> Job:
        """Submit the export job for execution."""
        self._job = self._manager.submit_job(
            name=f"Export: {self._job_name}",
            job_type=JobType.SEGY_EXPORT,
            custom_config={
                "traces_path": self._traces_path,
                "headers_path": self._headers_path,
                "output_path": self._output_path,
            },
        )

        self._token = self._manager.get_cancellation_token(self._job.id)

        logger.info(f"SEGY export job submitted: {self._job.id}")
        return self._job

    def cancel(self) -> bool:
        """Request cancellation of the export."""
        if self._job is None:
            return False
        return self._manager.cancel_job(self._job.id)


def create_import_job(
    segy_path: str,
    output_dir: str,
    header_mapping: Any,
    job_name: Optional[str] = None,
    **kwargs,
) -> SEGYImportJobAdapter:
    """
    Convenience function to create a SEGY import job.

    Parameters
    ----------
    segy_path : str
        Path to SEGY file
    output_dir : str
        Output directory for Zarr/Parquet
    header_mapping : HeaderMapping
        Header mapping configuration
    job_name : str, optional
        Human-readable job name
    **kwargs
        Additional ImportConfig options

    Returns
    -------
    SEGYImportJobAdapter
        Adapter ready to run
    """
    from utils.segy_import.multiprocess_import.coordinator import ImportConfig

    config = ImportConfig(
        segy_path=segy_path,
        output_dir=output_dir,
        header_mapping=header_mapping,
        **kwargs,
    )

    return SEGYImportJobAdapter(config, job_name=job_name)
