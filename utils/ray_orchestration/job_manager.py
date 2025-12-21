"""
Job Manager

Centralized job lifecycle management with Ray orchestration.
Handles job submission, monitoring, cancellation, and cleanup.
"""

import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from uuid import UUID

from models.job import Job, JobState, JobType, JobPriority
from models.job_progress import JobProgress, WorkerProgress, ProgressUpdate
from models.job_config import JobConfig

from .cluster import RayClusterManager, is_ray_initialized
from .cancellation import (
    CancellationToken,
    CancellationCoordinator,
    CancellationReason,
    get_cancellation_coordinator,
)
from .job_history import JobHistoryStorage, get_job_history_storage

logger = logging.getLogger(__name__)

# Lazy import ray
_ray = None


def _get_ray():
    """Lazy import Ray module."""
    global _ray
    if _ray is None:
        import ray
        _ray = ray
    return _ray


class JobManager:
    """
    Manages job lifecycle with Ray orchestration.

    Provides:
    - Job submission and queuing
    - Progress tracking
    - Cancellation/pause/resume
    - Job history and monitoring

    Thread-safe singleton pattern.

    Usage
    -----
    >>> manager = JobManager()
    >>> job = manager.submit_job("Import SEGY", JobType.SEGY_IMPORT, config)
    >>> # Monitor progress
    >>> progress = manager.get_progress(job.id)
    >>> # Cancel if needed
    >>> manager.cancel_job(job.id)
    """

    _instance: Optional['JobManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'JobManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, history_storage: Optional[JobHistoryStorage] = None):
        if self._initialized:
            return

        self._jobs: Dict[UUID, Job] = {}
        self._progress: Dict[UUID, JobProgress] = {}
        self._ray_refs: Dict[UUID, Any] = {}  # job_id -> ray.ObjectRef
        self._callbacks: Dict[str, List[Callable]] = {
            "on_job_queued": [],
            "on_job_started": [],
            "on_job_completed": [],
            "on_job_failed": [],
            "on_job_cancelled": [],
            "on_progress_update": [],
        }
        self._cancellation = get_cancellation_coordinator()
        self._history = history_storage or get_job_history_storage()
        self._initialized = True

    def submit_job(
        self,
        name: str,
        job_type: JobType,
        config: Optional[JobConfig] = None,
        priority: JobPriority = JobPriority.NORMAL,
        parent_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """
        Submit a new job for execution.

        Parameters
        ----------
        name : str
            Human-readable job name
        job_type : JobType
            Type of job
        config : JobConfig, optional
            Job configuration
        priority : JobPriority
            Job priority
        parent_id : UUID, optional
            Parent job ID for sub-jobs
        tags : list, optional
            Tags for categorization
        custom_config : dict, optional
            Additional configuration

        Returns
        -------
        Job
            The created job (in QUEUED state)
        """
        config = config or JobConfig()
        if custom_config:
            config.custom_config.update(custom_config)

        job = Job(
            name=name,
            job_type=job_type,
            priority=priority,
            parent_id=parent_id,
            tags=tags or [],
            config=config.to_dict(),
        )

        # Create cancellation token
        token = self._cancellation.create_token(job_id=job.id)

        with self._lock:
            self._jobs[job.id] = job
            self._progress[job.id] = JobProgress(job_id=job.id)

        job.mark_queued()
        logger.info(f"Job submitted: {job.name} ({job.id})")

        # Trigger callback so dashboard can display the queued job
        self._trigger_callbacks("on_job_queued", job)

        return job

    def start_job(self, job_id: UUID) -> bool:
        """
        Start a queued job.

        Parameters
        ----------
        job_id : UUID
            Job to start

        Returns
        -------
        bool
            True if job was started
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.error(f"Job not found: {job_id}")
            return False

        if job.state != JobState.QUEUED:
            logger.warning(f"Job {job_id} not in QUEUED state: {job.state}")
            return False

        job.mark_started()

        # Update progress
        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "running"
            progress.message = "Job started"

        # Trigger callbacks
        self._trigger_callbacks("on_job_started", job)

        logger.info(f"Job started: {job.name} ({job.id})")
        return True

    def update_progress(self, update: ProgressUpdate) -> None:
        """
        Update job progress from a worker.

        Parameters
        ----------
        update : ProgressUpdate
            Progress update from worker
        """
        progress = self._progress.get(update.job_id)
        if progress is None:
            logger.warning(f"No progress tracker for job {update.job_id}")
            return

        # Update or create worker progress
        worker = progress.get_worker(update.worker_id)
        if worker is None:
            worker = WorkerProgress(
                worker_id=update.worker_id,
                items_total=update.items_total,
            )
            progress.workers.append(worker)

        worker.items_processed = update.items_processed
        worker.items_total = update.items_total
        worker.current_item = update.current_item
        worker.last_update = update.timestamp
        if update.metrics:
            worker.metrics.update(update.metrics)
            # Also copy metrics to progress level for easy access
            progress.metrics.update(update.metrics)

        progress.update_from_workers()
        progress.message = update.message

        # Trigger callbacks
        self._trigger_callbacks("on_progress_update", progress)

    def complete_job(
        self,
        job_id: UUID,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a job as completed successfully.

        Parameters
        ----------
        job_id : UUID
            Job to complete
        result : dict, optional
            Job result data

        Returns
        -------
        bool
            True if job was completed
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job.mark_completed(result=result)

        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "completed"
            progress.overall_percent = 100.0
            progress.message = "Job completed successfully"

        # Save to history
        self._save_to_history(job)

        self._trigger_callbacks("on_job_completed", job)
        logger.info(f"Job completed: {job.name} ({job.id})")
        return True

    def fail_job(
        self,
        job_id: UUID,
        error: str,
        traceback: Optional[str] = None,
    ) -> bool:
        """
        Mark a job as failed.

        Parameters
        ----------
        job_id : UUID
            Job that failed
        error : str
            Error message
        traceback : str, optional
            Full traceback

        Returns
        -------
        bool
            True if job was marked failed
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job.mark_failed(error=error, traceback=traceback)

        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "failed"
            progress.message = f"Error: {error}"

        # Save to history
        self._save_to_history(job)

        self._trigger_callbacks("on_job_failed", job)
        logger.error(f"Job failed: {job.name} ({job.id}): {error}")
        return True

    def cancel_job(
        self,
        job_id: UUID,
        reason: CancellationReason = CancellationReason.USER_REQUESTED,
        message: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Cancel a running job.

        Parameters
        ----------
        job_id : UUID
            Job to cancel
        reason : CancellationReason
            Why cancellation is requested
        message : str, optional
            Human-readable message
        force : bool
            Force immediate cancellation

        Returns
        -------
        bool
            True if cancellation was initiated
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        if not job.can_cancel:
            logger.warning(f"Job {job_id} cannot be cancelled: {job.state}")
            return False

        # Mark job as cancelling
        job.mark_cancelling()

        # Signal cancellation through token
        self._cancellation.cancel_job(
            job_id=job_id,
            reason=reason,
            message=message,
            force=force,
        )

        # Cancel Ray task if active
        ray_ref = self._ray_refs.get(job_id)
        if ray_ref is not None and is_ray_initialized():
            try:
                ray = _get_ray()
                ray.cancel(ray_ref, force=force)
            except Exception as e:
                logger.warning(f"Error cancelling Ray task: {e}")

        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "cancelling"
            progress.message = message or "Cancellation requested"

        logger.info(f"Cancellation initiated for job: {job.name} ({job.id})")
        return True

    def finalize_cancellation(self, job_id: UUID) -> bool:
        """
        Finalize a job cancellation after cleanup.

        Called when worker has finished cleanup after cancellation.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job.mark_cancelled()

        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "cancelled"
            progress.message = "Job cancelled"

        # Save to history
        self._save_to_history(job)

        self._trigger_callbacks("on_job_cancelled", job)
        logger.info(f"Job cancelled: {job.name} ({job.id})")
        return True

    def _save_to_history(self, job: Job) -> None:
        """
        Save a terminal job to history storage.

        Parameters
        ----------
        job : Job
            Job to save (should be in terminal state)
        """
        if self._history is None:
            return

        try:
            self._history.save_job(job)
            logger.debug(f"Saved job {job.id} to history")
        except Exception as e:
            logger.warning(f"Failed to save job {job.id} to history: {e}")

    def pause_job(self, job_id: UUID) -> bool:
        """Pause a running job."""
        job = self._jobs.get(job_id)
        if job is None or not job.can_pause:
            return False

        job.mark_paused()
        self._cancellation.pause_job(job_id)

        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "paused"
            progress.message = "Job paused"

        logger.info(f"Job paused: {job.name} ({job.id})")
        return True

    def resume_job(self, job_id: UUID) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if job is None or not job.can_resume:
            return False

        job.mark_resumed()
        self._cancellation.resume_job(job_id)

        progress = self._progress.get(job_id)
        if progress:
            progress.phase = "running"
            progress.message = "Job resumed"

        logger.info(f"Job resumed: {job.name} ({job.id})")
        return True

    def get_job(self, job_id: UUID) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_progress(self, job_id: UUID) -> Optional[JobProgress]:
        """Get progress for a job."""
        return self._progress.get(job_id)

    def get_cancellation_token(self, job_id: UUID) -> Optional[CancellationToken]:
        """Get cancellation token for a job."""
        return self._cancellation.get_job_token(job_id)

    def list_jobs(
        self,
        states: Optional[List[JobState]] = None,
        job_types: Optional[List[JobType]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Parameters
        ----------
        states : list, optional
            Filter by job states
        job_types : list, optional
            Filter by job types
        tags : list, optional
            Filter by tags (any match)

        Returns
        -------
        list
            Matching jobs
        """
        jobs = list(self._jobs.values())

        if states is not None:
            jobs = [j for j in jobs if j.state in states]

        if job_types is not None:
            jobs = [j for j in jobs if j.job_type in job_types]

        if tags is not None:
            jobs = [j for j in jobs if any(t in j.tags for t in tags)]

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get_active_jobs(self) -> List[Job]:
        """Get all active (running) jobs."""
        return self.list_jobs(states=[JobState.RUNNING, JobState.CANCELLING])

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """
        Register a callback for job events.

        Parameters
        ----------
        event : str
            Event name: on_job_started, on_job_completed,
            on_job_failed, on_job_cancelled, on_progress_update
        callback : callable
            Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, data: Any) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")

    def cleanup(self) -> int:
        """
        Clean up completed/cancelled jobs.

        Returns
        -------
        int
            Number of jobs cleaned up
        """
        count = 0
        with self._lock:
            to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.is_terminal
            ]
            for job_id in to_remove:
                del self._jobs[job_id]
                self._progress.pop(job_id, None)
                self._ray_refs.pop(job_id, None)
                count += 1

        self._cancellation.cleanup_completed()
        return count

    # History access methods

    def get_job_history(
        self,
        limit: int = 50,
        states: Optional[List[JobState]] = None,
        job_types: Optional[List[JobType]] = None,
    ) -> List[Job]:
        """
        Get jobs from history.

        Parameters
        ----------
        limit : int
            Maximum jobs to return
        states : list, optional
            Filter by states
        job_types : list, optional
            Filter by job types

        Returns
        -------
        list
            List of historical jobs
        """
        if self._history is None:
            return []

        return self._history.query_jobs(
            states=states,
            job_types=job_types,
            limit=limit,
        )

    def get_history_statistics(self) -> Dict[str, Any]:
        """
        Get job history statistics.

        Returns
        -------
        dict
            Statistics from job history
        """
        if self._history is None:
            return {}

        return self._history.get_statistics()

    def get_recent_failures(self, limit: int = 10) -> List[Job]:
        """
        Get recent failed jobs from history.

        Parameters
        ----------
        limit : int
            Maximum jobs to return

        Returns
        -------
        list
            List of failed jobs
        """
        if self._history is None:
            return []

        return self._history.get_failed_jobs(limit=limit)

    @property
    def history_storage(self) -> Optional[JobHistoryStorage]:
        """Get the history storage instance."""
        return self._history

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        state_counts = {}
        for job in self._jobs.values():
            state_counts[job.state.name] = state_counts.get(job.state.name, 0) + 1

        return {
            "total_jobs": len(self._jobs),
            "state_counts": state_counts,
            "active_jobs": len(self.get_active_jobs()),
            "cancellation": self._cancellation.get_status(),
        }


# Module-level convenience functions

_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager."""
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
