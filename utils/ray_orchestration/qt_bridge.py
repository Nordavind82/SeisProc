"""
PyQt6 Signal Bridge for Job Manager

Provides thread-safe connection between Ray job execution and PyQt6 UI.
Uses QThread and signals to ensure UI updates happen on the main thread.
"""

import logging
from typing import Optional, Dict, Any
from uuid import UUID

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
from PyQt6.QtWidgets import QApplication

from models.job import Job, JobState
from models.job_progress import JobProgress
from .job_manager import JobManager, get_job_manager

logger = logging.getLogger(__name__)


class JobSignalEmitter(QObject):
    """
    Qt signal emitter for job events.

    Thread-safe: Can be called from any thread, signals are
    delivered on the main Qt thread.

    Signals
    -------
    job_created : UUID, dict
        Emitted when a new job is created
    job_queued : UUID, dict
        Emitted when a job is queued
    job_started : UUID, dict
        Emitted when a job starts executing
    job_progress : UUID, dict
        Emitted on progress updates
    job_state_changed : UUID, str
        Emitted when job state changes
    job_completed : UUID, dict
        Emitted when a job completes successfully
    job_failed : UUID, dict
        Emitted when a job fails
    job_cancelled : UUID
        Emitted when a job is cancelled
    """

    job_created = pyqtSignal(object, dict)      # UUID, job_info
    job_queued = pyqtSignal(object, dict)       # UUID, job_info
    job_started = pyqtSignal(object, dict)      # UUID, job_info
    job_progress = pyqtSignal(object, dict)     # UUID, progress_info
    job_state_changed = pyqtSignal(object, str) # UUID, state
    job_completed = pyqtSignal(object, dict)    # UUID, result
    job_failed = pyqtSignal(object, dict)       # UUID, error_info
    job_cancelled = pyqtSignal(object)          # UUID

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_job_created(self, job: Job):
        """Emit job created signal."""
        self.job_created.emit(job.id, {
            "name": job.name,
            "job_type": job.job_type.name,
            "priority": job.priority.name,
        })

    def emit_job_queued(self, job: Job):
        """Emit job queued signal."""
        self.job_queued.emit(job.id, {
            "name": job.name,
            "job_type": job.job_type.name,
            "priority": job.priority.name,
        })

    def emit_job_started(self, job: Job):
        """Emit job started signal."""
        logger.info(f"[QT_BRIDGE] emit_job_started: job_id={job.id}, name={job.name}")
        self.job_started.emit(job.id, {
            "name": job.name,
            "job_type": job.job_type.name,
            "started_at": job.started_at.isoformat() if job.started_at else None,
        })

    def emit_progress(self, progress: JobProgress):
        """Emit progress update signal with all available statistics."""
        # Calculate traces per second from aggregate rate
        traces_per_sec = progress.aggregate_rate if hasattr(progress, 'aggregate_rate') else 0

        # Extract extended metrics from progress.metrics if available
        metrics = progress.metrics if hasattr(progress, 'metrics') else {}

        self.job_progress.emit(progress.job_id, {
            "percent": progress.overall_percent,
            "message": progress.message,
            "phase": progress.phase,
            "eta_seconds": progress.eta_seconds,
            "workers": len(progress.workers),
            "active_workers": progress.active_workers,
            # Extended statistics for Dashboard display
            "current_gathers": metrics.get('current_gathers', 0),
            "total_gathers": metrics.get('total_gathers', 0),
            "current_traces": progress.total_items_processed,
            "total_traces": progress.total_items,
            "traces_per_sec": traces_per_sec,
            "compute_kernel": metrics.get('compute_kernel', ''),
        })

    def emit_state_changed(self, job: Job):
        """Emit state change signal."""
        self.job_state_changed.emit(job.id, job.state.name)

    def emit_job_completed(self, job: Job):
        """Emit job completed signal."""
        self.job_completed.emit(job.id, {
            "name": job.name,
            "result": job.result,
            "duration_seconds": job.duration_seconds,
        })

    def emit_job_failed(self, job: Job):
        """Emit job failed signal."""
        self.job_failed.emit(job.id, {
            "name": job.name,
            "error": job.error_message,
            "traceback": job.error_traceback,
        })

    def emit_job_cancelled(self, job: Job):
        """Emit job cancelled signal."""
        self.job_cancelled.emit(job.id)


class JobManagerBridge(QObject):
    """
    Bridge between JobManager and PyQt6 UI.

    Connects JobManager callbacks to Qt signals and provides
    a Qt-friendly API for job operations.

    Usage
    -----
    >>> bridge = JobManagerBridge()
    >>> bridge.signals.job_started.connect(dashboard.on_job_started)
    >>> bridge.signals.job_progress.connect(dashboard.on_progress_updated)
    >>> bridge.start()
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._manager = get_job_manager()
        self._signals = JobSignalEmitter(self)
        self._polling_timer: Optional[QTimer] = None
        self._running = False

    @property
    def signals(self) -> JobSignalEmitter:
        """Get the signal emitter."""
        return self._signals

    @property
    def manager(self) -> JobManager:
        """Get the underlying job manager."""
        return self._manager

    def start(self, poll_interval_ms: int = 100):
        """
        Start the bridge.

        Registers callbacks with JobManager for real-time updates.
        Note: Polling is disabled as direct callbacks provide sufficient updates
        and polling causes duplicate signals and excessive log spam.

        Parameters
        ----------
        poll_interval_ms : int
            Polling interval (unused - polling disabled)
        """
        if self._running:
            return

        # Register callbacks - these provide real-time progress updates
        self._manager.register_callback("on_job_queued", self._on_job_queued)
        self._manager.register_callback("on_job_started", self._on_job_started)
        self._manager.register_callback("on_job_completed", self._on_job_completed)
        self._manager.register_callback("on_job_failed", self._on_job_failed)
        self._manager.register_callback("on_job_cancelled", self._on_job_cancelled)
        self._manager.register_callback("on_progress_update", self._on_progress_update)

        # Polling disabled - direct callbacks are sufficient and polling causes:
        # 1. Duplicate progress updates (same data emitted twice)
        # 2. Excessive log spam
        # 3. Unnecessary CPU usage
        # If needed for fallback, uncomment with longer interval (e.g., 2000ms)
        # self._polling_timer = QTimer(self)
        # self._polling_timer.timeout.connect(self._poll_jobs)
        # self._polling_timer.start(poll_interval_ms)

        self._running = True
        logger.info("JobManagerBridge started")

    def stop(self):
        """Stop the bridge."""
        if not self._running:
            return

        if self._polling_timer:
            self._polling_timer.stop()
            self._polling_timer = None

        self._running = False
        logger.info("JobManagerBridge stopped")

    # Job operations (UI-friendly wrappers)

    def submit_job(self, name: str, job_type, **kwargs) -> UUID:
        """
        Submit a new job.

        Parameters
        ----------
        name : str
            Job name
        job_type : JobType
            Type of job
        **kwargs
            Additional job configuration

        Returns
        -------
        UUID
            Job ID
        """
        job = self._manager.submit_job(name=name, job_type=job_type, **kwargs)
        self._signals.emit_job_created(job)
        self._signals.emit_job_queued(job)
        return job.id

    def start_job(self, job_id: UUID) -> bool:
        """Start a queued job."""
        result = self._manager.start_job(job_id)
        if result:
            job = self._manager.get_job(job_id)
            if job:
                self._signals.emit_job_started(job)
        return result

    def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a job."""
        return self._manager.cancel_job(job_id)

    def pause_job(self, job_id: UUID) -> bool:
        """Pause a running job."""
        result = self._manager.pause_job(job_id)
        if result:
            job = self._manager.get_job(job_id)
            if job:
                self._signals.emit_state_changed(job)
        return result

    def resume_job(self, job_id: UUID) -> bool:
        """Resume a paused job."""
        result = self._manager.resume_job(job_id)
        if result:
            job = self._manager.get_job(job_id)
            if job:
                self._signals.emit_state_changed(job)
        return result

    def cancel_all_jobs(self) -> int:
        """Cancel all active jobs."""
        from .cancellation import cancel_all_jobs
        return cancel_all_jobs()

    def get_job(self, job_id: UUID) -> Optional[Job]:
        """Get a job by ID."""
        return self._manager.get_job(job_id)

    def get_progress(self, job_id: UUID) -> Optional[JobProgress]:
        """Get progress for a job."""
        return self._manager.get_progress(job_id)

    # Callbacks from JobManager

    def _on_job_queued(self, job: Job):
        """Handle job queued from manager."""
        self._signals.emit_job_queued(job)

    def _on_job_started(self, job: Job):
        """Handle job started from manager."""
        # Use invokeMethod for thread safety
        self._signals.emit_job_started(job)

    def _on_job_completed(self, job: Job):
        """Handle job completed from manager."""
        self._signals.emit_job_completed(job)
        self._signals.emit_state_changed(job)

    def _on_job_failed(self, job: Job):
        """Handle job failed from manager."""
        self._signals.emit_job_failed(job)
        self._signals.emit_state_changed(job)

    def _on_job_cancelled(self, job: Job):
        """Handle job cancelled from manager."""
        self._signals.emit_job_cancelled(job)
        self._signals.emit_state_changed(job)

    def _on_progress_update(self, progress: JobProgress):
        """Handle progress update from manager."""
        self._signals.emit_progress(progress)

    def _poll_jobs(self):
        """Poll for job updates (backup to callbacks)."""
        for job in self._manager.get_active_jobs():
            progress = self._manager.get_progress(job.id)
            if progress:
                self._signals.emit_progress(progress)


# Module-level singleton

_bridge: Optional[JobManagerBridge] = None


def get_job_bridge() -> JobManagerBridge:
    """Get the global JobManagerBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = JobManagerBridge()
    return _bridge


def connect_dashboard(dashboard):
    """
    Connect a JobDashboardWidget to the job bridge.

    Parameters
    ----------
    dashboard : JobDashboardWidget
        Dashboard to connect
    """
    bridge = get_job_bridge()

    # Connect signals from bridge to dashboard
    bridge.signals.job_started.connect(dashboard.on_job_started)
    bridge.signals.job_progress.connect(dashboard.on_progress_updated)
    bridge.signals.job_state_changed.connect(dashboard.on_job_state_changed)
    bridge.signals.job_queued.connect(dashboard.on_job_queued)

    # Connect signals from dashboard to bridge
    dashboard.cancel_job_requested.connect(bridge.cancel_job)
    dashboard.pause_job_requested.connect(bridge.pause_job)
    dashboard.resume_job_requested.connect(bridge.resume_job)
    dashboard.cancel_all_requested.connect(bridge.cancel_all_jobs)

    # Start the bridge
    bridge.start()

    logger.info("Dashboard connected to job bridge")
