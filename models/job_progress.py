"""
Job Progress Models

Defines progress tracking for jobs and workers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID


@dataclass
class WorkerProgress:
    """
    Progress information for a single worker.

    Attributes
    ----------
    worker_id : str
        Unique worker identifier
    items_processed : int
        Number of items processed
    items_total : int
        Total items to process
    current_item : str, optional
        Description of current item being processed
    started_at : datetime
        When worker started
    last_update : datetime
        Last progress update time
    error : str, optional
        Error message if worker failed
    """

    worker_id: str
    items_total: int
    items_processed: int = 0
    current_item: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.items_total == 0:
            return 0.0
        return (self.items_processed / self.items_total) * 100.0

    @property
    def items_remaining(self) -> int:
        """Get number of items remaining."""
        return max(0, self.items_total - self.items_processed)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def items_per_second(self) -> float:
        """Get processing rate."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.items_processed / elapsed

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        rate = self.items_per_second
        if rate == 0:
            return None
        return self.items_remaining / rate

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "worker_id": self.worker_id,
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "progress_percent": self.progress_percent,
            "current_item": self.current_item,
            "started_at": self.started_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            "error": self.error,
            "metrics": self.metrics,
            "items_per_second": self.items_per_second,
            "eta_seconds": self.eta_seconds,
        }


@dataclass
class JobProgress:
    """
    Aggregate progress information for a job.

    Attributes
    ----------
    job_id : UUID
        Job identifier
    phase : str
        Current phase of execution
    overall_percent : float
        Overall progress percentage (0-100)
    workers : list
        Progress for individual workers
    started_at : datetime
        When job started
    last_update : datetime
        Last progress update
    message : str, optional
        Current status message
    """

    job_id: UUID
    phase: str = "initializing"
    overall_percent: float = 0.0
    workers: List[WorkerProgress] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    message: Optional[str] = None
    checkpoint_file: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_items_processed(self) -> int:
        """Get total items processed across all workers."""
        return sum(w.items_processed for w in self.workers)

    @property
    def total_items(self) -> int:
        """Get total items across all workers."""
        return sum(w.items_total for w in self.workers)

    @property
    def active_workers(self) -> int:
        """Get number of active workers."""
        return sum(1 for w in self.workers if w.error is None)

    @property
    def failed_workers(self) -> int:
        """Get number of failed workers."""
        return sum(1 for w in self.workers if w.error is not None)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def aggregate_rate(self) -> float:
        """Get aggregate processing rate across all workers."""
        return sum(w.items_per_second for w in self.workers)

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining."""
        rate = self.aggregate_rate
        if rate == 0:
            return None
        remaining = self.total_items - self.total_items_processed
        return remaining / rate

    def update_from_workers(self) -> None:
        """Recalculate overall progress from worker progress."""
        if self.total_items == 0:
            self.overall_percent = 0.0
        else:
            self.overall_percent = (self.total_items_processed / self.total_items) * 100.0
        self.last_update = datetime.now()

    def get_worker(self, worker_id: str) -> Optional[WorkerProgress]:
        """Get worker progress by ID."""
        for w in self.workers:
            if w.worker_id == worker_id:
                return w
        return None

    def add_worker(self, worker: WorkerProgress) -> None:
        """Add or update worker progress."""
        existing = self.get_worker(worker.worker_id)
        if existing:
            self.workers.remove(existing)
        self.workers.append(worker)
        self.update_from_workers()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": str(self.job_id),
            "phase": self.phase,
            "overall_percent": self.overall_percent,
            "workers": [w.to_dict() for w in self.workers],
            "started_at": self.started_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            "message": self.message,
            "checkpoint_file": self.checkpoint_file,
            "metrics": self.metrics,
            "total_items_processed": self.total_items_processed,
            "total_items": self.total_items,
            "active_workers": self.active_workers,
            "failed_workers": self.failed_workers,
            "elapsed_seconds": self.elapsed_seconds,
            "aggregate_rate": self.aggregate_rate,
            "eta_seconds": self.eta_seconds,
        }


@dataclass
class ProgressUpdate:
    """
    A single progress update message.

    Used for sending progress updates through queues/signals.
    """

    job_id: UUID
    worker_id: str
    items_processed: int
    items_total: int
    message: Optional[str] = None
    current_item: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": str(self.job_id),
            "worker_id": self.worker_id,
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "message": self.message,
            "current_item": self.current_item,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }
