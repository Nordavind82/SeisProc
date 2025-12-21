"""
Job Model

Defines the core Job dataclass and related enums for job management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4


class JobState(Enum):
    """Possible states of a job in its lifecycle."""
    CREATED = auto()      # Job created but not yet queued
    QUEUED = auto()       # Job queued for execution
    RUNNING = auto()      # Job currently executing
    PAUSED = auto()       # Job paused, can be resumed
    CANCELLING = auto()   # Cancellation requested, cleanup in progress
    CANCELLED = auto()    # Job cancelled by user
    COMPLETED = auto()    # Job completed successfully
    FAILED = auto()       # Job failed with error
    TIMEOUT = auto()      # Job timed out


class JobType(Enum):
    """Types of jobs supported by the system."""
    SEGY_IMPORT = auto()       # Import SEGY files
    SEGY_EXPORT = auto()       # Export SEGY files
    BATCH_PROCESS = auto()     # Batch processing workflow
    SINGLE_PROCESS = auto()    # Single processor execution
    QC_ANALYSIS = auto()       # QC analysis job
    MIGRATION = auto()         # Seismic migration
    VOLUME_BUILD = auto()      # Volume construction


class JobPriority(Enum):
    """Priority levels for job scheduling."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Job:
    """
    Represents a processing job with full lifecycle tracking.

    Attributes
    ----------
    id : UUID
        Unique job identifier
    name : str
        Human-readable job name
    job_type : JobType
        Type of job being executed
    state : JobState
        Current state of the job
    priority : JobPriority
        Job priority for scheduling
    created_at : datetime
        When the job was created
    started_at : datetime, optional
        When execution started
    completed_at : datetime, optional
        When execution completed
    error_message : str, optional
        Error message if job failed
    config : dict
        Job-specific configuration
    result : dict, optional
        Job result data
    parent_id : UUID, optional
        Parent job ID for sub-jobs
    tags : list
        Tags for job categorization
    """

    # Required fields
    name: str
    job_type: JobType

    # Auto-generated fields
    id: UUID = field(default_factory=uuid4)
    state: JobState = JobState.CREATED
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)

    # Lifecycle timestamps
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error handling
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Configuration and results
    config: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    # Relationships
    parent_id: Optional[UUID] = None
    child_ids: List[UUID] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Resource tracking
    ray_task_id: Optional[str] = None
    worker_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "job_type": self.job_type.name,
            "state": self.state.name,
            "priority": self.priority.name,
            "created_at": self.created_at.isoformat(),
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "config": self.config,
            "result": self.result,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "child_ids": [str(c) for c in self.child_ids],
            "tags": self.tags,
            "metadata": self.metadata,
            "ray_task_id": self.ray_task_id,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Deserialize job from dictionary."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            job_type=JobType[data["job_type"]],
            state=JobState[data["state"]],
            priority=JobPriority[data["priority"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            queued_at=datetime.fromisoformat(data["queued_at"]) if data.get("queued_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            paused_at=datetime.fromisoformat(data["paused_at"]) if data.get("paused_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            error_traceback=data.get("error_traceback"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            config=data.get("config", {}),
            result=data.get("result"),
            parent_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
            child_ids=[UUID(c) for c in data.get("child_ids", [])],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            ray_task_id=data.get("ray_task_id"),
            worker_id=data.get("worker_id"),
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.TIMEOUT,
        )

    @property
    def is_active(self) -> bool:
        """Check if job is actively executing."""
        return self.state in (JobState.RUNNING, JobState.CANCELLING)

    @property
    def can_cancel(self) -> bool:
        """Check if job can be cancelled."""
        return self.state in (
            JobState.CREATED,
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.PAUSED,
        )

    @property
    def can_pause(self) -> bool:
        """Check if job can be paused."""
        return self.state == JobState.RUNNING

    @property
    def can_resume(self) -> bool:
        """Check if job can be resumed."""
        return self.state == JobState.PAUSED

    def mark_queued(self) -> None:
        """Mark job as queued."""
        self.state = JobState.QUEUED
        self.queued_at = datetime.now()

    def mark_started(self, worker_id: Optional[str] = None) -> None:
        """Mark job as started."""
        self.state = JobState.RUNNING
        self.started_at = datetime.now()
        self.worker_id = worker_id

    def mark_paused(self) -> None:
        """Mark job as paused."""
        self.state = JobState.PAUSED
        self.paused_at = datetime.now()

    def mark_resumed(self) -> None:
        """Mark job as resumed."""
        self.state = JobState.RUNNING
        self.paused_at = None

    def mark_cancelling(self) -> None:
        """Mark job as being cancelled."""
        self.state = JobState.CANCELLING

    def mark_cancelled(self) -> None:
        """Mark job as cancelled."""
        self.state = JobState.CANCELLED
        self.completed_at = datetime.now()

    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as completed successfully."""
        self.state = JobState.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def mark_failed(self, error: str, traceback: Optional[str] = None) -> None:
        """Mark job as failed."""
        self.state = JobState.FAILED
        self.completed_at = datetime.now()
        self.error_message = error
        self.error_traceback = traceback
