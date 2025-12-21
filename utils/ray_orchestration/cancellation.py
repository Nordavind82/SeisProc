"""
Multi-Level Cancellation System

Provides cancellation tokens that propagate across:
- Ray tasks (actor-based cancellation)
- Python threads (threading.Event)
- Future: Rust/PyO3 (AtomicBool)
- Future: Metal kernels (command buffer cancellation)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Union
from uuid import UUID, uuid4
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CancellationReason(Enum):
    """Reasons for cancellation."""
    USER_REQUESTED = auto()
    TIMEOUT = auto()
    ERROR = auto()
    PARENT_CANCELLED = auto()
    SYSTEM_SHUTDOWN = auto()
    RESOURCE_EXHAUSTED = auto()


@dataclass
class CancellationRequest:
    """Details about a cancellation request."""
    reason: CancellationReason
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    force: bool = False


class CancellationToken:
    """
    Thread-safe cancellation token with hierarchical support.

    This token can be checked by workers at any level to determine
    if they should stop processing. Supports parent-child relationships
    for cascading cancellation.

    Usage
    -----
    >>> token = CancellationToken()
    >>> # In worker:
    >>> while not token.is_cancelled:
    ...     process_item()
    >>> # From coordinator:
    >>> token.cancel(CancellationReason.USER_REQUESTED)
    """

    def __init__(
        self,
        parent: Optional['CancellationToken'] = None,
        token_id: Optional[UUID] = None,
    ):
        self._id = token_id or uuid4()
        self._cancelled = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._lock = threading.Lock()
        self._request: Optional[CancellationRequest] = None
        self._callbacks: List[Callable[['CancellationToken'], None]] = []
        self._parent = parent
        self._children: List['CancellationToken'] = []

        # Register with parent
        if parent is not None:
            parent._register_child(self)

    @property
    def id(self) -> UUID:
        """Get token ID."""
        return self._id

    @property
    def is_cancelled(self) -> bool:
        """
        Check if cancellation has been requested.

        Also checks parent token if one exists.
        """
        if self._cancelled.is_set():
            return True
        if self._parent is not None:
            return self._parent.is_cancelled
        return False

    @property
    def is_paused(self) -> bool:
        """Check if processing should be paused."""
        return not self._pause_event.is_set()

    @property
    def cancellation_request(self) -> Optional[CancellationRequest]:
        """Get cancellation request details."""
        return self._request

    def cancel(
        self,
        reason: CancellationReason = CancellationReason.USER_REQUESTED,
        message: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Request cancellation.

        Parameters
        ----------
        reason : CancellationReason
            Why cancellation is being requested
        message : str, optional
            Human-readable message
        force : bool
            Force immediate cancellation without cleanup
        """
        with self._lock:
            if self._cancelled.is_set():
                return  # Already cancelled

            self._request = CancellationRequest(
                reason=reason,
                message=message,
                force=force,
            )
            self._cancelled.set()
            self._pause_event.set()  # Unpause to allow cleanup

            logger.info(
                f"Cancellation requested for token {self._id}: "
                f"{reason.name} - {message or 'No message'}"
            )

        # Propagate to children
        for child in self._children:
            child.cancel(
                reason=CancellationReason.PARENT_CANCELLED,
                message=f"Parent token {self._id} cancelled: {message}",
                force=force,
            )

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in cancellation callback: {e}")

    def pause(self) -> None:
        """Request pause (workers should wait)."""
        self._pause_event.clear()
        logger.info(f"Pause requested for token {self._id}")

    def resume(self) -> None:
        """Resume from pause."""
        self._pause_event.set()
        logger.info(f"Resume requested for token {self._id}")

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """
        Block if paused, return immediately if not.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait

        Returns
        -------
        bool
            True if resumed, False if timed out or cancelled
        """
        if self.is_cancelled:
            return False
        return self._pause_event.wait(timeout)

    def check_cancelled(self, interval: float = 0.1) -> bool:
        """
        Check for cancellation with brief pause for responsiveness.

        Parameters
        ----------
        interval : float
            Time to wait between checks

        Returns
        -------
        bool
            True if cancelled
        """
        if self.is_cancelled:
            return True
        # Brief pause to allow cancellation signals
        time.sleep(interval)
        return self.is_cancelled

    def raise_if_cancelled(self) -> None:
        """Raise CancellationError if cancelled."""
        if self.is_cancelled:
            raise CancellationError(self._request)

    def on_cancel(self, callback: Callable[['CancellationToken'], None]) -> None:
        """
        Register a callback for when cancellation is requested.

        Parameters
        ----------
        callback : callable
            Function to call with the token when cancelled
        """
        with self._lock:
            if self._cancelled.is_set():
                # Already cancelled, call immediately
                callback(self)
            else:
                self._callbacks.append(callback)

    def create_child(self) -> 'CancellationToken':
        """
        Create a child token that will be cancelled when this token is.

        Returns
        -------
        CancellationToken
            Child token linked to this parent
        """
        return CancellationToken(parent=self)

    def _register_child(self, child: 'CancellationToken') -> None:
        """Register a child token."""
        with self._lock:
            self._children.append(child)
            # If already cancelled, cancel child immediately
            if self._cancelled.is_set():
                child.cancel(
                    reason=CancellationReason.PARENT_CANCELLED,
                    message=f"Parent token {self._id} was already cancelled",
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize token state."""
        return {
            "id": str(self._id),
            "is_cancelled": self.is_cancelled,
            "is_paused": self.is_paused,
            "request": {
                "reason": self._request.reason.name,
                "message": self._request.message,
                "timestamp": self._request.timestamp.isoformat(),
                "force": self._request.force,
            } if self._request else None,
            "child_count": len(self._children),
        }


class CancellationError(Exception):
    """Exception raised when operation is cancelled."""

    def __init__(self, request_or_message: Optional[Union[CancellationRequest, str]] = None):
        # Handle both CancellationRequest objects and plain string messages
        if isinstance(request_or_message, str):
            self.request = None
            message = f"Operation cancelled: {request_or_message}"
        elif request_or_message is not None:
            self.request = request_or_message
            message = f"Operation cancelled: {request_or_message.reason.name}"
            if request_or_message.message:
                message += f" - {request_or_message.message}"
        else:
            self.request = None
            message = "Operation cancelled"
        super().__init__(message)


class CancellationCoordinator:
    """
    Manages cancellation tokens for all active jobs.

    Provides centralized control for:
    - Creating tokens for new jobs
    - Cancelling specific jobs
    - Cancelling all jobs (system shutdown)
    - Tracking active tokens

    Thread-safe singleton pattern.
    """

    _instance: Optional['CancellationCoordinator'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'CancellationCoordinator':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tokens: Dict[UUID, CancellationToken] = {}
                cls._instance._job_tokens: Dict[UUID, UUID] = {}  # job_id -> token_id
            return cls._instance

    def create_token(
        self,
        job_id: Optional[UUID] = None,
        parent_token_id: Optional[UUID] = None,
    ) -> CancellationToken:
        """
        Create a new cancellation token.

        Parameters
        ----------
        job_id : UUID, optional
            Job to associate with this token
        parent_token_id : UUID, optional
            Parent token for hierarchical cancellation

        Returns
        -------
        CancellationToken
            New cancellation token
        """
        parent = None
        if parent_token_id is not None:
            parent = self._tokens.get(parent_token_id)

        token = CancellationToken(parent=parent)

        with self._lock:
            self._tokens[token.id] = token
            if job_id is not None:
                self._job_tokens[job_id] = token.id

        logger.debug(f"Created cancellation token {token.id} for job {job_id}")
        return token

    def get_token(self, token_id: UUID) -> Optional[CancellationToken]:
        """Get token by ID."""
        return self._tokens.get(token_id)

    def get_job_token(self, job_id: UUID) -> Optional[CancellationToken]:
        """Get token for a job."""
        token_id = self._job_tokens.get(job_id)
        if token_id is None:
            return None
        return self._tokens.get(token_id)

    def cancel_job(
        self,
        job_id: UUID,
        reason: CancellationReason = CancellationReason.USER_REQUESTED,
        message: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Cancel a specific job.

        Parameters
        ----------
        job_id : UUID
            Job to cancel
        reason : CancellationReason
            Cancellation reason
        message : str, optional
            Message for logging
        force : bool
            Force immediate cancellation

        Returns
        -------
        bool
            True if job was found and cancelled
        """
        token = self.get_job_token(job_id)
        if token is None:
            logger.warning(f"No cancellation token found for job {job_id}")
            return False

        token.cancel(reason=reason, message=message, force=force)
        return True

    def pause_job(self, job_id: UUID) -> bool:
        """Pause a specific job."""
        token = self.get_job_token(job_id)
        if token is None:
            return False
        token.pause()
        return True

    def resume_job(self, job_id: UUID) -> bool:
        """Resume a paused job."""
        token = self.get_job_token(job_id)
        if token is None:
            return False
        token.resume()
        return True

    def cancel_all(
        self,
        reason: CancellationReason = CancellationReason.SYSTEM_SHUTDOWN,
        message: Optional[str] = None,
    ) -> int:
        """
        Cancel all active jobs.

        Returns
        -------
        int
            Number of jobs cancelled
        """
        count = 0
        with self._lock:
            for token in self._tokens.values():
                if not token.is_cancelled:
                    token.cancel(reason=reason, message=message)
                    count += 1

        logger.info(f"Cancelled {count} jobs: {reason.name}")
        return count

    def cleanup_completed(self) -> int:
        """
        Remove tokens for completed jobs.

        Returns
        -------
        int
            Number of tokens removed
        """
        with self._lock:
            # Find completed job tokens
            completed = [
                (job_id, token_id)
                for job_id, token_id in self._job_tokens.items()
                if token_id in self._tokens
                and self._tokens[token_id].is_cancelled
            ]

            for job_id, token_id in completed:
                del self._job_tokens[job_id]
                del self._tokens[token_id]

            return len(completed)

    @property
    def active_count(self) -> int:
        """Get number of active (non-cancelled) tokens."""
        return sum(1 for t in self._tokens.values() if not t.is_cancelled)

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "total_tokens": len(self._tokens),
            "active_tokens": self.active_count,
            "job_tokens": len(self._job_tokens),
        }


# Module-level convenience functions

_coordinator: Optional[CancellationCoordinator] = None


def get_cancellation_coordinator() -> CancellationCoordinator:
    """Get the global cancellation coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = CancellationCoordinator()
    return _coordinator


def create_cancellation_token(
    job_id: Optional[UUID] = None,
) -> CancellationToken:
    """Create a new cancellation token."""
    return get_cancellation_coordinator().create_token(job_id=job_id)


def cancel_job(job_id: UUID, reason: CancellationReason = CancellationReason.USER_REQUESTED) -> bool:
    """Cancel a specific job."""
    return get_cancellation_coordinator().cancel_job(job_id, reason=reason)


def cancel_all_jobs() -> int:
    """Cancel all jobs."""
    return get_cancellation_coordinator().cancel_all()
