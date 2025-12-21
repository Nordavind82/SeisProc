"""
Base Worker Actor for Ray-based Processing

Provides foundational worker actor with:
- Multi-level cancellation checking
- Progress reporting to job manager
- Pause/resume support
- Checkpoint integration
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any, Callable
from uuid import UUID

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker lifecycle states."""
    IDLE = auto()
    INITIALIZING = auto()
    PROCESSING = auto()
    PAUSED = auto()
    COMPLETING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class WorkerProgress:
    """Progress update from a worker."""
    worker_id: str
    job_id: UUID
    items_processed: int
    items_total: int
    current_item: Optional[str] = None
    elapsed_seconds: float = 0.0
    state: WorkerState = WorkerState.PROCESSING
    metrics: Optional[Dict[str, Any]] = None

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.items_total == 0:
            return 0.0
        return (self.items_processed / self.items_total) * 100.0

    @property
    def eta_seconds(self) -> float:
        """Estimate time remaining."""
        if self.items_processed == 0 or self.elapsed_seconds == 0:
            return 0.0
        rate = self.items_processed / self.elapsed_seconds
        remaining = self.items_total - self.items_processed
        return remaining / rate if rate > 0 else 0.0


class CancellationChecker:
    """
    Lightweight cancellation checker for workers.

    Checks a shared cancellation state without heavy dependencies.
    Used by workers to detect cancellation requests.
    """

    def __init__(self, job_id: UUID):
        self.job_id = job_id
        self._cancelled = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    @property
    def is_paused(self) -> bool:
        """Check if worker is paused."""
        return self._paused

    def cancel(self):
        """Signal cancellation."""
        self._cancelled = True
        # Also unpause so worker can exit
        self._pause_event.set()

    def pause(self):
        """Signal pause."""
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        """Signal resume."""
        self._paused = False
        self._pause_event.set()

    def wait_if_paused(self, timeout: float = 1.0) -> bool:
        """
        Wait if paused, with periodic timeout for cancellation checks.

        Returns True if we should continue, False if cancelled.
        """
        while self._paused and not self._cancelled:
            self._pause_event.wait(timeout=timeout)
        return not self._cancelled

    def raise_if_cancelled(self):
        """Raise exception if cancelled."""
        if self._cancelled:
            from ..cancellation import CancellationError, CancellationRequest, CancellationReason
            raise CancellationError(
                CancellationRequest(
                    reason=CancellationReason.USER_REQUESTED,
                    message=f"Job {self.job_id} cancelled"
                )
            )


class BaseWorkerActor(ABC):
    """
    Base class for Ray worker actors.

    Provides common functionality for all worker types:
    - State management
    - Progress reporting
    - Cancellation checking
    - Pause/resume handling
    - Error handling

    Subclasses implement the actual processing logic.

    Usage (as Ray actor)
    --------------------
    @ray.remote
    class MyWorkerActor(BaseWorkerActor):
        def process_item(self, item):
            # Implementation
            pass

    worker = MyWorkerActor.remote(job_id, worker_id)
    result = ray.get(worker.process.remote(items))
    """

    def __init__(
        self,
        job_id: UUID,
        worker_id: str,
        progress_callback: Optional[Callable[[WorkerProgress], None]] = None,
    ):
        """
        Initialize worker actor.

        Parameters
        ----------
        job_id : UUID
            Parent job identifier
        worker_id : str
            Unique worker identifier
        progress_callback : callable, optional
            Callback for progress updates
        """
        self.job_id = job_id
        self.worker_id = worker_id
        self._progress_callback = progress_callback

        self._state = WorkerState.IDLE
        self._cancellation = CancellationChecker(job_id)
        self._start_time: Optional[float] = None
        self._items_processed = 0
        self._items_total = 0
        self._last_progress_time = 0.0
        self._progress_interval = 0.5  # Report progress every 0.5 seconds

        logger.debug(f"Worker {worker_id} initialized for job {job_id}")

    @property
    def state(self) -> WorkerState:
        """Get current worker state."""
        return self._state

    @property
    def is_cancelled(self) -> bool:
        """Check if worker was cancelled."""
        return self._cancellation.is_cancelled

    @property
    def is_paused(self) -> bool:
        """Check if worker is paused."""
        return self._cancellation.is_paused

    def cancel(self):
        """Request worker cancellation."""
        logger.info(f"Worker {self.worker_id} cancellation requested")
        self._cancellation.cancel()
        self._state = WorkerState.CANCELLED

    def pause(self):
        """Pause worker processing."""
        logger.info(f"Worker {self.worker_id} pause requested")
        self._cancellation.pause()
        self._state = WorkerState.PAUSED

    def resume(self):
        """Resume worker processing."""
        logger.info(f"Worker {self.worker_id} resume requested")
        self._cancellation.resume()
        if self._state == WorkerState.PAUSED:
            self._state = WorkerState.PROCESSING

    def get_progress(self) -> WorkerProgress:
        """Get current progress."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        return WorkerProgress(
            worker_id=self.worker_id,
            job_id=self.job_id,
            items_processed=self._items_processed,
            items_total=self._items_total,
            elapsed_seconds=elapsed,
            state=self._state,
        )

    def _report_progress(self, force: bool = False):
        """Report progress if interval has elapsed."""
        now = time.time()
        if not force and (now - self._last_progress_time) < self._progress_interval:
            return

        self._last_progress_time = now
        progress = self.get_progress()

        if self._progress_callback:
            try:
                self._progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _check_cancellation(self):
        """Check for cancellation and raise if cancelled."""
        self._cancellation.raise_if_cancelled()

    def _wait_if_paused(self) -> bool:
        """Wait if paused. Returns False if cancelled during pause."""
        return self._cancellation.wait_if_paused()

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main processing entry point.

        Subclasses must implement this method.
        Should call _check_cancellation() periodically.
        """
        pass

    def _run_with_lifecycle(
        self,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Run processing with full lifecycle management.

        Handles:
        - State transitions
        - Error handling
        - Progress reporting
        - Cancellation checking

        Parameters
        ----------
        process_func : callable
            The actual processing function
        *args, **kwargs
            Arguments to pass to process_func
        """
        self._state = WorkerState.INITIALIZING
        self._start_time = time.time()

        try:
            # Check cancellation before starting
            self._check_cancellation()

            # Run the processing
            self._state = WorkerState.PROCESSING
            result = process_func(*args, **kwargs)

            # Complete
            self._state = WorkerState.COMPLETING
            self._report_progress(force=True)
            self._state = WorkerState.COMPLETED

            return result

        except Exception as e:
            # Check if it was a cancellation
            from ..cancellation import CancellationError
            if isinstance(e, CancellationError):
                self._state = WorkerState.CANCELLED
                logger.info(f"Worker {self.worker_id} cancelled")
                raise

            # Other error
            self._state = WorkerState.FAILED
            logger.error(f"Worker {self.worker_id} failed: {e}")
            raise

        finally:
            self._report_progress(force=True)
