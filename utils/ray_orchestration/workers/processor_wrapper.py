"""
Processor Wrapper for Ray Workers

Wraps existing BaseProcessor subclasses for use in Ray actors,
adding cancellation checking and progress callbacks.
"""

import logging
import time
from typing import Optional, Callable, Dict, Any
from uuid import UUID

import numpy as np

from .base_worker import CancellationChecker, WorkerProgress

logger = logging.getLogger(__name__)


class ProcessorWrapper:
    """
    Wraps a BaseProcessor with cancellation and progress support.

    This wrapper can be used to add cancellation checking to existing
    processors without modifying their code.

    Usage
    -----
    >>> from processors.denoise_3d import Denoise3DProcessor
    >>>
    >>> processor = Denoise3DProcessor(wavelet='db4', level=4)
    >>> wrapper = ProcessorWrapper(
    ...     processor=processor,
    ...     job_id=job.id,
    ...     worker_id='worker-0',
    ...     cancellation_checker=checker,
    ... )
    >>>
    >>> result = wrapper.process(gather_data)
    """

    def __init__(
        self,
        processor: Any,
        job_id: UUID,
        worker_id: str,
        cancellation_checker: Optional[CancellationChecker] = None,
        progress_callback: Optional[Callable[[WorkerProgress], None]] = None,
    ):
        """
        Initialize processor wrapper.

        Parameters
        ----------
        processor : BaseProcessor
            The processor to wrap
        job_id : UUID
            Job identifier
        worker_id : str
            Worker identifier
        cancellation_checker : CancellationChecker, optional
            Checker for cancellation state
        progress_callback : callable, optional
            Progress callback
        """
        self._processor = processor
        self._job_id = job_id
        self._worker_id = worker_id
        self._cancellation = cancellation_checker
        self._progress_callback = progress_callback

        self._items_processed = 0
        self._items_total = 0
        self._start_time: Optional[float] = None

    @property
    def processor(self):
        """Get the wrapped processor."""
        return self._processor

    def set_total_items(self, total: int):
        """Set total items for progress tracking."""
        self._items_total = total

    def process(self, gather_data: Any) -> Any:
        """
        Process a single gather with cancellation checking.

        Parameters
        ----------
        gather_data : SeismicData
            Input gather data

        Returns
        -------
        SeismicData
            Processed gather data
        """
        if self._start_time is None:
            self._start_time = time.time()

        # Check cancellation before processing
        if self._cancellation:
            self._cancellation.raise_if_cancelled()

            # Wait if paused
            if not self._cancellation.wait_if_paused():
                self._cancellation.raise_if_cancelled()

        # Process the gather
        result = self._processor.process(gather_data)

        # Update progress
        self._items_processed += 1
        self._report_progress()

        # Check cancellation after processing
        if self._cancellation:
            self._cancellation.raise_if_cancelled()

        return result

    def _report_progress(self):
        """Report progress if callback is set."""
        if self._progress_callback is None:
            return

        elapsed = time.time() - self._start_time if self._start_time else 0

        progress = WorkerProgress(
            worker_id=self._worker_id,
            job_id=self._job_id,
            items_processed=self._items_processed,
            items_total=self._items_total,
            elapsed_seconds=elapsed,
        )

        try:
            self._progress_callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")

    def get_description(self) -> str:
        """Get processor description."""
        return self._processor.get_description()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize processor configuration."""
        return self._processor.to_dict()

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ProcessorWrapper':
        """
        Create wrapper from configuration.

        Note: This only recreates the processor, not the wrapper settings.
        """
        from processors.base_processor import BaseProcessor
        processor = BaseProcessor.from_dict(config)
        # Wrapper settings must be provided separately
        return cls(
            processor=processor,
            job_id=UUID(int=0),  # Placeholder
            worker_id='',
        )


def wrap_processor(
    processor: Any,
    job_id: UUID,
    worker_id: str,
    cancellation_checker: Optional[CancellationChecker] = None,
    progress_callback: Optional[Callable] = None,
) -> ProcessorWrapper:
    """
    Convenience function to wrap a processor.

    Parameters
    ----------
    processor : BaseProcessor
        Processor to wrap
    job_id : UUID
        Job identifier
    worker_id : str
        Worker identifier
    cancellation_checker : CancellationChecker, optional
        Cancellation checker
    progress_callback : callable, optional
        Progress callback

    Returns
    -------
    ProcessorWrapper
        Wrapped processor
    """
    return ProcessorWrapper(
        processor=processor,
        job_id=job_id,
        worker_id=worker_id,
        cancellation_checker=cancellation_checker,
        progress_callback=progress_callback,
    )
