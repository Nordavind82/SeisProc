"""
Tests for Ray Worker Actors

Tests worker actors, cancellation checking, and progress reporting.
"""

import pytest
import time
import threading
from uuid import uuid4


class TestCancellationChecker:
    """Tests for CancellationChecker."""

    def test_initial_state(self):
        """Test initial state is not cancelled or paused."""
        from utils.ray_orchestration.workers.base_worker import CancellationChecker

        checker = CancellationChecker(uuid4())

        assert not checker.is_cancelled
        assert not checker.is_paused

    def test_cancel(self):
        """Test cancellation."""
        from utils.ray_orchestration.workers.base_worker import CancellationChecker

        checker = CancellationChecker(uuid4())
        checker.cancel()

        assert checker.is_cancelled

    def test_pause_resume(self):
        """Test pause and resume."""
        from utils.ray_orchestration.workers.base_worker import CancellationChecker

        checker = CancellationChecker(uuid4())

        # Initial state
        assert not checker.is_paused

        # Pause
        checker.pause()
        assert checker.is_paused

        # Resume
        checker.resume()
        assert not checker.is_paused

    def test_wait_if_paused_not_paused(self):
        """Test wait_if_paused when not paused."""
        from utils.ray_orchestration.workers.base_worker import CancellationChecker

        checker = CancellationChecker(uuid4())

        # Should return immediately
        result = checker.wait_if_paused(timeout=0.1)
        assert result is True

    def test_wait_if_paused_cancelled(self):
        """Test wait_if_paused returns False when cancelled."""
        from utils.ray_orchestration.workers.base_worker import CancellationChecker

        checker = CancellationChecker(uuid4())
        checker.pause()
        checker.cancel()  # Also unpauses

        result = checker.wait_if_paused(timeout=0.1)
        assert result is False

    def test_raise_if_cancelled(self):
        """Test raise_if_cancelled."""
        from utils.ray_orchestration.workers.base_worker import CancellationChecker
        from utils.ray_orchestration.cancellation import CancellationError

        checker = CancellationChecker(uuid4())

        # Should not raise when not cancelled
        checker.raise_if_cancelled()

        # Should raise when cancelled
        checker.cancel()
        with pytest.raises(CancellationError):
            checker.raise_if_cancelled()


class TestWorkerProgress:
    """Tests for WorkerProgress."""

    def test_progress_creation(self):
        """Test creating a progress update."""
        from utils.ray_orchestration.workers.base_worker import WorkerProgress, WorkerState

        job_id = uuid4()
        progress = WorkerProgress(
            worker_id='worker-0',
            job_id=job_id,
            items_processed=50,
            items_total=100,
            elapsed_seconds=10.0,
            state=WorkerState.PROCESSING,
        )

        assert progress.worker_id == 'worker-0'
        assert progress.items_processed == 50
        assert progress.items_total == 100

    def test_percent_complete(self):
        """Test percent calculation."""
        from utils.ray_orchestration.workers.base_worker import WorkerProgress

        job_id = uuid4()

        # Normal case
        progress = WorkerProgress(
            worker_id='worker-0',
            job_id=job_id,
            items_processed=25,
            items_total=100,
        )
        assert progress.percent_complete == 25.0

        # Zero total
        progress_zero = WorkerProgress(
            worker_id='worker-0',
            job_id=job_id,
            items_processed=0,
            items_total=0,
        )
        assert progress_zero.percent_complete == 0.0

    def test_eta_calculation(self):
        """Test ETA calculation."""
        from utils.ray_orchestration.workers.base_worker import WorkerProgress

        job_id = uuid4()

        progress = WorkerProgress(
            worker_id='worker-0',
            job_id=job_id,
            items_processed=50,
            items_total=100,
            elapsed_seconds=10.0,
        )

        # 50 items in 10 seconds = 5 items/sec
        # 50 remaining / 5 items/sec = 10 seconds ETA
        assert progress.eta_seconds == 10.0


class TestCPUWorkerActorFactory:
    """Tests for create_cpu_worker_actor factory function."""

    def test_factory_creates_ray_actor_class(self):
        """Test that create_cpu_worker_actor returns a Ray actor class."""
        pytest.importorskip('ray')

        from utils.ray_orchestration.workers.cpu_worker import create_cpu_worker_actor

        actor_class = create_cpu_worker_actor()

        # Should be a Ray remote class
        assert hasattr(actor_class, 'remote')

    def test_worker_result_dataclass(self):
        """Test WorkerResult dataclass."""
        from utils.ray_orchestration.workers.cpu_worker import WorkerResult

        result = WorkerResult(
            worker_id='worker-0',
            n_gathers_processed=10,
            n_traces_processed=1000,
            elapsed_seconds=5.0,
            success=True,
        )

        assert result.worker_id == 'worker-0'
        assert result.n_gathers_processed == 10
        assert result.n_traces_processed == 1000
        assert result.success is True

    def test_gather_result_dataclass(self):
        """Test GatherResult dataclass."""
        from utils.ray_orchestration.workers.cpu_worker import GatherResult

        result = GatherResult(
            gather_idx=5,
            n_traces=100,
            elapsed_seconds=0.5,
            success=True,
        )

        assert result.gather_idx == 5
        assert result.n_traces == 100
        assert result.success is True


class TestProcessorWrapper:
    """Tests for ProcessorWrapper."""

    def test_wrapper_creation(self):
        """Test creating a processor wrapper."""
        from utils.ray_orchestration.workers.processor_wrapper import ProcessorWrapper

        # Create a mock processor
        class MockProcessor:
            def process(self, data):
                return data

            def get_description(self):
                return "Mock Processor"

            def to_dict(self):
                return {'class_name': 'MockProcessor'}

        processor = MockProcessor()
        job_id = uuid4()

        wrapper = ProcessorWrapper(
            processor=processor,
            job_id=job_id,
            worker_id='worker-0',
        )

        assert wrapper.processor == processor
        assert wrapper.get_description() == "Mock Processor"

    def test_wrapper_with_cancellation(self):
        """Test wrapper with cancellation checker."""
        from utils.ray_orchestration.workers.processor_wrapper import ProcessorWrapper
        from utils.ray_orchestration.workers.base_worker import CancellationChecker
        from utils.ray_orchestration.cancellation import CancellationError

        class MockProcessor:
            def process(self, data):
                return data

            def get_description(self):
                return "Mock"

            def to_dict(self):
                return {}

        processor = MockProcessor()
        job_id = uuid4()
        checker = CancellationChecker(job_id)

        wrapper = ProcessorWrapper(
            processor=processor,
            job_id=job_id,
            worker_id='worker-0',
            cancellation_checker=checker,
        )

        # Cancel and try to process
        checker.cancel()

        with pytest.raises(CancellationError):
            wrapper.process("test data")


class TestWrapProcessor:
    """Tests for wrap_processor convenience function."""

    def test_wrap_processor(self):
        """Test wrap_processor function."""
        from utils.ray_orchestration.workers.processor_wrapper import wrap_processor

        class MockProcessor:
            def process(self, data):
                return data

            def get_description(self):
                return "Mock"

            def to_dict(self):
                return {}

        processor = MockProcessor()
        job_id = uuid4()

        wrapper = wrap_processor(
            processor=processor,
            job_id=job_id,
            worker_id='worker-0',
        )

        assert wrapper.processor == processor
