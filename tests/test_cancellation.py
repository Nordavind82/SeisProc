"""
Tests for Multi-Level Cancellation System

Tests CancellationToken, CancellationCoordinator, and related functionality.
"""

import pytest
import threading
import time
from uuid import uuid4


class TestCancellationToken:
    """Tests for CancellationToken."""

    def test_token_creation(self):
        """Test creating a cancellation token."""
        from utils.ray_orchestration import CancellationToken

        token = CancellationToken()

        assert token.is_cancelled is False
        assert token.is_paused is False
        assert token.cancellation_request is None

    def test_token_cancellation(self):
        """Test cancelling a token."""
        from utils.ray_orchestration import CancellationToken, CancellationReason

        token = CancellationToken()
        token.cancel(reason=CancellationReason.USER_REQUESTED, message="Test cancel")

        assert token.is_cancelled is True
        assert token.cancellation_request is not None
        assert token.cancellation_request.reason == CancellationReason.USER_REQUESTED
        assert token.cancellation_request.message == "Test cancel"

    def test_token_pause_resume(self):
        """Test pausing and resuming a token."""
        from utils.ray_orchestration import CancellationToken

        token = CancellationToken()

        assert token.is_paused is False

        token.pause()
        assert token.is_paused is True

        token.resume()
        assert token.is_paused is False

    def test_token_wait_if_paused(self):
        """Test waiting when paused."""
        from utils.ray_orchestration import CancellationToken

        token = CancellationToken()

        # Should return immediately when not paused
        result = token.wait_if_paused(timeout=0.1)
        assert result is True

        token.pause()

        # Should timeout when paused
        result = token.wait_if_paused(timeout=0.1)
        assert result is False

    def test_token_raise_if_cancelled(self):
        """Test raising exception on cancellation."""
        from utils.ray_orchestration import (
            CancellationToken,
            CancellationError,
            CancellationReason,
        )

        token = CancellationToken()

        # Should not raise when not cancelled
        token.raise_if_cancelled()

        token.cancel(reason=CancellationReason.TIMEOUT)

        with pytest.raises(CancellationError) as exc_info:
            token.raise_if_cancelled()

        assert "TIMEOUT" in str(exc_info.value)

    def test_token_callbacks(self):
        """Test cancellation callbacks."""
        from utils.ray_orchestration import CancellationToken, CancellationReason

        token = CancellationToken()
        callback_called = []

        def on_cancel(t):
            callback_called.append(t.id)

        token.on_cancel(on_cancel)

        assert len(callback_called) == 0

        token.cancel(reason=CancellationReason.USER_REQUESTED)

        assert len(callback_called) == 1
        assert callback_called[0] == token.id

    def test_parent_child_tokens(self):
        """Test hierarchical token cancellation."""
        from utils.ray_orchestration import CancellationToken, CancellationReason

        parent = CancellationToken()
        child = parent.create_child()

        assert child.is_cancelled is False

        # Cancel parent
        parent.cancel(reason=CancellationReason.USER_REQUESTED)

        # Child should be cancelled
        assert child.is_cancelled is True
        assert child.cancellation_request.reason == CancellationReason.PARENT_CANCELLED

    def test_token_thread_safety(self):
        """Test token thread safety."""
        from utils.ray_orchestration import CancellationToken, CancellationReason

        token = CancellationToken()
        results = []

        def checker():
            for _ in range(100):
                results.append(token.is_cancelled)
                time.sleep(0.001)

        def canceller():
            time.sleep(0.05)
            token.cancel(reason=CancellationReason.USER_REQUESTED)

        threads = [
            threading.Thread(target=checker),
            threading.Thread(target=checker),
            threading.Thread(target=canceller),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some False followed by True
        assert False in results
        assert True in results

    def test_token_serialization(self):
        """Test token serialization."""
        from utils.ray_orchestration import CancellationToken, CancellationReason

        token = CancellationToken()
        token.cancel(reason=CancellationReason.USER_REQUESTED, message="Test")

        data = token.to_dict()

        assert "id" in data
        assert data["is_cancelled"] is True
        assert data["request"]["reason"] == "USER_REQUESTED"
        assert data["request"]["message"] == "Test"


class TestCancellationCoordinator:
    """Tests for CancellationCoordinator."""

    def setup_method(self):
        """Reset coordinator before each test."""
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        # Clear internal state
        coordinator = CancellationCoordinator()
        coordinator._tokens.clear()
        coordinator._job_tokens.clear()

    def test_coordinator_singleton(self):
        """Test that coordinator is a singleton."""
        from utils.ray_orchestration import get_cancellation_coordinator

        coord1 = get_cancellation_coordinator()
        coord2 = get_cancellation_coordinator()

        assert coord1 is coord2

    def test_create_job_token(self):
        """Test creating a token for a job."""
        from utils.ray_orchestration import get_cancellation_coordinator

        coordinator = get_cancellation_coordinator()
        job_id = uuid4()

        token = coordinator.create_token(job_id=job_id)

        assert token is not None
        assert coordinator.get_job_token(job_id) is token

    def test_cancel_job_by_id(self):
        """Test cancelling a job by ID."""
        from utils.ray_orchestration import (
            get_cancellation_coordinator,
            CancellationReason,
        )

        coordinator = get_cancellation_coordinator()
        job_id = uuid4()
        token = coordinator.create_token(job_id=job_id)

        result = coordinator.cancel_job(
            job_id=job_id,
            reason=CancellationReason.USER_REQUESTED,
            message="User cancelled",
        )

        assert result is True
        assert token.is_cancelled is True

    def test_cancel_nonexistent_job(self):
        """Test cancelling a non-existent job."""
        from utils.ray_orchestration import get_cancellation_coordinator

        coordinator = get_cancellation_coordinator()

        result = coordinator.cancel_job(job_id=uuid4())

        assert result is False

    def test_pause_resume_job(self):
        """Test pausing and resuming a job."""
        from utils.ray_orchestration import get_cancellation_coordinator

        coordinator = get_cancellation_coordinator()
        job_id = uuid4()
        token = coordinator.create_token(job_id=job_id)

        coordinator.pause_job(job_id)
        assert token.is_paused is True

        coordinator.resume_job(job_id)
        assert token.is_paused is False

    def test_cancel_all_jobs(self):
        """Test cancelling all jobs."""
        from utils.ray_orchestration import (
            get_cancellation_coordinator,
            CancellationReason,
        )

        coordinator = get_cancellation_coordinator()

        # Create multiple jobs
        job_ids = [uuid4() for _ in range(3)]
        tokens = [coordinator.create_token(job_id=jid) for jid in job_ids]

        count = coordinator.cancel_all(
            reason=CancellationReason.SYSTEM_SHUTDOWN,
            message="Shutting down",
        )

        assert count == 3
        assert all(t.is_cancelled for t in tokens)

    def test_coordinator_status(self):
        """Test getting coordinator status."""
        from utils.ray_orchestration import get_cancellation_coordinator

        coordinator = get_cancellation_coordinator()

        # Create some tokens
        for _ in range(3):
            coordinator.create_token(job_id=uuid4())

        status = coordinator.get_status()

        assert status["total_tokens"] == 3
        assert status["active_tokens"] == 3
        assert status["job_tokens"] == 3


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset coordinator before each test."""
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        coordinator = CancellationCoordinator()
        coordinator._tokens.clear()
        coordinator._job_tokens.clear()

    def test_create_cancellation_token(self):
        """Test module-level token creation."""
        from utils.ray_orchestration import create_cancellation_token

        job_id = uuid4()
        token = create_cancellation_token(job_id=job_id)

        assert token is not None
        assert token.is_cancelled is False

    def test_cancel_job(self):
        """Test module-level job cancellation."""
        from utils.ray_orchestration import (
            create_cancellation_token,
            cancel_job,
            get_cancellation_coordinator,
        )

        job_id = uuid4()
        token = create_cancellation_token(job_id=job_id)

        result = cancel_job(job_id)

        assert result is True
        assert token.is_cancelled is True

    def test_cancel_all_jobs(self):
        """Test module-level cancel all."""
        from utils.ray_orchestration import (
            create_cancellation_token,
            cancel_all_jobs,
        )

        job_ids = [uuid4() for _ in range(3)]
        tokens = [create_cancellation_token(job_id=jid) for jid in job_ids]

        count = cancel_all_jobs()

        assert count == 3
        assert all(t.is_cancelled for t in tokens)
