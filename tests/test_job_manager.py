"""
Tests for Job Manager

Tests JobManager lifecycle management and integration with cancellation.
"""

import pytest
from uuid import uuid4


class TestJobManager:
    """Tests for JobManager."""

    def setup_method(self):
        """Reset manager before each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator

        # Reset singleton state
        manager = JobManager()
        manager._jobs.clear()
        manager._progress.clear()
        manager._ray_refs.clear()

        coordinator = CancellationCoordinator()
        coordinator._tokens.clear()
        coordinator._job_tokens.clear()

    def test_manager_singleton(self):
        """Test that JobManager is a singleton."""
        from utils.ray_orchestration import get_job_manager

        manager1 = get_job_manager()
        manager2 = get_job_manager()

        assert manager1 is manager2

    def test_submit_job(self):
        """Test submitting a job."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()

        job = manager.submit_job(
            name="Test Import",
            job_type=JobType.SEGY_IMPORT,
            tags=["test"],
        )

        assert job is not None
        assert job.name == "Test Import"
        assert job.job_type == JobType.SEGY_IMPORT
        assert job.state == JobState.QUEUED
        assert "test" in job.tags

    def test_start_job(self):
        """Test starting a queued job."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)

        result = manager.start_job(job.id)

        assert result is True
        assert job.state == JobState.RUNNING
        assert job.started_at is not None

    def test_complete_job(self):
        """Test completing a job."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)

        result = manager.complete_job(job.id, result={"traces": 1000})

        assert result is True
        assert job.state == JobState.COMPLETED
        assert job.result == {"traces": 1000}

    def test_fail_job(self):
        """Test failing a job."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)

        result = manager.fail_job(job.id, error="Test error", traceback="...")

        assert result is True
        assert job.state == JobState.FAILED
        assert job.error_message == "Test error"

    def test_cancel_job(self):
        """Test cancelling a job."""
        from utils.ray_orchestration import get_job_manager, CancellationReason
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)

        result = manager.cancel_job(
            job.id,
            reason=CancellationReason.USER_REQUESTED,
            message="User cancelled",
        )

        assert result is True
        assert job.state == JobState.CANCELLING

        # Check cancellation token
        token = manager.get_cancellation_token(job.id)
        assert token is not None
        assert token.is_cancelled is True

    def test_finalize_cancellation(self):
        """Test finalizing job cancellation."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)
        manager.cancel_job(job.id)

        result = manager.finalize_cancellation(job.id)

        assert result is True
        assert job.state == JobState.CANCELLED

    def test_pause_resume_job(self):
        """Test pausing and resuming a job."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)

        # Pause
        result = manager.pause_job(job.id)
        assert result is True
        assert job.state == JobState.PAUSED

        token = manager.get_cancellation_token(job.id)
        assert token.is_paused is True

        # Resume
        result = manager.resume_job(job.id)
        assert result is True
        assert job.state == JobState.RUNNING
        assert token.is_paused is False

    def test_get_progress(self):
        """Test getting job progress."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)

        progress = manager.get_progress(job.id)

        assert progress is not None
        assert progress.job_id == job.id
        assert progress.phase == "initializing"

    def test_update_progress(self):
        """Test updating job progress from worker."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType
        from models.job_progress import ProgressUpdate

        manager = get_job_manager()
        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)

        update = ProgressUpdate(
            job_id=job.id,
            worker_id="worker-1",
            items_processed=50,
            items_total=100,
            message="Processing...",
        )

        manager.update_progress(update)

        progress = manager.get_progress(job.id)
        assert progress.overall_percent == 50.0
        assert len(progress.workers) == 1
        assert progress.workers[0].items_processed == 50

    def test_list_jobs(self):
        """Test listing jobs with filters."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()

        # Create various jobs
        job1 = manager.submit_job(
            name="Import 1",
            job_type=JobType.SEGY_IMPORT,
            tags=["import"],
        )
        job2 = manager.submit_job(
            name="Process 1",
            job_type=JobType.BATCH_PROCESS,
            tags=["process"],
        )
        manager.start_job(job1.id)

        # Filter by state
        running = manager.list_jobs(states=[JobState.RUNNING])
        assert len(running) == 1
        assert running[0].id == job1.id

        # Filter by type
        imports = manager.list_jobs(job_types=[JobType.SEGY_IMPORT])
        assert len(imports) == 1
        assert imports[0].id == job1.id

        # Filter by tags
        process_jobs = manager.list_jobs(tags=["process"])
        assert len(process_jobs) == 1
        assert process_jobs[0].id == job2.id

    def test_get_active_jobs(self):
        """Test getting active jobs."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType

        manager = get_job_manager()

        job1 = manager.submit_job(name="Test 1", job_type=JobType.BATCH_PROCESS)
        job2 = manager.submit_job(name="Test 2", job_type=JobType.BATCH_PROCESS)

        manager.start_job(job1.id)

        active = manager.get_active_jobs()

        assert len(active) == 1
        assert active[0].id == job1.id

    def test_job_callbacks(self):
        """Test job event callbacks."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType

        manager = get_job_manager()
        events = []

        manager.register_callback("on_job_started", lambda j: events.append(("started", j.id)))
        manager.register_callback("on_job_completed", lambda j: events.append(("completed", j.id)))

        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)
        manager.complete_job(job.id)

        assert ("started", job.id) in events
        assert ("completed", job.id) in events

    def test_cleanup(self):
        """Test cleaning up completed jobs."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType

        manager = get_job_manager()

        job1 = manager.submit_job(name="Test 1", job_type=JobType.BATCH_PROCESS)
        job2 = manager.submit_job(name="Test 2", job_type=JobType.BATCH_PROCESS)

        manager.start_job(job1.id)
        manager.complete_job(job1.id)
        manager.start_job(job2.id)

        count = manager.cleanup()

        assert count == 1
        assert manager.get_job(job1.id) is None
        assert manager.get_job(job2.id) is not None

    def test_manager_status(self):
        """Test getting manager status."""
        from utils.ray_orchestration import get_job_manager
        from models.job import JobType

        manager = get_job_manager()

        job = manager.submit_job(name="Test", job_type=JobType.BATCH_PROCESS)
        manager.start_job(job.id)

        status = manager.get_status()

        assert status["total_jobs"] == 1
        assert status["active_jobs"] == 1
        assert "RUNNING" in status["state_counts"]
        assert "cancellation" in status
