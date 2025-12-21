"""
Tests for JobManager History Integration

Tests that JobManager properly saves terminal jobs to history
and provides history access methods.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from models.job import Job, JobState, JobType, JobPriority
from models.job_config import JobConfig


class TestJobManagerHistoryIntegration:
    """Tests for JobManager history integration."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def history_storage(self, temp_db):
        """Create a history storage instance."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        yield storage
        storage.close()

    @pytest.fixture
    def job_manager(self, history_storage):
        """Create a job manager with custom history storage."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import get_cancellation_coordinator

        # Reset singleton for testing
        JobManager._instance = None

        # Create manager, then inject custom history storage
        manager = JobManager()
        manager._history = history_storage
        yield manager

        # Reset singleton after test
        JobManager._instance = None

    def test_manager_has_history(self, job_manager, history_storage):
        """Test that manager has history storage."""
        assert job_manager._history is history_storage
        assert job_manager.history_storage is history_storage

    def test_complete_job_saves_to_history(self, job_manager, history_storage):
        """Test completing a job saves it to history."""
        job = job_manager.submit_job(
            name="Complete Test",
            job_type=JobType.BATCH_PROCESS,
        )
        job_manager.start_job(job.id)
        job_manager.complete_job(job.id, result={"output": "data"})

        # Should be in history
        history_job = history_storage.get_job(job.id)
        assert history_job is not None
        assert history_job.state == JobState.COMPLETED
        assert history_job.result == {"output": "data"}

    def test_fail_job_saves_to_history(self, job_manager, history_storage):
        """Test failing a job saves it to history."""
        job = job_manager.submit_job(
            name="Fail Test",
            job_type=JobType.SEGY_IMPORT,
        )
        job_manager.start_job(job.id)
        job_manager.fail_job(job.id, error="Test error", traceback="stack trace")

        # Should be in history
        history_job = history_storage.get_job(job.id)
        assert history_job is not None
        assert history_job.state == JobState.FAILED
        assert history_job.error_message == "Test error"

    def test_cancel_job_saves_to_history(self, job_manager, history_storage):
        """Test cancelling a job saves it to history."""
        job = job_manager.submit_job(
            name="Cancel Test",
            job_type=JobType.QC_ANALYSIS,
        )
        job_manager.start_job(job.id)
        job_manager.cancel_job(job.id)
        job_manager.finalize_cancellation(job.id)

        # Should be in history
        history_job = history_storage.get_job(job.id)
        assert history_job is not None
        assert history_job.state == JobState.CANCELLED

    def test_get_job_history(self, job_manager, history_storage):
        """Test getting job history from manager."""
        # Complete some jobs
        for i in range(3):
            job = job_manager.submit_job(
                name=f"History Job {i}",
                job_type=JobType.BATCH_PROCESS,
            )
            job_manager.start_job(job.id)
            job_manager.complete_job(job.id)

        # Query history
        history = job_manager.get_job_history(limit=10)
        assert len(history) == 3

    def test_get_job_history_with_filters(self, job_manager, history_storage):
        """Test getting filtered job history."""
        # Complete job
        completed = job_manager.submit_job(
            name="Completed",
            job_type=JobType.SEGY_IMPORT,
        )
        job_manager.start_job(completed.id)
        job_manager.complete_job(completed.id)

        # Fail job
        failed = job_manager.submit_job(
            name="Failed",
            job_type=JobType.SEGY_EXPORT,
        )
        job_manager.start_job(failed.id)
        job_manager.fail_job(failed.id, error="Error")

        # Query only completed
        completed_history = job_manager.get_job_history(
            states=[JobState.COMPLETED]
        )
        assert len(completed_history) == 1
        assert completed_history[0].name == "Completed"

        # Query only failed
        failed_history = job_manager.get_job_history(
            states=[JobState.FAILED]
        )
        assert len(failed_history) == 1
        assert failed_history[0].name == "Failed"

    def test_get_history_statistics(self, job_manager, history_storage):
        """Test getting history statistics."""
        # Complete jobs
        for i in range(5):
            job = job_manager.submit_job(
                name=f"Stats Job {i}",
                job_type=JobType.BATCH_PROCESS,
            )
            job_manager.start_job(job.id)
            job_manager.complete_job(job.id)

        # Fail a job
        failed = job_manager.submit_job(
            name="Failed Stats",
            job_type=JobType.BATCH_PROCESS,
        )
        job_manager.start_job(failed.id)
        job_manager.fail_job(failed.id, error="Error")

        stats = job_manager.get_history_statistics()

        assert stats["total_jobs"] == 6
        assert stats["by_state"]["COMPLETED"] == 5
        assert stats["by_state"]["FAILED"] == 1

    def test_get_recent_failures(self, job_manager, history_storage):
        """Test getting recent failures."""
        # Complete some jobs
        for i in range(2):
            job = job_manager.submit_job(
                name=f"Complete {i}",
                job_type=JobType.BATCH_PROCESS,
            )
            job_manager.start_job(job.id)
            job_manager.complete_job(job.id)

        # Fail some jobs
        for i in range(3):
            job = job_manager.submit_job(
                name=f"Fail {i}",
                job_type=JobType.BATCH_PROCESS,
            )
            job_manager.start_job(job.id)
            job_manager.fail_job(job.id, error=f"Error {i}")

        failures = job_manager.get_recent_failures(limit=10)

        assert len(failures) == 3
        assert all(j.state == JobState.FAILED for j in failures)


class TestJobManagerHistoryDisabled:
    """Tests for JobManager when history is disabled."""

    def test_manager_without_history(self):
        """Test manager works when history storage is None."""
        from utils.ray_orchestration.job_manager import JobManager

        # Reset singleton
        JobManager._instance = None

        # Create manager with None history
        manager = JobManager.__new__(JobManager)
        manager._initialized = False
        manager._history = None
        manager._jobs = {}
        manager._progress = {}
        manager._ray_refs = {}
        manager._callbacks = {
            "on_job_started": [],
            "on_job_completed": [],
            "on_job_failed": [],
            "on_job_cancelled": [],
            "on_progress_update": [],
        }
        manager._lock = __import__('threading').Lock()
        manager._cancellation = None
        manager._initialized = True

        # Test history methods return empty
        assert manager.get_job_history() == []
        assert manager.get_history_statistics() == {}
        assert manager.get_recent_failures() == []

        # Reset singleton
        JobManager._instance = None
