"""
Tests for Job History Storage

Tests SQLite-based job history storage with persistence,
querying, and statistics.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from models.job import Job, JobState, JobType, JobPriority


class TestJobHistoryStorage:
    """Tests for JobHistoryStorage class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create a storage instance with temp database."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        yield storage
        storage.close()

    def test_storage_creation(self, storage):
        """Test storage initializes correctly."""
        assert storage is not None
        assert storage._db_path.exists()

    def test_save_and_get_job(self, storage):
        """Test saving and retrieving a job."""
        job = Job(
            name="Test Job",
            job_type=JobType.BATCH_PROCESS,
            state=JobState.COMPLETED,
        )
        job.mark_started()
        job.mark_completed({"output": "success"})

        storage.save_job(job)

        retrieved = storage.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.name == "Test Job"
        assert retrieved.state == JobState.COMPLETED
        assert retrieved.result == {"output": "success"}

    def test_get_nonexistent_job(self, storage):
        """Test getting a job that doesn't exist."""
        result = storage.get_job(uuid4())
        assert result is None

    def test_update_existing_job(self, storage):
        """Test updating an existing job."""
        job = Job(
            name="Update Test",
            job_type=JobType.SINGLE_PROCESS,
            state=JobState.RUNNING,
        )
        storage.save_job(job)

        # Update job
        job.mark_completed({"data": "updated"})
        storage.save_job(job)

        retrieved = storage.get_job(job.id)
        assert retrieved.state == JobState.COMPLETED
        assert retrieved.result == {"data": "updated"}

    def test_query_by_state(self, storage):
        """Test querying jobs by state."""
        # Create jobs with different states
        completed = Job(name="Completed", job_type=JobType.SEGY_IMPORT)
        completed.mark_completed()
        storage.save_job(completed)

        failed = Job(name="Failed", job_type=JobType.SEGY_EXPORT)
        failed.mark_failed("Error occurred")
        storage.save_job(failed)

        cancelled = Job(name="Cancelled", job_type=JobType.BATCH_PROCESS)
        cancelled.mark_cancelled()
        storage.save_job(cancelled)

        # Query completed only
        results = storage.query_jobs(states=[JobState.COMPLETED])
        assert len(results) == 1
        assert results[0].name == "Completed"

        # Query failed only
        results = storage.query_jobs(states=[JobState.FAILED])
        assert len(results) == 1
        assert results[0].name == "Failed"

        # Query multiple states
        results = storage.query_jobs(
            states=[JobState.COMPLETED, JobState.FAILED]
        )
        assert len(results) == 2

    def test_query_by_type(self, storage):
        """Test querying jobs by type."""
        import_job = Job(name="Import", job_type=JobType.SEGY_IMPORT)
        import_job.mark_completed()
        storage.save_job(import_job)

        export_job = Job(name="Export", job_type=JobType.SEGY_EXPORT)
        export_job.mark_completed()
        storage.save_job(export_job)

        results = storage.query_jobs(job_types=[JobType.SEGY_IMPORT])
        assert len(results) == 1
        assert results[0].name == "Import"

    def test_query_by_date_range(self, storage):
        """Test querying jobs by date range."""
        old_job = Job(name="Old Job", job_type=JobType.BATCH_PROCESS)
        old_job.created_at = datetime.now() - timedelta(days=10)
        old_job.mark_completed()
        storage.save_job(old_job)

        new_job = Job(name="New Job", job_type=JobType.BATCH_PROCESS)
        new_job.mark_completed()
        storage.save_job(new_job)

        # Query jobs from last 5 days
        start = datetime.now() - timedelta(days=5)
        results = storage.query_jobs(start_date=start)
        assert len(results) == 1
        assert results[0].name == "New Job"

    def test_query_with_name_search(self, storage):
        """Test querying jobs with name search."""
        job1 = Job(name="SEGY Import Line 100", job_type=JobType.SEGY_IMPORT)
        job1.mark_completed()
        storage.save_job(job1)

        job2 = Job(name="Batch Processing", job_type=JobType.BATCH_PROCESS)
        job2.mark_completed()
        storage.save_job(job2)

        results = storage.query_jobs(search_name="SEGY")
        assert len(results) == 1
        assert "SEGY" in results[0].name

    def test_query_with_limit_and_offset(self, storage):
        """Test pagination with limit and offset."""
        for i in range(10):
            job = Job(name=f"Job {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_completed()
            storage.save_job(job)

        # First page
        page1 = storage.query_jobs(limit=5, offset=0)
        assert len(page1) == 5

        # Second page
        page2 = storage.query_jobs(limit=5, offset=5)
        assert len(page2) == 5

        # All unique
        all_ids = [j.id for j in page1] + [j.id for j in page2]
        assert len(set(all_ids)) == 10

    def test_query_ordering(self, storage):
        """Test query result ordering."""
        job1 = Job(name="Job A", job_type=JobType.BATCH_PROCESS)
        job1.mark_completed()
        storage.save_job(job1)

        job2 = Job(name="Job B", job_type=JobType.BATCH_PROCESS)
        job2.mark_completed()
        storage.save_job(job2)

        # Default descending order
        results = storage.query_jobs(order_by="completed_at", order_desc=True)
        assert len(results) == 2
        # job2 should be first (newer)

        # Ascending order
        results = storage.query_jobs(order_by="name", order_desc=False)
        assert results[0].name == "Job A"

    def test_get_recent_jobs(self, storage):
        """Test getting recent jobs."""
        for i in range(5):
            job = Job(name=f"Recent {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_completed()
            storage.save_job(job)

        recent = storage.get_recent_jobs(limit=3)
        assert len(recent) == 3

    def test_get_failed_jobs(self, storage):
        """Test getting failed jobs."""
        completed = Job(name="Completed", job_type=JobType.BATCH_PROCESS)
        completed.mark_completed()
        storage.save_job(completed)

        failed1 = Job(name="Failed 1", job_type=JobType.BATCH_PROCESS)
        failed1.mark_failed("Error 1")
        storage.save_job(failed1)

        failed2 = Job(name="Failed 2", job_type=JobType.BATCH_PROCESS)
        failed2.mark_failed("Error 2")
        storage.save_job(failed2)

        failed = storage.get_failed_jobs()
        assert len(failed) == 2
        assert all(j.state == JobState.FAILED for j in failed)


class TestJobHistoryStatistics:
    """Tests for statistics functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create a storage instance with temp database."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        yield storage
        storage.close()

    def test_get_statistics_empty(self, storage):
        """Test statistics on empty storage."""
        stats = storage.get_statistics()

        assert stats["total_jobs"] == 0
        assert stats["error_rate_percent"] == 0

    def test_get_statistics_with_jobs(self, storage):
        """Test statistics with jobs."""
        # Add completed jobs
        for i in range(5):
            job = Job(name=f"Completed {i}", job_type=JobType.SEGY_IMPORT)
            job.mark_started()
            job.mark_completed()
            storage.save_job(job)

        # Add failed jobs
        for i in range(2):
            job = Job(name=f"Failed {i}", job_type=JobType.SEGY_EXPORT)
            job.mark_started()
            job.mark_failed("Error")
            storage.save_job(job)

        stats = storage.get_statistics()

        assert stats["total_jobs"] == 7
        assert stats["by_state"]["COMPLETED"] == 5
        assert stats["by_state"]["FAILED"] == 2
        assert stats["by_type"]["SEGY_IMPORT"] == 5
        assert stats["by_type"]["SEGY_EXPORT"] == 2
        # Error rate: 2/7 = 28.57%
        assert 28 <= stats["error_rate_percent"] <= 29

    def test_get_daily_counts(self, storage):
        """Test daily job counts."""
        # Add jobs for today
        for i in range(3):
            job = Job(name=f"Today {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_completed()
            storage.save_job(job)

        daily = storage.get_daily_counts(days=7)
        assert len(daily) >= 1

        today = datetime.now().strftime("%Y-%m-%d")
        today_entry = next((d for d in daily if d["date"] == today), None)
        assert today_entry is not None
        assert today_entry["total"] == 3


class TestJobHistoryManagement:
    """Tests for history management operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create a storage instance with temp database."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        yield storage
        storage.close()

    def test_delete_job(self, storage):
        """Test deleting a job."""
        job = Job(name="To Delete", job_type=JobType.BATCH_PROCESS)
        job.mark_completed()
        storage.save_job(job)

        assert storage.get_job(job.id) is not None

        deleted = storage.delete_job(job.id)
        assert deleted is True
        assert storage.get_job(job.id) is None

    def test_delete_nonexistent_job(self, storage):
        """Test deleting a job that doesn't exist."""
        deleted = storage.delete_job(uuid4())
        assert deleted is False

    def test_cleanup_old_entries(self, temp_db):
        """Test cleaning up old entries."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(
            db_path=temp_db,
            max_history_days=7,
            auto_cleanup=False,
        )

        # Add old job
        old_job = Job(name="Old", job_type=JobType.BATCH_PROCESS)
        old_job.created_at = datetime.now() - timedelta(days=30)
        old_job.mark_completed()
        storage.save_job(old_job)

        # Add recent job
        new_job = Job(name="New", job_type=JobType.BATCH_PROCESS)
        new_job.mark_completed()
        storage.save_job(new_job)

        # Cleanup
        deleted = storage.cleanup_old_entries()
        assert deleted == 1

        # Old job should be gone
        assert storage.get_job(old_job.id) is None
        # New job should remain
        assert storage.get_job(new_job.id) is not None

        storage.close()

    def test_clear_all(self, storage):
        """Test clearing all history."""
        for i in range(5):
            job = Job(name=f"Job {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_completed()
            storage.save_job(job)

        deleted = storage.clear_all()
        assert deleted == 5

        recent = storage.get_recent_jobs()
        assert len(recent) == 0


class TestJobHistoryExportImport:
    """Tests for export/import functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def storage(self, temp_db):
        """Create a storage instance with temp database."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        yield storage
        storage.close()

    def test_export_to_json(self, storage, tmp_path):
        """Test exporting to JSON."""
        for i in range(3):
            job = Job(name=f"Export Test {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_completed()
            storage.save_job(job)

        export_path = tmp_path / "export.json"
        count = storage.export_to_json(export_path)

        assert count == 3
        assert export_path.exists()

        import json
        with open(export_path) as f:
            data = json.load(f)
        assert data["job_count"] == 3
        assert len(data["jobs"]) == 3

    def test_import_from_json(self, temp_db, tmp_path):
        """Test importing from JSON."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        # First storage: create and export jobs
        storage1 = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        for i in range(3):
            job = Job(name=f"Import Test {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_completed()
            storage1.save_job(job)

        export_path = tmp_path / "export.json"
        storage1.export_to_json(export_path)
        storage1.close()

        # Second storage: import
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path2 = Path(f.name)

        storage2 = JobHistoryStorage(db_path=db_path2, auto_cleanup=False)
        imported = storage2.import_from_json(export_path)

        assert imported == 3
        assert len(storage2.get_recent_jobs()) == 3

        storage2.close()
        db_path2.unlink()


class TestJobHistorySingleton:
    """Tests for singleton access."""

    def test_get_job_history_storage(self, tmp_path):
        """Test singleton accessor."""
        import utils.ray_orchestration.job_history as jh

        # Reset singleton
        jh._history_storage = None

        db_path = tmp_path / "test.db"
        storage1 = jh.get_job_history_storage(db_path=db_path)
        storage2 = jh.get_job_history_storage()

        assert storage1 is storage2

        storage1.close()
        jh._history_storage = None
