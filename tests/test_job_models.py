"""
Tests for Job Data Models

Tests Job, JobProgress, JobConfig dataclasses and serialization.
"""

import pytest
from uuid import UUID, uuid4
from datetime import datetime


class TestJobModel:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        """Test creating a job with required fields."""
        from models.job import Job, JobType, JobState

        job = Job(
            name="Test Import",
            job_type=JobType.SEGY_IMPORT,
        )

        assert job.name == "Test Import"
        assert job.job_type == JobType.SEGY_IMPORT
        assert job.state == JobState.CREATED
        assert isinstance(job.id, UUID)
        assert isinstance(job.created_at, datetime)

    def test_job_serialization(self):
        """Job model serializes and deserializes correctly."""
        from models.job import Job, JobType, JobState, JobPriority

        job = Job(
            id=uuid4(),
            name="Test Import",
            job_type=JobType.SEGY_IMPORT,
            state=JobState.CREATED,
            priority=JobPriority.HIGH,
        )

        serialized = job.to_dict()
        deserialized = Job.from_dict(serialized)

        assert deserialized.id == job.id
        assert deserialized.name == job.name
        assert deserialized.job_type == job.job_type
        assert deserialized.state == job.state
        assert deserialized.priority == job.priority

    def test_job_state_transitions(self):
        """Test job state transitions."""
        from models.job import Job, JobType, JobState

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        assert job.state == JobState.CREATED

        job.mark_queued()
        assert job.state == JobState.QUEUED
        assert job.queued_at is not None

        job.mark_started()
        assert job.state == JobState.RUNNING
        assert job.started_at is not None

        job.mark_completed(result={"traces": 1000})
        assert job.state == JobState.COMPLETED
        assert job.completed_at is not None
        assert job.result == {"traces": 1000}

    def test_job_cancellation_states(self):
        """Test job cancellation state transitions."""
        from models.job import Job, JobType, JobState

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.mark_queued()
        job.mark_started()

        assert job.can_cancel is True

        job.mark_cancelling()
        assert job.state == JobState.CANCELLING

        job.mark_cancelled()
        assert job.state == JobState.CANCELLED
        assert job.is_terminal is True

    def test_job_pause_resume(self):
        """Test job pause and resume."""
        from models.job import Job, JobType, JobState

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.mark_queued()
        job.mark_started()

        assert job.can_pause is True

        job.mark_paused()
        assert job.state == JobState.PAUSED
        assert job.can_resume is True

        job.mark_resumed()
        assert job.state == JobState.RUNNING

    def test_job_failure(self):
        """Test job failure handling."""
        from models.job import Job, JobType, JobState

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.mark_queued()
        job.mark_started()

        job.mark_failed(error="Test error", traceback="Traceback...")

        assert job.state == JobState.FAILED
        assert job.error_message == "Test error"
        assert job.error_traceback == "Traceback..."
        assert job.is_terminal is True

    def test_job_duration(self):
        """Test job duration calculation."""
        from models.job import Job, JobType
        import time

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        assert job.duration_seconds is None

        job.mark_queued()
        job.mark_started()
        time.sleep(0.1)  # Small delay
        job.mark_completed()

        assert job.duration_seconds is not None
        assert job.duration_seconds >= 0.1


class TestJobProgress:
    """Tests for JobProgress dataclass."""

    def test_progress_creation(self):
        """Test creating job progress."""
        from models.job_progress import JobProgress

        job_id = uuid4()
        progress = JobProgress(job_id=job_id)

        assert progress.job_id == job_id
        assert progress.phase == "initializing"
        assert progress.overall_percent == 0.0
        assert len(progress.workers) == 0

    def test_worker_progress(self):
        """Test worker progress tracking."""
        from models.job_progress import WorkerProgress

        worker = WorkerProgress(
            worker_id="worker-1",
            items_total=100,
            items_processed=50,
        )

        assert worker.progress_percent == 50.0
        assert worker.items_remaining == 50
        assert worker.elapsed_seconds >= 0

    def test_aggregate_progress(self):
        """Test aggregating progress from multiple workers."""
        from models.job_progress import JobProgress, WorkerProgress

        job_id = uuid4()
        progress = JobProgress(job_id=job_id)

        # Add workers
        progress.add_worker(WorkerProgress(
            worker_id="worker-1",
            items_total=100,
            items_processed=50,
        ))
        progress.add_worker(WorkerProgress(
            worker_id="worker-2",
            items_total=100,
            items_processed=100,
        ))

        assert progress.total_items == 200
        assert progress.total_items_processed == 150
        assert progress.overall_percent == 75.0
        assert progress.active_workers == 2

    def test_progress_serialization(self):
        """Test progress serialization."""
        from models.job_progress import JobProgress, WorkerProgress

        job_id = uuid4()
        progress = JobProgress(job_id=job_id, phase="processing")
        progress.add_worker(WorkerProgress(
            worker_id="worker-1",
            items_total=100,
            items_processed=50,
        ))

        serialized = progress.to_dict()

        assert serialized["job_id"] == str(job_id)
        assert serialized["phase"] == "processing"
        assert serialized["overall_percent"] == 50.0
        assert len(serialized["workers"]) == 1


class TestJobConfig:
    """Tests for JobConfig dataclass."""

    def test_default_config(self):
        """Test default job configuration."""
        from models.job_config import JobConfig

        config = JobConfig()

        assert config.resources.num_cpus == 1.0
        assert config.checkpoint.enabled is True
        assert config.retry.max_retries == 3
        assert config.batch_size == 100

    def test_resource_requirements(self):
        """Test resource requirements."""
        from models.job_config import ResourceRequirements

        resources = ResourceRequirements.for_cpu_intensive(cores=8)

        assert resources.num_cpus == 8
        assert resources.memory_mb == 8192  # 1GB per core

    def test_segy_import_config(self):
        """Test SEGY import configuration factory."""
        from models.job_config import JobConfig

        config = JobConfig.for_segy_import(file_size_mb=1000)

        assert config.checkpoint.enabled is True
        assert config.num_workers is not None
        assert config.batch_size == 1000

    def test_gpu_processing_config(self):
        """Test GPU processing configuration factory."""
        from models.job_config import JobConfig

        config = JobConfig.for_gpu_processing()

        assert config.resources.num_gpus == 1.0
        assert config.batch_size == 5000  # Large batches for GPU
        assert config.num_workers == 1

    def test_config_serialization(self):
        """Test config serialization."""
        from models.job_config import JobConfig

        config = JobConfig(batch_size=500, timeout_seconds=3600)
        serialized = config.to_dict()

        assert serialized["batch_size"] == 500
        assert serialized["timeout_seconds"] == 3600
        assert "resources" in serialized
        assert "checkpoint" in serialized
