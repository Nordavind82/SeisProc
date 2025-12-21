"""
Tests for Processing Job Adapter

Tests the ProcessingJobAdapter and RayProcessingJobAdapter.
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, patch, MagicMock


class TestProcessingJobAdapter:
    """Tests for ProcessingJobAdapter."""

    def setup_method(self):
        """Reset singletons before each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        from utils.ray_orchestration.checkpoint import CheckpointManager
        JobManager._instance = None
        CancellationCoordinator._instance = None
        CheckpointManager._instance = None

    def teardown_method(self):
        """Cleanup after each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        from utils.ray_orchestration.checkpoint import CheckpointManager
        JobManager._instance = None
        CancellationCoordinator._instance = None
        CheckpointManager._instance = None

    def test_adapter_creation(self):
        """Test creating a processing adapter."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        # Create mock config
        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'TestProcessor'}
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = ProcessingJobAdapter(mock_config)

        assert adapter._processing_config == mock_config
        assert adapter.job is None

    def test_default_job_name(self):
        """Test default job name generation."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'Denoise3DProcessor'}
        mock_config.n_workers = 4

        adapter = ProcessingJobAdapter(mock_config)

        assert 'Denoise3DProcessor' in adapter._job_name

    def test_custom_job_name(self):
        """Test custom job name."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_config = Mock()
        mock_config.processor_config = {}

        adapter = ProcessingJobAdapter(mock_config, job_name="My Custom Job")

        assert adapter._job_name == "My Custom Job"

    def test_submit_creates_job(self):
        """Test that submit creates a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter
        from models.job import JobType

        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'TestProcessor'}
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = ProcessingJobAdapter(mock_config)
        job = adapter.submit()

        assert job is not None
        assert adapter.job == job
        assert adapter.job_id == job.id
        assert job.job_type == JobType.BATCH_PROCESS

    def test_cancel_without_job(self):
        """Test cancel returns False without a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_config = Mock()
        mock_config.processor_config = {}

        adapter = ProcessingJobAdapter(mock_config)

        result = adapter.cancel()
        assert result is False

    def test_pause_without_job(self):
        """Test pause returns False without a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_config = Mock()
        mock_config.processor_config = {}

        adapter = ProcessingJobAdapter(mock_config)

        result = adapter.pause()
        assert result is False

    def test_resume_without_job(self):
        """Test resume returns False without a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_config = Mock()
        mock_config.processor_config = {}

        adapter = ProcessingJobAdapter(mock_config)

        result = adapter.resume()
        assert result is False


class TestRayProcessingJobAdapter:
    """Tests for RayProcessingJobAdapter."""

    def setup_method(self):
        """Reset singletons before each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        JobManager._instance = None
        CancellationCoordinator._instance = None

    def teardown_method(self):
        """Cleanup after each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        JobManager._instance = None
        CancellationCoordinator._instance = None

    def test_adapter_creation(self):
        """Test creating a Ray processing adapter."""
        from utils.ray_orchestration.processing_job_adapter import RayProcessingJobAdapter

        mock_config = Mock()
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = RayProcessingJobAdapter(mock_config, use_ray=False)

        assert adapter._use_ray is False

    def test_submit_creates_job(self):
        """Test that submit creates a job."""
        from utils.ray_orchestration.processing_job_adapter import RayProcessingJobAdapter

        mock_config = Mock()
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = RayProcessingJobAdapter(mock_config, use_ray=False)
        job = adapter.submit()

        assert job is not None
        assert adapter.job == job


class TestCreateProcessingJob:
    """Tests for create_processing_job convenience function."""

    def setup_method(self):
        """Reset singletons before each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        JobManager._instance = None
        CancellationCoordinator._instance = None

    def teardown_method(self):
        """Cleanup after each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        JobManager._instance = None
        CancellationCoordinator._instance = None

    def test_create_local_job(self):
        """Test creating a local processing job."""
        from utils.ray_orchestration.processing_job_adapter import create_processing_job

        adapter = create_processing_job(
            input_storage_dir='/tmp/input',
            output_storage_dir='/tmp/output',
            processor_config={'class_name': 'TestProcessor'},
            job_name='Test Job',
            use_ray=False,
        )

        assert adapter is not None
        # Should be ProcessingJobAdapter, not RayProcessingJobAdapter
        assert adapter.__class__.__name__ == 'ProcessingJobAdapter'

    def test_create_ray_job(self):
        """Test creating a Ray processing job."""
        from utils.ray_orchestration.processing_job_adapter import create_processing_job

        adapter = create_processing_job(
            input_storage_dir='/tmp/input',
            output_storage_dir='/tmp/output',
            processor_config={'class_name': 'TestProcessor'},
            use_ray=True,
        )

        assert adapter is not None
        assert adapter.__class__.__name__ == 'RayProcessingJobAdapter'


class TestProcessingJobIntegration:
    """Integration tests for processing job adapters."""

    def setup_method(self):
        """Reset singletons before each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        from utils.ray_orchestration.checkpoint import CheckpointManager
        JobManager._instance = None
        CancellationCoordinator._instance = None
        CheckpointManager._instance = None

    def teardown_method(self):
        """Cleanup after each test."""
        from utils.ray_orchestration.job_manager import JobManager
        from utils.ray_orchestration.cancellation import CancellationCoordinator
        from utils.ray_orchestration.checkpoint import CheckpointManager
        JobManager._instance = None
        CancellationCoordinator._instance = None
        CheckpointManager._instance = None

    def test_submit_and_cancel(self):
        """Test submitting and cancelling a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter
        from models.job import JobState

        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'TestProcessor'}
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = ProcessingJobAdapter(mock_config)
        job = adapter.submit()

        # Cancel the job
        result = adapter.cancel()
        assert result is True

    def test_submit_and_pause_resume(self):
        """Test submitting, pausing, and resuming a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter
        from utils.ray_orchestration.job_manager import get_job_manager

        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'TestProcessor'}
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = ProcessingJobAdapter(mock_config)
        job = adapter.submit()

        # Start the job first (pause only works on running jobs)
        manager = get_job_manager()
        manager.start_job(job.id)

        # Pause
        pause_result = adapter.pause()
        assert pause_result is True

        # Resume
        resume_result = adapter.resume()
        assert resume_result is True
