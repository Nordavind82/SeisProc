"""
Tests for Processing Job Integration

Tests the ProcessingJobAdapter, processing_api, and Qt bridge integration
for Phase 4 Job Management.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4
from dataclasses import dataclass
from typing import Optional

from models.job import Job, JobType, JobState
from models.job_config import JobConfig


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_processing_config():
    """Create a mock ProcessingConfig."""
    config = Mock()
    config.input_storage_dir = '/tmp/input'
    config.output_storage_dir = '/tmp/output'
    config.processor_config = {'class_name': 'TestProcessor'}
    config.n_workers = 4
    return config


@pytest.fixture
def mock_qt_bridge():
    """Create a mock Qt bridge."""
    bridge = Mock()
    bridge.signals = Mock()
    bridge.signals.emit_job_queued = Mock()
    bridge.signals.emit_job_started = Mock()
    bridge.signals.emit_job_completed = Mock()
    bridge.signals.emit_job_failed = Mock()
    bridge.signals.emit_state_changed = Mock()
    bridge.signals.job_progress = Mock()
    bridge.signals.job_progress.emit = Mock()
    return bridge


@pytest.fixture
def mock_job_manager():
    """Create a mock JobManager."""
    manager = Mock()
    job = Job(name="Test Job", job_type=JobType.BATCH_PROCESS)
    manager.submit_job = Mock(return_value=job)
    manager.start_job = Mock()
    manager.complete_job = Mock()
    manager.fail_job = Mock()
    manager.cancel_job = Mock()
    manager.get_cancellation_token = Mock(return_value=Mock(is_cancelled=False))
    manager.update_progress = Mock()
    return manager


# =============================================================================
# ProcessingJobAdapter Tests
# =============================================================================

class TestProcessingJobAdapter:
    """Tests for ProcessingJobAdapter."""

    def test_adapter_creation(self, mock_processing_config):
        """Test creating a ProcessingJobAdapter."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            job_name="Test Processing",
        )

        assert adapter._job_name == "Test Processing"
        assert adapter._processing_config == mock_processing_config
        assert adapter._qt_bridge is None

    def test_adapter_with_qt_bridge(self, mock_processing_config, mock_qt_bridge):
        """Test creating adapter with Qt bridge."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            job_name="Test",
            qt_bridge=mock_qt_bridge,
        )

        assert adapter._qt_bridge == mock_qt_bridge

    def test_default_job_name(self, mock_processing_config):
        """Test default job name generation."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(mock_processing_config)

        # Should generate name from processor config
        assert "TestProcessor" in adapter._job_name

    @patch('utils.ray_orchestration.processing_job_adapter.get_job_manager')
    def test_submit_creates_job(self, mock_get_manager, mock_processing_config, mock_job_manager):
        """Test that submit() creates a job."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_get_manager.return_value = mock_job_manager

        adapter = ProcessingJobAdapter(mock_processing_config, job_name="Test")
        adapter._manager = mock_job_manager

        job = adapter.submit()

        assert job is not None
        mock_job_manager.submit_job.assert_called_once()

    @patch('utils.ray_orchestration.processing_job_adapter.get_job_manager')
    def test_submit_emits_queued_signal(self, mock_get_manager, mock_processing_config, mock_qt_bridge, mock_job_manager):
        """Test that submit() emits queued signal to Qt bridge."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_get_manager.return_value = mock_job_manager

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            job_name="Test",
            qt_bridge=mock_qt_bridge,
        )
        adapter._manager = mock_job_manager

        job = adapter.submit()

        mock_qt_bridge.signals.emit_job_queued.assert_called_once()


class TestProcessingJobAdapterEmitMethods:
    """Tests for ProcessingJobAdapter emit methods."""

    def test_emit_progress(self, mock_processing_config, mock_qt_bridge):
        """Test _emit_progress method."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            qt_bridge=mock_qt_bridge,
        )
        adapter._job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        progress_info = {
            'percent': 50.0,
            'message': 'Processing...',
            'phase': 'processing',
        }
        adapter._emit_progress(progress_info)

        mock_qt_bridge.signals.job_progress.emit.assert_called_once()

    def test_emit_job_started(self, mock_processing_config, mock_qt_bridge):
        """Test _emit_job_started method."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            qt_bridge=mock_qt_bridge,
        )
        adapter._job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        adapter._emit_job_started()

        mock_qt_bridge.signals.emit_job_started.assert_called_once()

    def test_emit_job_completed(self, mock_processing_config, mock_qt_bridge):
        """Test _emit_job_completed method."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            qt_bridge=mock_qt_bridge,
        )
        adapter._job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        adapter._emit_job_completed()

        mock_qt_bridge.signals.emit_job_completed.assert_called_once()

    def test_emit_job_failed(self, mock_processing_config, mock_qt_bridge):
        """Test _emit_job_failed method."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            qt_bridge=mock_qt_bridge,
        )
        adapter._job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        adapter._emit_job_failed()

        mock_qt_bridge.signals.emit_job_failed.assert_called_once()

    def test_emit_without_bridge_no_error(self, mock_processing_config):
        """Test emit methods don't error without Qt bridge."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        adapter = ProcessingJobAdapter(mock_processing_config)
        adapter._job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        # These should not raise
        adapter._emit_job_queued()
        adapter._emit_job_started()
        adapter._emit_progress({'percent': 50})
        adapter._emit_job_completed()
        adapter._emit_job_failed()
        adapter._emit_state_changed()


# =============================================================================
# Processing API Tests
# =============================================================================

class TestProcessingAPI:
    """Tests for the high-level processing API."""

    def test_get_optimal_workers(self):
        """Test get_optimal_workers function."""
        from utils.ray_orchestration.processing_api import get_optimal_workers

        workers = get_optimal_workers()

        # Should be at least 2
        assert workers >= 2

    def test_processing_result_creation(self):
        """Test ProcessingResult dataclass."""
        from utils.ray_orchestration.processing_api import ProcessingResult

        result = ProcessingResult(
            success=True,
            job_id=uuid4(),
            output_dir='/tmp/output',
            n_gathers=100,
            n_traces=10000,
            elapsed_time=60.0,
            throughput=166.7,
        )

        assert result.success is True
        assert result.n_gathers == 100
        assert result.n_traces == 10000

    def test_export_result_creation(self):
        """Test ExportResult dataclass."""
        from utils.ray_orchestration.processing_api import ExportResult

        result = ExportResult(
            success=True,
            job_id=uuid4(),
            output_file='/tmp/output.sgy',
            n_traces=10000,
            elapsed_time=30.0,
        )

        assert result.success is True
        assert result.n_traces == 10000

    def test_processing_result_from_coordinator(self):
        """Test ProcessingResult.from_coordinator_result."""
        from utils.ray_orchestration.processing_api import ProcessingResult

        # Mock coordinator result
        coord_result = Mock()
        coord_result.success = True
        coord_result.output_dir = '/tmp/out'
        coord_result.n_gathers = 50
        coord_result.n_traces = 5000
        coord_result.elapsed_time = 45.0
        coord_result.throughput_traces_per_sec = 111.1
        coord_result.error = None

        job_id = uuid4()
        result = ProcessingResult.from_coordinator_result(coord_result, job_id)

        assert result.success is True
        assert result.job_id == job_id
        assert result.n_gathers == 50
        assert result.throughput == 111.1


# =============================================================================
# Create Processing Job Function Tests
# =============================================================================

class TestCreateProcessingJob:
    """Tests for create_processing_job convenience function."""

    @patch('utils.ray_orchestration.processing_job_adapter.get_job_manager')
    def test_create_processing_job_local(self, mock_get_manager, mock_job_manager):
        """Test creating a local processing job."""
        from utils.ray_orchestration.processing_job_adapter import create_processing_job

        mock_get_manager.return_value = mock_job_manager

        adapter = create_processing_job(
            input_storage_dir='/tmp/input',
            output_storage_dir='/tmp/output',
            processor_config={'class_name': 'Test'},
            job_name='Test Job',
            use_ray=False,
        )

        assert adapter is not None
        assert adapter._job_name == 'Test Job'

    @patch('utils.ray_orchestration.processing_job_adapter.get_job_manager')
    def test_create_processing_job_with_qt_bridge(self, mock_get_manager, mock_job_manager, mock_qt_bridge):
        """Test creating a processing job with Qt bridge."""
        from utils.ray_orchestration.processing_job_adapter import create_processing_job

        mock_get_manager.return_value = mock_job_manager

        adapter = create_processing_job(
            input_storage_dir='/tmp/input',
            output_storage_dir='/tmp/output',
            processor_config={'class_name': 'Test'},
            qt_bridge=mock_qt_bridge,
        )

        assert adapter._qt_bridge == mock_qt_bridge


# =============================================================================
# Integration Tests
# =============================================================================

class TestJobManagementIntegration:
    """Integration tests for job management workflow."""

    def test_job_lifecycle_states(self):
        """Test job state transitions through lifecycle."""
        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)

        assert job.state == JobState.CREATED

        job.mark_started()
        assert job.state == JobState.RUNNING

        job.mark_completed()
        assert job.state == JobState.COMPLETED

    def test_job_failure_state(self):
        """Test job failure state transition."""
        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Test error")

        assert job.state == JobState.FAILED
        assert job.error_message == "Test error"

    def test_job_cancellation_state(self):
        """Test job cancellation state transition."""
        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_cancelling()

        assert job.state == JobState.CANCELLING

    @patch('utils.ray_orchestration.processing_job_adapter.get_job_manager')
    def test_full_adapter_workflow(self, mock_get_manager, mock_processing_config, mock_job_manager, mock_qt_bridge):
        """Test full adapter workflow with signals."""
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter

        mock_get_manager.return_value = mock_job_manager

        adapter = ProcessingJobAdapter(
            mock_processing_config,
            job_name="Integration Test",
            qt_bridge=mock_qt_bridge,
        )
        adapter._manager = mock_job_manager

        # Submit
        job = adapter.submit()

        # Verify job created and signal emitted
        mock_job_manager.submit_job.assert_called_once()
        mock_qt_bridge.signals.emit_job_queued.assert_called_once()


class TestQtBridgeSignals:
    """Tests for Qt bridge signal emission."""

    def test_signal_emitter_job_queued(self):
        """Test job_queued signal emission."""
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter

        emitter = JobSignalEmitter()

        # Create signal spy
        received = []
        emitter.job_queued.connect(lambda jid, info: received.append(('queued', jid, info)))

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        emitter.emit_job_queued(job)

        assert len(received) == 1
        assert received[0][0] == 'queued'
        assert received[0][1] == job.id

    def test_signal_emitter_job_started(self):
        """Test job_started signal emission."""
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter

        emitter = JobSignalEmitter()

        received = []
        emitter.job_started.connect(lambda jid, info: received.append(('started', jid, info)))

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        emitter.emit_job_started(job)

        assert len(received) == 1
        assert received[0][0] == 'started'

    def test_signal_emitter_job_completed(self):
        """Test job_completed signal emission."""
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter

        emitter = JobSignalEmitter()

        received = []
        emitter.job_completed.connect(lambda jid: received.append(('completed', jid)))

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        emitter.emit_job_completed(job)

        assert len(received) == 1
        assert received[0][0] == 'completed'

    def test_signal_emitter_job_failed(self):
        """Test job_failed signal emission."""
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter

        emitter = JobSignalEmitter()

        received = []
        emitter.job_failed.connect(lambda jid, err: received.append(('failed', jid, err)))

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.error = "Test error"
        emitter.emit_job_failed(job)

        assert len(received) == 1
        assert received[0][0] == 'failed'

    def test_signal_emitter_progress(self):
        """Test job_progress signal emission."""
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter

        emitter = JobSignalEmitter()

        received = []
        emitter.job_progress.connect(lambda jid, prog: received.append(('progress', jid, prog)))

        job_id = uuid4()
        progress = {'percent': 50, 'message': 'Testing'}
        emitter.job_progress.emit(job_id, progress)

        assert len(received) == 1
        assert received[0][0] == 'progress'
        assert received[0][2]['percent'] == 50


class TestJobManagerBridge:
    """Tests for JobManagerBridge."""

    def test_bridge_singleton(self):
        """Test that get_job_bridge returns consistent bridge."""
        from utils.ray_orchestration.qt_bridge import get_job_bridge

        bridge1 = get_job_bridge()
        bridge2 = get_job_bridge()

        # Should return the same bridge
        assert bridge1 is bridge2

    def test_bridge_has_signals(self):
        """Test that bridge has signal emitter."""
        from utils.ray_orchestration.qt_bridge import get_job_bridge

        bridge = get_job_bridge()

        assert hasattr(bridge, 'signals')
        assert bridge.signals is not None


# =============================================================================
# Alert Integration Tests
# =============================================================================

class TestAlertJobIntegration:
    """Tests for alert manager job integration."""

    def test_job_failure_triggers_alert(self):
        """Test that job failure triggers an alert."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        job = Job(name="Failed Job", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Test failure")

        alerts = manager.process_job_event(job)

        assert len(alerts) == 1
        assert "Failed Job" in alerts[0].title

    def test_successful_job_no_alert(self):
        """Test that successful job doesn't trigger failure alert."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        job = Job(name="Success", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_completed()

        alerts = manager.process_job_event(job)

        assert len(alerts) == 0


# =============================================================================
# Job History Integration Tests
# =============================================================================

class TestJobHistoryIntegration:
    """Tests for job history integration with processing."""

    def test_job_saved_to_history_on_complete(self):
        """Test that completed jobs are saved to history."""
        from utils.ray_orchestration.job_manager import get_job_manager

        # Get the singleton manager (has history by default)
        manager = get_job_manager()

        # Create and complete a job with unique name
        import uuid
        unique_name = f"History Test {uuid.uuid4().hex[:8]}"

        job = manager.submit_job(
            name=unique_name,
            job_type=JobType.BATCH_PROCESS,
        )
        manager.start_job(job.id)
        manager.complete_job(job.id, result={'test': 'data'})

        # Check history
        records = manager.get_job_history()
        assert len(records) >= 1

        # Find our job
        our_record = None
        for record in records:
            if record.name == unique_name:
                our_record = record
                break

        assert our_record is not None
        assert our_record.state == JobState.COMPLETED

    def test_job_saved_to_history_on_failure(self):
        """Test that failed jobs are saved to history."""
        from utils.ray_orchestration.job_manager import get_job_manager

        manager = get_job_manager()

        import uuid
        unique_name = f"Failure History Test {uuid.uuid4().hex[:8]}"

        job = manager.submit_job(
            name=unique_name,
            job_type=JobType.BATCH_PROCESS,
        )
        manager.start_job(job.id)
        manager.fail_job(job.id, error="Test error")

        records = manager.get_job_history()

        our_record = None
        for record in records:
            if record.name == unique_name:
                our_record = record
                break

        assert our_record is not None
        assert our_record.state == JobState.FAILED
        assert our_record.error_message == "Test error"
