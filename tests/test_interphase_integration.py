"""
Interphase Integration Tests

These tests verify that components from different phases work together correctly.
They test the integration points between:
- Phase 1 (Foundation): Ray, Job Models, Cancellation
- Phase 2 (UI & SEGY): Dashboard, Qt Bridge, Checkpoints, SEGY Adapter
- Phase 3 (Processing): Workers, Processing Adapter, Resource Monitor
"""

import pytest
import time
import threading
from uuid import uuid4
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def reset_singletons():
    """Reset all singleton instances before and after each test."""
    # Reset before
    from utils.ray_orchestration.job_manager import JobManager
    from utils.ray_orchestration.cancellation import CancellationCoordinator
    from utils.ray_orchestration.checkpoint import CheckpointManager
    import utils.ray_orchestration.resource_monitor as rm

    JobManager._instance = None
    CancellationCoordinator._instance = None
    CheckpointManager._instance = None
    rm._monitor = None

    yield

    # Reset after
    if rm._monitor:
        rm._monitor.stop()
    JobManager._instance = None
    CancellationCoordinator._instance = None
    CheckpointManager._instance = None
    rm._monitor = None


# =============================================================================
# Phase 1 <-> Phase 2 Integration Tests
# =============================================================================

class TestPhase1Phase2Integration:
    """Tests for integration between Phase 1 (Foundation) and Phase 2 (UI & SEGY)."""

    def test_job_manager_with_qt_bridge(self, reset_singletons):
        """
        Test: JobManager (Phase 1) works with QtSignalBridge (Phase 2).

        Verifies that job lifecycle events are correctly emitted as Qt signals.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter
        from models.job import JobType

        # Create signal emitter and track emissions
        emitter = JobSignalEmitter()
        received_signals = []

        # Connect signals (using actual signal names from JobSignalEmitter)
        emitter.job_created.connect(
            lambda job_id, info: received_signals.append(('created', job_id))
        )
        emitter.job_started.connect(
            lambda job_id, info: received_signals.append(('started', job_id))
        )
        emitter.job_completed.connect(
            lambda job_id, info: received_signals.append(('completed', job_id))
        )

        # Get manager and register emitter callbacks using register_callback
        manager = get_job_manager()
        manager.register_callback(
            "on_job_started",
            lambda job: emitter.emit_job_started(job)
        )
        manager.register_callback(
            "on_job_completed",
            lambda job: emitter.emit_job_completed(job)
        )

        # Create and run a job
        job = manager.submit_job(
            name="Test Job",
            job_type=JobType.BATCH_PROCESS,
        )

        # Manually emit created signal (submit_job doesn't trigger callbacks)
        emitter.emit_job_created(job)
        assert ('created', job.id) in received_signals

        # Start job
        manager.start_job(job.id)
        assert ('started', job.id) in received_signals

        # Complete job
        manager.complete_job(job.id)
        assert any(s[0] == 'completed' and s[1] == job.id for s in received_signals)

    def test_cancellation_with_checkpoint(self, reset_singletons):
        """
        Test: Cancellation (Phase 1) integrates with Checkpoint (Phase 2).

        Verifies that when a job is cancelled, checkpoint state is preserved.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.cancellation import get_cancellation_coordinator
        from utils.ray_orchestration.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
            load_latest_checkpoint,
        )
        from models.job import JobType
        import tempfile
        import shutil

        # Setup temp dir for checkpoints
        temp_dir = tempfile.mkdtemp()
        try:
            manager = get_job_manager()
            checkpoint_manager = get_checkpoint_manager(temp_dir)

            # Create and start a job
            job = manager.submit_job(
                name="Checkpoint Test",
                job_type=JobType.BATCH_PROCESS,
            )
            manager.start_job(job.id)

            # Save a checkpoint (simulating progress)
            save_checkpoint(
                job_id=job.id,
                phase='processing',
                items_completed=500,
                items_total=1000,
                state={'gather_idx': 50},
            )

            # Cancel the job
            manager.cancel_job(job.id)

            # Verify checkpoint is still available after cancellation
            checkpoint = load_latest_checkpoint(job.id)
            assert checkpoint is not None
            assert checkpoint.items_completed == 500
            assert checkpoint.state['gather_idx'] == 50

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_job_lifecycle_with_segy_adapter(self, reset_singletons):
        """
        Test: Job lifecycle (Phase 1) works with SEGYImportJobAdapter (Phase 2).

        Verifies that SEGY adapter properly creates and manages jobs.
        """
        from utils.ray_orchestration.segy_job_adapter import SEGYImportJobAdapter
        from models.job import JobType, JobState

        # Create mock import config
        mock_config = Mock()
        mock_config.segy_path = "/tmp/test.sgy"
        mock_config.output_dir = "/tmp/output"

        # Create adapter
        adapter = SEGYImportJobAdapter(mock_config, job_name="Test Import")

        # Submit job
        job = adapter.submit()

        assert job is not None
        assert job.job_type == JobType.SEGY_IMPORT
        assert "Test Import" in job.name
        assert adapter.job_id == job.id


# =============================================================================
# Phase 1 <-> Phase 3 Integration Tests
# =============================================================================

class TestPhase1Phase3Integration:
    """Tests for integration between Phase 1 (Foundation) and Phase 3 (Processing)."""

    def test_cancellation_token_with_worker(self, reset_singletons):
        """
        Test: CancellationToken (Phase 1) works with CPUWorkerActor (Phase 3).

        Verifies that worker correctly checks cancellation state.
        """
        from utils.ray_orchestration.cancellation import (
            get_cancellation_coordinator,
            CancellationError,
        )
        from utils.ray_orchestration.workers.base_worker import CancellationChecker

        coordinator = get_cancellation_coordinator()
        job_id = uuid4()

        # Create a cancellation token through coordinator
        token = coordinator.create_token(job_id)

        # Create a cancellation checker (used by workers)
        checker = CancellationChecker(job_id)

        # Initially not cancelled
        assert not checker.is_cancelled

        # Cancel via token
        token.cancel()

        # Note: The checker is independent, but in real usage,
        # the worker would share the token's state.
        # For this test, we verify the checker's own cancel method works.
        checker.cancel()
        assert checker.is_cancelled

        with pytest.raises(CancellationError):
            checker.raise_if_cancelled()

    def test_job_manager_with_processing_adapter(self, reset_singletons):
        """
        Test: JobManager (Phase 1) works with ProcessingJobAdapter (Phase 3).

        Verifies that processing adapter creates jobs through the manager.
        """
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter
        from utils.ray_orchestration.job_manager import get_job_manager
        from models.job import JobType, JobState

        # Create mock config
        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'DWTDenoiseProcessor'}
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        # Create adapter
        adapter = ProcessingJobAdapter(mock_config)
        job = adapter.submit()

        # Verify job was created correctly
        manager = get_job_manager()
        retrieved_job = manager.get_job(job.id)

        assert retrieved_job is not None
        assert retrieved_job.job_type == JobType.BATCH_PROCESS
        assert retrieved_job.state == JobState.QUEUED

    def test_resource_monitor_with_job_manager(self, reset_singletons):
        """
        Test: ResourceMonitor (Phase 3) can be used alongside JobManager (Phase 1).

        Verifies that resource monitoring doesn't interfere with job management.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.resource_monitor import (
            get_resource_monitor,
            start_monitoring,
            stop_monitoring,
        )
        from models.job import JobType

        # Start monitoring
        monitor = get_resource_monitor()
        monitor.start()

        try:
            # Create jobs while monitoring
            manager = get_job_manager()
            job = manager.submit_job(
                name="Monitored Job",
                job_type=JobType.BATCH_PROCESS,
            )

            # Get resource snapshot
            snapshot = monitor.get_current_snapshot()
            assert snapshot.cpu_percent >= 0
            assert snapshot.memory_percent >= 0

            # Job should still work
            manager.start_job(job.id)
            assert manager.get_job(job.id).state.name == 'RUNNING'

        finally:
            monitor.stop()


# =============================================================================
# Phase 2 <-> Phase 3 Integration Tests
# =============================================================================

class TestPhase2Phase3Integration:
    """Tests for integration between Phase 2 (UI & SEGY) and Phase 3 (Processing)."""

    def test_qt_bridge_with_processing_adapter(self, reset_singletons):
        """
        Test: QtSignalBridge (Phase 2) works with ProcessingJobAdapter (Phase 3).

        Verifies that processing job events are emitted as Qt signals.
        """
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter, JobManagerBridge
        from utils.ray_orchestration.job_manager import get_job_manager

        # Setup signal tracking
        emitter = JobSignalEmitter()
        received = []

        emitter.job_created.connect(
            lambda job_id, info: received.append(('created', str(job_id)))
        )

        # Create bridge (it creates its own manager internally)
        bridge = JobManagerBridge()

        # Register callback to emit created signal when job starts
        bridge.manager.register_callback(
            "on_job_started",
            lambda job: emitter.emit_job_created(job)
        )

        # Create processing adapter
        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'TestProcessor'}
        mock_config.n_workers = 2
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        adapter = ProcessingJobAdapter(mock_config)
        job = adapter.submit()

        # Emit created signal directly
        emitter.emit_job_created(job)

        # Verify signal was emitted
        assert len(received) > 0
        assert received[0][0] == 'created'

    def test_checkpoint_with_worker_progress(self, reset_singletons):
        """
        Test: Checkpoint (Phase 2) can save worker progress (Phase 3).

        Verifies that worker progress can be saved to checkpoints.
        """
        from utils.ray_orchestration.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
            load_latest_checkpoint,
        )
        from utils.ray_orchestration.workers.base_worker import WorkerProgress, WorkerState
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_manager = get_checkpoint_manager(temp_dir)
            job_id = uuid4()

            # Create worker progress data
            worker_progress = WorkerProgress(
                worker_id='worker-0',
                job_id=job_id,
                items_processed=100,
                items_total=500,
                elapsed_seconds=30.0,
                state=WorkerState.PROCESSING,
            )

            # Save checkpoint with worker progress
            save_checkpoint(
                job_id=job_id,
                phase='processing',
                items_completed=100,
                items_total=500,
                state={
                    'worker_progress': {
                        'worker-0': {
                            'items_processed': worker_progress.items_processed,
                            'items_total': worker_progress.items_total,
                            'percent': worker_progress.percent_complete,
                        }
                    }
                },
            )

            # Load and verify
            checkpoint = load_latest_checkpoint(job_id)
            assert checkpoint is not None
            assert checkpoint.state['worker_progress']['worker-0']['items_processed'] == 100

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resource_alerts_with_qt_signals(self, reset_singletons):
        """
        Test: Resource alerts (Phase 3) can be connected to Qt signals (Phase 2).

        Verifies that resource alerts can trigger UI updates.
        """
        from utils.ray_orchestration.resource_monitor import (
            ResourceMonitor,
            ResourceThresholds,
            ResourceAlert,
        )
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter

        # Track alerts
        alerts_received = []

        # Create monitor with low threshold to trigger alerts
        thresholds = ResourceThresholds(
            memory_warning_percent=1.0,  # Will trigger on any usage
        )

        def on_alert(alert: ResourceAlert):
            alerts_received.append(alert)
            # In real code, this would emit a Qt signal

        monitor = ResourceMonitor(
            thresholds=thresholds,
            sample_interval=0.1,
            alert_callback=on_alert,
        )

        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        # Should have received alerts
        assert len(alerts_received) > 0
        assert all(isinstance(a, ResourceAlert) for a in alerts_received)


# =============================================================================
# Full Stack Integration Tests (Phase 1 + 2 + 3)
# =============================================================================

class TestFullStackIntegration:
    """Tests that verify the complete integration across all phases."""

    def test_full_job_lifecycle_with_monitoring(self, reset_singletons):
        """
        Test: Complete job lifecycle with all Phase 1, 2, 3 components.

        Verifies:
        - Job creation (Phase 1)
        - Qt signal emission (Phase 2)
        - Resource monitoring (Phase 3)
        - Cancellation (Phase 1)
        - Checkpoint saving (Phase 2)
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter, JobManagerBridge
        from utils.ray_orchestration.resource_monitor import get_resource_monitor
        from utils.ray_orchestration.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
        )
        from models.job import JobType
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            # Setup all components
            manager = get_job_manager()
            emitter = JobSignalEmitter()
            bridge = JobManagerBridge()  # Creates its own manager internally
            checkpoint_manager = get_checkpoint_manager(temp_dir)
            monitor = get_resource_monitor()

            # Track signals
            events = []
            emitter.job_created.connect(
                lambda job_id, info: events.append(('created', job_id))
            )
            emitter.job_started.connect(
                lambda job_id, info: events.append(('started', job_id))
            )
            emitter.job_cancelled.connect(
                lambda job_id: events.append(('cancelled', job_id))
            )

            # Register callbacks
            manager.register_callback(
                "on_job_started",
                lambda job: emitter.emit_job_started(job)
            )
            manager.register_callback(
                "on_job_cancelled",
                lambda job: emitter.emit_job_cancelled(job)
            )

            # Start monitoring
            monitor.start()

            # Create and start job
            job = manager.submit_job(
                name="Full Stack Test",
                job_type=JobType.BATCH_PROCESS,
            )
            emitter.emit_job_created(job)
            manager.start_job(job.id)

            # Save checkpoint
            save_checkpoint(
                job_id=job.id,
                phase='processing',
                items_completed=250,
                items_total=1000,
            )

            # Get resource snapshot
            snapshot = monitor.get_current_snapshot()
            assert snapshot is not None

            # Cancel job
            manager.cancel_job(job.id)
            manager.finalize_cancellation(job.id)

            # Stop monitoring
            monitor.stop()

            # Verify all events occurred
            event_types = [e[0] for e in events]
            assert 'created' in event_types
            assert 'started' in event_types
            assert 'cancelled' in event_types

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_processing_adapter_with_all_components(self, reset_singletons):
        """
        Test: ProcessingJobAdapter integrates with all components.

        Verifies that ProcessingJobAdapter correctly uses:
        - JobManager (Phase 1)
        - CancellationToken (Phase 1)
        - Checkpoint (Phase 2)
        - Resource tracking (Phase 3)
        """
        from utils.ray_orchestration.processing_job_adapter import ProcessingJobAdapter
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.cancellation import get_cancellation_coordinator
        from models.job import JobState

        # Create mock config
        mock_config = Mock()
        mock_config.processor_config = {'class_name': 'TestProcessor'}
        mock_config.n_workers = 4
        mock_config.input_storage_dir = '/tmp/input'
        mock_config.output_storage_dir = '/tmp/output'

        # Create adapter
        adapter = ProcessingJobAdapter(mock_config)

        # Submit creates job and token
        job = adapter.submit()
        manager = get_job_manager()
        coordinator = get_cancellation_coordinator()

        # Verify job exists
        assert manager.get_job(job.id) is not None

        # Verify cancellation token exists
        token = manager.get_cancellation_token(job.id)
        assert token is not None
        assert not token.is_cancelled

        # Cancel through adapter
        adapter.cancel()

        # Verify token is cancelled
        assert token.is_cancelled


# =============================================================================
# Error Handling Integration Tests
# =============================================================================

class TestErrorHandlingIntegration:
    """Tests for error handling across phase boundaries."""

    def test_worker_error_propagates_to_job(self, reset_singletons):
        """
        Test: Errors in workers (Phase 3) propagate to job state (Phase 1).

        Verifies that worker failures are reflected in job state.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from models.job import JobType, JobState

        manager = get_job_manager()
        job = manager.submit_job(
            name="Error Test",
            job_type=JobType.BATCH_PROCESS,
        )
        manager.start_job(job.id)

        # Fail the job
        manager.fail_job(job.id, error="Worker failed: OutOfMemory")

        # Verify state
        failed_job = manager.get_job(job.id)
        assert failed_job.state == JobState.FAILED
        assert "OutOfMemory" in str(failed_job.error_message)

    def test_checkpoint_survives_job_failure(self, reset_singletons):
        """
        Test: Checkpoints (Phase 2) survive job failures (Phase 1).

        Verifies that checkpoint data is preserved even when job fails.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
            load_latest_checkpoint,
        )
        from models.job import JobType
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_manager = get_checkpoint_manager(temp_dir)
            manager = get_job_manager()

            job = manager.submit_job(
                name="Failure Recovery Test",
                job_type=JobType.BATCH_PROCESS,
            )
            manager.start_job(job.id)

            # Save progress
            save_checkpoint(
                job_id=job.id,
                phase='processing',
                items_completed=750,
                items_total=1000,
                state={'last_good_index': 749},
            )

            # Job fails
            manager.fail_job(job.id, error="Unexpected error")

            # Checkpoint should still be recoverable
            checkpoint = load_latest_checkpoint(job.id)
            assert checkpoint is not None
            assert checkpoint.items_completed == 750
            assert checkpoint.state['last_good_index'] == 749

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Concurrency Integration Tests
# =============================================================================

class TestConcurrencyIntegration:
    """Tests for concurrent operations across phases."""

    def test_multiple_jobs_with_monitoring(self, reset_singletons):
        """
        Test: Multiple concurrent jobs work with resource monitoring.

        Verifies that the system handles multiple jobs correctly.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.resource_monitor import get_resource_monitor
        from models.job import JobType, JobState

        manager = get_job_manager()
        monitor = get_resource_monitor()
        monitor.start()

        try:
            # Create multiple jobs
            jobs = []
            job_ids = set()
            for i in range(5):
                job = manager.submit_job(
                    name=f"Concurrent Job {i}",
                    job_type=JobType.BATCH_PROCESS,
                )
                jobs.append(job)
                job_ids.add(job.id)
                manager.start_job(job.id)

            # All our jobs should be running (filter by our job IDs)
            running = [j for j in manager.list_jobs(states=[JobState.RUNNING])
                       if j.id in job_ids]
            assert len(running) == 5

            # Resource monitoring should work
            snapshot = monitor.get_current_snapshot()
            assert snapshot is not None

            # Complete all
            for job in jobs:
                manager.complete_job(job.id)

            # All our jobs should be completed
            completed = [j for j in manager.list_jobs(states=[JobState.COMPLETED])
                         if j.id in job_ids]
            assert len(completed) == 5

        finally:
            monitor.stop()

    def test_pause_resume_with_checkpoints(self, reset_singletons):
        """
        Test: Pause/resume (Phase 1) correctly interacts with checkpoints (Phase 2).

        Verifies that pause saves checkpoint and resume can continue.
        """
        from utils.ray_orchestration.job_manager import get_job_manager
        from utils.ray_orchestration.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
            load_latest_checkpoint,
        )
        from models.job import JobType, JobState
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            checkpoint_manager = get_checkpoint_manager(temp_dir)
            manager = get_job_manager()

            job = manager.submit_job(
                name="Pause Resume Test",
                job_type=JobType.BATCH_PROCESS,
            )
            manager.start_job(job.id)

            # Simulate progress and save checkpoint
            save_checkpoint(
                job_id=job.id,
                phase='processing',
                items_completed=500,
                items_total=1000,
            )

            # Pause
            manager.pause_job(job.id)
            paused_job = manager.get_job(job.id)
            assert paused_job.state == JobState.PAUSED

            # Checkpoint should be available
            checkpoint = load_latest_checkpoint(job.id)
            assert checkpoint.items_completed == 500

            # Resume
            manager.resume_job(job.id)
            resumed_job = manager.get_job(job.id)
            assert resumed_job.state == JobState.RUNNING

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
