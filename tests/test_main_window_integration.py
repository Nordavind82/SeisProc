"""
Integration Tests for Main Window Job Integration

Tests the complete UI workflow for Phase 4 components:
- MainWindowJobIntegration initialization
- Dashboard signal connections
- Toast notification pipeline
- Alert manager integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4

from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu
from PyQt6.QtCore import Qt
import sys


# Global QApplication instance for tests
_app = None


def get_app():
    """Get or create QApplication instance."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    """Ensure QApplication exists for all tests."""
    app = get_app()
    yield app


class TestMainWindowJobIntegration:
    """Tests for MainWindowJobIntegration setup and configuration."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_integration_creation(self, main_window):
        """Test creating integration instance."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        assert integration is not None
        assert integration._window is main_window

    def test_integration_setup_creates_docks(self, main_window):
        """Test setup creates dock widgets."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        # Check docks were created
        assert integration._job_dock is not None
        assert integration._resource_dock is not None
        assert integration._analytics_dock is not None

    def test_integration_setup_creates_toast_manager(self, main_window):
        """Test setup creates toast manager."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        assert integration._toast_manager is not None
        assert integration._alert_toast_bridge is not None

    def test_integration_setup_adds_status_bar_monitor(self, main_window):
        """Test setup adds compact monitor to status bar."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        assert integration._compact_monitor is not None

    def test_docks_hidden_by_default(self, main_window):
        """Test docks are hidden by default."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup(
            show_job_dock=False,
            show_resource_dock=False,
            show_analytics_dock=False,
        )

        assert not integration._job_dock.isVisible()
        assert not integration._resource_dock.isVisible()
        assert not integration._analytics_dock.isVisible()

    def test_show_job_dashboard(self, main_window):
        """Test showing job dashboard dock."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()
        main_window.show()  # Need to show window for dock visibility

        integration.show_job_dashboard()

        assert integration._job_dock.isVisible()

    def test_show_resource_monitor(self, main_window):
        """Test showing resource monitor dock."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()
        main_window.show()

        integration.show_resource_monitor()

        assert integration._resource_dock.isVisible()

    def test_show_job_analytics(self, main_window):
        """Test showing job analytics dock."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()
        main_window.show()

        integration.show_job_analytics()

        assert integration._analytics_dock.isVisible()

    def test_show_toast(self, main_window):
        """Test showing toast notification."""
        from views.main_window_integration import MainWindowJobIntegration
        from views.widgets.toast_notification import ToastType

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        integration.show_toast("Test message", ToastType.INFO)

        assert integration._toast_manager.get_active_count() == 1

    def test_cleanup(self, main_window):
        """Test cleanup method."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        # Should not raise
        integration.cleanup()


class TestSignalBridgeConnection:
    """Tests for signal bridge connections."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_connect_bridge_signals_with_mock_bridge(self, main_window):
        """Test connecting signal bridge."""
        from views.main_window_integration import MainWindowJobIntegration

        # Create mock signal bridge
        mock_bridge = MagicMock()
        mock_signals = MagicMock()
        mock_bridge.signals = mock_signals

        integration = MainWindowJobIntegration(main_window)
        integration.setup(signal_bridge=mock_bridge)

        # Verify signals were connected
        mock_signals.job_queued.connect.assert_called()
        mock_signals.job_started.connect.assert_called()
        mock_signals.job_progress.connect.assert_called()

    def test_job_started_signal_updates_dashboard(self, main_window):
        """Test job started signal updates dashboard."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        dashboard = integration.get_job_dashboard()
        job_id = uuid4()
        job_info = {"name": "Test Job", "job_type": "BATCH_PROCESS"}

        # Trigger the slot directly
        dashboard.on_job_started(job_id, job_info)

        assert dashboard.active_jobs_count() == 1

    def test_progress_updated_signal(self, main_window):
        """Test progress update signal."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        dashboard = integration.get_job_dashboard()
        job_id = uuid4()
        job_info = {"name": "Test Job", "job_type": "BATCH_PROCESS"}

        # Start a job first
        dashboard.on_job_started(job_id, job_info)

        # Update progress
        progress = {"percent": 50, "message": "Processing..."}
        dashboard.on_progress_updated(job_id, progress)

        assert dashboard.get_job_progress(job_id) == 50

    def test_job_state_changed_to_completed(self, main_window):
        """Test job completed state change."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        dashboard = integration.get_job_dashboard()
        job_id = uuid4()
        job_info = {"name": "Test Job", "job_type": "BATCH_PROCESS"}

        # Start a job
        dashboard.on_job_started(job_id, job_info)
        assert dashboard.active_jobs_count() == 1

        # Complete the job
        dashboard.on_job_state_changed(job_id, "COMPLETED")

        # Job should be moved to history
        assert dashboard.active_jobs_count() == 0


class TestAlertToastPipeline:
    """Tests for alert to toast notification pipeline."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_alert_bridge_connection(self, main_window):
        """Test alert bridge connects to alert manager."""
        from views.main_window_integration import MainWindowJobIntegration
        from utils.ray_orchestration.alert_manager import AlertManager

        alert_manager = AlertManager()
        integration = MainWindowJobIntegration(main_window)
        integration.setup(alert_manager=alert_manager)

        # Bridge should be connected
        assert integration._alert_toast_bridge is not None
        assert len(alert_manager._callbacks) > 0

    def test_alert_triggers_toast(self, main_window):
        """Test alert triggers toast notification."""
        from views.main_window_integration import MainWindowJobIntegration
        from utils.ray_orchestration.alert_manager import (
            AlertManager, Alert, AlertSeverity, AlertCategory
        )

        alert_manager = AlertManager()
        integration = MainWindowJobIntegration(main_window)
        integration.setup(alert_manager=alert_manager)

        # Create and process an alert
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.JOB,
            title="Test Alert",
            message="This is a test alert",
        )

        # Trigger the bridge callback directly
        integration._alert_toast_bridge._on_alert(alert)

        # Toast should be shown
        assert integration._toast_manager.get_active_count() == 1


class TestViewMenuActions:
    """Tests for View menu action integration."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_add_view_menu_actions(self, main_window):
        """Test adding view menu actions."""
        from views.main_window_integration import (
            MainWindowJobIntegration,
            add_view_menu_actions,
        )

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        # Create a view menu
        view_menu = QMenu("View", main_window)

        # Add actions
        add_view_menu_actions(view_menu, integration)

        # Verify actions were added
        actions = view_menu.actions()
        action_texts = [a.text() for a in actions if not a.isSeparator()]

        assert "&Job Dashboard" in action_texts
        assert "&Resource Monitor" in action_texts
        assert "Job &Analytics" in action_texts


class TestJobDashboardControlSignals:
    """Tests for job control signals from dashboard."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_cancel_job_signal(self, main_window):
        """Test cancel job signal is forwarded."""
        from views.main_window_integration import MainWindowJobIntegration

        mock_manager = MagicMock()
        integration = MainWindowJobIntegration(main_window)
        integration.setup(job_manager=mock_manager)

        job_id = uuid4()
        integration._on_cancel_job(job_id)

        mock_manager.cancel_job.assert_called_once_with(job_id)

    def test_pause_job_signal(self, main_window):
        """Test pause job signal is forwarded."""
        from views.main_window_integration import MainWindowJobIntegration

        mock_manager = MagicMock()
        integration = MainWindowJobIntegration(main_window)
        integration.setup(job_manager=mock_manager)

        job_id = uuid4()
        integration._on_pause_job(job_id)

        mock_manager.pause_job.assert_called_once_with(job_id)

    def test_resume_job_signal(self, main_window):
        """Test resume job signal is forwarded."""
        from views.main_window_integration import MainWindowJobIntegration

        mock_manager = MagicMock()
        integration = MainWindowJobIntegration(main_window)
        integration.setup(job_manager=mock_manager)

        job_id = uuid4()
        integration._on_resume_job(job_id)

        mock_manager.resume_job.assert_called_once_with(job_id)

    def test_cancel_all_signal(self, main_window):
        """Test cancel all signal is forwarded."""
        from views.main_window_integration import MainWindowJobIntegration

        mock_manager = MagicMock()
        integration = MainWindowJobIntegration(main_window)
        integration.setup(job_manager=mock_manager)

        integration._on_cancel_all()

        mock_manager.cancel_all.assert_called_once()


class TestResourceAlertToast:
    """Tests for resource alert to toast conversion."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_critical_resource_alert_shows_error_toast(self, main_window):
        """Test critical resource alert shows error toast."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        # Create mock critical alert
        alert = MagicMock()
        alert.severity = 'critical'
        alert.message = "High CPU usage"

        integration._on_resource_alert_toast(alert)

        assert integration._toast_manager.get_active_count() == 1

    def test_warning_resource_alert_shows_warning_toast(self, main_window):
        """Test warning resource alert shows warning toast."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        # Create mock warning alert
        alert = MagicMock()
        alert.severity = 'warning'
        alert.message = "Memory usage increasing"

        integration._on_resource_alert_toast(alert)

        assert integration._toast_manager.get_active_count() == 1


class TestEndToEndWorkflow:
    """End-to-end tests for complete workflows."""

    @pytest.fixture
    def main_window(self):
        """Create a test main window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window
        window.close()

    def test_complete_job_lifecycle(self, main_window):
        """Test complete job lifecycle: queue -> start -> progress -> complete."""
        from views.main_window_integration import MainWindowJobIntegration

        integration = MainWindowJobIntegration(main_window)
        integration.setup()

        dashboard = integration.get_job_dashboard()
        job_id = uuid4()
        job_info = {"name": "Test Job", "job_type": "BATCH_PROCESS", "priority": "NORMAL"}

        # Queue job
        dashboard.on_job_queued(job_id, job_info)

        # Dequeue (moving to active)
        dashboard.on_job_dequeued(job_id)

        # Start job
        dashboard.on_job_started(job_id, job_info)
        assert dashboard.active_jobs_count() == 1

        # Update progress
        for percent in [25, 50, 75, 100]:
            progress = {"percent": percent, "message": f"Processing {percent}%"}
            dashboard.on_progress_updated(job_id, progress)

        # Complete job
        dashboard.on_job_state_changed(job_id, "COMPLETED")
        assert dashboard.active_jobs_count() == 0

    def test_job_failure_triggers_alert_and_toast(self, main_window):
        """Test job failure triggers alert and toast notification."""
        from views.main_window_integration import MainWindowJobIntegration
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )
        from models.job import Job, JobType

        # Setup with alert manager and failure rule
        alert_manager = AlertManager()
        alert_manager.add_rule(JobFailureRule(cooldown_seconds=0))

        integration = MainWindowJobIntegration(main_window)
        integration.setup(alert_manager=alert_manager)

        # Create failed job
        job = Job(name="Failed Job", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Test error")

        # Process job event
        alerts = alert_manager.process_job_event(job)

        # Alert should be generated and toast shown
        assert len(alerts) == 1
        assert integration._toast_manager.get_active_count() == 1
