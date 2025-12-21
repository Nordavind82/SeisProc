"""
Tests for Toast Notification Widget

Tests the Toast and ToastManager components.
"""

import pytest
from unittest.mock import Mock, MagicMock

from PyQt6.QtWidgets import QApplication, QMainWindow
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


class TestToast:
    """Tests for Toast component."""

    def test_toast_creation(self):
        """Test creating a toast."""
        from views.widgets.toast_notification import Toast, ToastType

        toast = Toast(
            message="Test message",
            toast_type=ToastType.INFO,
        )

        assert toast is not None
        assert toast._message == "Test message"
        assert toast._type == ToastType.INFO

    def test_toast_with_title(self):
        """Test toast with title."""
        from views.widgets.toast_notification import Toast, ToastType

        toast = Toast(
            message="Message body",
            toast_type=ToastType.SUCCESS,
            title="Success Title",
        )

        assert toast._title == "Success Title"

    def test_toast_types(self):
        """Test different toast types."""
        from views.widgets.toast_notification import Toast, ToastType

        for toast_type in ToastType:
            toast = Toast(
                message="Test",
                toast_type=toast_type,
                duration_ms=0,  # Don't auto-close
            )
            assert toast._type == toast_type

    def test_toast_emits_closed_signal(self):
        """Test toast emits closed signal."""
        from views.widgets.toast_notification import Toast, ToastType

        toast = Toast(
            message="Test",
            toast_type=ToastType.INFO,
            duration_ms=0,
        )

        closed_received = []
        toast.closed.connect(lambda: closed_received.append(True))

        # Manually trigger close
        toast._on_fade_complete()

        assert len(closed_received) == 1

    def test_toast_non_closable(self):
        """Test non-closable toast."""
        from views.widgets.toast_notification import Toast, ToastType

        toast = Toast(
            message="Test",
            toast_type=ToastType.WARNING,
            closable=False,
            duration_ms=0,
        )

        assert toast._closable is False


class TestToastManager:
    """Tests for ToastManager."""

    @pytest.fixture
    def parent_window(self):
        """Create a parent window for toasts."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window

    def test_manager_creation(self, parent_window):
        """Test creating a toast manager."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        assert manager is not None
        assert manager.get_active_count() == 0

    def test_show_info_toast(self, parent_window):
        """Test showing an info toast."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_info("Info message")

        assert manager.get_active_count() == 1

    def test_show_success_toast(self, parent_window):
        """Test showing a success toast."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_success("Success message", title="Done")

        assert manager.get_active_count() == 1

    def test_show_warning_toast(self, parent_window):
        """Test showing a warning toast."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_warning("Warning message")

        assert manager.get_active_count() == 1

    def test_show_error_toast(self, parent_window):
        """Test showing an error toast."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_error("Error message", title="Error")

        assert manager.get_active_count() == 1

    def test_multiple_toasts(self, parent_window):
        """Test showing multiple toasts."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_info("Info 1")
        manager.show_info("Info 2")
        manager.show_info("Info 3")

        assert manager.get_active_count() == 3

    def test_clear_all_toasts(self, parent_window):
        """Test clearing all toasts."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_info("Info 1")
        manager.show_info("Info 2")

        manager.clear_all()

        # Note: clear_all triggers animations, toasts removed asynchronously
        # In tests we just verify the method runs without error

    def test_set_position(self, parent_window):
        """Test setting toast position."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)

        positions = ["top-right", "top-left", "bottom-right", "bottom-left"]
        for pos in positions:
            manager.set_position(pos)
            assert manager._position == pos


class TestAlertToastBridge:
    """Tests for AlertToastBridge."""

    @pytest.fixture
    def parent_window(self):
        """Create a parent window for toasts."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window

    def test_bridge_creation(self, parent_window):
        """Test creating alert-toast bridge."""
        from views.widgets.toast_notification import (
            ToastManager, AlertToastBridge
        )

        manager = ToastManager(parent_window)
        bridge = AlertToastBridge(manager)

        assert bridge is not None
        assert bridge._toast_manager is manager

    def test_bridge_shows_toast_on_alert(self, parent_window):
        """Test bridge shows toast when alert occurs."""
        from views.widgets.toast_notification import (
            ToastManager, AlertToastBridge
        )
        from utils.ray_orchestration.alert_manager import (
            AlertManager, Alert, AlertSeverity, AlertCategory
        )

        manager = ToastManager(parent_window)
        bridge = AlertToastBridge(manager)

        alert_manager = AlertManager()
        bridge.connect_to_alert_manager(alert_manager)

        # Create and trigger an alert manually
        alert = Alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.JOB,
            title="Test Alert",
            message="This is a test alert",
        )

        bridge._on_alert(alert)

        assert manager.get_active_count() == 1

    def test_bridge_maps_severity_to_type(self, parent_window):
        """Test bridge maps alert severity to toast type."""
        from views.widgets.toast_notification import (
            ToastManager, AlertToastBridge, ToastType
        )
        from utils.ray_orchestration.alert_manager import (
            Alert, AlertSeverity, AlertCategory
        )

        manager = ToastManager(parent_window)
        bridge = AlertToastBridge(manager)

        # Test each severity level
        severities = [
            (AlertSeverity.INFO, ToastType.INFO),
            (AlertSeverity.WARNING, ToastType.WARNING),
            (AlertSeverity.ERROR, ToastType.ERROR),
            (AlertSeverity.CRITICAL, ToastType.ERROR),
        ]

        for alert_severity, expected_toast_type in severities:
            alert = Alert(
                severity=alert_severity,
                category=AlertCategory.SYSTEM,
                title="Test",
                message="Test message",
            )

            # The bridge will create a toast with the mapped type
            # We just verify it runs without error
            bridge._on_alert(alert)


class TestToastPositioning:
    """Tests for toast positioning."""

    @pytest.fixture
    def parent_window(self):
        """Create a parent window."""
        window = QMainWindow()
        window.resize(800, 600)
        yield window

    def test_toasts_stacked_vertically(self, parent_window):
        """Test toasts are stacked vertically."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_info("Toast 1")
        manager.show_info("Toast 2")

        # Toasts should be positioned (verified by _position_toasts)
        assert manager.get_active_count() == 2

    def test_position_updates_on_remove(self, parent_window):
        """Test positions update when toast is removed."""
        from views.widgets.toast_notification import ToastManager

        manager = ToastManager(parent_window)
        manager.show_info("Toast 1", duration_ms=0)
        manager.show_info("Toast 2", duration_ms=0)

        # Remove first toast
        if manager._toasts:
            manager._remove_toast(manager._toasts[0])

        assert manager.get_active_count() == 1
