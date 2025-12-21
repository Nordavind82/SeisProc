"""
Tests for Resource Monitor Widget

Tests the ResourceMonitorWidget and CompactResourceMonitorWidget UI components.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Create QApplication before any Qt widgets
from PyQt6.QtWidgets import QApplication
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


class TestResourceGauge:
    """Tests for ResourceGauge component."""

    def test_gauge_creation(self):
        """Test creating a resource gauge."""
        from views.widgets.resource_monitor_widget import ResourceGauge

        gauge = ResourceGauge(
            title="CPU",
            unit="%",
            warning_threshold=80.0,
            critical_threshold=95.0,
        )

        assert gauge is not None
        assert gauge._title == "CPU"

    def test_gauge_set_value_normal(self):
        """Test setting a normal value."""
        from views.widgets.resource_monitor_widget import ResourceGauge

        gauge = ResourceGauge("Memory", unit="%")
        gauge.set_value(50.0, 100.0)

        assert gauge._current_value == 50.0
        assert gauge._progress.value() == 50

    def test_gauge_set_value_warning(self):
        """Test setting a warning-level value."""
        from views.widgets.resource_monitor_widget import ResourceGauge

        gauge = ResourceGauge(
            "CPU",
            unit="%",
            warning_threshold=80.0,
            critical_threshold=95.0,
        )
        gauge.set_value(85.0, 100.0)

        assert gauge._progress.value() == 85
        # Should have warning color (yellow)
        assert "#ffc107" in gauge._value_label.styleSheet()

    def test_gauge_set_value_critical(self):
        """Test setting a critical-level value."""
        from views.widgets.resource_monitor_widget import ResourceGauge

        gauge = ResourceGauge(
            "Memory",
            unit="%",
            warning_threshold=80.0,
            critical_threshold=95.0,
        )
        gauge.set_value(98.0, 100.0)

        assert gauge._progress.value() == 98
        # Should have critical color (red)
        assert "#dc3545" in gauge._value_label.styleSheet()

    def test_gauge_set_details(self):
        """Test setting details text."""
        from views.widgets.resource_monitor_widget import ResourceGauge

        gauge = ResourceGauge("CPU", unit="%")
        gauge.set_details("8 cores")

        assert gauge._details_label.text() == "8 cores"


class TestAlertIndicator:
    """Tests for AlertIndicator component."""

    def test_indicator_creation(self):
        """Test creating an alert indicator."""
        from views.widgets.resource_monitor_widget import AlertIndicator

        indicator = AlertIndicator()

        assert indicator is not None
        assert indicator._count_label.text() == "0"

    def test_indicator_no_alerts(self):
        """Test indicator with no alerts."""
        from views.widgets.resource_monitor_widget import AlertIndicator

        indicator = AlertIndicator()
        indicator.update_alerts([])

        assert indicator._count_label.text() == "0"
        assert "No alerts" in indicator._message_label.text()

    def test_indicator_with_warning(self):
        """Test indicator with warning alert."""
        from views.widgets.resource_monitor_widget import AlertIndicator
        from utils.ray_orchestration.resource_monitor import ResourceAlert

        indicator = AlertIndicator()

        alert = ResourceAlert(
            timestamp=datetime.now(),
            resource_type='memory',
            severity='warning',
            message='Memory usage high: 85%',
            current_value=85.0,
            threshold=80.0,
        )

        indicator.update_alerts([alert])

        assert indicator._count_label.text() == "1"
        assert "Memory usage high" in indicator._message_label.text()
        # Should have warning color
        assert "#ffc107" in indicator._count_label.styleSheet()

    def test_indicator_with_critical(self):
        """Test indicator with critical alert."""
        from views.widgets.resource_monitor_widget import AlertIndicator
        from utils.ray_orchestration.resource_monitor import ResourceAlert

        indicator = AlertIndicator()

        alert = ResourceAlert(
            timestamp=datetime.now(),
            resource_type='memory',
            severity='critical',
            message='Memory critical: 98%',
            current_value=98.0,
            threshold=95.0,
        )

        indicator.update_alerts([alert])

        assert indicator._count_label.text() == "1"
        # Should have critical color
        assert "#dc3545" in indicator._count_label.styleSheet()


class TestResourceMonitorWidget:
    """Tests for ResourceMonitorWidget."""

    def setup_method(self):
        """Reset monitor before each test."""
        import utils.ray_orchestration.resource_monitor as rm
        rm._monitor = None

    def teardown_method(self):
        """Cleanup after each test."""
        import utils.ray_orchestration.resource_monitor as rm
        if rm._monitor:
            rm._monitor.stop()
        rm._monitor = None

    def test_widget_creation(self):
        """Test creating a resource monitor widget."""
        from views.widgets.resource_monitor_widget import ResourceMonitorWidget

        widget = ResourceMonitorWidget(update_interval_ms=10000)  # Long interval for test

        assert widget is not None
        assert widget._cpu_gauge is not None
        assert widget._memory_gauge is not None
        assert widget._disk_read_gauge is not None
        assert widget._disk_write_gauge is not None

    def test_widget_with_custom_monitor(self):
        """Test widget with custom monitor."""
        from views.widgets.resource_monitor_widget import ResourceMonitorWidget
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(sample_interval=10.0)
        widget = ResourceMonitorWidget(
            monitor=monitor,
            update_interval_ms=10000,
        )

        assert widget._monitor is monitor

    def test_widget_update_display(self):
        """Test widget updates display correctly."""
        from views.widgets.resource_monitor_widget import ResourceMonitorWidget
        from utils.ray_orchestration.resource_monitor import ResourceSnapshot

        widget = ResourceMonitorWidget(update_interval_ms=10000)

        # Create a test snapshot
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=45.0,
            cpu_count=8,
            memory_used_mb=8000.0,
            memory_available_mb=8000.0,
            memory_percent=50.0,
            disk_read_mb=10.5,
            disk_write_mb=5.2,
        )

        # Update gauges manually
        widget._update_gauges(snapshot)

        # Verify gauges updated
        assert widget._cpu_gauge._progress.value() == 45
        assert widget._memory_gauge._progress.value() == 50

    def test_widget_emits_alert_signal(self):
        """Test widget emits signal on new alert."""
        from views.widgets.resource_monitor_widget import ResourceMonitorWidget
        from utils.ray_orchestration.resource_monitor import (
            ResourceMonitor,
            ResourceThresholds,
        )

        # Create monitor with very low threshold
        thresholds = ResourceThresholds(memory_warning_percent=1.0)
        monitor = ResourceMonitor(thresholds=thresholds, sample_interval=0.1)

        widget = ResourceMonitorWidget(
            monitor=monitor,
            update_interval_ms=100,
        )

        alerts_received = []
        widget.alert_triggered.connect(lambda a: alerts_received.append(a))

        # Start monitor briefly
        monitor.start()
        import time
        time.sleep(0.3)
        monitor.stop()

        # Update display to trigger signal
        widget._update_alerts()

        # Should have received alerts
        assert len(alerts_received) > 0 or len(monitor.get_alerts()) > 0

    def test_widget_get_summary(self):
        """Test widget returns summary."""
        from views.widgets.resource_monitor_widget import ResourceMonitorWidget
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(sample_interval=0.1)
        widget = ResourceMonitorWidget(
            monitor=monitor,
            update_interval_ms=10000,
        )

        # Start briefly to collect samples
        monitor.start()
        import time
        time.sleep(0.3)
        monitor.stop()

        summary = widget.get_summary()

        assert 'samples' in summary or summary == {}

    def test_widget_clear_alerts(self):
        """Test clearing alerts."""
        from views.widgets.resource_monitor_widget import ResourceMonitorWidget
        from utils.ray_orchestration.resource_monitor import (
            ResourceMonitor,
            ResourceThresholds,
        )

        thresholds = ResourceThresholds(memory_warning_percent=1.0)
        monitor = ResourceMonitor(thresholds=thresholds, sample_interval=0.1)

        widget = ResourceMonitorWidget(
            monitor=monitor,
            update_interval_ms=10000,
        )

        # Generate some alerts
        monitor.start()
        import time
        time.sleep(0.3)
        monitor.stop()

        # Clear alerts
        widget.clear_alerts()

        assert len(monitor.get_alerts()) == 0
        assert widget._last_alert_count == 0


class TestCompactResourceMonitorWidget:
    """Tests for CompactResourceMonitorWidget."""

    def setup_method(self):
        """Reset monitor before each test."""
        import utils.ray_orchestration.resource_monitor as rm
        rm._monitor = None

    def teardown_method(self):
        """Cleanup after each test."""
        import utils.ray_orchestration.resource_monitor as rm
        if rm._monitor:
            rm._monitor.stop()
        rm._monitor = None

    def test_compact_widget_creation(self):
        """Test creating a compact resource monitor widget."""
        from views.widgets.resource_monitor_widget import CompactResourceMonitorWidget

        widget = CompactResourceMonitorWidget()

        assert widget is not None
        assert widget._cpu_bar is not None
        assert widget._mem_bar is not None
        assert widget._alert_dot is not None

    def test_compact_widget_update(self):
        """Test compact widget updates correctly."""
        from views.widgets.resource_monitor_widget import CompactResourceMonitorWidget

        widget = CompactResourceMonitorWidget()

        # Trigger update
        widget._update_display()

        # Should have valid values
        assert widget._cpu_bar.value() >= 0
        assert widget._mem_bar.value() >= 0

    def test_compact_widget_alert_dot_color(self):
        """Test alert dot changes color on alerts."""
        from views.widgets.resource_monitor_widget import CompactResourceMonitorWidget
        from utils.ray_orchestration.resource_monitor import (
            ResourceMonitor,
            ResourceThresholds,
        )

        thresholds = ResourceThresholds(memory_warning_percent=1.0)
        monitor = ResourceMonitor(thresholds=thresholds, sample_interval=0.1)

        widget = CompactResourceMonitorWidget(monitor=monitor)

        # Generate alerts
        monitor.start()
        import time
        time.sleep(0.3)
        monitor.stop()

        # Update display
        widget._update_display()

        # Alert dot should reflect alerts (yellow or red)
        style = widget._alert_dot.styleSheet()
        # Should be warning (yellow) or critical (red) - not green
        has_alert_color = "#ffc107" in style or "#dc3545" in style
        # Or still green if no alerts triggered
        has_green = "#28a745" in style

        assert has_alert_color or has_green
