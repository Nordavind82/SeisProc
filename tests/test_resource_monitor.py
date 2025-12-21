"""
Tests for Resource Monitor

Tests resource monitoring, alerts, and thresholds.
"""

import pytest
import time
from datetime import datetime


class TestResourceSnapshot:
    """Tests for ResourceSnapshot."""

    def test_snapshot_creation(self):
        """Test creating a resource snapshot."""
        from utils.ray_orchestration.resource_monitor import ResourceSnapshot

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            cpu_count=8,
            memory_used_mb=8000.0,
            memory_available_mb=8000.0,
            memory_percent=50.0,
        )

        assert snapshot.cpu_percent == 50.0
        assert snapshot.cpu_count == 8
        assert snapshot.memory_percent == 50.0

    def test_memory_total(self):
        """Test memory total calculation."""
        from utils.ray_orchestration.resource_monitor import ResourceSnapshot

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=0,
            cpu_count=1,
            memory_used_mb=4000.0,
            memory_available_mb=4000.0,
            memory_percent=50.0,
        )

        assert snapshot.memory_total_mb == 8000.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from utils.ray_orchestration.resource_monitor import ResourceSnapshot

        ts = datetime.now()
        snapshot = ResourceSnapshot(
            timestamp=ts,
            cpu_percent=75.0,
            cpu_count=4,
            memory_used_mb=12000.0,
            memory_available_mb=4000.0,
            memory_percent=75.0,
        )

        data = snapshot.to_dict()

        assert data['cpu_percent'] == 75.0
        assert data['memory_percent'] == 75.0
        assert 'timestamp' in data


class TestResourceAlert:
    """Tests for ResourceAlert."""

    def test_alert_creation(self):
        """Test creating a resource alert."""
        from utils.ray_orchestration.resource_monitor import ResourceAlert

        alert = ResourceAlert(
            timestamp=datetime.now(),
            resource_type='memory',
            severity='warning',
            message='Memory usage high: 85%',
            current_value=85.0,
            threshold=80.0,
        )

        assert alert.resource_type == 'memory'
        assert alert.severity == 'warning'
        assert alert.current_value == 85.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from utils.ray_orchestration.resource_monitor import ResourceAlert

        alert = ResourceAlert(
            timestamp=datetime.now(),
            resource_type='cpu',
            severity='critical',
            message='CPU critical',
            current_value=99.0,
            threshold=95.0,
        )

        data = alert.to_dict()

        assert data['resource_type'] == 'cpu'
        assert data['severity'] == 'critical'


class TestResourceThresholds:
    """Tests for ResourceThresholds."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        from utils.ray_orchestration.resource_monitor import ResourceThresholds

        thresholds = ResourceThresholds()

        assert thresholds.memory_warning_percent == 80.0
        assert thresholds.memory_critical_percent == 95.0
        assert thresholds.cpu_warning_percent == 90.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        from utils.ray_orchestration.resource_monitor import ResourceThresholds

        thresholds = ResourceThresholds(
            memory_warning_percent=70.0,
            memory_critical_percent=90.0,
        )

        assert thresholds.memory_warning_percent == 70.0
        assert thresholds.memory_critical_percent == 90.0


class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    def test_monitor_creation(self):
        """Test creating a resource monitor."""
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(sample_interval=1.0)

        assert not monitor._running

    def test_get_current_snapshot(self):
        """Test getting current snapshot without starting monitor."""
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()

        assert snapshot is not None
        assert snapshot.cpu_count >= 1

    def test_start_stop(self):
        """Test starting and stopping monitor."""
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(sample_interval=0.1)

        monitor.start()
        assert monitor._running

        # Wait for a few samples
        time.sleep(0.3)

        monitor.stop()
        assert not monitor._running

    def test_history_collection(self):
        """Test that history is collected."""
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(sample_interval=0.1)

        monitor.start()
        time.sleep(0.5)
        monitor.stop()

        history = monitor.get_history()
        assert len(history) > 0

    def test_history_limit(self):
        """Test history size limiting."""
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(
            sample_interval=0.05,
            history_size=5,
        )

        monitor.start()
        time.sleep(0.5)  # Should collect ~10 samples
        monitor.stop()

        history = monitor.get_history()
        assert len(history) <= 5

    def test_get_summary(self):
        """Test getting resource summary."""
        from utils.ray_orchestration.resource_monitor import ResourceMonitor

        monitor = ResourceMonitor(sample_interval=0.1)

        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        summary = monitor.get_summary()

        assert 'samples' in summary
        assert 'cpu' in summary
        assert 'memory' in summary
        assert 'alerts' in summary

    def test_alert_callback(self):
        """Test alert callback is called."""
        from utils.ray_orchestration.resource_monitor import (
            ResourceMonitor,
            ResourceThresholds,
        )

        alerts_received = []

        def on_alert(alert):
            alerts_received.append(alert)

        # Set very low thresholds to trigger alerts
        thresholds = ResourceThresholds(
            memory_warning_percent=1.0,  # Will trigger on any usage
            cpu_warning_percent=1.0,
        )

        monitor = ResourceMonitor(
            thresholds=thresholds,
            sample_interval=0.1,
            alert_callback=on_alert,
        )

        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        # Should have received some alerts
        assert len(alerts_received) > 0

    def test_clear_alerts(self):
        """Test clearing alerts."""
        from utils.ray_orchestration.resource_monitor import (
            ResourceMonitor,
            ResourceThresholds,
        )

        thresholds = ResourceThresholds(
            memory_warning_percent=1.0,
        )

        monitor = ResourceMonitor(
            thresholds=thresholds,
            sample_interval=0.1,
        )

        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        # Should have alerts
        assert len(monitor.get_alerts()) > 0

        # Clear alerts
        monitor.clear_alerts()
        assert len(monitor.get_alerts()) == 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset global monitor."""
        import utils.ray_orchestration.resource_monitor as rm
        rm._monitor = None

    def teardown_method(self):
        """Cleanup."""
        from utils.ray_orchestration.resource_monitor import stop_monitoring
        import utils.ray_orchestration.resource_monitor as rm
        stop_monitoring()
        rm._monitor = None

    def test_get_resource_monitor(self):
        """Test getting global monitor."""
        from utils.ray_orchestration.resource_monitor import get_resource_monitor

        monitor1 = get_resource_monitor()
        monitor2 = get_resource_monitor()

        # Should return same instance
        assert monitor1 is monitor2

    def test_start_stop_monitoring(self):
        """Test start/stop convenience functions."""
        from utils.ray_orchestration.resource_monitor import (
            start_monitoring,
            stop_monitoring,
            get_resource_monitor,
        )

        start_monitoring()
        monitor = get_resource_monitor()
        assert monitor._running

        stop_monitoring()
        assert not monitor._running

    def test_get_current_resources(self):
        """Test getting current resources."""
        from utils.ray_orchestration.resource_monitor import get_current_resources

        snapshot = get_current_resources()

        assert snapshot is not None
        assert snapshot.cpu_count >= 1
