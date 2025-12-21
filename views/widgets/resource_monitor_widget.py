"""
Resource Monitor Widget

Displays real-time system resource usage including CPU, memory, disk I/O,
and GPU metrics with visual gauges and alerts.
"""

import logging
from typing import Optional, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QProgressBar, QGridLayout, QSizePolicy,
    QScrollArea, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

from utils.ray_orchestration.resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    ResourceAlert,
    get_resource_monitor,
)

logger = logging.getLogger(__name__)


class ResourceGauge(QFrame):
    """
    A visual gauge showing resource usage percentage.

    Displays a progress bar with color coding based on thresholds.
    """

    def __init__(
        self,
        title: str,
        unit: str = "%",
        warning_threshold: float = 80.0,
        critical_threshold: float = 95.0,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._unit = unit
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._current_value = 0.0
        self._max_value = 100.0

        self._setup_ui()

    def _setup_ui(self):
        """Set up the gauge UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(150)
        self.setMaximumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Title row
        title_row = QHBoxLayout()
        self._title_label = QLabel(self._title)
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        title_row.addWidget(self._title_label)

        title_row.addStretch()

        self._value_label = QLabel("0%")
        self._value_label.setStyleSheet("color: #28a745;")  # Green
        title_row.addWidget(self._value_label)

        layout.addLayout(title_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setMaximumHeight(12)
        layout.addWidget(self._progress)

        # Details row
        self._details_label = QLabel("")
        self._details_label.setStyleSheet("color: #6c757d; font-size: 10px;")
        layout.addWidget(self._details_label)

    def set_value(self, value: float, max_value: Optional[float] = None):
        """Update the gauge value."""
        self._current_value = value
        if max_value is not None:
            self._max_value = max_value

        # Calculate percentage
        if self._max_value > 0:
            percent = (value / self._max_value) * 100
        else:
            percent = 0

        percent = min(100, max(0, percent))
        self._progress.setValue(int(percent))

        # Update value label
        if self._unit == "%":
            self._value_label.setText(f"{percent:.1f}%")
        elif self._unit == "GB":
            self._value_label.setText(f"{value / 1024:.1f} GB")
        elif self._unit == "MB/s":
            self._value_label.setText(f"{value:.1f} MB/s")
        else:
            self._value_label.setText(f"{value:.1f} {self._unit}")

        # Update color based on thresholds
        if percent >= self._critical_threshold:
            color = "#dc3545"  # Red
            bar_color = "background-color: #dc3545;"
        elif percent >= self._warning_threshold:
            color = "#ffc107"  # Yellow
            bar_color = "background-color: #ffc107;"
        else:
            color = "#28a745"  # Green
            bar_color = "background-color: #28a745;"

        self._value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                {bar_color}
            }}
        """)

    def set_details(self, text: str):
        """Set details text."""
        self._details_label.setText(text)


class AlertIndicator(QFrame):
    """
    Shows recent resource alerts with severity coloring.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alerts: List[ResourceAlert] = []
        self._setup_ui()

    def _setup_ui(self):
        """Set up the alert indicator UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMaximumHeight(60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Header
        header = QHBoxLayout()
        title = QLabel("Alerts")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)

        header.addStretch()

        self._count_label = QLabel("0")
        self._count_label.setStyleSheet(
            "background-color: #28a745; color: white; "
            "padding: 2px 8px; border-radius: 10px; font-size: 11px;"
        )
        header.addWidget(self._count_label)

        layout.addLayout(header)

        # Latest alert message
        self._message_label = QLabel("No alerts")
        self._message_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)

    def update_alerts(self, alerts: List[ResourceAlert]):
        """Update with new alerts."""
        self._alerts = alerts

        if not alerts:
            self._count_label.setText("0")
            self._count_label.setStyleSheet(
                "background-color: #28a745; color: white; "
                "padding: 2px 8px; border-radius: 10px; font-size: 11px;"
            )
            self._message_label.setText("No alerts")
            self._message_label.setStyleSheet("color: #6c757d; font-size: 11px;")
            return

        # Count by severity
        critical = sum(1 for a in alerts if a.severity == 'critical')
        warning = sum(1 for a in alerts if a.severity == 'warning')

        self._count_label.setText(str(len(alerts)))

        if critical > 0:
            self._count_label.setStyleSheet(
                "background-color: #dc3545; color: white; "
                "padding: 2px 8px; border-radius: 10px; font-size: 11px;"
            )
        elif warning > 0:
            self._count_label.setStyleSheet(
                "background-color: #ffc107; color: black; "
                "padding: 2px 8px; border-radius: 10px; font-size: 11px;"
            )

        # Show latest alert
        latest = alerts[-1]
        self._message_label.setText(latest.message)

        if latest.severity == 'critical':
            self._message_label.setStyleSheet("color: #dc3545; font-size: 11px;")
        else:
            self._message_label.setStyleSheet("color: #ffc107; font-size: 11px;")


class ResourceMonitorWidget(QWidget):
    """
    Real-time resource monitoring widget.

    Displays CPU, memory, disk I/O, and GPU usage with visual gauges.
    Shows alerts when thresholds are exceeded.

    Signals
    -------
    alert_triggered : ResourceAlert
        Emitted when a new alert is triggered
    """

    alert_triggered = pyqtSignal(object)  # ResourceAlert

    def __init__(
        self,
        monitor: Optional[ResourceMonitor] = None,
        update_interval_ms: int = 1000,
        parent=None,
    ):
        """
        Initialize the resource monitor widget.

        Parameters
        ----------
        monitor : ResourceMonitor, optional
            Resource monitor to use. If None, uses global monitor.
        update_interval_ms : int
            UI update interval in milliseconds
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self._monitor = monitor or get_resource_monitor()
        self._update_interval = update_interval_ms
        self._last_alert_count = 0

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Set up the widget UI."""
        self.setMinimumWidth(200)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Title
        title = QLabel("System Resources")
        font = title.font()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # Gauges grid
        gauges_layout = QGridLayout()
        gauges_layout.setSpacing(8)

        # CPU gauge
        self._cpu_gauge = ResourceGauge(
            "CPU",
            unit="%",
            warning_threshold=90.0,
            critical_threshold=99.0,
        )
        gauges_layout.addWidget(self._cpu_gauge, 0, 0)

        # Memory gauge
        self._memory_gauge = ResourceGauge(
            "Memory",
            unit="%",
            warning_threshold=80.0,
            critical_threshold=95.0,
        )
        gauges_layout.addWidget(self._memory_gauge, 0, 1)

        # Disk Read gauge
        self._disk_read_gauge = ResourceGauge(
            "Disk Read",
            unit="MB/s",
            warning_threshold=500.0,
            critical_threshold=1000.0,
        )
        gauges_layout.addWidget(self._disk_read_gauge, 1, 0)

        # Disk Write gauge
        self._disk_write_gauge = ResourceGauge(
            "Disk Write",
            unit="MB/s",
            warning_threshold=500.0,
            critical_threshold=1000.0,
        )
        gauges_layout.addWidget(self._disk_write_gauge, 1, 1)

        # GPU gauge (hidden by default)
        self._gpu_gauge = ResourceGauge(
            "GPU Memory",
            unit="%",
            warning_threshold=85.0,
            critical_threshold=95.0,
        )
        self._gpu_gauge.setVisible(False)
        gauges_layout.addWidget(self._gpu_gauge, 2, 0, 1, 2)

        layout.addLayout(gauges_layout)

        # Alert indicator
        self._alert_indicator = AlertIndicator()
        layout.addWidget(self._alert_indicator)

        # Status line
        self._status_label = QLabel("Monitoring...")
        self._status_label.setStyleSheet("color: #6c757d; font-size: 10px;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._status_label)

        layout.addStretch()

    def _setup_timer(self):
        """Set up the update timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_display)
        self._timer.start(self._update_interval)

    def _update_display(self):
        """Update the display with current resource data."""
        try:
            snapshot = self._monitor.get_current_snapshot()
            self._update_gauges(snapshot)
            self._update_alerts()
            self._update_status(snapshot)
        except Exception as e:
            logger.warning(f"Failed to update resource display: {e}")

    def _update_gauges(self, snapshot: ResourceSnapshot):
        """Update gauge displays."""
        # CPU
        self._cpu_gauge.set_value(snapshot.cpu_percent, 100)
        self._cpu_gauge.set_details(f"{snapshot.cpu_count} cores")

        # Memory
        self._memory_gauge.set_value(snapshot.memory_percent, 100)
        used_gb = snapshot.memory_used_mb / 1024
        total_gb = snapshot.memory_total_mb / 1024
        self._memory_gauge.set_details(f"{used_gb:.1f} / {total_gb:.1f} GB")

        # Disk I/O
        self._disk_read_gauge.set_value(snapshot.disk_read_mb, 1000)
        self._disk_read_gauge.set_details("")

        self._disk_write_gauge.set_value(snapshot.disk_write_mb, 1000)
        self._disk_write_gauge.set_details("")

        # GPU (if available)
        if snapshot.gpu_memory_used_mb is not None and snapshot.gpu_memory_total_mb:
            self._gpu_gauge.setVisible(True)
            gpu_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
            self._gpu_gauge.set_value(gpu_percent, 100)
            used_gb = snapshot.gpu_memory_used_mb / 1024
            total_gb = snapshot.gpu_memory_total_mb / 1024
            self._gpu_gauge.set_details(f"{used_gb:.1f} / {total_gb:.1f} GB")
        else:
            self._gpu_gauge.setVisible(False)

    def _update_alerts(self):
        """Update alert indicator."""
        alerts = self._monitor.get_alerts(last_n=10)
        self._alert_indicator.update_alerts(alerts)

        # Emit signal for new alerts
        if len(alerts) > self._last_alert_count:
            new_alerts = alerts[self._last_alert_count:]
            for alert in new_alerts:
                self.alert_triggered.emit(alert)
            self._last_alert_count = len(alerts)

    def _update_status(self, snapshot: ResourceSnapshot):
        """Update status line."""
        time_str = snapshot.timestamp.strftime("%H:%M:%S")
        self._status_label.setText(f"Updated: {time_str}")

    def start_monitoring(self):
        """Start the resource monitor if not running."""
        self._monitor.start()
        self._timer.start(self._update_interval)

    def stop_monitoring(self):
        """Stop the resource monitor."""
        self._timer.stop()
        self._monitor.stop()

    def get_summary(self):
        """Get resource usage summary."""
        return self._monitor.get_summary()

    def clear_alerts(self):
        """Clear all alerts."""
        self._monitor.clear_alerts()
        self._last_alert_count = 0
        self._alert_indicator.update_alerts([])

    def cleanup(self):
        """Clean up resources (alias for stop_monitoring)."""
        self.stop_monitoring()


class CompactResourceMonitorWidget(QWidget):
    """
    Compact version of resource monitor for status bar or toolbar.

    Shows CPU and memory as small inline gauges.
    """

    def __init__(
        self,
        monitor: Optional[ResourceMonitor] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._monitor = monitor or get_resource_monitor()

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Set up compact UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(12)

        # CPU
        cpu_layout = QHBoxLayout()
        cpu_layout.setSpacing(4)
        cpu_label = QLabel("CPU:")
        cpu_label.setStyleSheet("font-size: 11px;")
        cpu_layout.addWidget(cpu_label)

        self._cpu_bar = QProgressBar()
        self._cpu_bar.setRange(0, 100)
        self._cpu_bar.setMaximumWidth(60)
        self._cpu_bar.setMaximumHeight(14)
        self._cpu_bar.setTextVisible(True)
        self._cpu_bar.setFormat("%p%")
        cpu_layout.addWidget(self._cpu_bar)

        layout.addLayout(cpu_layout)

        # Memory
        mem_layout = QHBoxLayout()
        mem_layout.setSpacing(4)
        mem_label = QLabel("Mem:")
        mem_label.setStyleSheet("font-size: 11px;")
        mem_layout.addWidget(mem_label)

        self._mem_bar = QProgressBar()
        self._mem_bar.setRange(0, 100)
        self._mem_bar.setMaximumWidth(60)
        self._mem_bar.setMaximumHeight(14)
        self._mem_bar.setTextVisible(True)
        self._mem_bar.setFormat("%p%")
        mem_layout.addWidget(self._mem_bar)

        layout.addLayout(mem_layout)

        # Alert indicator (dot)
        self._alert_dot = QLabel()
        self._alert_dot.setFixedSize(12, 12)
        self._alert_dot.setStyleSheet(
            "background-color: #28a745; border-radius: 6px;"
        )
        layout.addWidget(self._alert_dot)

    def _setup_timer(self):
        """Set up update timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_display)
        self._timer.start(1000)

    def _update_display(self):
        """Update the compact display."""
        try:
            snapshot = self._monitor.get_current_snapshot()

            self._cpu_bar.setValue(int(snapshot.cpu_percent))
            self._mem_bar.setValue(int(snapshot.memory_percent))

            # Update alert dot
            alerts = self._monitor.get_alerts(last_n=5)
            if any(a.severity == 'critical' for a in alerts):
                self._alert_dot.setStyleSheet(
                    "background-color: #dc3545; border-radius: 6px;"
                )
            elif any(a.severity == 'warning' for a in alerts):
                self._alert_dot.setStyleSheet(
                    "background-color: #ffc107; border-radius: 6px;"
                )
            else:
                self._alert_dot.setStyleSheet(
                    "background-color: #28a745; border-radius: 6px;"
                )

        except Exception as e:
            logger.warning(f"Failed to update compact display: {e}")

    def cleanup(self):
        """Clean up resources."""
        self._timer.stop()
