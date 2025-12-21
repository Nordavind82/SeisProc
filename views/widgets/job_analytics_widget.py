"""
Job Analytics Widget

Provides visual analytics for job history including charts,
statistics, and performance metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QComboBox, QGroupBox,
    QScrollArea, QSizePolicy, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont

from utils.ray_orchestration.job_history import JobHistoryStorage, get_job_history_storage

logger = logging.getLogger(__name__)


class StatCard(QFrame):
    """
    A card displaying a single statistic.

    Displays a title, main value, and optional subtitle.
    """

    def __init__(
        self,
        title: str,
        value: str = "0",
        subtitle: str = "",
        color: str = "#007bff",
        parent=None,
    ):
        super().__init__(parent)

        self._title = title
        self._value = value
        self._subtitle = subtitle
        self._color = color

        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            StatCard {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        # Title
        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        layout.addWidget(self._title_label)

        # Value
        self._value_label = QLabel(self._value)
        self._value_label.setStyleSheet(
            f"color: {self._color}; font-size: 24px; font-weight: bold;"
        )
        layout.addWidget(self._value_label)

        # Subtitle
        self._subtitle_label = QLabel(self._subtitle)
        self._subtitle_label.setStyleSheet("color: #6c757d; font-size: 10px;")
        layout.addWidget(self._subtitle_label)

    def set_value(self, value: str, subtitle: str = ""):
        """Update the displayed value."""
        self._value_label.setText(value)
        if subtitle:
            self._subtitle_label.setText(subtitle)

    def set_color(self, color: str):
        """Update the value color."""
        self._color = color
        self._value_label.setStyleSheet(
            f"color: {self._color}; font-size: 24px; font-weight: bold;"
        )


class SimpleBarChart(QWidget):
    """
    A simple bar chart widget.

    Displays vertical bars with labels.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data: List[Dict[str, Any]] = []
        self._bar_color = QColor("#007bff")
        self._bar_width = 30
        self._spacing = 10

        self.setMinimumHeight(150)

    def set_data(self, data: List[Dict[str, Any]]):
        """
        Set chart data.

        Parameters
        ----------
        data : list
            List of dicts with 'label' and 'value' keys
        """
        self._data = data
        self.update()

    def set_bar_color(self, color: str):
        """Set the bar color."""
        self._bar_color = QColor(color)
        self.update()

    def paintEvent(self, event):
        """Paint the bar chart."""
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        max_value = max((d.get('value', 0) for d in self._data), default=1)
        if max_value == 0:
            max_value = 1

        chart_height = self.height() - 30  # Leave room for labels
        chart_width = self.width()

        # Calculate bar dimensions
        total_bars = len(self._data)
        available_width = chart_width - 20
        bar_width = min(self._bar_width, (available_width - (total_bars - 1) * self._spacing) / total_bars)
        total_bar_width = total_bars * bar_width + (total_bars - 1) * self._spacing
        start_x = (chart_width - total_bar_width) / 2

        # Draw bars
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._bar_color)

        for i, item in enumerate(self._data):
            value = item.get('value', 0)
            label = item.get('label', '')

            bar_height = (value / max_value) * (chart_height - 20) if max_value > 0 else 0
            x = start_x + i * (bar_width + self._spacing)
            y = chart_height - bar_height

            # Draw bar
            painter.drawRoundedRect(
                int(x), int(y),
                int(bar_width), int(bar_height),
                4, 4
            )

            # Draw label
            painter.setPen(QColor("#6c757d"))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(
                int(x), chart_height + 5,
                int(bar_width), 20,
                Qt.AlignmentFlag.AlignCenter,
                label
            )

            # Draw value on top of bar
            if bar_height > 20:
                painter.setPen(QColor("#ffffff"))
                painter.drawText(
                    int(x), int(y) + 5,
                    int(bar_width), 20,
                    Qt.AlignmentFlag.AlignCenter,
                    str(value)
                )

            painter.setPen(Qt.PenStyle.NoPen)


class SimplePieChart(QWidget):
    """
    A simple pie chart widget.

    Displays a pie chart with a legend.
    """

    # Colors for slices
    COLORS = [
        "#28a745",  # Green - completed
        "#dc3545",  # Red - failed
        "#ffc107",  # Yellow - cancelled
        "#17a2b8",  # Cyan - other
        "#6c757d",  # Gray
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._data: List[Dict[str, Any]] = []
        self.setMinimumSize(200, 150)

    def set_data(self, data: List[Dict[str, Any]]):
        """
        Set chart data.

        Parameters
        ----------
        data : list
            List of dicts with 'label' and 'value' keys
        """
        self._data = data
        self.update()

    def paintEvent(self, event):
        """Paint the pie chart."""
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate total
        total = sum(d.get('value', 0) for d in self._data)
        if total == 0:
            return

        # Pie dimensions
        size = min(self.width() - 100, self.height() - 20)
        pie_rect = (10, 10, size, size)

        # Draw slices
        start_angle = 90 * 16  # Start from top
        for i, item in enumerate(self._data):
            value = item.get('value', 0)
            if value == 0:
                continue

            span_angle = int((value / total) * 360 * 16)
            color = QColor(self.COLORS[i % len(self.COLORS)])

            painter.setPen(QPen(color.darker(110), 1))
            painter.setBrush(color)
            painter.drawPie(*pie_rect, start_angle, span_angle)

            start_angle += span_angle

        # Draw legend
        legend_x = size + 30
        legend_y = 15
        legend_spacing = 20

        painter.setFont(QFont("", 9))

        for i, item in enumerate(self._data):
            if item.get('value', 0) == 0:
                continue

            color = QColor(self.COLORS[i % len(self.COLORS)])
            label = item.get('label', '')
            value = item.get('value', 0)

            # Color box
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(legend_x, legend_y + i * legend_spacing, 12, 12)

            # Label
            painter.setPen(QColor("#333333"))
            painter.drawText(
                legend_x + 18, legend_y + i * legend_spacing,
                100, 15,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{label}: {value}"
            )


class JobAnalyticsWidget(QWidget):
    """
    Widget for displaying job analytics and statistics.

    Shows:
    - Summary statistics cards
    - Daily job counts chart
    - Job state distribution pie chart
    - Performance metrics

    Signals
    -------
    refresh_requested : None
        Emitted when refresh is requested
    """

    refresh_requested = pyqtSignal()

    def __init__(
        self,
        history_storage: Optional[JobHistoryStorage] = None,
        update_interval_ms: int = 30000,
        parent=None,
    ):
        """
        Initialize analytics widget.

        Parameters
        ----------
        history_storage : JobHistoryStorage, optional
            History storage to use, defaults to singleton
        update_interval_ms : int
            Auto-refresh interval in milliseconds
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._history = history_storage or get_job_history_storage()
        self._update_interval = update_interval_ms

        self._setup_ui()
        self._setup_timer()
        self._refresh_data()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Header
        header = self._create_header()
        layout.addLayout(header)

        # Statistics cards
        cards_layout = self._create_stat_cards()
        layout.addLayout(cards_layout)

        # Charts row
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(16)

        # Daily counts chart
        daily_group = QGroupBox("Daily Job Counts (7 Days)")
        daily_layout = QVBoxLayout(daily_group)
        self._daily_chart = SimpleBarChart()
        daily_layout.addWidget(self._daily_chart)
        charts_layout.addWidget(daily_group)

        # State distribution chart
        state_group = QGroupBox("Job State Distribution")
        state_layout = QVBoxLayout(state_group)
        self._state_chart = SimplePieChart()
        state_layout.addWidget(self._state_chart)
        charts_layout.addWidget(state_group)

        layout.addLayout(charts_layout)

        # Performance metrics
        metrics_group = self._create_metrics_group()
        layout.addWidget(metrics_group)

        layout.addStretch()

    def _create_header(self) -> QHBoxLayout:
        """Create header with title and controls."""
        header = QHBoxLayout()

        title = QLabel("Job Analytics")
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)

        header.addStretch()

        # Period selector
        self._period_combo = QComboBox()
        self._period_combo.addItems(["Last 7 Days", "Last 30 Days", "All Time"])
        self._period_combo.currentIndexChanged.connect(self._refresh_data)
        header.addWidget(self._period_combo)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_data)
        header.addWidget(refresh_btn)

        return header

    def _create_stat_cards(self) -> QHBoxLayout:
        """Create statistics cards."""
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)

        # Total jobs
        self._total_card = StatCard(
            "Total Jobs",
            "0",
            "",
            "#007bff"
        )
        cards_layout.addWidget(self._total_card)

        # Completed jobs
        self._completed_card = StatCard(
            "Completed",
            "0",
            "",
            "#28a745"
        )
        cards_layout.addWidget(self._completed_card)

        # Failed jobs
        self._failed_card = StatCard(
            "Failed",
            "0",
            "",
            "#dc3545"
        )
        cards_layout.addWidget(self._failed_card)

        # Error rate
        self._error_rate_card = StatCard(
            "Error Rate",
            "0%",
            "",
            "#ffc107"
        )
        cards_layout.addWidget(self._error_rate_card)

        return cards_layout

    def _create_metrics_group(self) -> QGroupBox:
        """Create performance metrics group."""
        group = QGroupBox("Performance Metrics")
        layout = QGridLayout(group)

        # Average duration
        layout.addWidget(QLabel("Average Duration:"), 0, 0)
        self._avg_duration_label = QLabel("--")
        self._avg_duration_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._avg_duration_label, 0, 1)

        # Max duration
        layout.addWidget(QLabel("Max Duration:"), 0, 2)
        self._max_duration_label = QLabel("--")
        self._max_duration_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._max_duration_label, 0, 3)

        # Total processing time
        layout.addWidget(QLabel("Total Processing Time:"), 1, 0)
        self._total_time_label = QLabel("--")
        self._total_time_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._total_time_label, 1, 1)

        # Jobs per day
        layout.addWidget(QLabel("Jobs per Day (avg):"), 1, 2)
        self._jobs_per_day_label = QLabel("--")
        self._jobs_per_day_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._jobs_per_day_label, 1, 3)

        return group

    def _setup_timer(self):
        """Set up auto-refresh timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_data)
        self._timer.start(self._update_interval)

    def _get_date_filter(self):
        """Get start date based on period selection."""
        period = self._period_combo.currentText()
        if period == "Last 7 Days":
            return datetime.now() - timedelta(days=7)
        elif period == "Last 30 Days":
            return datetime.now() - timedelta(days=30)
        return None  # All time

    @pyqtSlot()
    def _refresh_data(self):
        """Refresh all analytics data."""
        start_date = self._get_date_filter()

        try:
            stats = self._history.get_statistics(start_date=start_date)
            self._update_stat_cards(stats)
            self._update_state_chart(stats)
            self._update_performance_metrics(stats)

            # Daily counts
            daily = self._history.get_daily_counts(days=7)
            self._update_daily_chart(daily)

        except Exception as e:
            logger.error(f"Error refreshing analytics: {e}")

        self.refresh_requested.emit()

    def _update_stat_cards(self, stats: Dict[str, Any]):
        """Update statistics cards."""
        total = stats.get('total_jobs', 0)
        by_state = stats.get('by_state', {})

        completed = by_state.get('COMPLETED', 0)
        failed = by_state.get('FAILED', 0)
        error_rate = stats.get('error_rate_percent', 0)

        self._total_card.set_value(str(total))
        self._completed_card.set_value(str(completed))
        self._failed_card.set_value(str(failed))
        self._error_rate_card.set_value(f"{error_rate:.1f}%")

        # Color error rate based on value
        if error_rate >= 20:
            self._error_rate_card.set_color("#dc3545")  # Red
        elif error_rate >= 10:
            self._error_rate_card.set_color("#ffc107")  # Yellow
        else:
            self._error_rate_card.set_color("#28a745")  # Green

    def _update_state_chart(self, stats: Dict[str, Any]):
        """Update state distribution pie chart."""
        by_state = stats.get('by_state', {})

        data = [
            {'label': 'Completed', 'value': by_state.get('COMPLETED', 0)},
            {'label': 'Failed', 'value': by_state.get('FAILED', 0)},
            {'label': 'Cancelled', 'value': by_state.get('CANCELLED', 0)},
            {'label': 'Timeout', 'value': by_state.get('TIMEOUT', 0)},
        ]

        self._state_chart.set_data(data)

    def _update_daily_chart(self, daily: List[Dict[str, Any]]):
        """Update daily job counts chart."""
        # Prepare data for bar chart
        chart_data = []
        for entry in daily[-7:]:  # Last 7 days
            date = entry.get('date', '')
            # Use short date format
            short_date = date[-5:] if len(date) >= 5 else date
            chart_data.append({
                'label': short_date,
                'value': entry.get('total', 0)
            })

        self._daily_chart.set_data(chart_data)

    def _update_performance_metrics(self, stats: Dict[str, Any]):
        """Update performance metrics labels."""
        duration = stats.get('duration', {})

        avg_sec = duration.get('avg_seconds')
        max_sec = duration.get('max_seconds')
        total_sec = duration.get('total_seconds')

        if avg_sec is not None:
            self._avg_duration_label.setText(self._format_duration(avg_sec))
        else:
            self._avg_duration_label.setText("--")

        if max_sec is not None:
            self._max_duration_label.setText(self._format_duration(max_sec))
        else:
            self._max_duration_label.setText("--")

        if total_sec is not None:
            self._total_time_label.setText(self._format_duration(total_sec))
        else:
            self._total_time_label.setText("--")

        # Calculate jobs per day
        total_jobs = stats.get('total_jobs', 0)
        period = self._period_combo.currentText()
        if period == "Last 7 Days":
            days = 7
        elif period == "Last 30 Days":
            days = 30
        else:
            days = 30  # Default estimate

        jobs_per_day = total_jobs / days if days > 0 else 0
        self._jobs_per_day_label.setText(f"{jobs_per_day:.1f}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self._history.get_statistics(start_date=self._get_date_filter())

    def cleanup(self):
        """Clean up resources."""
        self._timer.stop()
