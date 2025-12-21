"""
Tests for Job Analytics Widget

Tests the JobAnalyticsWidget and its chart components.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

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


class TestStatCard:
    """Tests for StatCard component."""

    def test_card_creation(self):
        """Test creating a stat card."""
        from views.widgets.job_analytics_widget import StatCard

        card = StatCard(
            title="Test Metric",
            value="42",
            subtitle="items",
            color="#007bff",
        )

        assert card is not None
        assert card._title == "Test Metric"
        assert card._value == "42"

    def test_card_set_value(self):
        """Test setting card value."""
        from views.widgets.job_analytics_widget import StatCard

        card = StatCard("Count", "0")
        card.set_value("100", "updated")

        assert card._value_label.text() == "100"
        assert card._subtitle_label.text() == "updated"

    def test_card_set_color(self):
        """Test setting card color."""
        from views.widgets.job_analytics_widget import StatCard

        card = StatCard("Error Rate", "5%", color="#28a745")
        card.set_color("#dc3545")

        assert "#dc3545" in card._value_label.styleSheet()


class TestSimpleBarChart:
    """Tests for SimpleBarChart component."""

    def test_chart_creation(self):
        """Test creating a bar chart."""
        from views.widgets.job_analytics_widget import SimpleBarChart

        chart = SimpleBarChart()
        assert chart is not None

    def test_chart_set_data(self):
        """Test setting chart data."""
        from views.widgets.job_analytics_widget import SimpleBarChart

        chart = SimpleBarChart()
        data = [
            {'label': 'Mon', 'value': 10},
            {'label': 'Tue', 'value': 15},
            {'label': 'Wed', 'value': 8},
        ]
        chart.set_data(data)

        assert chart._data == data

    def test_chart_set_bar_color(self):
        """Test setting bar color."""
        from views.widgets.job_analytics_widget import SimpleBarChart
        from PyQt6.QtGui import QColor

        chart = SimpleBarChart()
        chart.set_bar_color("#28a745")

        assert chart._bar_color == QColor("#28a745")


class TestSimplePieChart:
    """Tests for SimplePieChart component."""

    def test_chart_creation(self):
        """Test creating a pie chart."""
        from views.widgets.job_analytics_widget import SimplePieChart

        chart = SimplePieChart()
        assert chart is not None

    def test_chart_set_data(self):
        """Test setting chart data."""
        from views.widgets.job_analytics_widget import SimplePieChart

        chart = SimplePieChart()
        data = [
            {'label': 'Completed', 'value': 80},
            {'label': 'Failed', 'value': 15},
            {'label': 'Cancelled', 'value': 5},
        ]
        chart.set_data(data)

        assert chart._data == data


class TestJobAnalyticsWidget:
    """Tests for JobAnalyticsWidget."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def history_storage(self, temp_db):
        """Create a history storage instance."""
        from utils.ray_orchestration.job_history import JobHistoryStorage

        storage = JobHistoryStorage(db_path=temp_db, auto_cleanup=False)
        yield storage
        storage.close()

    def test_widget_creation(self, history_storage):
        """Test creating an analytics widget."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget

        widget = JobAnalyticsWidget(
            history_storage=history_storage,
            update_interval_ms=60000,
        )

        assert widget is not None
        assert widget._history is history_storage
        assert widget._total_card is not None
        assert widget._daily_chart is not None

        widget.cleanup()

    def test_widget_displays_statistics(self, history_storage):
        """Test widget displays job statistics."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget
        from models.job import Job, JobType

        # Add some jobs to history
        for i in range(5):
            job = Job(name=f"Job {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_completed()
            history_storage.save_job(job)

        widget = JobAnalyticsWidget(
            history_storage=history_storage,
            update_interval_ms=60000,
        )

        # Total should show 5
        assert widget._total_card._value_label.text() == "5"
        assert widget._completed_card._value_label.text() == "5"

        widget.cleanup()

    def test_widget_shows_error_rate(self, history_storage):
        """Test widget shows error rate correctly."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget
        from models.job import Job, JobType

        # Add completed jobs
        for i in range(8):
            job = Job(name=f"Completed {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_completed()
            history_storage.save_job(job)

        # Add failed jobs
        for i in range(2):
            job = Job(name=f"Failed {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            history_storage.save_job(job)

        widget = JobAnalyticsWidget(
            history_storage=history_storage,
            update_interval_ms=60000,
        )

        # Error rate should be 20%
        assert "20" in widget._error_rate_card._value_label.text()

        widget.cleanup()

    def test_widget_get_statistics(self, history_storage):
        """Test getting statistics from widget."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget
        from models.job import Job, JobType

        job = Job(name="Test", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_completed()
        history_storage.save_job(job)

        widget = JobAnalyticsWidget(
            history_storage=history_storage,
            update_interval_ms=60000,
        )

        stats = widget.get_statistics()

        assert stats['total_jobs'] == 1
        assert 'by_state' in stats

        widget.cleanup()

    def test_widget_period_filter(self, history_storage):
        """Test widget period filter changes."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget

        widget = JobAnalyticsWidget(
            history_storage=history_storage,
            update_interval_ms=60000,
        )

        # Change period selection
        widget._period_combo.setCurrentIndex(1)  # Last 30 Days

        # Should trigger refresh
        assert widget._period_combo.currentText() == "Last 30 Days"

        widget.cleanup()

    def test_widget_refresh_signal(self, history_storage):
        """Test widget emits refresh signal."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget

        widget = JobAnalyticsWidget(
            history_storage=history_storage,
            update_interval_ms=60000,
        )

        signals_received = []
        widget.refresh_requested.connect(lambda: signals_received.append(True))

        # Trigger refresh
        widget._refresh_data()

        assert len(signals_received) == 1

        widget.cleanup()


class TestDurationFormatting:
    """Tests for duration formatting."""

    def test_format_seconds(self):
        """Test formatting seconds."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget

        widget = JobAnalyticsWidget.__new__(JobAnalyticsWidget)

        assert widget._format_duration(30.5) == "30.5s"
        assert widget._format_duration(0.5) == "0.5s"

    def test_format_minutes(self):
        """Test formatting minutes."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget

        widget = JobAnalyticsWidget.__new__(JobAnalyticsWidget)

        assert widget._format_duration(120) == "2.0m"
        assert widget._format_duration(90) == "1.5m"

    def test_format_hours(self):
        """Test formatting hours."""
        from views.widgets.job_analytics_widget import JobAnalyticsWidget

        widget = JobAnalyticsWidget.__new__(JobAnalyticsWidget)

        assert widget._format_duration(3600) == "1.0h"
        assert widget._format_duration(7200) == "2.0h"
