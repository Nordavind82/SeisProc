"""
Tests for Alert Manager with Rules Engine

Tests the AlertManager and various alert rules.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from models.job import Job, JobType, JobState


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        from utils.ray_orchestration.alert_manager import (
            Alert, AlertSeverity, AlertCategory
        )

        alert = Alert(
            severity=AlertSeverity.WARNING,
            category=AlertCategory.JOB,
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.category == AlertCategory.JOB
        assert alert.title == "Test Alert"
        assert alert.acknowledged is False

    def test_alert_acknowledge(self):
        """Test acknowledging an alert."""
        from utils.ray_orchestration.alert_manager import (
            Alert, AlertSeverity, AlertCategory
        )

        alert = Alert(
            severity=AlertSeverity.ERROR,
            category=AlertCategory.SYSTEM,
            title="Error",
            message="Test error",
        )

        assert alert.acknowledged is False
        alert.acknowledge()
        assert alert.acknowledged is True

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        from utils.ray_orchestration.alert_manager import (
            Alert, AlertSeverity, AlertCategory
        )

        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RESOURCE,
            title="Critical",
            message="Critical issue",
            source="test_rule",
        )

        data = alert.to_dict()

        assert data['severity'] == 'CRITICAL'
        assert data['category'] == 'RESOURCE'
        assert data['title'] == 'Critical'
        assert data['source'] == 'test_rule'


class TestJobFailureRule:
    """Tests for JobFailureRule."""

    def test_rule_triggers_on_failure(self):
        """Test rule triggers on job failure."""
        from utils.ray_orchestration.alert_manager import JobFailureRule

        rule = JobFailureRule(min_failures_threshold=1)

        # Create failed job
        job = Job(name="Test Job", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Test error")

        alert = rule.evaluate({'job': job})

        assert alert is not None
        assert "Test Job" in alert.title
        assert "Test error" in alert.message

    def test_rule_ignores_successful_jobs(self):
        """Test rule ignores successful jobs."""
        from utils.ray_orchestration.alert_manager import JobFailureRule

        rule = JobFailureRule()

        job = Job(name="Success", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_completed()

        alert = rule.evaluate({'job': job})
        assert alert is None

    def test_rule_respects_cooldown(self):
        """Test rule respects cooldown period."""
        from utils.ray_orchestration.alert_manager import JobFailureRule

        rule = JobFailureRule(cooldown_seconds=60)

        # First failure triggers
        job1 = Job(name="Fail 1", job_type=JobType.BATCH_PROCESS)
        job1.mark_started()
        job1.mark_failed("Error 1")
        alert1 = rule.evaluate({'job': job1})
        assert alert1 is not None

        # Second failure within cooldown doesn't trigger
        job2 = Job(name="Fail 2", job_type=JobType.BATCH_PROCESS)
        job2.mark_started()
        job2.mark_failed("Error 2")
        alert2 = rule.evaluate({'job': job2})
        assert alert2 is None


class TestConsecutiveFailuresRule:
    """Tests for ConsecutiveFailuresRule."""

    def test_triggers_after_threshold(self):
        """Test rule triggers after consecutive failures."""
        from utils.ray_orchestration.alert_manager import ConsecutiveFailuresRule

        rule = ConsecutiveFailuresRule(threshold=3, cooldown_seconds=0)

        # First two failures - no alert
        for i in range(2):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            alert = rule.evaluate({'job': job})
            assert alert is None

        # Third failure - triggers alert
        job = Job(name="Fail 3", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Error")
        alert = rule.evaluate({'job': job})
        assert alert is not None
        assert "3" in alert.title

    def test_resets_on_success(self):
        """Test count resets on successful job."""
        from utils.ray_orchestration.alert_manager import ConsecutiveFailuresRule

        rule = ConsecutiveFailuresRule(threshold=3, cooldown_seconds=0)

        # Two failures
        for i in range(2):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            rule.evaluate({'job': job})

        # One success - resets count
        success = Job(name="Success", job_type=JobType.BATCH_PROCESS)
        success.mark_started()
        success.mark_completed()
        rule.evaluate({'job': success})

        assert rule._consecutive_count == 0


class TestErrorRateRule:
    """Tests for ErrorRateRule."""

    def test_triggers_when_rate_exceeded(self):
        """Test rule triggers when error rate exceeds threshold."""
        from utils.ray_orchestration.alert_manager import ErrorRateRule

        rule = ErrorRateRule(
            error_rate_threshold=0.5,  # 50%
            min_jobs=4,
            cooldown_seconds=0,
        )

        # Add 2 successes
        for i in range(2):
            job = Job(name=f"Success {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_completed()
            rule.evaluate({'job': job})

        # Add 2 failures (50% error rate)
        for i in range(2):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            alert = rule.evaluate({'job': job})

        assert alert is not None
        assert "50" in alert.title or "0.5" in str(alert.data.get('error_rate'))

    def test_waits_for_minimum_jobs(self):
        """Test rule waits for minimum jobs."""
        from utils.ray_orchestration.alert_manager import ErrorRateRule

        rule = ErrorRateRule(
            error_rate_threshold=0.3,
            min_jobs=10,
            cooldown_seconds=0,
        )

        # Add 5 failures (but need 10 jobs minimum)
        for i in range(5):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            alert = rule.evaluate({'job': job})
            assert alert is None  # Not enough jobs yet


class TestLongRunningJobRule:
    """Tests for LongRunningJobRule."""

    def test_triggers_for_long_job(self):
        """Test rule triggers for long running job."""
        from utils.ray_orchestration.alert_manager import LongRunningJobRule

        rule = LongRunningJobRule(duration_minutes=1, cooldown_seconds=0)

        job = Job(name="Long Job", job_type=JobType.BATCH_PROCESS)
        job.state = JobState.RUNNING
        job.started_at = datetime.now() - timedelta(minutes=5)

        alert = rule.evaluate({'job': job})

        assert alert is not None
        assert "Long Running" in alert.title

    def test_ignores_short_jobs(self):
        """Test rule ignores jobs under threshold."""
        from utils.ray_orchestration.alert_manager import LongRunningJobRule

        rule = LongRunningJobRule(duration_minutes=30, cooldown_seconds=0)

        job = Job(name="Quick Job", job_type=JobType.BATCH_PROCESS)
        job.state = JobState.RUNNING
        job.started_at = datetime.now() - timedelta(minutes=5)

        alert = rule.evaluate({'job': job})
        assert alert is None

    def test_only_alerts_once_per_job(self):
        """Test rule only alerts once per job."""
        from utils.ray_orchestration.alert_manager import LongRunningJobRule

        rule = LongRunningJobRule(duration_minutes=1, cooldown_seconds=0)

        job = Job(name="Long Job", job_type=JobType.BATCH_PROCESS)
        job.state = JobState.RUNNING
        job.started_at = datetime.now() - timedelta(minutes=5)

        # First evaluation - alert
        alert1 = rule.evaluate({'job': job})
        assert alert1 is not None

        # Second evaluation - no alert (already alerted)
        alert2 = rule.evaluate({'job': job})
        assert alert2 is None


class TestAlertManager:
    """Tests for AlertManager."""

    def test_manager_creation(self):
        """Test creating an alert manager."""
        from utils.ray_orchestration.alert_manager import AlertManager

        manager = AlertManager()
        assert manager is not None
        assert len(manager.get_rules()) == 0

    def test_add_and_remove_rule(self):
        """Test adding and removing rules."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        rule = JobFailureRule(name="Test Rule")

        manager.add_rule(rule)
        assert len(manager.get_rules()) == 1

        manager.remove_rule("Test Rule")
        assert len(manager.get_rules()) == 0

    def test_process_job_event(self):
        """Test processing job events."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        job = Job(name="Failed Job", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Error")

        alerts = manager.process_job_event(job)

        assert len(alerts) == 1
        assert "Failed Job" in alerts[0].title

    def test_get_alerts(self):
        """Test getting alerts."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        # Generate alerts
        for i in range(3):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            manager.process_job_event(job)

        alerts = manager.get_alerts()
        assert len(alerts) == 3

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        job = Job(name="Fail", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Error")
        alerts = manager.process_job_event(job)

        alert_id = alerts[0].id
        manager.acknowledge_alert(alert_id)

        acknowledged = manager.get_alerts(acknowledged=True)
        assert len(acknowledged) == 1

    def test_acknowledge_all(self):
        """Test acknowledging all alerts."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        for i in range(3):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            manager.process_job_event(job)

        count = manager.acknowledge_all()
        assert count == 3
        assert manager.get_unacknowledged_count() == 0

    def test_clear_acknowledged(self):
        """Test clearing acknowledged alerts."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        for i in range(3):
            job = Job(name=f"Fail {i}", job_type=JobType.BATCH_PROCESS)
            job.mark_started()
            job.mark_failed("Error")
            manager.process_job_event(job)

        manager.acknowledge_all()
        cleared = manager.clear_acknowledged()

        assert cleared == 3
        assert len(manager.get_alerts()) == 0

    def test_enable_disable_rule(self):
        """Test enabling and disabling rules."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(name="Failure"))

        # Disable rule
        manager.disable_rule("Failure")

        job = Job(name="Fail", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Error")
        alerts = manager.process_job_event(job)

        assert len(alerts) == 0  # Rule disabled

        # Re-enable
        manager.enable_rule("Failure")
        alerts = manager.process_job_event(job)
        assert len(alerts) == 1  # Rule enabled

    def test_callback_notification(self):
        """Test callback is called for new alerts."""
        from utils.ray_orchestration.alert_manager import (
            AlertManager, JobFailureRule
        )

        manager = AlertManager()
        manager.add_rule(JobFailureRule(cooldown_seconds=0))

        received_alerts = []
        manager.register_callback(lambda a: received_alerts.append(a))

        job = Job(name="Fail", job_type=JobType.BATCH_PROCESS)
        job.mark_started()
        job.mark_failed("Error")
        manager.process_job_event(job)

        assert len(received_alerts) == 1


class TestDefaultAlertManager:
    """Tests for default alert manager creation."""

    def test_create_default_manager(self):
        """Test creating default manager with rules."""
        import utils.ray_orchestration.alert_manager as am
        from utils.ray_orchestration.alert_manager import create_default_alert_manager

        # Reset singleton to test fresh creation
        am._alert_manager = None

        manager = create_default_alert_manager()

        rules = manager.get_rules()
        assert len(rules) == 4  # Default rules

        # Cleanup
        am._alert_manager = None

    def test_singleton_manager(self):
        """Test singleton manager access."""
        import utils.ray_orchestration.alert_manager as am

        # Reset singleton
        am._alert_manager = None

        manager1 = am.get_alert_manager()
        manager2 = am.get_alert_manager()

        assert manager1 is manager2

        # Cleanup
        am._alert_manager = None

    def test_create_default_is_singleton(self):
        """Test create_default_alert_manager returns singleton."""
        import utils.ray_orchestration.alert_manager as am
        from utils.ray_orchestration.alert_manager import (
            create_default_alert_manager,
            get_alert_manager,
        )

        # Reset singleton
        am._alert_manager = None

        manager1 = create_default_alert_manager()
        manager2 = get_alert_manager()

        assert manager1 is manager2

        # Cleanup
        am._alert_manager = None
