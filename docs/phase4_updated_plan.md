# Phase 4: Polish & Optimization - Updated Plan

**Version:** 2.0
**Date:** 2024-12-19
**Status:** Ready for Implementation
**Prerequisites:** Phase 1, 2, 3 Complete (130 tests passing)

---

## Overview

This updated Phase 4 plan incorporates lessons learned from Phases 1-3 and addresses:
1. **Gaps from earlier phases** that should be filled before optimization
2. **Original Phase 4 tasks** updated based on current architecture
3. **New tasks** identified during implementation

---

## Pre-Phase 4: Fill Critical Gaps

Before starting optimization, complete these gaps from earlier phases:

### Gap 4.0.1: Resource Monitor UI Widget
**Priority:** High
**Effort:** 2 days

The backend `ResourceMonitor` exists but has no UI component.

**Files to Create:**
```
views/job_dashboard/resource_monitor_widget.py
```

**Implementation:**
```python
class ResourceMonitorWidget(QWidget):
    """Real-time resource monitoring display."""

    def __init__(self):
        # CPU gauge
        # Memory gauge
        # GPU gauge (if available)
        # Worker status table
        # Alert indicator
```

**Tests:**
```python
# tests/test_resource_monitor_widget.py
def test_widget_displays_cpu_usage(qtbot):
    """Resource widget shows CPU usage."""

def test_widget_shows_memory_alerts(qtbot):
    """Resource widget indicates memory alerts."""

def test_widget_updates_in_realtime(qtbot):
    """Resource widget updates every sample interval."""
```

---

### Gap 4.0.2: Metal Worker Actor
**Priority:** Medium
**Effort:** 3 days

GPU processing through Ray actors for Metal shaders.

**Files to Create:**
```
utils/ray_orchestration/workers/metal_worker.py
```

**Implementation:**
```python
class MetalWorkerActor(BaseWorkerActor):
    """Ray actor for Metal GPU processing."""

    def __init__(self, job_id, worker_id):
        super().__init__(job_id, worker_id)
        self._device = None
        self._command_queue = None

    def initialize_gpu(self):
        """Initialize Metal device for this worker."""

    def process_gather_gpu(self, gather_data, processor_config):
        """Process gather using GPU kernels."""
```

**Tests:**
```python
# tests/test_metal_worker.py
def test_metal_worker_initializes_gpu():
    """Metal worker initializes GPU device."""

def test_metal_worker_processes_gather():
    """Metal worker processes gather on GPU."""

def test_metal_worker_handles_gpu_memory():
    """Metal worker manages GPU memory correctly."""
```

---

### Gap 4.0.3: Main Window Integration
**Priority:** High
**Effort:** 1 day

Connect job dashboard to main application window.

**Files to Modify:**
```
main_window.py  # Add job dashboard dock widget
```

**Implementation:**
- Add "Job Dashboard" menu action
- Create dock widget for dashboard
- Connect cancel all to Escape key
- Add status bar job indicator

---

## Week 13: Job History & Persistence

### Task 4.1: Job History Storage
**Duration:** 3 days

Create persistent storage for job history with SQLite.

**Files to Create:**
```
utils/ray_orchestration/history/
├── __init__.py
├── storage.py          # JobHistoryStorage class
├── models.py           # SQLAlchemy models
└── queries.py          # Query interface
```

**Implementation:**
```python
class JobHistoryStorage:
    """SQLite-based job history storage."""

    def __init__(self, db_path: str = ".seisproc/job_history.db"):
        self._engine = create_engine(f"sqlite:///{db_path}")

    def save_job(self, job: Job) -> None:
        """Save completed job to history."""

    def get_recent(self, limit: int = 50) -> List[Job]:
        """Get recent jobs."""

    def query(self,
              job_types: List[JobType] = None,
              states: List[JobState] = None,
              date_from: datetime = None,
              date_to: datetime = None) -> List[Job]:
        """Query job history with filters."""

    def get_statistics(self, days: int = 30) -> JobStatistics:
        """Get aggregate statistics."""

    def cleanup(self, keep_days: int = 90) -> int:
        """Remove old history entries."""
```

**Tests:**
```python
# tests/test_job_history.py
def test_job_saved_to_history():
    """Completed jobs are saved to history."""

def test_job_queried_by_type():
    """Jobs can be queried by type."""

def test_statistics_calculated():
    """Statistics are calculated correctly."""

def test_history_cleanup():
    """Old jobs are cleaned up."""
```

---

### Task 4.2: History Integration with JobManager
**Duration:** 1 day

Automatically save jobs to history on completion/failure.

**Files to Modify:**
```
utils/ray_orchestration/job_manager.py
```

**Implementation:**
- Add history storage integration
- Save on complete_job(), fail_job(), finalize_cancellation()
- Add get_job_history() method

---

## Week 14: Analytics Dashboard

### Task 4.3: Job Analytics Widget
**Duration:** 3 days

Create analytics dashboard for job performance insights.

**Files to Create:**
```
views/job_dashboard/analytics_widget.py
```

**Implementation:**
```python
class JobAnalyticsWidget(QWidget):
    """Job analytics and statistics display."""

    def __init__(self):
        # Summary cards (total jobs, success rate, avg duration)
        # Job type breakdown chart
        # Daily/weekly job count chart
        # Performance trends
        # Top processors by usage

    def refresh(self):
        """Refresh analytics from history."""

    def export_report(self, path: str):
        """Export analytics report."""
```

**Charts to Include:**
1. Jobs by status (pie chart)
2. Jobs over time (line chart)
3. Average duration by job type (bar chart)
4. Success/failure rate trend
5. Resource utilization over time

**Tests:**
```python
# tests/test_analytics_widget.py
def test_analytics_shows_summary(qtbot):
    """Analytics shows summary statistics."""

def test_analytics_charts_render(qtbot):
    """Analytics charts render correctly."""

def test_analytics_export_works():
    """Analytics can be exported to file."""
```

---

## Week 15: Alert System Enhancement

### Task 4.4: Alert Manager with Rules Engine
**Duration:** 2 days

Enhance existing ResourceMonitor alerts with configurable rules.

**Files to Create:**
```
utils/ray_orchestration/alerts/
├── __init__.py
├── manager.py          # AlertManager
├── rules.py            # AlertRule, built-in rules
└── actions.py          # Alert actions (toast, log, sound)
```

**Implementation:**
```python
@dataclass
class AlertRule:
    """Configurable alert rule."""
    name: str
    condition: Callable[[ResourceSnapshot], bool]
    severity: str  # 'info', 'warning', 'critical'
    message_template: str
    cooldown_seconds: int = 60
    actions: List[str] = field(default_factory=lambda: ['toast', 'log'])

class AlertManager:
    """Manages alert rules and notifications."""

    def __init__(self):
        self._rules: List[AlertRule] = []
        self._last_triggered: Dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""

    def check(self, snapshot: ResourceSnapshot):
        """Check all rules against snapshot."""

    def get_default_rules(self) -> List[AlertRule]:
        """Get built-in alert rules."""
        return [
            AlertRule(
                name="high_memory",
                condition=lambda s: s.memory_percent > 85,
                severity="warning",
                message_template="Memory usage at {memory_percent:.0f}%"
            ),
            AlertRule(
                name="critical_memory",
                condition=lambda s: s.memory_percent > 95,
                severity="critical",
                message_template="CRITICAL: Memory at {memory_percent:.0f}%"
            ),
            AlertRule(
                name="job_stuck",
                condition=lambda s: s.metrics.get('job_no_progress_seconds', 0) > 300,
                severity="warning",
                message_template="Job appears stuck - no progress for 5 minutes"
            ),
        ]
```

**Tests:**
```python
# tests/test_alert_manager.py
def test_alert_triggers_on_condition():
    """Alert triggers when condition met."""

def test_alert_respects_cooldown():
    """Alert doesn't spam during cooldown."""

def test_custom_rule_works():
    """Custom alert rules can be added."""
```

---

### Task 4.5: Toast Notification Widget
**Duration:** 1 day

Create non-intrusive toast notifications for alerts.

**Files to Create:**
```
views/widgets/toast_notification.py
```

**Implementation:**
```python
class ToastNotification(QWidget):
    """Animated toast notification."""

    def __init__(self, message: str, severity: str, parent=None):
        # Slide-in animation
        # Auto-dismiss after timeout
        # Click to dismiss
        # Stack multiple toasts

    @classmethod
    def show_toast(cls, message: str, severity: str = 'info',
                   duration_ms: int = 5000):
        """Show a toast notification."""
```

---

## Week 16: Stress Testing & Documentation

### Task 4.6: Comprehensive Stress Tests
**Duration:** 2 days

Create stress tests to validate system under load.

**Files to Create:**
```
tests/test_stress.py
benchmarks/job_throughput.py
```

**Tests:**
```python
# tests/test_stress.py
class TestStress:
    """Stress tests for job system."""

    def test_many_concurrent_jobs(self):
        """System handles 20 concurrent jobs."""

    def test_rapid_job_submission(self):
        """System handles rapid job submission."""

    def test_cancellation_under_load(self):
        """Cancellation works under heavy load."""

    def test_memory_stability(self):
        """Memory doesn't leak over many jobs."""

    def test_checkpoint_under_load(self):
        """Checkpointing works under load."""

# benchmarks/job_throughput.py
def benchmark_job_throughput():
    """Measure job throughput."""
    # Submit 100 small jobs
    # Measure time to complete all
    # Report jobs/second
```

---

### Task 4.7: Performance Benchmarks
**Duration:** 1 day

Create benchmarks for key operations.

**Files to Create:**
```
benchmarks/
├── __init__.py
├── cancellation_latency.py
├── checkpoint_throughput.py
└── progress_update_rate.py
```

**Benchmark Targets:**
| Operation | Target |
|-----------|--------|
| Cancellation latency | < 2 seconds |
| Checkpoint save | < 100ms |
| Progress update | < 10ms |
| Job submission | < 50ms |

---

### Task 4.8: User Documentation
**Duration:** 2 days

Create comprehensive user documentation.

**Files to Create:**
```
docs/
├── user_guide_job_management.md    # How to use job dashboard
├── migration_guide_ray.md          # Migrating from old system
├── troubleshooting_jobs.md         # Common issues and solutions
└── api_reference_jobs.md           # API documentation
```

**Documentation Topics:**
1. Job Dashboard Overview
2. Submitting and Managing Jobs
3. Understanding Job States
4. Pause/Resume/Cancel Operations
5. Checkpoint and Recovery
6. Resource Monitoring
7. Troubleshooting Guide

---

## Phase 4 Completion Checklist

| Task | Priority | Duration | Test Count | Status |
|------|----------|----------|------------|--------|
| 4.0.1 Resource Monitor Widget | High | 2 days | 3 | ☐ |
| 4.0.2 Metal Worker Actor | Medium | 3 days | 3 | ☐ |
| 4.0.3 Main Window Integration | High | 1 day | 1 | ☐ |
| 4.1 Job History Storage | High | 3 days | 4 | ☐ |
| 4.2 History Integration | Medium | 1 day | 2 | ☐ |
| 4.3 Analytics Widget | Medium | 3 days | 3 | ☐ |
| 4.4 Alert Manager | Medium | 2 days | 3 | ☐ |
| 4.5 Toast Notifications | Low | 1 day | 2 | ☐ |
| 4.6 Stress Tests | High | 2 days | 5 | ☐ |
| 4.7 Performance Benchmarks | Medium | 1 day | 4 | ☐ |
| 4.8 Documentation | High | 2 days | 0 | ☐ |

**Total New Tests:** 30
**Total Duration:** ~3 weeks (21 days of effort)

---

## Deferred to Future Phase

The following items from the original plan are deferred:

### Rust SEGY Module
**Reason:** The existing Python SEGY adapter works well. Rust optimization should be a separate initiative when performance bottlenecks are proven.

**Recommended Approach:**
1. Profile existing SEGY I/O to identify bottlenecks
2. If bottlenecks exist, create targeted Rust optimization
3. Keep Python fallback for compatibility

### Numba Worker Actor
**Reason:** Metal GPU acceleration covers most compute-intensive operations. Numba can be added later if CPU-only optimization is needed.

---

## Success Criteria

Phase 4 is complete when:

1. **Functionality:**
   - [ ] Job history persists across sessions
   - [ ] Analytics show meaningful statistics
   - [ ] Alerts trigger and display correctly
   - [ ] Dashboard accessible from main window

2. **Performance:**
   - [ ] Cancellation < 2 seconds under load
   - [ ] 20 concurrent jobs run without issues
   - [ ] No memory leaks after 100 jobs

3. **Quality:**
   - [ ] All 160 tests passing (130 existing + 30 new)
   - [ ] Documentation complete
   - [ ] No critical bugs

---

## Dependencies

```
Phase 4 Dependencies:
├── Gap 4.0.1 (Resource Monitor Widget)
│   └── Depends on: ResourceMonitor (Phase 3) ✅
├── Gap 4.0.2 (Metal Worker)
│   └── Depends on: BaseWorkerActor (Phase 3) ✅
├── Gap 4.0.3 (Main Window Integration)
│   └── Depends on: JobDashboardWidget (Phase 2) ✅
├── Task 4.1 (Job History)
│   └── Depends on: Job models (Phase 1) ✅
├── Task 4.3 (Analytics)
│   └── Depends on: Job History (Task 4.1)
├── Task 4.4 (Alert Manager)
│   └── Depends on: ResourceMonitor (Phase 3) ✅
└── Task 4.6 (Stress Tests)
    └── Depends on: All previous tasks
```

---

## Recommended Order of Implementation

```
Week 1: Gaps + History
├── Day 1-2: Gap 4.0.1 Resource Monitor Widget
├── Day 3: Gap 4.0.3 Main Window Integration
├── Day 4-5: Task 4.1 Job History Storage

Week 2: Analytics + Alerts
├── Day 1: Task 4.2 History Integration
├── Day 2-4: Task 4.3 Analytics Widget
├── Day 5: Task 4.4 Alert Manager

Week 3: Polish + Testing
├── Day 1: Task 4.5 Toast Notifications
├── Day 2-3: Gap 4.0.2 Metal Worker Actor
├── Day 4: Task 4.6-4.7 Stress Tests + Benchmarks
├── Day 5: Task 4.8 Documentation
```

---

## Notes

1. **Metal Worker** can be implemented in parallel with other tasks if resources allow
2. **Stress tests** should be run on the target hardware configuration
3. **Documentation** should be written incrementally as features are completed
4. Consider adding **telemetry** for production usage patterns (optional)
