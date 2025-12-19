# Hybrid Architecture Implementation Plan
## Detailed Tasks, Tests, and File Mapping

**Version:** 1.0
**Date:** 2024-12-19
**Approach:** Hybrid (New orchestration + Keep existing computation)

---

## Table of Contents

1. [Phase 1: Foundation](#phase-1-foundation-weeks-1-4)
2. [Phase 2: UI & SEGY](#phase-2-ui--segy-weeks-5-8)
3. [Phase 3: Processing Integration](#phase-3-processing-integration-weeks-9-12)
4. [Phase 4: Polish & Optimization](#phase-4-polish--optimization-weeks-13-16)
5. [File Mapping](#file-mapping)
6. [Test Summary](#test-summary)

---

## Phase 1: Foundation (Weeks 1-4)

### Week 1: Ray Infrastructure Setup

#### Task 1.1: Install and Configure Ray
**Duration:** 2 days

**Actions:**
1. Add Ray to dependencies in `pyproject.toml`
2. Create `utils/ray_orchestration/__init__.py`
3. Create basic Ray initialization module
4. Test Ray cluster startup/shutdown

**Files to Create:**
```
utils/ray_orchestration/
├── __init__.py
├── cluster.py          # Ray cluster management
└── config.py           # Ray configuration
```

**Test 1.1.1: Ray Cluster Initialization**
```python
# tests/test_ray_cluster.py
def test_ray_cluster_starts():
    """Ray cluster initializes successfully."""
    from utils.ray_orchestration import initialize_ray, shutdown_ray

    result = initialize_ray()

    assert result is True
    assert ray.is_initialized()

    shutdown_ray()
    assert not ray.is_initialized()
```

**Expected Output:**
```
tests/test_ray_cluster.py::test_ray_cluster_starts PASSED
Ray initialized with 12 CPUs, 32GB memory
Ray dashboard available at http://127.0.0.1:8265
Ray shutdown complete
```

---

#### Task 1.2: Create Job Data Models
**Duration:** 2 days

**Actions:**
1. Create Job, JobProgress, JobState dataclasses
2. Create ResourceRequirements model
3. Create WorkerProgress model
4. Add serialization/deserialization methods

**Files to Create:**
```
models/
├── job.py              # Job, JobState, JobType
├── job_progress.py     # JobProgress, WorkerProgress
└── job_config.py       # JobConfig, ResourceRequirements
```

**Test 1.2.1: Job Model Serialization**
```python
# tests/test_job_models.py
def test_job_serialization():
    """Job model serializes and deserializes correctly."""
    from models.job import Job, JobType, JobState

    job = Job(
        id=uuid.uuid4(),
        name="Test Import",
        job_type=JobType.SEGY_IMPORT,
        state=JobState.CREATED,
        priority=5
    )

    serialized = job.to_dict()
    restored = Job.from_dict(serialized)

    assert restored.name == job.name
    assert restored.job_type == job.job_type
    assert restored.state == job.state
```

**Expected Output:**
```
tests/test_job_models.py::test_job_serialization PASSED
Job serialized to 245 bytes
Job restored with id=7f3a2b1c-...
```

---

#### Task 1.3: Create Job State Machine
**Duration:** 2 days

**Actions:**
1. Implement state transition logic
2. Add validation for allowed transitions
3. Create state change event emission
4. Add timestamp tracking for transitions

**Files to Create:**
```
utils/ray_orchestration/
└── state_machine.py    # JobStateMachine class
```

**Test 1.3.1: State Transitions**
```python
# tests/test_job_state_machine.py
def test_valid_state_transitions():
    """Valid state transitions are allowed."""
    from utils.ray_orchestration.state_machine import JobStateMachine

    sm = JobStateMachine()

    assert sm.can_transition(JobState.CREATED, JobState.QUEUED)
    assert sm.can_transition(JobState.QUEUED, JobState.RUNNING)
    assert sm.can_transition(JobState.RUNNING, JobState.PAUSED)
    assert not sm.can_transition(JobState.COMPLETED, JobState.RUNNING)

def test_invalid_transition_raises():
    """Invalid state transitions raise exception."""
    sm = JobStateMachine()

    with pytest.raises(InvalidStateTransition):
        sm.transition(JobState.COMPLETED, JobState.QUEUED)
```

**Expected Output:**
```
tests/test_job_state_machine.py::test_valid_state_transitions PASSED
tests/test_job_state_machine.py::test_invalid_transition_raises PASSED
Validated 12 valid transitions, blocked 8 invalid transitions
```

---

### Week 2: Job Queue and Scheduler

#### Task 1.4: Create Priority Job Queue
**Duration:** 2 days

**Actions:**
1. Implement thread-safe priority queue
2. Add job dependency tracking
3. Create queue persistence (optional)
4. Implement queue statistics

**Files to Create:**
```
utils/ray_orchestration/
├── job_queue.py        # PriorityJobQueue class
└── dependency_graph.py # Job dependency tracking
```

**Test 1.4.1: Priority Queue Ordering**
```python
# tests/test_job_queue.py
def test_priority_queue_ordering():
    """Higher priority jobs are scheduled first."""
    from utils.ray_orchestration.job_queue import PriorityJobQueue

    queue = PriorityJobQueue()

    job_low = create_job(priority=3)
    job_high = create_job(priority=8)
    job_medium = create_job(priority=5)

    queue.enqueue(job_low)
    queue.enqueue(job_high)
    queue.enqueue(job_medium)

    assert queue.dequeue().priority == 8
    assert queue.dequeue().priority == 5
    assert queue.dequeue().priority == 3
```

**Expected Output:**
```
tests/test_job_queue.py::test_priority_queue_ordering PASSED
Queue size: 3 -> 2 -> 1 -> 0
Dequeue order: priority 8, 5, 3
```

---

#### Task 1.5: Create Job Scheduler
**Duration:** 3 days

**Actions:**
1. Implement resource-aware scheduling
2. Add dependency resolution
3. Create worker pool management
4. Implement scheduling policies

**Files to Create:**
```
utils/ray_orchestration/
├── scheduler.py        # JobScheduler class
└── resource_manager.py # Resource tracking
```

**Test 1.5.1: Resource-Aware Scheduling**
```python
# tests/test_scheduler.py
def test_resource_aware_scheduling():
    """Scheduler respects memory constraints."""
    from utils.ray_orchestration.scheduler import JobScheduler

    scheduler = JobScheduler(max_memory_gb=16)

    job_large = create_job(memory_gb=12)
    job_small = create_job(memory_gb=4)

    scheduler.submit(job_large)
    scheduler.submit(job_small)

    # Large job starts first (higher resource need)
    running = scheduler.get_running_jobs()
    assert len(running) == 1
    assert running[0].memory_gb == 12

    # Small job waits (not enough memory)
    queued = scheduler.get_queued_jobs()
    assert len(queued) == 1
```

**Expected Output:**
```
tests/test_scheduler.py::test_resource_aware_scheduling PASSED
Available memory: 16 GB
Scheduled job requiring 12 GB
Queued job requiring 4 GB (waiting for resources)
```

---

### Week 3: Cancellation Infrastructure

#### Task 1.6: Implement Multi-Level Cancellation
**Duration:** 3 days

**Actions:**
1. Create CancellationToken class
2. Implement Ray-level cancellation
3. Add Python threading.Event propagation
4. Create cancellation timeout handling

**Files to Create:**
```
utils/ray_orchestration/
├── cancellation.py     # CancellationToken, CancellationManager
└── timeout.py          # Timeout handling utilities
```

**Test 1.6.1: Cancellation Propagation**
```python
# tests/test_cancellation.py
def test_cancellation_propagates():
    """Cancellation signal propagates to workers."""
    from utils.ray_orchestration.cancellation import CancellationManager

    manager = CancellationManager()
    token = manager.create_token("job-123")

    # Simulate worker checking token
    assert not token.is_cancelled()

    # Cancel the job
    manager.cancel("job-123")

    # Worker sees cancellation
    assert token.is_cancelled()

def test_cancellation_timeout():
    """Cancellation with timeout forces termination."""
    manager = CancellationManager()
    token = manager.create_token("job-456")

    # Request cancellation with 2 second timeout
    result = manager.cancel("job-456", timeout_seconds=2, force_after_timeout=True)

    assert result.graceful_shutdown or result.force_terminated
```

**Expected Output:**
```
tests/test_cancellation.py::test_cancellation_propagates PASSED
Cancellation token created for job-123
Cancellation requested at 14:23:05.123
Worker received cancellation at 14:23:05.125 (2ms latency)

tests/test_cancellation.py::test_cancellation_timeout PASSED
Graceful shutdown completed in 1.2 seconds
```

---

#### Task 1.7: Upgrade pybind11 for Free-Threading
**Duration:** 2 days

**Actions:**
1. Update pybind11 to v2.13.0+
2. Add `mod_gil_not_used` to bindings.cpp
3. Add thread safety to device_manager
4. Test with Python 3.13t (if available)

**Files to Modify:**
```
seismic_metal/
├── src/bindings.cpp       # Add mod_gil_not_used
└── src/device_manager.mm  # Add mutex locking
```

**Test 1.7.1: Metal Kernels Thread Safety**
```python
# tests/test_metal_threading.py
def test_metal_concurrent_calls():
    """Metal kernels handle concurrent calls safely."""
    import concurrent.futures
    from seismic_metal import dwt_denoise

    data = np.random.randn(1000, 100).astype(np.float32)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(dwt_denoise, data, "db4", 4, "soft", 3.0)
            for _ in range(4)
        ]

        results = [f.result() for f in futures]

    # All calls should succeed without crashes
    assert len(results) == 4
    for result, metrics in results:
        assert result.shape == data.shape
```

**Expected Output:**
```
tests/test_metal_threading.py::test_metal_concurrent_calls PASSED
Thread 0: DWT completed in 12.3ms
Thread 1: DWT completed in 13.1ms
Thread 2: DWT completed in 12.8ms
Thread 3: DWT completed in 13.5ms
All 4 concurrent Metal calls succeeded
```

---

### Week 4: Basic Ray Workers

#### Task 1.8: Create Base Worker Actor
**Duration:** 2 days

**Actions:**
1. Create base Ray actor class
2. Implement progress reporting
3. Add cancellation checking
4. Create worker lifecycle management

**Files to Create:**
```
utils/ray_orchestration/
└── workers/
    ├── __init__.py
    ├── base_worker.py      # BaseWorkerActor
    └── progress_reporter.py # ProgressReporter
```

**Test 1.8.1: Worker Progress Reporting**
```python
# tests/test_worker_progress.py
def test_worker_reports_progress():
    """Worker actor reports progress correctly."""
    from utils.ray_orchestration.workers import BaseWorkerActor

    worker = BaseWorkerActor.remote()
    progress_updates = []

    def on_progress(update):
        progress_updates.append(update)

    ray.get(worker.process_with_callback.remote(
        data=test_data,
        callback=on_progress
    ))

    assert len(progress_updates) > 0
    assert progress_updates[-1].percent == 100.0
```

**Expected Output:**
```
tests/test_worker_progress.py::test_worker_reports_progress PASSED
Progress: 0% -> 25% -> 50% -> 75% -> 100%
Total updates received: 5
Final status: completed
```

---

#### Task 1.9: Create Metal Worker Actor
**Duration:** 2 days

**Actions:**
1. Create MetalWorkerActor for GPU processing
2. Handle Metal device initialization per process
3. Integrate with existing kernel_backend.py
4. Add GPU memory monitoring

**Files to Create:**
```
utils/ray_orchestration/
└── workers/
    └── metal_worker.py     # MetalWorkerActor
```

**Test 1.9.1: Metal Worker Initialization**
```python
# tests/test_metal_worker.py
def test_metal_worker_initializes():
    """Metal worker initializes GPU correctly."""
    from utils.ray_orchestration.workers import MetalWorkerActor

    worker = MetalWorkerActor.remote()

    info = ray.get(worker.get_device_info.remote())

    assert info['available'] == True
    assert 'Apple' in info['device_name']

def test_metal_worker_processes():
    """Metal worker processes data correctly."""
    worker = MetalWorkerActor.remote()

    data = np.random.randn(1000, 50).astype(np.float32)

    result = ray.get(worker.dwt_denoise.remote(
        data, wavelet="db4", level=4
    ))

    assert result.shape == data.shape
    assert not np.allclose(result, data)  # Actually processed
```

**Expected Output:**
```
tests/test_metal_worker.py::test_metal_worker_initializes PASSED
Metal device: Apple M4 Max
GPU memory: 64 GB unified

tests/test_metal_worker.py::test_metal_worker_processes PASSED
DWT processing completed in 8.2ms
GPU utilization: 94%
```

---

#### Task 1.10: Integration Test - Basic Job Execution
**Duration:** 2 days

**Actions:**
1. Create end-to-end test with simple job
2. Verify job lifecycle (create -> queue -> run -> complete)
3. Test cancellation during execution
4. Verify progress reporting

**Test 1.10.1: End-to-End Job Execution**
```python
# tests/test_integration_phase1.py
def test_end_to_end_job_execution():
    """Complete job lifecycle works correctly."""
    from utils.ray_orchestration import JobManager

    manager = JobManager()

    # Create job
    job = manager.create_job(
        name="Test Processing",
        job_type=JobType.PROCESSING,
        config={'processor': 'dwt_denoise', 'wavelet': 'db4'}
    )
    assert job.state == JobState.CREATED

    # Submit job
    manager.submit(job.id)
    assert job.state == JobState.QUEUED

    # Wait for completion
    result = manager.wait_for_completion(job.id, timeout=60)

    assert result.success == True
    assert job.state == JobState.COMPLETED
    assert result.progress.percent == 100.0

def test_job_cancellation_during_execution():
    """Job can be cancelled while running."""
    manager = JobManager()

    # Create long-running job
    job = manager.create_job(
        name="Long Processing",
        job_type=JobType.PROCESSING,
        config={'processor': 'slow_test', 'duration': 30}
    )

    manager.submit(job.id)

    # Wait until running
    while job.state != JobState.RUNNING:
        time.sleep(0.1)

    # Cancel
    start = time.time()
    manager.cancel(job.id)
    elapsed = time.time() - start

    assert job.state == JobState.CANCELLED
    assert elapsed < 2.0  # Cancelled within 2 seconds
```

**Expected Output:**
```
tests/test_integration_phase1.py::test_end_to_end_job_execution PASSED
Job created: id=abc123, state=CREATED
Job submitted: state=QUEUED
Job started: state=RUNNING
Progress: 25% -> 50% -> 75% -> 100%
Job completed: state=COMPLETED, duration=5.2s

tests/test_integration_phase1.py::test_job_cancellation_during_execution PASSED
Job running at 35% progress
Cancellation requested
Worker received cancellation in 45ms
Job cancelled: state=CANCELLED
Total cancellation time: 1.2s
```

---

## Phase 1 Completion Checklist

| Task | Status | Test Count | Notes |
|------|--------|------------|-------|
| 1.1 Ray Setup | ☐ | 2 | |
| 1.2 Job Models | ☐ | 3 | |
| 1.3 State Machine | ☐ | 2 | |
| 1.4 Job Queue | ☐ | 3 | |
| 1.5 Scheduler | ☐ | 2 | |
| 1.6 Cancellation | ☐ | 4 | |
| 1.7 pybind11 Upgrade | ☐ | 2 | |
| 1.8 Base Worker | ☐ | 2 | |
| 1.9 Metal Worker | ☐ | 3 | |
| 1.10 Integration | ☐ | 2 | |

**Total Phase 1 Tests:** 25
**Expected Phase 1 Duration:** 4 weeks

---

## Phase 2: UI & SEGY (Weeks 5-8)

### Week 5: Job Dashboard Widget

#### Task 2.1: Create Job Dashboard Widget
**Duration:** 3 days

**Actions:**
1. Create QWidget-based job dashboard
2. Add active jobs list with progress bars
3. Add queued jobs list with controls
4. Add recent jobs history

**Files to Create:**
```
views/
├── job_dashboard.py           # JobDashboardWidget
└── widgets/
    ├── job_card.py            # JobCardWidget
    ├── progress_bar.py        # EnhancedProgressBar
    └── job_queue_widget.py    # JobQueueWidget
```

**Test 2.1.1: Job Dashboard Updates**
```python
# tests/test_job_dashboard.py
def test_dashboard_shows_active_jobs(qtbot):
    """Dashboard displays active jobs correctly."""
    from views.job_dashboard import JobDashboardWidget

    dashboard = JobDashboardWidget()
    qtbot.addWidget(dashboard)

    # Simulate job started
    dashboard.on_job_started("job-123", {
        'name': 'Test Job',
        'progress': 0
    })

    # Check job appears in active list
    assert dashboard.active_jobs_count() == 1
    assert "Test Job" in dashboard.get_active_job_names()

def test_dashboard_progress_updates(qtbot):
    """Dashboard updates progress in real-time."""
    dashboard = JobDashboardWidget()
    qtbot.addWidget(dashboard)

    dashboard.on_job_started("job-123", {'name': 'Test', 'progress': 0})

    # Simulate progress updates
    for pct in [25, 50, 75, 100]:
        dashboard.on_progress_updated("job-123", {'percent': pct})
        qtbot.wait(10)

    # Check final progress shown
    progress = dashboard.get_job_progress("job-123")
    assert progress == 100
```

**Expected Output:**
```
tests/test_job_dashboard.py::test_dashboard_shows_active_jobs PASSED
Dashboard initialized
Job card created for "Test Job"
Active jobs: 1

tests/test_job_dashboard.py::test_dashboard_progress_updates PASSED
Progress updates: 25% -> 50% -> 75% -> 100%
UI updated 4 times
Final progress bar: 100%
```

---

#### Task 2.2: Create Resource Monitor Widget
**Duration:** 2 days

**Actions:**
1. Create resource monitoring widget
2. Add CPU/Memory/GPU gauges
3. Add worker status table
4. Integrate with existing memory_monitor.py

**Files to Create:**
```
views/
└── resource_monitor_widget.py  # ResourceMonitorWidget
```

**Test 2.2.1: Resource Monitor Updates**
```python
# tests/test_resource_monitor.py
def test_resource_monitor_displays(qtbot):
    """Resource monitor shows system stats."""
    from views.resource_monitor_widget import ResourceMonitorWidget

    monitor = ResourceMonitorWidget()
    qtbot.addWidget(monitor)

    # Trigger update
    monitor.refresh()

    assert monitor.cpu_usage >= 0
    assert monitor.memory_used_gb >= 0
    assert monitor.gpu_available in [True, False]
```

**Expected Output:**
```
tests/test_resource_monitor.py::test_resource_monitor_displays PASSED
CPU: 45%
Memory: 12.4 / 32.0 GB (39%)
GPU: Apple M4 Max (available)
Workers: 0 active
```

---

### Week 6: PyQt6 Signal Integration

#### Task 2.3: Create Job Manager Signals
**Duration:** 2 days

**Actions:**
1. Create QObject with job signals
2. Connect to Ray event handlers
3. Bridge async Ray events to Qt main thread
4. Implement signal throttling for performance

**Files to Create:**
```
utils/ray_orchestration/
└── qt_bridge.py           # QtSignalBridge, JobManagerSignals
```

**Test 2.3.1: Signal Emission**
```python
# tests/test_qt_bridge.py
def test_signals_emitted_on_job_events(qtbot):
    """Qt signals emitted when job state changes."""
    from utils.ray_orchestration.qt_bridge import JobManagerSignals

    signals = JobManagerSignals()
    received_signals = []

    signals.job_started.connect(lambda job_id: received_signals.append(('started', job_id)))
    signals.job_completed.connect(lambda job_id, success: received_signals.append(('completed', job_id)))

    # Simulate events
    signals.emit_job_started("job-123")
    signals.emit_job_completed("job-123", True)

    qtbot.wait(10)

    assert ('started', 'job-123') in received_signals
    assert ('completed', 'job-123') in received_signals
```

**Expected Output:**
```
tests/test_qt_bridge.py::test_signals_emitted_on_job_events PASSED
Signal emitted: job_started(job-123)
Signal emitted: job_completed(job-123, True)
Received 2 signals in main thread
```

---

#### Task 2.4: Connect Dashboard to Main Window
**Duration:** 2 days

**Actions:**
1. Add job dashboard to main window
2. Create menu actions for job management
3. Add toolbar buttons for common actions
4. Implement dock widget for dashboard

**Files to Modify:**
```
main_window.py              # Add job dashboard integration
```

**Test 2.4.1: Dashboard Integration**
```python
# tests/test_main_window_jobs.py
def test_job_dashboard_accessible(qtbot):
    """Job dashboard is accessible from main window."""
    from main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    # Open job dashboard
    window.show_job_dashboard()

    assert window.job_dashboard is not None
    assert window.job_dashboard.isVisible()
```

**Expected Output:**
```
tests/test_main_window_jobs.py::test_job_dashboard_accessible PASSED
Job dashboard docked on right side
Dashboard visible: True
Menu action: View -> Job Dashboard [checked]
```

---

### Week 7: Rust SEGY Module Setup

#### Task 2.5: Create Rust Project Structure
**Duration:** 2 days

**Actions:**
1. Create Cargo.toml with dependencies
2. Create pyproject.toml for maturin
3. Set up basic PyO3 module structure
4. Configure build system

**Files to Create:**
```
seisproc_rust/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs
│   └── segy/
│       └── mod.rs
└── tests/
    └── test_basic.rs
```

**Test 2.5.1: Rust Module Builds**
```bash
# Build test
cd seisproc_rust
maturin build --release

# Import test
python -c "import seisproc_rust; print(seisproc_rust.__version__)"
```

**Expected Output:**
```
$ maturin build --release
   Compiling seisproc_rust v0.1.0
    Finished release [optimized] target(s) in 12.34s
    Built wheel for python 3.13

$ python -c "import seisproc_rust; print(seisproc_rust.__version__)"
0.1.0
```

---

#### Task 2.6: Implement Rust SEGY Reader
**Duration:** 3 days

**Actions:**
1. Implement memory-mapped file reading
2. Add IBM float to IEEE float conversion
3. Implement header parsing with Rayon
4. Create Python bindings for reader

**Files to Create:**
```
seisproc_rust/src/
├── segy/
│   ├── reader.rs
│   ├── header.rs
│   └── formats.rs
└── lib.rs            # Update with bindings
```

**Test 2.6.1: Rust SEGY Reading**
```python
# tests/test_rust_segy.py
def test_rust_segy_reads_file():
    """Rust SEGY reader reads file correctly."""
    from seisproc_rust import read_segy_traces

    traces, headers = read_segy_traces(
        "test_data/small.sgy",
        start_trace=0,
        end_trace=100
    )

    assert traces.shape == (100, 2000)  # 100 traces, 2000 samples
    assert len(headers) == 100

def test_rust_segy_faster_than_python():
    """Rust SEGY reader is faster than Python."""
    from seisproc_rust import read_segy_traces
    from utils.segy_import.segy_reader import SEGYReader

    # Rust timing
    start = time.time()
    rust_traces, _ = read_segy_traces("test_data/medium.sgy", 0, 10000)
    rust_time = time.time() - start

    # Python timing
    start = time.time()
    reader = SEGYReader("test_data/medium.sgy")
    python_traces = reader.read_traces(0, 10000)
    python_time = time.time() - start

    speedup = python_time / rust_time
    assert speedup >= 2.0  # At least 2x faster
```

**Expected Output:**
```
tests/test_rust_segy.py::test_rust_segy_reads_file PASSED
Read 100 traces x 2000 samples in 1.2ms
Headers parsed: 100 with 40 fields each

tests/test_rust_segy.py::test_rust_segy_faster_than_python PASSED
Rust time: 0.45s
Python time: 2.34s
Speedup: 5.2x
```

---

### Week 8: Rust SEGY Integration

#### Task 2.7: Create Rust Import Worker
**Duration:** 2 days

**Actions:**
1. Create RustImportWorker Ray actor
2. Integrate with Rust SEGY reader
3. Add progress reporting
4. Implement cancellation support

**Files to Create:**
```
utils/ray_orchestration/
└── workers/
    └── rust_segy_worker.py  # RustSEGYWorker
```

**Test 2.7.1: Rust Import Worker**
```python
# tests/test_rust_import_worker.py
def test_rust_import_worker():
    """Rust import worker processes SEGY file."""
    from utils.ray_orchestration.workers import RustSEGYWorker

    worker = RustSEGYWorker.remote()

    result = ray.get(worker.import_segment.remote(
        segy_path="test_data/medium.sgy",
        start_trace=0,
        end_trace=10000,
        output_zarr="test_output.zarr"
    ))

    assert result.success == True
    assert result.traces_imported == 10000
```

**Expected Output:**
```
tests/test_rust_import_worker.py::test_rust_import_worker PASSED
Imported 10000 traces in 0.89s
Throughput: 11,236 traces/second
Written to test_output.zarr
```

---

#### Task 2.8: Implement Pause/Resume with Checkpoints
**Duration:** 3 days

**Actions:**
1. Create CheckpointManager class
2. Implement checkpoint writing during processing
3. Add checkpoint reading for resume
4. Create pause event handling

**Files to Create:**
```
utils/ray_orchestration/
├── checkpoint.py           # CheckpointManager
└── pause_resume.py         # PauseResumeManager
```

**Test 2.8.1: Checkpoint Save and Resume**
```python
# tests/test_checkpoint.py
def test_checkpoint_save_and_resume():
    """Job can be paused and resumed from checkpoint."""
    from utils.ray_orchestration.checkpoint import CheckpointManager

    manager = CheckpointManager(checkpoint_dir="/tmp/checkpoints")

    # Simulate partial progress
    state = {
        'job_id': 'job-123',
        'completed_segments': [0, 1, 2],
        'total_segments': 10,
        'progress_percent': 30.0
    }

    # Save checkpoint
    manager.save(state)

    # Resume from checkpoint
    restored = manager.load('job-123')

    assert restored['completed_segments'] == [0, 1, 2]
    assert restored['progress_percent'] == 30.0

def test_resume_skips_completed_segments():
    """Resume skips already completed segments."""
    manager = CheckpointManager()
    job_manager = JobManager()

    # Create checkpoint indicating segments 0-4 complete
    manager.save({
        'job_id': 'job-456',
        'completed_segments': [0, 1, 2, 3, 4],
        'total_segments': 10
    })

    # Resume job
    job = job_manager.resume_from_checkpoint('job-456')

    # Only segments 5-9 should be scheduled
    pending = job_manager.get_pending_segments(job.id)
    assert pending == [5, 6, 7, 8, 9]
```

**Expected Output:**
```
tests/test_checkpoint.py::test_checkpoint_save_and_resume PASSED
Checkpoint saved: /tmp/checkpoints/job-123.json (245 bytes)
Checkpoint loaded: 3/10 segments complete

tests/test_checkpoint.py::test_resume_skips_completed_segments PASSED
Resumed job-456: skipping 5 completed segments
Pending segments: [5, 6, 7, 8, 9]
```

---

## Phase 2 Completion Checklist

| Task | Status | Test Count | Notes |
|------|--------|------------|-------|
| 2.1 Job Dashboard | ☐ | 3 | |
| 2.2 Resource Monitor | ☐ | 2 | |
| 2.3 Qt Signals | ☐ | 2 | |
| 2.4 Main Window Integration | ☐ | 2 | |
| 2.5 Rust Project Setup | ☐ | 1 | |
| 2.6 Rust SEGY Reader | ☐ | 3 | |
| 2.7 Rust Import Worker | ☐ | 2 | |
| 2.8 Checkpoints | ☐ | 3 | |

**Total Phase 2 Tests:** 18
**Expected Phase 2 Duration:** 4 weeks

---

## Phase 3: Processing Integration (Weeks 9-12)

### Week 9: CPU Worker Actors

#### Task 3.1: Create CPU Worker Actor
**Duration:** 2 days

**Actions:**
1. Create CPUWorkerActor for Numba/Joblib processors
2. Wrap existing processors in Ray actors
3. Add resource limits (CPU, memory)
4. Implement batch processing

**Files to Create:**
```
utils/ray_orchestration/
└── workers/
    ├── cpu_worker.py        # CPUWorkerActor
    └── processor_wrapper.py # ProcessorWrapper
```

**Test 3.1.1: CPU Worker Processes**
```python
# tests/test_cpu_worker.py
def test_cpu_worker_processes_dwt():
    """CPU worker processes DWT correctly."""
    from utils.ray_orchestration.workers import CPUWorkerActor

    worker = CPUWorkerActor.remote(processor_type='dwt_denoise')

    data = np.random.randn(1000, 50).astype(np.float32)

    result = ray.get(worker.process.remote(data, wavelet='db4', level=4))

    assert result.shape == data.shape
```

**Expected Output:**
```
tests/test_cpu_worker.py::test_cpu_worker_processes_dwt PASSED
CPU worker initialized with DWTDenoise
Processed 50 traces in 234ms
Using Joblib backend with 6 workers
```

---

#### Task 3.2: Create Numba Worker Actor
**Duration:** 2 days

**Actions:**
1. Create NumbaWorkerActor for NMO, Deconvolution
2. Handle Numba JIT compilation in workers
3. Add cache warming on initialization
4. Implement batch processing

**Files to Create:**
```
utils/ray_orchestration/
└── workers/
    └── numba_worker.py      # NumbaWorkerActor
```

**Test 3.2.1: Numba Worker NMO**
```python
# tests/test_numba_worker.py
def test_numba_worker_nmo():
    """Numba worker processes NMO correctly."""
    from utils.ray_orchestration.workers import NumbaWorkerActor

    worker = NumbaWorkerActor.remote(processor_type='nmo')

    # Warm up JIT
    ray.get(worker.warmup.remote())

    # Process data
    data = create_test_gather()
    result = ray.get(worker.process.remote(data, velocity=2000.0))

    assert result.shape == data.shape
```

**Expected Output:**
```
tests/test_numba_worker.py::test_numba_worker_nmo PASSED
Numba JIT compiled in 1.2s (first call)
NMO correction applied in 45ms
Stretch mute: 30% applied
```

---

### Week 10: Processing Coordinator

#### Task 3.3: Create Processing Coordinator
**Duration:** 3 days

**Actions:**
1. Create coordinator for multi-worker processing
2. Implement gather partitioning
3. Add result aggregation
4. Integrate with job manager

**Files to Create:**
```
utils/ray_orchestration/
├── processing_coordinator.py  # ProcessingCoordinator
└── partitioner.py             # GatherPartitioner (enhanced)
```

**Test 3.3.1: Parallel Processing Coordination**
```python
# tests/test_processing_coordinator.py
def test_parallel_processing():
    """Coordinator distributes work across workers."""
    from utils.ray_orchestration.processing_coordinator import ProcessingCoordinator

    coordinator = ProcessingCoordinator(n_workers=4)

    result = coordinator.process(
        input_path="test_data/survey.zarr",
        output_path="test_output.zarr",
        processor_config={'type': 'dwt_denoise', 'wavelet': 'db4'}
    )

    assert result.success == True
    assert result.gathers_processed == 100
    assert result.traces_processed == 50000
```

**Expected Output:**
```
tests/test_processing_coordinator.py::test_parallel_processing PASSED
Partitioned 100 gathers across 4 workers
Worker 0: 25 gathers (12,500 traces)
Worker 1: 25 gathers (12,500 traces)
Worker 2: 25 gathers (12,500 traces)
Worker 3: 25 gathers (12,500 traces)
Total processing time: 4.5s
Throughput: 11,111 traces/second
```

---

#### Task 3.4: Migrate Existing Parallel Processing
**Duration:** 3 days

**Actions:**
1. Create adapter for existing coordinator
2. Add fallback to ProcessPoolExecutor
3. Implement gradual migration path
4. Update qc_batch_engine integration

**Files to Modify:**
```
utils/parallel_processing/
└── coordinator.py          # Add Ray backend option

processors/
├── qc_batch_engine.py      # Update to use Ray
└── qc_stacking_engine.py   # Update to use Ray
```

**Test 3.4.1: Migration Compatibility**
```python
# tests/test_migration_compatibility.py
def test_ray_produces_same_results():
    """Ray backend produces same results as ProcessPoolExecutor."""
    from utils.parallel_processing.coordinator import ParallelProcessingCoordinator

    # Process with old backend
    old_coordinator = ParallelProcessingCoordinator(backend='process_pool')
    old_result = old_coordinator.process(test_config)

    # Process with Ray backend
    new_coordinator = ParallelProcessingCoordinator(backend='ray')
    new_result = new_coordinator.process(test_config)

    # Compare results
    np.testing.assert_allclose(
        old_result.traces,
        new_result.traces,
        rtol=1e-5
    )
```

**Expected Output:**
```
tests/test_migration_compatibility.py::test_ray_produces_same_results PASSED
ProcessPoolExecutor: 100 gathers in 12.3s
Ray: 100 gathers in 8.7s
Results match within tolerance (rtol=1e-5)
Speedup: 1.41x
```

---

### Week 11: Multi-Level Cancellation Implementation

#### Task 3.5: Implement Rust Cancellation
**Duration:** 2 days

**Actions:**
1. Add AtomicBool cancellation flag to Rust
2. Check flag in Rayon parallel loops
3. Create Python bindings for cancellation
4. Test cancellation latency

**Files to Create/Modify:**
```
seisproc_rust/src/
├── parallel/
│   └── cancellation.rs     # Rust cancellation
└── lib.rs                  # Update bindings
```

**Test 3.5.1: Rust Cancellation**
```python
# tests/test_rust_cancellation.py
def test_rust_cancellation_fast():
    """Rust operations cancel within 100ms."""
    from seisproc_rust import read_segy_with_cancel

    cancel_flag = CancellationFlag()

    # Start reading in thread
    def read_async():
        return read_segy_with_cancel("large_file.sgy", cancel_flag)

    thread = threading.Thread(target=read_async)
    thread.start()

    # Cancel after 100ms
    time.sleep(0.1)
    start = time.time()
    cancel_flag.set()
    thread.join()
    cancel_time = time.time() - start

    assert cancel_time < 0.1  # Cancelled within 100ms
```

**Expected Output:**
```
tests/test_rust_cancellation.py::test_rust_cancellation_fast PASSED
Reading started at trace 0
Cancellation requested at trace 45,234
Operation cancelled in 23ms
Cleanup completed
```

---

#### Task 3.6: Implement Metal Cancellation
**Duration:** 2 days

**Actions:**
1. Add cancellation support to Metal kernels
2. Handle command buffer cleanup
3. Integrate with Python cancellation
4. Test GPU resource cleanup

**Files to Modify:**
```
seismic_metal/
├── src/bindings.cpp        # Add cancellation parameter
├── src/device_manager.mm   # Add cleanup methods
└── src/dwt_kernel.mm       # Check cancellation
```

**Test 3.6.1: Metal Cancellation**
```python
# tests/test_metal_cancellation.py
def test_metal_cancellation():
    """Metal operations handle cancellation correctly."""
    from seismic_metal import dwt_denoise_with_cancel

    cancel_flag = CancellationFlag()
    large_data = np.random.randn(10000, 1000).astype(np.float32)

    # Start in thread
    def process_async():
        return dwt_denoise_with_cancel(large_data, cancel_flag)

    thread = threading.Thread(target=process_async)
    thread.start()

    # Cancel after 50ms
    time.sleep(0.05)
    cancel_flag.set()
    thread.join()

    # Verify GPU resources released
    assert get_metal_command_buffer_count() == 0
```

**Expected Output:**
```
tests/test_metal_cancellation.py::test_metal_cancellation PASSED
Metal DWT started on 10,000 traces
Cancellation received at trace 2,345
Command buffer completed
GPU resources released
```

---

### Week 12: End-to-End Integration

#### Task 3.7: Full Processing Pipeline Test
**Duration:** 3 days

**Actions:**
1. Create comprehensive integration test
2. Test all worker types together
3. Verify cancellation at all levels
4. Measure end-to-end performance

**Test 3.7.1: Full Pipeline**
```python
# tests/test_full_pipeline.py
def test_full_processing_pipeline():
    """Complete processing pipeline works end-to-end."""
    from utils.ray_orchestration import JobManager

    manager = JobManager()

    # Create processing job
    job = manager.create_job(
        name="Full Pipeline Test",
        job_type=JobType.PROCESSING,
        config={
            'input': 'test_data/survey.zarr',
            'output': 'test_output.zarr',
            'processors': [
                {'type': 'agc', 'window_ms': 500},
                {'type': 'dwt_denoise', 'wavelet': 'db4'},
                {'type': 'bandpass', 'low': 10, 'high': 80}
            ]
        }
    )

    # Execute
    manager.submit(job.id)
    result = manager.wait_for_completion(job.id)

    assert result.success == True

    # Verify output
    output = zarr.open('test_output.zarr', 'r')
    assert output['traces'].shape[1] == input_trace_count

def test_cancellation_at_all_levels():
    """Cancellation works at Ray, Python, Rust, and Metal levels."""
    manager = JobManager()

    # Test each backend
    for backend in ['ray', 'rust', 'metal', 'numba']:
        job = manager.create_job(
            name=f"Cancel Test ({backend})",
            config={'backend': backend, 'duration': 30}
        )

        manager.submit(job.id)

        # Wait until running
        while job.state != JobState.RUNNING:
            time.sleep(0.1)

        # Cancel and measure time
        start = time.time()
        manager.cancel(job.id)
        cancel_time = time.time() - start

        assert job.state == JobState.CANCELLED
        assert cancel_time < 2.0, f"{backend} cancelled in {cancel_time}s"
```

**Expected Output:**
```
tests/test_full_pipeline.py::test_full_processing_pipeline PASSED
Job created: Full Pipeline Test
Phase 1: AGC (12,345 traces) - 2.3s
Phase 2: DWT (12,345 traces) - 4.5s
Phase 3: Bandpass (12,345 traces) - 1.2s
Total time: 8.0s
Output verified: 12,345 traces

tests/test_full_pipeline.py::test_cancellation_at_all_levels PASSED
ray: cancelled in 0.45s
rust: cancelled in 0.12s
metal: cancelled in 0.08s
numba: cancelled in 0.67s
All cancellations under 2s threshold
```

---

## Phase 3 Completion Checklist

| Task | Status | Test Count | Notes |
|------|--------|------------|-------|
| 3.1 CPU Worker | ☐ | 2 | |
| 3.2 Numba Worker | ☐ | 2 | |
| 3.3 Processing Coordinator | ☐ | 2 | |
| 3.4 Migration Compatibility | ☐ | 2 | |
| 3.5 Rust Cancellation | ☐ | 2 | |
| 3.6 Metal Cancellation | ☐ | 2 | |
| 3.7 Full Pipeline | ☐ | 3 | |

**Total Phase 3 Tests:** 15
**Expected Phase 3 Duration:** 4 weeks

---

## Phase 4: Polish & Optimization (Weeks 13-16)

### Week 13: Rust SEGY Export

#### Task 4.1: Implement Rust SEGY Writer
**Duration:** 3 days

**Actions:**
1. Implement buffered SEGY writing in Rust
2. Add header serialization
3. Implement segment merging
4. Add mute application

**Files to Create:**
```
seisproc_rust/src/
└── segy/
    ├── writer.rs
    └── merger.rs
```

**Test 4.1.1: Rust SEGY Export**
```python
# tests/test_rust_export.py
def test_rust_segy_export():
    """Rust SEGY export writes valid file."""
    from seisproc_rust import write_segy

    traces = np.random.randn(1000, 2000).astype(np.float32)
    headers = create_test_headers(1000)

    write_segy("test_output.sgy", traces, headers)

    # Verify with segyio
    with segyio.open("test_output.sgy") as f:
        assert len(f.trace) == 1000
        assert len(f.trace[0]) == 2000
```

**Expected Output:**
```
tests/test_rust_export.py::test_rust_segy_export PASSED
Wrote 1000 traces in 0.34s
Throughput: 2,941 traces/second
File size: 8.2 MB
Verified with segyio: OK
```

---

### Week 14: Job History and Analytics

#### Task 4.2: Create Job History Storage
**Duration:** 2 days

**Actions:**
1. Create SQLite job history database
2. Implement job persistence
3. Add query interface
4. Create history cleanup

**Files to Create:**
```
utils/ray_orchestration/
└── history/
    ├── __init__.py
    ├── storage.py          # JobHistoryStorage
    └── queries.py          # Query interface
```

**Test 4.2.1: Job History**
```python
# tests/test_job_history.py
def test_job_history_persists():
    """Completed jobs are saved to history."""
    from utils.ray_orchestration.history import JobHistoryStorage

    storage = JobHistoryStorage("test_history.db")

    # Save completed job
    job = create_completed_job()
    storage.save(job)

    # Query history
    history = storage.get_recent(limit=10)

    assert len(history) >= 1
    assert history[0].id == job.id
```

**Expected Output:**
```
tests/test_job_history.py::test_job_history_persists PASSED
Job saved to history: id=abc123
History contains 1 job
Query time: 2ms
```

---

#### Task 4.3: Create Analytics Dashboard
**Duration:** 3 days

**Actions:**
1. Create job analytics widget
2. Add performance charts
3. Implement summary statistics
4. Add export functionality

**Files to Create:**
```
views/
└── job_analytics_widget.py  # JobAnalyticsWidget
```

**Test 4.3.1: Analytics Display**
```python
# tests/test_analytics.py
def test_analytics_shows_stats(qtbot):
    """Analytics widget shows job statistics."""
    from views.job_analytics_widget import JobAnalyticsWidget

    widget = JobAnalyticsWidget()
    qtbot.addWidget(widget)

    # Load history
    widget.load_history()

    stats = widget.get_summary_stats()

    assert 'total_jobs' in stats
    assert 'avg_duration' in stats
    assert 'success_rate' in stats
```

**Expected Output:**
```
tests/test_analytics.py::test_analytics_shows_stats PASSED
Total jobs: 47
Success rate: 87%
Avg duration: 12m 34s
Total traces processed: 124.5M
```

---

### Week 15: Alert System

#### Task 4.4: Create Alert Manager
**Duration:** 2 days

**Actions:**
1. Create alert rule engine
2. Implement notification channels
3. Add toast notifications
4. Create system tray integration

**Files to Create:**
```
utils/ray_orchestration/
└── alerts/
    ├── __init__.py
    ├── manager.py          # AlertManager
    ├── rules.py            # AlertRule definitions
    └── notifications.py    # Notification channels

views/
└── widgets/
    └── toast_notification.py  # ToastNotification
```

**Test 4.4.1: Alert Triggering**
```python
# tests/test_alerts.py
def test_memory_alert_triggers():
    """Memory alert triggers when threshold exceeded."""
    from utils.ray_orchestration.alerts import AlertManager

    manager = AlertManager()
    alerts_received = []

    manager.on_alert(lambda a: alerts_received.append(a))

    # Simulate high memory
    manager.check_memory(used_percent=92)

    assert len(alerts_received) == 1
    assert alerts_received[0].severity == 'critical'
    assert 'memory' in alerts_received[0].message.lower()
```

**Expected Output:**
```
tests/test_alerts.py::test_memory_alert_triggers PASSED
Memory check: 92% used
Alert triggered: CRITICAL - Memory usage at 92%
Notification sent to: toast, log
```

---

### Week 16: Documentation and Testing

#### Task 4.5: Comprehensive Integration Tests
**Duration:** 3 days

**Actions:**
1. Create stress tests
2. Add performance benchmarks
3. Test error recovery
4. Verify all cancellation paths

**Test 4.5.1: Stress Test**
```python
# tests/test_stress.py
def test_many_concurrent_jobs():
    """System handles many concurrent jobs."""
    manager = JobManager()

    # Submit 20 jobs
    jobs = []
    for i in range(20):
        job = manager.create_job(name=f"Stress Test {i}")
        manager.submit(job.id)
        jobs.append(job)

    # Wait for all to complete
    for job in jobs:
        result = manager.wait_for_completion(job.id, timeout=300)
        assert result.success == True

    # Verify no resource leaks
    assert ray.available_resources()['CPU'] > 0
    assert manager.active_job_count() == 0
```

**Expected Output:**
```
tests/test_stress.py::test_many_concurrent_jobs PASSED
Submitted 20 jobs
Running concurrently: max 8 (limited by resources)
All jobs completed in 45.2s
No resource leaks detected
Final CPU available: 12
```

---

#### Task 4.6: User Documentation
**Duration:** 2 days

**Actions:**
1. Write user guide for job management
2. Document new UI features
3. Create migration guide from old system
4. Add troubleshooting section

**Files to Create:**
```
docs/
├── user_guide_job_management.md
├── migration_guide_ray.md
└── troubleshooting_jobs.md
```

---

## Phase 4 Completion Checklist

| Task | Status | Test Count | Notes |
|------|--------|------------|-------|
| 4.1 Rust SEGY Export | ☐ | 3 | |
| 4.2 Job History | ☐ | 2 | |
| 4.3 Analytics Dashboard | ☐ | 2 | |
| 4.4 Alert System | ☐ | 3 | |
| 4.5 Stress Tests | ☐ | 3 | |
| 4.6 Documentation | ☐ | 0 | No tests |

**Total Phase 4 Tests:** 13
**Expected Phase 4 Duration:** 4 weeks

---

## File Mapping

### Legend
- ✅ **KEEP** - No changes needed
- 🔧 **MODIFY** - Update existing file
- 🆕 **CREATE** - New file to create
- ❌ **DELETE** - Remove file (after migration)
- ⚠️ **DEPRECATE** - Mark as legacy, keep for fallback

---

### Core Application

| File | Action | Notes |
|------|--------|-------|
| `main.py` | ✅ KEEP | Entry point unchanged |
| `main_window.py` | 🔧 MODIFY | Add job dashboard integration |
| `main_debug.py` | ✅ KEEP | Debug entry point |
| `pyproject.toml` | 🔧 MODIFY | Add Ray, maturin dependencies |

---

### Models

| File | Action | Notes |
|------|--------|-------|
| `models/__init__.py` | 🔧 MODIFY | Export new job models |
| `models/app_settings.py` | ✅ KEEP | |
| `models/anisotropy_model.py` | ✅ KEEP | |
| `models/binning.py` | ✅ KEEP | |
| `models/dataset_navigator.py` | ✅ KEEP | |
| `models/fk_config.py` | ✅ KEEP | |
| `models/fkk_config.py` | ✅ KEEP | |
| `models/gather_navigator.py` | ✅ KEEP | |
| `models/header_mapping.py` | ✅ KEEP | |
| `models/header_schema.py` | ✅ KEEP | |
| `models/lazy_seismic_data.py` | ✅ KEEP | |
| `models/migration_config.py` | ✅ KEEP | |
| `models/migration_geometry.py` | ✅ KEEP | |
| `models/migration_job.py` | ✅ KEEP | |
| `models/seismic_data.py` | ✅ KEEP | |
| `models/seismic_volume.py` | ✅ KEEP | |
| `models/velocity_model.py` | ✅ KEEP | |
| `models/viewport_state.py` | ✅ KEEP | |
| `models/job.py` | 🆕 CREATE | Job, JobState, JobType |
| `models/job_progress.py` | 🆕 CREATE | JobProgress, WorkerProgress |
| `models/job_config.py` | 🆕 CREATE | JobConfig, ResourceRequirements |

---

### Processors

| File | Action | Notes |
|------|--------|-------|
| `processors/__init__.py` | ✅ KEEP | |
| `processors/agc.py` | ✅ KEEP | Wrap in Ray worker |
| `processors/bandpass_filter.py` | ✅ KEEP | Wrap in Ray worker |
| `processors/base_processor.py` | ✅ KEEP | |
| `processors/cdp_stacker.py` | ✅ KEEP | Wrap in Ray worker |
| `processors/chunked_processor.py` | ✅ KEEP | |
| `processors/deconvolution.py` | ✅ KEEP | Wrap in Numba worker |
| `processors/denoise_3d.py` | ✅ KEEP | Wrap in Numba worker |
| `processors/dwt_denoise.py` | ✅ KEEP | Uses kernel_backend |
| `processors/emd_denoise.py` | ✅ KEEP | Wrap in CPU worker |
| `processors/fk_filter.py` | ✅ KEEP | Wrap in Ray worker |
| `processors/fkk_coordinate_filter.py` | ✅ KEEP | |
| `processors/fkk_filter_gpu.py` | ✅ KEEP | |
| `processors/gabor_denoise.py` | ✅ KEEP | Wrap in CPU worker |
| `processors/gain_processor.py` | ✅ KEEP | Wrap in Ray worker |
| `processors/kernel_backend.py` | 🔧 MODIFY | Add Ray backend option |
| `processors/mute_processor.py` | ✅ KEEP | Wrap in Ray worker |
| `processors/nmo_processor.py` | ✅ KEEP | Wrap in Numba worker |
| `processors/omp_denoise.py` | ✅ KEEP | Wrap in CPU worker |
| `processors/qc_batch_engine.py` | 🔧 MODIFY | Use Ray coordinator |
| `processors/qc_stacking_engine.py` | 🔧 MODIFY | Use Ray coordinator |
| `processors/spectral_analyzer.py` | ✅ KEEP | |
| `processors/sst_denoise.py` | ✅ KEEP | Wrap in CPU worker |
| `processors/stft_denoise.py` | ✅ KEEP | Uses kernel_backend |
| `processors/stockwell_denoise.py` | ✅ KEEP | Wrap in CPU worker |
| `processors/tf_denoise.py` | ✅ KEEP | Wrap in CPU worker |
| `processors/tf_denoise_gpu.py` | ✅ KEEP | |

---

### Processors - GPU

| File | Action | Notes |
|------|--------|-------|
| `processors/gpu/__init__.py` | ✅ KEEP | |
| `processors/gpu/device_manager.py` | ✅ KEEP | |
| `processors/gpu/stft_gpu.py` | ✅ KEEP | |
| `processors/gpu/stransform_gpu.py` | ✅ KEEP | |
| `processors/gpu/thresholding_gpu.py` | ✅ KEEP | |
| `processors/gpu/utils_gpu.py` | ✅ KEEP | |

---

### Processors - Migration

| File | Action | Notes |
|------|--------|-------|
| `processors/migration/__init__.py` | ✅ KEEP | |
| `processors/migration/antialias.py` | ✅ KEEP | |
| `processors/migration/aperture.py` | ✅ KEEP | |
| `processors/migration/aperture_adaptive.py` | ✅ KEEP | |
| `processors/migration/base_migrator.py` | ✅ KEEP | |
| `processors/migration/checkpoint.py` | ✅ KEEP | |
| `processors/migration/config_adapter.py` | ✅ KEEP | |
| `processors/migration/geometry_preprocessor.py` | ✅ KEEP | |
| `processors/migration/gpu_memory.py` | ✅ KEEP | |
| `processors/migration/interpolation.py` | ✅ KEEP | |
| `processors/migration/kirchhoff_kernel.py` | ✅ KEEP | |
| `processors/migration/kirchhoff_migrator.py` | ✅ KEEP | |
| `processors/migration/migration_engine.py` | 🔧 MODIFY | Add Ray integration |
| `processors/migration/optimized_kirchhoff_migrator.py` | ✅ KEEP | |
| `processors/migration/trace_index.py` | ✅ KEEP | |
| `processors/migration/traveltime.py` | ✅ KEEP | |
| `processors/migration/traveltime_cache.py` | ✅ KEEP | |
| `processors/migration/traveltime_curved.py` | ✅ KEEP | |
| `processors/migration/traveltime_lut.py` | ✅ KEEP | |
| `processors/migration/traveltime_vti.py` | ✅ KEEP | |
| `processors/migration/velocity_model.py` | ✅ KEEP | |
| `processors/migration/weights.py` | ✅ KEEP | |

---

### Utils - Parallel Processing

| File | Action | Notes |
|------|--------|-------|
| `utils/parallel_processing/__init__.py` | ⚠️ DEPRECATE | Keep for fallback |
| `utils/parallel_processing/config.py` | ⚠️ DEPRECATE | Keep for fallback |
| `utils/parallel_processing/coordinator.py` | 🔧 MODIFY | Add Ray backend option |
| `utils/parallel_processing/partitioner.py` | ✅ KEEP | Reuse in Ray |
| `utils/parallel_processing/shared_data.py` | ⚠️ DEPRECATE | Ray handles this |
| `utils/parallel_processing/worker.py` | ⚠️ DEPRECATE | Replace with Ray workers |

---

### Utils - Parallel Export

| File | Action | Notes |
|------|--------|-------|
| `utils/parallel_export/__init__.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/parallel_export/config.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/parallel_export/coordinator.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/parallel_export/header_vectorizer.py` | 🔧 MODIFY | Move to Rust |
| `utils/parallel_export/merger.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/parallel_export/worker.py` | ⚠️ DEPRECATE | Replace with Rust |

---

### Utils - SEGY Import

| File | Action | Notes |
|------|--------|-------|
| `utils/segy_import/__init__.py` | 🔧 MODIFY | Add Rust backend option |
| `utils/segy_import/computed_headers.py` | ✅ KEEP | |
| `utils/segy_import/data_storage.py` | ✅ KEEP | |
| `utils/segy_import/header_mapping.py` | ✅ KEEP | |
| `utils/segy_import/segy_export.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/segy_import/segy_reader.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/segy_import/segy_reader_fast.py` | ⚠️ DEPRECATE | Replace with Rust |
| `utils/segy_import/multiprocess_import/__init__.py` | ⚠️ DEPRECATE | Replace with Ray+Rust |
| `utils/segy_import/multiprocess_import/coordinator.py` | ⚠️ DEPRECATE | Replace with Ray+Rust |
| `utils/segy_import/multiprocess_import/partitioner.py` | ✅ KEEP | Reuse logic |
| `utils/segy_import/multiprocess_import/worker.py` | ⚠️ DEPRECATE | Replace with Rust worker |

---

### Utils - Other

| File | Action | Notes |
|------|--------|-------|
| `utils/__init__.py` | ✅ KEEP | |
| `utils/anisotropy_io.py` | ✅ KEEP | |
| `utils/binning_presets.py` | ✅ KEEP | |
| `utils/dataset_indexer.py` | ✅ KEEP | |
| `utils/fix_metadata_segy_path.py` | ✅ KEEP | |
| `utils/geometry_utils.py` | ✅ KEEP | |
| `utils/header_calculator.py` | ✅ KEEP | |
| `utils/memory_monitor.py` | 🔧 MODIFY | Integrate with alerts |
| `utils/memory_profiler_diagnostic.py` | ✅ KEEP | |
| `utils/migration_output_storage.py` | ✅ KEEP | |
| `utils/parquet_io.py` | ✅ KEEP | |
| `utils/pptx_export.py` | ✅ KEEP | |
| `utils/sample_data.py` | ✅ KEEP | |
| `utils/sort_detector.py` | ✅ KEEP | |
| `utils/storage_manager.py` | ✅ KEEP | |
| `utils/subgather_detector.py` | ✅ KEEP | |
| `utils/theme_manager.py` | ✅ KEEP | |
| `utils/trace_sorter.py` | ✅ KEEP | |
| `utils/trace_spacing.py` | ✅ KEEP | |
| `utils/unit_conversion.py` | ✅ KEEP | |
| `utils/update_dataset_paths.py` | ✅ KEEP | |
| `utils/velocity_io.py` | ✅ KEEP | |
| `utils/window_cache.py` | ✅ KEEP | |

---

### Utils - Ray Orchestration (NEW)

| File | Action | Notes |
|------|--------|-------|
| `utils/ray_orchestration/__init__.py` | 🆕 CREATE | Module init |
| `utils/ray_orchestration/cluster.py` | 🆕 CREATE | Ray cluster management |
| `utils/ray_orchestration/config.py` | 🆕 CREATE | Configuration |
| `utils/ray_orchestration/job_queue.py` | 🆕 CREATE | Priority queue |
| `utils/ray_orchestration/scheduler.py` | 🆕 CREATE | Job scheduler |
| `utils/ray_orchestration/state_machine.py` | 🆕 CREATE | State transitions |
| `utils/ray_orchestration/cancellation.py` | 🆕 CREATE | Cancellation manager |
| `utils/ray_orchestration/checkpoint.py` | 🆕 CREATE | Checkpoint manager |
| `utils/ray_orchestration/pause_resume.py` | 🆕 CREATE | Pause/resume manager |
| `utils/ray_orchestration/qt_bridge.py` | 🆕 CREATE | Qt signal bridge |
| `utils/ray_orchestration/processing_coordinator.py` | 🆕 CREATE | Processing coordinator |
| `utils/ray_orchestration/dependency_graph.py` | 🆕 CREATE | Job dependencies |
| `utils/ray_orchestration/resource_manager.py` | 🆕 CREATE | Resource tracking |
| `utils/ray_orchestration/workers/__init__.py` | 🆕 CREATE | Workers module |
| `utils/ray_orchestration/workers/base_worker.py` | 🆕 CREATE | Base worker actor |
| `utils/ray_orchestration/workers/metal_worker.py` | 🆕 CREATE | Metal GPU worker |
| `utils/ray_orchestration/workers/cpu_worker.py` | 🆕 CREATE | CPU worker |
| `utils/ray_orchestration/workers/numba_worker.py` | 🆕 CREATE | Numba worker |
| `utils/ray_orchestration/workers/rust_segy_worker.py` | 🆕 CREATE | Rust SEGY worker |
| `utils/ray_orchestration/workers/processor_wrapper.py` | 🆕 CREATE | Processor adapter |
| `utils/ray_orchestration/workers/progress_reporter.py` | 🆕 CREATE | Progress reporting |
| `utils/ray_orchestration/history/__init__.py` | 🆕 CREATE | History module |
| `utils/ray_orchestration/history/storage.py` | 🆕 CREATE | History storage |
| `utils/ray_orchestration/history/queries.py` | 🆕 CREATE | Query interface |
| `utils/ray_orchestration/alerts/__init__.py` | 🆕 CREATE | Alerts module |
| `utils/ray_orchestration/alerts/manager.py` | 🆕 CREATE | Alert manager |
| `utils/ray_orchestration/alerts/rules.py` | 🆕 CREATE | Alert rules |
| `utils/ray_orchestration/alerts/notifications.py` | 🆕 CREATE | Notifications |

---

### Views

| File | Action | Notes |
|------|--------|-------|
| `views/__init__.py` | 🔧 MODIFY | Export new widgets |
| `views/control_panel.py` | ✅ KEEP | |
| `views/export_options_dialog.py` | ✅ KEEP | |
| `views/fk_designer_dialog.py` | ✅ KEEP | |
| `views/fkk_designer_dialog.py` | ✅ KEEP | |
| `views/flip_window.py` | ✅ KEEP | |
| `views/gather_navigation_panel.py` | ✅ KEEP | |
| `views/isa_window.py` | ✅ KEEP | |
| `views/migration_monitor_dialog.py` | 🔧 MODIFY | Connect to new signals |
| `views/processing_chain_widget.py` | ✅ KEEP | |
| `views/pstm_wizard_dialog.py` | ✅ KEEP | |
| `views/qc_batch_dialog.py` | 🔧 MODIFY | Use job manager |
| `views/qc_presentation_control_panel.py` | ✅ KEEP | |
| `views/qc_presentation_window.py` | ✅ KEEP | |
| `views/qc_stack_viewer.py` | ✅ KEEP | |
| `views/qc_stacking_dialog.py` | 🔧 MODIFY | Use job manager |
| `views/segy_import_dialog.py` | 🔧 MODIFY | Use Rust backend |
| `views/seismic_viewer.py` | ✅ KEEP | |
| `views/seismic_viewer_pyqtgraph.py` | ✅ KEEP | |
| `views/settings_dialog.py` | 🔧 MODIFY | Add job settings |
| `views/volume_header_dialog.py` | ✅ KEEP | |
| `views/job_dashboard.py` | 🆕 CREATE | Job dashboard |
| `views/resource_monitor_widget.py` | 🆕 CREATE | Resource monitor |
| `views/job_analytics_widget.py` | 🆕 CREATE | Analytics |
| `views/widgets/__init__.py` | 🔧 MODIFY | Export new widgets |
| `views/widgets/kernel_selector.py` | ✅ KEEP | |
| `views/widgets/job_card.py` | 🆕 CREATE | Job card widget |
| `views/widgets/progress_bar.py` | 🆕 CREATE | Enhanced progress bar |
| `views/widgets/job_queue_widget.py` | 🆕 CREATE | Queue widget |
| `views/widgets/toast_notification.py` | 🆕 CREATE | Toast notifications |

---

### Seismic Metal (C++/Objective-C++)

| File | Action | Notes |
|------|--------|-------|
| `seismic_metal/include/common_types.h` | ✅ KEEP | |
| `seismic_metal/include/device_manager.h` | ✅ KEEP | |
| `seismic_metal/include/dwt_kernel.h` | ✅ KEEP | |
| `seismic_metal/include/fkk_kernel.h` | ✅ KEEP | |
| `seismic_metal/include/stft_kernel.h` | ✅ KEEP | |
| `seismic_metal/include/vdsp_fft.h` | ✅ KEEP | |
| `seismic_metal/python/__init__.py` | ✅ KEEP | |
| `seismic_metal/shaders/dwt_decompose.metal` | ✅ KEEP | |
| `seismic_metal/shaders/dwt_reconstruct.metal` | ✅ KEEP | |
| `seismic_metal/shaders/fkk_mask.metal` | ✅ KEEP | |
| `seismic_metal/shaders/mad_threshold.metal` | ✅ KEEP | |
| `seismic_metal/shaders/stft_forward.metal` | ✅ KEEP | |
| `seismic_metal/shaders/stft_inverse.metal` | ✅ KEEP | |
| `seismic_metal/shaders/swt_reconstruct.metal` | ✅ KEEP | |
| `seismic_metal/src/bindings.cpp` | 🔧 MODIFY | Add mod_gil_not_used |
| `seismic_metal/src/device_manager.mm` | 🔧 MODIFY | Add thread safety |
| `seismic_metal/src/dwt_kernel.mm` | ✅ KEEP | |
| `seismic_metal/src/fkk_kernel.mm` | ✅ KEEP | |
| `seismic_metal/src/stft_kernel.mm` | ✅ KEEP | |
| `seismic_metal/src/vdsp_fft.mm` | ✅ KEEP | |
| `seismic_metal/benchmarks/benchmark_metal_vs_numba.py` | ✅ KEEP | |

---

### Rust SEGY Module (NEW)

| File | Action | Notes |
|------|--------|-------|
| `seisproc_rust/Cargo.toml` | 🆕 CREATE | Rust dependencies |
| `seisproc_rust/pyproject.toml` | 🆕 CREATE | Maturin config |
| `seisproc_rust/src/lib.rs` | 🆕 CREATE | PyO3 module |
| `seisproc_rust/src/segy/mod.rs` | 🆕 CREATE | SEGY module |
| `seisproc_rust/src/segy/reader.rs` | 🆕 CREATE | SEGY reader |
| `seisproc_rust/src/segy/writer.rs` | 🆕 CREATE | SEGY writer |
| `seisproc_rust/src/segy/header.rs` | 🆕 CREATE | Header parsing |
| `seisproc_rust/src/segy/formats.rs` | 🆕 CREATE | Data formats |
| `seisproc_rust/src/segy/merger.rs` | 🆕 CREATE | Segment merger |
| `seisproc_rust/src/parallel/mod.rs` | 🆕 CREATE | Parallel module |
| `seisproc_rust/src/parallel/partitioner.rs` | 🆕 CREATE | Smart partitioner |
| `seisproc_rust/src/parallel/progress.rs` | 🆕 CREATE | Progress reporting |
| `seisproc_rust/src/parallel/cancellation.rs` | 🆕 CREATE | Cancellation |
| `seisproc_rust/src/utils/mod.rs` | 🆕 CREATE | Utils module |
| `seisproc_rust/src/utils/sorting.rs` | 🆕 CREATE | Parallel sort |
| `seisproc_rust/src/utils/validation.rs` | 🆕 CREATE | Validation |

---

### Tests

| File | Action | Notes |
|------|--------|-------|
| `tests/conftest.py` | 🔧 MODIFY | Add Ray fixtures |
| `tests/test_ray_cluster.py` | 🆕 CREATE | Phase 1 |
| `tests/test_job_models.py` | 🆕 CREATE | Phase 1 |
| `tests/test_job_state_machine.py` | 🆕 CREATE | Phase 1 |
| `tests/test_job_queue.py` | 🆕 CREATE | Phase 1 |
| `tests/test_scheduler.py` | 🆕 CREATE | Phase 1 |
| `tests/test_cancellation.py` | 🆕 CREATE | Phase 1 |
| `tests/test_metal_threading.py` | 🆕 CREATE | Phase 1 |
| `tests/test_worker_progress.py` | 🆕 CREATE | Phase 1 |
| `tests/test_metal_worker.py` | 🆕 CREATE | Phase 1 |
| `tests/test_integration_phase1.py` | 🆕 CREATE | Phase 1 |
| `tests/test_job_dashboard.py` | 🆕 CREATE | Phase 2 |
| `tests/test_resource_monitor.py` | 🆕 CREATE | Phase 2 |
| `tests/test_qt_bridge.py` | 🆕 CREATE | Phase 2 |
| `tests/test_main_window_jobs.py` | 🆕 CREATE | Phase 2 |
| `tests/test_rust_segy.py` | 🆕 CREATE | Phase 2 |
| `tests/test_rust_import_worker.py` | 🆕 CREATE | Phase 2 |
| `tests/test_checkpoint.py` | 🆕 CREATE | Phase 2 |
| `tests/test_cpu_worker.py` | 🆕 CREATE | Phase 3 |
| `tests/test_numba_worker.py` | 🆕 CREATE | Phase 3 |
| `tests/test_processing_coordinator.py` | 🆕 CREATE | Phase 3 |
| `tests/test_migration_compatibility.py` | 🆕 CREATE | Phase 3 |
| `tests/test_rust_cancellation.py` | 🆕 CREATE | Phase 3 |
| `tests/test_metal_cancellation.py` | 🆕 CREATE | Phase 3 |
| `tests/test_full_pipeline.py` | 🆕 CREATE | Phase 3 |
| `tests/test_rust_export.py` | 🆕 CREATE | Phase 4 |
| `tests/test_job_history.py` | 🆕 CREATE | Phase 4 |
| `tests/test_analytics.py` | 🆕 CREATE | Phase 4 |
| `tests/test_alerts.py` | 🆕 CREATE | Phase 4 |
| `tests/test_stress.py` | 🆕 CREATE | Phase 4 |

---

### Documentation

| File | Action | Notes |
|------|--------|-------|
| `docs/unified_hybrid_architecture_proposal.md` | ✅ KEEP | This proposal |
| `docs/hybrid_implementation_plan.md` | 🆕 CREATE | This document |
| `docs/user_guide_job_management.md` | 🆕 CREATE | Phase 4 |
| `docs/migration_guide_ray.md` | 🆕 CREATE | Phase 4 |
| `docs/troubleshooting_jobs.md` | 🆕 CREATE | Phase 4 |

---

## Test Summary

| Phase | New Tests | Modified Tests | Total |
|-------|-----------|----------------|-------|
| Phase 1 | 25 | 0 | 25 |
| Phase 2 | 18 | 0 | 18 |
| Phase 3 | 15 | 2 | 17 |
| Phase 4 | 13 | 0 | 13 |
| **Total** | **71** | **2** | **73** |

---

## File Count Summary

| Category | Keep | Modify | Create | Deprecate | Delete |
|----------|------|--------|--------|-----------|--------|
| Models | 18 | 1 | 3 | 0 | 0 |
| Processors | 42 | 4 | 0 | 0 | 0 |
| Utils (existing) | 32 | 4 | 0 | 14 | 0 |
| Utils (Ray new) | 0 | 0 | 28 | 0 | 0 |
| Views | 17 | 6 | 5 | 0 | 0 |
| Metal | 17 | 2 | 0 | 0 | 0 |
| Rust (new) | 0 | 0 | 16 | 0 | 0 |
| Tests | ~80 | 1 | 30 | 0 | 0 |
| **Total** | **206** | **18** | **82** | **14** | **0** |

---

## Conclusion

This implementation plan provides:

1. **71 new tests** across 4 phases
2. **82 new files** to create
3. **18 files** to modify
4. **14 files** to deprecate (keep as fallback)
5. **0 files** to delete immediately

The hybrid approach preserves **206 existing files** while adding new orchestration infrastructure. All deprecated files remain available as fallbacks during the transition period.
