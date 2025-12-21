#!/usr/bin/env python3
"""
Debug script to trace job dashboard progress flow and output dataset creation.

Run with: python debug_job_dashboard.py
"""

import sys
import logging
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DEBUG_TEST")

def test_1_job_progress_models():
    """Test 1: Verify JobProgress and related models work correctly."""
    print("\n" + "="*60)
    print("TEST 1: Job Progress Models")
    print("="*60)

    try:
        from models.job_progress import JobProgress, WorkerProgress, ProgressUpdate
        from models.job import Job, JobType, JobState

        # Create a test job progress
        job_id = uuid4()
        progress = JobProgress(
            job_id=job_id,
            phase="processing",
            overall_percent=50.0,
            message="Test progress",
        )

        # Add a worker
        worker = WorkerProgress(
            worker_id="test-worker-1",
            items_total=1000,
            items_processed=500,
        )
        progress.add_worker(worker)

        # Add metrics
        progress.metrics = {
            'current_gathers': 100,
            'total_gathers': 200,
            'compute_kernel': 'Metal GPU',
            'traces_per_sec': 50000,
        }

        print(f"✓ JobProgress created: {progress.overall_percent}%")
        print(f"  - total_items_processed: {progress.total_items_processed}")
        print(f"  - total_items: {progress.total_items}")
        print(f"  - metrics: {progress.metrics}")
        print(f"  - active_workers: {progress.active_workers}")

        # Test ProgressUpdate
        update = ProgressUpdate(
            job_id=job_id,
            worker_id="coordinator",
            items_processed=500,
            items_total=1000,
            message="Phase: processing",
            metrics={
                'current_gathers': 100,
                'total_gathers': 200,
                'compute_kernel': 'Metal GPU',
            }
        )
        print(f"✓ ProgressUpdate created: {update.items_processed}/{update.items_total}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_job_manager_progress():
    """Test 2: Verify JobManager receives and stores progress correctly."""
    print("\n" + "="*60)
    print("TEST 2: JobManager Progress Storage")
    print("="*60)

    try:
        from utils.ray_orchestration.job_manager import JobManager, get_job_manager
        from models.job_progress import ProgressUpdate
        from models.job import JobType
        from models.job_config import JobConfig

        manager = get_job_manager()

        # Submit a test job
        job = manager.submit_job(
            name="Test Job",
            job_type=JobType.BATCH_PROCESS,
            config=JobConfig.for_batch_processing(trace_count=100000),
        )
        print(f"✓ Job submitted: {job.id}")

        # Start the job
        manager.start_job(job.id)
        print(f"✓ Job started")

        # Get initial progress
        progress_before = manager.get_progress(job.id)
        print(f"  Progress before update:")
        print(f"    - overall_percent: {progress_before.overall_percent}")
        print(f"    - total_items: {progress_before.total_items}")
        print(f"    - metrics: {progress_before.metrics}")

        # Send progress update
        update = ProgressUpdate(
            job_id=job.id,
            worker_id="coordinator",
            items_processed=5000,
            items_total=100000,
            message="Phase: processing",
            metrics={
                'current_gathers': 50,
                'total_gathers': 1000,
                'compute_kernel': 'Metal GPU',
                'traces_per_sec': 25000,
                'active_workers': 9,
            }
        )
        manager.update_progress(update)
        print(f"✓ Progress update sent")

        # Get updated progress
        progress_after = manager.get_progress(job.id)
        print(f"  Progress after update:")
        print(f"    - overall_percent: {progress_after.overall_percent}")
        print(f"    - total_items_processed: {progress_after.total_items_processed}")
        print(f"    - total_items: {progress_after.total_items}")
        print(f"    - metrics: {progress_after.metrics}")
        print(f"    - active_workers: {progress_after.active_workers}")

        # Verify metrics were stored
        if progress_after.metrics.get('compute_kernel') == 'Metal GPU':
            print(f"✓ Metrics stored correctly at progress level")
        else:
            print(f"✗ Metrics NOT stored at progress level!")
            print(f"    Expected 'Metal GPU', got: {progress_after.metrics.get('compute_kernel')}")
            return False

        # Cancel the test job
        manager.cancel_job(job.id)

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_qt_bridge_emit():
    """Test 3: Verify Qt bridge emits correct progress dict."""
    print("\n" + "="*60)
    print("TEST 3: Qt Bridge Progress Emission")
    print("="*60)

    try:
        from utils.ray_orchestration.qt_bridge import JobSignalEmitter
        from models.job_progress import JobProgress, WorkerProgress
        from uuid import uuid4

        # Create emitter
        emitter = JobSignalEmitter()

        # Track what gets emitted
        emitted_data = []
        def capture_emit(job_id, progress_dict):
            emitted_data.append((job_id, progress_dict))
            print(f"  Signal emitted: job_id={job_id}")
            for k, v in progress_dict.items():
                print(f"    - {k}: {v}")

        emitter.job_progress.connect(capture_emit)

        # Create test progress with all fields
        job_id = uuid4()
        progress = JobProgress(
            job_id=job_id,
            phase="processing",
            overall_percent=25.0,
            message="Test message",
        )

        # Add worker
        worker = WorkerProgress(
            worker_id="coordinator",
            items_total=22000000,
            items_processed=5500000,
        )
        progress.add_worker(worker)

        # Add metrics
        progress.metrics = {
            'current_gathers': 1700,
            'total_gathers': 7000,
            'compute_kernel': 'Metal GPU',
            'traces_per_sec': 150000,
            'active_workers': 9,
        }

        # Emit progress
        emitter.emit_progress(progress)

        if len(emitted_data) == 0:
            print(f"✗ No signal emitted!")
            return False

        _, emitted = emitted_data[0]

        # Verify all fields
        expected_fields = [
            'percent', 'message', 'phase', 'eta_seconds',
            'current_gathers', 'total_gathers',
            'current_traces', 'total_traces',
            'traces_per_sec', 'compute_kernel', 'active_workers'
        ]

        missing = [f for f in expected_fields if f not in emitted]
        if missing:
            print(f"✗ Missing fields in emitted data: {missing}")
            return False

        print(f"✓ All expected fields present")

        # Verify values
        if emitted.get('current_traces') != 5500000:
            print(f"✗ current_traces wrong: expected 5500000, got {emitted.get('current_traces')}")
            return False
        if emitted.get('total_traces') != 22000000:
            print(f"✗ total_traces wrong: expected 22000000, got {emitted.get('total_traces')}")
            return False
        if emitted.get('compute_kernel') != 'Metal GPU':
            print(f"✗ compute_kernel wrong: expected 'Metal GPU', got {emitted.get('compute_kernel')}")
            return False

        print(f"✓ All values correct")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_processing_progress_dataclass():
    """Test 4: Verify ProcessingProgress has new fields."""
    print("\n" + "="*60)
    print("TEST 4: ProcessingProgress Dataclass")
    print("="*60)

    try:
        from utils.parallel_processing import ProcessingProgress

        progress = ProcessingProgress(
            phase="processing",
            current_traces=5000000,
            total_traces=22000000,
            current_gathers=1500,
            total_gathers=7000,
            active_workers=9,
            elapsed_time=30.5,
            eta_seconds=120.0,
            traces_per_sec=150000.0,
            compute_kernel="Metal GPU",
        )

        print(f"✓ ProcessingProgress created with new fields:")
        print(f"  - traces_per_sec: {progress.traces_per_sec}")
        print(f"  - compute_kernel: {progress.compute_kernel}")

        return True

    except TypeError as e:
        print(f"✗ FAILED - Missing fields in ProcessingProgress: {e}")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_output_dataset_creation():
    """Test 5: Check output dataset structure and headers."""
    print("\n" + "="*60)
    print("TEST 5: Output Dataset Creation")
    print("="*60)

    import os

    # Find recent processing output directories
    processing_dir = Path(os.path.expanduser("~/SeismicData/processing"))
    if not processing_dir.exists():
        print(f"✗ Processing directory not found: {processing_dir}")
        return False

    # List recent outputs
    outputs = sorted(processing_dir.glob("processed_*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not outputs:
        print(f"✗ No processing outputs found in {processing_dir}")
        return False

    print(f"Found {len(outputs)} output directories")

    # Check most recent
    latest = outputs[0]
    print(f"\nChecking latest output: {latest.name}")

    # Check expected files
    expected_files = [
        "output/traces.zarr",
        "output/metadata.json",
        "output/headers.parquet",
        "output/ensemble_index.parquet",
    ]

    for expected in expected_files:
        path = latest / expected
        if path.exists():
            if path.is_dir():
                # Zarr directory
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                print(f"  ✓ {expected} exists ({size / 1024 / 1024:.1f} MB)")
            else:
                size = path.stat().st_size
                print(f"  ✓ {expected} exists ({size / 1024:.1f} KB)")
        else:
            print(f"  ✗ {expected} MISSING!")

    # Check output subdirectory
    output_dir = latest / "output"
    if output_dir.exists():
        print(f"\nContents of {output_dir}:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                print(f"    {item.name}/ ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f"    {item.name} ({item.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  ✗ output/ directory MISSING!")

    # Check if headers exist at root level
    root_headers = latest / "headers.parquet"
    if root_headers.exists():
        print(f"\n  Note: headers.parquet exists at root level")

    return True


def test_6_ray_coordinator_progress_callback():
    """Test 6: Simulate coordinator progress callback flow."""
    print("\n" + "="*60)
    print("TEST 6: Ray Coordinator Progress Callback Flow")
    print("="*60)

    try:
        from utils.parallel_processing import ProcessingProgress

        # Simulate what the coordinator sends
        progress = ProcessingProgress(
            phase="processing",
            current_traces=0,
            total_traces=22324543,
            current_gathers=0,
            total_gathers=7074,
            active_workers=9,
            elapsed_time=0.5,
            eta_seconds=0,
            traces_per_sec=0,
            compute_kernel="Metal GPU",
        )

        print(f"Simulated initial progress from coordinator:")
        print(f"  - current_traces: {progress.current_traces}")
        print(f"  - total_traces: {progress.total_traces}")
        print(f"  - active_workers: {progress.active_workers}")
        print(f"  - compute_kernel: {progress.compute_kernel}")

        # Simulate what wrapped_progress would emit
        percent = (progress.current_traces / progress.total_traces * 100
                   if progress.total_traces > 0 else 0)

        traces_per_sec = getattr(progress, 'traces_per_sec', 0)
        compute_kernel = getattr(progress, 'compute_kernel', '')

        emit_dict = {
            'percent': percent,
            'message': f"Phase: {progress.phase}",
            'phase': progress.phase,
            'eta_seconds': progress.eta_seconds,
            'active_workers': progress.active_workers,
            'current_gathers': progress.current_gathers,
            'total_gathers': progress.total_gathers,
            'current_traces': progress.current_traces,
            'total_traces': progress.total_traces,
            'traces_per_sec': traces_per_sec,
            'compute_kernel': compute_kernel,
        }

        print(f"\nEmit dict that would be sent:")
        for k, v in emit_dict.items():
            print(f"  - {k}: {v}")

        # Check critical fields
        issues = []
        if emit_dict['total_traces'] == 0:
            issues.append("total_traces is 0!")
        if emit_dict['total_gathers'] == 0:
            issues.append("total_gathers is 0!")
        if not emit_dict['compute_kernel']:
            issues.append("compute_kernel is empty!")

        if issues:
            print(f"\n✗ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print(f"\n✓ Progress dict looks correct")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_signal_connection_check():
    """Test 7: Check if signals are properly connected."""
    print("\n" + "="*60)
    print("TEST 7: Signal Connection Check")
    print("="*60)

    try:
        from utils.ray_orchestration.qt_bridge import get_job_bridge

        bridge = get_job_bridge()

        print(f"Bridge instance: {bridge}")
        print(f"Bridge signals: {bridge.signals}")

        # Check signal receivers
        job_progress_signal = bridge.signals.job_progress

        # Get receiver count (PyQt6 doesn't expose this directly, but we can test)
        print(f"\nTesting signal emission...")

        received = []
        def test_receiver(job_id, progress):
            received.append((job_id, progress))

        # Connect test receiver
        job_progress_signal.connect(test_receiver)

        # Emit test signal
        from uuid import uuid4
        test_id = uuid4()
        test_progress = {'test': True, 'percent': 50}
        job_progress_signal.emit(test_id, test_progress)

        # Check if received
        if received:
            print(f"✓ Signal received: {received[0]}")
        else:
            print(f"✗ Signal NOT received!")
            return False

        # Disconnect test receiver
        job_progress_signal.disconnect(test_receiver)

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all debug tests."""
    print("="*60)
    print("JOB DASHBOARD DEBUG TESTS")
    print("="*60)

    results = {}

    # Run tests
    results['1_job_progress_models'] = test_1_job_progress_models()
    results['2_job_manager_progress'] = test_2_job_manager_progress()
    results['3_qt_bridge_emit'] = test_3_qt_bridge_emit()
    results['4_processing_progress'] = test_4_processing_progress_dataclass()
    results['5_output_dataset'] = test_5_output_dataset_creation()
    results['6_coordinator_callback'] = test_6_ray_coordinator_progress_callback()
    results['7_signal_connection'] = test_7_signal_connection_check()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n⚠️  Some tests failed - review output above for details")
        return 1
    else:
        print("\n✓ All tests passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
