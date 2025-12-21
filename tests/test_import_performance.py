"""
Performance comparison test for SEGY import.

Compares:
1. Direct coordinator call (old code path)
2. SEGYImportWorker (new code path)

Usage:
    python tests/test_import_performance.py /path/to/test.sgy /path/to/output_dir
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.segy_import.multiprocess_import.coordinator import (
    ParallelImportCoordinator,
    ImportConfig,
    ImportProgress,
    get_optimal_workers,
)
from utils.segy_import.header_mapping import HeaderMapping


def test_direct_coordinator(segy_path: str, output_dir: str) -> float:
    """
    Test direct coordinator call (old code path).

    Returns elapsed time in seconds.
    """
    print("\n" + "="*60)
    print("TEST 1: Direct Coordinator Call (Old Code Path)")
    print("="*60)

    # Create config
    config = ImportConfig(
        segy_path=segy_path,
        output_dir=output_dir,
        header_mapping=HeaderMapping(),
        ensemble_key=None,
        n_workers=get_optimal_workers(),
        chunk_size=10000,
    )

    coordinator = ParallelImportCoordinator(config)

    progress_count = [0]
    last_print = [0]

    def on_progress(prog: ImportProgress):
        progress_count[0] += 1
        # Print every 5 seconds
        if prog.elapsed_time - last_print[0] >= 5:
            last_print[0] = prog.elapsed_time
            rate = prog.current_traces / prog.elapsed_time if prog.elapsed_time > 0 else 0
            print(f"  Progress: {prog.current_traces:,}/{prog.total_traces:,} traces "
                  f"({prog.current_traces/prog.total_traces*100:.1f}%) "
                  f"@ {rate:,.0f} traces/sec")

    print(f"\nStarting import with {config.n_workers} workers...")
    start_time = time.time()

    result = coordinator.run(progress_callback=on_progress)

    elapsed = time.time() - start_time

    if result.success:
        rate = result.n_traces / elapsed
        print(f"\n✓ SUCCESS: {result.n_traces:,} traces in {elapsed:.1f}s")
        print(f"  Rate: {rate:,.0f} traces/sec")
        print(f"  Progress callbacks: {progress_count[0]}")
    else:
        print(f"\n✗ FAILED: {result.error}")

    return elapsed


def test_worker_import(segy_path: str, output_dir: str) -> float:
    """
    Test SEGYImportWorker (new code path).

    Returns elapsed time in seconds.
    """
    print("\n" + "="*60)
    print("TEST 2: SEGYImportWorker (New Code Path)")
    print("="*60)

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QEventLoop, QTimer
    from utils.ray_orchestration.segy_workers import SEGYImportWorker, SEGYImportResult

    # Need QApplication for signals
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Create config
    config = ImportConfig(
        segy_path=segy_path,
        output_dir=output_dir,
        header_mapping=HeaderMapping(),
        ensemble_key=None,
        n_workers=get_optimal_workers(),
        chunk_size=10000,
    )

    progress_count = [0]
    last_print = [0.0]
    result_holder = [None]
    error_holder = [None]

    def on_progress(prog):
        progress_count[0] += 1
        elapsed = prog.elapsed_time if hasattr(prog, 'elapsed_time') else 0
        if elapsed - last_print[0] >= 5:
            last_print[0] = elapsed
            current = prog.current_traces if hasattr(prog, 'current_traces') else 0
            total = prog.total_traces if hasattr(prog, 'total_traces') else 1
            rate = current / elapsed if elapsed > 0 else 0
            print(f"  Progress: {current:,}/{total:,} traces "
                  f"({current/total*100:.1f}%) "
                  f"@ {rate:,.0f} traces/sec")

    def on_complete(result: SEGYImportResult):
        result_holder[0] = result

    def on_error(error: str):
        error_holder[0] = error

    worker = SEGYImportWorker(config, job_name="perf_test")
    worker.progress_updated.connect(on_progress)
    worker.finished_with_result.connect(on_complete)
    worker.error_occurred.connect(on_error)

    print(f"\nStarting worker with {config.n_workers} workers...")
    start_time = time.time()

    worker.start()

    # Wait for completion with event loop
    while worker.isRunning():
        app.processEvents()
        time.sleep(0.1)

    elapsed = time.time() - start_time

    if result_holder[0] and result_holder[0].success:
        result = result_holder[0]
        rate = result.n_traces / elapsed
        print(f"\n✓ SUCCESS: {result.n_traces:,} traces in {elapsed:.1f}s")
        print(f"  Rate: {rate:,.0f} traces/sec")
        print(f"  Progress callbacks: {progress_count[0]}")
    elif error_holder[0]:
        print(f"\n✗ FAILED: {error_holder[0]}")
    else:
        print(f"\n✗ FAILED: Unknown error")

    return elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_import_performance.py <segy_file> [output_base_dir]")
        print("\nThis test compares import performance between:")
        print("  1. Direct ParallelImportCoordinator.run() (old code)")
        print("  2. SEGYImportWorker (new code)")
        sys.exit(1)

    segy_path = sys.argv[1]
    base_output = sys.argv[2] if len(sys.argv) > 2 else tempfile.gettempdir()

    if not Path(segy_path).exists():
        print(f"Error: SEGY file not found: {segy_path}")
        sys.exit(1)

    # Get file info
    file_size_gb = Path(segy_path).stat().st_size / (1024**3)
    print(f"\nSEGY File: {segy_path}")
    print(f"File Size: {file_size_gb:.2f} GB")
    print(f"Workers: {get_optimal_workers()}")

    # Test 1: Direct coordinator
    output_dir1 = Path(base_output) / "perf_test_direct"
    if output_dir1.exists():
        shutil.rmtree(output_dir1)
    output_dir1.mkdir(parents=True)

    time1 = test_direct_coordinator(segy_path, str(output_dir1))

    # Cleanup
    shutil.rmtree(output_dir1)

    # Test 2: Worker
    output_dir2 = Path(base_output) / "perf_test_worker"
    if output_dir2.exists():
        shutil.rmtree(output_dir2)
    output_dir2.mkdir(parents=True)

    time2 = test_worker_import(segy_path, str(output_dir2))

    # Cleanup
    shutil.rmtree(output_dir2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Direct Coordinator: {time1:.1f}s")
    print(f"SEGYImportWorker:   {time2:.1f}s")
    print(f"Difference:         {time2 - time1:+.1f}s ({(time2/time1 - 1)*100:+.1f}%)")

    if abs(time2 - time1) < time1 * 0.1:
        print("\n✓ Performance is similar (within 10%)")
    elif time2 > time1 * 1.5:
        print("\n⚠ Worker is significantly slower - investigation needed")
    else:
        print("\n✓ Performance difference is acceptable")


if __name__ == "__main__":
    main()
