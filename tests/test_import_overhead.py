"""
Quick overhead test for SEGY import setup.

Measures the time taken for:
1. Direct coordinator setup
2. Worker + job management setup

This doesn't actually import data - just measures the overhead.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_direct_setup():
    """Time direct coordinator setup."""
    from utils.segy_import.multiprocess_import.coordinator import (
        ParallelImportCoordinator,
        ImportConfig,
    )
    from utils.segy_import.header_mapping import HeaderMapping

    start = time.time()

    config = ImportConfig(
        segy_path="/tmp/fake.sgy",  # Doesn't need to exist for setup
        output_dir="/tmp/fake_output",
        header_mapping=HeaderMapping(),
        ensemble_key=None,
        n_workers=8,
        chunk_size=10000,
    )

    coordinator = ParallelImportCoordinator(config)

    elapsed = time.time() - start
    print(f"Direct coordinator setup: {elapsed*1000:.2f}ms")
    return elapsed


def test_worker_setup():
    """Time worker + job management setup."""
    from utils.segy_import.multiprocess_import.coordinator import ImportConfig
    from utils.segy_import.header_mapping import HeaderMapping
    from utils.ray_orchestration.segy_workers import SEGYImportWorker
    from utils.ray_orchestration.job_manager import get_job_manager
    from utils.ray_orchestration.qt_bridge import get_job_bridge

    start = time.time()

    config = ImportConfig(
        segy_path="/tmp/fake.sgy",
        output_dir="/tmp/fake_output",
        header_mapping=HeaderMapping(),
        ensemble_key=None,
        n_workers=8,
        chunk_size=10000,
    )

    # These are the extra steps in the worker
    job_manager = get_job_manager()
    qt_bridge = get_job_bridge()

    worker = SEGYImportWorker(config, job_name="test")

    elapsed = time.time() - start
    print(f"Worker + job management setup: {elapsed*1000:.2f}ms")
    return elapsed


def test_signal_emission_overhead():
    """Time Qt signal emission overhead."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    class Emitter(QObject):
        signal = pyqtSignal(dict)

    emitter = Emitter()
    received = [0]

    def on_receive(data):
        received[0] += 1

    emitter.signal.connect(on_receive)

    # Time 1000 emissions
    start = time.time()
    for i in range(1000):
        emitter.signal.emit({'percent': i / 10, 'message': 'test', 'traces': i * 100})

    # Process events
    app.processEvents()

    elapsed = time.time() - start
    print(f"1000 signal emissions: {elapsed*1000:.2f}ms ({elapsed:.3f}ms per signal)")
    return elapsed


def main():
    print("="*60)
    print("SEGY Import Overhead Analysis")
    print("="*60)

    t1 = test_direct_setup()
    t2 = test_worker_setup()
    t3 = test_signal_emission_overhead()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Worker overhead vs direct: {(t2-t1)*1000:.2f}ms")
    print(f"Signal emission (per 0.5s @ 2Hz): {t3/1000*2*1000:.4f}ms")
    print("\nConclusion:")

    total_overhead = (t2 - t1) + (t3 / 1000 * 2)  # Per 0.5s update
    if total_overhead < 0.01:  # Less than 10ms overhead
        print("✓ Overhead is negligible - slowdown must be elsewhere")
    else:
        print(f"⚠ Overhead is {total_overhead*1000:.2f}ms - may contribute to slowdown")


if __name__ == "__main__":
    main()
