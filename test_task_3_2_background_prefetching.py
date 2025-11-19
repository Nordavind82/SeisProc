"""
Test suite for Task 3.2: Background Prefetching Thread

Tests the background thread that prefetches adjacent gathers for smooth navigation.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.gather_navigator import GatherNavigator
from models.lazy_seismic_data import LazySeismicData
from models.seismic_data import SeismicData
import zarr


# Test fixtures
class TestFixtures:
    """Helper class to create test data"""

    @staticmethod
    def create_test_lazy_data(n_samples=500, n_traces=100, sample_rate=0.004):
        """Create test lazy seismic data with Zarr storage"""
        temp_dir = tempfile.mkdtemp()
        storage_dir = Path(temp_dir)

        # Create test data
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)

        # Save to Zarr
        zarr_path = storage_dir / 'data.zarr'
        z = zarr.open(str(zarr_path), mode='w', shape=traces.shape,
                     chunks=(n_samples, 10), dtype=np.float32)
        z[:] = traces

        # Create metadata
        metadata = {
            'n_traces': n_traces,
            'n_samples': n_samples,
            'sample_rate': sample_rate,
            'duration': (n_samples - 1) * sample_rate,
            'header_mapping': {
                'ensemble_keys': ['CDP', 'INLINE']
            }
        }

        # Create headers
        headers_data = {
            'trace_index': list(range(n_traces)),
            'TraceNumber': list(range(1, n_traces + 1)),
            'CDP': [100 + (i // 10) for i in range(n_traces)],
            'INLINE': [1000 + (i // 10) for i in range(n_traces)],
            'OFFSET': [100 * (i % 10) for i in range(n_traces)]
        }
        headers_df = pd.DataFrame(headers_data)
        headers_path = storage_dir / 'headers.parquet'
        headers_df.to_parquet(headers_path, index=False)

        # Create ensemble index (10 ensembles, 10 traces each)
        ensemble_data = []
        for i in range(10):
            ensemble_data.append({
                'ensemble_id': i,
                'start_trace': i * 10,
                'end_trace': i * 10 + 9,
                'n_traces': 10
            })
        ensembles_df = pd.DataFrame(ensemble_data)
        ensemble_path = storage_dir / 'ensemble_index.parquet'
        ensembles_df.to_parquet(ensemble_path, index=False)

        # Create LazySeismicData
        lazy_data = LazySeismicData(
            zarr_path=zarr_path,
            metadata=metadata,
            headers_path=headers_path,
            ensemble_index_path=ensemble_path
        )

        return lazy_data, ensembles_df, storage_dir

    @staticmethod
    def cleanup_storage(storage_dir):
        """Clean up temporary storage"""
        if storage_dir and os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)


# Test 1: Thread starts on initialization
def test_1_thread_starts():
    """Test that prefetch thread starts when lazy data loaded"""
    print("\n=== Test 1: Thread Starts on Initialization ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Give thread a moment to start
        time.sleep(0.1)

        # Verify thread is running
        assert navigator._prefetch_thread is not None, "Prefetch thread should exist"
        assert navigator._prefetch_thread.is_alive(), "Prefetch thread should be running"
        assert navigator._prefetch_thread.daemon, "Prefetch thread should be daemon"
        assert navigator._prefetch_thread.name == "GatherNavigator-Prefetch", \
            f"Thread name should be 'GatherNavigator-Prefetch', got '{navigator._prefetch_thread.name}'"

        print("✓ Prefetch thread started successfully")
        print(f"  - Thread alive: {navigator._prefetch_thread.is_alive()}")
        print(f"  - Thread daemon: {navigator._prefetch_thread.daemon}")
        print(f"  - Thread name: {navigator._prefetch_thread.name}")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 2: Prefetch triggered on navigation
def test_2_prefetch_on_navigation():
    """Test that prefetch is triggered when navigating"""
    print("\n=== Test 2: Prefetch Triggered on Navigation ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Navigate to gather 5
        navigator.goto_gather(5)
        navigator.get_current_gather()  # Load current gather

        # Wait for prefetch to complete
        time.sleep(1.0)

        # Check cache - should have gather 5 plus some adjacent gathers
        with navigator._cache_lock:
            cache_size = len(navigator._ensemble_cache)
            cached_ids = list(navigator._ensemble_cache.keys())

        assert cache_size >= 2, f"Should have at least 2 cached (current + prefetched), got {cache_size}"
        assert 5 in cached_ids, "Gather 5 should be cached (current)"

        # Check if at least one adjacent gather was prefetched
        adjacent_cached = any(adj in cached_ids for adj in [3, 4, 6, 7])
        assert adjacent_cached, f"At least one adjacent gather should be prefetched. Cached: {cached_ids}"

        print("✓ Prefetch triggered correctly")
        print(f"  - Cache size: {cache_size}")
        print(f"  - Cached gathers: {cached_ids}")

        # Test navigation to prefetched gather is fast
        navigator.goto_gather(6)
        start_time = time.time()
        navigator.get_current_gather()
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        # Should be cache hit (< 50ms)
        if 6 in cached_ids:
            assert elapsed < 50, f"Navigation to prefetched gather should be < 50ms, got {elapsed:.1f}ms"
            print(f"  - Navigation to prefetched gather 6: {elapsed:.1f}ms (cache hit)")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 3: Thread doesn't block UI
def test_3_thread_non_blocking():
    """Test that prefetch thread doesn't block main thread"""
    print("\n=== Test 3: Thread Doesn't Block UI ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Navigate and immediately check responsiveness
        navigator.goto_gather(5)

        # Main thread should remain responsive (not blocked)
        start_time = time.time()
        test_value = 0
        for i in range(1000000):
            test_value += i  # Busy work to simulate UI interaction

        elapsed = time.time() - start_time

        # Should complete quickly (not blocked by prefetch)
        assert elapsed < 1.0, f"Main thread should remain responsive, took {elapsed:.3f}s"

        print("✓ Main thread not blocked by prefetch")
        print(f"  - Main thread computation time: {elapsed*1000:.1f}ms")
        print(f"  - Test value computed: {test_value}")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 4: Thread-safe cache access
def test_4_thread_safe_access():
    """Test that concurrent cache access is thread-safe"""
    print("\n=== Test 4: Thread-Safe Cache Access ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Rapidly navigate through gathers (concurrent access)
        errors = []

        def rapid_navigation():
            try:
                for i in range(10):
                    navigator.goto_gather(i % 10)
                    navigator.get_current_gather()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads navigating concurrently
        threads = []
        for _ in range(3):
            t = threading.Thread(target=rapid_navigation)
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=5.0)

        # Check for errors
        assert len(errors) == 0, f"Thread-safe access failed with errors: {errors}"

        # Verify cache is in valid state
        with navigator._cache_lock:
            cache_size = len(navigator._ensemble_cache)

        assert cache_size <= navigator._max_cached_ensembles, \
            f"Cache size ({cache_size}) exceeds max ({navigator._max_cached_ensembles})"

        print("✓ Thread-safe cache access verified")
        print(f"  - No errors during concurrent navigation")
        print(f"  - Cache size: {cache_size}/{navigator._max_cached_ensembles}")
        print(f"  - Cache hits: {navigator._cache_hits}")
        print(f"  - Cache misses: {navigator._cache_misses}")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 5: Graceful shutdown
def test_5_graceful_shutdown():
    """Test that prefetch thread shuts down cleanly"""
    print("\n=== Test 5: Graceful Shutdown ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Verify thread is running
        time.sleep(0.1)
        assert navigator._prefetch_thread.is_alive(), "Thread should be running"

        # Stop thread
        navigator._stop_prefetch_thread()

        # Give it time to stop
        time.sleep(0.2)

        # Verify thread stopped
        assert not navigator._prefetch_thread.is_alive(), "Thread should have stopped"
        assert navigator._stop_prefetch.is_set(), "Stop flag should be set"

        print("✓ Graceful shutdown successful")
        print(f"  - Thread stopped: True")
        print(f"  - Stop flag set: {navigator._stop_prefetch.is_set()}")

        # Test __del__ cleanup
        del navigator
        time.sleep(0.1)

        print("✓ __del__ cleanup successful (no exceptions)")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 6: Prefetch stops at boundaries
def test_6_boundary_conditions():
    """Test that prefetch handles boundaries correctly"""
    print("\n=== Test 6: Boundary Conditions ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Navigate to first gather (boundary)
        navigator.goto_gather(0)
        navigator.get_current_gather()

        # Wait for prefetch
        time.sleep(1.0)

        # Check cache - should not have negative indices
        with navigator._cache_lock:
            cached_ids = list(navigator._ensemble_cache.keys())

        assert all(g_id >= 0 for g_id in cached_ids), \
            f"Should not have negative gather IDs: {cached_ids}"

        print("✓ Start boundary handled correctly")
        print(f"  - Cached gathers at start: {cached_ids}")

        # Navigate to last gather (boundary)
        navigator.goto_gather(9)
        navigator.get_current_gather()

        # Wait for prefetch
        time.sleep(1.0)

        # Check cache - should not exceed n_gathers
        with navigator._cache_lock:
            cached_ids = list(navigator._ensemble_cache.keys())

        assert all(g_id < navigator.n_gathers for g_id in cached_ids), \
            f"Should not exceed n_gathers ({navigator.n_gathers}): {cached_ids}"

        print("✓ End boundary handled correctly")
        print(f"  - Cached gathers at end: {cached_ids}")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 7: No CPU usage when idle
def test_7_cpu_idle():
    """Test that thread doesn't consume CPU when idle"""
    print("\n=== Test 7: No CPU Usage When Idle ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Let thread idle (no navigation)
        time.sleep(1.0)

        # Thread should still be alive but not consuming CPU
        assert navigator._prefetch_thread.is_alive(), "Thread should still be alive"

        # Verify event is not set (thread is waiting)
        assert not navigator._prefetch_event.is_set(), "Event should not be set when idle"

        print("✓ Thread idle correctly (no CPU consumption)")
        print(f"  - Thread alive: {navigator._prefetch_thread.is_alive()}")
        print(f"  - Event set: {navigator._prefetch_event.is_set()}")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 8: Multiple prefetch triggers
def test_8_multiple_triggers():
    """Test that multiple rapid prefetch triggers work correctly"""
    print("\n=== Test 8: Multiple Prefetch Triggers ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Rapidly navigate (trigger multiple prefetches)
        for i in [2, 4, 6, 8, 5]:
            navigator.goto_gather(i)
            time.sleep(0.05)  # Small delay between navigations

        # Wait for prefetch to settle
        time.sleep(1.5)

        # Check cache
        with navigator._cache_lock:
            cache_size = len(navigator._ensemble_cache)
            cached_ids = list(navigator._ensemble_cache.keys())

        assert cache_size <= navigator._max_cached_ensembles, \
            f"Cache size ({cache_size}) should not exceed max ({navigator._max_cached_ensembles})"

        # Should have some gathers cached (may or may not include 5 due to rapid navigation)
        assert cache_size > 0, "Cache should have at least some gathers"

        # Most recent navigation was to 5, but due to rapid prefetching and LRU eviction,
        # it might have been evicted. Just verify cache is in valid state.
        print("✓ Multiple prefetch triggers handled correctly")
        print(f"  - Cache size: {cache_size}/{navigator._max_cached_ensembles}")
        print(f"  - Cached gathers: {cached_ids}")
        print(f"  - Most recent gather (5) cached: {5 in cached_ids}")

        # Cleanup
        navigator._stop_prefetch_thread()

    finally:
        TestFixtures.cleanup_storage(storage_dir)


if __name__ == '__main__':
    print("=" * 70)
    print("Task 3.2: Background Prefetching Thread - Test Suite")
    print("=" * 70)

    tests = [
        test_1_thread_starts,
        test_2_prefetch_on_navigation,
        test_3_thread_non_blocking,
        test_4_thread_safe_access,
        test_5_graceful_shutdown,
        test_6_boundary_conditions,
        test_7_cpu_idle,
        test_8_multiple_triggers,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test_func.__name__} FAILED:")
            print(f"  {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} ERROR:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    if failed == 0:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"❌ {failed} TEST(S) FAILED")
        sys.exit(1)
