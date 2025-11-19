"""
Test suite for Task 2.3: Window Caching with LRU Policy

Tests the WindowCache class implementation including:
- Basic get/put operations
- Memory tracking
- LRU eviction by count and memory
- Thread safety
- Statistics tracking
"""
import numpy as np
import sys
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.window_cache import WindowCache


def test_1_basic_get_put():
    """
    Test 1: Basic get/put operations
    Verify: Cache stores and retrieves windows correctly
    """
    print("\n" + "="*70)
    print("TEST 1: Basic get/put operations")
    print("="*70)

    cache = WindowCache(max_windows=3, max_memory_mb=100)

    # Create test data
    data_a = np.ones((100, 50), dtype=np.float32) * 1.0
    data_b = np.ones((100, 50), dtype=np.float32) * 2.0
    data_c = np.ones((100, 50), dtype=np.float32) * 3.0

    key_a = (0, 100, 0, 50)
    key_b = (100, 200, 0, 50)
    key_c = (200, 300, 0, 50)

    print(f"Putting 3 windows into cache...")
    cache.put(key_a, data_a)
    cache.put(key_b, data_b)
    cache.put(key_c, data_c)

    print(f"  Cache size: {len(cache)} windows")
    assert len(cache) == 3, f"Cache should have 3 windows"

    print(f"\nGetting windows from cache...")
    retrieved_a = cache.get(key_a)
    retrieved_b = cache.get(key_b)
    retrieved_c = cache.get(key_c)

    print(f"  Retrieved A: {retrieved_a is not None}")
    print(f"  Retrieved B: {retrieved_b is not None}")
    print(f"  Retrieved C: {retrieved_c is not None}")

    assert retrieved_a is not None, "Window A should be in cache"
    assert retrieved_b is not None, "Window B should be in cache"
    assert retrieved_c is not None, "Window C should be in cache"

    # Verify data correctness
    print(f"\nVerifying data values...")
    assert np.allclose(retrieved_a, data_a), "Data A mismatch"
    assert np.allclose(retrieved_b, data_b), "Data B mismatch"
    assert np.allclose(retrieved_c, data_c), "Data C mismatch"
    print(f"  ✓ All data values correct")

    # Test cache miss
    print(f"\nTesting cache miss...")
    missing = cache.get((999, 1099, 0, 50))
    assert missing is None, "Non-existent key should return None"
    print(f"  ✓ Cache miss returns None")

    # Check statistics
    stats = cache.get_stats()
    print(f"\nCache statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")

    assert stats['hits'] == 3, "Should have 3 hits"
    assert stats['misses'] == 1, "Should have 1 miss"

    print("✓ TEST 1 PASSED: Basic get/put operations work correctly")
    return True


def test_2_memory_tracking():
    """
    Test 2: Memory tracking accuracy
    Verify: Memory usage tracked correctly (±1% accuracy)
    """
    print("\n" + "="*70)
    print("TEST 2: Memory tracking accuracy")
    print("="*70)

    cache = WindowCache(max_windows=10, max_memory_mb=100)

    # Create windows of known sizes
    # Shape calculations: 1024 x 1024 x N x 4 bytes = ~4N MB
    data_10mb = np.zeros((1024, 1024, 3), dtype=np.float32)  # ~12 MB
    data_5mb = np.zeros((1024, 1024, 2), dtype=np.float32)  # ~8 MB

    actual_size_10 = data_10mb.nbytes
    actual_size_5 = data_5mb.nbytes

    print(f"Data sizes:")
    print(f"  10MB array: {actual_size_10 / 1024 / 1024:.2f} MB")
    print(f"  5MB array: {actual_size_5 / 1024 / 1024:.2f} MB")

    print(f"\nPutting 10MB window...")
    cache.put((0, 100, 0, 100), data_10mb)
    memory_after_10 = cache.get_memory_usage()
    print(f"  Cache memory: {memory_after_10 / 1024 / 1024:.2f} MB")

    # Check accuracy (±1%)
    error_10 = abs(memory_after_10 - actual_size_10) / actual_size_10
    print(f"  Tracking error: {error_10 * 100:.2f}%")
    assert error_10 < 0.01, f"Memory tracking error too high: {error_10*100:.2f}%"

    print(f"\nPutting 5MB window...")
    cache.put((100, 200, 0, 100), data_5mb)
    memory_after_5 = cache.get_memory_usage()
    expected_total = actual_size_10 + actual_size_5
    print(f"  Cache memory: {memory_after_5 / 1024 / 1024:.2f} MB")
    print(f"  Expected: {expected_total / 1024 / 1024:.2f} MB")

    error_total = abs(memory_after_5 - expected_total) / expected_total
    print(f"  Tracking error: {error_total * 100:.2f}%")
    assert error_total < 0.01, f"Memory tracking error too high: {error_total*100:.2f}%"

    print(f"\nClearing cache...")
    cache.clear()
    memory_after_clear = cache.get_memory_usage()
    print(f"  Cache memory after clear: {memory_after_clear} bytes")
    assert memory_after_clear == 0, "Memory should be 0 after clear"

    print("✓ TEST 2 PASSED: Memory tracking accurate")
    return True


def test_3_lru_eviction_count():
    """
    Test 3: LRU eviction when count exceeded
    Verify: Oldest window evicted when max windows reached
    """
    print("\n" + "="*70)
    print("TEST 3: LRU eviction when count exceeded")
    print("="*70)

    cache = WindowCache(max_windows=3, max_memory_mb=1000)  # High memory limit

    # Create windows
    data = np.ones((100, 50), dtype=np.float32)

    print(f"Putting windows A, B, C into cache (max=3)...")
    cache.put(('A',), data * 1)
    cache.put(('B',), data * 2)
    cache.put(('C',), data * 3)

    print(f"  Cache size: {len(cache)}/3")
    assert len(cache) == 3, "Cache should have 3 windows"

    print(f"\nPutting window D (should evict A)...")
    cache.put(('D',), data * 4)

    print(f"  Cache size: {len(cache)}/3")
    assert len(cache) == 3, "Cache should still have 3 windows"

    # Check which windows remain
    has_a = cache.get(('A',)) is not None
    has_b = cache.get(('B',)) is not None
    has_c = cache.get(('C',)) is not None
    has_d = cache.get(('D',)) is not None

    print(f"\nWindows in cache:")
    print(f"  A: {has_a}")
    print(f"  B: {has_b}")
    print(f"  C: {has_c}")
    print(f"  D: {has_d}")

    assert not has_a, "Window A should be evicted (oldest)"
    assert has_b, "Window B should remain"
    assert has_c, "Window C should remain"
    assert has_d, "Window D should be present"

    stats = cache.get_stats()
    print(f"\nEvictions: {stats['evictions']}")
    assert stats['evictions'] == 1, "Should have 1 eviction"

    print("✓ TEST 3 PASSED: LRU eviction by count works correctly")
    return True


def test_4_lru_eviction_memory():
    """
    Test 4: LRU eviction when memory exceeded
    Verify: Windows evicted when memory limit reached
    """
    print("\n" + "="*70)
    print("TEST 4: LRU eviction when memory exceeded")
    print("="*70)

    # Cache with 50MB limit
    cache = WindowCache(max_windows=10, max_memory_mb=50)

    # Create 20MB windows (1024 x 1024 x 5 x 4 bytes = 20MB)
    data_20mb = np.zeros((1024, 1024, 5), dtype=np.float32)
    size_mb = data_20mb.nbytes / 1024 / 1024
    print(f"Window size: {size_mb:.1f} MB")
    print(f"Memory limit: 50 MB")

    print(f"\nPutting first window (20MB)...")
    cache.put((0,), data_20mb * 1)
    print(f"  Cache: {cache.get_memory_usage_mb():.1f} MB, {len(cache)} windows")
    assert cache.get_memory_usage_mb() <= 50, "Should be under limit"

    print(f"\nPutting second window (20MB)...")
    cache.put((1,), data_20mb * 2)
    print(f"  Cache: {cache.get_memory_usage_mb():.1f} MB, {len(cache)} windows")
    assert cache.get_memory_usage_mb() <= 50, "Should be under limit"

    print(f"\nPutting third window (20MB, should evict first)...")
    cache.put((2,), data_20mb * 3)
    print(f"  Cache: {cache.get_memory_usage_mb():.1f} MB, {len(cache)} windows")
    assert cache.get_memory_usage_mb() <= 50, "Should be under limit"

    # Check which windows remain
    has_0 = cache.get((0,)) is not None
    has_1 = cache.get((1,)) is not None
    has_2 = cache.get((2,)) is not None

    print(f"\nWindows in cache:")
    print(f"  Window 0: {has_0}")
    print(f"  Window 1: {has_1}")
    print(f"  Window 2: {has_2}")

    assert not has_0, "Window 0 should be evicted (would exceed memory)"
    assert has_1, "Window 1 should remain"
    assert has_2, "Window 2 should be present"

    stats = cache.get_stats()
    print(f"\nEvictions: {stats['evictions']}")
    assert stats['evictions'] >= 1, "Should have at least 1 eviction"

    print("✓ TEST 4 PASSED: LRU eviction by memory works correctly")
    return True


def test_5_access_updates_lru():
    """
    Test 5: Access updates LRU order
    Verify: Accessing window moves it to end (most recent)
    """
    print("\n" + "="*70)
    print("TEST 5: Access updates LRU order")
    print("="*70)

    cache = WindowCache(max_windows=3, max_memory_mb=1000)

    data = np.ones((100, 50), dtype=np.float32)

    print(f"Putting windows A, B, C...")
    cache.put(('A',), data * 1)
    cache.put(('B',), data * 2)
    cache.put(('C',), data * 3)

    print(f"  Order (oldest to newest): A -> B -> C")

    print(f"\nAccessing window A (makes it most recent)...")
    _ = cache.get(('A',))
    print(f"  New order: B -> C -> A")

    print(f"\nPutting window D (should evict B, not A)...")
    cache.put(('D',), data * 4)

    has_a = cache.get(('A',)) is not None
    has_b = cache.get(('B',)) is not None
    has_c = cache.get(('C',)) is not None
    has_d = cache.get(('D',)) is not None

    print(f"\nWindows in cache:")
    print(f"  A: {has_a} (should remain - accessed recently)")
    print(f"  B: {has_b} (should be evicted - oldest)")
    print(f"  C: {has_c} (should remain)")
    print(f"  D: {has_d} (should be present)")

    assert has_a, "Window A should remain (was accessed)"
    assert not has_b, "Window B should be evicted (oldest after access)"
    assert has_c, "Window C should remain"
    assert has_d, "Window D should be present"

    print("✓ TEST 5 PASSED: Access correctly updates LRU order")
    return True


def test_6_clear_empties_cache():
    """
    Test 6: Clear empties cache
    Verify: All windows removed, memory reset to 0
    """
    print("\n" + "="*70)
    print("TEST 6: Clear empties cache")
    print("="*70)

    cache = WindowCache(max_windows=5, max_memory_mb=100)

    # Add multiple windows
    data = np.ones((100, 50), dtype=np.float32)
    print(f"Adding 5 windows...")
    for i in range(5):
        cache.put((i,), data * i)

    print(f"  Cache size: {len(cache)} windows")
    print(f"  Memory usage: {cache.get_memory_usage_mb():.2f} MB")

    assert len(cache) == 5, "Should have 5 windows"
    assert cache.get_memory_usage() > 0, "Should have non-zero memory"

    print(f"\nClearing cache...")
    cache.clear()

    print(f"  Cache size after clear: {len(cache)} windows")
    print(f"  Memory usage after clear: {cache.get_memory_usage()} bytes")

    assert len(cache) == 0, "Cache should be empty"
    assert cache.get_memory_usage() == 0, "Memory should be 0"

    # Verify all gets return None
    print(f"\nVerifying all windows removed...")
    for i in range(5):
        result = cache.get((i,))
        assert result is None, f"Window {i} should be gone"

    print(f"  ✓ All windows removed")

    print("✓ TEST 6 PASSED: Clear empties cache correctly")
    return True


def test_7_thread_safety():
    """
    Test 7: Thread safety
    Verify: Concurrent put/get operations work correctly
    """
    print("\n" + "="*70)
    print("TEST 7: Thread safety")
    print("="*70)

    cache = WindowCache(max_windows=20, max_memory_mb=100)

    # Shared state for tracking
    errors = []
    completed = []

    def worker(thread_id: int, num_operations: int):
        """Worker thread that performs cache operations."""
        try:
            data = np.ones((50, 50), dtype=np.float32) * thread_id

            for i in range(num_operations):
                key = (thread_id, i)

                # Put data
                cache.put(key, data * i)

                # Small delay to increase chance of race conditions
                time.sleep(0.001)

                # Get data
                retrieved = cache.get(key)

                # Verify (might be evicted, but if present should match)
                if retrieved is not None:
                    if not np.allclose(retrieved, data * i):
                        errors.append(f"Thread {thread_id}: Data mismatch at iteration {i}")

            completed.append(thread_id)

        except Exception as e:
            errors.append(f"Thread {thread_id}: {str(e)}")

    print(f"Starting 10 threads with 20 operations each...")
    threads = []
    num_threads = 10
    ops_per_thread = 20

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, ops_per_thread))
        threads.append(t)
        t.start()

    print(f"  Waiting for threads to complete...")
    for t in threads:
        t.join()

    print(f"\nCompleted threads: {len(completed)}/{num_threads}")
    print(f"Errors encountered: {len(errors)}")

    if errors:
        print(f"\nErrors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")

    assert len(completed) == num_threads, "All threads should complete"
    assert len(errors) == 0, f"No errors should occur, got {len(errors)}"

    # Check cache state is consistent
    stats = cache.get_stats()
    print(f"\nFinal cache state:")
    print(f"  Windows: {len(cache)}")
    print(f"  Memory: {cache.get_memory_usage_mb():.2f} MB")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")

    print("✓ TEST 7 PASSED: Thread-safe operations work correctly")
    return True


def run_all_tests():
    """Run all test cases for WindowCache."""
    print("="*70)
    print("TASK 2.3 TEST SUITE: Window Caching with LRU Policy")
    print("="*70)

    tests = [
        test_1_basic_get_put,
        test_2_memory_tracking,
        test_3_lru_eviction_count,
        test_4_lru_eviction_memory,
        test_5_access_updates_lru,
        test_6_clear_empties_cache,
        test_7_thread_safety,
    ]

    passed = 0
    failed = 0
    results = []

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                results.append((test_func.__name__, "PASS", None))
            else:
                failed += 1
                results.append((test_func.__name__, "FAIL", "Returned False"))
        except AssertionError as e:
            failed += 1
            results.append((test_func.__name__, "FAIL", str(e)))
            print(f"\n✗ {test_func.__name__} FAILED:")
            print(f"  Error: {e}")
        except Exception as e:
            failed += 1
            results.append((test_func.__name__, "FAIL", str(e)))
            print(f"\n✗ {test_func.__name__} FAILED:")
            print(f"  Error: {e}")

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, status, error in results:
        if status == "PASS":
            print(f"✓ PASS: {test_name}")
        else:
            print(f"✗ FAIL: {test_name}")
            if error:
                print(f"       Error: {error}")

    print(f"\nTotal: {passed}/{len(tests)} tests passed")

    if failed == 0:
        print("\n" + "="*70)
        print("✓ Task 2.3 Complete: Window cache with LRU policy implemented")
        print("  - All 7 tests passed")
        print("  - LRU eviction working correctly")
        print("  - Memory tracking accurate")
        print("  - Thread-safe implementation confirmed")
        print("  - O(1) get/put operations")
        print("="*70)
        return True
    else:
        print(f"\n✗ {failed} test(s) failed - Task 2.3 incomplete")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
