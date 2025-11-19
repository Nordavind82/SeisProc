"""
Test suite for Task 3.1: Lazy Ensemble Loading in GatherNavigator

Tests the lazy loading functionality with LRU caching for ensemble/gather navigation.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

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
            'trace_index': list(range(n_traces)),  # Required by LazySeismicData
            'TraceNumber': list(range(1, n_traces + 1)),
            'CDP': [100 + (i // 10) for i in range(n_traces)],  # 10 traces per CDP
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


# Test 1: Load lazy data and verify initialization
def test_1_load_lazy_data():
    """Test loading lazy data into GatherNavigator"""
    print("\n=== Test 1: Load Lazy Data ===")

    # Create test data
    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        # Create navigator
        navigator = GatherNavigator()

        # Load lazy data
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Verify initialization
        assert navigator.lazy_data is not None, "lazy_data should be set"
        assert navigator.full_data is None, "full_data should be None in lazy mode"
        assert navigator.headers_df is None, "headers_df should be None in lazy mode"
        assert navigator.n_gathers == 10, f"Expected 10 gathers, got {navigator.n_gathers}"
        assert navigator.current_gather_id == 0, "Should start at gather 0"
        assert len(navigator._ensemble_cache) == 0, "Cache should be empty initially"
        assert navigator.ensemble_keys == ['CDP', 'INLINE'], "Ensemble keys should be extracted"

        print("✓ Lazy data loaded successfully")
        print(f"  - n_gathers: {navigator.n_gathers}")
        print(f"  - ensemble_keys: {navigator.ensemble_keys}")
        print(f"  - cache size: {len(navigator._ensemble_cache)}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 2: Get current gather with lazy loading
def test_2_get_current_gather_lazy():
    """Test getting current gather in lazy mode"""
    print("\n=== Test 2: Get Current Gather (Lazy Mode) ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Get first gather
        gather_data, gather_headers, gather_info = navigator.get_current_gather()

        # Verify gather data
        assert isinstance(gather_data, SeismicData), "Should return SeismicData"
        assert gather_data.n_traces == 10, f"Expected 10 traces, got {gather_data.n_traces}"
        assert gather_data.n_samples == 500, f"Expected 500 samples, got {gather_data.n_samples}"

        # Verify headers
        assert isinstance(gather_headers, pd.DataFrame), "Should return DataFrame"
        assert len(gather_headers) == 10, f"Expected 10 headers, got {len(gather_headers)}"
        assert 'CDP' in gather_headers.columns, "CDP should be in headers"

        # Verify gather info
        assert gather_info['gather_id'] == 0, "Should be gather 0"
        assert gather_info['n_traces'] == 10, "Should have 10 traces"
        assert 'CDP' in gather_info['key_values'], "Should have CDP key value"

        # Verify cache was used
        assert len(navigator._ensemble_cache) == 1, "Gather should be cached"
        assert 0 in navigator._ensemble_cache, "Gather 0 should be in cache"
        assert navigator._cache_misses == 1, "Should have 1 cache miss"
        assert navigator._cache_hits == 0, "Should have 0 cache hits initially"

        print("✓ Gather loaded successfully")
        print(f"  - Gather data shape: {gather_data.traces.shape}")
        print(f"  - Gather headers: {len(gather_headers)} rows")
        print(f"  - Gather info: {gather_info['description']}")
        print(f"  - Cache size: {len(navigator._ensemble_cache)}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 3: Ensemble caching with LRU eviction
def test_3_ensemble_caching_lru():
    """Test LRU cache eviction when cache is full"""
    print("\n=== Test 3: Ensemble Caching (LRU Eviction) ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Set max cache to 3 for testing
        navigator._max_cached_ensembles = 3

        # Load 5 gathers (should evict 2)
        for i in range(5):
            navigator.goto_gather(i)
            navigator.get_current_gather()

        # Verify cache size is limited
        assert len(navigator._ensemble_cache) == 3, f"Cache should have max 3 ensembles, got {len(navigator._ensemble_cache)}"

        # Verify most recent gathers are cached (2, 3, 4)
        assert 2 in navigator._ensemble_cache, "Gather 2 should be cached"
        assert 3 in navigator._ensemble_cache, "Gather 3 should be cached"
        assert 4 in navigator._ensemble_cache, "Gather 4 should be cached"
        assert 0 not in navigator._ensemble_cache, "Gather 0 should be evicted"
        assert 1 not in navigator._ensemble_cache, "Gather 1 should be evicted"

        # Verify cache stats
        assert navigator._cache_misses == 5, f"Should have 5 misses, got {navigator._cache_misses}"

        # Access gather 2 (should be cache hit)
        navigator.goto_gather(2)
        navigator.get_current_gather()
        assert navigator._cache_hits == 1, f"Should have 1 hit, got {navigator._cache_hits}"

        # Access gather 0 (should be cache miss and reload)
        navigator.goto_gather(0)
        navigator.get_current_gather()
        assert navigator._cache_misses == 6, f"Should have 6 misses, got {navigator._cache_misses}"

        # Verify LRU eviction (gather 3 should be evicted now)
        assert len(navigator._ensemble_cache) == 3, "Cache should still be 3"
        assert 0 in navigator._ensemble_cache, "Gather 0 should be cached"
        assert 2 in navigator._ensemble_cache, "Gather 2 should be cached"
        assert 4 in navigator._ensemble_cache, "Gather 4 should be cached"
        assert 3 not in navigator._ensemble_cache, "Gather 3 should be evicted (LRU)"

        print("✓ LRU caching working correctly")
        print(f"  - Cache size: {len(navigator._ensemble_cache)}")
        print(f"  - Cached gathers: {list(navigator._ensemble_cache.keys())}")
        print(f"  - Cache hits: {navigator._cache_hits}")
        print(f"  - Cache misses: {navigator._cache_misses}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 4: Cache statistics
def test_4_cache_statistics():
    """Test cache statistics reporting"""
    print("\n=== Test 4: Cache Statistics ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Initial stats
        stats = navigator.get_cache_stats()
        assert stats is not None, "Should return stats in lazy mode"
        assert stats['cache_size'] == 0, "Cache should be empty"
        assert stats['cache_hits'] == 0, "Should have 0 hits"
        assert stats['cache_misses'] == 0, "Should have 0 misses"
        assert stats['hit_rate'] == 0.0, "Hit rate should be 0%"

        # Load some gathers
        navigator.get_current_gather()  # Load gather 0
        navigator.next_gather()
        navigator.get_current_gather()  # Load gather 1
        navigator.goto_gather(0)
        navigator.get_current_gather()  # Cache hit for gather 0

        # Check stats
        stats = navigator.get_cache_stats()
        assert stats['cache_size'] == 2, f"Should have 2 cached, got {stats['cache_size']}"
        assert stats['cache_hits'] == 1, f"Should have 1 hit, got {stats['cache_hits']}"
        assert stats['cache_misses'] == 2, f"Should have 2 misses, got {stats['cache_misses']}"
        assert stats['total_requests'] == 3, f"Should have 3 requests, got {stats['total_requests']}"

        expected_hit_rate = (1 / 3) * 100
        assert abs(stats['hit_rate'] - expected_hit_rate) < 0.1, \
            f"Hit rate should be ~{expected_hit_rate:.1f}%, got {stats['hit_rate']:.1f}%"

        assert 0 in stats['cached_gather_ids'], "Gather 0 should be in cache"
        assert 1 in stats['cached_gather_ids'], "Gather 1 should be in cache"

        print("✓ Cache statistics correct")
        print(f"  - Cache size: {stats['cache_size']}/{stats['max_cache_size']}")
        print(f"  - Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}")
        print(f"  - Hit rate: {stats['hit_rate']:.1f}%")
        print(f"  - Cached gathers: {stats['cached_gather_ids']}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 5: Prefetch adjacent gathers
def test_5_prefetch_adjacent():
    """Test prefetching adjacent gathers"""
    print("\n=== Test 5: Prefetch Adjacent Gathers ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Navigate to gather 5
        navigator.goto_gather(5)
        navigator.get_current_gather()

        assert len(navigator._ensemble_cache) == 1, "Should have 1 cached"
        assert 5 in navigator._ensemble_cache, "Gather 5 should be cached"

        # Prefetch adjacent
        navigator.prefetch_adjacent()

        # Should now have 3 cached (4, 5, 6)
        assert len(navigator._ensemble_cache) == 3, f"Should have 3 cached, got {len(navigator._ensemble_cache)}"
        assert 4 in navigator._ensemble_cache, "Gather 4 should be prefetched"
        assert 5 in navigator._ensemble_cache, "Gather 5 should remain cached"
        assert 6 in navigator._ensemble_cache, "Gather 6 should be prefetched"

        # Navigate to next gather (should be cache hit now)
        initial_hits = navigator._cache_hits
        navigator.next_gather()  # Go to gather 6
        navigator.get_current_gather()

        assert navigator._cache_hits == initial_hits + 1, "Should be a cache hit due to prefetch"

        print("✓ Prefetching working correctly")
        print(f"  - Cached gathers after prefetch: {list(navigator._ensemble_cache.keys())}")
        print(f"  - Next navigation was cache hit: True")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 6: Get gather info without loading traces
def test_6_get_gather_info_lazy():
    """Test getting gather info without loading full traces"""
    print("\n=== Test 6: Get Gather Info (Lazy Mode) ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Get info without loading gather
        info = navigator._get_gather_info(3)

        # Verify info is returned
        assert info['gather_id'] == 3, f"Should be gather 3, got {info['gather_id']}"
        assert info['ensemble_id'] == 3, f"Should be ensemble 3, got {info['ensemble_id']}"
        assert info['n_traces'] == 10, f"Should have 10 traces, got {info['n_traces']}"
        assert info['start_trace'] == 30, f"Should start at trace 30, got {info['start_trace']}"
        assert info['end_trace'] == 39, f"Should end at trace 39, got {info['end_trace']}"
        assert 'CDP' in info['key_values'], "Should have CDP key"
        assert 'INLINE' in info['key_values'], "Should have INLINE key"

        # Verify gather was NOT loaded into cache (only header was loaded)
        assert 3 not in navigator._ensemble_cache, "Gather should not be fully loaded"

        print("✓ Gather info retrieved without loading traces")
        print(f"  - Gather description: {info['description']}")
        print(f"  - Key values: {info['key_values']}")
        print(f"  - Cache size: {len(navigator._ensemble_cache)}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 7: Backward compatibility with full data mode
def test_7_backward_compatibility():
    """Test that legacy full data mode still works"""
    print("\n=== Test 7: Backward Compatibility (Full Data Mode) ===")

    # Create full data (legacy mode)
    n_samples = 500
    n_traces = 100
    sample_rate = 0.004
    traces = np.random.randn(n_samples, n_traces).astype(np.float32)

    full_data = SeismicData(
        traces=traces,
        sample_rate=sample_rate,
        metadata={'test': True}
    )

    # Create headers
    headers_data = {
        'TraceNumber': list(range(1, n_traces + 1)),
        'CDP': [100 + (i // 10) for i in range(n_traces)],
    }
    headers_df = pd.DataFrame(headers_data)

    # Create ensembles
    ensemble_data = []
    for i in range(10):
        ensemble_data.append({
            'ensemble_id': i,
            'start_trace': i * 10,
            'end_trace': i * 10 + 9,
            'n_traces': 10
        })
    ensembles_df = pd.DataFrame(ensemble_data)

    # Load into navigator (legacy mode)
    navigator = GatherNavigator()
    navigator.load_data(full_data, headers_df, ensembles_df)

    # Verify legacy mode
    assert navigator.full_data is not None, "full_data should be set"
    assert navigator.lazy_data is None, "lazy_data should be None in legacy mode"
    assert navigator.headers_df is not None, "headers_df should be set in legacy mode"

    # Get gather (should use legacy path)
    gather_data, gather_headers, gather_info = navigator.get_current_gather()

    assert isinstance(gather_data, SeismicData), "Should return SeismicData"
    assert gather_data.n_traces == 10, f"Expected 10 traces, got {gather_data.n_traces}"
    assert len(gather_headers) == 10, f"Expected 10 headers, got {len(gather_headers)}"

    # Verify cache not used in legacy mode
    stats = navigator.get_cache_stats()
    assert stats is None, "Should return None in legacy mode (no cache)"

    print("✓ Backward compatibility maintained")
    print(f"  - Full data mode: active")
    print(f"  - Gather loaded: {gather_info['description']}")


# Test 8: In-gather sorting with lazy mode
def test_8_sorting_with_lazy_mode():
    """Test in-gather sorting works with lazy loading"""
    print("\n=== Test 8: In-Gather Sorting (Lazy Mode) ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Set sort keys
        navigator.set_sort_keys(['OFFSET'])

        # Get gather (should be sorted)
        gather_data, gather_headers, gather_info = navigator.get_current_gather()

        # Verify sorting
        offsets = gather_headers['OFFSET'].values
        assert len(offsets) == 10, "Should have 10 traces"

        # Check if sorted
        is_sorted = all(offsets[i] <= offsets[i+1] for i in range(len(offsets)-1))
        assert is_sorted, f"Offsets should be sorted, got {offsets}"

        # Verify sort metadata
        assert 'sorted' in gather_data.metadata, "Should have sorted flag"
        assert gather_data.metadata['sorted'] == True, "Should be marked as sorted"

        print("✓ Sorting works in lazy mode")
        print(f"  - Sort keys: {navigator.sort_keys}")
        print(f"  - Offsets: {offsets}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 9: Get available sort headers in lazy mode
def test_9_get_available_sort_headers_lazy():
    """Test getting available sort headers in lazy mode"""
    print("\n=== Test 9: Get Available Sort Headers (Lazy Mode) ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Get available headers
        headers = navigator.get_available_sort_headers()

        assert isinstance(headers, list), "Should return list"
        assert len(headers) > 0, "Should have some headers"
        assert 'TraceNumber' in headers, "Should have TraceNumber"
        assert 'CDP' in headers, "Should have CDP"
        assert 'INLINE' in headers, "Should have INLINE"
        assert 'OFFSET' in headers, "Should have OFFSET"

        print("✓ Available headers retrieved in lazy mode")
        print(f"  - Headers: {headers}")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


# Test 10: Statistics with cache info
def test_10_statistics_with_cache():
    """Test that statistics include cache info in lazy mode"""
    print("\n=== Test 10: Statistics with Cache Info ===")

    lazy_data, ensembles_df, storage_dir = TestFixtures.create_test_lazy_data()

    try:
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df)

        # Load some gathers
        navigator.get_current_gather()
        navigator.next_gather()
        navigator.get_current_gather()

        # Get statistics
        stats = navigator.get_statistics()

        assert stats is not None, "Should return stats"
        assert stats['n_gathers'] == 10, "Should have 10 gathers"
        assert stats['mode'] == 'multi-gather', "Should be multi-gather mode"
        assert 'cache' in stats, "Should include cache stats"

        cache_info = stats['cache']
        assert cache_info['cache_size'] == 2, "Should have 2 cached"
        assert cache_info['cache_misses'] == 2, "Should have 2 misses"

        print("✓ Statistics include cache info")
        print(f"  - Mode: {stats['mode']}")
        print(f"  - Total gathers: {stats['n_gathers']}")
        print(f"  - Cache size: {cache_info['cache_size']}")
        print(f"  - Hit rate: {cache_info['hit_rate']:.1f}%")

    finally:
        TestFixtures.cleanup_storage(storage_dir)


if __name__ == '__main__':
    print("=" * 70)
    print("Task 3.1: Lazy Ensemble Loading in GatherNavigator - Test Suite")
    print("=" * 70)

    tests = [
        test_1_load_lazy_data,
        test_2_get_current_gather_lazy,
        test_3_ensemble_caching_lru,
        test_4_cache_statistics,
        test_5_prefetch_adjacent,
        test_6_get_gather_info_lazy,
        test_7_backward_compatibility,
        test_8_sorting_with_lazy_mode,
        test_9_get_available_sort_headers_lazy,
        test_10_statistics_with_cache,
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
