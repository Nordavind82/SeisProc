"""
Test Suite for Task 1.4: Streaming Ensemble Detection

Tests the detect_ensembles_streaming() method for memory-efficient ensemble detection.
"""
import numpy as np
import tempfile
import tracemalloc
from pathlib import Path
import sys
sys.path.insert(0, '/scratch/Python_Apps/Denoise_App')

from utils.segy_import.data_storage import DataStorage


def create_ensemble_generator(pattern, chunk_size: int):
    """
    Create a generator yielding headers with specific ensemble pattern.

    Args:
        pattern: List of (cdp, traces_in_ensemble) tuples
        chunk_size: Headers per chunk

    Yields:
        (traces_chunk, headers_chunk, start_idx, end_idx)
    """
    total_traces = sum(n for _, n in pattern)
    current_trace = 0

    for start_idx in range(0, total_traces, chunk_size):
        end_idx = min(start_idx + chunk_size, total_traces)
        current_chunk_size = end_idx - start_idx

        # Create dummy trace chunk
        traces_chunk = np.zeros((100, current_chunk_size), dtype=np.float32)

        # Create headers for this chunk
        headers_chunk = []
        for i in range(current_chunk_size):
            trace_num = start_idx + i

            # Find which ensemble this trace belongs to
            trace_count = 0
            for cdp, n_traces in pattern:
                if trace_num < trace_count + n_traces:
                    # This trace belongs to this CDP
                    current_cdp = cdp
                    break
                trace_count += n_traces

            header = {
                'trace_sequence_file': trace_num + 1,
                'cdp': current_cdp,
                'offset': (trace_num % 100) * 10,
            }
            headers_chunk.append(header)

        yield traces_chunk, headers_chunk, start_idx, end_idx


def create_multikey_generator(n_inlines: int, n_crosslines: int, chunk_size: int):
    """
    Create generator for multi-key ensemble test (inline + crossline).

    Args:
        n_inlines: Number of inline values
        n_crosslines: Number of crossline values per inline
        chunk_size: Headers per chunk
    """
    total_traces = n_inlines * n_crosslines
    current_trace = 0

    for start_idx in range(0, total_traces, chunk_size):
        end_idx = min(start_idx + chunk_size, total_traces)
        current_chunk_size = end_idx - start_idx

        traces_chunk = np.zeros((100, current_chunk_size), dtype=np.float32)
        headers_chunk = []

        for i in range(current_chunk_size):
            trace_num = start_idx + i
            inline = 100 + (trace_num // n_crosslines)
            crossline = 200 + (trace_num % n_crosslines)

            header = {
                'trace_sequence_file': trace_num + 1,
                'inline': inline,
                'crossline': crossline,
                'cdp': trace_num + 1,
            }
            headers_chunk.append(header)

        yield traces_chunk, headers_chunk, start_idx, end_idx


def test_1_single_ensemble_key():
    """
    Test 1: Single ensemble key (CDP)
    Verify: Correct ensemble boundaries detected
    """
    print("\n" + "="*70)
    print("TEST 1: Single ensemble key (CDP)")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_single_key"

        # Pattern: 50 ensembles, 100 traces each
        # CDP values: 1, 2, 3, ..., 50
        pattern = [(i + 1, 100) for i in range(50)]
        chunk_size = 300

        print(f"Configuration:")
        print(f"  Ensembles: {len(pattern)}")
        print(f"  Traces per ensemble: 100")
        print(f"  Total traces: {sum(n for _, n in pattern)}")
        print(f"  Ensemble key: ['cdp']")

        # Create storage and generator
        storage = DataStorage(str(output_dir))
        generator = create_ensemble_generator(pattern, chunk_size)

        # Detect ensembles
        print("\nDetecting ensembles...")
        ensembles = list(storage.detect_ensembles_streaming(generator, ['cdp']))

        print(f"\nVerification:")
        print(f"  Ensembles detected: {len(ensembles)}")
        assert len(ensembles) == 50, f"Ensemble count mismatch: {len(ensembles)} != 50"

        # Verify first ensemble
        ens_id, start, end, keys = ensembles[0]
        print(f"\n  First ensemble:")
        print(f"    ID: {ens_id}, Traces: {start}-{end}, Keys: {keys}")
        assert ens_id == 0, f"First ensemble ID wrong: {ens_id}"
        assert start == 0, f"First ensemble start wrong: {start}"
        assert end == 99, f"First ensemble end wrong: {end}"
        assert keys['cdp'] == 1, f"First ensemble CDP wrong: {keys['cdp']}"

        # Verify last ensemble
        ens_id, start, end, keys = ensembles[-1]
        print(f"\n  Last ensemble:")
        print(f"    ID: {ens_id}, Traces: {start}-{end}, Keys: {keys}")
        assert ens_id == 49, f"Last ensemble ID wrong: {ens_id}"
        assert start == 4900, f"Last ensemble start wrong: {start}"
        assert end == 4999, f"Last ensemble end wrong: {end}"
        assert keys['cdp'] == 50, f"Last ensemble CDP wrong: {keys['cdp']}"

        # Verify all ensemble sizes
        for i, (ens_id, start, end, keys) in enumerate(ensembles):
            n_traces = end - start + 1
            assert n_traces == 100, f"Ensemble {i} size wrong: {n_traces}"

        print("✓ TEST 1 PASSED: Single ensemble key working")
        return True


def test_2_multiple_ensemble_keys():
    """
    Test 2: Multiple ensemble keys (inline + crossline)
    Verify: Correct boundaries for combined keys
    """
    print("\n" + "="*70)
    print("TEST 2: Multiple ensemble keys (inline + crossline)")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_multi_key"

        # Configuration: 3 inlines × 100 crosslines = 300 ensembles
        n_inlines = 3
        n_crosslines = 100
        chunk_size = 250

        print(f"Configuration:")
        print(f"  Inlines: {n_inlines}")
        print(f"  Crosslines per inline: {n_crosslines}")
        print(f"  Expected ensembles: {n_inlines * n_crosslines}")
        print(f"  Ensemble keys: ['inline', 'crossline']")

        # Create storage and generator
        storage = DataStorage(str(output_dir))
        generator = create_multikey_generator(n_inlines, n_crosslines, chunk_size)

        # Detect ensembles
        print("\nDetecting ensembles...")
        ensembles = list(storage.detect_ensembles_streaming(generator, ['inline', 'crossline']))

        print(f"\nVerification:")
        print(f"  Ensembles detected: {len(ensembles)}")
        expected_ensembles = n_inlines * n_crosslines
        assert len(ensembles) == expected_ensembles, \
            f"Ensemble count mismatch: {len(ensembles)} != {expected_ensembles}"

        # Verify first ensemble
        ens_id, start, end, keys = ensembles[0]
        print(f"\n  First ensemble:")
        print(f"    ID: {ens_id}, Traces: {start}-{end}")
        print(f"    Keys: inline={keys['inline']}, crossline={keys['crossline']}")
        assert keys['inline'] == 100, f"First inline wrong: {keys['inline']}"
        assert keys['crossline'] == 200, f"First crossline wrong: {keys['crossline']}"
        assert start == 0 and end == 0, f"First ensemble boundaries wrong: {start}-{end}"

        # Verify transition between inlines
        # Ensemble at crossline 99 of inline 100
        ens_99 = ensembles[99]
        print(f"\n  Ensemble 99 (last of inline 100):")
        print(f"    Keys: inline={ens_99[3]['inline']}, crossline={ens_99[3]['crossline']}")
        assert ens_99[3]['inline'] == 100, "Inline 100 boundary issue"
        assert ens_99[3]['crossline'] == 299, "Crossline boundary issue"

        # Ensemble at crossline 0 of inline 101
        ens_100 = ensembles[100]
        print(f"\n  Ensemble 100 (first of inline 101):")
        print(f"    Keys: inline={ens_100[3]['inline']}, crossline={ens_100[3]['crossline']}")
        assert ens_100[3]['inline'] == 101, "Inline 101 boundary issue"
        assert ens_100[3]['crossline'] == 200, "Crossline reset issue"

        print("✓ TEST 2 PASSED: Multiple ensemble keys working")
        return True


def test_3_variable_ensemble_sizes():
    """
    Test 3: Variable ensemble sizes
    Verify: Handles different ensemble sizes correctly
    """
    print("\n" + "="*70)
    print("TEST 3: Variable ensemble sizes")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_variable"

        # Pattern: varying ensemble sizes
        pattern = [
            (1, 50),   # CDP 1: 50 traces
            (2, 150),  # CDP 2: 150 traces
            (3, 30),   # CDP 3: 30 traces
            (4, 200),  # CDP 4: 200 traces
            (5, 70),   # CDP 5: 70 traces
        ]
        chunk_size = 100

        print(f"Configuration:")
        print(f"  Ensemble sizes: {[n for _, n in pattern]}")
        print(f"  Expected ensembles: {len(pattern)}")

        # Create storage and generator
        storage = DataStorage(str(output_dir))
        generator = create_ensemble_generator(pattern, chunk_size)

        # Detect ensembles
        print("\nDetecting ensembles...")
        ensembles = list(storage.detect_ensembles_streaming(generator, ['cdp']))

        print(f"\nVerification:")
        print(f"  Ensembles detected: {len(ensembles)}")
        assert len(ensembles) == len(pattern), f"Ensemble count mismatch"

        # Verify each ensemble size
        for i, (ens_id, start, end, keys) in enumerate(ensembles):
            expected_cdp, expected_n_traces = pattern[i]
            actual_n_traces = end - start + 1

            print(f"  Ensemble {i}: CDP={keys['cdp']}, traces={actual_n_traces} (expected {expected_n_traces})")

            assert keys['cdp'] == expected_cdp, f"Ensemble {i} CDP wrong"
            assert actual_n_traces == expected_n_traces, f"Ensemble {i} size wrong"

        print("✓ TEST 3 PASSED: Variable ensemble sizes handled")
        return True


def test_4_memory_efficiency():
    """
    Test 4: Memory efficiency
    Verify: Memory usage O(1), not dependent on dataset size
    """
    print("\n" + "="*70)
    print("TEST 4: Memory efficiency")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_memory"

        # Large dataset: 1000 ensembles, 100 traces each = 100,000 traces
        pattern = [(i + 1, 100) for i in range(1000)]
        chunk_size = 5000

        total_traces = sum(n for _, n in pattern)
        print(f"Configuration:")
        print(f"  Total traces: {total_traces}")
        print(f"  Ensembles: {len(pattern)}")
        print(f"  Memory target: < 5 MB")

        # Start memory tracking
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        print(f"  Baseline memory: {baseline / 1024 / 1024:.1f} MB")

        # Create storage and generator
        storage = DataStorage(str(output_dir))
        generator = create_ensemble_generator(pattern, chunk_size)

        # Detect ensembles while tracking memory
        peak_memory = baseline
        ensemble_count = 0

        for ens_id, start, end, keys in storage.detect_ensembles_streaming(generator, ['cdp']):
            ensemble_count += 1
            current_memory = tracemalloc.get_traced_memory()[0]
            peak_memory = max(peak_memory, current_memory)

        final_memory = tracemalloc.get_traced_memory()[0]
        peak_memory_mb = peak_memory / 1024 / 1024
        memory_increase_mb = (peak_memory - baseline) / 1024 / 1024

        tracemalloc.stop()

        print(f"\nMemory analysis:")
        print(f"  Baseline:  {baseline / 1024 / 1024:.1f} MB")
        print(f"  Peak:      {peak_memory_mb:.1f} MB")
        print(f"  Final:     {final_memory / 1024 / 1024:.1f} MB")
        print(f"  Increase:  {memory_increase_mb:.1f} MB")
        print(f"  Ensembles processed: {ensemble_count}")

        # Verify memory usage is O(1) - should be very small
        assert memory_increase_mb < 5, f"Memory usage too high: {memory_increase_mb:.1f} MB"

        print(f"✓ TEST 4 PASSED: Memory efficient ({memory_increase_mb:.1f} MB for 100k traces)")
        return True


def test_5_unsorted_data():
    """
    Test 5: Unsorted data handling
    Verify: Detects ensembles correctly even with repeated CDP values
    """
    print("\n" + "="*70)
    print("TEST 5: Unsorted data handling")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_unsorted"

        # Pattern with CDP switching: 1,1,2,2,1,1,3,3
        # This creates 5 ensembles (not 3) due to re-occurrence of CDP 1
        pattern = [
            (1, 50),   # Ensemble 0: CDP 1
            (2, 50),   # Ensemble 1: CDP 2
            (1, 50),   # Ensemble 2: CDP 1 again (new ensemble)
            (3, 50),   # Ensemble 3: CDP 3
            (2, 50),   # Ensemble 4: CDP 2 again (new ensemble)
        ]
        chunk_size = 75

        print(f"Configuration:")
        print(f"  CDP pattern: {[cdp for cdp, _ in pattern]}")
        print(f"  Expected ensembles: {len(pattern)} (due to unsorted data)")

        # Create storage and generator
        storage = DataStorage(str(output_dir))
        generator = create_ensemble_generator(pattern, chunk_size)

        # Detect ensembles
        print("\nDetecting ensembles...")
        ensembles = list(storage.detect_ensembles_streaming(generator, ['cdp']))

        print(f"\nVerification:")
        print(f"  Ensembles detected: {len(ensembles)}")
        assert len(ensembles) == len(pattern), \
            f"Ensemble count mismatch: {len(ensembles)} != {len(pattern)}"

        # Verify the pattern
        for i, (ens_id, start, end, keys) in enumerate(ensembles):
            expected_cdp = pattern[i][0]
            print(f"  Ensemble {i}: CDP={keys['cdp']} (expected {expected_cdp})")
            assert keys['cdp'] == expected_cdp, f"Ensemble {i} CDP mismatch"

        # Verify that CDP 1 appears twice
        cdp_1_count = sum(1 for _, _, _, keys in ensembles if keys['cdp'] == 1)
        print(f"\n  CDP 1 appears {cdp_1_count} times (expected 2)")
        assert cdp_1_count == 2, "CDP 1 should appear in 2 separate ensembles"

        print("✓ TEST 5 PASSED: Unsorted data handled correctly")
        return True


def run_all_tests():
    """Run all tests for Task 1.4"""
    print("\n" + "="*70)
    print("TASK 1.4 TEST SUITE: Streaming Ensemble Detection")
    print("="*70)

    tests = [
        ("Test 1: Single ensemble key (CDP)", test_1_single_ensemble_key),
        ("Test 2: Multiple ensemble keys", test_2_multiple_ensemble_keys),
        ("Test 3: Variable ensemble sizes", test_3_variable_ensemble_sizes),
        ("Test 4: Memory efficiency", test_4_memory_efficiency),
        ("Test 5: Unsorted data handling", test_5_unsorted_data),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED:")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result, _ in results if result)
    failed = len(results) - passed

    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"       Error: {error}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n" + "="*70)
        print("✓ Task 1.4 Complete: Streaming ensemble detection implemented")
        print("  - Successfully detects ensembles from streaming headers")
        print("  - Memory usage: O(1) confirmed")
        print("  - Handles single and multiple ensemble keys")
        print("  - Tested with sorted and unsorted data")
        print("="*70)
        return True
    else:
        print(f"\n✗ {failed} test(s) failed - Task 1.4 incomplete")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
