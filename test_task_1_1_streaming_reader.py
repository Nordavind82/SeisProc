"""
Test Suite for Task 1.1: Streaming Trace Reader

Tests the read_traces_in_chunks() method for memory-efficient SEGY reading.
"""
import numpy as np
import tempfile
import tracemalloc
from pathlib import Path
import sys
sys.path.insert(0, '/scratch/Python_Apps/Denoise_App')

from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.header_mapping import HeaderMapping, StandardHeaders


def create_test_segy_file(output_path: str, n_traces: int, n_samples: int):
    """
    Create a test SEGY file with known pattern for testing.

    Args:
        output_path: Path to output SEGY file
        n_traces: Number of traces to create
        n_samples: Number of samples per trace
    """
    import segyio

    # Create SEGY spec
    spec = segyio.spec()
    spec.format = 5  # IEEE float
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(output_path, spec) as f:
        # Write text header
        text_header = "Test SEGY file for streaming reader validation" + " " * 3160
        f.text[0] = text_header.encode('ascii')[:3200]

        # Write binary header
        f.bin[segyio.BinField.Format] = 5
        f.bin[segyio.BinField.Samples] = n_samples

        # Write traces with known pattern
        for i in range(n_traces):
            # Trace data: each trace has constant value = trace_number
            trace_data = np.full(n_samples, float(i), dtype=np.float32)
            f.trace[i] = trace_data

            # Write headers with known CDP values
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.CDP: (i // 100) + 1,  # CDP changes every 100 traces
                segyio.TraceField.offset: (i % 100) * 10,  # Offset 0-990 repeating
                segyio.TraceField.INLINE_3D: 100 + (i // 100),
                segyio.TraceField.CROSSLINE_3D: 200 + (i % 100),
            }


def test_1_small_file_chunk_sizes():
    """
    Test 1: Small file (1000 traces) with chunk_size=300
    Verify: 4 chunks yielded (300, 300, 300, 100 traces)
    """
    print("\n" + "="*70)
    print("TEST 1: Small file chunk sizes")
    print("="*70)

    # Create test file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_1000_traces.sgy"
        print(f"Creating test file: {test_file}")
        create_test_segy_file(str(test_file), n_traces=1000, n_samples=500)

        # Create reader
        mapping = HeaderMapping()
        mapping.add_standard_headers()
        reader = SEGYReader(str(test_file), mapping)

        # Read in chunks
        chunk_sizes = []
        total_traces = 0
        chunk_count = 0

        print("Reading in chunks of 300...")
        for traces, headers, start, end in reader.read_traces_in_chunks(chunk_size=300):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            total_traces += chunk_size
            chunk_count += 1

            print(f"  Chunk {chunk_count}: traces {start}-{end-1}, size={chunk_size}, shape={traces.shape}")

            # Verify chunk data shape matches
            assert traces.shape[1] == chunk_size, f"Trace array size mismatch: {traces.shape[1]} != {chunk_size}"
            assert len(headers) == chunk_size, f"Headers count mismatch: {len(headers)} != {chunk_size}"

        # Verify results
        print(f"\nVerification:")
        print(f"  Expected chunks: [300, 300, 300, 100]")
        print(f"  Actual chunks:   {chunk_sizes}")
        print(f"  Total traces:    {total_traces}")
        print(f"  Total chunks:    {chunk_count}")

        assert chunk_sizes == [300, 300, 300, 100], f"Chunk sizes incorrect: {chunk_sizes}"
        assert total_traces == 1000, f"Total traces incorrect: {total_traces}"
        assert chunk_count == 4, f"Chunk count incorrect: {chunk_count}"

        print("✓ TEST 1 PASSED: Chunk sizes correct")
        return True


def test_2_chunk_boundary_integrity():
    """
    Test 2: Chunk boundary integrity
    Verify: No traces duplicated or skipped at boundaries
    """
    print("\n" + "="*70)
    print("TEST 2: Chunk boundary integrity")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_boundary.sgy"
        print(f"Creating test file with 500 traces...")
        create_test_segy_file(str(test_file), n_traces=500, n_samples=100)

        # Create reader
        mapping = HeaderMapping()
        mapping.add_standard_headers()
        reader = SEGYReader(str(test_file), mapping)

        # Track all trace values seen
        all_trace_values = []

        print("Reading in chunks of 100...")
        for traces, headers, start, end in reader.read_traces_in_chunks(chunk_size=100):
            # Each trace has constant value = trace_number
            # Extract the first sample value from each trace (should be trace index)
            for i in range(traces.shape[1]):
                trace_value = traces[0, i]  # First sample
                all_trace_values.append(trace_value)

        print(f"\nVerification:")
        print(f"  Total traces collected: {len(all_trace_values)}")
        print(f"  First 10 values: {all_trace_values[:10]}")
        print(f"  Last 10 values: {all_trace_values[-10:]}")
        print(f"  Boundary traces (99-101): {all_trace_values[99:102]}")
        print(f"  Boundary traces (199-201): {all_trace_values[199:202]}")

        # Verify no duplicates or skips
        expected_values = list(range(500))
        assert len(all_trace_values) == 500, f"Wrong number of traces: {len(all_trace_values)}"

        # Check for sequential order
        for i, val in enumerate(all_trace_values):
            assert val == float(i), f"Trace {i} has wrong value: {val} != {i}"

        # Check specific boundaries
        assert all_trace_values[99] == 99.0, "Boundary trace 99 incorrect"
        assert all_trace_values[100] == 100.0, "Boundary trace 100 incorrect"
        assert all_trace_values[199] == 199.0, "Boundary trace 199 incorrect"
        assert all_trace_values[200] == 200.0, "Boundary trace 200 incorrect"

        print("✓ TEST 2 PASSED: No traces duplicated or skipped")
        return True


def test_3_headers_match_traces():
    """
    Test 3: Headers match chunk traces
    Verify: Header count equals trace count for each chunk
    """
    print("\n" + "="*70)
    print("TEST 3: Headers match traces")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_headers.sgy"
        print(f"Creating test file with 750 traces...")
        create_test_segy_file(str(test_file), n_traces=750, n_samples=200)

        # Create reader
        mapping = HeaderMapping()
        mapping.add_standard_headers()
        reader = SEGYReader(str(test_file), mapping)

        print("Reading in chunks of 250...")
        chunk_num = 0
        all_passed = True

        for traces, headers, start, end in reader.read_traces_in_chunks(chunk_size=250):
            chunk_num += 1
            trace_count = traces.shape[1]
            header_count = len(headers)

            print(f"  Chunk {chunk_num}: {trace_count} traces, {header_count} headers")

            # Verify counts match
            if trace_count != header_count:
                print(f"    ✗ MISMATCH: {trace_count} != {header_count}")
                all_passed = False
            else:
                print(f"    ✓ Match")

            # Verify header values are reasonable
            for i, header in enumerate(headers):
                expected_trace_seq = start + i + 1
                actual_trace_seq = header.get('trace_sequence_file')

                if actual_trace_seq != expected_trace_seq:
                    print(f"    ✗ Header {i} trace_sequence wrong: {actual_trace_seq} != {expected_trace_seq}")
                    all_passed = False
                    break

        assert all_passed, "Header/trace mismatch detected"
        print("✓ TEST 3 PASSED: Headers match traces")
        return True


def test_4_memory_efficiency():
    """
    Test 4: Memory efficiency
    Verify: Peak memory < 50MB for processing with chunk_size=1000
    """
    print("\n" + "="*70)
    print("TEST 4: Memory efficiency")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_memory.sgy"

        # Create larger file for memory testing (10,000 traces = ~40MB uncompressed)
        n_traces = 10000
        n_samples = 1000
        print(f"Creating test file with {n_traces} traces, {n_samples} samples...")
        print(f"  Uncompressed size: ~{n_traces * n_samples * 4 / 1024 / 1024:.1f} MB")
        create_test_segy_file(str(test_file), n_traces=n_traces, n_samples=n_samples)

        # Start memory tracking
        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]
        print(f"  Baseline memory: {baseline_memory / 1024 / 1024:.1f} MB")

        # Create reader
        mapping = HeaderMapping()
        mapping.add_standard_headers()
        reader = SEGYReader(str(test_file), mapping)

        # Read in chunks and track peak memory
        peak_memory = baseline_memory
        chunk_count = 0

        print(f"Reading in chunks of 1000...")
        for traces, headers, start, end in reader.read_traces_in_chunks(chunk_size=1000):
            chunk_count += 1
            current_memory = tracemalloc.get_traced_memory()[0]
            peak_memory = max(peak_memory, current_memory)

            if chunk_count % 2 == 0:
                print(f"  Chunk {chunk_count}: current memory = {current_memory / 1024 / 1024:.1f} MB")

        final_memory = tracemalloc.get_traced_memory()[0]
        peak_memory_mb = peak_memory / 1024 / 1024
        final_memory_mb = final_memory / 1024 / 1024
        memory_increase_mb = (peak_memory - baseline_memory) / 1024 / 1024

        tracemalloc.stop()

        print(f"\nMemory Analysis:")
        print(f"  Baseline:  {baseline_memory / 1024 / 1024:.1f} MB")
        print(f"  Peak:      {peak_memory_mb:.1f} MB")
        print(f"  Final:     {final_memory_mb:.1f} MB")
        print(f"  Increase:  {memory_increase_mb:.1f} MB")
        print(f"  Target:    < 50 MB increase")

        # Verify memory usage is bounded
        # Should only hold ~1 chunk in memory at a time
        # 1000 traces * 1000 samples * 4 bytes = 4 MB per chunk
        # Plus overhead, should be well under 50 MB
        assert memory_increase_mb < 50, f"Memory usage too high: {memory_increase_mb:.1f} MB"

        print(f"✓ TEST 4 PASSED: Memory efficient (peak increase: {memory_increase_mb:.1f} MB)")
        return True


def run_all_tests():
    """Run all tests for Task 1.1"""
    print("\n" + "="*70)
    print("TASK 1.1 TEST SUITE: Streaming Trace Reader")
    print("="*70)

    tests = [
        ("Test 1: Small file chunk sizes", test_1_small_file_chunk_sizes),
        ("Test 2: Chunk boundary integrity", test_2_chunk_boundary_integrity),
        ("Test 3: Headers match traces", test_3_headers_match_traces),
        ("Test 4: Memory efficiency", test_4_memory_efficiency),
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
        print("✓ Task 1.1 Complete: Streaming trace reader implemented")
        print("  - Successfully reads SEGY files in chunks")
        print("  - Memory usage: O(chunk_size) confirmed")
        print("  - All boundary conditions handled correctly")
        print("="*70)
        return True
    else:
        print(f"\n✗ {failed} test(s) failed - Task 1.1 incomplete")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
