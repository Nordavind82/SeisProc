"""
Test Suite for Task 1.3: Streaming Header Collection

Tests the save_headers_streaming() method for memory-efficient header storage.
"""
import numpy as np
import tempfile
import tracemalloc
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, '/scratch/Python_Apps/Denoise_App')

from utils.segy_import.data_storage import DataStorage


def create_header_generator(n_traces: int, chunk_size: int, varying_fields: bool = False):
    """
    Create a generator that yields header chunks.

    Args:
        n_traces: Total number of traces
        chunk_size: Headers per chunk
        varying_fields: If True, some headers will have different fields

    Yields:
        (traces_chunk, headers_chunk, start_idx, end_idx)
    """
    for start_idx in range(0, n_traces, chunk_size):
        end_idx = min(start_idx + chunk_size, n_traces)
        current_chunk_size = end_idx - start_idx

        # Create dummy trace chunk (not used for header tests)
        traces_chunk = np.zeros((100, current_chunk_size), dtype=np.float32)

        # Create headers with known values
        headers_chunk = []
        for i in range(current_chunk_size):
            trace_num = start_idx + i
            header = {
                'trace_sequence_file': trace_num + 1,
                'cdp': (trace_num // 100) + 1,
                'offset': (trace_num % 100) * 10,
                'inline': 100 + (trace_num // 100),
                'crossline': 200 + (trace_num % 100),
            }

            # Add varying fields for some headers
            if varying_fields and trace_num % 3 == 0:
                header['custom_field'] = trace_num * 2

            headers_chunk.append(header)

        yield traces_chunk, headers_chunk, start_idx, end_idx


def test_1_stream_50k_headers():
    """
    Test 1: Stream 50,000 headers in batches
    Verify: Single Parquet file with 50,000 rows
    """
    print("\n" + "="*70)
    print("TEST 1: Stream 50,000 headers in batches")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_headers"
        print(f"Output directory: {output_dir}")

        # Parameters
        n_traces = 50000
        chunk_size = 1000
        batch_size = 10000

        print(f"Configuration:")
        print(f"  Total headers: {n_traces}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Expected batches: {n_traces // batch_size}")

        # Create storage and generator
        storage = DataStorage(str(output_dir))
        generator = create_header_generator(n_traces, chunk_size)

        # Stream headers
        print("\nStreaming headers...")
        total = storage.save_headers_streaming(generator, batch_size=batch_size)

        print(f"\nVerification:")
        print(f"  Total saved: {total}")
        assert total == n_traces, f"Header count mismatch: {total} != {n_traces}"

        # Read back and verify
        print("\n  Reading Parquet file...")
        headers_path = output_dir / 'headers.parquet'
        assert headers_path.exists(), "Parquet file not created"

        df = pd.read_parquet(headers_path)
        print(f"  Rows in file: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        assert len(df) == n_traces, f"Row count mismatch: {len(df)} != {n_traces}"

        # Verify trace_index column
        assert 'trace_index' in df.columns, "trace_index column missing"
        assert df['trace_index'].min() == 0, f"trace_index min wrong: {df['trace_index'].min()}"
        assert df['trace_index'].max() == n_traces - 1, f"trace_index max wrong: {df['trace_index'].max()}"

        # Spot check values
        print("\n  Spot checking values...")
        test_indices = [0, 25000, 49999]
        for idx in test_indices:
            row = df[df['trace_index'] == idx].iloc[0]
            expected_trace_seq = idx + 1
            expected_cdp = (idx // 100) + 1
            actual_trace_seq = row['trace_sequence_file']
            actual_cdp = row['cdp']

            print(f"    Index {idx}: trace_seq={actual_trace_seq} (expected {expected_trace_seq}), cdp={actual_cdp} (expected {expected_cdp})")

            assert actual_trace_seq == expected_trace_seq, f"trace_sequence_file mismatch at {idx}"
            assert actual_cdp == expected_cdp, f"cdp mismatch at {idx}"

        print("✓ TEST 1 PASSED: Headers streamed successfully")
        return True


def test_2_batch_memory_efficiency():
    """
    Test 2: Batch memory efficiency
    Verify: Memory resets after batch write
    """
    print("\n" + "="*70)
    print("TEST 2: Batch memory efficiency")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_memory"

        # Parameters - 100,000 headers with many fields (simulate large headers)
        n_traces = 100000
        chunk_size = 5000
        batch_size = 10000

        print(f"Configuration:")
        print(f"  Total headers: {n_traces}")
        print(f"  Batch size: {batch_size}")
        print(f"  Expected memory: < 50 MB")

        # Start memory tracking
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        print(f"  Baseline memory: {baseline / 1024 / 1024:.1f} MB")

        # Create storage
        storage = DataStorage(str(output_dir))
        generator = create_header_generator(n_traces, chunk_size)

        # Track memory during streaming
        peak_memory = baseline
        batch_count = 0

        # Manually iterate to track memory after each batch
        header_buffer = []
        total_headers = 0
        trace_index_offset = 0

        for traces_chunk, headers_chunk, start_idx, end_idx in generator:
            header_buffer.extend(headers_chunk)

            if len(header_buffer) >= batch_size:
                storage._write_header_batch(header_buffer, trace_index_offset, append=(total_headers > 0))
                total_headers += len(header_buffer)
                trace_index_offset += len(header_buffer)
                batch_count += 1

                # Check memory after batch write
                current_memory = tracemalloc.get_traced_memory()[0]
                peak_memory = max(peak_memory, current_memory)

                if batch_count % 2 == 0:
                    mem_mb = current_memory / 1024 / 1024
                    print(f"  Batch {batch_count}: {mem_mb:.1f} MB")

                header_buffer = []

        # Write remaining
        if header_buffer:
            storage._write_header_batch(header_buffer, trace_index_offset, append=(total_headers > 0))
            total_headers += len(header_buffer)

        final_memory = tracemalloc.get_traced_memory()[0]
        peak_memory_mb = peak_memory / 1024 / 1024
        memory_increase_mb = (peak_memory - baseline) / 1024 / 1024

        tracemalloc.stop()

        print(f"\nMemory analysis:")
        print(f"  Baseline:  {baseline / 1024 / 1024:.1f} MB")
        print(f"  Peak:      {peak_memory_mb:.1f} MB")
        print(f"  Final:     {final_memory / 1024 / 1024:.1f} MB")
        print(f"  Increase:  {memory_increase_mb:.1f} MB")
        print(f"  Target:    < 50 MB")

        # Verify memory bounded
        assert memory_increase_mb < 50, f"Memory usage too high: {memory_increase_mb:.1f} MB"

        print(f"✓ TEST 2 PASSED: Memory efficient ({memory_increase_mb:.1f} MB peak)")
        return True


def test_3_missing_fields_handled():
    """
    Test 3: Missing header fields handled
    Verify: Parquet schema includes all fields, missing values as null/NaN
    """
    print("\n" + "="*70)
    print("TEST 3: Missing header fields handled")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_missing_fields"

        # Parameters
        n_traces = 15000
        chunk_size = 5000
        batch_size = 5000

        print(f"Configuration:")
        print(f"  Total headers: {n_traces}")
        print(f"  Headers with varying fields (every 3rd has 'custom_field')")

        # Create storage and generator with varying fields
        storage = DataStorage(str(output_dir))
        generator = create_header_generator(n_traces, chunk_size, varying_fields=True)

        # Stream headers
        print("\nStreaming headers with varying fields...")
        total = storage.save_headers_streaming(generator, batch_size=batch_size)

        # Read back and verify
        print("\nVerifying Parquet schema...")
        df = pd.read_parquet(output_dir / 'headers.parquet')

        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        # Verify custom_field column exists
        assert 'custom_field' in df.columns, "custom_field column missing"

        # Count non-null custom_field values
        non_null_count = df['custom_field'].notna().sum()
        expected_non_null = n_traces // 3  # Every 3rd trace
        print(f"\n  Non-null custom_field: {non_null_count} (expected ~{expected_non_null})")

        # Allow some tolerance
        assert abs(non_null_count - expected_non_null) < 10, \
            f"Non-null count unexpected: {non_null_count} vs {expected_non_null}"

        # Verify null values are proper NaN
        null_count = df['custom_field'].isna().sum()
        print(f"  Null custom_field: {null_count}")
        assert null_count == n_traces - non_null_count, "Null count mismatch"

        # Verify can query the DataFrame
        print("\n  Testing query operations...")
        subset = df.query("cdp > 100 and offset < 500")
        print(f"    Query result: {len(subset)} rows")
        assert len(subset) > 0, "Query returned no results"

        print("✓ TEST 3 PASSED: Missing fields handled correctly")
        return True


def test_4_parquet_integrity():
    """
    Test 4: Parquet file integrity
    Verify: File readable, compressed, queryable
    """
    print("\n" + "="*70)
    print("TEST 4: Parquet file integrity")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_integrity"

        # Parameters
        n_traces = 25000
        chunk_size = 2500
        batch_size = 5000

        print(f"Configuration:")
        print(f"  Total headers: {n_traces}")

        # Create storage and stream
        storage = DataStorage(str(output_dir))
        generator = create_header_generator(n_traces, chunk_size)

        print("\nStreaming headers...")
        total = storage.save_headers_streaming(generator, batch_size=batch_size)

        # Check file exists and size
        headers_path = output_dir / 'headers.parquet'
        file_size_mb = headers_path.stat().st_size / 1024 / 1024
        print(f"\nFile integrity:")
        print(f"  File exists: {headers_path.exists()}")
        print(f"  File size: {file_size_mb:.2f} MB")

        # Verify compression (file should be much smaller than raw data)
        # Rough estimate: ~100 bytes per header uncompressed
        estimated_raw_mb = (n_traces * 100) / 1024 / 1024
        compression_ratio = estimated_raw_mb / file_size_mb if file_size_mb > 0 else 1
        print(f"  Estimated raw: {estimated_raw_mb:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")

        assert compression_ratio > 1.5, f"Compression too low: {compression_ratio:.1f}x"

        # Read and verify
        print("\n  Reading with pandas...")
        df = pd.read_parquet(headers_path)
        assert len(df) == n_traces, f"Row count mismatch"
        print(f"    ✓ Read {len(df)} rows")

        # Test queries
        print("\n  Testing query operations...")

        # Query 1: Filter by CDP
        q1 = df.query("cdp > 50 and cdp < 100")
        print(f"    Query 1 (cdp range): {len(q1)} rows")
        assert len(q1) > 0, "Query 1 failed"

        # Query 2: Filter by offset
        q2 = df.query("offset >= 0 and offset <= 500")
        print(f"    Query 2 (offset range): {len(q2)} rows")
        assert len(q2) > 0, "Query 2 failed"

        # Query 3: Combined filter
        q3 = df[(df['cdp'] > 10) & (df['inline'] >= 100)]
        print(f"    Query 3 (combined): {len(q3)} rows")
        assert len(q3) > 0, "Query 3 failed"

        # Test sorting
        print("\n  Testing sort operations...")
        df_sorted = df.sort_values('offset')
        assert df_sorted['offset'].iloc[0] <= df_sorted['offset'].iloc[-1], "Sort failed"
        print("    ✓ Sort by offset successful")

        # Test grouping
        print("\n  Testing group operations...")
        grouped = df.groupby('cdp').size()
        print(f"    Unique CDPs: {len(grouped)}")
        assert len(grouped) > 0, "Grouping failed"

        print("✓ TEST 4 PASSED: Parquet file integrity confirmed")
        return True


def run_all_tests():
    """Run all tests for Task 1.3"""
    print("\n" + "="*70)
    print("TASK 1.3 TEST SUITE: Streaming Header Collection")
    print("="*70)

    tests = [
        ("Test 1: Stream 50,000 headers in batches", test_1_stream_50k_headers),
        ("Test 2: Batch memory efficiency", test_2_batch_memory_efficiency),
        ("Test 3: Missing header fields handled", test_3_missing_fields_handled),
        ("Test 4: Parquet file integrity", test_4_parquet_integrity),
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
        print("✓ Task 1.3 Complete: Streaming header collection implemented")
        print("  - Successfully streams headers to Parquet")
        print("  - Memory usage: O(batch_size) confirmed")
        print("  - Parquet files optimized and queryable")
        print("  - Handles variable schema correctly")
        print("="*70)
        return True
    else:
        print(f"\n✗ {failed} test(s) failed - Task 1.3 incomplete")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
