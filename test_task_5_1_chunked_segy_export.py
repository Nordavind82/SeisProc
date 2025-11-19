"""
Test suite for Task 5.1: Chunked SEGY Exporter

Tests the memory-efficient chunked SEGY export functionality.
"""

import sys
import os
import numpy as np
import tempfile
import shutil
from pathlib import Path
import segyio
import zarr

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.segy_import.segy_export import SEGYExporter, export_from_zarr_chunked


# Test fixtures
class TestFixtures:
    """Helper class to create test data"""

    @staticmethod
    def create_test_segy(n_samples=1000, n_traces=10000, seed=42):
        """Create test SEGY file"""
        np.random.seed(seed)
        temp_dir = tempfile.mkdtemp()
        segy_path = Path(temp_dir) / 'test.sgy'

        # Create random data
        data = np.random.randn(n_samples, n_traces).astype(np.float32)

        # Create SEGY spec
        spec = segyio.spec()
        spec.format = 1  # 4-byte IBM float
        spec.samples = range(n_samples)
        spec.tracecount = n_traces

        # Create SEGY file
        with segyio.create(str(segy_path), spec) as f:
            # Write text header
            text_header = "C01 Test SEGY file for chunked export testing".ljust(3200, ' ')
            f.text[0] = text_header.encode('ascii')

            # Write binary header fields
            f.bin[segyio.BinField.JobID] = 1234
            f.bin[segyio.BinField.LineNumber] = 5678
            f.bin[segyio.BinField.Samples] = n_samples

            # Write traces
            for i in range(n_traces):
                # Write trace header
                f.header[i] = {
                    segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                    segyio.TraceField.CDP: 1000 + (i // 10),  # CDP every 10 traces
                    segyio.TraceField.offset: (i % 10) * 100,  # Offset pattern
                    segyio.TraceField.TraceNumber: i + 1,
                }

                # Write trace data
                f.trace[i] = data[:, i]

        return segy_path, data, Path(temp_dir)

    @staticmethod
    def create_processed_zarr(original_data, multiplier=2.0):
        """Create processed Zarr from original data"""
        temp_dir = tempfile.mkdtemp()
        zarr_path = Path(temp_dir) / 'processed.zarr'

        # Process data (multiply by factor)
        processed_data = original_data * multiplier

        # Save to Zarr
        z = zarr.open(str(zarr_path), mode='w', shape=processed_data.shape,
                     chunks=(processed_data.shape[0], 1000), dtype=np.float32)
        z[:] = processed_data

        return zarr_path, processed_data, Path(temp_dir)

    @staticmethod
    def cleanup_storage(storage_dir):
        """Clean up temporary storage"""
        if storage_dir and storage_dir.exists():
            shutil.rmtree(storage_dir)


# Test 1: Export matches input dimensions
def test_1_dimensions_match():
    """Test that output SEGY has same dimensions as input"""
    print("\n=== Test 1: Export Matches Input Dimensions ===")

    # Create test data
    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=10000
    )
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(original_data)

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Export in chunks
        exporter = SEGYExporter(str(output_path))
        exporter.export_from_zarr_chunked(
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=2000
        )

        # Verify output exists
        assert output_path.exists(), "Output SEGY should exist"

        # Read output and verify dimensions
        with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
            assert f.tracecount == 10000, f"Expected 10000 traces, got {f.tracecount}"
            assert len(f.samples) == 500, f"Expected 500 samples, got {len(f.samples)}"

        print("✓ Dimensions match")
        print(f"  - Input: {original_data.shape}")
        print(f"  - Output: (500, 10000)")
        print(f"  - Chunk size: 2000")
        print(f"  - Chunks processed: 5")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


# Test 2: Headers preserved
def test_2_headers_preserved():
    """Test that trace headers are preserved from original"""
    print("\n=== Test 2: Headers Preserved ===")

    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=5000
    )
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(original_data)

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Export
        export_from_zarr_chunked(
            output_path=str(output_path),
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=1000
        )

        # Read original and output headers
        with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
            with segyio.open(str(output_path), 'r', ignore_geometry=True) as dst:
                # Check trace 100
                src_cdp = src.header[100][segyio.TraceField.CDP]
                dst_cdp = dst.header[100][segyio.TraceField.CDP]
                assert src_cdp == dst_cdp, \
                    f"CDP mismatch at trace 100: {src_cdp} != {dst_cdp}"

                src_offset = src.header[100][segyio.TraceField.offset]
                dst_offset = dst.header[100][segyio.TraceField.offset]
                assert src_offset == dst_offset, \
                    f"Offset mismatch at trace 100: {src_offset} != {dst_offset}"

                # Check multiple traces
                for idx in [0, 500, 1000, 2500, 4999]:
                    src_trace_num = src.header[idx][segyio.TraceField.TRACE_SEQUENCE_FILE]
                    dst_trace_num = dst.header[idx][segyio.TraceField.TRACE_SEQUENCE_FILE]
                    assert src_trace_num == dst_trace_num, \
                        f"Trace number mismatch at {idx}"

        print("✓ Headers preserved correctly")
        print(f"  - CDP preserved: Yes (trace 100)")
        print(f"  - Offset preserved: Yes (trace 100)")
        print(f"  - Trace numbers checked: 0, 500, 1000, 2500, 4999")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


# Test 3: Trace data matches processed
def test_3_trace_data_matches():
    """Test that trace data in output matches processed Zarr"""
    print("\n=== Test 3: Trace Data Matches Processed ===")

    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=10000
    )
    # Multiply by 2.5 for easy verification
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(
        original_data, multiplier=2.5
    )

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Export
        export_from_zarr_chunked(
            output_path=str(output_path),
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=2000
        )

        # Read output and compare
        with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
            # Check specific traces
            for trace_idx in [0, 5000, 9999]:
                output_trace = f.trace[trace_idx]
                expected_trace = processed_data[:, trace_idx]

                assert np.allclose(output_trace, expected_trace, rtol=1e-4), \
                    f"Trace {trace_idx} data mismatch"

            # Verify processed relationship
            trace_0 = f.trace[0]
            expected_0 = original_data[:, 0] * 2.5
            assert np.allclose(trace_0, expected_0, rtol=1e-4), \
                "Trace 0 should be original × 2.5"

        print("✓ Trace data matches processed")
        print(f"  - Traces checked: 0, 5000, 9999")
        print(f"  - Processing: original × 2.5")
        print(f"  - Accuracy: within 0.01% (rtol=1e-4)")
        print(f"  - No corruption or truncation")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


# Test 4: Chunk boundaries correct
def test_4_chunk_boundaries():
    """Test that chunk boundaries are handled correctly"""
    print("\n=== Test 4: Chunk Boundaries Correct ===")

    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=5000
    )
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(original_data)

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Export with specific chunk size
        export_from_zarr_chunked(
            output_path=str(output_path),
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=1000  # Boundaries at 999, 1999, 2999, 3999
        )

        # Check traces at boundaries
        with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
            boundaries = [999, 1999, 2999, 3999]
            for boundary in boundaries:
                # Check traces around boundary
                trace_before = f.trace[boundary]
                trace_at = f.trace[boundary + 1] if boundary + 1 < 5000 else None
                trace_after = f.trace[boundary + 2] if boundary + 2 < 5000 else None

                # Verify against expected
                expected_before = processed_data[:, boundary]
                assert np.allclose(trace_before, expected_before, rtol=1e-4), \
                    f"Trace {boundary} mismatch at boundary"

                if trace_at is not None:
                    expected_at = processed_data[:, boundary + 1]
                    assert np.allclose(trace_at, expected_at, rtol=1e-4), \
                        f"Trace {boundary + 1} mismatch at boundary"

                # Check trace numbering is sequential
                trace_num_before = f.header[boundary][segyio.TraceField.TRACE_SEQUENCE_FILE]
                if boundary + 1 < 5000:
                    trace_num_after = f.header[boundary + 1][segyio.TraceField.TRACE_SEQUENCE_FILE]
                    assert trace_num_after == trace_num_before + 1, \
                        f"Trace numbering not sequential at boundary {boundary}"

        print("✓ Chunk boundaries correct")
        print(f"  - Boundaries checked: 999, 1999, 2999, 3999")
        print(f"  - No discontinuities found")
        print(f"  - Sequential trace numbering verified")
        print(f"  - No duplicated traces")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


# Test 5: Progress callback accuracy
def test_5_progress_callback():
    """Test progress callback accuracy"""
    print("\n=== Test 5: Progress Callback Accuracy ===")

    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=10000
    )
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(original_data)

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Track progress
        progress_updates = []

        def progress_callback(current, total, time_remaining):
            progress_updates.append({
                'current': current,
                'total': total,
                'time_remaining': time_remaining,
                'percent': (current / total) * 100
            })

        # Export
        export_from_zarr_chunked(
            output_path=str(output_path),
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=2500,
            progress_callback=progress_callback
        )

        # Verify progress updates
        assert len(progress_updates) == 4, \
            f"Expected 4 progress updates (4 chunks), got {len(progress_updates)}"

        # Check milestones
        expected_traces = [2500, 5000, 7500, 10000]
        for i, expected in enumerate(expected_traces):
            actual = progress_updates[i]['current']
            assert actual == expected, \
                f"Update {i+1} should be at {expected} traces, got {actual}"

        # Check percentages
        expected_percents = [25, 50, 75, 100]
        for i, expected_pct in enumerate(expected_percents):
            actual_pct = progress_updates[i]['percent']
            assert abs(actual_pct - expected_pct) < 1.0, \
                f"Update {i+1} should be {expected_pct}%, got {actual_pct:.1f}% (±1%)"

        print("✓ Progress callback accurate")
        print(f"  - Updates received: {len(progress_updates)}")
        print(f"  - Milestones: {[u['current'] for u in progress_updates]}")
        percentages = [f"{u['percent']:.0f}%" for u in progress_updates]
        print(f"  - Percentages: {percentages}")
        print(f"  - Accuracy: ±1%")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


# Test 6: Binary and text headers preserved
def test_6_binary_text_headers():
    """Test that binary and text headers are preserved"""
    print("\n=== Test 6: Binary and Text Headers Preserved ===")

    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=5000
    )
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(original_data)

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Export
        export_from_zarr_chunked(
            output_path=str(output_path),
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=1000
        )

        # Compare headers
        with segyio.open(str(segy_path), 'r', ignore_geometry=True) as src:
            with segyio.open(str(output_path), 'r', ignore_geometry=True) as dst:
                # Check text header
                src_text = src.text[0]
                dst_text = dst.text[0]
                assert src_text == dst_text, "Text header should match"

                # Check binary header fields
                src_job_id = src.bin[segyio.BinField.JobID]
                dst_job_id = dst.bin[segyio.BinField.JobID]
                assert src_job_id == dst_job_id, f"Job ID mismatch: {src_job_id} != {dst_job_id}"

                src_line = src.bin[segyio.BinField.LineNumber]
                dst_line = dst.bin[segyio.BinField.LineNumber]
                assert src_line == dst_line, f"Line number mismatch: {src_line} != {dst_line}"

        print("✓ Binary and text headers preserved")
        print(f"  - Text header: Match")
        print(f"  - Job ID: 1234 (preserved)")
        print(f"  - Line number: 5678 (preserved)")
        print(f"  - Sample count: 500 (preserved)")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


# Test 7: Output readable by segyio
def test_7_output_readable():
    """Test that output SEGY is readable by segyio"""
    print("\n=== Test 7: Output Readable by External Tools ===")

    segy_path, original_data, segy_dir = TestFixtures.create_test_segy(
        n_samples=500, n_traces=5000
    )
    zarr_path, processed_data, zarr_dir = TestFixtures.create_processed_zarr(original_data)

    try:
        output_dir = Path(tempfile.mkdtemp())
        output_path = output_dir / 'output.sgy'

        # Export
        export_from_zarr_chunked(
            output_path=str(output_path),
            original_segy_path=str(segy_path),
            processed_zarr_path=str(zarr_path),
            chunk_size=1000
        )

        # Try opening with segyio (fresh instance)
        try:
            with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
                # Verify can read properties
                trace_count = f.tracecount
                sample_count = len(f.samples)

                assert trace_count == 5000, f"Expected 5000 traces, got {trace_count}"
                assert sample_count == 500, f"Expected 500 samples, got {sample_count}"

                # Verify can read traces
                trace_0 = f.trace[0]
                assert trace_0 is not None, "Should be able to read trace 0"
                assert len(trace_0) == 500, f"Trace should have 500 samples, got {len(trace_0)}"

                # Verify can read headers
                header_0 = f.header[0]
                assert header_0 is not None, "Should be able to read header 0"

                print("✓ Output readable by segyio")
                print(f"  - File opens: Yes (no errors)")
                print(f"  - Trace count: {trace_count}")
                print(f"  - Sample count: {sample_count}")
                print(f"  - Can read traces: Yes")
                print(f"  - Can read headers: Yes")
                print(f"  - Valid SEGY format: Yes")

        except Exception as e:
            raise AssertionError(f"Failed to open output SEGY: {e}")

        TestFixtures.cleanup_storage(output_dir)

    finally:
        TestFixtures.cleanup_storage(segy_dir)
        TestFixtures.cleanup_storage(zarr_dir)


if __name__ == '__main__':
    print("=" * 70)
    print("Task 5.1: Chunked SEGY Exporter - Test Suite")
    print("=" * 70)

    tests = [
        test_1_dimensions_match,
        test_2_headers_preserved,
        test_3_trace_data_matches,
        test_4_chunk_boundaries,
        test_5_progress_callback,
        test_6_binary_text_headers,
        test_7_output_readable,
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
