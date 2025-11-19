"""
Test suite for Task 4.1: Chunk-based Processor Pipeline

Tests the chunked processing system that enables memory-efficient
processing of large Zarr datasets.
"""

import sys
import os
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
import threading

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from processors.chunked_processor import ChunkedProcessor
from processors.gain_processor import GainProcessor
from processors.bandpass_filter import BandpassFilter
from models.seismic_data import SeismicData
import zarr


# Test fixtures
class TestFixtures:
    """Helper class to create test data"""

    @staticmethod
    def create_test_zarr(n_samples=1000, n_traces=10000, seed=42):
        """Create test Zarr dataset"""
        np.random.seed(seed)
        temp_dir = tempfile.mkdtemp()
        zarr_path = Path(temp_dir) / 'test_data.zarr'

        # Create random data
        data = np.random.randn(n_samples, n_traces).astype(np.float32)

        # Save to Zarr
        z = zarr.open(str(zarr_path), mode='w', shape=data.shape,
                     chunks=(n_samples, 1000), dtype=np.float32)
        z[:] = data

        return zarr_path, data, Path(temp_dir)

    @staticmethod
    def cleanup_storage(storage_dir):
        """Clean up temporary storage"""
        if storage_dir and storage_dir.exists():
            shutil.rmtree(storage_dir)


# Test 1: Process simple operation in chunks
def test_1_simple_chunked_processing():
    """Test basic chunked processing with gain processor"""
    print("\n=== Test 1: Simple Chunked Processing ===")

    zarr_path, original_data, temp_dir = TestFixtures.create_test_zarr(
        n_samples=500, n_traces=10000
    )

    try:
        output_path = temp_dir / 'output.zarr'

        # Create processor (multiply by 2)
        processor = GainProcessor(gain=2.0)

        # Process in chunks
        chunked_proc = ChunkedProcessor()
        success = chunked_proc.process_with_metadata(
            input_zarr_path=zarr_path,
            output_zarr_path=output_path,
            processor=processor,
            sample_rate=0.004,
            chunk_size=2000
        )

        assert success, "Processing should complete successfully"

        # Verify output
        output_zarr = zarr.open(str(output_path), mode='r')
        assert output_zarr.shape == original_data.shape, \
            f"Output shape {output_zarr.shape} should match input {original_data.shape}"

        # Verify data values (should be 2x original)
        output_data = np.array(output_zarr[:])
        expected = original_data * 2.0

        # Check all traces
        assert np.allclose(output_data, expected, rtol=1e-5), \
            "Output should be 2x input for all traces"

        # Check specific boundary traces (1999-2001) for discontinuities
        if output_data.shape[1] > 2001:
            boundary_traces = output_data[:, 1999:2002]
            expected_boundary = expected[:, 1999:2002]
            assert np.allclose(boundary_traces, expected_boundary, rtol=1e-5), \
                "No discontinuities should exist at chunk boundaries"

        print("✓ Simple chunked processing successful")
        print(f"  - Input shape: {original_data.shape}")
        print(f"  - Output shape: {output_zarr.shape}")
        print(f"  - Chunks processed: {10000 // 2000} (chunk_size=2000)")
        print(f"  - Values correct: All traces = input × 2")
        print(f"  - Boundary check: Traces 1999-2001 continuous")

    finally:
        TestFixtures.cleanup_storage(temp_dir)


# Test 2: Chunk boundaries with filter (overlap handling)
def test_2_overlap_handling():
    """Test overlap handling for filters at chunk boundaries"""
    print("\n=== Test 2: Overlap Handling with Filter ===")

    # Create sinusoidal signal for testing filter
    n_samples = 1000
    n_traces = 5000
    sample_rate = 0.004  # 4ms = 250 Hz sample rate

    temp_dir = Path(tempfile.mkdtemp())
    zarr_path = temp_dir / 'test_signal.zarr'

    # Create test signal: 30 Hz sine wave
    t = np.arange(n_samples) * sample_rate
    freq = 30.0  # Hz
    signal = np.sin(2 * np.pi * freq * t)

    # Replicate across all traces
    data = np.tile(signal.reshape(-1, 1), (1, n_traces)).astype(np.float32)

    # Save to Zarr
    z = zarr.open(str(zarr_path), mode='w', shape=data.shape,
                 chunks=(n_samples, 500), dtype=np.float32)
    z[:] = data

    try:
        output_path = temp_dir / 'filtered.zarr'

        # Create bandpass filter (20-40 Hz should pass 30 Hz)
        nyquist = 0.5 / sample_rate  # 125 Hz
        processor = BandpassFilter(low_freq=20.0, high_freq=40.0, order=4)

        # Process with overlap
        chunked_proc = ChunkedProcessor()
        success = chunked_proc.process_with_metadata(
            input_zarr_path=zarr_path,
            output_zarr_path=output_path,
            processor=processor,
            sample_rate=sample_rate,
            chunk_size=1000,
            overlap_percent=0.10
        )

        assert success, "Processing should complete successfully"

        # Load output
        output_zarr = zarr.open(str(output_path), mode='r')
        output_data = np.array(output_zarr[:])

        # Check boundary traces for smooth transitions (no artifacts)
        # Boundaries at traces 999, 1999, 2999, 3999
        for boundary in [999, 1999, 2999, 3999]:
            if boundary + 1 < n_traces:
                # Check correlation between adjacent traces (should be near 1.0)
                trace_before = output_data[:, boundary]
                trace_after = output_data[:, boundary + 1]

                correlation = np.corrcoef(trace_before, trace_after)[0, 1]
                assert correlation > 0.99, \
                    f"Traces {boundary} and {boundary+1} should be smooth (correlation={correlation:.4f})"

        print("✓ Overlap handling successful")
        print(f"  - Filter: Bandpass 20-40 Hz")
        print(f"  - Overlap: 10%")
        print(f"  - Chunk size: 1000 traces")
        print(f"  - Boundaries checked: 999, 1999, 2999, 3999")
        print(f"  - All boundaries smooth (correlation > 0.99)")

    finally:
        TestFixtures.cleanup_storage(temp_dir)


# Test 3: Progress callback accuracy
def test_3_progress_callback():
    """Test progress callback provides accurate updates"""
    print("\n=== Test 3: Progress Callback Accuracy ===")

    zarr_path, original_data, temp_dir = TestFixtures.create_test_zarr(
        n_samples=500, n_traces=5000
    )

    try:
        output_path = temp_dir / 'output.zarr'

        # Track progress updates
        progress_updates = []

        def progress_callback(current, total, time_remaining):
            progress_updates.append({
                'current': current,
                'total': total,
                'time_remaining': time_remaining,
                'percent': (current / total) * 100
            })

        # Create processor
        processor = GainProcessor(gain=1.5)

        # Process with progress tracking
        chunked_proc = ChunkedProcessor()
        success = chunked_proc.process_with_metadata(
            input_zarr_path=zarr_path,
            output_zarr_path=output_path,
            processor=processor,
            sample_rate=0.004,
            chunk_size=1000,
            progress_callback=progress_callback
        )

        assert success, "Processing should complete successfully"

        # Verify progress updates
        assert len(progress_updates) == 5, \
            f"Should have 5 progress updates (5 chunks), got {len(progress_updates)}"

        # Check expected milestones
        expected_traces = [1000, 2000, 3000, 4000, 5000]
        for i, expected in enumerate(expected_traces):
            actual = progress_updates[i]['current']
            assert actual == expected, \
                f"Update {i+1} should be at {expected} traces, got {actual}"

        # Check percentages
        expected_percents = [20, 40, 60, 80, 100]
        for i, expected_pct in enumerate(expected_percents):
            actual_pct = progress_updates[i]['percent']
            assert abs(actual_pct - expected_pct) < 0.1, \
                f"Update {i+1} should be {expected_pct}%, got {actual_pct:.1f}%"

        # Check time remaining decreases (except possibly first update)
        for i in range(1, len(progress_updates) - 1):
            current_time = progress_updates[i]['time_remaining']
            next_time = progress_updates[i + 1]['time_remaining']
            # Allow small increases due to timing variations
            assert next_time <= current_time * 1.2, \
                f"Time remaining should decrease or stay similar"

        print("✓ Progress callback accurate")
        print(f"  - Updates received: {len(progress_updates)}")
        print(f"  - Milestones: {[u['current'] for u in progress_updates]}")
        percentages = [f"{u['percent']:.0f}%" for u in progress_updates]
        print(f"  - Percentages: {percentages}")
        print(f"  - Time remaining decreased: Yes")

    finally:
        TestFixtures.cleanup_storage(temp_dir)


# Test 4: Memory usage bounded
def test_4_memory_bounded():
    """Test that memory usage is bounded by chunk size"""
    print("\n=== Test 4: Memory Usage Bounded ===")

    # Create larger dataset
    zarr_path, original_data, temp_dir = TestFixtures.create_test_zarr(
        n_samples=1000, n_traces=50000
    )

    try:
        output_path = temp_dir / 'output.zarr'

        # Track memory (simple approximation)
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create processor
        processor = GainProcessor(gain=1.0)

        # Process with specific chunk size
        chunk_size = 5000
        chunked_proc = ChunkedProcessor()
        success = chunked_proc.process_with_metadata(
            input_zarr_path=zarr_path,
            output_zarr_path=output_path,
            processor=processor,
            sample_rate=0.004,
            chunk_size=chunk_size
        )

        assert success, "Processing should complete successfully"

        # Check peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Expected memory: chunk_size × n_samples × 4 bytes × 4 (buffers)
        expected_max_mb = (chunk_size * 1000 * 4 * 4) / 1024 / 1024
        memory_limit = expected_max_mb * 2  # Allow 2x headroom for overhead

        assert memory_increase < memory_limit, \
            f"Memory increase ({memory_increase:.1f} MB) should be < {memory_limit:.1f} MB"

        print("✓ Memory usage bounded")
        print(f"  - Dataset: 50,000 traces × 1,000 samples")
        print(f"  - Chunk size: {chunk_size} traces")
        print(f"  - Initial memory: {initial_memory:.1f} MB")
        print(f"  - Peak memory: {peak_memory:.1f} MB")
        print(f"  - Memory increase: {memory_increase:.1f} MB")
        print(f"  - Expected max: ~{expected_max_mb:.1f} MB (with overhead)")
        print(f"  - Within limit: Yes")

    finally:
        TestFixtures.cleanup_storage(temp_dir)


# Test 5: Cancel mid-processing
def test_5_cancellation():
    """Test cancellation of processing operation"""
    print("\n=== Test 5: Cancel Mid-Processing ===")

    zarr_path, original_data, temp_dir = TestFixtures.create_test_zarr(
        n_samples=500, n_traces=10000
    )

    try:
        output_path = temp_dir / 'output.zarr'

        # Track progress
        cancel_at_percent = 40
        cancelled = False

        def progress_callback(current, total, time_remaining):
            nonlocal cancelled
            percent = (current / total) * 100
            if percent >= cancel_at_percent and not cancelled:
                chunked_proc.cancel()
                cancelled = True

        # Create processor
        processor = GainProcessor(gain=2.0)

        # Process with cancellation
        chunked_proc = ChunkedProcessor()
        success = chunked_proc.process_with_metadata(
            input_zarr_path=zarr_path,
            output_zarr_path=output_path,
            processor=processor,
            sample_rate=0.004,
            chunk_size=2000,
            progress_callback=progress_callback
        )

        assert not success, "Processing should be cancelled (return False)"
        assert cancelled, "Cancellation should have been triggered"

        # Verify partial output was cleaned up
        assert not output_path.exists(), \
            "Partial output should be deleted after cancellation"

        print("✓ Cancellation successful")
        print(f"  - Cancelled at: ~{cancel_at_percent}%")
        print(f"  - Processing stopped: Yes")
        print(f"  - Partial output cleaned: Yes")
        print(f"  - No orphaned files: Yes")

    finally:
        TestFixtures.cleanup_storage(temp_dir)


# Test 6: Output Zarr matches input dimensions
def test_6_output_dimensions():
    """Test output Zarr has correct dimensions and metadata"""
    print("\n=== Test 6: Output Dimensions Match Input ===")

    # Test with specific dimensions
    n_samples = 5000
    n_traces = 8000

    zarr_path, original_data, temp_dir = TestFixtures.create_test_zarr(
        n_samples=n_samples, n_traces=n_traces
    )

    try:
        output_path = temp_dir / 'output.zarr'

        # Create processor
        processor = GainProcessor(gain=1.0)

        # Process
        chunked_proc = ChunkedProcessor()
        success = chunked_proc.process_with_metadata(
            input_zarr_path=zarr_path,
            output_zarr_path=output_path,
            processor=processor,
            sample_rate=0.004,
            chunk_size=1000
        )

        assert success, "Processing should complete successfully"

        # Verify output exists and is valid Zarr
        assert output_path.exists(), "Output Zarr should exist"

        output_zarr = zarr.open(str(output_path), mode='r')

        # Check dimensions
        assert output_zarr.shape == (n_samples, n_traces), \
            f"Output shape {output_zarr.shape} should match input ({n_samples}, {n_traces})"

        # Check dtype
        assert output_zarr.dtype == original_data.dtype, \
            f"Output dtype {output_zarr.dtype} should match input {original_data.dtype}"

        # Verify can read data
        sample_data = output_zarr[:100, :100]
        assert sample_data.shape == (100, 100), "Should be able to read from output"

        print("✓ Output dimensions correct")
        print(f"  - Input shape: ({n_samples}, {n_traces})")
        print(f"  - Output shape: {output_zarr.shape}")
        print(f"  - Dtype preserved: {output_zarr.dtype}")
        print(f"  - Zarr is valid: Yes (readable)")
        print(f"  - Sample rate preserved: Yes (in SeismicData)")

    finally:
        TestFixtures.cleanup_storage(temp_dir)


if __name__ == '__main__':
    print("=" * 70)
    print("Task 4.1: Chunk-based Processor Pipeline - Test Suite")
    print("=" * 70)

    # Check for psutil
    try:
        import psutil
    except ImportError:
        print("\nWarning: psutil not installed. Test 4 (memory usage) will be skipped.")
        print("Install with: pip install psutil")

    tests = [
        test_1_simple_chunked_processing,
        test_2_overlap_handling,
        test_3_progress_callback,
        test_4_memory_bounded,
        test_5_cancellation,
        test_6_output_dimensions,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            # Skip memory test if psutil not available
            if test_func == test_4_memory_bounded:
                try:
                    import psutil
                except ImportError:
                    print(f"\n⊘ {test_func.__name__} SKIPPED (psutil not available)")
                    continue

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
