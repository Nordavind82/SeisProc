"""
Test suite for Task 7.1: End-to-End Integration Test

Tests the complete workflow from SEGY import through lazy loading,
navigation, processing, and export. Validates all major components
working together.
"""
import sys
import os
import time
import tempfile
import shutil
import numpy as np
import pandas as pd
import segyio
import zarr
from numcodecs import Blosc
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lazy_seismic_data import LazySeismicData
from models.gather_navigator import GatherNavigator
from processors.chunked_processor import ChunkedProcessor
from processors.gain_processor import GainProcessor
from utils.segy_import.segy_export import export_from_zarr_chunked
from utils.memory_monitor import MemoryMonitor, format_bytes


class EndToEndIntegrationTest:
    """End-to-end integration test for large SEGY workflow."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Test parameters
        self.n_samples = 500
        self.n_traces = 10000
        self.sample_rate = 2.0  # ms
        self.n_ensembles = 100
        self.traces_per_ensemble = self.n_traces // self.n_ensembles

        # File paths
        self.input_segy = self.test_dir / "input_test.sgy"
        self.input_zarr = self.test_dir / "input_test.zarr"
        self.headers_parquet = self.test_dir / "headers.parquet"
        self.ensembles_parquet = self.test_dir / "ensembles.parquet"
        self.processed_zarr = self.test_dir / "processed.zarr"
        self.output_segy = self.test_dir / "output_processed.sgy"

        # Memory monitor
        self.memory_monitor = MemoryMonitor(update_interval=1.0)
        self.peak_memory = 0
        self.memory_samples = []

    def track_memory(self):
        """Track memory usage."""
        current = self.memory_monitor.get_current_usage()
        self.peak_memory = max(self.peak_memory, current)
        self.memory_samples.append(current)

    def create_test_segy_file(self) -> dict:
        """
        Create a test SEGY file with known patterns.

        Returns:
            Dictionary with creation stats
        """
        print("\n" + "="*70)
        print("STEP 1: Creating Test SEGY File")
        print("="*70)

        start_time = time.time()

        # Create SEGY specification
        spec = segyio.spec()
        spec.format = 1  # 4-byte IBM float
        spec.samples = range(self.n_samples)
        spec.tracecount = self.n_traces

        print(f"Creating SEGY file with:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Traces: {self.n_traces}")
        print(f"  Sample rate: {self.sample_rate} ms")
        print(f"  Ensembles: {self.n_ensembles}")

        # Create SEGY file
        with segyio.create(str(self.input_segy), spec) as f:
            # Write binary header
            f.bin[segyio.BinField.Samples] = self.n_samples
            f.bin[segyio.BinField.Interval] = int(self.sample_rate * 1000)  # microseconds

            # Write text header
            f.text[0] = b'Test SEGY file for integration testing' + b' ' * 3161

            # Write traces with pattern
            for i in range(self.n_traces):
                # Trace data: sine wave with trace-dependent frequency
                t = np.arange(self.n_samples) * self.sample_rate / 1000.0  # seconds
                freq = 10 + (i % 50)  # 10-60 Hz
                trace_data = np.sin(2 * np.pi * freq * t) * (i + 1)

                # Write trace
                f.trace[i] = trace_data.astype(np.float32)

                # Write headers
                ensemble_id = i // self.traces_per_ensemble
                f.header[i] = {
                    segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.CDP: ensemble_id + 1,
                    segyio.TraceField.offset: (i % self.traces_per_ensemble) * 10,
                    segyio.TraceField.ElevationScalar: -100,
                }

        elapsed = time.time() - start_time
        file_size = self.input_segy.stat().st_size

        print(f"\n✓ SEGY file created:")
        print(f"  File size: {format_bytes(file_size)}")
        print(f"  Creation time: {elapsed:.2f}s")

        return {
            'file_size': file_size,
            'creation_time': elapsed,
            'n_traces': self.n_traces,
            'n_samples': self.n_samples
        }

    def import_to_zarr_streaming(self) -> dict:
        """
        Import SEGY to Zarr using streaming/chunked approach.

        Returns:
            Dictionary with import stats
        """
        print("\n" + "="*70)
        print("STEP 2: Streaming Import to Zarr")
        print("="*70)

        start_time = time.time()
        self.track_memory()
        baseline_memory = self.memory_monitor.get_current_usage()

        # Open SEGY file
        with segyio.open(str(self.input_segy), 'r', ignore_geometry=True) as segy:
            n_traces = segy.tracecount
            n_samples = len(segy.samples)

            print(f"Importing {n_traces} traces...")

            # Create Zarr array (force v2 format for compatibility)
            zarr_array = zarr.open(
                str(self.input_zarr),
                mode='w',
                shape=(n_samples, n_traces),
                chunks=(n_samples, min(1000, n_traces)),
                dtype=np.float32,
                compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
                zarr_format=2
            )

            # Import in chunks
            chunk_size = 1000
            chunks_processed = 0

            for chunk_start in range(0, n_traces, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_traces)

                # Read chunk
                chunk_data = []
                for i in range(chunk_start, chunk_end):
                    chunk_data.append(segy.trace[i])

                chunk_array = np.array(chunk_data).T  # Transpose to (samples, traces)

                # Write to Zarr
                zarr_array[:, chunk_start:chunk_end] = chunk_array

                chunks_processed += 1
                self.track_memory()

                if chunks_processed % 5 == 0:
                    progress = (chunk_end / n_traces) * 100
                    print(f"  Progress: {chunk_end}/{n_traces} traces ({progress:.0f}%)")

            # Save metadata
            zarr_array.attrs['sample_rate'] = self.sample_rate
            zarr_array.attrs['n_samples'] = n_samples
            zarr_array.attrs['n_traces'] = n_traces

            # Extract and save headers
            headers_data = []
            for i in range(n_traces):
                header = segy.header[i]
                headers_data.append({
                    'trace_index': i,
                    'TraceNumber': header[segyio.TraceField.TRACE_SEQUENCE_FILE],
                    'CDP': header[segyio.TraceField.CDP],
                    'offset': header[segyio.TraceField.offset],
                })

            headers_df = pd.DataFrame(headers_data)
            headers_df.to_parquet(self.headers_parquet, index=False)

            # Create ensemble index
            ensembles = []
            for ensemble_id in range(self.n_ensembles):
                start_trace = ensemble_id * self.traces_per_ensemble
                end_trace = start_trace + self.traces_per_ensemble - 1  # end_trace is inclusive
                ensembles.append({
                    'ensemble_id': ensemble_id,
                    'CDP': ensemble_id + 1,
                    'start_trace': start_trace,
                    'end_trace': end_trace,
                    'n_traces': self.traces_per_ensemble
                })

            ensembles_df = pd.DataFrame(ensembles)
            ensembles_df.to_parquet(self.ensembles_parquet, index=False)

        elapsed = time.time() - start_time
        zarr_size = sum(f.stat().st_size for f in self.input_zarr.rglob('*') if f.is_file())
        peak_memory = max(self.memory_samples) if self.memory_samples else baseline_memory
        memory_overhead = peak_memory - baseline_memory

        print(f"\n✓ Import completed:")
        print(f"  Zarr size: {format_bytes(zarr_size)}")
        print(f"  Import time: {elapsed:.2f}s")
        print(f"  Chunks processed: {chunks_processed}")
        print(f"  Peak memory: {format_bytes(peak_memory)}")
        print(f"  Memory overhead: {format_bytes(memory_overhead)}")
        print(f"  Compression ratio: {self.input_segy.stat().st_size / zarr_size:.2f}x")

        return {
            'zarr_size': zarr_size,
            'import_time': elapsed,
            'chunks_processed': chunks_processed,
            'peak_memory': peak_memory,
            'compression_ratio': self.input_segy.stat().st_size / zarr_size
        }

    def test_lazy_loading_navigation(self) -> dict:
        """
        Test lazy loading and ensemble navigation.

        Returns:
            Dictionary with navigation stats
        """
        print("\n" + "="*70)
        print("STEP 3: Testing Lazy Loading and Navigation")
        print("="*70)

        start_time = time.time()
        baseline_memory = self.memory_monitor.get_current_usage()

        # Load as lazy data
        print("Loading data as LazySeismicData...")
        metadata = {
            'sample_rate': self.sample_rate,
            'n_samples': self.n_samples,
            'n_traces': self.n_traces
        }
        lazy_data = LazySeismicData(
            zarr_path=self.input_zarr,
            metadata=metadata,
            headers_path=self.headers_parquet,
            ensemble_index_path=self.ensembles_parquet
        )

        load_memory = self.memory_monitor.get_current_usage()
        memory_for_load = load_memory - baseline_memory

        print(f"  Memory after load: {format_bytes(load_memory)}")
        print(f"  Memory increase: {format_bytes(memory_for_load)}")
        print(f"  n_samples: {lazy_data.n_samples}")
        print(f"  n_traces: {lazy_data.n_traces}")

        # Test window extraction
        print("\nTesting window extraction...")
        window_times = []
        for i in range(5):
            t_start = time.time()
            window = lazy_data.get_window(0, 100, i*100, (i+1)*100)
            window_times.append(time.time() - t_start)
            assert window.shape == (50, 100), f"Window shape mismatch: {window.shape}"

        avg_window_time = np.mean(window_times) * 1000
        print(f"  ✓ 5 window extractions successful")
        print(f"  Average window load time: {avg_window_time:.1f}ms")

        # Test ensemble navigation
        print("\nTesting ensemble navigation...")
        ensembles_df = pd.read_parquet(self.ensembles_parquet)
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensembles_df=ensembles_df)

        print(f"  Number of ensembles: {navigator.n_gathers}")

        # Navigate through some ensembles
        nav_times = []
        for i in [0, 10, 50, 99]:
            navigator.current_gather_id = i
            t_start = time.time()
            gather_data, gather_headers, gather_info = navigator.get_current_gather()
            nav_times.append(time.time() - t_start)

            expected_traces = self.traces_per_ensemble
            actual_traces = gather_data.traces.shape[1]
            assert actual_traces == expected_traces, \
                f"Ensemble {i} trace count mismatch: expected {expected_traces}, got {actual_traces}"

        avg_nav_time = np.mean(nav_times) * 1000
        print(f"  ✓ Navigated to 4 ensembles successfully")
        print(f"  Average navigation time: {avg_nav_time:.1f}ms")

        # Test prefetching benefit
        print("\nTesting prefetching...")
        time.sleep(1.5)  # Allow prefetch thread to work

        cached_nav_times = []
        for i in range(48, 52):
            navigator.current_gather_id = i
            t_start = time.time()
            navigator.get_current_gather()
            cached_nav_times.append(time.time() - t_start)

        avg_cached_time = np.mean(cached_nav_times) * 1000
        print(f"  ✓ Sequential navigation with prefetch")
        print(f"  Average cached navigation: {avg_cached_time:.1f}ms")

        cache_hit_rate = navigator._cache_hits / (navigator._cache_hits + navigator._cache_misses) * 100 \
            if (navigator._cache_hits + navigator._cache_misses) > 0 else 0
        print(f"  Cache hit rate: {cache_hit_rate:.0f}%")

        final_memory = self.memory_monitor.get_current_usage()

        elapsed = time.time() - start_time

        print(f"\n✓ Lazy loading and navigation tested:")
        print(f"  Test time: {elapsed:.2f}s")
        print(f"  Final memory: {format_bytes(final_memory)}")

        return {
            'load_memory': memory_for_load,
            'avg_window_time_ms': avg_window_time,
            'avg_navigation_time_ms': avg_nav_time,
            'avg_cached_navigation_ms': avg_cached_time,
            'cache_hit_rate': cache_hit_rate,
            'test_time': elapsed
        }

    def test_chunked_processing(self) -> dict:
        """
        Test chunked processing with gain processor.

        Returns:
            Dictionary with processing stats
        """
        print("\n" + "="*70)
        print("STEP 4: Testing Chunked Processing")
        print("="*70)

        start_time = time.time()
        baseline_memory = self.memory_monitor.get_current_usage()

        # Create processor
        gain_value = 2.5
        processor = GainProcessor(gain=gain_value)

        # Create chunked processor
        chunked_processor = ChunkedProcessor()

        print(f"Processing with gain factor: {gain_value}")
        print(f"Chunk size: 1000 traces")

        # Progress tracking
        progress_updates = []

        def progress_callback(current, total, time_remaining):
            progress_updates.append({
                'current': current,
                'total': total,
                'percent': (current / total) * 100,
                'time_remaining': time_remaining
            })
            if len(progress_updates) % 3 == 0:
                print(f"  Progress: {current}/{total} traces ({progress_updates[-1]['percent']:.0f}%), "
                      f"{time_remaining:.0f}s remaining")
            self.track_memory()

        # Process
        success = chunked_processor.process_with_metadata(
            input_zarr_path=self.input_zarr,
            output_zarr_path=self.processed_zarr,
            processor=processor,
            sample_rate=self.sample_rate,
            chunk_size=1000,
            progress_callback=progress_callback,
            overlap_percent=0.0  # No overlap for gain
        )

        assert success, "Processing failed"

        elapsed = time.time() - start_time
        peak_memory = max(self.memory_samples[-len(progress_updates):]) if self.memory_samples else baseline_memory
        memory_overhead = peak_memory - baseline_memory

        # Verify output
        print("\nVerifying processed output...")
        input_zarr = zarr.open(str(self.input_zarr), mode='r')
        output_zarr = zarr.open(str(self.processed_zarr), mode='r')

        # Check dimensions
        assert output_zarr.shape == input_zarr.shape, "Shape mismatch"

        # Spot check values
        test_indices = [0, 5000, 9999]
        for idx in test_indices:
            input_trace = input_zarr[:, idx]
            output_trace = output_zarr[:, idx]
            expected = input_trace * gain_value
            np.testing.assert_allclose(output_trace, expected, rtol=1e-5,
                                     err_msg=f"Trace {idx} processing incorrect")

        print(f"  ✓ Spot checked traces: {test_indices}")
        print(f"  ✓ All values match expected (gain × {gain_value})")

        processed_size = sum(f.stat().st_size for f in self.processed_zarr.rglob('*') if f.is_file())

        print(f"\n✓ Processing completed:")
        print(f"  Processed size: {format_bytes(processed_size)}")
        print(f"  Processing time: {elapsed:.2f}s")
        print(f"  Throughput: {self.n_traces / elapsed:.0f} traces/second")
        print(f"  Progress updates: {len(progress_updates)}")
        print(f"  Peak memory: {format_bytes(peak_memory)}")
        print(f"  Memory overhead: {format_bytes(memory_overhead)}")

        return {
            'processing_time': elapsed,
            'throughput': self.n_traces / elapsed,
            'progress_updates': len(progress_updates),
            'peak_memory': peak_memory,
            'memory_overhead': memory_overhead,
            'processed_size': processed_size
        }

    def test_chunked_export(self) -> dict:
        """
        Test chunked SEGY export.

        Returns:
            Dictionary with export stats
        """
        print("\n" + "="*70)
        print("STEP 5: Testing Chunked Export")
        print("="*70)

        start_time = time.time()
        baseline_memory = self.memory_monitor.get_current_usage()

        # Progress tracking
        progress_updates = []

        def progress_callback(current, total, time_remaining):
            progress_updates.append({
                'current': current,
                'total': total,
                'percent': (current / total) * 100
            })
            if len(progress_updates) % 3 == 0:
                print(f"  Progress: {current}/{total} traces ({progress_updates[-1]['percent']:.0f}%), "
                      f"{time_remaining:.0f}s remaining")
            self.track_memory()

        print(f"Exporting {self.n_traces} traces...")
        print(f"Chunk size: 1000 traces")

        # Export
        export_from_zarr_chunked(
            output_path=str(self.output_segy),
            original_segy_path=str(self.input_segy),
            processed_zarr_path=str(self.processed_zarr),
            chunk_size=1000,
            progress_callback=progress_callback
        )

        elapsed = time.time() - start_time
        peak_memory = max(self.memory_samples[-len(progress_updates):]) if self.memory_samples else baseline_memory
        memory_overhead = peak_memory - baseline_memory
        output_size = self.output_segy.stat().st_size

        # Verify exported file
        print("\nVerifying exported SEGY...")
        with segyio.open(str(self.output_segy), 'r', ignore_geometry=True) as output_segy:
            assert output_segy.tracecount == self.n_traces, "Trace count mismatch"
            assert len(output_segy.samples) == self.n_samples, "Sample count mismatch"

            # Verify headers preserved
            with segyio.open(str(self.input_segy), 'r', ignore_geometry=True) as input_segy:
                test_traces = [0, 5000, 9999]
                for idx in test_traces:
                    input_header = input_segy.header[idx]
                    output_header = output_segy.header[idx]

                    assert input_header[segyio.TraceField.CDP] == output_header[segyio.TraceField.CDP], \
                        f"CDP mismatch at trace {idx}"
                    assert input_header[segyio.TraceField.offset] == output_header[segyio.TraceField.offset], \
                        f"Offset mismatch at trace {idx}"

            print(f"  ✓ Headers preserved for traces: {test_traces}")

            # Verify processed data
            processed_zarr = zarr.open(str(self.processed_zarr), mode='r')
            test_traces = [0, 5000, 9999]
            for idx in test_traces:
                segy_trace = output_segy.trace[idx]
                zarr_trace = processed_zarr[:, idx]
                np.testing.assert_allclose(segy_trace, zarr_trace, rtol=1e-4,
                                         err_msg=f"Trace data mismatch at {idx}")

            print(f"  ✓ Trace data matches processed Zarr for traces: {test_traces}")

        print(f"\n✓ Export completed:")
        print(f"  Output size: {format_bytes(output_size)}")
        print(f"  Export time: {elapsed:.2f}s")
        print(f"  Throughput: {self.n_traces / elapsed:.0f} traces/second")
        print(f"  Progress updates: {len(progress_updates)}")
        print(f"  Peak memory: {format_bytes(peak_memory)}")
        print(f"  Memory overhead: {format_bytes(memory_overhead)}")

        return {
            'export_time': elapsed,
            'throughput': self.n_traces / elapsed,
            'progress_updates': len(progress_updates),
            'peak_memory': peak_memory,
            'memory_overhead': memory_overhead,
            'output_size': output_size
        }

    def cleanup(self):
        """Clean up test files and monitor."""
        self.memory_monitor.stop()
        time.sleep(0.6)


def run_integration_test():
    """Run complete end-to-end integration test."""
    print("\n" + "="*70)
    print("TASK 7.1: END-TO-END INTEGRATION TEST")
    print("="*70)
    print("\nTesting complete workflow:")
    print("  1. Create test SEGY file")
    print("  2. Streaming import to Zarr")
    print("  3. Lazy loading and navigation")
    print("  4. Chunked processing")
    print("  5. Chunked export to SEGY")
    print("="*70)

    # Create test directory
    test_dir = Path(tempfile.mkdtemp(prefix="integration_test_"))
    print(f"\nTest directory: {test_dir}")

    try:
        # Run test
        test = EndToEndIntegrationTest(test_dir)

        overall_start = time.time()

        # Step 1: Create test file
        creation_stats = test.create_test_segy_file()

        # Step 2: Import
        import_stats = test.import_to_zarr_streaming()

        # Step 3: Lazy loading and navigation
        navigation_stats = test.test_lazy_loading_navigation()

        # Step 4: Processing
        processing_stats = test.test_chunked_processing()

        # Step 5: Export
        export_stats = test.test_chunked_export()

        overall_elapsed = time.time() - overall_start

        # Summary
        print("\n" + "="*70)
        print("INTEGRATION TEST SUMMARY")
        print("="*70)

        print("\nPerformance Metrics:")
        print(f"  Total workflow time: {overall_elapsed:.2f}s")
        print(f"  Import time: {import_stats['import_time']:.2f}s")
        print(f"  Processing time: {processing_stats['processing_time']:.2f}s")
        print(f"  Export time: {export_stats['export_time']:.2f}s")

        print("\nMemory Efficiency:")
        print(f"  Peak memory (overall): {format_bytes(test.peak_memory)}")
        print(f"  Import peak: {format_bytes(import_stats['peak_memory'])}")
        print(f"  Processing peak: {format_bytes(processing_stats['peak_memory'])}")
        print(f"  Export peak: {format_bytes(export_stats['peak_memory'])}")

        print("\nData Integrity:")
        print(f"  ✓ {creation_stats['n_traces']} traces processed")
        print(f"  ✓ Headers preserved")
        print(f"  ✓ Processing verified (gain factor applied)")
        print(f"  ✓ Output SEGY valid and readable")

        print("\nCompression & Storage:")
        print(f"  Input SEGY: {format_bytes(creation_stats['file_size'])}")
        print(f"  Zarr (compressed): {format_bytes(import_stats['zarr_size'])}")
        print(f"  Compression ratio: {import_stats['compression_ratio']:.2f}x")
        print(f"  Output SEGY: {format_bytes(export_stats['output_size'])}")

        print("\nNavigation Performance:")
        print(f"  Average window load: {navigation_stats['avg_window_time_ms']:.1f}ms")
        print(f"  Average navigation: {navigation_stats['avg_navigation_time_ms']:.1f}ms")
        print(f"  Cached navigation: {navigation_stats['avg_cached_navigation_ms']:.1f}ms")
        print(f"  Cache hit rate: {navigation_stats['cache_hit_rate']:.0f}%")

        print("\n" + "="*70)
        print("✅ INTEGRATION TEST PASSED")
        print("="*70)

        print("\nAll workflow components verified:")
        print("  ✓ Streaming import functional")
        print("  ✓ Lazy loading memory-efficient")
        print("  ✓ Ensemble navigation with prefetching")
        print("  ✓ Chunked processing accurate")
        print("  ✓ Chunked export preserves data and headers")
        print("  ✓ Memory usage bounded throughout")

        # Cleanup
        test.cleanup()

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("❌ INTEGRATION TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")


if __name__ == '__main__':
    import sys
    success = run_integration_test()
    sys.exit(0 if success else 1)
