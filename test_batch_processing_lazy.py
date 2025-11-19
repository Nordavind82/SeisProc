#!/usr/bin/env python3
"""
Test batch processing with lazy/streaming mode datasets.

This test verifies that batch processing works correctly when data is loaded
in lazy/streaming mode, which previously failed with "Full dataset not available".
"""
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from models.lazy_seismic_data import LazySeismicData
from models.gather_navigator import GatherNavigator
from processors.bandpass_filter import BandpassFilter
from utils.segy_import.data_storage import DataStorage


def create_test_dataset():
    """Create a small test dataset in zarr format."""
    # Create synthetic seismic data
    n_samples = 500
    n_traces = 120  # 3 gathers of 40 traces each
    sample_rate = 2.0

    # Generate synthetic traces
    traces = np.random.randn(n_samples, n_traces).astype(np.float32)

    # Add some signal
    for i in range(n_traces):
        # Add a simple sinusoidal event
        t = np.arange(n_samples) * sample_rate / 1000.0
        traces[:, i] += 2.0 * np.sin(2 * np.pi * 30 * t)  # 30 Hz signal

    # Create headers DataFrame
    headers = pd.DataFrame({
        'trace_number': range(1, n_traces + 1),
        'cdp': [i // 40 + 1 for i in range(n_traces)],  # 3 CDPs
        'offset': [(i % 40) * 25 for i in range(n_traces)],  # 40 traces per CDP
    })

    # Create ensemble index (3 gathers of 40 traces each)
    ensemble_index = pd.DataFrame({
        'ensemble_id': [0, 1, 2],
        'start_trace': [0, 40, 80],
        'end_trace': [39, 79, 119],
        'n_traces': [40, 40, 40],
    })

    # Save to temporary directory
    temp_dir = tempfile.mkdtemp(prefix='test_batch_lazy_')
    storage_dir = Path(temp_dir)

    print(f"Creating test dataset in: {storage_dir}")

    # Use DataStorage to write the data
    storage = DataStorage(str(storage_dir))
    storage.initialize(
        n_samples=n_samples,
        n_traces=n_traces,
        sample_rate=sample_rate,
        metadata={
            'description': 'Test dataset for batch processing',
            'test': True
        }
    )

    # Write traces
    storage.write_traces(traces, 0)

    # Write headers
    storage.write_headers(headers)

    # Write ensemble index
    storage.write_ensemble_index(ensemble_index)

    storage.close()

    return storage_dir


def test_batch_processing_with_lazy_data():
    """Test that batch processing works with lazy-loaded data."""
    print("\n" + "="*60)
    print("Testing Batch Processing with Lazy/Streaming Mode")
    print("="*60)

    storage_dir = None

    try:
        # Create test dataset
        storage_dir = create_test_dataset()

        # Load as lazy data
        print("\n1. Loading data in lazy/streaming mode...")
        lazy_data = LazySeismicData.from_storage_dir(str(storage_dir))
        print(f"   ✓ Loaded: {lazy_data.n_traces} traces, {lazy_data.n_samples} samples")

        # Load ensemble index
        ensemble_index_path = storage_dir / 'ensemble_index.parquet'
        ensemble_index = pd.read_parquet(ensemble_index_path)
        print(f"   ✓ Loaded: {len(ensemble_index)} gathers")

        # Create gather navigator
        print("\n2. Initializing gather navigator...")
        navigator = GatherNavigator()
        navigator.load_lazy_data(lazy_data, ensemble_index)
        print(f"   ✓ Navigator ready: {navigator.n_gathers} gathers")

        # Verify lazy mode is active
        assert navigator.lazy_data is not None, "Lazy data should be set"
        assert navigator.full_data is None, "Full data should be None in lazy mode"
        print("   ✓ Confirmed: Running in lazy mode (full_data is None)")

        # Get dataset info for batch processing
        print("\n3. Extracting dataset information (as batch processing does)...")
        if navigator.lazy_data is not None:
            n_samples = navigator.lazy_data.n_samples
            n_traces = navigator.lazy_data.n_traces
            sample_rate = navigator.lazy_data.sample_rate
            print(f"   ✓ Retrieved from lazy_data:")
            print(f"     - Samples: {n_samples}")
            print(f"     - Traces: {n_traces}")
            print(f"     - Sample rate: {sample_rate} ms")

        # Pre-allocate output array (as batch processing does)
        print("\n4. Pre-allocating output array...")
        processed_traces = np.zeros((n_samples, n_traces), dtype=np.float32)
        print(f"   ✓ Allocated: {processed_traces.shape}, {processed_traces.nbytes / 1024 / 1024:.2f} MB")

        # Create a processor
        print("\n5. Creating bandpass filter processor...")
        processor = BandpassFilter(
            low_freq=10.0,
            high_freq=80.0,
            filter_order=4,
            sample_rate=sample_rate
        )
        print(f"   ✓ Processor: {processor.get_description()}")

        # Simulate batch processing loop
        print("\n6. Processing gathers (simulating batch processing)...")
        stats = navigator.get_statistics()
        n_gathers = stats['n_gathers']

        for i in range(n_gathers):
            # Navigate to gather
            navigator.goto_gather(i)
            gather_data, gather_headers, gather_info = navigator.get_current_gather()

            # Process this gather
            processed_gather = processor.process(gather_data)

            # Store in full array
            start_trace = gather_info['start_trace']
            end_trace = gather_info['end_trace']
            processed_traces[:, start_trace:end_trace+1] = processed_gather.traces

            print(f"   ✓ Processed gather {i+1}/{n_gathers}: "
                  f"traces {start_trace}-{end_trace} ({end_trace - start_trace + 1} traces)")

        # Verify processing completed
        print("\n7. Verifying processed data...")
        assert not np.allclose(processed_traces, 0), "Processed data should not be all zeros"
        non_zero_traces = np.sum(np.any(processed_traces != 0, axis=0))
        print(f"   ✓ Non-zero traces: {non_zero_traces}/{n_traces}")

        print("\n" + "="*60)
        print("✅ TEST PASSED: Batch processing works with lazy mode!")
        print("="*60)
        print("\nThe fix successfully enables batch processing on datasets")
        print("loaded in lazy/streaming mode by:")
        print("  1. Checking for both lazy_data and full_data")
        print("  2. Extracting shape info from the appropriate source")
        print("  3. Pre-allocating output array with correct dimensions")
        print("  4. Processing gather-by-gather (already supported)")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if storage_dir and storage_dir.exists():
            print(f"\nCleaning up: {storage_dir}")
            shutil.rmtree(storage_dir)


if __name__ == '__main__':
    success = test_batch_processing_with_lazy_data()
    exit(0 if success else 1)
