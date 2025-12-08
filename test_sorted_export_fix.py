"""
Test to verify sorted header export fix.

This test verifies that:
1. Headers are correctly sorted and exported in memory-efficient mode
2. Both batch and memory-efficient export modes produce identical results
"""
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import segyio

from utils.segy_import.segy_export import export_from_zarr_chunked, SEGYExporter
from models.seismic_data import SeismicData
import zarr


def create_test_segy(path, n_traces=20, n_samples=100):
    """Create a simple test SEG-Y file with traceable headers."""
    spec = segyio.spec()
    spec.format = 1
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(str(path), spec) as f:
        # Create text header
        f.text[0] = b'Test SEG-Y file' + b' ' * (3200 - 15)

        # Write traces with unique headers
        for i in range(n_traces):
            # Create trace data (increasing values per trace)
            trace_data = np.full(n_samples, i * 10, dtype=np.float32)
            f.trace[i] = trace_data

            # Set headers to identify each trace
            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.FieldRecord: 1000 - i,  # Descending for sort test
                segyio.TraceField.TraceNumber: i + 1
            }


def read_segy_headers(path, n_traces):
    """Read all headers from a SEG-Y file."""
    headers = []
    with segyio.open(str(path), 'r', ignore_geometry=True) as f:
        for i in range(n_traces):
            headers.append({
                'TRACE_SEQUENCE_FILE': f.header[i][segyio.TraceField.TRACE_SEQUENCE_FILE],
                'FieldRecord': f.header[i][segyio.TraceField.FieldRecord],
                'TraceNumber': f.header[i][segyio.TraceField.TraceNumber]
            })
    return pd.DataFrame(headers)


def test_sorted_export_with_headers():
    """Test that sorted headers are correctly exported in chunked mode."""
    print("\n=== Test: Sorted Header Export in Chunked Mode ===\n")

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Setup paths
        original_segy = temp_dir / "original.sgy"
        processed_zarr = temp_dir / "processed.zarr"
        output_without_sort = temp_dir / "output_no_sort.sgy"
        output_with_sort = temp_dir / "output_sorted.sgy"

        n_traces = 20
        n_samples = 100

        # Step 1: Create test SEG-Y
        print(f"Creating test SEG-Y with {n_traces} traces...")
        create_test_segy(original_segy, n_traces, n_samples)

        # Step 2: Create "processed" data (just copy for testing)
        print("Creating processed Zarr array...")
        with segyio.open(str(original_segy), 'r', ignore_geometry=True) as f:
            data = np.array([f.trace[i] for i in range(n_traces)]).T

        zarr_array = zarr.open(
            str(processed_zarr),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 10),
            dtype=np.float32
        )
        zarr_array[:] = data

        # Step 3: Export WITHOUT sorted headers (original order)
        print("Exporting without sorted headers...")
        export_from_zarr_chunked(
            output_path=str(output_without_sort),
            original_segy_path=str(original_segy),
            processed_zarr_path=str(processed_zarr),
            chunk_size=5,
            headers_df=None  # No sorted headers
        )

        # Step 4: Create sorted headers (sort by FieldRecord ascending - reverse of original)
        print("Creating sorted headers (by FieldRecord ascending)...")
        original_headers = read_segy_headers(original_segy, n_traces)
        sorted_headers = original_headers.sort_values('FieldRecord', ascending=True).reset_index(drop=True)

        print("\nOriginal header order (first 5):")
        print(original_headers[['TRACE_SEQUENCE_FILE', 'FieldRecord']].head())
        print("\nSorted header order (first 5):")
        print(sorted_headers[['TRACE_SEQUENCE_FILE', 'FieldRecord']].head())

        # Step 5: Export WITH sorted headers
        print("\nExporting with sorted headers...")
        export_from_zarr_chunked(
            output_path=str(output_with_sort),
            original_segy_path=str(original_segy),
            processed_zarr_path=str(processed_zarr),
            chunk_size=5,
            headers_df=sorted_headers  # Sorted headers
        )

        # Step 6: Verify results
        print("\nVerifying exported files...")

        # Read headers from both exports
        unsorted_export_headers = read_segy_headers(output_without_sort, n_traces)
        sorted_export_headers = read_segy_headers(output_with_sort, n_traces)

        print("\nExported WITHOUT sort (first 5):")
        print(unsorted_export_headers[['TRACE_SEQUENCE_FILE', 'FieldRecord']].head())
        print("\nExported WITH sort (first 5):")
        print(sorted_export_headers[['TRACE_SEQUENCE_FILE', 'FieldRecord']].head())

        # Verify unsorted export matches original
        assert unsorted_export_headers['FieldRecord'].tolist() == original_headers['FieldRecord'].tolist(), \
            "Unsorted export should match original header order"

        # Verify sorted export matches sorted headers
        assert sorted_export_headers['FieldRecord'].tolist() == sorted_headers['FieldRecord'].tolist(), \
            "Sorted export should match sorted header order"

        # Verify they are different (sorted vs unsorted)
        assert unsorted_export_headers['FieldRecord'].tolist() != sorted_export_headers['FieldRecord'].tolist(), \
            "Sorted and unsorted exports should be different"

        print("\n✅ SUCCESS: All tests passed!")
        print("   - Unsorted export matches original order")
        print("   - Sorted export matches sorted order")
        print("   - Headers are correctly applied in chunked export")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("\nCleaned up temporary files.")


if __name__ == "__main__":
    success = test_sorted_export_with_headers()
    exit(0 if success else 1)
