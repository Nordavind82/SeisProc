"""
Test to verify both export modes produce identical results.

Compares batch processing export vs memory-efficient export
to ensure headers and data are identical in both modes.
"""
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import segyio

from utils.segy_import.segy_export import export_processed_segy, export_from_zarr_chunked
from models.seismic_data import SeismicData
import zarr


def create_test_segy(path, n_traces=20, n_samples=100):
    """Create a simple test SEG-Y file."""
    spec = segyio.spec()
    spec.format = 1
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(str(path), spec) as f:
        f.text[0] = b'Test SEG-Y file' + b' ' * (3200 - 15)

        for i in range(n_traces):
            trace_data = np.full(n_samples, i * 10, dtype=np.float32)
            f.trace[i] = trace_data

            f.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.FieldRecord: 1000 - i,
                segyio.TraceField.TraceNumber: i + 1
            }


def read_segy_full(path):
    """Read both data and headers from SEG-Y."""
    with segyio.open(str(path), 'r', ignore_geometry=True) as f:
        n_traces = f.tracecount
        data = np.array([f.trace[i] for i in range(n_traces)]).T

        headers = []
        for i in range(n_traces):
            headers.append({
                'TRACE_SEQUENCE_FILE': f.header[i][segyio.TraceField.TRACE_SEQUENCE_FILE],
                'FieldRecord': f.header[i][segyio.TraceField.FieldRecord],
                'TraceNumber': f.header[i][segyio.TraceField.TraceNumber]
            })

        return data, pd.DataFrame(headers)


def test_export_mode_consistency():
    """Test that batch and memory-efficient exports produce identical results."""
    print("\n=== Test: Export Mode Consistency ===\n")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Setup
        original_segy = temp_dir / "original.sgy"
        processed_zarr = temp_dir / "processed.zarr"
        batch_export = temp_dir / "batch_export.sgy"
        chunked_export = temp_dir / "chunked_export.sgy"

        n_traces = 20
        n_samples = 100

        # Create test data
        print(f"Creating test SEG-Y with {n_traces} traces...")
        create_test_segy(original_segy, n_traces, n_samples)

        # Read original
        original_data, original_headers = read_segy_full(original_segy)

        # Create sorted headers (ascending by FieldRecord)
        print("Creating sorted headers...")
        sorted_headers = original_headers.sort_values('FieldRecord', ascending=True).reset_index(drop=True)

        # Sort data to match sorted headers
        sort_indices = original_headers.sort_values('FieldRecord', ascending=True).index.values
        sorted_data = original_data[:, sort_indices]

        print("\nOriginal order (first 5):")
        print(original_headers[['FieldRecord']].head())
        print("\nSorted order (first 5):")
        print(sorted_headers[['FieldRecord']].head())

        # Create processed SeismicData (for batch export)
        processed_seismic = SeismicData(
            traces=sorted_data,
            sample_rate=0.004,
            metadata={'sorted': True}
        )

        # Create Zarr (for chunked export)
        print("\nCreating Zarr array...")
        zarr_array = zarr.open(
            str(processed_zarr),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 10),
            dtype=np.float32
        )
        zarr_array[:] = sorted_data

        # Export using batch mode
        print("Exporting with batch mode...")
        export_processed_segy(
            output_path=str(batch_export),
            original_segy_path=str(original_segy),
            processed_data=processed_seismic,
            headers_df=sorted_headers
        )

        # Export using chunked mode
        print("Exporting with chunked mode...")
        export_from_zarr_chunked(
            output_path=str(chunked_export),
            original_segy_path=str(original_segy),
            processed_zarr_path=str(processed_zarr),
            chunk_size=5,
            headers_df=sorted_headers
        )

        # Compare results
        print("\nComparing exported files...")
        batch_data, batch_headers = read_segy_full(batch_export)
        chunked_data, chunked_headers = read_segy_full(chunked_export)

        print("\nBatch export headers (first 5):")
        print(batch_headers[['FieldRecord']].head())
        print("\nChunked export headers (first 5):")
        print(chunked_headers[['FieldRecord']].head())

        # Verify headers match
        assert batch_headers['FieldRecord'].tolist() == chunked_headers['FieldRecord'].tolist(), \
            "Headers should match between batch and chunked export"

        assert batch_headers['TRACE_SEQUENCE_FILE'].tolist() == chunked_headers['TRACE_SEQUENCE_FILE'].tolist(), \
            "Trace sequence should match between batch and chunked export"

        # Verify data matches
        assert np.allclose(batch_data, chunked_data), \
            "Trace data should match between batch and chunked export"

        # Verify sorted order is maintained
        assert batch_headers['FieldRecord'].tolist() == sorted_headers['FieldRecord'].tolist(), \
            "Exported headers should match sorted order"

        print("\n✅ SUCCESS: Both export modes produce identical results!")
        print("   - Headers are identical")
        print("   - Trace data is identical")
        print("   - Sorted order is preserved in both modes")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("\nCleaned up temporary files.")


if __name__ == "__main__":
    success = test_export_mode_consistency()
    exit(0 if success else 1)
