#!/usr/bin/env python3
"""
Test script for SEG-Y import functionality.
Demonstrates header mapping, ensemble detection, and Zarr/Parquet storage.
"""
import sys
import numpy as np

from utils.segy_import.header_mapping import HeaderMapping, HeaderField, StandardHeaders
from utils.segy_import.data_storage import DataStorage
from models.seismic_data import SeismicData


def test_header_mapping():
    """Test header mapping configuration."""
    print("="*60)
    print("Testing Header Mapping Configuration")
    print("="*60 + "\n")

    # Create mapping
    mapping = HeaderMapping()

    # Add standard headers
    print("Adding standard headers...")
    mapping.add_standard_headers(StandardHeaders.get_minimal())
    print(f"  ✓ Added {len(mapping.fields)} standard headers")

    # Add custom header
    print("\nAdding custom header...")
    custom = HeaderField(
        name='processing_code',
        byte_position=233,
        format='i',
        description='Custom processing identifier'
    )
    mapping.add_field(custom)
    print(f"  ✓ Added custom header: {custom.name}")

    # Set ensemble keys
    print("\nConfiguring ensemble boundaries...")
    mapping.set_ensemble_keys(['cdp'])
    print(f"  ✓ Ensemble keys: {mapping.ensemble_keys}")

    # Save to file
    print("\nSaving configuration...")
    mapping.save_to_file('/tmp/test_mapping.json')
    print("  ✓ Saved to /tmp/test_mapping.json")

    # Load from file
    print("\nLoading configuration...")
    loaded = HeaderMapping.load_from_file('/tmp/test_mapping.json')
    print(f"  ✓ Loaded: {loaded}")

    print("\n✓ Header mapping test passed\n")
    return mapping


def test_data_storage():
    """Test Zarr/Parquet storage."""
    print("="*60)
    print("Testing Zarr/Parquet Storage")
    print("="*60 + "\n")

    # Create sample seismic data
    print("Creating sample data...")
    traces = np.random.randn(1000, 100).astype(np.float32)
    seismic_data = SeismicData(
        traces=traces,
        sample_rate=2.0,
        metadata={'source': 'test_script', 'description': 'Test data'}
    )
    print(f"  ✓ Created: {seismic_data}")

    # Create sample headers
    print("\nCreating sample headers...")
    headers = []
    for i in range(100):
        headers.append({
            'trace_sequence_file': i + 1,
            'cdp': 1000 + (i // 10),  # 10 traces per CDP
            'offset': 100 * (i % 10),
            'sample_count': 1000,
            'sample_interval': 2000,
        })
    print(f"  ✓ Created headers for {len(headers)} traces")

    # Create ensemble boundaries (10 CDPs with 10 traces each)
    print("\nCreating ensemble boundaries...")
    ensembles = []
    for i in range(10):
        ensembles.append((i * 10, (i + 1) * 10 - 1))
    print(f"  ✓ Created {len(ensembles)} ensembles")

    # Save to storage
    print("\nSaving to Zarr/Parquet...")
    storage = DataStorage('/tmp/test_seismic_data')
    storage.save_seismic_data(seismic_data, headers, ensembles, chunk_size=50)
    print("  ✓ Data saved")

    # Get statistics
    print("\nStorage statistics:")
    stats = storage.get_statistics()
    print(f"  Zarr size: {stats['zarr']['size_mb']:.2f} MB")
    print(f"  Headers size: {stats['headers']['size_mb']:.2f} MB")
    print(f"  Compression ratio: {stats['zarr']['size_mb'] / (traces.nbytes / 1024 / 1024):.2f}x")
    print(f"  Ensembles: {stats['ensembles']['n_ensembles']}")

    # Load data back
    print("\nLoading data back...")
    loaded_data, headers_df, ensembles_df = storage.load_seismic_data()
    print(f"  ✓ Loaded: {loaded_data}")
    print(f"  ✓ Headers shape: {headers_df.shape}")
    print(f"  ✓ Ensembles: {len(ensembles_df)}")

    # Verify data integrity
    print("\nVerifying data integrity...")
    assert np.allclose(loaded_data.traces, seismic_data.traces), "Traces don't match!"
    assert len(headers_df) == len(headers), "Header count doesn't match!"
    assert len(ensembles_df) == len(ensembles), "Ensemble count doesn't match!"
    print("  ✓ Data integrity verified")

    # Test ensemble access
    print("\nTesting ensemble access...")
    ensemble_traces, ensemble_headers = storage.get_ensemble_traces(ensemble_id=5)
    print(f"  ✓ Ensemble 5: {ensemble_traces.shape}")
    print(f"  ✓ Headers: {len(ensemble_headers)} traces")

    # Test header queries
    print("\nTesting header queries...")
    subset = storage.query_headers("cdp == 1005 and offset < 500")
    print(f"  ✓ Query result: {len(subset)} traces")

    print("\n✓ Data storage test passed\n")


def test_read_value():
    """Test reading header values from bytes."""
    print("="*60)
    print("Testing Header Value Reading")
    print("="*60 + "\n")

    # Create test header bytes (240 bytes)
    header_bytes = bytearray(240)

    # Write test values
    # Trace sequence at byte 5 (int32)
    import struct
    struct.pack_into('>i', header_bytes, 4, 12345)  # byte 5 (0-indexed as 4)

    # CDP at byte 21 (int32)
    struct.pack_into('>i', header_bytes, 20, 9999)

    # Create header field
    trace_seq_field = HeaderField('trace_sequence_file', 5, 'i', 'Trace sequence')
    cdp_field = HeaderField('cdp', 21, 'i', 'CDP number')

    # Read values
    trace_seq = trace_seq_field.read_value(bytes(header_bytes))
    cdp = cdp_field.read_value(bytes(header_bytes))

    print(f"Trace sequence: {trace_seq}")
    print(f"CDP: {cdp}")

    assert trace_seq == 12345, "Trace sequence mismatch!"
    assert cdp == 9999, "CDP mismatch!"

    print("\n✓ Header reading test passed\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SEG-Y IMPORT SYSTEM - Component Tests")
    print("="*60 + "\n")

    try:
        test_header_mapping()
        test_read_value()
        test_data_storage()

        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nThe SEG-Y import system is ready to use!")
        print("\nNext steps:")
        print("  1. Run the GUI: python main.py")
        print("  2. File → Load SEG-Y File...")
        print("  3. Configure headers and ensemble keys")
        print("  4. Import your data")
        print("\nSee SEGY_IMPORT_GUIDE.md for detailed documentation.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
