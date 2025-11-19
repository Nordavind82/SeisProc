#!/usr/bin/env python3
"""
Test script to verify core components work correctly.
Tests data model, processing, and basic functionality without GUI.
"""
import sys

from models.seismic_data import SeismicData
from processors.bandpass_filter import BandpassFilter, ProcessingPipeline
from utils.sample_data import generate_sample_seismic_data
import numpy as np


def test_data_model():
    """Test SeismicData model."""
    print("Testing SeismicData model...")

    # Create sample data
    data = generate_sample_seismic_data(
        n_samples=500,
        n_traces=50,
        sample_rate=2.0,
        noise_level=0.1
    )

    print(f"  ✓ Created: {data}")
    print(f"  ✓ Duration: {data.duration:.0f}ms")
    print(f"  ✓ Nyquist frequency: {data.nyquist_freq:.1f}Hz")

    assert data.n_samples == 500
    assert data.n_traces == 50
    assert data.sample_rate == 2.0
    assert data.nyquist_freq == 250.0

    print("  ✓ All assertions passed\n")


def test_bandpass_filter():
    """Test bandpass filter processor."""
    print("Testing BandpassFilter...")

    # Create sample data
    data = generate_sample_seismic_data(
        n_samples=500,
        n_traces=50,
        sample_rate=2.0
    )

    # Create bandpass filter
    processor = BandpassFilter(
        low_freq=10.0,
        high_freq=80.0,
        order=4
    )

    print(f"  ✓ Created filter: {processor.get_description()}")

    # Process data
    processed = processor.process(data)

    print(f"  ✓ Processed data: {processed}")
    print(f"  ✓ Input shape: {data.traces.shape}")
    print(f"  ✓ Output shape: {processed.traces.shape}")

    assert processed.traces.shape == data.traces.shape
    assert processed.sample_rate == data.sample_rate
    assert 'processing_history' in processed.metadata

    # Verify input data unchanged (immutability)
    assert data.metadata.get('processing_history') is None

    print("  ✓ All assertions passed\n")


def test_nyquist_validation():
    """Test that filter validates against Nyquist frequency."""
    print("Testing Nyquist frequency validation...")

    data = generate_sample_seismic_data(
        n_samples=500,
        n_traces=50,
        sample_rate=2.0  # Nyquist = 250 Hz
    )

    # Should fail: high frequency >= Nyquist
    try:
        processor = BandpassFilter(
            low_freq=10.0,
            high_freq=260.0,  # Exceeds Nyquist!
            order=4
        )
        processed = processor.process(data)
        print("  ✗ Should have raised ValueError!")
        assert False
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {e}")

    print("  ✓ Nyquist validation working\n")


def test_processing_pipeline():
    """Test processing pipeline."""
    print("Testing ProcessingPipeline...")

    data = generate_sample_seismic_data(
        n_samples=500,
        n_traces=50,
        sample_rate=2.0
    )

    # Create pipeline with multiple filters
    pipeline = ProcessingPipeline()
    pipeline.add_processor(BandpassFilter(low_freq=10.0, high_freq=80.0, order=4))

    print(f"  ✓ Pipeline: {pipeline}")

    # Process
    result = pipeline.process(data)

    print(f"  ✓ Processed: {result}")
    print(f"  ✓ {pipeline.get_description()}")

    assert result.traces.shape == data.traces.shape
    assert len(result.metadata['processing_history']) == 1

    print("  ✓ All assertions passed\n")


def test_difference_calculation():
    """Test difference calculation for QC."""
    print("Testing difference calculation...")

    # Create data
    data = generate_sample_seismic_data(
        n_samples=500,
        n_traces=50,
        sample_rate=2.0
    )

    # Process
    processor = BandpassFilter(low_freq=10.0, high_freq=80.0, order=4)
    processed = processor.process(data)

    # Calculate difference (residual)
    difference = data.traces - processed.traces

    print(f"  ✓ Input RMS: {np.sqrt(np.mean(data.traces**2)):.4f}")
    print(f"  ✓ Processed RMS: {np.sqrt(np.mean(processed.traces**2)):.4f}")
    print(f"  ✓ Difference RMS: {np.sqrt(np.mean(difference**2)):.4f}")

    # Difference should be smaller than input (filtering removes some energy)
    assert np.abs(difference).max() < np.abs(data.traces).max()

    print("  ✓ Difference calculation working\n")


def main():
    """Run all tests."""
    print("="*60)
    print("SEISMIC QC APP - Component Tests")
    print("="*60 + "\n")

    try:
        test_data_model()
        test_bandpass_filter()
        test_nyquist_validation()
        test_processing_pipeline()
        test_difference_calculation()

        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYou can now run the GUI application:")
        print("  python main.py")
        print("\nOr with:")
        print("  cd seismic_qc_app && python main.py")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
