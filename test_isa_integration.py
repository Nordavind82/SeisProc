"""
Integration test for ISA functionality - no GUI required.
"""
import numpy as np
from models.seismic_data import SeismicData
from processors.spectral_analyzer import SpectralAnalyzer


def test_spectral_analyzer():
    """Test spectral analyzer with synthetic seismic data."""
    print("=" * 60)
    print("Testing Spectral Analyzer")
    print("=" * 60)

    # Generate test data
    n_samples = 500
    n_traces = 50
    sample_rate_ms = 2.0

    sample_rate_hz = 1000.0 / sample_rate_ms
    time_s = np.arange(n_samples) * (sample_rate_ms / 1000.0)

    traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        # 30 Hz dominant frequency
        traces[:, i] = np.sin(2 * np.pi * 30 * time_s)
        # Add noise
        traces[:, i] += 0.1 * np.random.randn(n_samples)

    # Create seismic data
    data = SeismicData(
        traces=traces,
        sample_rate=sample_rate_ms,
        metadata={'description': 'Test'}
    )

    print(f"\nTest Data:")
    print(f"  Traces: {data.n_traces}")
    print(f"  Samples: {data.n_samples}")
    print(f"  Sample rate: {data.sample_rate} ms")
    print(f"  Nyquist: {data.nyquist_freq:.1f} Hz")

    # Create analyzer
    analyzer = SpectralAnalyzer(data.sample_rate)

    # Test 1: Single trace spectrum
    print(f"\nTest 1: Single trace spectrum")
    trace_idx = 0
    freqs, amps = analyzer.compute_spectrum(data.traces[:, trace_idx])
    dominant = analyzer.find_dominant_frequency(freqs, amps)

    print(f"  Frequency bins: {len(freqs)}")
    print(f"  Expected dominant: 30 Hz")
    print(f"  Detected dominant: {dominant:.1f} Hz")

    assert abs(dominant - 30.0) < 1.0, f"Expected 30 Hz, got {dominant:.1f} Hz"
    print(f"  ✓ PASS")

    # Test 2: Average spectrum
    print(f"\nTest 2: Average spectrum (all traces)")
    avg_freqs, avg_amps = analyzer.compute_average_spectrum(data.traces)
    avg_dominant = analyzer.find_dominant_frequency(avg_freqs, avg_amps)

    print(f"  Frequency bins: {len(avg_freqs)}")
    print(f"  Expected dominant: 30 Hz")
    print(f"  Detected dominant: {avg_dominant:.1f} Hz")

    assert abs(avg_dominant - 30.0) < 1.0, f"Expected 30 Hz, got {avg_dominant:.1f} Hz"
    print(f"  ✓ PASS")

    # Test 3: Find dominant in frequency range
    print(f"\nTest 3: Dominant frequency in range [20-40 Hz]")
    range_dominant = analyzer.find_dominant_frequency(freqs, amps, freq_range=(20, 40))

    print(f"  Expected: 30 Hz")
    print(f"  Detected: {range_dominant:.1f} Hz")

    assert abs(range_dominant - 30.0) < 1.0, f"Expected 30 Hz, got {range_dominant:.1f} Hz"
    print(f"  ✓ PASS")

    print(f"\n" + "=" * 60)
    print("All tests PASSED! ✓")
    print("=" * 60)


def test_multi_frequency():
    """Test with multiple frequency components."""
    print("\n" + "=" * 60)
    print("Testing Multi-Frequency Signal")
    print("=" * 60)

    # Generate signal with multiple frequencies
    n_samples = 1000
    sample_rate_ms = 2.0
    sample_rate_hz = 1000.0 / sample_rate_ms
    time_s = np.arange(n_samples) * (sample_rate_ms / 1000.0)

    # Three frequency components
    signal = (
        1.0 * np.sin(2 * np.pi * 20 * time_s) +  # 20 Hz
        2.0 * np.sin(2 * np.pi * 50 * time_s) +  # 50 Hz (dominant)
        0.5 * np.sin(2 * np.pi * 80 * time_s)    # 80 Hz
    )

    analyzer = SpectralAnalyzer(sample_rate_ms)
    freqs, amps = analyzer.compute_spectrum(signal)

    print(f"\nSignal components:")
    print(f"  20 Hz (amplitude: 1.0)")
    print(f"  50 Hz (amplitude: 2.0) <- dominant")
    print(f"  80 Hz (amplitude: 0.5)")

    # Overall dominant should be 50 Hz
    overall_dominant = analyzer.find_dominant_frequency(freqs, amps)
    print(f"\nOverall dominant: {overall_dominant:.1f} Hz")
    assert abs(overall_dominant - 50.0) < 2.0, f"Expected ~50 Hz, got {overall_dominant:.1f} Hz"
    print(f"  ✓ PASS")

    # Test range searches
    low_dominant = analyzer.find_dominant_frequency(freqs, amps, freq_range=(10, 30))
    mid_dominant = analyzer.find_dominant_frequency(freqs, amps, freq_range=(40, 60))
    high_dominant = analyzer.find_dominant_frequency(freqs, amps, freq_range=(70, 90))

    print(f"\nRange searches:")
    print(f"  [10-30 Hz]: {low_dominant:.1f} Hz (expected ~20 Hz)")
    print(f"  [40-60 Hz]: {mid_dominant:.1f} Hz (expected ~50 Hz)")
    print(f"  [70-90 Hz]: {high_dominant:.1f} Hz (expected ~80 Hz)")

    assert abs(low_dominant - 20.0) < 2.0
    assert abs(mid_dominant - 50.0) < 2.0
    assert abs(high_dominant - 80.0) < 2.0
    print(f"  ✓ ALL PASS")

    print(f"\n" + "=" * 60)


if __name__ == '__main__':
    test_spectral_analyzer()
    test_multi_frequency()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED! ✓✓✓")
    print("=" * 60)
    print("\nISA functionality is ready to use:")
    print("  1. Run main application: python main.py")
    print("  2. Load data (File → Generate Sample Data)")
    print("  3. Open ISA: View → Open ISA Window (Ctrl+I)")
    print("  4. Click on traces to view their spectrum")
