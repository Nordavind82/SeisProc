"""
Test Interactive Spectral Analysis (ISA) window.
"""
import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from models.seismic_data import SeismicData
from views.isa_window import ISAWindow


def generate_test_data():
    """Generate simple test seismic data with known frequency content."""
    n_samples = 500
    n_traces = 50
    sample_rate_ms = 2.0  # 2ms

    # Create time axis
    sample_rate_hz = 1000.0 / sample_rate_ms  # 500 Hz
    time_s = np.arange(n_samples) * (sample_rate_ms / 1000.0)

    # Create traces with different frequency content
    traces = np.zeros((n_samples, n_traces))

    for i in range(n_traces):
        # Add multiple frequency components
        # Main signal: 30 Hz
        traces[:, i] += np.sin(2 * np.pi * 30 * time_s)

        # Higher frequency: 60 Hz (weaker)
        traces[:, i] += 0.5 * np.sin(2 * np.pi * 60 * time_s)

        # Low frequency: 10 Hz
        traces[:, i] += 0.3 * np.sin(2 * np.pi * 10 * time_s)

        # Add noise
        traces[:, i] += 0.1 * np.random.randn(n_samples)

    return SeismicData(
        traces=traces,
        sample_rate=sample_rate_ms,
        metadata={'description': 'Test data for ISA'}
    )


def main():
    """Test ISA window."""
    print("Generating test data...")
    test_data = generate_test_data()

    print(f"Test data created:")
    print(f"  Traces: {test_data.n_traces}")
    print(f"  Samples: {test_data.n_samples}")
    print(f"  Duration: {test_data.duration:.1f} ms")
    print(f"  Sample rate: {test_data.sample_rate} ms")
    print(f"  Nyquist: {test_data.nyquist_freq:.1f} Hz")
    print(f"\nExpected frequencies: 10 Hz, 30 Hz (dominant), 60 Hz")

    print("\nOpening ISA window...")
    app = QApplication(sys.argv)

    # Create ISA window (viewport_state is optional, will create new one if not provided)
    isa_window = ISAWindow(test_data)
    isa_window.show()

    print("ISA window opened. You can:")
    print("  - Click on any trace to view its spectrum")
    print("  - Use trace selector spinbox")
    print("  - Enable 'Show average spectrum' to see ensemble average")
    print("  - Adjust frequency range")

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
