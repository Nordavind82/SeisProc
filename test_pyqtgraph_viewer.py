#!/usr/bin/env python3
"""
Test PyQtGraph seismic viewer with sample data.
Demonstrates performance and mouse controls.
"""
import sys

from PyQt6.QtWidgets import QApplication
from models.seismic_data import SeismicData
from models.viewport_state import ViewportState
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from utils.sample_data import generate_sample_seismic_data
import time


def test_pyqtgraph_viewer():
    """Test PyQtGraph viewer with sample data."""
    print("="*60)
    print("PyQtGraph Seismic Viewer Test")
    print("="*60 + "\n")

    # Create Qt application
    app = QApplication(sys.argv)

    # Create sample data
    print("Generating sample data...")
    start = time.time()
    data = generate_sample_seismic_data(
        n_samples=1000,
        n_traces=100,
        sample_rate=2.0,
        noise_level=0.1
    )
    elapsed = time.time() - start
    print(f"  ✓ Generated {data.n_traces} traces in {elapsed:.3f}s")
    print(f"  ✓ Data: {data}\n")

    # Create viewport state
    viewport_state = ViewportState()
    viewport_state.reset_to_data(data.duration, data.n_traces - 1)

    # Create viewer
    print("Creating PyQtGraph viewer...")
    start = time.time()
    viewer = SeismicViewerPyQtGraph("Test Viewer", viewport_state)
    elapsed = time.time() - start
    print(f"  ✓ Viewer created in {elapsed:.3f}s\n")

    # Set data
    print("Rendering data...")
    start = time.time()
    viewer.set_data(data)
    elapsed = time.time() - start
    print(f"  ✓ Data rendered in {elapsed:.3f}s\n")

    # Show viewer
    viewer.setWindowTitle("PyQtGraph Seismic Viewer Test")
    viewer.resize(1200, 800)
    viewer.show()

    print("="*60)
    print("Viewer Controls:")
    print("="*60)
    print("Mouse Wheel:        Zoom both axes")
    print("Ctrl+Wheel:         Zoom X-axis (traces) only")
    print("Shift+Wheel:        Zoom Y-axis (time) only")
    print("Left Drag:          Pan view")
    print("Right Drag:         Box zoom")
    print("Middle Click:       Reset view")
    print("Toolbar Dropdown:   Change zoom mode")
    print("Reset View Button:  Reset to full extent")
    print("="*60 + "\n")

    print("Close the window to exit...")

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    test_pyqtgraph_viewer()
