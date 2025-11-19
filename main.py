#!/usr/bin/env python3
"""
Seismic Data Processing QC Tool - Main Entry Point

Professional seismic data quality control application with:
- Three synchronized viewers (Input, Processed, Difference)
- Interactive zoom, pan, and gain controls
- Extensible processing pipeline
- Zero-phase bandpass filtering

Usage:
    python main.py
"""
import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    """Main entry point for the application."""
    # Create Qt application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Seismic QC Tool")
    app.setOrganizationName("Geophysical Software")
    app.setApplicationVersion("1.0.0")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
