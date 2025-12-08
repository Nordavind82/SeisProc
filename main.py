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
import os
import sys

# Fix for segmentation fault: Set Qt/display environment variables BEFORE importing PyQt6
# This prevents OpenGL integration issues that cause crashes on some systems
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')          # Use X11 backend
os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')  # Disable OpenGL (prevents segfault)
os.environ.setdefault('MPLBACKEND', 'Agg')              # Non-interactive matplotlib backend

# CRITICAL: Force Qt to use portable settings format instead of native D-Bus/system services
# This prevents QSettings from crashing when accessing system services with display issues
os.environ.setdefault('QT_SETTINGS_PATH', '/tmp')        # Use temp directory for settings
os.environ['QT_SCALE_FACTOR'] = '1'                      # Disable automatic scaling

from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
from utils.theme_manager import get_theme_manager


def main():
    """Main entry point for the application."""
    # Create Qt application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Seismic QC Tool")
    app.setOrganizationName("Geophysical Software")
    app.setApplicationVersion("1.0.0")

    # Apply theme from saved preferences
    theme_manager = get_theme_manager()
    theme_manager.apply_to_app()

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
