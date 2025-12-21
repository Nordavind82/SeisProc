#!/usr/bin/env python3
"""
Seismic Data Processing QC Tool - Main Entry Point

Professional seismic data quality control application with:
- Three synchronized viewers (Input, Processed, Difference)
- Interactive zoom, pan, and gain controls
- Extensible processing pipeline
- Zero-phase bandpass filtering
- Ray-based distributed processing (Phase 1-4)
- Job monitoring and resource tracking

Usage:
    python main.py
"""
import os
import sys
import logging
import atexit

# Fix for segmentation fault: Set Qt/display environment variables BEFORE importing PyQt6
# This prevents OpenGL integration issues that cause crashes on some systems
if sys.platform == 'linux':
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')          # Use X11 backend on Linux
    os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')   # Disable OpenGL (prevents segfault)
# On macOS, use default 'cocoa' platform (no need to set)
os.environ.setdefault('MPLBACKEND', 'Agg')              # Non-interactive matplotlib backend

# CRITICAL: Force Qt to use portable settings format instead of native D-Bus/system services
# This prevents QSettings from crashing when accessing system services with display issues
os.environ.setdefault('QT_SETTINGS_PATH', '/tmp')        # Use temp directory for settings
os.environ['QT_SCALE_FACTOR'] = '1'                      # Disable automatic scaling

from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
from utils.theme_manager import get_theme_manager

# Set up logging
logger = logging.getLogger(__name__)


def _initialize_ray_orchestration():
    """
    Initialize Phase 1-4 Ray orchestration components.

    This sets up:
    - Alert manager with default rules for job monitoring
    - Resource monitoring (started on demand by UI)
    - Job history storage (initialized lazily)

    Ray cluster itself is initialized lazily when first job runs.
    """
    try:
        # Initialize alert manager with default rules
        from utils.ray_orchestration.alert_manager import (
            get_alert_manager,
            create_default_alert_manager,
        )

        # Use default alert manager with pre-configured rules
        # This ensures job failure alerts work from the start
        alert_manager = create_default_alert_manager()

        logger.debug("Ray orchestration components initialized")
        return True

    except ImportError as e:
        logger.debug(f"Ray orchestration not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize Ray orchestration: {e}")
        return False


def _cleanup_ray_orchestration():
    """
    Clean up Ray orchestration on application exit.

    Ensures clean shutdown of:
    - Ray cluster (if initialized)
    - Resource monitors
    - Active job cancellation
    """
    try:
        from utils.ray_orchestration import (
            is_ray_initialized,
            shutdown_ray,
            stop_monitoring,
        )

        # Stop resource monitoring
        try:
            stop_monitoring()
            logger.debug("Resource monitoring stopped")
        except Exception as e:
            logger.debug(f"Resource monitoring stop failed: {e}")

        # Shutdown Ray cluster if it was initialized
        if is_ray_initialized():
            logger.info("Shutting down Ray cluster...")
            shutdown_ray()
            logger.info("Ray cluster shutdown complete")

    except ImportError:
        pass  # Ray not available
    except Exception as e:
        logger.debug(f"Ray cleanup error (non-fatal): {e}")


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

    # Initialize Phase 1-4 Ray orchestration components
    _initialize_ray_orchestration()

    # Register cleanup handler for Ray shutdown
    atexit.register(_cleanup_ray_orchestration)

    # Also connect to Qt's aboutToQuit signal for cleanup
    app.aboutToQuit.connect(_cleanup_ray_orchestration)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
