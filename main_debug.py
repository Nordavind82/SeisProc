#!/usr/bin/env python3
"""
Debug version of main.py to trace segmentation fault.
"""
import sys
import os
import logging
import traceback

# Set up logging BEFORE any imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/scratch/Python_Apps/SeisProc/debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("STARTING APPLICATION IN DEBUG MODE")
logger.info("=" * 70)

# Force Qt platform to xcb (X11)
logger.info("Setting QT_QPA_PLATFORM to xcb")
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Disable OpenGL if it's causing issues
logger.info("Setting QT_XCB_GL_INTEGRATION to none")
os.environ['QT_XCB_GL_INTEGRATION'] = 'none'

# Set matplotlib backend BEFORE importing pyplot
logger.info("Setting matplotlib backend to Agg (non-interactive)")
os.environ['MPLBACKEND'] = 'Agg'

try:
    logger.info("Step 1: Importing PyQt6.QtWidgets...")
    from PyQt6.QtWidgets import QApplication
    logger.info("  ✓ PyQt6.QtWidgets imported successfully")
except Exception as e:
    logger.error(f"  ✗ Failed to import PyQt6.QtWidgets: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 2: Importing PyQt6.QtCore...")
    from PyQt6.QtCore import Qt
    logger.info("  ✓ PyQt6.QtCore imported successfully")
except Exception as e:
    logger.error(f"  ✗ Failed to import PyQt6.QtCore: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 3: Importing matplotlib...")
    import matplotlib
    matplotlib.use('Agg')  # Force non-GUI backend
    logger.info(f"  ✓ Matplotlib imported, backend: {matplotlib.get_backend()}")
except Exception as e:
    logger.error(f"  ✗ Failed to import matplotlib: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 4: Importing numpy...")
    import numpy as np
    logger.info(f"  ✓ Numpy imported, version: {np.__version__}")
except Exception as e:
    logger.error(f"  ✗ Failed to import numpy: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 5: Importing pyqtgraph...")
    import pyqtgraph as pg
    logger.info(f"  ✓ pyqtgraph imported")
except Exception as e:
    logger.error(f"  ✗ Failed to import pyqtgraph: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 6: Creating QApplication instance...")
    app = QApplication(sys.argv)
    logger.info("  ✓ QApplication created successfully")

    logger.info("Step 7: Setting application metadata...")
    app.setApplicationName("Seismic QC Tool")
    app.setOrganizationName("Geophysical Software")
    app.setApplicationVersion("1.0.0")
    logger.info("  ✓ Application metadata set")

except Exception as e:
    logger.error(f"  ✗ Failed to create QApplication: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 8: Importing MainWindow...")
    from main_window import MainWindow
    logger.info("  ✓ MainWindow imported successfully")
except Exception as e:
    logger.error(f"  ✗ Failed to import MainWindow: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 9: Creating MainWindow instance...")
    window = MainWindow()
    logger.info("  ✓ MainWindow created successfully")
except Exception as e:
    logger.error(f"  ✗ Failed to create MainWindow: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 10: Showing MainWindow...")
    window.show()
    logger.info("  ✓ MainWindow shown successfully")
except Exception as e:
    logger.error(f"  ✗ Failed to show MainWindow: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    logger.info("Step 11: Starting event loop...")
    logger.info("Application is running. Use Ctrl+C to exit.")
    logger.info("=" * 70)
    sys.exit(app.exec())
except Exception as e:
    logger.error(f"  ✗ Event loop crashed: {e}")
    traceback.print_exc()
    sys.exit(1)
