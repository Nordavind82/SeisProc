"""
Main application window - coordinates three synchronized seismic viewers.
"""
import logging
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QMenuBar, QMenu, QFileDialog, QMessageBox, QSplitter,
                              QPushButton, QProgressDialog, QApplication)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
import numpy as np
import sys
import zarr
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
from models.seismic_data import SeismicData
from models.viewport_state import ViewportState
from models.gather_navigator import GatherNavigator
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from views.control_panel import ControlPanel
from views.gather_navigation_panel import GatherNavigationPanel
from views.segy_import_dialog import SEGYImportDialog
from views.flip_window import FlipWindow
from views.fk_designer_dialog import FKDesignerDialog
from processors.base_processor import BaseProcessor
from processors.fk_filter import FKFilter
from models.fk_config import FKConfigManager, FKFilterConfig
from utils.trace_spacing import calculate_trace_spacing_with_stats


class MainWindow(QMainWindow):
    """
    Main application window with three synchronized seismic viewers.

    Layout:
    - Left: Control panel
    - Right: Three horizontally arranged seismic viewers (Input, Processed, Difference)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Data Processing QC Tool")
        self.setGeometry(100, 100, 1600, 900)

        # Data storage
        self.input_data = None
        self.processed_data = None
        self.difference_data = None
        self.headers_df = None  # Pandas DataFrame with trace headers
        self.ensembles_df = None  # Pandas DataFrame with ensemble boundaries
        self.original_segy_path = None  # Path to original SEG-Y file (for export)

        # Full dataset storage (for batch processing)
        self.full_processed_data = None  # Full processed dataset (all gathers)
        self.sorted_headers_df = None  # Sorted headers DataFrame (for export)
        self.is_full_dataset_processed = False  # Flag to track if all gathers processed

        # Gather navigation
        self.gather_navigator = GatherNavigator()

        # Auto-processing state
        self.auto_process_enabled = False
        self.last_processor = None  # Store last used processor

        # Shared viewport state for synchronized views
        self.viewport_state = ViewportState()

        # Flip window (created on demand)
        self.flip_window = None

        # Create UI
        self._init_ui()
        self._create_menu_bar()

        # Show welcome message
        self.statusBar().showMessage("Ready. Load seismic data to begin.")

    def _init_ui(self):
        """Initialize user interface."""
        # Central widget with horizontal splitter
        central = QWidget()
        main_layout = QHBoxLayout()

        # Left side: Control panel and gather navigation
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Gather navigation panel
        self.gather_nav_panel = GatherNavigationPanel(self.gather_navigator)
        self.gather_nav_panel.gather_navigation_requested.connect(self._on_gather_navigation)
        self.gather_nav_panel.auto_process_changed.connect(self._on_auto_process_changed)
        left_layout.addWidget(self.gather_nav_panel)

        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.process_requested.connect(self._on_process_requested)
        self.control_panel.amplitude_range_changed.connect(self.viewport_state.set_amplitude_range)
        self.control_panel.colormap_changed.connect(self.viewport_state.set_colormap)
        self.control_panel.interpolation_changed.connect(self.viewport_state.set_interpolation)
        self.control_panel.sort_keys_changed.connect(self._on_sort_keys_changed)
        self.control_panel.zoom_in_requested.connect(lambda: self.viewport_state.zoom_in(0.5))
        self.control_panel.zoom_out_requested.connect(lambda: self.viewport_state.zoom_out(2.0))
        self.control_panel.reset_view_requested.connect(self._reset_view)
        self.control_panel.fk_design_requested.connect(self._on_fk_design_requested)
        self.control_panel.fk_config_selected.connect(self._on_fk_config_selected)
        # Connect auto-scale button to auto-scale method
        auto_scale_btn = self.control_panel.findChild(QPushButton, "Auto Scale from Data")
        if not auto_scale_btn:
            # Find by text
            for btn in self.control_panel.findChildren(QPushButton):
                if "Auto Scale" in btn.text():
                    btn.clicked.connect(self._auto_scale_amplitude)
                    break
        left_layout.addWidget(self.control_panel)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Right side: Three seismic viewers (PyQtGraph-based)
        viewer_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.input_viewer = SeismicViewerPyQtGraph("Input Data", self.viewport_state)
        self.processed_viewer = SeismicViewerPyQtGraph("Difference (Removed Noise)", self.viewport_state)
        self.difference_viewer = SeismicViewerPyQtGraph("Processed (Denoised)", self.viewport_state)

        viewer_splitter.addWidget(self.input_viewer)
        viewer_splitter.addWidget(self.processed_viewer)
        viewer_splitter.addWidget(self.difference_viewer)

        # Equal width for all three viewers
        viewer_splitter.setSizes([400, 400, 400])

        main_layout.addWidget(viewer_splitter, stretch=1)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Load data action
        load_action = QAction("&Load SEG-Y File...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._import_segy_dialog)
        file_menu.addAction(load_action)

        # Load from Zarr/Parquet
        load_zarr_action = QAction("Load from &Zarr/Parquet...", self)
        load_zarr_action.triggered.connect(self._load_from_zarr)
        file_menu.addAction(load_zarr_action)

        file_menu.addSeparator()

        # Export processed SEG-Y action
        export_action = QAction("&Export Processed SEG-Y...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setToolTip("Export processed data to SEG-Y file")
        export_action.triggered.connect(self._export_processed_segy)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Generate sample data action
        sample_action = QAction("&Generate Sample Data", self)
        sample_action.setShortcut("Ctrl+G")
        sample_action.triggered.connect(self._generate_sample_data)
        file_menu.addAction(sample_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Processing menu
        process_menu = menubar.addMenu("&Processing")

        # Batch process all gathers action
        batch_process_action = QAction("&Batch Process All Gathers", self)
        batch_process_action.setShortcut("Ctrl+B")
        batch_process_action.setToolTip("Process all gathers in dataset with current filter settings")
        batch_process_action.triggered.connect(self._batch_process_all_gathers)
        process_menu.addAction(batch_process_action)

        # Memory-efficient batch process and export action
        batch_export_action = QAction("Process and Export (Memory Efficient)...", self)
        batch_export_action.setShortcut("Ctrl+Shift+E")
        batch_export_action.setToolTip("Process all gathers and export directly to SEG-Y without loading into memory")
        batch_export_action.triggered.connect(self._batch_process_and_export_streaming)
        process_menu.addAction(batch_export_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Flip window action
        flip_action = QAction("Open &Flip Window", self)
        flip_action.setShortcut("Ctrl+F")
        flip_action.setToolTip("Open flip window to cycle through Input/Processed/Difference views")
        flip_action.triggered.connect(self._open_flip_window)
        view_menu.addAction(flip_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _import_segy_dialog(self):
        """Open SEG-Y import dialog."""
        logger.info("_import_segy_dialog() - START")
        try:
            logger.info("  → Creating SEGYImportDialog...")
            dialog = SEGYImportDialog(self)
            logger.info("  ✓ SEGYImportDialog created successfully")

            logger.info("  → Connecting import_completed signal...")
            dialog.import_completed.connect(self._on_segy_imported)
            logger.info("  ✓ Signal connected")

            logger.info("  → Calling dialog.exec()...")
            dialog.exec()
            logger.info("  ✓ Dialog exec() returned")

            logger.info("_import_segy_dialog() - COMPLETE")
        except Exception as e:
            logger.error(f"_import_segy_dialog() - FAILED: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open SEG-Y import dialog:\n{e}")
            raise

    def _on_segy_imported(self, seismic_data, headers_df, ensembles_df, file_path=None):
        """Handle completed SEG-Y import."""
        from models.lazy_seismic_data import LazySeismicData

        # Store original SEG-Y file path for export (if available)
        if file_path is not None:
            self.original_segy_path = file_path

        # Load data into gather navigator - use lazy loading if available
        if isinstance(seismic_data, LazySeismicData):
            self.gather_navigator.load_lazy_data(seismic_data, ensembles_df)
        else:
            self.gather_navigator.load_data(seismic_data, headers_df, ensembles_df)

        # Display first gather
        self._display_current_gather()

        # Update control panel with Nyquist frequency
        self.control_panel.update_nyquist(seismic_data.nyquist_freq)

        # Populate available sort headers in control panel
        available_headers = self.gather_navigator.get_available_sort_headers()
        self.control_panel.set_available_sort_headers(available_headers)

        # Status message
        if self.gather_navigator.has_gathers():
            stats = self.gather_navigator.get_statistics()
            self.statusBar().showMessage(
                f"Loaded SEG-Y: {stats['n_gathers']} gathers, "
                f"{stats['total_traces']} total traces"
            )
        else:
            self.statusBar().showMessage(
                f"Loaded SEG-Y: {seismic_data.n_traces} traces, "
                f"{seismic_data.duration:.0f}ms"
            )

    def _on_gather_navigation(self, action: str):
        """Handle gather navigation request."""
        # Display current gather
        self._display_current_gather()

        # Auto-apply processing if enabled
        if self.auto_process_enabled and self.last_processor is not None:
            self._apply_processing(self.last_processor)

    def _on_auto_process_changed(self, enabled: bool):
        """Handle auto-process checkbox change."""
        self.auto_process_enabled = enabled

        # Update status message
        if enabled:
            self.statusBar().showMessage(
                "Auto-processing enabled - processing will apply automatically on navigation",
                3000
            )
        else:
            self.statusBar().showMessage(
                "Auto-processing disabled",
                3000
            )

    def _display_current_gather(self):
        """Display the current gather in all viewers."""
        try:
            # Get current gather data
            gather_data, gather_headers, gather_info = self.gather_navigator.get_current_gather()

            # Set as input data
            self.input_data = gather_data
            self.headers_df = gather_headers

            # Display in input viewer
            self.input_viewer.set_data(self.input_data)

            # Clear processed and difference viewers
            self.processed_viewer.clear()
            self.difference_viewer.clear()
            self.processed_data = None
            self.difference_data = None

            # Reset viewport to show all data of this gather
            self._reset_view()

            # Auto-scale amplitude range from data
            self._auto_scale_amplitude()

            # Update flip window if open
            self._update_flip_window()

            # Update status
            self.statusBar().showMessage(
                f"Viewing: {gather_info['description']} - "
                f"{gather_info['n_traces']} traces"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display gather:\n{str(e)}")

    def _auto_scale_amplitude(self):
        """Auto-scale amplitude range from current data."""
        if self.input_data is None:
            return

        # Calculate statistics from data
        # Use 98th percentile to avoid extreme outliers
        abs_data = np.abs(self.input_data.traces)
        max_amp = np.percentile(abs_data, 98.0)

        # Set symmetric range around zero
        min_amp = -max_amp
        max_amp = max_amp

        # Ensure non-zero range
        if max_amp == 0:
            max_amp = 1.0
            min_amp = -1.0

        # Update viewport state and control panel
        self.viewport_state.set_amplitude_range(min_amp, max_amp)
        self.control_panel.set_amplitude_range(min_amp, max_amp)

    def _load_from_zarr(self):
        """Load previously imported data from Zarr/Parquet."""
        from utils.segy_import.data_storage import DataStorage

        # Select directory
        data_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Zarr/Parquet Data Directory",
            ""
        )

        if not data_dir:
            return

        try:
            # Load data using lazy loading for memory efficiency
            from models.lazy_seismic_data import LazySeismicData

            lazy_data = LazySeismicData.from_storage_dir(data_dir)
            storage = DataStorage(data_dir)
            ensembles_df = storage.get_ensemble_index()

            # Extract original SEG-Y path from metadata if available
            original_segy_path = None
            if hasattr(lazy_data, 'metadata') and lazy_data.metadata:
                # Try to get it from top-level metadata first (new format)
                original_segy_path = lazy_data.metadata.get('original_segy_path')

                # Fall back to seismic_metadata section (streaming import format)
                if not original_segy_path:
                    seismic_meta = lazy_data.metadata.get('seismic_metadata', {})
                    original_segy_path = seismic_meta.get('original_segy_path') or seismic_meta.get('source_file')

            # Load into gather navigator and display
            self._on_segy_imported(lazy_data, None, ensembles_df, original_segy_path)

            # Show statistics
            stats = storage.get_statistics()
            if ensembles_df is not None and len(ensembles_df) > 0:
                msg = (f"Successfully loaded data from:\n{data_dir}\n\n"
                       f"Gathers: {len(ensembles_df)}\n"
                       f"Total Traces: {stats['headers']['n_traces']:,}\n\n"
                       f"Use the gather navigation panel to browse gathers.")
            else:
                msg = (f"Successfully loaded data from:\n{data_dir}\n\n"
                       f"Traces: {stats['headers']['n_traces']:,}")

            # Add note about original SEG-Y path availability
            if original_segy_path:
                if Path(original_segy_path).exists():
                    msg += f"\n\n✅ Original SEG-Y file found - batch processing available"
                else:
                    msg += f"\n\n⚠️  Original SEG-Y file not found at:\n{original_segy_path}\n\nBatch processing will require you to locate the file."
            else:
                msg += "\n\n⚠️  Original SEG-Y path not in metadata.\nTo enable batch processing, please re-import from SEG-Y."

            QMessageBox.information(self, "Data Loaded", msg)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load data:\n{str(e)}"
            )

    def _generate_sample_data(self):
        """Generate synthetic seismic data for testing."""
        from utils.sample_data import generate_sample_seismic_data

        try:
            # Generate sample data
            self.input_data = generate_sample_seismic_data(
                n_samples=1000,
                n_traces=100,
                sample_rate=2.0,  # 2ms
                noise_level=0.1
            )

            # Display in input viewer
            self.input_viewer.set_data(self.input_data)

            # Clear processed and difference viewers
            self.processed_viewer.clear()
            self.difference_viewer.clear()
            self.processed_data = None
            self.difference_data = None

            # Reset viewport to show all data
            self._reset_view()

            # Update control panel with Nyquist frequency
            self.control_panel.update_nyquist(self.input_data.nyquist_freq)

            # Update flip window if open
            self._update_flip_window()

            self.statusBar().showMessage(
                f"Generated sample data: {self.input_data.n_traces} traces, "
                f"{self.input_data.duration:.0f}ms, Nyquist={self.input_data.nyquist_freq:.1f}Hz"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate sample data:\n{str(e)}")

    def _on_sort_keys_changed(self, sort_keys: list):
        """
        Handle sort keys change from control panel.

        Args:
            sort_keys: List of header names to sort by
        """
        # Set sort keys in gather navigator
        self.gather_navigator.set_sort_keys(sort_keys)

        # Refresh current display with new sort
        self._display_current_gather()

        # Refresh flip window if open
        if hasattr(self, 'flip_window') and self.flip_window is not None:
            if self.flip_window.isVisible():
                self._update_flip_window()

        # Status message
        if sort_keys:
            sort_text = " → ".join(sort_keys)
            self.statusBar().showMessage(f"Applied in-gather sort: {sort_text}", 3000)
        else:
            self.statusBar().showMessage("Cleared in-gather sort", 3000)

    def _on_process_requested(self, processor: BaseProcessor):
        """
        Handle processing request from control panel.

        Args:
            processor: The processor to apply
        """
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load or generate seismic data first."
            )
            return

        # Store processor for auto-processing
        self.last_processor = processor

        # Apply the processing
        self._apply_processing(processor)

    def _apply_processing(self, processor: BaseProcessor):
        """
        Apply processing to current data.

        Args:
            processor: The processor to apply
        """
        if self.input_data is None:
            return

        try:
            self.statusBar().showMessage("Processing...")

            # Apply processor to input data
            self.processed_data = processor.process(self.input_data)

            # Calculate difference (residual)
            difference_traces = self.input_data.traces - self.processed_data.traces
            self.difference_data = SeismicData(
                traces=difference_traces,
                sample_rate=self.input_data.sample_rate,
                metadata={'description': 'Difference (Input - Processed)'}
            )

            # Update viewers
            self.processed_viewer.set_data(self.processed_data)
            self.difference_viewer.set_data(self.difference_data)

            # Update flip window if open
            self._update_flip_window()

            # Status message
            if self.auto_process_enabled:
                self.statusBar().showMessage(
                    f"Auto-processed: {processor.get_description()}"
                )
            else:
                self.statusBar().showMessage(
                    f"Processing complete: {processor.get_description()}"
                )

        except ValueError as e:
            QMessageBox.critical(
                self,
                "Processing Error",
                f"Failed to process data:\n{str(e)}"
            )
            self.statusBar().showMessage("Processing failed.")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Unexpected Error",
                f"An unexpected error occurred:\n{str(e)}"
            )
            self.statusBar().showMessage("Processing failed.")

    def _reset_view(self):
        """Reset viewport to show all data."""
        if self.input_data is not None:
            self.viewport_state.reset_to_data(
                self.input_data.duration,
                self.input_data.n_traces - 1
            )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Seismic QC Tool",
            "<h2>Seismic Data Processing QC Tool</h2>"
            "<p>Professional seismic data quality control application</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Three synchronized seismic viewers</li>"
            "<li>Zero-phase bandpass filtering</li>"
            "<li>Interactive zoom, pan, and gain controls</li>"
            "<li>Difference/residual analysis</li>"
            "<li>Flip window for quick view cycling</li>"
            "</ul>"
            "<p><b>Architecture Principles:</b></p>"
            "<ul>"
            "<li>Separation of data and visualization</li>"
            "<li>Extensible processing pipeline</li>"
            "<li>Amplitude-preserving workflows</li>"
            "<li>Professional seismic conventions</li>"
            "</ul>"
        )

    def _open_flip_window(self):
        """Open or focus the flip window."""
        if self.flip_window is None:
            # Create flip window
            self.flip_window = FlipWindow(self.viewport_state, self)
            # Connect close event to clear reference
            self.flip_window.destroyed.connect(self._on_flip_window_closed)

        # Update with current data
        self._update_flip_window()

        # Show and raise to front
        self.flip_window.show()
        self.flip_window.raise_()
        self.flip_window.activateWindow()

        # Status message
        self.statusBar().showMessage(
            "Flip window opened. LMB: cycle forward, RMB: cycle backward",
            3000
        )

    def _update_flip_window(self):
        """Update flip window with current data."""
        if self.flip_window is not None:
            # Only update if we have at least input data
            if self.input_data is not None:
                self.flip_window.set_data(
                    self.input_data,
                    self.processed_data,
                    self.difference_data
                )

    def _on_flip_window_closed(self):
        """Handle flip window being closed."""
        self.flip_window = None

    def _batch_process_all_gathers(self):
        """Batch process all gathers in the dataset with current processor."""
        # Check if we have multi-gather data
        if not self.gather_navigator.has_gathers():
            QMessageBox.warning(
                self,
                "No Multi-Gather Data",
                "Batch processing requires multi-gather SEG-Y data.\n\n"
                "Please load SEG-Y data with ensemble information."
            )
            return

        # Check if processor is configured
        if self.last_processor is None:
            QMessageBox.warning(
                self,
                "No Processor Configured",
                "Please apply processing to at least one gather first.\n\n"
                "This sets the processing parameters for batch mode.\n\n"
                "Steps:\n"
                "1. Configure filter parameters\n"
                "2. Click 'Apply Filter'\n"
                "3. Then run batch processing"
            )
            return

        # Confirm with user
        stats = self.gather_navigator.get_statistics()
        n_gathers = stats['n_gathers']
        total_traces = stats['total_traces']

        # Add memory warning for large datasets in lazy mode
        message = (
            f"This will process all {n_gathers} gathers with:\n\n"
            f"{self.last_processor.get_description()}\n\n"
            f"Total traces: {total_traces}\n\n"
        )

        if self.gather_navigator.lazy_data is not None:
            # Estimate memory usage (rough estimate: 4 bytes per sample)
            n_samples = self.gather_navigator.lazy_data.n_samples
            estimated_mb = (n_samples * total_traces * 4) / (1024 * 1024)
            message += (
                f"⚠️  Memory Note: Batch processing loads all processed\n"
                f"data into memory (~{estimated_mb:.0f} MB for output).\n\n"
            )

        message += "This may take some time. Continue?"

        reply = QMessageBox.question(
            self,
            "Batch Process All Gathers",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Processing gathers...",
            "Cancel",
            0,
            n_gathers,
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Batch Processing")
        progress.setMinimumDuration(0)

        try:
            # Get data source (works with both full data and lazy loading modes)
            if self.gather_navigator.full_data is not None:
                # Full data mode
                full_data = self.gather_navigator.full_data
                n_samples = full_data.n_samples
                n_traces = full_data.n_traces
                sample_rate = full_data.sample_rate
                metadata = full_data.metadata.copy()
            elif self.gather_navigator.lazy_data is not None:
                # Lazy loading mode
                lazy_data = self.gather_navigator.lazy_data
                n_samples = lazy_data.n_samples
                n_traces = lazy_data.n_traces
                sample_rate = lazy_data.sample_rate
                metadata = lazy_data.metadata.copy()
            else:
                raise ValueError("No dataset loaded")

            # Initialize array for processed data
            processed_traces = np.zeros((n_samples, n_traces), dtype=np.float32)

            # Initialize array for sorted headers (if sorting is enabled)
            sorted_headers_list = []

            # Process each gather
            current_gather_id = self.gather_navigator.current_gather_id

            for i in range(n_gathers):
                # Update progress
                progress.setValue(i)
                progress.setLabelText(f"Processing gather {i+1} of {n_gathers}...")

                # Check if cancelled
                if progress.wasCanceled():
                    QMessageBox.information(
                        self,
                        "Cancelled",
                        f"Batch processing cancelled after {i} gathers."
                    )
                    return

                # Navigate to gather
                self.gather_navigator.goto_gather(i)
                gather_data, gather_headers, gather_info = self.gather_navigator.get_current_gather()

                # Process this gather
                processed_gather = self.last_processor.process(gather_data)

                # Store in full array
                start_trace = gather_info['start_trace']
                end_trace = gather_info['end_trace']
                processed_traces[:, start_trace:end_trace+1] = processed_gather.traces

                # Store sorted headers for this gather (for export)
                sorted_headers_list.append(gather_headers)

            # Concatenate sorted headers for export
            if sorted_headers_list:
                import pandas as pd
                self.sorted_headers_df = pd.concat(sorted_headers_list, ignore_index=True)
            else:
                self.sorted_headers_df = None

            # Create full processed dataset
            self.full_processed_data = SeismicData(
                traces=processed_traces,
                sample_rate=sample_rate,
                metadata={
                    **metadata,
                    'description': f'Batch Processed - {self.last_processor.get_description()}',
                    'n_gathers': n_gathers,
                    'sorted': len(self.gather_navigator.sort_keys) > 0,
                    'sort_keys': self.gather_navigator.sort_keys.copy() if self.gather_navigator.sort_keys else []
                }
            )

            self.is_full_dataset_processed = True

            # Return to original gather
            self.gather_navigator.goto_gather(current_gather_id)
            self._display_current_gather()

            # If auto-processing enabled, apply to current gather
            if self.auto_process_enabled:
                self._apply_processing(self.last_processor)

            # Complete
            progress.setValue(n_gathers)

            QMessageBox.information(
                self,
                "Batch Processing Complete",
                f"Successfully processed all {n_gathers} gathers!\n\n"
                f"Processing: {self.last_processor.get_description()}\n"
                f"Total traces: {stats['total_traces']}\n\n"
                f"You can now export the processed dataset to SEG-Y."
            )

            self.statusBar().showMessage(
                f"Batch processing complete: {n_gathers} gathers processed",
                5000
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Batch Processing Error",
                f"Failed to process gathers:\n\n{str(e)}"
            )
            self.statusBar().showMessage("Batch processing failed.")

    def _export_processed_segy(self):
        """Export processed data to SEG-Y file."""
        # Check if we have full processed dataset
        if not self.is_full_dataset_processed or self.full_processed_data is None:
            QMessageBox.warning(
                self,
                "No Batch Processed Data",
                "Please run batch processing first before exporting.\n\n"
                "Steps:\n"
                "1. Load multi-gather SEG-Y data\n"
                "2. Configure and apply filter to one gather\n"
                "3. Run 'Processing → Batch Process All Gathers'\n"
                "4. Then export\n\n"
                "Note: Only batch processed data can be exported.\n"
                "Single gather export is not supported."
            )
            return

        # Check if we have original SEG-Y file path
        if self.original_segy_path is None:
            QMessageBox.warning(
                self,
                "No Original SEG-Y Path",
                "Export requires the original SEG-Y file for headers.\n\n"
                "This data was imported before the original path tracking\n"
                "feature was added. Please re-import the SEG-Y file to\n"
                "enable export functionality."
            )
            return

        # Validate that the original SEG-Y file still exists
        if not Path(self.original_segy_path).exists():
            reply = QMessageBox.critical(
                self,
                "Original SEG-Y File Not Found",
                f"The original SEG-Y file is required for export but cannot be found:\n\n"
                f"{self.original_segy_path}\n\n"
                f"Do you want to browse for the file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                new_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Locate Original SEG-Y File",
                    str(Path(self.original_segy_path).parent),
                    "SEG-Y Files (*.sgy *.segy);;All Files (*)"
                )

                if new_path and Path(new_path).exists():
                    self.original_segy_path = new_path
                    # Don't recursively call here - just let user try export again
                    QMessageBox.information(
                        self,
                        "Path Updated",
                        f"Original SEG-Y path updated. Please try exporting again."
                    )
                return
            else:
                return

        # Ask user for output file
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed SEG-Y",
            "",
            "SEG-Y Files (*.sgy *.segy);;All Files (*)"
        )

        if not output_file:
            return  # User cancelled

        try:
            from utils.segy_import.segy_export import export_processed_segy

            # Show progress message
            self.statusBar().showMessage("Exporting processed SEG-Y...")

            # Get full headers - use sorted headers if sorting was applied
            if self.sorted_headers_df is not None:
                full_headers_df = self.sorted_headers_df
                sort_info = f" (sorted by: {', '.join(self.gather_navigator.sort_keys)})"
            else:
                full_headers_df = self.gather_navigator.headers_df
                sort_info = ""

            # Export full processed dataset
            export_processed_segy(
                output_path=output_file,
                original_segy_path=self.original_segy_path,
                processed_data=self.full_processed_data,
                headers_df=full_headers_df
            )

            # Success message
            self.statusBar().showMessage(
                f"Successfully exported to: {output_file}",
                5000
            )

            stats = self.gather_navigator.get_statistics()
            QMessageBox.information(
                self,
                "Export Complete",
                f"Batch processed data successfully exported to:\n\n{output_file}\n\n"
                f"Gathers: {stats['n_gathers']}\n"
                f"Total Traces: {self.full_processed_data.n_traces}\n"
                f"Samples: {self.full_processed_data.n_samples}\n"
                f"Sample rate: {self.full_processed_data.sample_rate} ms\n\n"
                f"Processing: {self.full_processed_data.metadata.get('description', 'N/A')}\n"
                f"{sort_info}\n\n"
                f"All original headers have been preserved."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export SEG-Y file:\n\n{str(e)}"
            )
            self.statusBar().showMessage("Export failed.")

    def _batch_process_and_export_streaming(self):
        """
        Memory-efficient batch processing and export.

        Processes all gathers and exports directly to SEG-Y without loading
        everything into memory. Uses temporary Zarr storage and chunked export.

        Memory usage: ~100-200 MB regardless of dataset size.
        """
        # Check if we have multi-gather data
        if not self.gather_navigator.has_gathers():
            QMessageBox.warning(
                self,
                "No Multi-Gather Data",
                "This feature requires multi-gather SEG-Y data.\n\n"
                "Please load SEG-Y data with ensemble information."
            )
            return

        # Check if processor is configured
        if self.last_processor is None:
            QMessageBox.warning(
                self,
                "No Processor Configured",
                "Please apply processing to at least one gather first.\n\n"
                "This sets the processing parameters.\n\n"
                "Steps:\n"
                "1. Configure filter parameters\n"
                "2. Click 'Apply Filter'\n"
                "3. Then use this feature"
            )
            return

        # Check if we have original SEG-Y file path
        if self.original_segy_path is None:
            QMessageBox.warning(
                self,
                "No Original SEG-Y Path",
                "Export requires the original SEG-Y file for headers.\n\n"
                "This data was imported before the original path tracking\n"
                "feature was added. Please re-import the SEG-Y file to\n"
                "enable batch processing and export."
            )
            return

        # Validate that the original SEG-Y file still exists
        if not Path(self.original_segy_path).exists():
            # Show helpful error with the path
            reply = QMessageBox.critical(
                self,
                "Original SEG-Y File Not Found",
                f"The original SEG-Y file is required for export but cannot be found:\n\n"
                f"{self.original_segy_path}\n\n"
                f"This file may have been moved or deleted.\n\n"
                f"Options:\n"
                f"• Move the file back to its original location\n"
                f"• Re-import the SEG-Y data from its current location\n\n"
                f"Do you want to browse for the file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Let user browse for the file
                new_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Locate Original SEG-Y File",
                    str(Path(self.original_segy_path).parent),
                    "SEG-Y Files (*.sgy *.segy);;All Files (*)"
                )

                if new_path and Path(new_path).exists():
                    self.original_segy_path = new_path
                    QMessageBox.information(
                        self,
                        "Path Updated",
                        f"Original SEG-Y path updated to:\n{new_path}\n\n"
                        f"You can now proceed with batch processing."
                    )
                    # Recursively call this function to continue
                    self._batch_process_and_export_streaming()
                return
            else:
                return

        # Get statistics
        stats = self.gather_navigator.get_statistics()
        n_gathers = stats['n_gathers']
        total_traces = stats['total_traces']

        # Get data dimensions
        if self.gather_navigator.lazy_data is not None:
            n_samples = self.gather_navigator.lazy_data.n_samples
        elif self.gather_navigator.full_data is not None:
            n_samples = self.gather_navigator.full_data.n_samples
        else:
            QMessageBox.warning(self, "Error", "No dataset loaded")
            return

        # Estimate processing time (rough estimate)
        estimated_minutes = (total_traces / 10000) * 0.5  # Very rough estimate

        # Ask user for output file FIRST (before doing any work)
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed SEG-Y (Memory Efficient)",
            "",
            "SEG-Y Files (*.sgy *.segy);;All Files (*)"
        )

        if not output_file:
            return  # User cancelled

        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Process and Export (Memory Efficient)",
            f"This will process and export all {n_gathers} gathers:\n\n"
            f"{self.last_processor.get_description()}\n\n"
            f"Total traces: {total_traces:,}\n"
            f"Estimated time: ~{estimated_minutes:.1f} minutes\n\n"
            f"✅ Memory efficient: Uses only ~100-200 MB\n"
            f"✅ Processes and exports in one pass\n"
            f"✅ No memory limits for large datasets\n\n"
            f"Output: {output_file}\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create temporary directory for processed zarr
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix='processed_zarr_')
        processed_zarr_path = Path(temp_dir) / 'processed_traces.zarr'

        # Create progress dialog
        progress = QProgressDialog(
            "Processing and exporting gathers...",
            "Cancel",
            0,
            n_gathers + 1,  # +1 for export step
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Memory-Efficient Processing")
        progress.setMinimumDuration(0)

        try:
            # Step 1: Initialize Zarr array for processed data
            self.statusBar().showMessage("Initializing output array...")
            progress.setLabelText("Initializing output array...")
            progress.setValue(0)

            processed_zarr = zarr.open(
                str(processed_zarr_path),
                mode='w',
                shape=(n_samples, total_traces),
                chunks=(n_samples, 1000),  # Chunk by traces
                dtype=np.float32
            )

            # Step 2: Process each gather and write to Zarr
            current_gather_id = self.gather_navigator.current_gather_id
            import time
            start_time = time.time()

            for i in range(n_gathers):
                # Update progress
                progress.setValue(i)
                elapsed = time.time() - start_time
                if i > 0:
                    rate = i / elapsed
                    remaining = (n_gathers - i) / rate if rate > 0 else 0
                    progress.setLabelText(
                        f"Processing gather {i+1}/{n_gathers}...\n"
                        f"Time remaining: ~{remaining/60:.1f} minutes"
                    )
                else:
                    progress.setLabelText(f"Processing gather {i+1}/{n_gathers}...")

                # Check if cancelled
                if progress.wasCanceled():
                    QMessageBox.information(
                        self,
                        "Cancelled",
                        f"Processing cancelled after {i} gathers.\n"
                        f"Temporary files will be cleaned up."
                    )
                    return

                # Navigate to gather
                self.gather_navigator.goto_gather(i)
                gather_data, gather_headers, gather_info = self.gather_navigator.get_current_gather()

                # Process this gather
                processed_gather = self.last_processor.process(gather_data)

                # Write to Zarr array
                start_trace = gather_info['start_trace']
                end_trace = gather_info['end_trace']
                processed_zarr[:, start_trace:end_trace+1] = processed_gather.traces

            # Step 3: Export using chunked export
            progress.setValue(n_gathers)
            progress.setLabelText("Exporting to SEG-Y (chunked)...")
            self.statusBar().showMessage("Exporting to SEG-Y...")

            from utils.segy_import.segy_export import export_from_zarr_chunked

            def export_progress_callback(current, total, time_remaining):
                """Update progress during export."""
                if not progress.wasCanceled():
                    progress.setLabelText(
                        f"Exporting traces {current:,}/{total:,}...\n"
                        f"Time remaining: ~{time_remaining/60:.1f} minutes"
                    )
                    QApplication.processEvents()  # Keep UI responsive

            export_from_zarr_chunked(
                output_path=output_file,
                original_segy_path=self.original_segy_path,
                processed_zarr_path=str(processed_zarr_path),
                chunk_size=5000,
                progress_callback=export_progress_callback
            )

            # Return to original gather
            self.gather_navigator.goto_gather(current_gather_id)
            self._display_current_gather()

            # Complete
            progress.setValue(n_gathers + 1)
            total_time = time.time() - start_time

            self.statusBar().showMessage(
                f"Processing and export complete: {n_gathers} gathers in {total_time/60:.1f} minutes",
                5000
            )

            QMessageBox.information(
                self,
                "Processing Complete",
                f"Successfully processed and exported all {n_gathers} gathers!\n\n"
                f"Processing: {self.last_processor.get_description()}\n"
                f"Total traces: {total_traces:,}\n"
                f"Total time: {total_time/60:.1f} minutes\n"
                f"Rate: {total_traces/total_time:.0f} traces/second\n\n"
                f"Output file: {output_file}\n\n"
                f"✅ Memory-efficient processing completed successfully!"
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Processing Error",
                f"Failed to process and export:\n\n{str(e)}"
            )
            self.statusBar().showMessage("Processing failed.")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up temporary zarr directory
            progress.close()
            if processed_zarr_path.parent.exists():
                try:
                    shutil.rmtree(processed_zarr_path.parent)
                    print(f"Cleaned up temporary directory: {processed_zarr_path.parent}")
                except Exception as e:
                    print(f"Warning: Could not clean up temp directory: {e}")

    # FK Filter handlers

    def _on_fk_design_requested(self):
        """Handle FK filter design request - open designer dialog."""
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load seismic data first before designing FK filter."
            )
            return

        # Get trace spacing from current gather
        trace_spacing = self._get_trace_spacing()

        # Get current gather info and headers for display
        gather_headers = None
        if self.gather_navigator.has_gathers():
            _, gather_headers, gather_info = self.gather_navigator.get_current_gather()
            gather_index = gather_info['gather_id']
        else:
            gather_index = 0

        # Open FK Designer dialog
        dialog = FKDesignerDialog(
            gather_data=self.input_data,
            gather_index=gather_index,
            trace_spacing=trace_spacing,
            gather_headers=gather_headers,
            parent=self
        )

        # Connect signal to refresh config list when saved
        dialog.config_saved.connect(lambda _: self.control_panel.refresh_fk_configs())

        dialog.exec()

    def _on_fk_config_selected(self, config_name: str):
        """Handle FK filter configuration selection for applying (with sub-gathers and AGC)."""
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load seismic data first."
            )
            return

        # Load configuration
        fk_manager = FKConfigManager()
        config = fk_manager.get_config(config_name)
        if config is None:
            QMessageBox.warning(
                self,
                "Configuration Not Found",
                f"FK filter configuration '{config_name}' not found."
            )
            return

        try:
            # Check if using sub-gathers
            if config.use_subgathers and config.boundary_header:
                # Apply FK filter with sub-gathers
                self._apply_fk_with_subgathers(config)
            else:
                # Standard full-gather processing
                self._apply_fk_full_gather(config)

            # Store config for auto-processing
            # (Wrapped to handle sub-gathers automatically)
            self.last_processor = config  # Store config instead of processor

        except Exception as e:
            QMessageBox.critical(
                self,
                "FK Filter Error",
                f"Failed to apply FK filter:\n\n{str(e)}"
            )

    def _apply_fk_full_gather(self, config: FKFilterConfig):
        """Apply FK filter to full gather (no sub-gathers)."""
        from processors.agc import apply_agc_to_gather, remove_agc

        # Get trace spacing
        trace_spacing = self._get_trace_spacing()

        # Get input traces
        input_traces = self.input_data.traces

        # Apply AGC if configured
        agc_scale_factors = None
        if config.apply_agc:
            # Convert sample rate from milliseconds to Hz
            sample_rate_hz = 1000.0 / self.input_data.sample_rate
            input_traces, agc_scale_factors = apply_agc_to_gather(
                input_traces,
                sample_rate_hz,
                config.agc_window_ms
            )

        # Create temporary SeismicData with AGC-applied traces (or original)
        temp_data = SeismicData(
            traces=input_traces,
            sample_rate=self.input_data.sample_rate,
            metadata={'agc_applied': config.apply_agc}
        )

        # Create FK processor
        processor = FKFilter(**config.to_processor_params(trace_spacing))

        # Apply filter
        filtered_data = processor.process(temp_data)

        # Remove AGC if it was applied
        if config.apply_agc and agc_scale_factors is not None:
            filtered_traces = remove_agc(filtered_data.traces, agc_scale_factors)
            self.processed_data = SeismicData(
                traces=filtered_traces,
                sample_rate=filtered_data.sample_rate,
                headers=self.input_data.headers,
                metadata={'description': f'FK filtered (AGC removed): {config.name}'}
            )
        else:
            self.processed_data = filtered_data

        # Calculate difference
        difference_traces = self.input_data.traces - self.processed_data.traces
        self.difference_data = SeismicData(
            traces=difference_traces,
            sample_rate=self.input_data.sample_rate,
            metadata={'description': 'Difference (Input - Processed)'}
        )

        # Update viewers
        self.processed_viewer.set_data(self.processed_data)
        self.difference_viewer.set_data(self.difference_data)

        # Status message
        self.statusBar().showMessage(f"FK filter applied: {config.name}")

    def _apply_fk_with_subgathers(self, config: FKFilterConfig):
        """Apply FK filter with sub-gather boundaries."""
        from utils.subgather_detector import (
            detect_subgathers, extract_subgather_traces
        )
        from utils.trace_spacing import calculate_subgather_trace_spacing_with_stats
        from processors.agc import apply_agc_to_gather, remove_agc

        # Get gather headers
        _, gather_headers, _ = self.gather_navigator.get_current_gather()

        if gather_headers is None:
            raise ValueError("Sub-gathers require trace headers, but none available")

        # Detect sub-gathers
        subgathers = detect_subgathers(
            gather_headers,
            config.boundary_header,
            min_traces=8
        )

        self.statusBar().showMessage(
            f"Processing {len(subgathers)} sub-gathers with FK filter..."
        )

        # Process each sub-gather
        full_filtered_traces = np.zeros_like(self.input_data.traces)

        for sg in subgathers:
            # Extract sub-gather traces (view, not copy)
            sg_traces = extract_subgather_traces(self.input_data.traces, sg)

            # Calculate trace spacing for this sub-gather using enhanced calculation
            # This properly handles coordinates with SEGY scalar support
            sg_stats = calculate_subgather_trace_spacing_with_stats(
                gather_headers,
                sg.start_trace,
                sg.end_trace,
                default_spacing=25.0
            )
            sg_spacing = sg_stats.spacing

            # Print spacing info for this sub-gather
            print(f"  Sub-gather '{sg.description}': spacing = {sg_spacing:.2f} m (from {sg_stats.coordinate_source})")

            # Apply AGC if configured
            agc_scale_factors = None
            if config.apply_agc:
                # Convert sample rate from milliseconds to Hz
                sample_rate_hz = 1000.0 / self.input_data.sample_rate
                sg_traces, agc_scale_factors = apply_agc_to_gather(
                    sg_traces,
                    sample_rate_hz,
                    config.agc_window_ms
                )

            # Create SeismicData for sub-gather
            sg_data = SeismicData(
                traces=sg_traces,
                sample_rate=self.input_data.sample_rate,
                metadata={'subgather': sg.description}
            )

            # Create FK processor for this sub-gather
            processor = FKFilter(**config.to_processor_params(sg_spacing))

            # Apply filter
            sg_filtered = processor.process(sg_data)

            # Remove AGC if applied
            if config.apply_agc and agc_scale_factors is not None:
                sg_filtered_traces = remove_agc(sg_filtered.traces, agc_scale_factors)
            else:
                sg_filtered_traces = sg_filtered.traces

            # Place filtered traces back into full gather
            full_filtered_traces[:, sg.start_trace:sg.end_trace + 1] = sg_filtered_traces

        # Create filtered SeismicData
        self.processed_data = SeismicData(
            traces=full_filtered_traces,
            sample_rate=self.input_data.sample_rate,
            headers=self.input_data.headers,
            metadata={'description': f'FK filtered (sub-gathers): {config.name}'}
        )

        # Calculate difference
        difference_traces = self.input_data.traces - self.processed_data.traces
        self.difference_data = SeismicData(
            traces=difference_traces,
            sample_rate=self.input_data.sample_rate,
            metadata={'description': 'Difference (Input - Processed)'}
        )

        # Update viewers
        self.processed_viewer.set_data(self.processed_data)
        self.difference_viewer.set_data(self.difference_data)

        # Update flip window if open
        self._update_flip_window()

        # Status message
        self.statusBar().showMessage(
            f"FK filter applied: {config.name} ({len(subgathers)} sub-gathers)"
        )

    def _get_trace_spacing(self) -> float:
        """
        Get trace spacing for current gather.

        Uses enhanced calculation with SEGY scalar support and multiple
        coordinate sources (ReceiverX, GroupX, SourceX, etc.).

        Returns:
            Trace spacing in meters (default 25m if not available)
        """
        try:
            # Try to get from gather headers
            _, gather_headers, _ = self.gather_navigator.get_current_gather()

            # Use enhanced trace spacing calculation with SEGY scalar support
            stats = calculate_trace_spacing_with_stats(gather_headers, default_spacing=25.0)

            # Print statistics for verification
            print(f"Trace spacing: {stats.spacing:.2f} m (from {stats.coordinate_source})")
            if stats.coordinate_source not in ['default', 'd3', 'provided']:
                print(f"  SEGY scalar: {stats.scalar_applied}, Quality: {stats.n_spacings} measurements")
                if stats.n_spacings > 0:
                    cv = (stats.std / stats.mean) * 100 if stats.mean > 0 else 0
                    print(f"  Mean: {stats.mean:.2f} m, Std: {stats.std:.2f} m, CV: {cv:.1f}%")

            return stats.spacing

        except Exception as e:
            print(f"Warning: Could not determine trace spacing from headers: {e}")
            return 25.0


# Import will be added after creating sample data module
