"""
Main application window - coordinates three synchronized seismic viewers.

Integrates:
- DatasetNavigator for multi-dataset management
- AppSettings for persistent session state
- GatherNavigator for ensemble navigation
"""
import logging
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QMenuBar, QMenu, QFileDialog, QMessageBox, QSplitter,
                              QPushButton, QProgressDialog, QApplication)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QShortcut, QKeySequence
import numpy as np
import pandas as pd
import sys
import zarr
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Models
from models.seismic_data import SeismicData
from models.viewport_state import ViewportState
from models.gather_navigator import GatherNavigator
from models.dataset_navigator import DatasetNavigator
from models.app_settings import get_settings
from models.lazy_seismic_data import LazySeismicData

# Views
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from views.control_panel import ControlPanel
from views.gather_navigation_panel import GatherNavigationPanel
from views.segy_import_dialog import SEGYImportDialog
from views.flip_window import FlipWindow
from views.fk_designer_dialog import FKDesignerDialog
from views.fkk_designer_dialog import FKKDesignerDialog
from views.volume_header_dialog import VolumeHeaderDialog, build_volume_with_dialog
from models.seismic_volume import SeismicVolume, create_synthetic_volume
from models.fkk_config import FKKConfig

# Processors
from processors.base_processor import BaseProcessor
from processors.fk_filter import FKFilter
from processors.fkk_filter_gpu import get_fkk_filter
from models.fk_config import FKConfigManager, FKFilterConfig

# Utils
from utils.trace_spacing import calculate_trace_spacing_with_stats
from utils.storage_manager import ProcessingStorageManager
from utils.volume_builder import build_volume_from_gathers, extract_traces_from_volume


class MainWindow(QMainWindow):
    """
    Main application window with three synchronized seismic viewers.

    Layout:
    - Left: Control panel
    - Right: Three horizontally arranged seismic viewers (Input, Processed, Difference)
    """

    def __init__(self):
        super().__init__()
        logger.debug("MainWindow.__init__ START")

        self.setWindowTitle("Seismic Data Processing QC Tool")
        self.setGeometry(100, 100, 1600, 900)

        # Application settings (singleton) - replaces old QSettings usage
        self.app_settings = get_settings()

        # Recent files (migrated to use app_settings internally)
        self.max_recent_files = 10
        self.recent_files = self._load_recent_files()

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

        # Dataset navigator - manages multiple loaded datasets
        self.dataset_navigator = DatasetNavigator(
            max_cached_datasets=self.app_settings.get_dataset_cache_limit()
        )
        self._connect_dataset_navigator_signals()

        # Gather navigation - manages navigation within a single dataset
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
        self._setup_keyboard_shortcuts()

        # Restore last session if enabled
        if self.app_settings.get_auto_load_last_dataset():
            self._restore_last_session()
        else:
            self.statusBar().showMessage("Ready. Load seismic data to begin.")

        logger.debug("MainWindow.__init__ COMPLETE")

    def _connect_dataset_navigator_signals(self):
        """Connect DatasetNavigator signals to handlers."""
        self.dataset_navigator.active_dataset_changed.connect(
            self._on_active_dataset_changed
        )
        self.dataset_navigator.dataset_added.connect(
            self._on_dataset_added
        )
        self.dataset_navigator.dataset_removed.connect(
            self._on_dataset_removed
        )
        self.dataset_navigator.datasets_cleared.connect(
            self._on_datasets_cleared
        )

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
        self.control_panel.fkk_design_requested.connect(self._on_fkk_design_requested)
        self.control_panel.fkk_apply_requested.connect(self._on_fkk_apply_requested)
        # PSTM signals
        self.control_panel.pstm_apply_requested.connect(self._on_pstm_apply_requested)
        self.control_panel.pstm_wizard_requested.connect(self._on_pstm_wizard_requested)
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

        # Recent files submenu
        self.recent_menu = file_menu.addMenu("Recent &Files")
        self._update_recent_files_menu()

        # Datasets submenu (multi-dataset management)
        self.datasets_menu = file_menu.addMenu("&Datasets")
        self._update_datasets_menu()

        file_menu.addSeparator()

        # Close current dataset
        close_dataset_action = QAction("&Close Current Dataset", self)
        close_dataset_action.setShortcut("Ctrl+W")
        close_dataset_action.triggered.connect(self._close_current_dataset)
        file_menu.addAction(close_dataset_action)

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
        batch_process_action = QAction("&Batch Process All Gathers (Legacy)", self)
        batch_process_action.setShortcut("Ctrl+B")
        batch_process_action.setToolTip("DEPRECATED: Use 'Parallel Batch Process' for 10-14x better performance")
        batch_process_action.triggered.connect(self._batch_process_all_gathers)
        process_menu.addAction(batch_process_action)

        # Memory-efficient batch process and export action
        batch_export_action = QAction("Process and Export (Legacy)...", self)
        batch_export_action.setShortcut("Ctrl+Shift+E")
        batch_export_action.setToolTip("DEPRECATED: Use 'Parallel Batch Process' + 'Parallel Export' for better performance")
        batch_export_action.triggered.connect(self._batch_process_and_export_streaming)
        process_menu.addAction(batch_export_action)

        process_menu.addSeparator()

        # Parallel batch processing action (new optimized version)
        parallel_process_action = QAction("&Parallel Batch Process...", self)
        parallel_process_action.setShortcut("Ctrl+Shift+B")
        parallel_process_action.setToolTip("Process all gathers using parallel multiprocessing (10-14x faster)")
        parallel_process_action.triggered.connect(self._batch_process_parallel)
        process_menu.addAction(parallel_process_action)

        # Parallel export action
        parallel_export_action = QAction("Parallel &Export SEG-Y...", self)
        parallel_export_action.setShortcut("Ctrl+Alt+E")
        parallel_export_action.setToolTip("Export processed data to SEG-Y using parallel multiprocessing (6-10x faster)")
        parallel_export_action.triggered.connect(self._export_parallel)
        process_menu.addAction(parallel_export_action)

        process_menu.addSeparator()

        # 3D FKK Filter section
        fkk_action = QAction("3D &FKK Filter Designer...", self)
        fkk_action.setShortcut("Ctrl+Shift+F")
        fkk_action.setToolTip("Open 3D FKK filter designer for velocity cone filtering of 3D volumes")
        fkk_action.triggered.connect(self._open_fkk_designer)
        process_menu.addAction(fkk_action)

        # Generate synthetic 3D volume for testing
        fkk_test_action = QAction("Generate Test 3D Volume...", self)
        fkk_test_action.setToolTip("Generate synthetic 3D volume for testing FKK filter")
        fkk_test_action.triggered.connect(self._generate_test_3d_volume)
        process_menu.addAction(fkk_test_action)

        process_menu.addSeparator()

        # Migration section
        pstm_action = QAction("&Kirchhoff PSTM...", self)
        pstm_action.setShortcut("Ctrl+M")
        pstm_action.setToolTip("Apply Pre-Stack Time Migration to current gather")
        pstm_action.triggered.connect(self._open_pstm_from_menu)
        process_menu.addAction(pstm_action)

        pstm_wizard_action = QAction("PSTM &Wizard...", self)
        pstm_wizard_action.setShortcut("Ctrl+Shift+M")
        pstm_wizard_action.setToolTip("Open the full PSTM configuration wizard")
        pstm_wizard_action.triggered.connect(self._on_pstm_wizard_requested)
        process_menu.addAction(pstm_wizard_action)

        resume_migration_action = QAction("&Resume Migration...", self)
        resume_migration_action.setToolTip("Resume an interrupted migration job from checkpoint")
        resume_migration_action.triggered.connect(self._resume_migration)
        process_menu.addAction(resume_migration_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Flip window action
        flip_action = QAction("Open &Flip Window", self)
        flip_action.setShortcut("Ctrl+F")
        flip_action.setToolTip("Open flip window to cycle through Input/Processed/Difference views")
        flip_action.triggered.connect(self._open_flip_window)
        view_menu.addAction(flip_action)

        # ISA window action
        isa_action = QAction("Open &ISA Window", self)
        isa_action.setShortcut("Ctrl+I")
        isa_action.setToolTip("Open Interactive Spectral Analysis window")
        isa_action.triggered.connect(self._open_isa_window)
        view_menu.addAction(isa_action)

        view_menu.addSeparator()

        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")

        self.light_theme_action = QAction("&Light Mode", self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.triggered.connect(lambda: self._set_theme('light'))
        theme_menu.addAction(self.light_theme_action)

        self.dark_theme_action = QAction("&Dark Mode", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.triggered.connect(lambda: self._set_theme('dark'))
        theme_menu.addAction(self.dark_theme_action)

        # Set initial check state
        self._update_theme_menu_state()

        # Edit menu (Settings)
        edit_menu = menubar.addMenu("&Edit")

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setToolTip("Open application settings")
        settings_action.triggered.connect(self._open_settings)
        edit_menu.addAction(settings_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts for common actions."""
        # Gather navigation shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._on_prev_gather_shortcut)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._on_next_gather_shortcut)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, self._on_first_gather_shortcut)
        QShortcut(QKeySequence(Qt.Key.Key_End), self, self._on_last_gather_shortcut)

        # Processing shortcuts
        QShortcut(QKeySequence("Ctrl+P"), self, self._on_apply_filter_shortcut)
        QShortcut(QKeySequence("Ctrl+R"), self, self._reset_view)

        # View shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_flip_view)
        QShortcut(QKeySequence("Ctrl++"), self, lambda: self.viewport_state.zoom_in(0.5))
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self.viewport_state.zoom_out(2.0))

    def _on_prev_gather_shortcut(self):
        """Handle left arrow key - go to previous gather."""
        if self.gather_navigator.has_gathers() and self.gather_navigator.can_go_previous():
            self.gather_navigator.previous_gather()
            self._on_gather_navigation("prev")

    def _on_next_gather_shortcut(self):
        """Handle right arrow key - go to next gather."""
        if self.gather_navigator.has_gathers() and self.gather_navigator.can_go_next():
            self.gather_navigator.next_gather()
            self._on_gather_navigation("next")

    def _on_first_gather_shortcut(self):
        """Handle Home key - go to first gather."""
        if self.gather_navigator.has_gathers():
            self.gather_navigator.goto_gather(0)
            self._on_gather_navigation("first")

    def _on_last_gather_shortcut(self):
        """Handle End key - go to last gather."""
        if self.gather_navigator.has_gathers():
            self.gather_navigator.goto_gather(self.gather_navigator.n_gathers - 1)
            self._on_gather_navigation("last")

    def _on_apply_filter_shortcut(self):
        """Handle Ctrl+P - apply current filter settings."""
        if self.input_data is not None and self.last_processor is not None:
            self._apply_processing(self.last_processor)
        elif self.input_data is not None:
            # Trigger apply button click on control panel
            self.control_panel._on_apply_clicked()

    def _toggle_flip_view(self):
        """Handle Space key - toggle flip window or cycle view."""
        if self.flip_window is not None and self.flip_window.isVisible():
            self.flip_window.cycle_view()
        else:
            self._open_flip_window()

    # Recent files management

    def _load_recent_files(self) -> list:
        """Load recent files list from settings."""
        files = self.app_settings.get_recent_files()
        if files is None:
            return []
        # Filter out files that no longer exist
        return [f for f in files if Path(f).exists()]

    def _save_recent_files(self):
        """Save recent files list to settings."""
        # Update app_settings with current list
        for f in self.recent_files:
            self.app_settings.add_recent_file(f)

    def _add_to_recent_files(self, file_path: str):
        """Add a file to the recent files list."""
        file_path = str(Path(file_path).resolve())

        # Remove if already in list
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)

        # Add to beginning of list
        self.recent_files.insert(0, file_path)

        # Trim list to max size
        self.recent_files = self.recent_files[:self.max_recent_files]

        # Save to app_settings and update menu
        self.app_settings.add_recent_file(file_path)
        self._update_recent_files_menu()

    def _update_recent_files_menu(self):
        """Update the recent files submenu."""
        self.recent_menu.clear()

        if not self.recent_files:
            no_files_action = QAction("(No recent files)", self)
            no_files_action.setEnabled(False)
            self.recent_menu.addAction(no_files_action)
        else:
            for i, file_path in enumerate(self.recent_files):
                # Show filename with number shortcut
                file_name = Path(file_path).name
                action = QAction(f"&{i+1}. {file_name}", self)
                action.setToolTip(file_path)
                action.setData(file_path)
                action.triggered.connect(lambda checked, path=file_path: self._open_recent_file(path))
                self.recent_menu.addAction(action)

            # Add separator and clear option
            self.recent_menu.addSeparator()
            clear_action = QAction("&Clear Recent Files", self)
            clear_action.triggered.connect(self._clear_recent_files)
            self.recent_menu.addAction(clear_action)

    def _open_recent_file(self, file_path: str):
        """Open a file from the recent files list."""
        path = Path(file_path)

        if not path.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The file no longer exists:\n{file_path}"
            )
            # Remove from recent files
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self._save_recent_files()
                self._update_recent_files_menu()
            return

        # Determine file type and open appropriately
        if path.is_dir() and (path / 'traces.zarr').exists():
            # Zarr storage directory
            self._load_from_zarr_path(file_path)
        elif path.suffix.lower() in ['.sgy', '.segy']:
            # SEG-Y file - open import dialog with this file pre-selected
            self._import_segy_with_path(file_path)
        else:
            QMessageBox.warning(
                self,
                "Unknown File Type",
                f"Cannot determine how to open:\n{file_path}"
            )

    def _import_segy_with_path(self, file_path: str):
        """Open SEG-Y import dialog with a pre-selected file."""
        try:
            dialog = SEGYImportDialog(self, initial_file=file_path)
            dialog.import_completed.connect(self._on_segy_imported)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to open SEG-Y file: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open SEG-Y file:\n{e}")

    def _load_from_zarr_path(self, dir_path: str):
        """Load data from a Zarr storage directory."""
        try:
            from models.lazy_seismic_data import LazySeismicData

            lazy_data = LazySeismicData.from_storage_dir(dir_path)

            # Load ensemble index
            ensembles_path = Path(dir_path) / 'ensemble_index.parquet'
            if ensembles_path.exists():
                import pyarrow.parquet as pq
                ensembles_df = pq.read_table(ensembles_path).to_pandas()
            else:
                ensembles_df = None

            # Load into navigator
            self.gather_navigator.load_lazy_data(lazy_data, ensembles_df)
            self._display_current_gather()

            # Populate sort keys from headers
            self._populate_sort_keys_from_storage(Path(dir_path))

            # Update recent files
            self._add_to_recent_files(dir_path)

            self.statusBar().showMessage(f"Loaded from: {dir_path}")

        except Exception as e:
            logger.error(f"Failed to load Zarr data: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{e}")

    def _populate_sort_keys_from_storage(self, storage_dir: Path):
        """
        Populate the control panel sort keys list from headers.parquet.

        Args:
            storage_dir: Path to storage directory containing headers.parquet
        """
        headers_path = storage_dir / 'headers.parquet'
        sort_keys = []

        if headers_path.exists():
            try:
                import pandas as pd
                headers_df = pd.read_parquet(headers_path)
                # Get numeric columns suitable for sorting
                numeric_cols = headers_df.select_dtypes(include=['number']).columns.tolist()
                # Prioritize common sort keys
                priority_keys = ['offset', 'TRACE_SEQUENCE_LINE', 'TraceNumber', 'FieldRecord',
                                'CDP', 'SOURCE_X', 'SOURCE_Y', 'GROUP_X', 'GROUP_Y']
                sort_keys = [k for k in priority_keys if k in numeric_cols]
                sort_keys += [k for k in numeric_cols if k not in priority_keys]
                logger.debug(f"Found {len(sort_keys)} sortable columns")
            except Exception as e:
                logger.warning(f"Could not read headers for sort keys: {e}")

        # Fallback to defaults if no headers found
        if not sort_keys:
            sort_keys = ['offset', 'TraceNumber', 'FieldRecord', 'CDP', 'TRACE_SEQUENCE_LINE']
            logger.info("Using default sort keys")

        # Update control panel
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.set_available_sort_headers(sort_keys)

    def _populate_sort_keys_from_dataframe(self, headers_df):
        """
        Populate the control panel sort keys list from a headers DataFrame.

        Args:
            headers_df: pandas DataFrame with header columns
        """
        sort_keys = []

        try:
            # Get numeric columns suitable for sorting
            numeric_cols = headers_df.select_dtypes(include=['number']).columns.tolist()
            # Prioritize common sort keys
            priority_keys = ['offset', 'TRACE_SEQUENCE_LINE', 'TraceNumber', 'FieldRecord',
                            'CDP', 'SOURCE_X', 'SOURCE_Y', 'GROUP_X', 'GROUP_Y']
            sort_keys = [k for k in priority_keys if k in numeric_cols]
            sort_keys += [k for k in numeric_cols if k not in priority_keys]
            logger.debug(f"Found {len(sort_keys)} sortable columns from DataFrame")
        except Exception as e:
            logger.warning(f"Could not extract sort keys from DataFrame: {e}")

        # Fallback to defaults if no headers found
        if not sort_keys:
            sort_keys = ['offset', 'TraceNumber', 'FieldRecord', 'CDP', 'TRACE_SEQUENCE_LINE']
            logger.info("Using default sort keys")

        # Update control panel
        if hasattr(self, 'control_panel') and self.control_panel is not None:
            self.control_panel.set_available_sort_headers(sort_keys)

    def _clear_recent_files(self):
        """Clear the recent files list."""
        self.recent_files = []
        self._save_recent_files()
        self._update_recent_files_menu()

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
        # Store original SEG-Y file path for export (if available)
        if file_path is not None:
            self.original_segy_path = file_path
            # Add to recent files
            self._add_to_recent_files(file_path)

        # Register dataset with DatasetNavigator (for multi-dataset management)
        storage_path = None
        if isinstance(seismic_data, LazySeismicData):
            # Get storage path from lazy data
            storage_path = Path(seismic_data.zarr_path).parent
            dataset_name = Path(file_path).stem if file_path else "Imported Data"

            # Register with dataset navigator
            dataset_id = self.dataset_navigator.add_dataset(
                source_path=Path(file_path) if file_path else Path(),
                storage_path=storage_path,
                lazy_data=seismic_data,
                name=dataset_name,
                metadata={
                    'original_segy_path': str(file_path) if file_path else None,
                    'ensembles_count': len(ensembles_df) if ensembles_df is not None else 0
                }
            )

            # Persist to settings
            info = self.dataset_navigator.get_dataset_info(dataset_id)
            if info:
                self.app_settings.add_loaded_dataset(info.to_dict())

            logger.info(f"Dataset registered: {dataset_name} ({dataset_id[:8]}...)")

        # Load data into gather navigator - use lazy loading if available
        if isinstance(seismic_data, LazySeismicData):
            self.gather_navigator.load_lazy_data(seismic_data, ensembles_df)
        else:
            self.gather_navigator.load_data(seismic_data, headers_df, ensembles_df)

        # Store ensembles for reference
        self.ensembles_df = ensembles_df

        # Display first gather
        self._display_current_gather()

        # Update control panel with Nyquist frequency
        self.control_panel.update_nyquist(seismic_data.nyquist_freq)

        # Populate sort keys - try storage first, then headers_df, then gather_navigator
        if storage_path is not None:
            self._populate_sort_keys_from_storage(storage_path)
        elif headers_df is not None:
            self._populate_sort_keys_from_dataframe(headers_df)
        else:
            # Fallback to gather_navigator method
            available_headers = self.gather_navigator.get_available_sort_headers()
            if available_headers:
                self.control_panel.set_available_sort_headers(available_headers)
            else:
                # Ultimate fallback - use defaults
                self.control_panel.set_available_sort_headers(
                    ['offset', 'TraceNumber', 'FieldRecord', 'CDP', 'TRACE_SEQUENCE_LINE']
                )

        # Update datasets menu
        self._update_datasets_menu()

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

            # Set up progress callback for status bar updates
            def progress_callback(current: int, total: int, message: str):
                if total > 0:
                    pct = int(current / total * 100)
                    self.statusBar().showMessage(f"Processing: {pct}% - {message}")
                    # Allow UI to update during processing
                    QApplication.processEvents()

            # Attach progress callback to processor
            processor.set_progress_callback(progress_callback)

            # Apply processor to input data
            self.processed_data = processor.process(self.input_data)

            # Clear progress callback
            processor.set_progress_callback(None)

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

    def _set_theme(self, theme: str):
        """Set application theme."""
        from utils.theme_manager import get_theme_manager
        theme_manager = get_theme_manager()
        theme_manager.set_theme(theme)
        self._update_theme_menu_state()
        self._update_viewer_themes()
        self.statusBar().showMessage(f"Theme changed to {theme} mode", 3000)

    def _update_theme_menu_state(self):
        """Update theme menu checkmarks based on current theme."""
        from utils.theme_manager import get_theme_manager
        theme_manager = get_theme_manager()
        is_dark = theme_manager.is_dark
        self.light_theme_action.setChecked(not is_dark)
        self.dark_theme_action.setChecked(is_dark)

    def _update_viewer_themes(self):
        """Update viewer backgrounds based on current theme."""
        from utils.theme_manager import get_theme_manager
        theme_manager = get_theme_manager()
        plot_colors = theme_manager.get_plot_colors()

        # Update PyQtGraph viewers
        for viewer in [self.input_viewer, self.processed_viewer, self.difference_viewer]:
            if hasattr(viewer, 'graphics_widget'):
                viewer.graphics_widget.setBackground(plot_colors['background'])

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

    def _open_settings(self):
        """Open the settings dialog."""
        from views.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self)
        if dialog.exec():
            # Settings were saved, apply any immediate changes
            logger.info("Settings updated")

            # Update gather cache limit if changed
            new_limit = self.app_settings.get_gather_cache_limit()
            if hasattr(self.gather_navigator, 'set_max_cached_ensembles'):
                self.gather_navigator.set_max_cached_ensembles(new_limit)

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

    def _open_isa_window(self):
        """Open Interactive Spectral Analysis window."""
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load or generate seismic data first."
            )
            return

        # Import ISA window
        from views.isa_window import ISAWindow

        # Create and show ISA window with shared viewport state for synchronization
        # Pass processed data if available for comparison
        isa_window = ISAWindow(
            self.input_data,
            self.viewport_state,
            processed_data=self.processed_data,
            parent=self
        )
        isa_window.show()

        # Status message
        compare_hint = " Check 'Compare with processed' to overlay spectra." if self.processed_data else ""
        self.statusBar().showMessage(
            f"ISA window opened. Click on traces to view their spectrum.{compare_hint}",
            3000
        )

    def _batch_process_all_gathers(self):
        """
        Batch process all gathers in the dataset with current processor.

        DEPRECATED: This method loads all processed data into memory.
        For large datasets, use _batch_process_parallel() instead which:
        - Uses multiprocessing for 10-14x speedup
        - Writes directly to disk without memory accumulation
        - Supports datasets of any size
        """
        import warnings
        warnings.warn(
            "_batch_process_all_gathers is deprecated. "
            "Use Parallel Batch Process (Ctrl+Shift+B) for better performance.",
            DeprecationWarning,
            stacklevel=2
        )

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

        DEPRECATED: This method uses single-threaded processing.
        For better performance, use:
        - _batch_process_parallel() for processing (10-14x faster)
        - _export_parallel() for export (6-10x faster)
        """
        import warnings
        warnings.warn(
            "_batch_process_and_export_streaming is deprecated. "
            "Use Parallel Batch Process (Ctrl+Shift+B) + Parallel Export (Ctrl+Alt+E) "
            "for better performance.",
            DeprecationWarning,
            stacklevel=2
        )

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
            sorted_headers_list = []  # Collect sorted headers for export
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

                # Collect sorted headers (if sorting is enabled)
                sorted_headers_list.append(gather_headers)

                # Process this gather
                processed_gather = self.last_processor.process(gather_data)

                # Write to Zarr array
                start_trace = gather_info['start_trace']
                end_trace = gather_info['end_trace']
                processed_zarr[:, start_trace:end_trace+1] = processed_gather.traces

            # Step 3: Prepare sorted headers for export
            if sorted_headers_list:
                # Concatenate all sorted headers
                sorted_headers_df = pd.concat(sorted_headers_list, ignore_index=True)
            else:
                sorted_headers_df = None

            # Step 4: Export using chunked export
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
                progress_callback=export_progress_callback,
                headers_df=sorted_headers_df
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

    # =========================================================================
    # Parallel Batch Processing (Multiprocess - Bypasses GIL)
    # =========================================================================

    def _batch_process_parallel(self):
        """
        Parallel batch processing using all CPU cores.

        Uses multiprocessing to bypass Python GIL, achieving 10-14x speedup
        on multi-core systems. Processes gathers in parallel and writes
        directly to shared output Zarr array.
        """
        # Check if we have multi-gather data with lazy loading
        if not self.gather_navigator.has_gathers():
            QMessageBox.warning(
                self,
                "No Multi-Gather Data",
                "Parallel batch processing requires multi-gather SEG-Y data\n"
                "loaded in lazy mode.\n\n"
                "Please load SEG-Y data with ensemble information."
            )
            return

        if self.gather_navigator.lazy_data is None:
            QMessageBox.warning(
                self,
                "Lazy Loading Required",
                "Parallel batch processing requires data loaded in lazy mode.\n\n"
                "The current dataset was loaded in full-memory mode.\n"
                "Please re-import the SEG-Y file to use parallel processing."
            )
            return

        # Check if processor is configured
        if self.last_processor is None:
            QMessageBox.warning(
                self,
                "No Processor Configured",
                "Please apply processing to at least one gather first.\n\n"
                "Steps:\n"
                "1. Configure filter parameters\n"
                "2. Click 'Apply Filter'\n"
                "3. Then run parallel batch processing"
            )
            return

        # Get statistics
        stats = self.gather_navigator.get_statistics()
        n_gathers = stats['n_gathers']
        total_traces = stats['total_traces']
        n_samples = self.gather_navigator.lazy_data.n_samples

        # Get input storage directory
        input_storage_dir = self.gather_navigator.lazy_data.zarr_path.parent

        # Check for ensemble index - auto-create if missing
        ensemble_path = input_storage_dir / 'ensemble_index.parquet'
        if not ensemble_path.exists():
            from utils.segy_import.data_storage import DataStorage
            storage = DataStorage(str(input_storage_dir))
            if not storage.ensure_ensemble_index(n_traces=total_traces):
                QMessageBox.warning(
                    self,
                    "Ensemble Index Error",
                    "Could not create ensemble index file.\n\n"
                    f"Expected: {ensemble_path}\n\n"
                    "Please re-import the SEG-Y data."
                )
                return
            self.statusBar().showMessage("Created missing ensemble index")

        # Initialize storage manager and validate disk space
        storage_manager = ProcessingStorageManager()
        is_sufficient, space_message = storage_manager.validate_disk_space(
            total_traces, n_samples
        )

        if not is_sufficient:
            QMessageBox.critical(
                self,
                "Insufficient Disk Space",
                space_message
            )
            return

        # Get optimal worker count
        from utils.parallel_processing import get_optimal_workers, SortOptions
        n_workers = get_optimal_workers()

        # Estimate time (rough: ~200k traces/sec with parallel)
        estimated_seconds = total_traces / (200000 * n_workers / 14)
        estimated_minutes = max(1, estimated_seconds / 60)

        # Show processing options dialog with sorting configuration
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QComboBox, QLabel, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Parallel Batch Processing Options")
        dialog.setMinimumWidth(450)

        layout = QVBoxLayout(dialog)

        # Info section
        info_label = QLabel(
            f"<b>Dataset:</b> {n_gathers:,} gathers, {total_traces:,} traces<br>"
            f"<b>Processor:</b> {self.last_processor.get_description()}<br>"
            f"<b>Workers:</b> {n_workers} CPU cores<br>"
            f"<b>Estimated time:</b> ~{estimated_minutes:.0f} minutes"
        )
        layout.addWidget(info_label)

        # Sorting options group
        sort_group = QGroupBox("In-Gather Sorting")
        sort_layout = QVBoxLayout(sort_group)

        sort_checkbox = QCheckBox("Sort traces within each gather")
        sort_layout.addWidget(sort_checkbox)

        # Sort key selection
        sort_key_layout = QHBoxLayout()
        sort_key_label = QLabel("Sort by:")
        sort_key_combo = QComboBox()

        # Get available header columns for sorting
        headers_path = input_storage_dir / 'headers.parquet'
        sorted_cols = []
        if headers_path.exists():
            import pandas as pd
            try:
                sample_headers = pd.read_parquet(headers_path)
                # Filter to numeric columns suitable for sorting (use 'number' to catch all numeric types)
                numeric_cols = sample_headers.select_dtypes(include=['number']).columns.tolist()
                # Prioritize common sort keys
                priority_keys = ['offset', 'TRACE_SEQUENCE_LINE', 'TraceNumber', 'FieldRecord', 'CDP', 'SOURCE_X', 'SOURCE_Y', 'GROUP_X', 'GROUP_Y']
                sorted_cols = [k for k in priority_keys if k in numeric_cols] + [k for k in numeric_cols if k not in priority_keys]
                logger.debug(f"Found {len(sorted_cols)} sortable columns in headers.parquet")
            except Exception as e:
                logger.warning(f"Could not read headers.parquet: {e}")
        else:
            logger.warning(f"Headers file not found: {headers_path}")

        # Fallback to default columns if none found
        if not sorted_cols:
            sorted_cols = ['offset', 'TraceNumber', 'FieldRecord', 'CDP', 'TRACE_SEQUENCE_LINE']
            logger.info("Using default sort columns (no headers found or all non-numeric)")

        sort_key_combo.addItems(sorted_cols)
        # Default to offset if available
        if 'offset' in sorted_cols:
            sort_key_combo.setCurrentText('offset')

        sort_key_layout.addWidget(sort_key_label)
        sort_key_layout.addWidget(sort_key_combo)
        sort_layout.addLayout(sort_key_layout)

        # Sort direction
        sort_dir_layout = QHBoxLayout()
        sort_dir_label = QLabel("Direction:")
        sort_dir_combo = QComboBox()
        sort_dir_combo.addItems(["Ascending", "Descending"])
        sort_dir_layout.addWidget(sort_dir_label)
        sort_dir_layout.addWidget(sort_dir_combo)
        sort_layout.addLayout(sort_dir_layout)

        # Enable/disable sort options based on checkbox
        def toggle_sort_options(checked):
            sort_key_combo.setEnabled(checked)
            sort_dir_combo.setEnabled(checked)

        sort_checkbox.toggled.connect(toggle_sort_options)
        sort_key_combo.setEnabled(False)
        sort_dir_combo.setEnabled(False)

        layout.addWidget(sort_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Build sort options from dialog
        sort_options = None
        if sort_checkbox.isChecked():
            sort_options = SortOptions(
                enabled=True,
                sort_key=sort_key_combo.currentText(),
                ascending=(sort_dir_combo.currentText() == "Ascending")
            )

        # Create processing session
        dataset_name = input_storage_dir.name
        session = storage_manager.create_session(f"processed_{dataset_name}")

        # Create progress dialog
        progress = QProgressDialog(
            "Initializing parallel processing...",
            "Cancel",
            0,
            total_traces,
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Parallel Batch Processing")
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)

        try:
            from utils.parallel_processing import (
                ParallelProcessingCoordinator,
                ProcessingConfig,
                ProcessingProgress
            )

            # Serialize processor configuration
            processor_config = self.last_processor.to_dict()

            # Create processing config
            config = ProcessingConfig(
                input_storage_dir=str(input_storage_dir),
                output_storage_dir=str(session.output_dir),
                processor_config=processor_config,
                n_workers=n_workers,
                sort_options=sort_options
            )

            # Progress callback
            import time
            start_time = time.time()

            def on_progress(prog: ProcessingProgress):
                if progress.wasCanceled():
                    return

                progress.setValue(prog.current_traces)

                # Format ETA
                if prog.eta_seconds > 0 and prog.eta_seconds < float('inf'):
                    if prog.eta_seconds > 60:
                        eta_str = f"{prog.eta_seconds / 60:.1f} min"
                    else:
                        eta_str = f"{prog.eta_seconds:.0f} sec"
                else:
                    eta_str = "calculating..."

                # Calculate rate
                elapsed = time.time() - start_time
                rate = prog.current_traces / elapsed if elapsed > 0 else 0

                progress.setLabelText(
                    f"Phase: {prog.phase}\n"
                    f"Gathers: {prog.current_gathers:,} / {prog.total_gathers:,}\n"
                    f"Traces: {prog.current_traces:,} / {prog.total_traces:,}\n"
                    f"Workers: {prog.active_workers} active\n"
                    f"Rate: {rate:,.0f} traces/sec\n"
                    f"ETA: {eta_str}"
                )
                QApplication.processEvents()

            # Run parallel processing
            coordinator = ParallelProcessingCoordinator(config)
            result = coordinator.run(progress_callback=on_progress)

            if progress.wasCanceled():
                # User cancelled - cleanup
                session.cleanup_all()
                QMessageBox.information(
                    self,
                    "Cancelled",
                    "Parallel processing was cancelled.\n"
                    "Temporary files have been cleaned up."
                )
                return

            if not result.success:
                # Processing failed
                session.cleanup_all()
                QMessageBox.critical(
                    self,
                    "Processing Failed",
                    f"Parallel processing failed:\n\n{result.error}"
                )
                return

            # Success - mark session complete
            session.mark_complete()

            # Show success message
            QMessageBox.information(
                self,
                "Parallel Processing Complete",
                f"Successfully processed all {n_gathers:,} gathers!\n\n"
                f"Statistics:\n"
                f"  - Traces processed: {result.n_traces:,}\n"
                f"  - Time: {result.elapsed_time:.1f} seconds\n"
                f"  - Throughput: {result.throughput_traces_per_sec:,.0f} traces/sec\n"
                f"  - Workers used: {result.n_workers_used}\n\n"
                f"Output saved to:\n{result.output_dir}\n\n"
                f"Would you like to load the processed data?"
            )

            # Offer to load the processed data
            reply = QMessageBox.question(
                self,
                "Load Processed Data?",
                f"Load the processed dataset?\n\n{result.output_dir}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._load_from_zarr_path(result.output_dir)

            self.statusBar().showMessage(
                f"Parallel processing complete: {result.n_traces:,} traces "
                f"in {result.elapsed_time:.1f}s "
                f"({result.throughput_traces_per_sec:,.0f} traces/sec)",
                10000
            )

        except Exception as e:
            progress.close()
            session.cleanup_all()
            QMessageBox.critical(
                self,
                "Processing Error",
                f"Failed to run parallel processing:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

        finally:
            progress.close()

    # =========================================================================
    # Parallel SEG-Y Export (Multiprocess - Bypasses GIL)
    # =========================================================================

    def _export_parallel(self):
        """
        Parallel SEG-Y export using all CPU cores.

        Uses multiprocessing to bypass Python GIL, achieving 6-10x speedup
        on multi-core systems. Exports processed Zarr data to SEG-Y with
        vectorized header access and segment merging.
        """
        import os
        from pathlib import Path
        from utils.parallel_export import (
            ParallelExportCoordinator,
            ExportConfig,
            ExportProgress
        )

        # Check if we have processed data in lazy storage
        if self.gather_navigator.lazy_data is None:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load data first using 'Import SEG-Y (Optimized)'."
            )
            return

        # Check if we have the original SEG-Y path
        if self.original_segy_path is None:
            QMessageBox.warning(
                self,
                "No Original SEG-Y Path",
                "Export requires the original SEG-Y file for headers.\n\n"
                "Please ensure data was imported with 'Import SEG-Y (Optimized)'."
            )
            return

        # Check if we have headers
        if self.headers_df is None:
            QMessageBox.warning(
                self,
                "No Headers",
                "No headers available for export.\n\n"
                "Headers are required to write correct trace headers."
            )
            return

        # Get the processed Zarr path
        # First check if there's a parallel-processed output available
        storage_manager = ProcessingStorageManager()
        sessions = storage_manager.list_sessions()

        # Look for the most recent completed session with matching data
        processed_zarr_path = None
        processed_output_dir = None
        is_processed_data = False  # Track if we're exporting processed or input data

        for session in sessions:
            info_path = session.session_dir / 'session_info.json'
            if info_path.exists():
                import json
                with open(info_path, 'r') as f:
                    info = json.load(f)
                if info.get('status') == 'complete':
                    zarr_path = session.session_dir / 'output' / 'traces.zarr'
                    if zarr_path.exists():
                        processed_zarr_path = str(zarr_path)
                        processed_output_dir = str(session.session_dir / 'output')
                        is_processed_data = True
                        print(f"  Found processed data: {processed_output_dir}")
                        break

        if processed_zarr_path is None:
            # No processed session found - ask user if they want to export input data
            lazy_data = self.gather_navigator.lazy_data
            if lazy_data is not None and hasattr(lazy_data, 'zarr_path'):
                storage_dir = lazy_data.zarr_path.parent
                zarr_path = storage_dir / 'traces.zarr'
                if zarr_path.exists():
                    # Ask user if they want to export unprocessed data
                    reply = QMessageBox.question(
                        self,
                        "No Processed Data Found",
                        "No parallel-processed data was found.\n\n"
                        "Would you like to export the ORIGINAL (unprocessed) data instead?\n\n"
                        f"Source: {storage_dir}\n\n"
                        "Note: To export processed data, first run 'Parallel Batch Process...'",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    processed_zarr_path = str(zarr_path)
                    processed_output_dir = str(storage_dir)
                    is_processed_data = False
                    print(f"  Using input data (unprocessed): {processed_output_dir}")
                else:
                    QMessageBox.warning(
                        self,
                        "No Data Found",
                        "No Zarr data found to export.\n\n"
                        "Please load SEG-Y data first using 'Import SEG-Y (Optimized)'."
                    )
                    return
            else:
                QMessageBox.warning(
                    self,
                    "No Data Loaded",
                    "No data available for export.\n\n"
                    "Please load SEG-Y data first using 'Import SEG-Y (Optimized)'."
                )
                return

        # Get headers path - prefer from processed output, fall back to input
        headers_path = Path(processed_output_dir) / 'headers.parquet'
        if not headers_path.exists():
            # Fall back to lazy data headers
            lazy_data = self.gather_navigator.lazy_data
            if lazy_data is not None and hasattr(lazy_data, 'zarr_path'):
                headers_path = lazy_data.zarr_path.parent / 'headers.parquet'

        if not headers_path.exists():
            QMessageBox.warning(
                self,
                "No Headers File",
                "Headers parquet file not found.\n\n"
                "Cannot export without trace headers."
            )
            return

        # Get number of workers
        n_workers = os.cpu_count() or 4
        n_workers = max(2, n_workers - 2)  # Leave 2 cores for system

        # Get trace count from Zarr
        import zarr
        z = zarr.open(processed_zarr_path, mode='r')
        n_samples, n_traces = z.shape

        # Estimate file size
        estimated_size_gb = (n_traces * (240 + n_samples * 4) + 3600) / (1024 ** 3)

        # Show confirmation dialog with data type indicator
        data_type_str = "PROCESSED" if is_processed_data else "ORIGINAL (unprocessed)"
        reply = QMessageBox.question(
            self,
            "Parallel SEG-Y Export",
            f"Export {data_type_str} data to SEG-Y using {n_workers} CPU cores?\n\n"
            f"Source: {processed_output_dir}\n"
            f"Traces: {n_traces:,}\n"
            f"Samples: {n_samples}\n"
            f"Estimated file size: {estimated_size_gb:.2f} GB\n\n"
            f"This will be approximately 6-10x faster than standard export.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Get output file path
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed SEG-Y (Parallel)",
            "",
            "SEG-Y Files (*.sgy *.segy);;All Files (*)"
        )

        if not output_file:
            return

        # Create temp directory for segment files
        temp_dir = Path(processed_output_dir) / 'export_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create export config
        config = ExportConfig(
            original_segy_path=self.original_segy_path,
            processed_zarr_path=processed_zarr_path,
            headers_parquet_path=str(headers_path),
            output_path=output_file,
            temp_dir=str(temp_dir),
            n_workers=n_workers
        )

        # Create progress dialog
        progress = QProgressDialog(
            "Initializing parallel export...",
            "Cancel",
            0, n_traces,
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Parallel SEG-Y Export")
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)

        def update_progress(prog: ExportProgress):
            """Update progress dialog."""
            if progress.wasCanceled():
                coordinator.cancel()
                return

            progress.setValue(prog.current_traces)

            if prog.phase == 'vectorizing':
                progress.setLabelText("Vectorizing headers for fast access...")
            elif prog.phase == 'exporting':
                rate = prog.current_traces / prog.elapsed_time if prog.elapsed_time > 0 else 0
                eta_min = prog.eta_seconds / 60 if prog.eta_seconds > 0 else 0
                progress.setLabelText(
                    f"Exporting traces ({prog.active_workers} workers)...\n"
                    f"{prog.current_traces:,} / {prog.total_traces:,} traces\n"
                    f"Rate: {rate:,.0f} traces/sec | ETA: {eta_min:.1f} min"
                )
            elif prog.phase == 'merging':
                progress.setLabelText("Merging segment files...")
            elif prog.phase == 'finalizing':
                progress.setLabelText("Finalizing export...")

            QApplication.processEvents()

        # Run export
        try:
            self.statusBar().showMessage(f"Exporting to SEG-Y with {n_workers} workers...")

            coordinator = ParallelExportCoordinator(config)

            # Track cancellation state before running (wasCanceled can be unreliable after close)
            was_canceled = False

            result = coordinator.run(progress_callback=update_progress)

            # Check cancellation BEFORE closing dialog
            was_canceled = progress.wasCanceled()
            progress.close()

            if was_canceled and not result.success:
                # Only treat as canceled if export actually failed
                # (successful export should not be deleted even if dialog shows canceled)
                self.statusBar().showMessage("Export canceled.")
                return

            if not result.success:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Parallel export failed:\n\n{result.error}"
                )
                return

            # Success
            file_size_mb = result.file_size_bytes / (1024 ** 2)
            data_status = "processed" if is_processed_data else "unprocessed (original)"
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {result.n_traces:,} traces!\n\n"
                f"Data type: {data_status}\n\n"
                f"Statistics:\n"
                f"  - File size: {file_size_mb:.1f} MB\n"
                f"  - Time: {result.elapsed_time:.1f} seconds\n"
                f"  - Throughput: {result.throughput_traces_per_sec:,.0f} traces/sec\n"
                f"  - Workers used: {result.n_workers_used}\n\n"
                f"Output: {output_file}"
            )

            self.statusBar().showMessage(
                f"Export complete: {result.n_traces:,} traces "
                f"in {result.elapsed_time:.1f}s "
                f"({result.throughput_traces_per_sec:,.0f} traces/sec)",
                10000
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to run parallel export:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

        finally:
            progress.close()
            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

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

    # =========================================================================
    # 3D FKK Filter Handlers
    # =========================================================================

    def _on_fkk_design_requested(self):
        """Handle FKK design request - build volume from current gather and open designer."""
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load seismic data first before using 3D FKK filter."
            )
            return

        # Use current active gather (same as other processing algorithms)
        try:
            # Get traces from current gather
            traces_data = self.input_data.traces
            n_samples, n_traces = traces_data.shape

            # Use headers_df (set when gather is loaded)
            if self.headers_df is None or self.headers_df.empty:
                # No headers available - cannot build 3D volume
                QMessageBox.warning(
                    self,
                    "No Headers",
                    "Current gather has no trace headers.\n"
                    "3D FKK filter requires headers to define inline/crossline axes."
                )
                return

            headers_df = self.headers_df.copy()

            if len(headers_df) != n_traces:
                # Headers don't match traces
                logger.warning(f"Header count ({len(headers_df)}) != trace count ({n_traces})")
                headers_df = headers_df.iloc[:n_traces] if len(headers_df) > n_traces else headers_df

            self.statusBar().showMessage(
                f"Building 3D volume from current gather ({n_traces} traces)..."
            )
            QApplication.processEvents()

            # Show header selection dialog and build volume
            result = build_volume_with_dialog(
                traces_data=traces_data,
                headers_df=headers_df,
                sample_rate_ms=self.input_data.sample_rate,
                coordinate_units=self.input_data.metadata.get('coordinate_units', 'meters'),
                parent=self
            )

            if result is None:
                self.statusBar().showMessage("Volume building cancelled")
                return

            volume, geometry = result

            # Store for later use
            self._current_3d_volume = volume
            self._fkk_volume_geometry = geometry

            self.statusBar().showMessage(
                f"Volume built: {volume.shape}, {volume.memory_mb():.1f} MB, "
                f"{geometry.coverage_percent:.1f}% coverage"
            )

            # Open FKK designer
            dialog = FKKDesignerDialog(volume, parent=self)
            dialog.filter_applied.connect(self._on_fkk_filter_applied)
            dialog.exec()

        except Exception as e:
            logger.error(f"FKK volume building failed: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to build 3D volume:\n\n{str(e)}"
            )
            self.statusBar().showMessage("Volume building failed")

    def _on_fkk_apply_requested(self, config: FKKConfig):
        """Handle FKK apply request - apply filter directly without designer."""
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load seismic data first."
            )
            return

        # Check if we have a built volume
        if not hasattr(self, '_current_3d_volume') or self._current_3d_volume is None:
            QMessageBox.warning(
                self,
                "No Volume",
                "Please use Design mode first to build a 3D volume\n"
                "from your data before applying a filter."
            )
            return

        try:
            self.statusBar().showMessage("Applying FKK filter...")
            QApplication.processEvents()

            # Get filter processor
            fkk_filter = get_fkk_filter(prefer_gpu=True)

            # Apply filter
            filtered_volume = fkk_filter.apply_filter(self._current_3d_volume, config)

            # Store result
            self._current_3d_volume_filtered = filtered_volume
            self._current_fkk_config = config

            # Extract filtered traces back to 2D gather format
            if hasattr(self, '_fkk_volume_geometry') and self.headers_df is not None:
                geom = self._fkk_volume_geometry

                filtered_traces = extract_traces_from_volume(
                    filtered_volume,
                    self.headers_df,
                    geom.inline_key,
                    geom.xline_key
                )

                # Update processed data with filtered result
                self.processed_data = SeismicData(
                    traces=filtered_traces,
                    sample_rate=self.input_data.sample_rate,
                    metadata={**self.input_data.metadata, 'fkk_filtered': True}
                )

                # Calculate difference
                difference_traces = self.input_data.traces - self.processed_data.traces
                self.difference_data = SeismicData(
                    traces=difference_traces,
                    sample_rate=self.input_data.sample_rate,
                    metadata={'description': 'Difference (Input - FKK Filtered)'}
                )

                # Update viewers
                self.processed_viewer.set_data(self.processed_data)
                self.difference_viewer.set_data(self.difference_data)

                self.statusBar().showMessage(
                    f"FKK filter applied: {config.get_summary()}"
                )
                return

            self.statusBar().showMessage(f"FKK filter applied: {config.get_summary()}")

        except Exception as e:
            logger.error(f"FKK filter application failed: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to apply FKK filter:\n\n{str(e)}"
            )
            self.statusBar().showMessage("FKK filter failed")

    def _on_fkk_filter_applied(self, filtered_volume, config):
        """Handle FKK filter result from designer dialog."""
        self._current_3d_volume_filtered = filtered_volume
        self._current_fkk_config = config

        # Extract filtered traces back to current gather format
        if hasattr(self, '_fkk_volume_geometry') and self.input_data is not None:
            try:
                geom = self._fkk_volume_geometry

                # Use headers_df (set when gather is loaded)
                if self.headers_df is not None:
                    filtered_traces = extract_traces_from_volume(
                        filtered_volume,
                        self.headers_df,
                        geom.inline_key,
                        geom.xline_key
                    )

                    # Update processed data
                    self.processed_data = SeismicData(
                        traces=filtered_traces,
                        sample_rate=self.input_data.sample_rate,
                        metadata={'description': f'FKK filtered: {config.get_summary()}'}
                    )

                    # Calculate difference
                    difference_traces = self.input_data.traces - self.processed_data.traces
                    self.difference_data = SeismicData(
                        traces=difference_traces,
                        sample_rate=self.input_data.sample_rate,
                        metadata={'description': 'Difference (Input - FKK Filtered)'}
                    )

                    # Update viewers
                    self.processed_viewer.set_data(self.processed_data)
                    self.difference_viewer.set_data(self.difference_data)

                    self.statusBar().showMessage(
                        f"FKK filter applied and displayed: {config.get_summary()}"
                    )
                    return

            except Exception as e:
                logger.warning(f"Could not update gather view: {e}")

        self.statusBar().showMessage(
            f"FKK filter applied: {config.get_summary()}"
        )

    # =========================================================================
    # PSTM (Pre-Stack Time Migration) Handlers
    # =========================================================================

    def _on_pstm_apply_requested(self, velocity: float, aperture: float, max_angle: float):
        """
        Handle PSTM apply request from control panel.

        Args:
            velocity: Constant velocity in m/s
            aperture: Migration aperture in meters
            max_angle: Maximum migration angle in degrees
        """
        if self.input_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load seismic data first."
            )
            return

        # Check if we have geometry headers
        if self.headers_df is None or self.headers_df.empty:
            QMessageBox.warning(
                self,
                "No Geometry",
                "No trace headers available.\n\n"
                "PSTM requires source/receiver coordinates from trace headers.\n"
                "Please ensure your SEG-Y file has geometry information."
            )
            return

        try:
            self.statusBar().showMessage("Preparing PSTM migration...")
            QApplication.processEvents()

            # Import PSTM components
            from processors.migration import KirchhoffMigrator
            from models.velocity_model import create_constant_velocity
            from models.migration_geometry import MigrationGeometry

            # Create constant velocity model
            velocity_model = create_constant_velocity(velocity, is_time=True)

            # Create geometry from headers
            geometry = self._create_geometry_from_headers()
            if geometry is None:
                return  # Error already shown

            # Create migrator
            migrator = KirchhoffMigrator(
                velocity=velocity_model,
                aperture=aperture,
                max_angle=max_angle,
                dt=self.input_data.sample_rate,
                prefer_gpu=self.control_panel.use_gpu() if hasattr(self.control_panel, 'use_gpu') else True,
            )

            self.statusBar().showMessage(f"Running PSTM (v={velocity:.0f} m/s, aperture={aperture:.0f}m)...")
            QApplication.processEvents()

            # Migrate the current gather
            output_image, output_fold = migrator.migrate_gather(
                gather=self.input_data,
                geometry=geometry,
            )

            # Convert output to SeismicData for display
            # The migrated image is the zero-offset section
            import torch
            if isinstance(output_image, torch.Tensor):
                migrated_traces = output_image.cpu().numpy()
            else:
                migrated_traces = output_image

            # Ensure 2D array (time x traces)
            if migrated_traces.ndim == 1:
                migrated_traces = migrated_traces.reshape(-1, 1)

            self.processed_data = SeismicData(
                traces=migrated_traces,
                sample_rate=self.input_data.sample_rate,
                metadata={
                    **self.input_data.metadata,
                    'pstm_migrated': True,
                    'pstm_velocity': velocity,
                    'pstm_aperture': aperture,
                    'pstm_max_angle': max_angle,
                }
            )

            # Calculate difference (input - migrated)
            # Note: Shapes may differ, so handle carefully
            if self.processed_data.traces.shape == self.input_data.traces.shape:
                difference_traces = self.input_data.traces - self.processed_data.traces
                self.difference_data = SeismicData(
                    traces=difference_traces,
                    sample_rate=self.input_data.sample_rate,
                    metadata={'description': 'Difference (Input - PSTM)'}
                )
                self.difference_viewer.set_data(self.difference_data)
            else:
                # Clear difference view if shapes don't match
                self.difference_viewer.clear()
                self.difference_data = None

            # Update processed viewer
            self.processed_viewer.set_data(self.processed_data)

            self.statusBar().showMessage(
                f"PSTM completed: velocity={velocity:.0f} m/s, "
                f"aperture={aperture:.0f}m, max_angle={max_angle:.0f}°"
            )

        except Exception as e:
            logger.error(f"PSTM migration failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "PSTM Error",
                f"Migration failed:\n\n{str(e)}"
            )
            self.statusBar().showMessage("PSTM migration failed")

    def _on_pstm_wizard_requested(self):
        """Handle request to open PSTM wizard dialog."""
        from views.pstm_wizard_dialog import PSTMWizard

        # Get the active dataset's storage path (Zarr directory)
        initial_file = None
        active_dataset_id = self.dataset_navigator.get_active_dataset_id()

        if active_dataset_id:
            info = self.dataset_navigator.get_dataset_info(active_dataset_id)
            if info and info.storage_path and info.storage_path.exists():
                initial_file = str(info.storage_path)
                logger.info(f"PSTM Wizard: Using active dataset: {info.name} at {initial_file}")

        # Fallback to original SEG-Y path if no active dataset
        if not initial_file and hasattr(self, 'original_segy_path') and self.original_segy_path:
            initial_file = self.original_segy_path

        wizard = PSTMWizard(parent=self, initial_file=initial_file)
        wizard.job_configured.connect(self._on_pstm_job_configured)
        wizard.exec()

    def _on_pstm_job_configured(self, config: dict):
        """Handle completed PSTM wizard configuration."""
        from views.migration_monitor_dialog import MigrationMonitorDialog

        # Ask to start now
        reply = QMessageBox.question(
            self,
            "Start Migration",
            f"Start migration job '{config.get('name', 'Migration')}' now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Create and show monitor dialog
            monitor = MigrationMonitorDialog(config, parent=self)
            monitor.job_completed.connect(self._on_migration_job_completed)
            monitor.show()
            monitor.start_job()

            # Track running jobs
            if not hasattr(self, '_migration_monitors'):
                self._migration_monitors = []
            self._migration_monitors.append(monitor)

            self.statusBar().showMessage(f"Migration job '{config.get('name')}' started")
        else:
            # Offer to save configuration
            save_reply = QMessageBox.question(
                self,
                "Save Configuration",
                "Would you like to save the job configuration for later?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if save_reply == QMessageBox.StandardButton.Yes:
                self._save_migration_config(config)

    def _on_migration_job_completed(self, success: bool, output_path: str):
        """Handle migration job completion."""
        if success:
            self.statusBar().showMessage(f"Migration completed: {output_path}")
            logger.info(f"Migration job completed successfully: {output_path}")
        else:
            self.statusBar().showMessage("Migration job failed or was cancelled")

    def _save_migration_config(self, config: dict):
        """Save migration configuration to file."""
        import json

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Migration Configuration",
            f"{config.get('name', 'migration_job')}.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.statusBar().showMessage(f"Configuration saved: {file_path}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save configuration:\n{e}"
                )

    def _resume_migration(self):
        """Resume an interrupted migration job from checkpoint."""
        from pathlib import Path
        import json

        # Let user select output directory or config file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Migration Configuration or Checkpoint",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Load configuration
            with open(file_path, 'r') as f:
                config = json.load(f)

            # Check if this is a valid migration config
            if 'name' not in config:
                QMessageBox.warning(
                    self,
                    "Invalid File",
                    "This doesn't appear to be a valid migration configuration file."
                )
                return

            # Check for checkpoint in output directory
            output_dir = config.get('output_directory', '')
            checkpoint_file = Path(output_dir) / 'checkpoint.json' if output_dir else None

            if checkpoint_file and checkpoint_file.exists():
                # Found checkpoint - offer to resume
                reply = QMessageBox.question(
                    self,
                    "Checkpoint Found",
                    f"Found checkpoint for job '{config.get('name')}'.\n\n"
                    "Resume from checkpoint?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Load checkpoint info
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)

                    completed_bins = checkpoint.get('completed_bins', [])
                    total_bins = checkpoint.get('total_bins', 0)

                    self.statusBar().showMessage(
                        f"Resuming: {len(completed_bins)}/{total_bins} bins completed"
                    )

                    # Mark config as resume
                    config['_resume_from_checkpoint'] = True
                    config['_checkpoint_file'] = str(checkpoint_file)
            else:
                # No checkpoint - start fresh
                reply = QMessageBox.question(
                    self,
                    "Start Job",
                    f"No checkpoint found.\n\n"
                    f"Start job '{config.get('name')}' from beginning?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply != QMessageBox.StandardButton.Yes:
                    return

            # Launch the job
            from views.migration_monitor_dialog import MigrationMonitorDialog

            monitor = MigrationMonitorDialog(config, parent=self)
            monitor.job_completed.connect(self._on_migration_job_completed)
            monitor.show()
            monitor.start_job()

            if not hasattr(self, '_migration_monitors'):
                self._migration_monitors = []
            self._migration_monitors.append(monitor)

            self.statusBar().showMessage(f"Migration job '{config.get('name')}' started")

        except json.JSONDecodeError:
            QMessageBox.critical(
                self,
                "Invalid File",
                "The selected file is not a valid JSON configuration."
            )
        except Exception as e:
            logger.error(f"Failed to resume migration: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Resume Error",
                f"Failed to resume migration:\n{e}"
            )

    def _open_pstm_from_menu(self):
        """Open PSTM panel from Processing menu."""
        # Switch to PSTM algorithm in control panel
        if hasattr(self.control_panel, 'algorithm_combo'):
            # Find Kirchhoff PSTM index (should be 4)
            combo = self.control_panel.algorithm_combo
            for i in range(combo.count()):
                if 'PSTM' in combo.itemText(i) or 'Kirchhoff' in combo.itemText(i):
                    combo.setCurrentIndex(i)
                    self.statusBar().showMessage(
                        "PSTM selected - configure parameters and click 'Apply PSTM'"
                    )
                    return

            QMessageBox.warning(
                self,
                "PSTM Not Available",
                "Kirchhoff PSTM algorithm not found in algorithm list."
            )

    def _create_geometry_from_headers(self):
        """
        Create MigrationGeometry from current gather's trace headers.

        Returns:
            MigrationGeometry or None if geometry cannot be created
        """
        from models.migration_geometry import MigrationGeometry

        if self.headers_df is None or self.headers_df.empty:
            return None

        # Try to find coordinate columns
        # Common SEG-Y header names for coordinates
        sx_keys = ['SourceX', 'SX', 'sx', 'source_x', 'SourceXCoordinate']
        sy_keys = ['SourceY', 'SY', 'sy', 'source_y', 'SourceYCoordinate']
        gx_keys = ['GroupX', 'GX', 'gx', 'ReceiverX', 'RX', 'rx', 'receiver_x', 'GroupXCoordinate']
        gy_keys = ['GroupY', 'GY', 'gy', 'ReceiverY', 'RY', 'ry', 'receiver_y', 'GroupYCoordinate']

        available_cols = set(self.headers_df.columns)

        def find_key(key_options):
            for k in key_options:
                if k in available_cols:
                    return k
            return None

        sx_key = find_key(sx_keys)
        sy_key = find_key(sy_keys)
        gx_key = find_key(gx_keys)
        gy_key = find_key(gy_keys)

        missing = []
        if sx_key is None:
            missing.append("Source X")
        if sy_key is None:
            missing.append("Source Y")
        if gx_key is None:
            missing.append("Receiver X")
        if gy_key is None:
            missing.append("Receiver Y")

        if missing:
            QMessageBox.warning(
                self,
                "Missing Geometry",
                f"Cannot find coordinate headers for:\n  {', '.join(missing)}\n\n"
                f"Available headers:\n  {', '.join(sorted(available_cols)[:20])}\n\n"
                "PSTM requires source and receiver coordinates."
            )
            return None

        try:
            # Get coordinate arrays
            source_x = self.headers_df[sx_key].values.astype(np.float32)
            source_y = self.headers_df[sy_key].values.astype(np.float32)
            receiver_x = self.headers_df[gx_key].values.astype(np.float32)
            receiver_y = self.headers_df[gy_key].values.astype(np.float32)

            # Check for coordinate scalar (common in SEG-Y)
            # Coordinates might be in tenths of meters or other units
            coord_scalar = 1.0
            if 'SourceGroupScalar' in available_cols:
                scalar_val = self.headers_df['SourceGroupScalar'].iloc[0]
                if scalar_val < 0:
                    coord_scalar = 1.0 / abs(scalar_val)
                elif scalar_val > 0:
                    coord_scalar = float(scalar_val)

            # Apply scalar if needed
            if coord_scalar != 1.0:
                source_x *= coord_scalar
                source_y *= coord_scalar
                receiver_x *= coord_scalar
                receiver_y *= coord_scalar
                logger.info(f"Applied coordinate scalar: {coord_scalar}")

            geometry = MigrationGeometry(
                source_x=source_x,
                source_y=source_y,
                receiver_x=receiver_x,
                receiver_y=receiver_y,
                metadata={
                    'sx_key': sx_key,
                    'sy_key': sy_key,
                    'gx_key': gx_key,
                    'gy_key': gy_key,
                    'coord_scalar': coord_scalar,
                }
            )

            stats = geometry.get_statistics()
            logger.info(
                f"Created geometry: {stats['n_traces']} traces, "
                f"offset range: {stats['offset_range'][0]:.0f}-{stats['offset_range'][1]:.0f}m"
            )

            return geometry

        except Exception as e:
            logger.error(f"Failed to create geometry: {e}")
            QMessageBox.warning(
                self,
                "Geometry Error",
                f"Failed to create geometry from headers:\n\n{str(e)}"
            )
            return None

    def _open_fkk_designer(self):
        """Open 3D FKK filter designer dialog (menu shortcut)."""
        self._on_fkk_design_requested()

    def _generate_test_3d_volume(self):
        """Generate a synthetic 3D volume for testing."""
        from PyQt6.QtWidgets import QInputDialog

        # Ask for volume size
        size, ok = QInputDialog.getInt(
            self,
            "Volume Size",
            "Enter cube size (e.g., 128 for 128x128x128):",
            128, 32, 512, 32
        )
        if not ok:
            return

        self.statusBar().showMessage(f"Generating {size}x{size}x{size} synthetic volume...")
        QApplication.processEvents()

        try:
            # Create synthetic volume
            volume = create_synthetic_volume(
                nt=size,
                nx=size,
                ny=size,
                dt=0.004,  # 4ms
                dx=25.0,   # 25m
                dy=25.0,   # 25m
                add_events=True
            )

            # Store and open designer
            self._current_3d_volume = volume

            self.statusBar().showMessage(
                f"Created synthetic volume: {volume.shape}, {volume.memory_mb():.1f} MB"
            )

            # Open the designer
            dialog = FKKDesignerDialog(volume, parent=self)
            dialog.filter_applied.connect(self._on_fkk_filter_applied)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create synthetic volume:\n\n{str(e)}"
            )
            self.statusBar().showMessage("Volume creation failed")

    # =========================================================================
    # Dataset Navigator Handlers
    # =========================================================================

    def _on_active_dataset_changed(self, dataset_id: str):
        """Handle active dataset change from DatasetNavigator."""
        if not dataset_id:
            # No active dataset - clear viewers
            self.input_viewer.clear()
            self.processed_viewer.clear()
            self.difference_viewer.clear()
            self.input_data = None
            self.processed_data = None
            self.difference_data = None
            self.statusBar().showMessage("No dataset loaded")
            return

        # Get the dataset
        lazy_data = self.dataset_navigator.get_dataset(dataset_id)
        if lazy_data is None:
            logger.warning(f"Could not load dataset: {dataset_id[:8]}...")
            return

        # Get dataset info
        info = self.dataset_navigator.get_dataset_info(dataset_id)
        if info is None:
            return

        # Update original SEG-Y path
        self.original_segy_path = str(info.source_path) if info.source_path else None

        # Get ensemble index
        ensembles_df = lazy_data._ensemble_index

        # Load into gather navigator
        self.gather_navigator.load_lazy_data(lazy_data, ensembles_df)

        # Reset batch processing state for new dataset
        self.full_processed_data = None
        self.sorted_headers_df = None
        self.is_full_dataset_processed = False

        # Display first gather
        self._display_current_gather()

        # Update control panel
        self.control_panel.update_nyquist(lazy_data.nyquist_freq)
        available_headers = self.gather_navigator.get_available_sort_headers()
        self.control_panel.set_available_sort_headers(available_headers)

        # Update datasets menu to show active
        self._update_datasets_menu()

        # Update app settings with active dataset
        self.app_settings.set_active_dataset_id(dataset_id)

        # Status message
        self.statusBar().showMessage(f"Switched to: {info.name}")

    def _on_dataset_added(self, dataset_id: str, info_dict: dict):
        """Handle new dataset added to DatasetNavigator."""
        self._update_datasets_menu()
        logger.info(f"Dataset added: {info_dict.get('name', 'Unknown')}")

    def _on_dataset_removed(self, dataset_id: str):
        """Handle dataset removed from DatasetNavigator."""
        self._update_datasets_menu()

        # Remove from persistent settings
        self.app_settings.remove_loaded_dataset(dataset_id)

        logger.info(f"Dataset removed: {dataset_id[:8]}...")

    def _on_datasets_cleared(self):
        """Handle all datasets cleared."""
        self._update_datasets_menu()

        # Clear viewers
        self.input_viewer.clear()
        self.processed_viewer.clear()
        self.difference_viewer.clear()
        self.input_data = None
        self.processed_data = None
        self.difference_data = None

        self.statusBar().showMessage("All datasets closed")

    def _update_datasets_menu(self):
        """Update the Datasets submenu with current datasets."""
        self.datasets_menu.clear()

        datasets = self.dataset_navigator.list_datasets()
        active_id = self.dataset_navigator.get_active_dataset_id()

        if not datasets:
            no_datasets = QAction("(No datasets loaded)", self)
            no_datasets.setEnabled(False)
            self.datasets_menu.addAction(no_datasets)
        else:
            for i, ds in enumerate(datasets):
                # Create action with dataset name
                name = ds.get('name', 'Unknown')
                action = QAction(f"&{i+1}. {name}", self)
                action.setCheckable(True)
                action.setChecked(ds.get('dataset_id') == active_id)
                action.setToolTip(f"Traces: {ds.get('n_traces', 0):,}, "
                                 f"Ensembles: {ds.get('n_ensembles', 0)}")

                # Connect to switch dataset
                dataset_id = ds.get('dataset_id')
                action.triggered.connect(
                    lambda checked, did=dataset_id: self._switch_to_dataset(did)
                )
                self.datasets_menu.addAction(action)

            # Add separator and management options
            self.datasets_menu.addSeparator()

            # Close all action
            close_all_action = QAction("Close &All Datasets", self)
            close_all_action.triggered.connect(self._close_all_datasets)
            self.datasets_menu.addAction(close_all_action)

    def _switch_to_dataset(self, dataset_id: str):
        """Switch to a specific dataset."""
        if dataset_id:
            self.dataset_navigator.set_active_dataset(dataset_id)

    def _close_current_dataset(self):
        """Close the currently active dataset."""
        active_id = self.dataset_navigator.get_active_dataset_id()
        if active_id:
            # Confirm if there's processed data
            if self.is_full_dataset_processed:
                reply = QMessageBox.question(
                    self,
                    "Close Dataset",
                    "This dataset has batch processed data that will be lost.\n\n"
                    "Are you sure you want to close it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            self.dataset_navigator.remove_dataset(active_id)
        else:
            self.statusBar().showMessage("No dataset to close", 3000)

    def _close_all_datasets(self):
        """Close all loaded datasets."""
        if not self.dataset_navigator.has_datasets():
            return

        reply = QMessageBox.question(
            self,
            "Close All Datasets",
            f"This will close all {self.dataset_navigator.get_dataset_count()} loaded datasets.\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.dataset_navigator.clear_all()

            # Clear persistent settings
            self.app_settings.save_loaded_datasets([])
            self.app_settings.set_active_dataset_id(None)

    # =========================================================================
    # Session State Management
    # =========================================================================

    def closeEvent(self, event):
        """Save session state before closing the application."""
        try:
            self._save_current_session()
            logger.info("Session saved on close")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

        # Accept the close event
        event.accept()

    def _save_current_session(self):
        """Save current session state to settings."""
        # Save viewport state
        limits = self.viewport_state.limits
        self.app_settings.save_viewport_state(
            limits.time_min,
            limits.time_max,
            limits.trace_min,
            limits.trace_max
        )

        # Save display state
        self.app_settings.save_display_state(
            colormap=self.viewport_state.colormap,
            interpolation=self.viewport_state.interpolation
        )

        # Save navigation state
        self.app_settings.save_navigation_state(
            current_gather_id=self.gather_navigator.current_gather_id,
            sort_keys=self.gather_navigator.sort_keys
        )

        # Save active dataset
        if self.dataset_navigator.has_datasets():
            active_id = self.dataset_navigator.get_active_dataset_id()
            self.app_settings.set_active_dataset_id(active_id)

            # Save dataset list
            datasets = self.dataset_navigator.list_datasets()
            self.app_settings.save_loaded_datasets(datasets)

        logger.debug("Session state saved")

    def _restore_last_session(self):
        """Restore last session on startup."""
        try:
            # Get saved datasets
            saved_datasets = self.app_settings.get_loaded_datasets()

            if not saved_datasets:
                self.statusBar().showMessage("Ready. Load seismic data to begin.")
                return

            # Restore datasets to navigator (data loaded on-demand)
            restored = self.dataset_navigator.restore_from_serialized({
                'datasets': saved_datasets,
                'active_dataset_id': self.app_settings.get_active_dataset_id()
            })

            if restored == 0:
                self.statusBar().showMessage(
                    "Previous datasets not found. Load seismic data to begin."
                )
                return

            # Update datasets menu
            self._update_datasets_menu()

            # Restore session state (viewport, display, navigation)
            session = self.app_settings.get_session_state()

            # Restore viewport (will be applied after data loads)
            self._pending_viewport = session.get('viewport', {})

            # Restore display settings
            colormap = session.get('colormap')
            if colormap:
                try:
                    self.viewport_state.set_colormap(colormap)
                except Exception:
                    pass

            interpolation = session.get('interpolation')
            if interpolation:
                try:
                    self.viewport_state.set_interpolation(interpolation)
                except Exception:
                    pass

            # Restore navigation state (will be applied after data loads)
            self._pending_gather_id = session.get('current_gather_id', 0)
            self._pending_sort_keys = session.get('sort_keys', [])

            # Status message
            self.statusBar().showMessage(
                f"Restored {restored} dataset(s) from last session"
            )

            logger.info(f"Restored {restored} datasets from last session")

        except Exception as e:
            logger.error(f"Failed to restore session: {e}", exc_info=True)
            self.statusBar().showMessage("Ready. Load seismic data to begin.")

    def _apply_pending_session_state(self):
        """Apply pending session state after data is loaded."""
        # Apply pending viewport
        if hasattr(self, '_pending_viewport') and self._pending_viewport:
            vp = self._pending_viewport
            try:
                self.viewport_state.set_limits(
                    vp.get('time_min', 0),
                    vp.get('time_max', 1000),
                    vp.get('trace_min', 0),
                    vp.get('trace_max', 100)
                )
            except Exception:
                pass
            self._pending_viewport = None

        # Apply pending gather navigation
        if hasattr(self, '_pending_gather_id') and self._pending_gather_id:
            try:
                if self._pending_gather_id < self.gather_navigator.n_gathers:
                    self.gather_navigator.goto_gather(self._pending_gather_id)
            except Exception:
                pass
            self._pending_gather_id = None

        # Apply pending sort keys
        if hasattr(self, '_pending_sort_keys') and self._pending_sort_keys:
            try:
                self.gather_navigator.set_sort_keys(self._pending_sort_keys)
            except Exception:
                pass
            self._pending_sort_keys = None


# Import will be added after creating sample data module
