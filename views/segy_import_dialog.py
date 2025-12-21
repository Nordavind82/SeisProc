"""
SEG-Y import dialog with header mapping configuration.
"""
import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QTableWidget, QTableWidgetItem,
                              QFileDialog, QGroupBox, QLineEdit, QComboBox,
                              QTextEdit, QMessageBox, QSpinBox, QCheckBox,
                              QHeaderView, QProgressDialog, QTabWidget, QScrollArea,
                              QWidget, QSplitter, QFrame, QFormLayout, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from typing import Optional, List
from pathlib import Path
import sys

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
from utils.segy_import.header_mapping import HeaderMapping, HeaderField, StandardHeaders
from utils.segy_import.computed_headers import ComputedHeaderField
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.data_storage import DataStorage
from utils.segy_import import (
    DEFAULT_CHUNK_SIZE,
    HEADER_BATCH_SIZE,
    PARALLEL_THRESHOLD,
    ParallelImportCoordinator,
    ImportConfig,
    ImportProgress,
    ImportStageResult,
    get_optimal_workers,
)
from utils.ray_orchestration.segy_workers import SEGYImportWorker, SEGYImportResult
from models.seismic_data import SeismicData
from models.app_settings import get_settings, AppSettings


class SEGYImportDialog(QDialog):
    """
    Dialog for importing SEG-Y files with custom header mapping.

    Allows user to:
    - Select SEG-Y file
    - Configure header byte mapping
    - Specify ensemble boundary keys
    - Preview headers
    - Import and save to Zarr/Parquet
    """

    import_completed = pyqtSignal(object, object, object, str)  # data (SeismicData or LazySeismicData), headers_df, ensembles_df, file_path
    import_started = pyqtSignal(object, str, dict)  # worker, output_dir, file_info - for async import

    def __init__(self, parent=None, initial_file: str = None):
        logger.info("SEGYImportDialog.__init__() - START")
        try:
            # CRITICAL: Set attribute to disable OpenGL before any widget creation
            # This prevents segfault when creating QTableWidget and other complex widgets
            logger.info("  → Setting Qt::AA_UseSoftwareOpenGL attribute...")
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import QApplication
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
            logger.info("  ✓ Software OpenGL enabled")

            super().__init__(parent)
            logger.info("  ✓ QDialog parent init complete")

            self.setWindowTitle("SEG-Y Import")
            logger.info("  ✓ Window title set")

            self.resize(900, 700)
            logger.info("  ✓ Window resized")

            self.segy_file = None
            self.header_mapping = HeaderMapping()
            self.reader = None
            self._initial_file = initial_file  # Store for loading after UI init
            self._import_worker: Optional[SEGYImportWorker] = None
            self._import_output_dir: Optional[str] = None  # Store for completion handler
            self._import_file_info: Optional[dict] = None  # Store for completion handler
            logger.info("  ✓ Instance variables initialized")

            logger.info("  → Calling _init_ui()...")
            self._init_ui()
            logger.info("  ✓ _init_ui() complete")

            logger.info("  → Calling _load_standard_headers()...")
            self._load_standard_headers()
            logger.info("  ✓ _load_standard_headers() complete")

            # Load initial file if provided (from recent files)
            if self._initial_file:
                logger.info(f"  → Loading initial file: {self._initial_file}")
                self._load_initial_file()
                logger.info("  ✓ Initial file loaded")

            logger.info("SEGYImportDialog.__init__() - COMPLETE")
        except Exception as e:
            logger.error(f"SEGYImportDialog.__init__() - FAILED: {e}", exc_info=True)
            raise

    def _load_initial_file(self):
        """Load the initial file specified at construction (for recent files)."""
        if self._initial_file and Path(self._initial_file).exists():
            self.file_path_edit.setText(self._initial_file)
            self._on_file_selected(self._initial_file)

    def _init_ui(self):
        """Initialize user interface with tabbed layout."""
        logger.info("    _init_ui() - Creating tabbed layout...")
        layout = QVBoxLayout()
        logger.info("    ✓ Layout created")

        # File selection - always visible at top
        logger.info("    → Creating file selection group...")
        layout.addWidget(self._create_file_selection_group())
        logger.info("    ✓ File selection group added")

        # Tab widget for main content
        logger.info("    → Creating tab widget...")
        self.tab_widget = QTabWidget()

        # Create tabs
        logger.info("    → Creating Headers tab...")
        self.tab_widget.addTab(self._create_headers_tab(), "Headers")
        logger.info("    ✓ Headers tab added")

        logger.info("    → Creating Import Settings tab...")
        self.tab_widget.addTab(self._create_import_settings_tab(), "Import Settings")
        logger.info("    ✓ Import Settings tab added")

        logger.info("    → Creating QC Inlines tab...")
        self.tab_widget.addTab(self._create_qc_inlines_tab(), "QC Inlines")
        logger.info("    ✓ QC Inlines tab added")

        logger.info("    → Creating Preview tab...")
        self.tab_widget.addTab(self._create_preview_tab(), "Preview")
        logger.info("    ✓ Preview tab added")

        layout.addWidget(self.tab_widget)
        logger.info("    ✓ Tab widget added")

        # Summary bar
        logger.info("    → Creating summary bar...")
        layout.addWidget(self._create_summary_bar())
        logger.info("    ✓ Summary bar added")

        # Action buttons - always visible at bottom
        logger.info("    → Creating action buttons...")
        layout.addLayout(self._create_action_buttons())
        logger.info("    ✓ Action buttons added")

        logger.info("    → Setting layout on dialog...")
        self.setLayout(layout)
        logger.info("    ✓ Layout set")

    def _create_file_selection_group(self) -> QGroupBox:
        """Create file selection group."""
        group = QGroupBox("SEG-Y File Selection")
        layout = QHBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select SEG-Y file...")
        self.file_path_edit.setReadOnly(True)
        layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_segy_file)
        layout.addWidget(browse_btn)

        info_btn = QPushButton("File Info")
        info_btn.clicked.connect(self._show_file_info)
        layout.addWidget(info_btn)

        group.setLayout(layout)
        return group

    # =========================================================================
    # Tab Creation Methods
    # =========================================================================

    def _create_headers_tab(self) -> QWidget:
        """Create Headers tab with header mapping and computed headers."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Use splitter for adjustable sizing
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Header mapping group (main table)
        splitter.addWidget(self._create_header_mapping_group())

        # Computed headers group
        splitter.addWidget(self._create_computed_headers_group())

        # Set initial sizes (70% for mapping, 30% for computed)
        splitter.setSizes([400, 150])

        layout.addWidget(splitter)
        return tab

    def _create_import_settings_tab(self) -> QWidget:
        """Create Import Settings tab with ensemble and output configuration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Ensemble configuration
        ensemble_group = QGroupBox("Ensemble Configuration")
        ensemble_layout = QFormLayout(ensemble_group)

        # Ensemble keys
        self.ensemble_keys_edit = QLineEdit()
        self.ensemble_keys_edit.setPlaceholderText("e.g., cdp or inline,crossline")
        self.ensemble_keys_edit.setToolTip(
            "Headers that define ensemble boundaries.\n"
            "Traces with the same ensemble key values are grouped together."
        )
        ensemble_layout.addRow("Ensemble Keys:", self.ensemble_keys_edit)

        # Preset buttons
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        cdp_btn = QPushButton("CDP")
        cdp_btn.clicked.connect(lambda: self.ensemble_keys_edit.setText("cdp"))
        preset_layout.addWidget(cdp_btn)

        shot_btn = QPushButton("Shot")
        shot_btn.clicked.connect(lambda: self.ensemble_keys_edit.setText("source_station"))
        preset_layout.addWidget(shot_btn)

        inline_btn = QPushButton("Inline/Xline")
        inline_btn.clicked.connect(lambda: self.ensemble_keys_edit.setText("inline,crossline"))
        preset_layout.addWidget(inline_btn)

        preset_layout.addStretch()
        ensemble_layout.addRow("", preset_layout)

        layout.addWidget(ensemble_group)

        # Coordinate settings
        coord_group = QGroupBox("Coordinate Settings")
        coord_layout = QFormLayout(coord_group)

        # Spatial units
        self.spatial_units_combo = QComboBox()
        self.spatial_units_combo.addItem("Meters (m)", AppSettings.METERS)
        self.spatial_units_combo.addItem("Feet (ft)", AppSettings.FEET)
        self.spatial_units_combo.setToolTip(
            "Select spatial units for coordinates and distances.\n"
            "This will be applied throughout the entire application."
        )

        # Set current value from AppSettings
        try:
            from models.app_settings import get_settings
            app_settings = get_settings()
            current_units = app_settings.get_spatial_units()
            if current_units == 'feet':
                self.spatial_units_combo.setCurrentIndex(1)
        except Exception as e:
            logger.warning(f"Could not load units from settings: {e}")

        coord_layout.addRow("Spatial Units:", self.spatial_units_combo)

        # Coordinate scalar info
        scalar_info = QLabel(
            "Coordinate scalars are automatically applied from trace headers.\n"
            "Negative scalars indicate division, positive indicate multiplication."
        )
        scalar_info.setStyleSheet("color: #666; font-style: italic;")
        scalar_info.setWordWrap(True)
        coord_layout.addRow("", scalar_info)

        layout.addWidget(coord_group)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)

        output_info = QLabel(
            "Output is automatically saved to the configured storage directory.\n"
            "Configure storage location in: Edit → Settings"
        )
        output_info.setStyleSheet("color: #666;")
        output_info.setWordWrap(True)
        output_layout.addRow("", output_info)

        layout.addWidget(output_group)

        layout.addStretch()
        return tab

    def _create_qc_inlines_tab(self) -> QWidget:
        """Create QC Inlines tab for configuring QC inline selection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Enable checkbox
        enable_group = QGroupBox("QC Inlines Configuration")
        enable_layout = QVBoxLayout(enable_group)

        self.qc_inlines_enabled = QCheckBox("Enable QC inline marking during import")
        self.qc_inlines_enabled.setToolTip(
            "Marks specific inlines for later QC stacking and batch processing.\n"
            "This information is stored in the dataset metadata."
        )
        self.qc_inlines_enabled.stateChanged.connect(self._on_qc_enabled_changed)
        enable_layout.addWidget(self.qc_inlines_enabled)

        layout.addWidget(enable_group)

        # Inline selection (initially disabled)
        self.qc_selection_group = QGroupBox("Inline Selection")
        self.qc_selection_group.setEnabled(False)
        selection_layout = QVBoxLayout(self.qc_selection_group)

        # Available range info
        self.qc_range_label = QLabel("Available range: (load file to detect)")
        self.qc_range_label.setStyleSheet("color: #666;")
        selection_layout.addWidget(self.qc_range_label)

        # Inline input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("QC Inlines:"))
        self.qc_inlines_edit = QLineEdit()
        self.qc_inlines_edit.setPlaceholderText("e.g., 100, 500, 1000-1100, 2000")
        self.qc_inlines_edit.setToolTip(
            "Enter inline numbers separated by commas.\n"
            "Use ranges with hyphen: 100-110\n"
            "Example: 100, 500, 1000-1100, 2000"
        )
        self.qc_inlines_edit.textChanged.connect(self._update_qc_count)
        input_layout.addWidget(self.qc_inlines_edit)
        selection_layout.addLayout(input_layout)

        # Quick select buttons
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick select:"))

        every_10_btn = QPushButton("Every 10th")
        every_10_btn.clicked.connect(lambda: self._qc_quick_select(10))
        quick_layout.addWidget(every_10_btn)

        every_50_btn = QPushButton("Every 50th")
        every_50_btn.clicked.connect(lambda: self._qc_quick_select(50))
        quick_layout.addWidget(every_50_btn)

        every_100_btn = QPushButton("Every 100th")
        every_100_btn.clicked.connect(lambda: self._qc_quick_select(100))
        quick_layout.addWidget(every_100_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.qc_inlines_edit.clear())
        quick_layout.addWidget(clear_btn)

        quick_layout.addStretch()
        selection_layout.addLayout(quick_layout)

        # Selected count
        self.qc_count_label = QLabel("Selected: 0 QC inlines")
        selection_layout.addWidget(self.qc_count_label)

        layout.addWidget(self.qc_selection_group)

        # Post-import options
        post_group = QGroupBox("Post-Import Options")
        post_layout = QVBoxLayout(post_group)

        self.auto_open_qc_batch = QCheckBox("Open QC Batch Processing dialog after import")
        self.auto_open_qc_batch.setToolTip(
            "Automatically opens the QC Batch Processing dialog\n"
            "after import completes, with QC inlines pre-configured."
        )
        self.auto_open_qc_batch.setEnabled(False)
        post_layout.addWidget(self.auto_open_qc_batch)

        layout.addWidget(post_group)

        layout.addStretch()
        return tab

    def _create_preview_tab(self) -> QWidget:
        """Create Preview tab with header preview and file statistics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header preview section
        preview_group = QGroupBox("Header Preview")
        preview_layout = QVBoxLayout(preview_group)

        # Preview buttons
        btn_layout = QHBoxLayout()
        preview_btn = QPushButton("Preview First 10 Traces")
        preview_btn.clicked.connect(self._preview_headers)
        btn_layout.addWidget(preview_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.preview_text.clear())
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()
        preview_layout.addLayout(btn_layout)

        # Preview text area
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Courier New", 9))
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        # File statistics section
        stats_group = QGroupBox("File Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlaceholderText("Load a SEG-Y file to view statistics")
        stats_layout.addWidget(self.stats_text)

        layout.addWidget(stats_group)

        return tab

    def _create_summary_bar(self) -> QFrame:
        """Create summary bar showing import configuration summary."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        frame.setStyleSheet("background-color: #f5f5f5; padding: 2px;")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)

        self.summary_label = QLabel("Select a SEG-Y file to begin")
        self.summary_label.setStyleSheet("color: #666;")
        layout.addWidget(self.summary_label)

        layout.addStretch()

        # Validation indicator
        self.validation_label = QLabel()
        layout.addWidget(self.validation_label)

        return frame

    # =========================================================================
    # QC Inlines Tab Helper Methods
    # =========================================================================

    def _on_qc_enabled_changed(self, state: int):
        """Handle QC inlines enabled checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.qc_selection_group.setEnabled(enabled)
        self.auto_open_qc_batch.setEnabled(enabled)
        self._update_summary()

    def _qc_quick_select(self, interval: int):
        """Quick select QC inlines at regular interval."""
        if not hasattr(self, '_inline_range') or self._inline_range is None:
            QMessageBox.warning(
                self, "No Data",
                "Load a SEG-Y file first to detect inline range."
            )
            return

        min_il, max_il = self._inline_range
        inlines = list(range(min_il, max_il + 1, interval))
        self.qc_inlines_edit.setText(", ".join(str(i) for i in inlines))

    def _update_qc_count(self):
        """Update QC line count display."""
        inlines = self._parse_qc_inlines()
        self.qc_count_label.setText(f"Selected: {len(inlines)} QC inlines")
        self._update_summary()

    def _parse_qc_inlines(self) -> List[int]:
        """Parse QC inline numbers from input field."""
        text = self.qc_inlines_edit.text().strip()
        if not text:
            return []

        inlines = []
        try:
            for part in text.split(','):
                part = part.strip()
                if not part:
                    continue
                if '-' in part:
                    start, end = part.split('-')
                    inlines.extend(range(int(start), int(end) + 1))
                else:
                    inlines.append(int(part))
            return sorted(set(inlines))
        except Exception:
            return []

    def _update_summary(self):
        """Update summary bar with current configuration."""
        parts = []

        if self.reader:
            try:
                info = self.reader.read_file_info()
                parts.append(f"{info['n_traces']:,} traces")
                parts.append(f"{info['n_samples']} samples")
            except Exception:
                pass

        if hasattr(self, 'ensemble_keys_edit') and self.ensemble_keys_edit.text():
            parts.append(f"Sort: {self.ensemble_keys_edit.text()}")

        if hasattr(self, 'qc_inlines_enabled') and self.qc_inlines_enabled.isChecked():
            n_qc = len(self._parse_qc_inlines())
            if n_qc > 0:
                parts.append(f"{n_qc} QC inlines")

        if parts:
            self.summary_label.setText(" | ".join(parts))
            self.summary_label.setStyleSheet("color: #333;")
        else:
            self.summary_label.setText("Select a SEG-Y file to begin")
            self.summary_label.setStyleSheet("color: #666;")

    def _update_file_statistics(self):
        """Update file statistics display."""
        if not self.reader:
            if hasattr(self, 'stats_text'):
                self.stats_text.clear()
            return

        try:
            info = self.reader.read_file_info()

            stats_html = f"""
            <table style="width: 100%;">
            <tr><td><b>Traces:</b></td><td>{info['n_traces']:,}</td>
                <td><b>Samples:</b></td><td>{info['n_samples']}</td></tr>
            <tr><td><b>Sample Interval:</b></td><td>{info['sample_interval']:.2f} ms</td>
                <td><b>Trace Length:</b></td><td>{info['trace_length_ms']:.1f} ms</td></tr>
            <tr><td><b>Data Format:</b></td><td>{info['format']}</td>
                <td><b>Est. Size:</b></td><td>{(info['n_traces'] * info['n_samples'] * 4 / 1024 / 1024):.1f} MB</td></tr>
            </table>
            """

            if hasattr(self, 'stats_text'):
                self.stats_text.setHtml(stats_html)

            # Try to detect inline range for QC
            self._detect_inline_range()

        except Exception as e:
            if hasattr(self, 'stats_text'):
                self.stats_text.setText(f"Error reading file info: {e}")

    def _detect_inline_range(self):
        """Detect inline range from first few traces using header mapping."""
        self._inline_range = None

        if not self.reader:
            return

        try:
            # First, find the inline field name from the header mapping
            inline_field_name = None
            inline_patterns = ['inline', 'inline_no', 'inline_3d', 'il', 'iline']

            # Search through defined header fields
            for field_name in self.header_mapping.fields.keys():
                if any(pattern in field_name.lower() for pattern in inline_patterns):
                    inline_field_name = field_name
                    break

            # Also check computed headers
            if not inline_field_name:
                for computed_field in self.header_mapping.computed_fields:
                    if any(pattern in computed_field.name.lower() for pattern in inline_patterns):
                        inline_field_name = computed_field.name
                        break

            if not inline_field_name:
                if hasattr(self, 'qc_range_label'):
                    self.qc_range_label.setText(
                        "No inline header detected in mapping.\n"
                        "Add an 'inline' or 'INLINE_NO' field to header mapping."
                    )
                return

            # Read sample headers using the mapping
            headers = self.reader.read_sample_headers(n_traces=100)

            # Extract inline values using the detected field name
            inline_values = []
            for header in headers:
                if inline_field_name in header:
                    val = header[inline_field_name]
                    if val is not None and val != 0:
                        inline_values.append(val)

            if inline_values:
                min_il = min(inline_values)
                max_il = max(inline_values)
                # Estimate full range
                self._inline_range = (min_il, max_il)
                if hasattr(self, 'qc_range_label'):
                    self.qc_range_label.setText(
                        f"Detected range (sample): {min_il} - {max_il}\n"
                        f"Using header field: '{inline_field_name}'"
                    )
            else:
                if hasattr(self, 'qc_range_label'):
                    self.qc_range_label.setText(
                        f"Field '{inline_field_name}' found but no valid values.\n"
                        "Check header mapping configuration."
                    )

        except Exception as e:
            logger.warning(f"Could not detect inline range: {e}")
            if hasattr(self, 'qc_range_label'):
                self.qc_range_label.setText(f"Error detecting range: {e}")

    def _on_file_selected(self, filepath: str):
        """Handle file selection - update reader and UI."""
        self.segy_file = filepath
        self._update_reader()
        self.import_btn.setEnabled(True)

        # Load sample header values if table has headers
        if self.header_table.rowCount() > 0:
            self._load_sample_header_values()

        # Update file statistics
        self._update_file_statistics()

        # Update summary
        self._update_summary()

    # =========================================================================
    # Original Group Creation Methods (now used by tabs)
    # =========================================================================

    def _create_header_mapping_group(self) -> QGroupBox:
        """Create header mapping configuration table."""
        logger.info("      _create_header_mapping_group() - START")
        group = QGroupBox("Trace Header Mapping Configuration")
        layout = QVBoxLayout()

        # Toolbar
        logger.info("        → Creating toolbar...")
        toolbar = QHBoxLayout()

        add_btn = QPushButton("Add Custom Header")
        add_btn.clicked.connect(self._add_custom_header)
        toolbar.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_header)
        toolbar.addWidget(remove_btn)

        toolbar.addStretch()

        # Save/Load mapping buttons
        save_mapping_btn = QPushButton("Save Mapping...")
        save_mapping_btn.clicked.connect(self._save_mapping_to_file)
        save_mapping_btn.setToolTip("Save current header mapping configuration to file")
        toolbar.addWidget(save_mapping_btn)

        load_mapping_btn = QPushButton("Load Mapping...")
        load_mapping_btn.clicked.connect(self._load_mapping_from_file)
        load_mapping_btn.setToolTip("Load header mapping configuration from file")
        toolbar.addWidget(load_mapping_btn)

        load_preset_btn = QPushButton("Load Standard Headers")
        load_preset_btn.clicked.connect(self._load_standard_headers)
        toolbar.addWidget(load_preset_btn)

        layout.addLayout(toolbar)
        logger.info("        ✓ Toolbar created")

        # Table
        logger.info("        → Creating QTableWidget...")
        self.header_table = QTableWidget()
        logger.info("        ✓ QTableWidget created")

        logger.info("        → Setting column count...")
        self.header_table.setColumnCount(5)
        logger.info("        ✓ Column count set")

        logger.info("        → Setting horizontal headers...")
        self.header_table.setHorizontalHeaderLabels([
            'Header Name', 'Byte Position', 'Format', 'Description', 'Sample Values (5 traces)'
        ])
        logger.info("        ✓ Headers set")

        # Set column stretch modes
        logger.info("        → Configuring header resize modes...")
        header = self.header_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Byte
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Format
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Description
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Sample Values
        logger.info("        ✓ Resize modes set")

        logger.info("        → Setting selection behavior...")
        self.header_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        logger.info("        ✓ Selection behavior set")

        logger.info("        → Adding table to layout...")
        layout.addWidget(self.header_table)
        logger.info("        ✓ Table added to layout")

        logger.info("        → Setting layout on group...")
        group.setLayout(layout)
        logger.info("        ✓ Layout set on group")

        logger.info("      _create_header_mapping_group() - COMPLETE")
        return group

    def _create_computed_headers_group(self) -> QGroupBox:
        """Create computed headers configuration table."""
        group = QGroupBox("Computed Headers (Trace Header Math)")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Define new headers computed from existing ones using math equations.")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)

        # Toolbar
        toolbar = QHBoxLayout()

        add_computed_btn = QPushButton("Add Computed Header")
        add_computed_btn.clicked.connect(self._add_computed_header)
        toolbar.addWidget(add_computed_btn)

        remove_computed_btn = QPushButton("Remove Selected")
        remove_computed_btn.clicked.connect(self._remove_selected_computed_header)
        toolbar.addWidget(remove_computed_btn)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Table
        self.computed_header_table = QTableWidget()
        self.computed_header_table.setColumnCount(4)
        self.computed_header_table.setHorizontalHeaderLabels([
            'Header Name', 'Expression', 'Format', 'Description'
        ])
        # Set column stretch modes
        header = self.computed_header_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)           # Expression
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Format
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Description
        self.computed_header_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.computed_header_table.setMaximumHeight(150)
        layout.addWidget(self.computed_header_table)

        # Help text
        help_text = QLabel(
            "Example: <b>receiver_line = round(receiver_station / 1000)</b><br>"
            "Available: +, -, *, /, //, %, **, round, floor, ceil, abs, min, max, sqrt, sin, cos, tan, atan2"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #444; padding: 5px; background: #f0f0f0; border-radius: 3px;")
        layout.addWidget(help_text)

        group.setLayout(layout)
        return group

    # NOTE: _create_ensemble_group removed - now in _create_import_settings_tab
    # NOTE: _create_preview_group removed - now in _create_preview_tab

    def _create_action_buttons(self) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()

        layout.addStretch()

        self.import_btn = QPushButton("Import SEG-Y")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.import_btn.clicked.connect(self._import_segy)
        self.import_btn.setEnabled(False)
        layout.addWidget(self.import_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        return layout

    def _browse_segy_file(self):
        """Browse for SEG-Y file."""
        logger.info("_browse_segy_file() - START")
        try:
            logger.info("  → Opening QFileDialog...")
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Select SEG-Y File",
                "",
                "SEG-Y Files (*.sgy *.segy *.SGY *.SEGY);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog  # Prevent native dialog crash
            )
            logger.info(f"  ✓ QFileDialog returned: {filename}")

            if filename:
                self.file_path_edit.setText(filename)
                self._on_file_selected(filename)

            logger.info("_browse_segy_file() - COMPLETE")
        except Exception as e:
            logger.error(f"_browse_segy_file() - FAILED: {e}", exc_info=True)
            raise

    def _show_file_info(self):
        """Show SEG-Y file information in a nice scrollable dialog."""
        if not self.reader:
            QMessageBox.warning(self, "No File", "Please select a SEG-Y file first.")
            return

        try:
            info = self.reader.read_file_info()

            # Create custom dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("SEG-Y File Information")
            dialog.resize(700, 600)

            layout = QVBoxLayout()

            # Create tab widget
            tabs = QTabWidget()

            # ==================== Overview Tab ====================
            overview_widget = QWidget()
            overview_layout = QVBoxLayout()

            overview_text = QTextEdit()
            overview_text.setReadOnly(True)
            overview_text.setHtml(f"""
<h2>File Overview</h2>
<table style="width: 100%; border-collapse: collapse;">
    <tr style="background-color: #f0f0f0;">
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Filename</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['filename']}</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Number of Traces</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['n_traces']:,}</td>
    </tr>
    <tr style="background-color: #f0f0f0;">
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Samples per Trace</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['n_samples']}</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Sample Interval</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['sample_interval']:.2f} ms</td>
    </tr>
    <tr style="background-color: #f0f0f0;">
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Trace Length</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['trace_length_ms']:.1f} ms</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Data Format</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['format']}</td>
    </tr>
</table>

<h3 style="margin-top: 20px;">File Size</h3>
<table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Total Data Points</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['n_traces'] * info['n_samples']:,}</td>
    </tr>
    <tr style="background-color: #f0f0f0;">
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Estimated Size (uncompressed)</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{(info['n_traces'] * info['n_samples'] * 4 / 1024 / 1024):.1f} MB</td>
    </tr>
</table>
            """)
            overview_layout.addWidget(overview_text)
            overview_widget.setLayout(overview_layout)
            tabs.addTab(overview_widget, "Overview")

            # ==================== Binary Header Tab ====================
            binary_widget = QWidget()
            binary_layout = QVBoxLayout()

            binary_text = QTextEdit()
            binary_text.setReadOnly(True)
            binary_text.setHtml(f"""
<h2>Binary Header</h2>
<table style="width: 100%; border-collapse: collapse;">
    <tr style="background-color: #f0f0f0;">
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Job ID</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['binary_header']['job_id']}</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Line Number</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['binary_header']['line_number']}</td>
    </tr>
    <tr style="background-color: #f0f0f0;">
        <td style="padding: 8px; border: 1px solid #ddd;"><b>Reel Number</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">{info['binary_header']['reel_number']}</td>
    </tr>
</table>

<p style="margin-top: 20px; color: #666; font-style: italic;">
The binary header contains file-level metadata about the entire SEG-Y dataset.
</p>
            """)
            binary_layout.addWidget(binary_text)
            binary_widget.setLayout(binary_layout)
            tabs.addTab(binary_widget, "Binary Header")

            # ==================== Text Header Tab ====================
            text_widget = QWidget()
            text_layout = QVBoxLayout()

            text_header_edit = QTextEdit()
            text_header_edit.setReadOnly(True)

            # Use monospace font for text header
            mono_font = QFont("Courier New", 9)
            text_header_edit.setFont(mono_font)

            # Format text header - SEG-Y text headers are 3200 bytes (40 lines of 80 chars)
            text_header = info['text_header']
            formatted_lines = []
            for i in range(0, len(text_header), 80):
                line_num = i // 80 + 1
                line = text_header[i:i+80]
                formatted_lines.append(f"C{line_num:02d} {line}")

            text_header_edit.setPlainText('\n'.join(formatted_lines))

            text_layout.addWidget(QLabel("SEG-Y Textual File Header (3200 bytes):"))
            text_layout.addWidget(text_header_edit)

            text_widget.setLayout(text_layout)
            tabs.addTab(text_widget, "Text Header")

            layout.addWidget(tabs)

            # Close button
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setDefault(True)
            button_layout.addWidget(close_btn)
            layout.addLayout(button_layout)

            dialog.setLayout(layout)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file info:\n{str(e)}")

    def _update_reader(self):
        """Update SEG-Y reader with current header mapping."""
        if self.segy_file:
            self._update_mapping_from_table()
            # Use simple SEGYReader for preview and small operations
            # Large file import uses multiprocess coordinator
            self.reader = SEGYReader(self.segy_file, self.header_mapping)

    def _create_format_combo(self, default_format: str = 'i') -> QComboBox:
        """
        Create a combobox with SEG-Y format options.

        Format codes:
        - h: 2-byte signed integer (2i)
        - H: 2-byte unsigned integer (2u)
        - i: 4-byte signed integer (4i)
        - I: 4-byte unsigned integer (4u)
        - f: 4-byte float (4r)
        - d: 8-byte double (8r)
        - b: 1-byte signed integer (1i)
        - B: 1-byte unsigned integer (1u)
        """
        combo = QComboBox()
        formats = [
            ('h', '2i - 2-byte signed int'),
            ('H', '2u - 2-byte unsigned int'),
            ('i', '4i - 4-byte signed int (default)'),
            ('I', '4u - 4-byte unsigned int'),
            ('f', '4r - 4-byte float'),
            ('d', '8r - 8-byte double'),
            ('b', '1i - 1-byte signed int'),
            ('B', '1u - 1-byte unsigned int'),
        ]

        for code, label in formats:
            combo.addItem(label, code)

        # Set default
        index = combo.findData(default_format)
        if index >= 0:
            combo.setCurrentIndex(index)

        return combo

    def _save_mapping_to_file(self):
        """Save current header mapping configuration to file."""
        try:
            # Update mapping from table first
            self._update_mapping_from_table()

            # Ask for save location
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Header Mapping",
                "",
                "JSON Files (*.json);;All Files (*)"
            )

            if filename:
                # Add .json extension if not present
                if not filename.endswith('.json'):
                    filename += '.json'

                self.header_mapping.save_to_file(filename)
                QMessageBox.information(
                    self,
                    "Mapping Saved",
                    f"Header mapping configuration saved to:\n{filename}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save mapping:\n{str(e)}"
            )

    def _load_mapping_from_file(self):
        """Load header mapping configuration from file."""
        try:
            # Ask for file location
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load Header Mapping",
                "",
                "JSON Files (*.json);;All Files (*)"
            )

            if filename:
                self.header_mapping = HeaderMapping.load_from_file(filename)
                self._populate_table_from_mapping()

                # Load ensemble keys if present
                if self.header_mapping.ensemble_keys:
                    self.ensemble_keys_edit.setText(','.join(self.header_mapping.ensemble_keys))

                QMessageBox.information(
                    self,
                    "Mapping Loaded",
                    f"Header mapping loaded from:\n{filename}\n\n"
                    f"Fields loaded: {len(self.header_mapping.fields)}"
                )

                # Update reader if file is selected
                if self.segy_file:
                    self._update_reader()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load mapping:\n{str(e)}"
            )

    def _load_standard_headers(self):
        """Load standard SEG-Y headers into table."""
        self.header_mapping.add_standard_headers(StandardHeaders.get_all_standard())
        self._populate_table_from_mapping()

    def _add_custom_header(self):
        """Add custom header row to table."""
        row = self.header_table.rowCount()
        self.header_table.insertRow(row)

        # Default values
        self.header_table.setItem(row, 0, QTableWidgetItem("custom_header"))
        self.header_table.setItem(row, 1, QTableWidgetItem("1"))

        # Format combobox
        format_combo = self._create_format_combo('i')
        self.header_table.setCellWidget(row, 2, format_combo)

        self.header_table.setItem(row, 3, QTableWidgetItem("Custom header description"))
        self.header_table.setItem(row, 4, QTableWidgetItem("N/A"))

    def _remove_selected_header(self):
        """Remove selected header rows."""
        selected_rows = set(index.row() for index in self.header_table.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            self.header_table.removeRow(row)

    def _add_computed_header(self):
        """Add computed header row to table."""
        row = self.computed_header_table.rowCount()
        self.computed_header_table.insertRow(row)

        # Default values
        self.computed_header_table.setItem(row, 0, QTableWidgetItem("receiver_line"))
        self.computed_header_table.setItem(row, 1, QTableWidgetItem("round(receiver_station / 1000)"))

        # Format combobox
        format_combo = self._create_format_combo('i')
        self.computed_header_table.setCellWidget(row, 2, format_combo)

        self.computed_header_table.setItem(row, 3, QTableWidgetItem("Computed receiver line"))

    def _remove_selected_computed_header(self):
        """Remove selected computed header rows."""
        selected_rows = set(index.row() for index in self.computed_header_table.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            self.computed_header_table.removeRow(row)

    def _populate_table_from_mapping(self):
        """Populate table from header mapping."""
        # Populate raw headers table
        self.header_table.setRowCount(0)

        for field in self.header_mapping.fields.values():
            row = self.header_table.rowCount()
            self.header_table.insertRow(row)

            self.header_table.setItem(row, 0, QTableWidgetItem(field.name))
            self.header_table.setItem(row, 1, QTableWidgetItem(str(field.byte_position)))

            # Format combobox
            format_combo = self._create_format_combo(field.format)
            self.header_table.setCellWidget(row, 2, format_combo)

            self.header_table.setItem(row, 3, QTableWidgetItem(field.description))
            self.header_table.setItem(row, 4, QTableWidgetItem("Loading..."))

        # Populate computed headers table
        self.computed_header_table.setRowCount(0)

        for field in self.header_mapping.computed_fields:
            row = self.computed_header_table.rowCount()
            self.computed_header_table.insertRow(row)

            self.computed_header_table.setItem(row, 0, QTableWidgetItem(field.name))
            self.computed_header_table.setItem(row, 1, QTableWidgetItem(field.expression))

            # Format combobox
            format_combo = self._create_format_combo(field.format)
            self.computed_header_table.setCellWidget(row, 2, format_combo)

            self.computed_header_table.setItem(row, 3, QTableWidgetItem(field.description))

        # Load sample values if file is selected
        if self.segy_file:
            self._load_sample_header_values()

    def _update_mapping_from_table(self):
        """Update header mapping from table contents."""
        # Update raw headers
        self.header_mapping.fields.clear()

        for row in range(self.header_table.rowCount()):
            try:
                name = self.header_table.item(row, 0).text()
                byte_pos = int(self.header_table.item(row, 1).text())

                # Get format from combobox
                format_widget = self.header_table.cellWidget(row, 2)
                if isinstance(format_widget, QComboBox):
                    format_str = format_widget.currentData()
                else:
                    # Fallback for old text items
                    format_str = self.header_table.item(row, 2).text()

                description = self.header_table.item(row, 3).text()

                field = HeaderField(name, byte_pos, format_str, description)
                self.header_mapping.add_field(field)
            except Exception as e:
                print(f"Warning: Skipping invalid row {row}: {e}")

        # Update computed headers
        self.header_mapping.computed_fields.clear()
        self.header_mapping._computed_processor = None

        for row in range(self.computed_header_table.rowCount()):
            try:
                name = self.computed_header_table.item(row, 0).text()
                expression = self.computed_header_table.item(row, 1).text()

                # Get format from combobox
                format_widget = self.computed_header_table.cellWidget(row, 2)
                if isinstance(format_widget, QComboBox):
                    format_str = format_widget.currentData()
                else:
                    format_str = 'f'  # Default to float

                description = self.computed_header_table.item(row, 3).text()

                field = ComputedHeaderField(name, expression, description, format_str)
                self.header_mapping.add_computed_field(field)
            except Exception as e:
                print(f"Warning: Skipping invalid computed header row {row}: {e}")

    def _preview_headers(self):
        """Preview headers from SEG-Y file."""
        if not self.reader:
            QMessageBox.warning(self, "No File", "Please select a SEG-Y file first.")
            return

        try:
            self._update_reader()
            headers = self.reader.read_sample_headers(n_traces=10)

            # Format preview text
            preview = "Sample Headers (first 10 traces):\n\n"
            for i, header in enumerate(headers):
                preview += f"Trace {i+1}:\n"
                for key, value in header.items():
                    preview += f"  {key:20s} = {value}\n"
                preview += "\n"

            self.preview_text.setPlainText(preview)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preview headers:\n{str(e)}")

    def _load_sample_header_values(self):
        """Load sample header values from first 5 traces and update table."""
        if not self.segy_file:
            return

        try:
            import segyio

            # Open SEG-Y file
            with segyio.open(self.segy_file, 'r', ignore_geometry=True) as f:
                n_traces = min(5, f.tracecount)  # Read up to 5 traces

                # Read headers for first n_traces
                sample_values = {}

                for row in range(self.header_table.rowCount()):
                    # Get header name and byte position from table items
                    name_item = self.header_table.item(row, 0)
                    pos_item = self.header_table.item(row, 1)

                    if not name_item or not pos_item:
                        continue

                    header_name = name_item.text()
                    byte_pos = int(pos_item.text())

                    # Get format from combobox widget (column 2 is a widget, not an item)
                    format_combo = self.header_table.cellWidget(row, 2)
                    if format_combo:
                        fmt = format_combo.currentData()  # Get the format code (e.g., 'i', 'f', 'h')
                    else:
                        fmt = 'i'  # Default format

                    values = []
                    for trace_idx in range(n_traces):
                        try:
                            # Get header bytes
                            header_dict = f.header[trace_idx]
                            header_bytes = bytes(header_dict.buf)

                            # Read value at byte position
                            from utils.segy_import.header_mapping import HeaderField
                            field = HeaderField(
                                name=header_name,
                                byte_position=byte_pos,
                                format=fmt,
                                description=""
                            )
                            value = field.read_value(header_bytes)
                            values.append(str(value))

                        except Exception:
                            values.append("N/A")

                    # Format values string
                    values_str = ", ".join(values)
                    if len(values_str) > 60:
                        values_str = values_str[:57] + "..."

                    # Update table
                    self.header_table.setItem(row, 4, QTableWidgetItem(values_str))

        except Exception as e:
            # If error, just mark as unavailable
            for row in range(self.header_table.rowCount()):
                self.header_table.setItem(row, 4, QTableWidgetItem(f"Error: {str(e)[:20]}"))

    def _import_segy(self):
        """Import SEG-Y file with current configuration."""
        logger.info("_import_segy() - START")
        if not self.reader:
            QMessageBox.warning(self, "No File", "Please select a SEG-Y file first.")
            return

        try:
            # Update mapping and ensemble keys
            logger.info("  → Updating reader...")
            self._update_reader()
            logger.info("  ✓ Reader updated")

            logger.info("  → Processing ensemble keys...")
            ensemble_keys_text = self.ensemble_keys_edit.text().strip()
            ensemble_keys = []
            if ensemble_keys_text:
                ensemble_keys = [k.strip() for k in ensemble_keys_text.split(',')]
                self.header_mapping.set_ensemble_keys(ensemble_keys)
            logger.info(f"  ✓ Ensemble keys: {ensemble_keys}")

            # Get spatial units selection and save to AppSettings
            logger.info("  → Getting selected units...")
            selected_units = self.spatial_units_combo.currentData()
            logger.info(f"  ✓ Selected units: {selected_units}")

            # Get AppSettings instance
            from models.app_settings import get_settings
            app_settings = get_settings()

            # Save to AppSettings (uses JSON file, no QSettings crash)
            try:
                app_settings.set_spatial_units(selected_units)
                logger.info(f"  ✓ Spatial units saved to settings: {selected_units}")
            except Exception as e:
                logger.warning(f"  Could not save spatial units: {e}")

            # Determine output directory
            # Use configured storage directory + filename-based subdirectory
            from pathlib import Path
            from datetime import datetime

            base_storage_dir = app_settings.get_effective_storage_directory()

            # Create unique subdirectory based on filename and timestamp
            segy_name = Path(self.segy_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subdir_name = f"{segy_name}_{timestamp}"

            output_dir = str(base_storage_dir / subdir_name)
            logger.info(f"  ✓ Using storage directory: {output_dir}")

            # Get file info to determine import strategy
            file_info = self.reader.read_file_info()
            n_traces = file_info['n_traces']
            n_samples = file_info['n_samples']

            import os
            file_size_mb = os.path.getsize(self.segy_file) / 1024 / 1024

            # Decide between streaming and batch import
            # Use streaming if: file > 500MB OR traces > 50,000
            use_streaming = file_size_mb > 500 or n_traces > 50000

            if use_streaming:
                print(f"Using STREAMING import (file: {file_size_mb:.1f} MB, traces: {n_traces:,})")
                self._import_streaming(output_dir, file_info, ensemble_keys)
            else:
                print(f"Using BATCH import (file: {file_size_mb:.1f} MB, traces: {n_traces:,})")
                self._import_batch(output_dir)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Import Error", f"Failed to import SEG-Y:\n{str(e)}")

    def _import_batch(self, output_dir: str):
        """Import using traditional batch method (for small files)."""
        # Create progress dialog
        progress = QProgressDialog("Importing SEG-Y file...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(10)

        # Read SEG-Y file
        progress.setLabelText("Reading SEG-Y file...")
        seismic_data, headers, ensembles = self.reader.read_to_seismic_data()
        progress.setValue(50)

        # Add original SEG-Y path to metadata for export functionality
        seismic_data.metadata['original_segy_path'] = self.segy_file

        # Save to Zarr/Parquet
        progress.setLabelText("Saving to Zarr/Parquet...")
        storage = DataStorage(output_dir)
        storage.save_seismic_data(seismic_data, headers, ensembles)
        progress.setValue(85)

        # Add QC configuration if enabled
        if hasattr(self, 'qc_inlines_enabled') and self.qc_inlines_enabled.isChecked():
            import json
            from pathlib import Path
            qc_inlines = self._parse_qc_inlines()
            if qc_inlines:
                metadata_path = Path(output_dir) / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['qc_config'] = {
                        'enabled': True,
                        'inline_numbers': qc_inlines,
                        'auto_open_batch': self.auto_open_qc_batch.isChecked(),
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

        progress.setValue(90)

        # Load back for verification
        progress.setLabelText("Loading data...")
        loaded_data, headers_df, ensembles_df = storage.load_seismic_data()
        progress.setValue(100)

        # Print computed header errors if any
        error_summary = self.reader.get_computed_header_errors()
        if error_summary:
            print(error_summary)

        # Show statistics
        self._show_success_stats(storage, loaded_data, output_dir)

        # Emit signal with loaded data and file path
        self.import_completed.emit(loaded_data, headers_df, ensembles_df, self.segy_file)
        self.accept()

    def _import_streaming(self, output_dir: str, file_info: dict, ensemble_keys: List[str]):
        """Import using multiprocess parallel method (for large files).

        Uses SEGYImportWorker for background execution with centralized
        job dashboard progress tracking (no built-in QProgressDialogs).
        """
        import json

        n_workers = get_optimal_workers()

        # Configure parallel import
        config = ImportConfig(
            segy_path=str(self.segy_file),
            output_dir=output_dir,
            header_mapping=self.header_mapping,
            ensemble_key=ensemble_keys[0] if ensemble_keys else None,
            n_workers=n_workers,
            chunk_size=DEFAULT_CHUNK_SIZE
        )

        # Store state for completion handler
        self._import_output_dir = output_dir
        self._import_file_info = file_info

        # Create worker
        self._import_worker = SEGYImportWorker(
            config,
            job_name=Path(self.segy_file).name,
            parent=self
        )

        # Show job dashboard if available (through parent main window)
        self._show_job_dashboard()

        # Start background import
        logger.info(f"Starting parallel import with {n_workers} workers (Job Dashboard)...")
        self._import_worker.start()

        # Emit signal for main window to take over worker handling
        # Pass worker, output_dir, and file_info so main window can handle completion
        self.import_started.emit(
            self._import_worker,
            output_dir,
            {
                'file_info': file_info,
                'segy_file': str(self.segy_file),
                'header_mapping': self.header_mapping.to_dict() if self.header_mapping else {},
                'qc_config': self._get_qc_config() if hasattr(self, '_get_qc_config') else None,
            }
        )

        # Close dialog - main window handles the rest via import_started signal
        self.accept()

    def _get_qc_config(self) -> Optional[dict]:
        """Get QC configuration if enabled."""
        if hasattr(self, 'qc_inlines_enabled') and self.qc_inlines_enabled.isChecked():
            qc_inlines = self._parse_qc_inlines()
            if qc_inlines:
                return {
                    'enabled': True,
                    'inline_numbers': qc_inlines,
                    'auto_open_batch': self.auto_open_qc_batch.isChecked(),
                }
        return None

    def _show_job_dashboard(self):
        """Show the job dashboard in main window if available."""
        try:
            parent = self.parent()
            if parent and hasattr(parent, '_job_integration') and parent._job_integration:
                parent._job_integration.show_job_dashboard()
        except Exception as e:
            logger.debug(f"Could not show job dashboard: {e}")

    def _on_streaming_import_complete(self, result: SEGYImportResult):
        """Handle successful streaming import completion."""
        import json

        output_dir = self._import_output_dir
        file_info = self._import_file_info

        try:
            # Update metadata with source file info
            metadata_path = Path(output_dir) / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['seismic_metadata'] = {
                    'source_file': str(self.segy_file),
                    'original_segy_path': str(self.segy_file),
                    'file_info': file_info,
                    'header_mapping': self.header_mapping.to_dict(),
                }

                # Add QC configuration if enabled
                if hasattr(self, 'qc_inlines_enabled') and self.qc_inlines_enabled.isChecked():
                    qc_inlines = self._parse_qc_inlines()
                    if qc_inlines:
                        metadata['qc_config'] = {
                            'enabled': True,
                            'inline_numbers': qc_inlines,
                            'auto_open_batch': self.auto_open_qc_batch.isChecked(),
                        }

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # Show success statistics
            n_traces = file_info['n_traces']
            n_samples = file_info['n_samples']
            n_workers = get_optimal_workers()
            elapsed = result.elapsed_time

            stats_text = f"""
Parallel Import Successful!

Data Statistics:
- Traces: {n_traces:,}
- Samples: {n_samples}

Performance:
- Import time: {elapsed:.1f}s
- Throughput: {n_traces / elapsed:,.0f} traces/sec
- Workers used: {n_workers}

Output: {output_dir}
            """

            QMessageBox.information(self, "Import Complete", stats_text)

            # Load back for viewer using lazy loading
            from models.lazy_seismic_data import LazySeismicData

            lazy_data = LazySeismicData.from_storage_dir(output_dir)

            # Load ensemble index if exists
            storage = DataStorage(output_dir)
            ensembles_df = storage.get_ensemble_index()

            # Emit signal with lazy data
            self.import_completed.emit(lazy_data, None, ensembles_df, self.segy_file)

            self.accept()

        except Exception as e:
            logger.error(f"Error in import completion handler: {e}")
            QMessageBox.critical(self, "Error", f"Error loading imported data:\n{e}")
            self.setEnabled(True)
            self.setWindowTitle("SEG-Y Import")

    def _on_streaming_import_error(self, error_msg: str):
        """Handle streaming import error or cancellation."""
        import shutil

        output_dir = self._import_output_dir

        # Clean up partial output
        if output_dir:
            output_path = Path(output_dir)
            if output_path.exists():
                try:
                    shutil.rmtree(output_path, ignore_errors=True)
                except Exception:
                    pass

        # Re-enable dialog
        self.setEnabled(True)
        self.setWindowTitle("SEG-Y Import")

        if "cancelled" in error_msg.lower():
            QMessageBox.information(self, "Import Cancelled", "Import was cancelled by user.")
        else:
            QMessageBox.critical(self, "Import Failed", f"Import failed:\n{error_msg}")

    def _show_success_stats(self, storage: DataStorage, loaded_data, output_dir: str):
        """Show success statistics dialog."""
        stats = storage.get_statistics()
        stats_text = f"""
Import Successful!

Data Statistics:
- Traces: {stats['headers']['n_traces']:,}
- Samples: {loaded_data.n_samples}
- Ensembles: {stats.get('ensembles', {}).get('n_ensembles', 'N/A')}

Storage:
- Zarr size: {stats['zarr']['size_mb']:.1f} MB
- Headers size: {stats['headers']['size_mb']:.1f} MB
- Compression: {stats['zarr']['size_mb'] / (loaded_data.traces.nbytes / 1024 / 1024):.2f}x

Output: {output_dir}
        """

        QMessageBox.information(self, "Import Complete", stats_text)
