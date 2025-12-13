"""
Export Options Dialog - Comprehensive UI for SEG-Y export configuration.

Combines dataset selection, validation, export type, and mute options in one dialog.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import zarr
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QRadioButton,
    QCheckBox, QDoubleSpinBox, QSpinBox, QFileDialog,
    QDialogButtonBox, QFrame, QMessageBox, QButtonGroup,
    QComboBox, QTabWidget, QWidget, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import segyio


@dataclass
class DatasetInfo:
    """Information about a Zarr dataset."""
    path: Path
    n_traces: int = 0
    n_samples: int = 0
    sample_rate_ms: float = 0.0
    duration_ms: float = 0.0
    sort_key: Optional[str] = None
    sort_ascending: bool = True
    is_sorted: bool = False
    ensemble_key: Optional[str] = None
    n_ensembles: int = 0
    valid: bool = False
    error: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path) -> 'DatasetInfo':
        """Load dataset info from a Zarr storage directory."""
        info = cls(path=path)

        try:
            zarr_path = path / 'traces.zarr'
            metadata_path = path / 'metadata.json'
            headers_path = path / 'headers.parquet'
            ensemble_path = path / 'ensemble_index.parquet'

            if not zarr_path.exists():
                info.error = "traces.zarr not found"
                return info

            # Load Zarr dimensions
            z = zarr.open(str(zarr_path), mode='r')
            info.n_samples, info.n_traces = z.shape

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                info.sample_rate_ms = meta.get('sample_rate', 0)
                info.duration_ms = meta.get('duration_ms', info.n_samples * info.sample_rate_ms)

                # Check for sorting info
                if 'sorting' in meta:
                    sort_info = meta['sorting']
                    info.is_sorted = sort_info.get('enabled', False)
                    info.sort_key = sort_info.get('sort_key')
                    info.sort_ascending = sort_info.get('ascending', True)

                # Check header mapping for ensemble keys
                if 'header_mapping' in meta:
                    mapping = meta['header_mapping']
                    ensemble_keys = mapping.get('ensemble_keys', [])
                    if ensemble_keys:
                        info.ensemble_key = ensemble_keys[0]

            # Load ensemble info
            if ensemble_path.exists():
                ensemble_df = pd.read_parquet(ensemble_path)
                info.n_ensembles = len(ensemble_df)

            # Check headers for sort order if not in metadata
            if not info.is_sorted and headers_path.exists():
                # Quick check - read first few rows to detect sorting
                headers_df = pd.read_parquet(headers_path, columns=['offset'] if 'offset' in pd.read_parquet(headers_path, columns=[]).columns else None)
                if headers_df is not None and len(headers_df) > 100:
                    # Check if offset appears sorted within first ensemble
                    first_100 = headers_df.iloc[:100]
                    if 'offset' in first_100.columns:
                        offsets = first_100['offset'].values
                        is_asc = all(offsets[i] <= offsets[i+1] for i in range(min(50, len(offsets)-1)))
                        is_desc = all(offsets[i] >= offsets[i+1] for i in range(min(50, len(offsets)-1)))
                        if is_asc or is_desc:
                            info.is_sorted = True
                            info.sort_key = 'offset'
                            info.sort_ascending = is_asc

            info.valid = True

        except Exception as e:
            info.error = str(e)

        return info


@dataclass
class ExportHeaderField:
    """Header field mapping for export."""
    parquet_column: str      # Column name in headers.parquet
    segy_byte_pos: int       # SEG-Y byte position (1-based)
    format: str              # 'i' for int32, 'h' for int16
    enabled: bool = True     # Whether to export this field


@dataclass
class ExportOptions:
    """Export configuration returned by the dialog."""
    # Data sources
    input_path: str
    processed_path: str
    headers_path: str

    # Export type
    export_type: str  # 'processed' or 'noise'

    # Mute settings
    mute_enabled: bool
    mute_velocity: float
    mute_top: bool
    mute_bottom: bool
    mute_taper: int
    mute_target: str  # 'output', 'input', or 'processed'

    # Header mapping for export
    header_mapping: Dict[str, ExportHeaderField]  # parquet_col -> ExportHeaderField

    # Validation
    shapes_match: bool
    sorting_compatible: bool


class ExportOptionsDialog(QDialog):
    """
    Comprehensive export options dialog.

    Allows user to:
    - View and change input/processed dataset paths
    - See dataset statistics and sorting information
    - Check compatibility for noise export
    - Select export type (Processed vs Noise)
    - Configure mute options
    """

    # Standard SEG-Y header byte positions (name -> (byte_pos, format, description))
    SEGY_HEADER_DEFS = {
        'TRACE_SEQUENCE_LINE': (1, 'i', 'Trace sequence number within line'),
        'TRACE_SEQUENCE_FILE': (5, 'i', 'Trace sequence number within file'),
        'FieldRecord': (9, 'i', 'Original field record number'),
        'TraceNumber': (13, 'i', 'Trace number within field record'),
        'EnergySourcePoint': (17, 'i', 'Energy source point number'),
        'CDP': (21, 'i', 'CDP ensemble number'),
        'TRACE_NUMBER_CDP': (25, 'i', 'Trace number within CDP'),
        'TraceIdentificationCode': (29, 'h', 'Trace identification code'),
        'offset': (37, 'i', 'Distance from source to receiver'),
        'ReceiverGroupElevation': (41, 'i', 'Receiver group elevation'),
        'SourceSurfaceElevation': (45, 'i', 'Surface elevation at source'),
        'SourceDepth': (49, 'i', 'Source depth below surface'),
        'SourceX': (73, 'i', 'Source coordinate X'),
        'SourceY': (77, 'i', 'Source coordinate Y'),
        'GroupX': (81, 'i', 'Receiver coordinate X'),
        'GroupY': (85, 'i', 'Receiver coordinate Y'),
        'CDP_X': (181, 'i', 'CDP X coordinate'),
        'CDP_Y': (185, 'i', 'CDP Y coordinate'),
        'INLINE_3D': (189, 'i', 'Inline number (3D)'),
        'CROSSLINE_3D': (193, 'i', 'Crossline number (3D)'),
        'SAMPLE_COUNT': (115, 'h', 'Number of samples in trace'),
        'SAMPLE_INTERVAL': (117, 'h', 'Sample interval (microseconds)'),
    }

    def __init__(
        self,
        input_path: Optional[Path] = None,
        processed_path: Optional[Path] = None,
        mute_config: Optional[Any] = None,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("SEG-Y Export Options")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        # Store dataset info
        self.input_info: Optional[DatasetInfo] = None
        self.processed_info: Optional[DatasetInfo] = None
        self.mute_config = mute_config

        # Header mapping storage
        self.header_mapping: Dict[str, ExportHeaderField] = {}
        self.available_columns: list = []

        self._init_ui()

        # Load initial datasets if provided
        if input_path:
            self._load_dataset(input_path, is_input=True)
        if processed_path:
            self._load_dataset(processed_path, is_input=False)

        self._update_compatibility()

    def _init_ui(self):
        """Initialize the dialog UI with tabs."""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Tab 1: Datasets
        self.tab_widget.addTab(self._create_datasets_tab(), "1. Datasets")

        # Tab 2: Headers
        self.tab_widget.addTab(self._create_headers_tab(), "2. Headers")

        # Tab 3: Export Options
        self.tab_widget.addTab(self._create_export_options_tab(), "3. Export Options")

        layout.addWidget(self.tab_widget)

        # Dialog Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _create_datasets_tab(self) -> QWidget:
        """Create the datasets selection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # INPUT Dataset Section
        input_group = QGroupBox("INPUT Dataset (Original Data)")
        input_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        input_layout = QVBoxLayout()

        input_path_layout = QHBoxLayout()
        self.input_path_label = QLabel("No dataset selected")
        self.input_path_label.setStyleSheet("color: #666;")
        input_path_layout.addWidget(self.input_path_label, stretch=1)

        self.input_browse_btn = QPushButton("Browse...")
        self.input_browse_btn.setMaximumWidth(100)
        self.input_browse_btn.clicked.connect(lambda: self._browse_dataset(is_input=True))
        input_path_layout.addWidget(self.input_browse_btn)
        input_layout.addLayout(input_path_layout)

        self.input_stats_frame = self._create_stats_frame()
        input_layout.addWidget(self.input_stats_frame)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # PROCESSED Dataset Section
        processed_group = QGroupBox("PROCESSED Dataset (Denoised Output)")
        processed_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        processed_layout = QVBoxLayout()

        processed_path_layout = QHBoxLayout()
        self.processed_path_label = QLabel("No dataset selected")
        self.processed_path_label.setStyleSheet("color: #666;")
        processed_path_layout.addWidget(self.processed_path_label, stretch=1)

        self.processed_browse_btn = QPushButton("Browse...")
        self.processed_browse_btn.setMaximumWidth(100)
        self.processed_browse_btn.clicked.connect(lambda: self._browse_dataset(is_input=False))
        processed_path_layout.addWidget(self.processed_browse_btn)
        processed_layout.addLayout(processed_path_layout)

        self.processed_stats_frame = self._create_stats_frame(is_processed=True)
        processed_layout.addWidget(self.processed_stats_frame)
        processed_group.setLayout(processed_layout)
        layout.addWidget(processed_group)

        # Compatibility Status
        self.compat_frame = QFrame()
        self.compat_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.compat_frame.setStyleSheet("QFrame { background: #f5f5f5; border-radius: 4px; padding: 5px; }")
        compat_layout = QVBoxLayout()

        self.compat_label = QLabel("Select both datasets to check compatibility")
        self.compat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compat_layout.addWidget(self.compat_label)

        self.sort_warning_label = QLabel("")
        self.sort_warning_label.setStyleSheet("color: #ff6600;")
        self.sort_warning_label.setWordWrap(True)
        self.sort_warning_label.hide()
        compat_layout.addWidget(self.sort_warning_label)

        self.compat_frame.setLayout(compat_layout)
        layout.addWidget(self.compat_frame)

        layout.addStretch()
        return tab

    def _create_headers_tab(self) -> QWidget:
        """Create the header mapping tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Instructions
        info_label = QLabel(
            "Configure which parquet columns map to SEG-Y trace header byte positions.\n"
            "Use 'Auto-populate' to detect columns, or load a saved configuration."
        )
        info_label.setStyleSheet("color: #666;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Button bar
        btn_layout = QHBoxLayout()

        self.auto_populate_btn = QPushButton("Auto-populate")
        self.auto_populate_btn.setToolTip("Detect columns from headers.parquet and match to SEG-Y fields")
        self.auto_populate_btn.clicked.connect(self._auto_populate_headers)
        btn_layout.addWidget(self.auto_populate_btn)

        self.add_header_btn = QPushButton("Add Row")
        self.add_header_btn.clicked.connect(self._add_header_row)
        btn_layout.addWidget(self.add_header_btn)

        self.remove_header_btn = QPushButton("Remove Row")
        self.remove_header_btn.clicked.connect(self._remove_header_row)
        btn_layout.addWidget(self.remove_header_btn)

        btn_layout.addStretch()

        self.load_config_btn = QPushButton("Load...")
        self.load_config_btn.clicked.connect(self._load_header_config)
        btn_layout.addWidget(self.load_config_btn)

        self.save_config_btn = QPushButton("Save...")
        self.save_config_btn.clicked.connect(self._save_header_config)
        btn_layout.addWidget(self.save_config_btn)

        layout.addLayout(btn_layout)

        # Header mapping table
        self.header_table = QTableWidget()
        self.header_table.setColumnCount(5)
        self.header_table.setHorizontalHeaderLabels([
            "Enable", "Parquet Column", "SEG-Y Field", "Byte Position", "Format"
        ])

        # Set column widths
        header = self.header_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.header_table.setColumnWidth(0, 50)
        self.header_table.setColumnWidth(3, 80)
        self.header_table.setColumnWidth(4, 60)

        layout.addWidget(self.header_table)

        # Available columns info
        self.columns_info_label = QLabel("Available columns: (select a dataset first)")
        self.columns_info_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.columns_info_label.setWordWrap(True)
        layout.addWidget(self.columns_info_label)

        return tab

    def _create_export_options_tab(self) -> QWidget:
        """Create the export options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Export Type Selection
        export_type_group = QGroupBox("Export Type")
        export_type_layout = QVBoxLayout()

        self.export_type_group = QButtonGroup()

        self.processed_radio = QRadioButton("Processed Data - Export denoised/filtered output")
        self.processed_radio.setChecked(True)
        self.export_type_group.addButton(self.processed_radio, 0)
        export_type_layout.addWidget(self.processed_radio)

        self.noise_radio = QRadioButton("Noise = Input - Processed - Export removed signal")
        self.export_type_group.addButton(self.noise_radio, 1)
        export_type_layout.addWidget(self.noise_radio)

        self.noise_warning_label = QLabel("")
        self.noise_warning_label.setStyleSheet("color: red; margin-left: 20px;")
        self.noise_warning_label.hide()
        export_type_layout.addWidget(self.noise_warning_label)

        export_type_group.setLayout(export_type_layout)
        layout.addWidget(export_type_group)

        # Mute Options
        mute_group = QGroupBox("Mute Options (Applied During Export)")
        mute_layout = QGridLayout()

        self.mute_top_cb = QCheckBox("Top Mute")
        self.mute_top_cb.setToolTip("Zero samples before mute time: T = |offset| / velocity")
        mute_layout.addWidget(self.mute_top_cb, 0, 0)

        self.mute_bottom_cb = QCheckBox("Bottom Mute")
        self.mute_bottom_cb.setToolTip("Zero samples after mute time: T = |offset| / velocity")
        mute_layout.addWidget(self.mute_bottom_cb, 0, 1)

        mute_layout.addWidget(QLabel("Apply mute to:"), 1, 0)
        self.mute_target_combo = QComboBox()
        self.mute_target_combo.addItems([
            "Output (after subtraction)",
            "Input (before subtraction)",
            "Processed (before subtraction)"
        ])
        self.mute_target_combo.setToolTip(
            "Output: Mute applied to final result\n"
            "Input: Mute applied to Input before subtraction\n"
            "Processed: Mute applied to Processed before subtraction"
        )
        mute_layout.addWidget(self.mute_target_combo, 1, 1)

        mute_layout.addWidget(QLabel("Velocity (m/s):"), 2, 0)
        self.mute_velocity_spin = QDoubleSpinBox()
        self.mute_velocity_spin.setRange(500, 8000)
        self.mute_velocity_spin.setValue(2500)
        self.mute_velocity_spin.setSingleStep(100)
        self.mute_velocity_spin.setDecimals(0)
        mute_layout.addWidget(self.mute_velocity_spin, 2, 1)

        mute_layout.addWidget(QLabel("Taper (samples):"), 3, 0)
        self.mute_taper_spin = QSpinBox()
        self.mute_taper_spin.setRange(0, 100)
        self.mute_taper_spin.setValue(20)
        self.mute_taper_spin.setToolTip("Cosine taper length at mute boundary")
        mute_layout.addWidget(self.mute_taper_spin, 3, 1)

        mute_info = QLabel("Linear mute formula: T_mute = |offset| / velocity")
        mute_info.setStyleSheet("color: #666; font-size: 9pt;")
        mute_layout.addWidget(mute_info, 4, 0, 1, 2)

        mute_group.setLayout(mute_layout)
        layout.addWidget(mute_group)

        # Pre-fill mute from config if provided
        if self.mute_config:
            self.mute_top_cb.setChecked(self.mute_config.top_mute)
            self.mute_bottom_cb.setChecked(self.mute_config.bottom_mute)
            self.mute_velocity_spin.setValue(self.mute_config.velocity)
            self.mute_taper_spin.setValue(self.mute_config.taper_samples)

        layout.addStretch()
        return tab

    def _add_header_row(self, parquet_col: str = "", segy_field: str = "",
                        byte_pos: int = 0, fmt: str = "i", enabled: bool = True):
        """Add a row to the header mapping table."""
        row = self.header_table.rowCount()
        self.header_table.insertRow(row)

        # Enable checkbox
        enable_cb = QCheckBox()
        enable_cb.setChecked(enabled)
        self.header_table.setCellWidget(row, 0, enable_cb)

        # Parquet column combo
        parquet_combo = QComboBox()
        parquet_combo.setEditable(True)
        parquet_combo.addItems(self.available_columns)
        if parquet_col:
            parquet_combo.setCurrentText(parquet_col)
        self.header_table.setCellWidget(row, 1, parquet_combo)

        # SEG-Y field combo
        segy_combo = QComboBox()
        segy_combo.setEditable(True)
        segy_fields = list(self.SEGY_HEADER_DEFS.keys())
        segy_combo.addItems(segy_fields)
        if segy_field:
            segy_combo.setCurrentText(segy_field)
        segy_combo.currentTextChanged.connect(lambda text: self._on_segy_field_changed(row, text))
        self.header_table.setCellWidget(row, 2, segy_combo)

        # Byte position
        byte_item = QTableWidgetItem(str(byte_pos) if byte_pos > 0 else "")
        self.header_table.setItem(row, 3, byte_item)

        # Format combo
        fmt_combo = QComboBox()
        fmt_combo.addItems(["i", "h"])  # i=int32, h=int16
        fmt_combo.setCurrentText(fmt)
        self.header_table.setCellWidget(row, 4, fmt_combo)

    def _on_segy_field_changed(self, row: int, field_name: str):
        """Update byte position and format when SEG-Y field is selected."""
        if field_name in self.SEGY_HEADER_DEFS:
            byte_pos, fmt, _ = self.SEGY_HEADER_DEFS[field_name]
            self.header_table.item(row, 3).setText(str(byte_pos))
            fmt_combo = self.header_table.cellWidget(row, 4)
            if fmt_combo:
                fmt_combo.setCurrentText(fmt)

    def _remove_header_row(self):
        """Remove the selected row from the header table."""
        current_row = self.header_table.currentRow()
        if current_row >= 0:
            self.header_table.removeRow(current_row)

    def _auto_populate_headers(self):
        """Auto-populate header mapping from available columns."""
        if not self.available_columns:
            QMessageBox.warning(
                self, "No Columns",
                "Please select a dataset first to detect available columns."
            )
            return

        # Clear existing rows
        self.header_table.setRowCount(0)

        # Common mappings: parquet column pattern -> SEG-Y field
        common_mappings = {
            'TRACE_SEQUENCE_LINE': ['trace_sequence_line', 'TRACE_SEQUENCE_LINE', 'seq_line'],
            'TRACE_SEQUENCE_FILE': ['trace_sequence_file', 'TRACE_SEQUENCE_FILE', 'seq_file'],
            'FieldRecord': ['field_record', 'FieldRecord', 'FIELD_RECORD', 'ffid', 'FFID'],
            'TraceNumber': ['trace_number', 'TraceNumber', 'TRACE_NUMBER', 'traceno'],
            'EnergySourcePoint': ['energy_source_point', 'EnergySourcePoint', 'source_point', 'SP'],
            'CDP': ['cdp', 'CDP', 'cmp', 'CMP'],
            'TRACE_NUMBER_CDP': ['trace_number_cdp', 'TRACE_NUMBER_CDP', 'cdp_trace'],
            'offset': ['offset', 'OFFSET', 'Offset'],
            'SourceX': ['source_x', 'SourceX', 'SOURCE_X', 'sx', 'SX'],
            'SourceY': ['source_y', 'SourceY', 'SOURCE_Y', 'sy', 'SY'],
            'GroupX': ['receiver_x', 'GroupX', 'GROUP_X', 'gx', 'GX', 'rec_x'],
            'GroupY': ['receiver_y', 'GroupY', 'GROUP_Y', 'gy', 'GY', 'rec_y'],
            'CDP_X': ['cdp_x', 'CDP_X', 'cmp_x'],
            'CDP_Y': ['cdp_y', 'CDP_Y', 'cmp_y'],
            'INLINE_3D': ['inline', 'INLINE', 'inline_no', 'INLINE_NO', 'INLINE_3D', 'il'],
            'CROSSLINE_3D': ['crossline', 'CROSSLINE', 'xline', 'XLINE', 'CROSSLINE_3D', 'xl'],
        }

        # Find matching columns
        for segy_field, patterns in common_mappings.items():
            for pattern in patterns:
                # Case-insensitive match
                for col in self.available_columns:
                    if col.lower() == pattern.lower():
                        byte_pos, fmt, _ = self.SEGY_HEADER_DEFS[segy_field]
                        self._add_header_row(col, segy_field, byte_pos, fmt, True)
                        break
                else:
                    continue
                break

    def _load_header_config(self):
        """Load header configuration from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Header Configuration",
            "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)

            self.header_table.setRowCount(0)
            for item in config.get('header_mapping', []):
                self._add_header_row(
                    item.get('parquet_column', ''),
                    item.get('segy_field', ''),
                    item.get('byte_position', 0),
                    item.get('format', 'i'),
                    item.get('enabled', True)
                )
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load configuration:\n{e}")

    def _save_header_config(self):
        """Save header configuration to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Header Configuration",
            "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            config = {'header_mapping': []}
            for row in range(self.header_table.rowCount()):
                enable_cb = self.header_table.cellWidget(row, 0)
                parquet_combo = self.header_table.cellWidget(row, 1)
                segy_combo = self.header_table.cellWidget(row, 2)
                byte_item = self.header_table.item(row, 3)
                fmt_combo = self.header_table.cellWidget(row, 4)

                config['header_mapping'].append({
                    'enabled': enable_cb.isChecked() if enable_cb else True,
                    'parquet_column': parquet_combo.currentText() if parquet_combo else '',
                    'segy_field': segy_combo.currentText() if segy_combo else '',
                    'byte_position': int(byte_item.text()) if byte_item and byte_item.text() else 0,
                    'format': fmt_combo.currentText() if fmt_combo else 'i'
                })

            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)

            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save configuration:\n{e}")

    def _get_header_mapping_from_table(self) -> Dict[str, ExportHeaderField]:
        """Extract header mapping from the table."""
        mapping = {}
        for row in range(self.header_table.rowCount()):
            enable_cb = self.header_table.cellWidget(row, 0)
            parquet_combo = self.header_table.cellWidget(row, 1)
            byte_item = self.header_table.item(row, 3)
            fmt_combo = self.header_table.cellWidget(row, 4)

            if not enable_cb or not enable_cb.isChecked():
                continue

            parquet_col = parquet_combo.currentText() if parquet_combo else ''
            byte_pos = int(byte_item.text()) if byte_item and byte_item.text() else 0
            fmt = fmt_combo.currentText() if fmt_combo else 'i'

            if parquet_col and byte_pos > 0:
                mapping[parquet_col] = ExportHeaderField(
                    parquet_column=parquet_col,
                    segy_byte_pos=byte_pos,
                    format=fmt,
                    enabled=True
                )
        return mapping

    def _create_stats_frame(self, is_processed: bool = False) -> QFrame:
        """Create a statistics display frame."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { background: #fafafa; border-radius: 4px; }")

        # Store the prefix as a property for later retrieval
        frame.setProperty("dataset_type", "processed" if is_processed else "input")

        layout = QGridLayout()
        layout.setContentsMargins(10, 5, 10, 5)

        # Create labels
        prefix = "processed" if is_processed else "input"

        # Row 1: Traces and Samples
        layout.addWidget(QLabel("Traces:"), 0, 0)
        traces_label = QLabel("-")
        traces_label.setObjectName(f"{prefix}_traces")
        layout.addWidget(traces_label, 0, 1)

        layout.addWidget(QLabel("Samples:"), 0, 2)
        samples_label = QLabel("-")
        samples_label.setObjectName(f"{prefix}_samples")
        layout.addWidget(samples_label, 0, 3)

        # Row 2: Duration and Sample Rate
        layout.addWidget(QLabel("Duration:"), 1, 0)
        duration_label = QLabel("-")
        duration_label.setObjectName(f"{prefix}_duration")
        layout.addWidget(duration_label, 1, 1)

        layout.addWidget(QLabel("Sample Rate:"), 1, 2)
        rate_label = QLabel("-")
        rate_label.setObjectName(f"{prefix}_rate")
        layout.addWidget(rate_label, 1, 3)

        # Row 3: Ensembles and Sorting
        layout.addWidget(QLabel("Ensembles:"), 2, 0)
        ensembles_label = QLabel("-")
        ensembles_label.setObjectName(f"{prefix}_ensembles")
        layout.addWidget(ensembles_label, 2, 1)

        layout.addWidget(QLabel("Sorting:"), 2, 2)
        sorting_label = QLabel("-")
        sorting_label.setObjectName(f"{prefix}_sorting")
        layout.addWidget(sorting_label, 2, 3)

        frame.setLayout(layout)
        return frame

    def _update_stats_frame(self, frame: QFrame, info: DatasetInfo):
        """Update stats frame with dataset info."""
        # Use stored property to determine prefix (not findChild which is unreliable)
        prefix = frame.property("dataset_type") or "input"

        def set_label(name: str, value: str):
            label = frame.findChild(QLabel, f"{prefix}_{name}")
            if label:
                label.setText(value)

        if info.valid:
            set_label("traces", f"{info.n_traces:,}")
            set_label("samples", str(info.n_samples))
            set_label("duration", f"{info.duration_ms:.0f} ms" if info.duration_ms else "-")
            set_label("rate", f"{info.sample_rate_ms:.2f} ms" if info.sample_rate_ms else "-")
            set_label("ensembles", f"{info.n_ensembles:,}" if info.n_ensembles else "-")

            if info.is_sorted:
                sort_dir = "ASC" if info.sort_ascending else "DESC"
                set_label("sorting", f"{info.sort_key} ({sort_dir})")
            else:
                set_label("sorting", "None")
        else:
            for name in ["traces", "samples", "duration", "rate", "ensembles", "sorting"]:
                set_label(name, "-")

    def _browse_dataset(self, is_input: bool):
        """Open file browser to select a dataset directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            f"Select {'Input' if is_input else 'Processed'} Dataset Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if dir_path:
            self._load_dataset(Path(dir_path), is_input)

    def _load_dataset(self, path: Path, is_input: bool):
        """Load and validate a dataset from path."""
        info = DatasetInfo.from_path(path)

        if is_input:
            self.input_info = info
            self.input_path_label.setText(str(path))
            self.input_path_label.setStyleSheet("" if info.valid else "color: red;")
            self._update_stats_frame(self.input_stats_frame, info)
        else:
            self.processed_info = info
            self.processed_path_label.setText(str(path))
            self.processed_path_label.setStyleSheet("" if info.valid else "color: red;")
            self._update_stats_frame(self.processed_stats_frame, info)

        # Load available columns from headers.parquet
        if info.valid:
            self._load_available_columns(path)

        if not info.valid:
            QMessageBox.warning(
                self,
                "Dataset Error",
                f"Could not load dataset:\n{path}\n\nError: {info.error}"
            )

        self._update_compatibility()

    def _load_available_columns(self, path: Path):
        """Load available column names from headers.parquet."""
        headers_path = path / 'headers.parquet'
        if headers_path.exists():
            try:
                # Read just the schema to get column names
                df = pd.read_parquet(headers_path, columns=[])
                self.available_columns = list(df.columns) if hasattr(df, 'columns') else []

                # Actually read columns - parquet needs at least one column
                schema_df = pd.read_parquet(headers_path)
                self.available_columns = list(schema_df.columns)

                # Update the columns info label
                if hasattr(self, 'columns_info_label'):
                    col_text = ", ".join(self.available_columns[:15])
                    if len(self.available_columns) > 15:
                        col_text += f"... ({len(self.available_columns)} total)"
                    self.columns_info_label.setText(f"Available columns: {col_text}")

                # Update combo boxes in existing rows
                for row in range(self.header_table.rowCount()):
                    parquet_combo = self.header_table.cellWidget(row, 1)
                    if parquet_combo:
                        current_text = parquet_combo.currentText()
                        parquet_combo.clear()
                        parquet_combo.addItems(self.available_columns)
                        parquet_combo.setCurrentText(current_text)

            except Exception as e:
                self.available_columns = []
                if hasattr(self, 'columns_info_label'):
                    self.columns_info_label.setText(f"Error reading columns: {e}")

    def _update_compatibility(self):
        """Update compatibility status based on loaded datasets."""
        if not self.input_info or not self.processed_info:
            self.compat_label.setText("Select both datasets to check compatibility")
            self.compat_label.setStyleSheet("color: #666;")
            self.sort_warning_label.hide()
            self.noise_radio.setEnabled(False)
            self.noise_warning_label.setText("Select both datasets first")
            self.noise_warning_label.show()
            return

        if not self.input_info.valid or not self.processed_info.valid:
            self.compat_label.setText("One or more datasets failed to load")
            self.compat_label.setStyleSheet("color: red;")
            self.noise_radio.setEnabled(False)
            return

        # Check shape compatibility
        shapes_match = (
            self.input_info.n_traces == self.processed_info.n_traces and
            self.input_info.n_samples == self.processed_info.n_samples
        )

        # Check sorting compatibility
        sorting_compatible = True
        sort_warning = ""

        if self.input_info.is_sorted != self.processed_info.is_sorted:
            sorting_compatible = False
            sort_warning = "WARNING: One dataset is sorted, the other is not!"
        elif self.input_info.is_sorted and self.processed_info.is_sorted:
            if self.input_info.sort_key != self.processed_info.sort_key:
                sorting_compatible = False
                sort_warning = f"WARNING: Different sort keys ({self.input_info.sort_key} vs {self.processed_info.sort_key})"
            elif self.input_info.sort_ascending != self.processed_info.sort_ascending:
                sorting_compatible = False
                sort_warning = "WARNING: Different sort directions (ASC vs DESC)"

        # Update UI
        if shapes_match:
            self.compat_label.setText(
                f"COMPATIBLE - Both datasets: {self.input_info.n_traces:,} traces x {self.input_info.n_samples} samples"
            )
            self.compat_label.setStyleSheet("color: green; font-weight: bold;")
            self.noise_radio.setEnabled(True)
            self.noise_warning_label.hide()
        else:
            self.compat_label.setText(
                f"INCOMPATIBLE - Shape mismatch!\n"
                f"Input: {self.input_info.n_traces:,} x {self.input_info.n_samples} | "
                f"Processed: {self.processed_info.n_traces:,} x {self.processed_info.n_samples}"
            )
            self.compat_label.setStyleSheet("color: red; font-weight: bold;")
            self.noise_radio.setEnabled(False)
            self.noise_warning_label.setText("Shape mismatch - noise export unavailable")
            self.noise_warning_label.show()
            if self.noise_radio.isChecked():
                self.processed_radio.setChecked(True)

        # Show sorting warning
        if sort_warning:
            self.sort_warning_label.setText(sort_warning)
            self.sort_warning_label.show()
        else:
            self.sort_warning_label.hide()

    def _on_accept(self):
        """Validate and accept dialog."""
        if not self.processed_info or not self.processed_info.valid:
            QMessageBox.warning(
                self,
                "No Dataset",
                "Please select a valid processed dataset to export."
            )
            return

        # If noise export selected, validate input too
        if self.noise_radio.isChecked():
            if not self.input_info or not self.input_info.valid:
                QMessageBox.warning(
                    self,
                    "No Input Dataset",
                    "Noise export requires a valid input dataset."
                )
                return

            # Check shapes
            if (self.input_info.n_traces != self.processed_info.n_traces or
                self.input_info.n_samples != self.processed_info.n_samples):
                QMessageBox.warning(
                    self,
                    "Shape Mismatch",
                    "Cannot export noise: Input and Processed datasets have different shapes."
                )
                return

        self.accept()

    def get_options(self) -> Optional[ExportOptions]:
        """Get the configured export options."""
        if not self.processed_info:
            return None

        # Determine headers path
        headers_path = self.processed_info.path / 'headers.parquet'
        if not headers_path.exists() and self.input_info:
            headers_path = self.input_info.path / 'headers.parquet'

        mute_enabled = self.mute_top_cb.isChecked() or self.mute_bottom_cb.isChecked()

        # Map combo box index to mute target string
        mute_target_map = {0: 'output', 1: 'input', 2: 'processed'}
        mute_target = mute_target_map.get(self.mute_target_combo.currentIndex(), 'output')

        # Get header mapping from table
        header_mapping = self._get_header_mapping_from_table()

        return ExportOptions(
            input_path=str(self.input_info.path) if self.input_info else "",
            processed_path=str(self.processed_info.path),
            headers_path=str(headers_path),
            export_type='noise' if self.noise_radio.isChecked() else 'processed',
            mute_enabled=mute_enabled,
            mute_velocity=self.mute_velocity_spin.value() if mute_enabled else 0,
            mute_top=self.mute_top_cb.isChecked(),
            mute_bottom=self.mute_bottom_cb.isChecked(),
            mute_taper=self.mute_taper_spin.value(),
            mute_target=mute_target,
            header_mapping=header_mapping,
            shapes_match=(
                self.input_info and self.processed_info and
                self.input_info.n_traces == self.processed_info.n_traces and
                self.input_info.n_samples == self.processed_info.n_samples
            ),
            sorting_compatible=True  # User has been warned
        )

    def get_dataset_info(self) -> Tuple[Optional[DatasetInfo], Optional[DatasetInfo]]:
        """Get the loaded dataset info objects."""
        return self.input_info, self.processed_info
