"""
SEG-Y import dialog with header mapping configuration.
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QTableWidget, QTableWidgetItem,
                              QFileDialog, QGroupBox, QLineEdit, QComboBox,
                              QTextEdit, QMessageBox, QSpinBox, QCheckBox,
                              QHeaderView, QProgressDialog, QTabWidget, QScrollArea,
                              QWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import Optional, List
import sys
from utils.segy_import.header_mapping import HeaderMapping, HeaderField, StandardHeaders
from utils.segy_import.computed_headers import ComputedHeaderField
from utils.segy_import.segy_reader import SEGYReader
from utils.segy_import.data_storage import DataStorage
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SEG-Y Import")
        self.resize(900, 700)

        self.segy_file = None
        self.header_mapping = HeaderMapping()
        self.reader = None

        self._init_ui()
        self._load_standard_headers()

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()

        # File selection
        layout.addWidget(self._create_file_selection_group())

        # Header mapping table
        layout.addWidget(self._create_header_mapping_group())

        # Computed headers configuration
        layout.addWidget(self._create_computed_headers_group())

        # Ensemble configuration
        layout.addWidget(self._create_ensemble_group())

        # Preview area
        layout.addWidget(self._create_preview_group())

        # Action buttons
        layout.addLayout(self._create_action_buttons())

        self.setLayout(layout)

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

    def _create_header_mapping_group(self) -> QGroupBox:
        """Create header mapping configuration table."""
        group = QGroupBox("Trace Header Mapping Configuration")
        layout = QVBoxLayout()

        # Toolbar
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

        # Table
        self.header_table = QTableWidget()
        self.header_table.setColumnCount(5)
        self.header_table.setHorizontalHeaderLabels([
            'Header Name', 'Byte Position', 'Format', 'Description', 'Sample Values (5 traces)'
        ])
        # Set column stretch modes
        header = self.header_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Byte
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Format
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Description
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Sample Values
        self.header_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.header_table)

        group.setLayout(layout)
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

    def _create_ensemble_group(self) -> QGroupBox:
        """Create ensemble configuration group."""
        group = QGroupBox("Import Configuration")
        layout = QVBoxLayout()

        # Ensemble keys row
        ensemble_row = QHBoxLayout()
        ensemble_row.addWidget(QLabel("Ensemble Keys (comma-separated):"))

        self.ensemble_keys_edit = QLineEdit()
        self.ensemble_keys_edit.setPlaceholderText("e.g., cdp or inline,crossline")
        ensemble_row.addWidget(self.ensemble_keys_edit)

        ensemble_row.addWidget(QLabel("(Headers that define ensemble boundaries)"))
        layout.addLayout(ensemble_row)

        # Spatial units row
        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Spatial Units:"))

        self.spatial_units_combo = QComboBox()
        self.spatial_units_combo.addItem("Meters (m)", AppSettings.METERS)
        self.spatial_units_combo.addItem("Feet (ft)", AppSettings.FEET)

        # Set current value from settings
        current_units = get_settings().get_spatial_units()
        index = 0 if current_units == AppSettings.METERS else 1
        self.spatial_units_combo.setCurrentIndex(index)

        self.spatial_units_combo.setToolTip(
            "Select spatial units for coordinates and distances.\n"
            "This will be applied throughout the entire application."
        )
        units_row.addWidget(self.spatial_units_combo)

        units_row.addWidget(QLabel("(Used for coordinates, offsets, and distances)"))
        units_row.addStretch()
        layout.addLayout(units_row)

        group.setLayout(layout)
        return group

    def _create_preview_group(self) -> QGroupBox:
        """Create header preview group."""
        group = QGroupBox("Header Preview (First 10 Traces)")
        layout = QVBoxLayout()

        preview_btn = QPushButton("Preview Headers")
        preview_btn.clicked.connect(self._preview_headers)
        layout.addWidget(preview_btn)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        layout.addWidget(self.preview_text)

        group.setLayout(layout)
        return group

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
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select SEG-Y File",
            "",
            "SEG-Y Files (*.sgy *.segy *.SGY *.SEGY);;All Files (*)"
        )

        if filename:
            self.segy_file = filename
            self.file_path_edit.setText(filename)
            self._update_reader()
            self.import_btn.setEnabled(True)

            # Load sample header values if table has headers
            if self.header_table.rowCount() > 0:
                self._load_sample_header_values()

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
        if not self.reader:
            QMessageBox.warning(self, "No File", "Please select a SEG-Y file first.")
            return

        try:
            # Update mapping and ensemble keys
            self._update_reader()

            ensemble_keys_text = self.ensemble_keys_edit.text().strip()
            ensemble_keys = []
            if ensemble_keys_text:
                ensemble_keys = [k.strip() for k in ensemble_keys_text.split(',')]
                self.header_mapping.set_ensemble_keys(ensemble_keys)

            # Save spatial units selection to app settings
            selected_units = self.spatial_units_combo.currentData()
            get_settings().set_spatial_units(selected_units)
            print(f"Spatial units set to: {selected_units}")

            # Ask for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Output Directory for Zarr/Parquet Storage",
                ""
            )

            if not output_dir:
                return

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
        """Import using streaming method (for large files)."""
        import time
        start_time = time.time()

        n_traces = file_info['n_traces']
        n_samples = file_info['n_samples']

        # Create progress dialog
        progress = QProgressDialog("Streaming import in progress...", "Cancel", 0, n_traces, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        storage = DataStorage(output_dir)

        try:
            # OPTIMIZED SINGLE-PASS IMPORT
            # Reads the SEG-Y file ONCE and processes everything simultaneously
            progress.setLabelText(f"Streaming {n_traces:,} traces (single-pass optimization)...")
            trace_generator = self.reader.read_traces_in_chunks(chunk_size=5000)

            def import_progress(current, total, phase):
                if progress.wasCanceled():
                    raise InterruptedError("Import cancelled by user")
                progress.setValue(current)
                progress.setLabelText(
                    f"Processing: {current:,}/{total:,} traces\n"
                    f"(Writing traces, headers, detecting ensembles simultaneously)"
                )

            # Single-pass import: traces + headers + ensembles all at once!
            import_stats = storage.save_all_streaming(
                trace_generator,
                n_samples=n_samples,
                n_traces=n_traces,
                ensemble_keys=ensemble_keys if ensemble_keys else None,
                chunk_size=5000,
                header_batch_size=10000,
                progress_callback=import_progress
            )

            if progress.wasCanceled():
                raise InterruptedError("Import cancelled by user")

            # Extract statistics from single-pass import
            compression_ratio = import_stats['compression_ratio']
            n_ensembles = import_stats['n_ensembles']
            total_headers = import_stats['total_headers']

            # Phase 4: Save metadata
            progress.setLabelText("Saving metadata...")
            metadata = {
                'shape': [n_samples, n_traces],
                'sample_rate': file_info['sample_interval'],
                'n_samples': n_samples,
                'n_traces': n_traces,
                'duration_ms': (n_samples - 1) * file_info['sample_interval'],
                'nyquist_freq': 1000.0 / (2.0 * file_info['sample_interval']),
                'seismic_metadata': {
                    'source_file': str(self.segy_file),
                    'original_segy_path': str(self.segy_file),  # For export functionality
                    'file_info': file_info,
                    'header_mapping': self.header_mapping.to_dict(),
                },
                'storage_info': {
                    'zarr_chunks': f"({n_samples}, 1000)",
                    'parquet_compression': 'snappy',
                    'zarr_compression': 'zstd'
                }
            }

            import json
            with open(storage.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Also save trace index
            import pandas as pd
            import numpy as np
            df_index = pd.DataFrame({
                'trace_index': np.arange(n_traces),
                'global_trace_id': np.arange(n_traces)
            })
            df_index.to_parquet(
                storage.trace_index_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

            progress.setValue(n_traces)

            elapsed_time = time.time() - start_time

            # Print computed header errors if any
            error_summary = self.reader.get_computed_header_errors()
            if error_summary:
                print(error_summary)

            # Show success statistics
            stats_text = f"""
Streaming Import Successful!

Data Statistics:
- Traces: {n_traces:,}
- Samples: {n_samples}
- Ensembles: {n_ensembles if ensemble_keys else 'Not configured'}

Performance:
- Import time: {elapsed_time:.1f}s
- Throughput: {n_traces / elapsed_time:.0f} traces/sec
- Compression ratio: {compression_ratio:.2f}x

Output: {output_dir}
            """

            QMessageBox.information(self, "Import Complete", stats_text)

            # Load back for viewer using lazy loading for memory efficiency
            from models.lazy_seismic_data import LazySeismicData

            progress.setLabelText("Preparing data for viewing...")
            lazy_data = LazySeismicData.from_storage_dir(str(storage.output_dir))
            ensembles_df = storage.get_ensemble_index()

            # Emit signal with lazy data and file path
            self.import_completed.emit(lazy_data, None, ensembles_df, self.segy_file)

            self.accept()

        except InterruptedError as e:
            # User cancelled - clean up
            import shutil
            if storage.output_dir.exists():
                shutil.rmtree(storage.output_dir, ignore_errors=True)
            QMessageBox.information(self, "Import Cancelled", "Import was cancelled by user.")
        except Exception as e:
            # Error - clean up
            import shutil
            if storage.output_dir.exists():
                shutil.rmtree(storage.output_dir, ignore_errors=True)
            raise

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
