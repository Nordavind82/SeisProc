"""
PSTM Configuration Wizard

Multi-page wizard for configuring Kirchhoff Pre-Stack Time Migration jobs.
Follows the FKK designer dialog pattern from SeisProc.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QRadioButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar,
    QHeaderView, QMessageBox, QButtonGroup
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class PSTMWizard(QWizard):
    """
    Multi-page PSTM configuration wizard.

    Pages:
    1. Job Setup - name, input file, output directory, template
    2. Velocity Model - constant, gradient, or from file
    3. Header Mapping - map SEG-Y headers to migration fields
    4. Output Grid - time/inline/crossline dimensions
    5. Binning Configuration - offset/azimuth bins
    6. Advanced Options - traveltime mode, aperture, antialiasing
    7. Review & Launch - summary, validation, launch
    """

    job_configured = pyqtSignal(dict)  # Emits configuration dictionary

    def __init__(self, parent=None, initial_file: str = None):
        super().__init__(parent)
        self.setWindowTitle("Kirchhoff PSTM Wizard")
        self.resize(900, 700)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        self.initial_file = initial_file
        self._config = self._create_default_config()

        if initial_file:
            self._config['input_file'] = initial_file

        # Add pages
        self.job_setup_page = JobSetupPage(self)
        self.velocity_page = VelocityModelPage(self)
        self.header_page = HeaderMappingPage(self)
        self.input_grid_page = InputGridPage(self)  # Scan and detect input grid
        self.image_output_page = ImageOutputPage(self)  # User-defined output grid
        self.binning_page = BinningConfigPage(self)
        self.advanced_page = AdvancedOptionsPage(self)
        self.review_page = ReviewLaunchPage(self)

        # Keep output_grid_page as alias for backward compatibility
        self.output_grid_page = self.image_output_page

        self.addPage(self.job_setup_page)
        self.addPage(self.velocity_page)
        self.addPage(self.header_page)
        self.addPage(self.input_grid_page)
        self.addPage(self.image_output_page)
        self.addPage(self.binning_page)
        self.addPage(self.advanced_page)
        self.addPage(self.review_page)

        self.finished.connect(self._on_finished)

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration dictionary."""
        return {
            'name': 'New Migration Job',
            'input_file': '',
            'input_type': 'zarr',  # 'zarr' or 'segy'
            'output_directory': '',
            'template': 'custom',
            # Velocity
            'velocity_type': 'constant',
            'velocity_v0': 2500.0,
            'velocity_gradient': 0.0,
            'velocity_file': '',
            # Output grid - time axis
            'time_min_ms': 0,
            'time_max_ms': 4000,
            'dt_ms': 4.0,
            # Output grid - corner points (3 corners define rectangular grid)
            'corner1_il': 1, 'corner1_xl': 1, 'corner1_x': 0.0, 'corner1_y': 0.0,  # Origin (IL min, XL min)
            'corner2_il': 100, 'corner2_xl': 1, 'corner2_x': 2475.0, 'corner2_y': 0.0,  # IL max, XL min
            'corner3_il': 1, 'corner3_xl': 100, 'corner3_x': 0.0, 'corner3_y': 2475.0,  # IL min, XL max
            # Computed corner 4 (IL max, XL max)
            'corner4_il': 100, 'corner4_xl': 100, 'corner4_x': 2475.0, 'corner4_y': 2475.0,
            # Bin sizes
            'bin_size_il': 25.0,
            'bin_size_xl': 25.0,
            # Legacy fields for backward compatibility
            'inline_min': 1,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100,
            'xline_step': 1,
            # Grid origin (computed from corner1)
            'x_origin': 0.0,
            'y_origin': 0.0,
            # Binning
            'binning_preset': 'full_stack',
            'binning_table': [],
            # Advanced
            'traveltime_mode': 'straight',
            'max_aperture_m': 3000.0,
            'max_angle_deg': 60.0,
            'antialias_enabled': True,
            'checkpoint_enabled': True,
            'use_gpu': True,
            'n_workers': 4,
            # Performance tuning
            'use_time_domain': False,
            'tile_size': 100,
            'use_kdtree': False,
            'use_time_dependent_aperture': True,
            'sample_batch_size': 200,
            # Header mapping
            'header_mapping': {},
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def set_config(self, config: Dict[str, Any]):
        """Set configuration (e.g., when loading from file)."""
        self._config.update(config)

    def _on_finished(self, result):
        """Handle wizard completion."""
        if result == QWizard.DialogCode.Accepted:
            # Collect final configuration from all pages
            self._collect_all_config()
            self.job_configured.emit(self._config)

    def _collect_all_config(self):
        """Collect configuration from all pages."""
        # Job Setup
        self._config['name'] = self.job_setup_page.name_edit.text()
        self._config['input_file'] = self.job_setup_page.input_edit.text()
        self._config['output_directory'] = self.job_setup_page.output_edit.text()

        # Velocity
        if self.velocity_page.const_radio.isChecked():
            self._config['velocity_type'] = 'constant'
        elif self.velocity_page.gradient_radio.isChecked():
            self._config['velocity_type'] = 'gradient'
        else:
            self._config['velocity_type'] = 'file'
        self._config['velocity_v0'] = self.velocity_page.v0_spin.value()
        self._config['velocity_gradient'] = self.velocity_page.gradient_spin.value()
        self._config['velocity_file'] = self.velocity_page.vel_file_edit.text()

        # Input Grid (from scan results)
        scan_results = self.input_grid_page.get_scan_results()
        if scan_results:
            # Store scanned input grid geometry
            self._config['input_il_min'] = scan_results['il_min']
            self._config['input_il_max'] = scan_results['il_max']
            self._config['input_xl_min'] = scan_results['xl_min']
            self._config['input_xl_max'] = scan_results['xl_max']
            self._config['input_bin_il'] = scan_results['bin_il']
            self._config['input_bin_xl'] = scan_results['bin_xl']
            self._config['input_il_extent'] = scan_results['il_distance']
            self._config['input_xl_extent'] = scan_results['xl_distance']
            # Grid origin from corner 1 coordinates
            self._config['x_origin'] = scan_results['c1_x']
            self._config['y_origin'] = scan_results['c1_y']
            # Grid azimuth (inline direction from north)
            self._config['grid_azimuth_deg'] = scan_results.get('grid_azimuth_deg', 0.0)
            # Time axis from input
            self._config['input_dt_ms'] = scan_results.get('dt_ms', 4.0)
            self._config['input_n_samples'] = scan_results.get('n_samples', 1000)
            self._config['input_t_max_ms'] = scan_results.get('t_max_ms', 4000.0)
            # Store all corners for reference
            self._config['corner1_il'] = scan_results['il_min']
            self._config['corner1_xl'] = scan_results['xl_min']
            self._config['corner1_x'] = scan_results['c1_x']
            self._config['corner1_y'] = scan_results['c1_y']
            self._config['corner2_il'] = scan_results['il_max']
            self._config['corner2_xl'] = scan_results['xl_min']
            self._config['corner2_x'] = scan_results['c2_x']
            self._config['corner2_y'] = scan_results['c2_y']
            self._config['corner3_il'] = scan_results['il_min']
            self._config['corner3_xl'] = scan_results['xl_max']
            self._config['corner3_x'] = scan_results['c3_x']
            self._config['corner3_y'] = scan_results['c3_y']
            self._config['corner4_il'] = scan_results['il_max']
            self._config['corner4_xl'] = scan_results['xl_max']
            self._config['corner4_x'] = scan_results['c4_x']
            self._config['corner4_y'] = scan_results['c4_y']

        # Image Output Parameters (user-defined output grid)
        output_params = self.image_output_page.get_output_params()
        self._config['inline_min'] = output_params['il_start']
        self._config['inline_max'] = output_params['il_end']
        self._config['inline_step'] = output_params['il_step']
        self._config['xline_min'] = output_params['xl_start']
        self._config['xline_max'] = output_params['xl_end']
        self._config['xline_step'] = output_params['xl_step']
        self._config['time_min_ms'] = output_params['time_start_ms']
        self._config['time_max_ms'] = output_params['time_end_ms']
        self._config['dt_ms'] = output_params['dt_ms']
        # Output bin sizes (can differ from input)
        self._config['output_bin_il'] = output_params['output_bin_il']
        self._config['output_bin_xl'] = output_params['output_bin_xl']
        # Ensure grid origin and azimuth are set from output params if available
        if 'x_origin' in output_params:
            self._config['x_origin'] = output_params['x_origin']
        if 'y_origin' in output_params:
            self._config['y_origin'] = output_params['y_origin']
        if 'grid_azimuth_deg' in output_params:
            self._config['grid_azimuth_deg'] = output_params['grid_azimuth_deg']

        # Binning
        self._config['binning_preset'] = self.binning_page.preset_combo.currentText()
        self._config['binning_table'] = self.binning_page.get_binning_table()

        # Advanced
        self._config['traveltime_mode'] = self.advanced_page.tt_mode_combo.currentText().lower().replace(' ', '_')
        self._config['max_aperture_m'] = self.advanced_page.aperture_spin.value()
        self._config['max_angle_deg'] = self.advanced_page.angle_spin.value()
        self._config['antialias_enabled'] = self.advanced_page.antialias_check.isChecked()
        self._config['checkpoint_enabled'] = self.advanced_page.checkpoint_check.isChecked()
        self._config['use_gpu'] = self.advanced_page.gpu_check.isChecked()
        self._config['n_workers'] = self.advanced_page.workers_spin.value()

        # Performance tuning (NEW)
        algorithm_text = self.advanced_page.algorithm_combo.currentText()
        self._config['use_time_domain'] = 'Time-Domain' in algorithm_text
        self._config['tile_size'] = self.advanced_page.tile_size_spin.value()
        self._config['use_kdtree'] = self.advanced_page.kdtree_check.isChecked()
        self._config['use_time_dependent_aperture'] = self.advanced_page.time_dep_aperture_check.isChecked()
        self._config['sample_batch_size'] = self.advanced_page.sample_batch_spin.value()

        # Header mapping
        self._config['header_mapping'] = self.header_page.get_mapping()


class JobSetupPage(QWizardPage):
    """Page 1: Basic job configuration."""

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Job Setup")
        self.setSubTitle("Configure basic job parameters and select input/output files")

        layout = QVBoxLayout(self)

        # Job name
        name_group = QGroupBox("Job Information")
        name_layout = QVBoxLayout(name_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Job Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter descriptive job name")
        self.name_edit.setText("New Migration Job")
        name_row.addWidget(self.name_edit)
        name_layout.addLayout(name_row)

        # Template selection
        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Custom",
            "Land 3D",
            "Marine Streamer",
            "Wide Azimuth OVT",
            "Full Stack"
        ])
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        template_row.addWidget(self.template_combo)
        template_row.addStretch()
        name_layout.addLayout(template_row)

        layout.addWidget(name_group)

        # Input file
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select input SEG-Y file or Zarr dataset")
        file_layout.addWidget(self.input_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_input)
        file_layout.addWidget(browse_btn)
        input_layout.addLayout(file_layout)

        # File info preview
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: gray; font-style: italic;")
        input_layout.addWidget(self.file_info_label)

        # Set initial file if provided (from active dataset)
        if wizard.initial_file:
            self.input_edit.setText(wizard.initial_file)
            # Update file info display
            self._update_file_info(wizard.initial_file)

        layout.addWidget(input_group)

        # Output directory
        output_group = QGroupBox("Output Location")
        output_layout = QVBoxLayout(output_group)

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory")
        output_row.addWidget(self.output_edit)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_browse_btn)
        output_layout.addLayout(output_row)

        layout.addWidget(output_group)

        layout.addStretch()

        # Register fields for validation
        # Note: Using asterisk (*) makes fields mandatory - Next is disabled until filled
        # We'll handle validation in isComplete() instead for better UX
        self.registerField("jobName", self.name_edit)
        self.registerField("inputFile", self.input_edit)
        self.registerField("outputDir", self.output_edit)

        # Connect text changes to re-evaluate completion
        self.name_edit.textChanged.connect(self.completeChanged)
        self.input_edit.textChanged.connect(self.completeChanged)
        self.output_edit.textChanged.connect(self.completeChanged)

    def isComplete(self) -> bool:
        """Check if page has minimum required data to proceed."""
        # Job name is required
        if not self.name_edit.text().strip():
            return False
        # Input file OR output dir - at least one should be specified
        # For PSTM on currently loaded data, input file may come from main window
        # Allow proceeding with just job name for flexibility
        return True

    def _browse_input(self):
        """Browse for input file - supports both Zarr directories and SEG-Y files."""
        from PyQt6.QtWidgets import QMenu

        # Show menu to choose between Zarr directory or SEG-Y file
        menu = QMenu(self)
        zarr_action = menu.addAction("Select Zarr Dataset (Recommended)")
        zarr_action.setToolTip("Select a previously imported dataset directory")
        segy_action = menu.addAction("Select SEG-Y File (Direct)")
        segy_action.setToolTip("Select raw SEG-Y file - will need import first")

        action = menu.exec(self.sender().mapToGlobal(self.sender().rect().bottomLeft()))

        if action == zarr_action:
            # Browse for Zarr directory
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Imported Dataset Directory",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            if dir_path:
                self.input_edit.setText(dir_path)
                self._update_file_info(dir_path)
        elif action == segy_action:
            # Browse for SEG-Y file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select SEG-Y File",
                "",
                "SEG-Y Files (*.sgy *.segy *.SGY *.SEGY);;All Files (*)"
            )
            if file_path:
                self.input_edit.setText(file_path)
                self._update_file_info(file_path)

    def _browse_output(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if dir_path:
            self.output_edit.setText(dir_path)

    def _update_file_info(self, file_path: str):
        """Update file info label with file details."""
        path = Path(file_path)
        if not path.exists():
            self.file_info_label.setText("File/directory not found")
            self.file_info_label.setStyleSheet("color: red;")
            return

        # Check if it's a Zarr dataset directory
        if path.is_dir():
            # Look for Zarr markers
            zarr_path = path / "traces.zarr"
            metadata_path = path / "metadata.json"

            if zarr_path.exists() or (path / ".zarray").exists():
                # It's a Zarr dataset
                try:
                    import json
                    info_parts = [f"Zarr Dataset: {path.name}"]

                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        if 'n_traces' in metadata:
                            info_parts.append(f"{metadata['n_traces']:,} traces")
                        if 'n_samples' in metadata:
                            info_parts.append(f"{metadata['n_samples']} samples")

                    self.file_info_label.setText(" | ".join(info_parts))
                    self.file_info_label.setStyleSheet("color: green; font-weight: bold;")

                    # Store dataset type for later use
                    self.wizard_ref._config['input_type'] = 'zarr'
                except Exception as e:
                    self.file_info_label.setText(f"Zarr dataset (error reading: {e})")
                    self.file_info_label.setStyleSheet("color: orange;")
            else:
                self.file_info_label.setText(f"Directory: {path.name} (not a valid Zarr dataset)")
                self.file_info_label.setStyleSheet("color: orange;")
        else:
            # It's a file (SEG-Y)
            size_mb = path.stat().st_size / (1024 * 1024)
            self.file_info_label.setText(f"SEG-Y: {path.name} ({size_mb:.1f} MB)")
            self.file_info_label.setStyleSheet("color: blue;")

            # Store dataset type
            self.wizard_ref._config['input_type'] = 'segy'

    def _on_template_changed(self, index: int):
        """Apply template settings."""
        templates = {
            0: {},  # Custom - no changes
            1: {'binning': 'land_3d', 'traveltime': 'curved'},  # Land 3D
            2: {'binning': 'marine', 'traveltime': 'straight'},  # Marine
            3: {'binning': 'ovt', 'traveltime': 'curved'},  # Wide Azimuth
            4: {'binning': 'full_stack', 'traveltime': 'straight'},  # Full Stack
        }
        # Templates will be applied when pages initialize


class VelocityModelPage(QWizardPage):
    """Page 2: Velocity model configuration."""

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Velocity Model")
        self.setSubTitle("Define the velocity model for migration")

        layout = QVBoxLayout(self)

        # Velocity type selection
        type_group = QGroupBox("Velocity Type")
        type_layout = QVBoxLayout(type_group)

        self.type_button_group = QButtonGroup(self)

        self.const_radio = QRadioButton("Constant Velocity")
        self.const_radio.setChecked(True)
        self.type_button_group.addButton(self.const_radio, 0)
        type_layout.addWidget(self.const_radio)

        self.gradient_radio = QRadioButton("Linear Gradient: v(z) = v₀ + k·z")
        self.type_button_group.addButton(self.gradient_radio, 1)
        type_layout.addWidget(self.gradient_radio)

        self.file_radio = QRadioButton("Load from File")
        self.type_button_group.addButton(self.file_radio, 2)
        type_layout.addWidget(self.file_radio)

        layout.addWidget(type_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)

        # V0
        params_layout.addWidget(QLabel("V₀ (m/s):"), 0, 0)
        self.v0_spin = QDoubleSpinBox()
        self.v0_spin.setRange(500, 8000)
        self.v0_spin.setValue(2500)
        self.v0_spin.setSingleStep(100)
        self.v0_spin.valueChanged.connect(self._update_preview)
        params_layout.addWidget(self.v0_spin, 0, 1)

        # Gradient
        params_layout.addWidget(QLabel("Gradient k (1/s):"), 1, 0)
        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setRange(-2.0, 2.0)
        self.gradient_spin.setValue(0.0)
        self.gradient_spin.setSingleStep(0.1)
        self.gradient_spin.setDecimals(3)
        self.gradient_spin.valueChanged.connect(self._update_preview)
        params_layout.addWidget(self.gradient_spin, 1, 1)

        self.gradient_info = QLabel("(v increases with depth if k > 0)")
        self.gradient_info.setStyleSheet("color: gray; font-style: italic;")
        params_layout.addWidget(self.gradient_info, 1, 2)

        # File path
        params_layout.addWidget(QLabel("Velocity File:"), 2, 0)
        self.vel_file_edit = QLineEdit()
        self.vel_file_edit.setEnabled(False)
        params_layout.addWidget(self.vel_file_edit, 2, 1)
        self.vel_browse_btn = QPushButton("Browse...")
        self.vel_browse_btn.setEnabled(False)
        self.vel_browse_btn.clicked.connect(self._browse_velocity_file)
        params_layout.addWidget(self.vel_browse_btn, 2, 2)

        layout.addWidget(params_group)

        # Preview
        preview_group = QGroupBox("Velocity Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.vel_preview_label = QLabel()
        self.vel_preview_label.setFont(QFont("Courier", 10))
        preview_layout.addWidget(self.vel_preview_label)
        layout.addWidget(preview_group)

        layout.addStretch()

        # Connect radio buttons
        self.const_radio.toggled.connect(self._update_ui_state)
        self.gradient_radio.toggled.connect(self._update_ui_state)
        self.file_radio.toggled.connect(self._update_ui_state)

        self._update_ui_state()
        self._update_preview()

    def _update_ui_state(self):
        """Update UI based on selected velocity type."""
        is_file = self.file_radio.isChecked()
        is_gradient = self.gradient_radio.isChecked()

        self.gradient_spin.setEnabled(is_gradient)
        self.gradient_info.setVisible(is_gradient)
        self.vel_file_edit.setEnabled(is_file)
        self.vel_browse_btn.setEnabled(is_file)

        self._update_preview()

    def _update_preview(self):
        """Update velocity preview text."""
        v0 = self.v0_spin.value()

        if self.const_radio.isChecked():
            text = f"Constant velocity: {v0:.0f} m/s\n\n"
            text += "Velocity profile:\n"
            text += f"  z=0.0s:  v = {v0:.0f} m/s\n"
            text += f"  z=1.0s:  v = {v0:.0f} m/s\n"
            text += f"  z=2.0s:  v = {v0:.0f} m/s\n"
            text += f"  z=3.0s:  v = {v0:.0f} m/s"
        elif self.gradient_radio.isChecked():
            k = self.gradient_spin.value()
            text = f"Linear gradient: v(z) = {v0:.0f} + {k:.3f}·z\n\n"
            text += "Velocity profile:\n"
            for z in [0.0, 1.0, 2.0, 3.0]:
                v = v0 + k * z
                text += f"  z={z:.1f}s:  v = {v:.0f} m/s\n"
        else:
            text = "Velocity from file (not yet loaded)"

        self.vel_preview_label.setText(text)

    def _browse_velocity_file(self):
        """Browse for velocity file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Velocity File",
            "",
            "Velocity Files (*.txt *.vel *.dat);;All Files (*)"
        )
        if file_path:
            self.vel_file_edit.setText(file_path)


class HeaderMappingPage(QWizardPage):
    """Page 3: Map input headers to required migration fields."""

    REQUIRED_HEADERS = [
        ('source_x', 'Source X', True, ['SourceX', 'SX', 'sx', 'source_x']),
        ('source_y', 'Source Y', True, ['SourceY', 'SY', 'sy', 'source_y']),
        ('receiver_x', 'Receiver X', True, ['GroupX', 'GX', 'gx', 'ReceiverX', 'receiver_x']),
        ('receiver_y', 'Receiver Y', True, ['GroupY', 'GY', 'gy', 'ReceiverY', 'receiver_y']),
        ('offset', 'Offset', False, ['offset', 'Offset', 'OFFSET']),
        ('azimuth', 'Azimuth', False, ['azimuth', 'Azimuth', 'AZIMUTH']),
        ('inline', 'Inline', False, ['INLINE_3D', 'inline', 'Inline', 'IL']),
        ('crossline', 'Crossline', False, ['CROSSLINE_3D', 'crossline', 'Crossline', 'XL']),
    ]

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Header Mapping")
        self.setSubTitle("Map SEG-Y trace headers to migration parameters")

        self._available_headers: List[str] = []
        self._header_combos: Dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            "Map the required fields to your input file headers. "
            "Fields marked with (*) are required. "
            "Offset and Azimuth can be computed if not available."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Auto-detect button
        auto_layout = QHBoxLayout()
        auto_btn = QPushButton("Auto-Detect Headers")
        auto_btn.clicked.connect(self._auto_detect)
        auto_layout.addWidget(auto_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        auto_layout.addWidget(clear_btn)

        auto_layout.addStretch()
        layout.addLayout(auto_layout)

        # Mapping table
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(4)
        self.mapping_table.setHorizontalHeaderLabels([
            "Required Field", "Input Header", "Status", "Sample Value"
        ])
        self.mapping_table.setRowCount(len(self.REQUIRED_HEADERS))
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )

        for i, (field_id, label, required, _) in enumerate(self.REQUIRED_HEADERS):
            # Field name
            name_text = f"{label} *" if required else label
            name_item = QTableWidgetItem(name_text)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(i, 0, name_item)

            # Header dropdown
            combo = QComboBox()
            combo.addItem("-- Not Mapped --")
            combo.currentIndexChanged.connect(self._on_mapping_changed)
            self.mapping_table.setCellWidget(i, 1, combo)
            self._header_combos[field_id] = combo

            # Status
            status_item = QTableWidgetItem("Not mapped")
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(i, 2, status_item)

            # Sample value
            sample_item = QTableWidgetItem("-")
            sample_item.setFlags(sample_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mapping_table.setItem(i, 3, sample_item)

        layout.addWidget(self.mapping_table)

        # Validation summary
        self.validation_label = QLabel("Map required headers (*) to continue")
        self.validation_label.setStyleSheet("color: orange;")
        layout.addWidget(self.validation_label)

    def initializePage(self):
        """Called when page is shown - load available headers."""
        input_file = self.wizard_ref.job_setup_page.input_edit.text()
        if input_file:
            self._load_available_headers(input_file)

    def _load_available_headers(self, file_path: str):
        """Load available headers from input file (Zarr or SEG-Y)."""
        path = Path(file_path)
        self._available_headers = []
        self._sample_values = {}  # Store sample values for display

        if path.is_dir():
            # Zarr/Parquet dataset - load from parquet headers
            self._load_headers_from_zarr(path)
        elif path.exists() and path.suffix.lower() in ['.sgy', '.segy']:
            # SEG-Y file - load standard headers
            self._load_headers_from_segy(path)
        else:
            # Fallback to common headers
            self._available_headers = [
                'SourceX', 'SourceY', 'GroupX', 'GroupY',
                'offset', 'azimuth', 'INLINE_3D', 'CROSSLINE_3D',
            ]

        # Update combos
        for combo in self._header_combos.values():
            combo.clear()
            combo.addItem("-- Not Mapped --")
            for header in sorted(self._available_headers):
                combo.addItem(header)

        # Try auto-detect
        self._auto_detect()

        # Update sample values in table
        self._update_sample_values()

    def _load_headers_from_zarr(self, dataset_path: Path):
        """Load available headers from Zarr/Parquet dataset."""
        try:
            # Try to load headers from parquet file
            headers_path = dataset_path / "headers.parquet"
            if headers_path.exists():
                import pyarrow.parquet as pq
                # Read just the schema to get column names
                parquet_file = pq.ParquetFile(headers_path)
                schema = parquet_file.schema_arrow
                self._available_headers = [field.name for field in schema]

                # Read first few rows for sample values
                table = parquet_file.read_row_group(0)
                df = table.to_pandas()
                if len(df) > 0:
                    for col in df.columns:
                        values = df[col].head(3).tolist()
                        self._sample_values[col] = values

                logger.info(f"Loaded {len(self._available_headers)} headers from Zarr dataset")
                return

            # Try metadata.json for header list
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'headers' in metadata:
                    self._available_headers = list(metadata['headers'].keys())
                    return

            logger.warning(f"No headers found in Zarr dataset: {dataset_path}")

        except Exception as e:
            logger.error(f"Error loading headers from Zarr: {e}")
            # Fallback to common headers
            self._available_headers = [
                'SourceX', 'SourceY', 'GroupX', 'GroupY',
                'offset', 'azimuth', 'INLINE_3D', 'CROSSLINE_3D',
            ]

    def _load_headers_from_segy(self, segy_path: Path):
        """Load available headers from SEG-Y file."""
        # Use standard SEG-Y trace header names
        self._available_headers = [
            'TRACE_SEQUENCE_LINE', 'TRACE_SEQUENCE_FILE',
            'FieldRecord', 'TraceNumber', 'EnergySourcePoint',
            'CDP', 'CDP_TRACE', 'TraceIdentificationCode',
            'SourceX', 'SourceY', 'GroupX', 'GroupY',
            'CoordinateUnits', 'SourceGroupScalar',
            'offset', 'ReceiverGroupElevation',
            'SourceSurfaceElevation', 'SourceDepth',
            'INLINE_3D', 'CROSSLINE_3D',
            'ShotPoint', 'ShotPointScalar',
        ]

    def _update_sample_values(self):
        """Update sample values column in mapping table."""
        for i, (field_id, _, _, patterns) in enumerate(self.REQUIRED_HEADERS):
            combo = self._header_combos[field_id]
            selected_header = combo.currentText()

            if selected_header in self._sample_values:
                values = self._sample_values[selected_header]
                sample_text = ", ".join(str(v) for v in values[:3])
                self.mapping_table.item(i, 3).setText(sample_text)
            else:
                self.mapping_table.item(i, 3).setText("-")

    def _auto_detect(self):
        """Auto-detect header mappings based on name patterns."""
        for field_id, _, _, patterns in self.REQUIRED_HEADERS:
            combo = self._header_combos[field_id]

            # Try to find a matching header
            for pattern in patterns:
                if pattern in self._available_headers:
                    idx = combo.findText(pattern)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        break

        self._update_validation()

    def _clear_all(self):
        """Clear all mappings."""
        for combo in self._header_combos.values():
            combo.setCurrentIndex(0)
        self._update_validation()

    def _on_mapping_changed(self):
        """Handle mapping change."""
        self._update_validation()
        self._update_sample_values()

    def _update_validation(self):
        """Update validation status."""
        missing_required = []
        mapped_count = 0

        for i, (field_id, label, required, _) in enumerate(self.REQUIRED_HEADERS):
            combo = self._header_combos[field_id]
            is_mapped = combo.currentIndex() > 0

            # Update status
            status_item = self.mapping_table.item(i, 2)
            if is_mapped:
                status_item.setText("OK")
                status_item.setForeground(Qt.GlobalColor.darkGreen)
                mapped_count += 1
            elif required:
                status_item.setText("Required")
                status_item.setForeground(Qt.GlobalColor.red)
                missing_required.append(label)
            else:
                status_item.setText("Optional")
                status_item.setForeground(Qt.GlobalColor.gray)

        # Update summary
        if missing_required:
            self.validation_label.setText(
                f"Missing required: {', '.join(missing_required)}"
            )
            self.validation_label.setStyleSheet("color: red;")
        else:
            self.validation_label.setText(
                f"All required headers mapped ({mapped_count}/{len(self.REQUIRED_HEADERS)} total)"
            )
            self.validation_label.setStyleSheet("color: green;")

    def get_mapping(self) -> Dict[str, str]:
        """Get current header mapping."""
        mapping = {}
        for field_id, _, _, _ in self.REQUIRED_HEADERS:
            combo = self._header_combos[field_id]
            if combo.currentIndex() > 0:
                mapping[field_id] = combo.currentText()
        return mapping

    def isComplete(self) -> bool:
        """Check if required mappings are complete."""
        for field_id, _, required, _ in self.REQUIRED_HEADERS:
            if required:
                combo = self._header_combos[field_id]
                if combo.currentIndex() == 0:
                    return False
        return True


class InputGridPage(QWizardPage):
    """Page 4: Scan and display input data grid geometry (read-only from scan)."""

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Input Data Grid")
        self.setSubTitle("Scan input data to determine grid geometry and bin sizes")

        self._scan_results = None

        layout = QVBoxLayout(self)

        # Scan button and status
        scan_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Scan Input Data")
        self.scan_btn.setToolTip("Scan input file to detect grid geometry from trace headers")
        self.scan_btn.clicked.connect(self._scan_input_data)
        scan_layout.addWidget(self.scan_btn)

        self.scan_status_label = QLabel("Click 'Scan Input Data' to analyze the input file")
        self.scan_status_label.setStyleSheet("color: gray; font-style: italic;")
        scan_layout.addWidget(self.scan_status_label)
        scan_layout.addStretch()
        layout.addLayout(scan_layout)

        # Grid geometry display (read-only, populated by scan)
        grid_group = QGroupBox("Detected Grid Geometry")
        grid_layout = QGridLayout(grid_group)

        # Headers
        grid_layout.addWidget(QLabel(""), 0, 0)
        grid_layout.addWidget(QLabel("Inline"), 0, 1)
        grid_layout.addWidget(QLabel("Crossline"), 0, 2)
        grid_layout.addWidget(QLabel("X (m)"), 0, 3)
        grid_layout.addWidget(QLabel("Y (m)"), 0, 4)

        # Min corner
        grid_layout.addWidget(QLabel("Min (IL min, XL min):"), 1, 0)
        self.min_il_label = QLabel("-")
        self.min_xl_label = QLabel("-")
        self.min_x_label = QLabel("-")
        self.min_y_label = QLabel("-")
        grid_layout.addWidget(self.min_il_label, 1, 1)
        grid_layout.addWidget(self.min_xl_label, 1, 2)
        grid_layout.addWidget(self.min_x_label, 1, 3)
        grid_layout.addWidget(self.min_y_label, 1, 4)

        # Max corner
        grid_layout.addWidget(QLabel("Max (IL max, XL max):"), 2, 0)
        self.max_il_label = QLabel("-")
        self.max_xl_label = QLabel("-")
        self.max_x_label = QLabel("-")
        self.max_y_label = QLabel("-")
        grid_layout.addWidget(self.max_il_label, 2, 1)
        grid_layout.addWidget(self.max_xl_label, 2, 2)
        grid_layout.addWidget(self.max_x_label, 2, 3)
        grid_layout.addWidget(self.max_y_label, 2, 4)

        layout.addWidget(grid_group)

        # Calculated bin sizes (read-only)
        bin_group = QGroupBox("Calculated Bin Sizes")
        bin_layout = QGridLayout(bin_group)

        bin_layout.addWidget(QLabel("Inline Bin Size:"), 0, 0)
        self.bin_il_label = QLabel("-")
        self.bin_il_label.setStyleSheet("font-weight: bold;")
        bin_layout.addWidget(self.bin_il_label, 0, 1)

        bin_layout.addWidget(QLabel("Crossline Bin Size:"), 0, 2)
        self.bin_xl_label = QLabel("-")
        self.bin_xl_label.setStyleSheet("font-weight: bold;")
        bin_layout.addWidget(self.bin_xl_label, 0, 3)

        layout.addWidget(bin_group)

        # Summary display
        summary_group = QGroupBox("Grid Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("No data scanned yet")
        self.summary_label.setFont(QFont("Courier", 10))
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_group)

        layout.addStretch()

    def initializePage(self):
        """Auto-scan when page is shown if not already scanned."""
        if self._scan_results is None:
            # Auto-trigger scan
            self._scan_input_data()

    def _scan_input_data(self):
        """Scan input data to find grid geometry from actual trace positions."""
        input_file = self.wizard_ref._config.get('input_file', '')
        if not input_file:
            input_file = self.wizard_ref.job_setup_page.input_edit.text()
        if not input_file:
            QMessageBox.warning(self, "No Input", "Please select an input file first.")
            return

        self.scan_status_label.setText("Scanning input data...")
        self.scan_status_label.setStyleSheet("color: blue;")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from pathlib import Path
            import pandas as pd
            import numpy as np
            import json
            import math

            input_path = Path(input_file)
            headers_path = input_path / 'headers.parquet' if input_path.is_dir() else None
            metadata_path = input_path / 'metadata.json' if input_path.is_dir() else None

            if not headers_path or not headers_path.exists():
                QMessageBox.warning(self, "No Headers",
                    "Headers file not found.\nPlease use a Zarr dataset with headers.parquet.")
                self.scan_status_label.setText("Scan failed: no headers file")
                self.scan_status_label.setStyleSheet("color: red;")
                return

            # Read time axis parameters from metadata.json
            dt_ms = 4.0  # default
            n_samples = 1000  # default
            t_max_ms = 4000.0  # default

            if metadata_path and metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    dt_ms = float(metadata.get('sample_rate', 4.0))
                    n_samples = int(metadata.get('n_samples', 1000))
                    t_max_ms = float(metadata.get('duration_ms', (n_samples - 1) * dt_ms))
                    logger.info(f"Read time axis from metadata: dt={dt_ms}ms, n_samples={n_samples}, t_max={t_max_ms}ms")
                except Exception as e:
                    logger.warning(f"Could not read metadata.json: {e}")

            # Read headers
            df = pd.read_parquet(headers_path)
            n_traces = len(df)

            self.scan_status_label.setText(f"Analyzing {n_traces:,} traces...")
            QApplication.processEvents()

            # Find coordinate scalar
            coord_scalar = 1.0
            for scalar_name in ['scalar_coord', 'coordinate_scalar', 'scalco', 'ScalarCoord']:
                if scalar_name in df.columns:
                    scalar_val = df[scalar_name].mode().iloc[0] if len(df[scalar_name].mode()) > 0 else df[scalar_name].iloc[0]
                    if scalar_val < 0:
                        coord_scalar = 1.0 / abs(scalar_val)
                    elif scalar_val > 0:
                        coord_scalar = float(scalar_val)
                    break

            # Find IL/XL columns - NO FALLBACKS, user must have correct mapping
            il_col = xl_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'inline' in col_lower or col_lower == 'il':
                    il_col = col
                elif 'crossline' in col_lower or 'xline' in col_lower or col_lower == 'xl':
                    xl_col = col

            # Find coordinate columns
            src_x_col = src_y_col = rec_x_col = rec_y_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'source_x' in col_lower or col_lower == 'sx':
                    src_x_col = col
                elif 'source_y' in col_lower or col_lower == 'sy':
                    src_y_col = col
                elif 'receiver_x' in col_lower or 'group_x' in col_lower or col_lower in ['gx', 'rx']:
                    rec_x_col = col
                elif 'receiver_y' in col_lower or 'group_y' in col_lower or col_lower in ['gy', 'ry']:
                    rec_y_col = col

            # Check if IL/XL columns exist and have valid data
            if not il_col or not xl_col:
                QMessageBox.warning(self, "Missing Headers",
                    f"Could not find inline/crossline columns in headers.\n\n"
                    f"Inline column: {'found' if il_col else 'NOT FOUND'}\n"
                    f"Crossline column: {'found' if xl_col else 'NOT FOUND'}\n\n"
                    f"Available columns:\n{', '.join(df.columns)}\n\n"
                    f"Please check your header mapping during SEG-Y import.")
                self.scan_status_label.setText("Scan failed: no IL/XL columns")
                self.scan_status_label.setStyleSheet("color: red;")
                return

            # Check if columns have valid data (not all same value)
            il_unique = df[il_col].nunique()
            xl_unique = df[xl_col].nunique()

            if il_unique <= 1:
                il_val = df[il_col].iloc[0] if len(df) > 0 else 'N/A'
                QMessageBox.warning(self, "Invalid Inline Data",
                    f"Inline column '{il_col}' has no valid data!\n\n"
                    f"All {len(df):,} traces have the same value: {il_val}\n\n"
                    f"This usually means:\n"
                    f"1. Wrong byte position in header mapping, or\n"
                    f"2. SEG-Y import bug reading non-standard headers\n\n"
                    f"Please re-import with correct header mapping.")
                self.scan_status_label.setText(f"Scan failed: inline column all same value ({il_val})")
                self.scan_status_label.setStyleSheet("color: red;")
                return

            if xl_unique <= 1:
                xl_val = df[xl_col].iloc[0] if len(df) > 0 else 'N/A'
                QMessageBox.warning(self, "Invalid Crossline Data",
                    f"Crossline column '{xl_col}' has no valid data!\n\n"
                    f"All {len(df):,} traces have the same value: {xl_val}\n\n"
                    f"Please re-import with correct header mapping.")
                self.scan_status_label.setText(f"Scan failed: crossline column all same value ({xl_val})")
                self.scan_status_label.setStyleSheet("color: red;")
                return

            if not src_x_col or not src_y_col:
                QMessageBox.warning(self, "Missing Headers",
                    "Could not find coordinate columns in headers.")
                self.scan_status_label.setText("Scan failed: no coordinate columns")
                self.scan_status_label.setStyleSheet("color: red;")
                return

            # Compute midpoint coordinates
            if rec_x_col and rec_y_col:
                df['_mid_x'] = ((df[src_x_col] + df[rec_x_col]) / 2) * coord_scalar
                df['_mid_y'] = ((df[src_y_col] + df[rec_y_col]) / 2) * coord_scalar
            else:
                df['_mid_x'] = df[src_x_col] * coord_scalar
                df['_mid_y'] = df[src_y_col] * coord_scalar

            # Find traces at IL/XL extremes
            il_min = df[il_col].min()
            il_max = df[il_col].max()
            xl_min = df[xl_col].min()
            xl_max = df[xl_col].max()

            self.scan_status_label.setText("Finding corner coordinates...")
            QApplication.processEvents()

            # Get coordinates at min IL (average across all XL at min IL)
            min_il_traces = df[df[il_col] == il_min]
            min_il_x = min_il_traces['_mid_x'].mean()
            min_il_y = min_il_traces['_mid_y'].mean()

            # Get coordinates at max IL
            max_il_traces = df[df[il_col] == il_max]
            max_il_x = max_il_traces['_mid_x'].mean()
            max_il_y = max_il_traces['_mid_y'].mean()

            # Get coordinates at min XL
            min_xl_traces = df[df[xl_col] == xl_min]
            min_xl_x = min_xl_traces['_mid_x'].mean()
            min_xl_y = min_xl_traces['_mid_y'].mean()

            # Get coordinates at max XL
            max_xl_traces = df[df[xl_col] == xl_max]
            max_xl_x = max_xl_traces['_mid_x'].mean()
            max_xl_y = max_xl_traces['_mid_y'].mean()

            # Find actual corner coordinates (traces near IL min/XL min, etc.)
            # Corner 1: IL min, XL min
            corner1_traces = df[(df[il_col] == il_min) & (df[xl_col] == df[df[il_col] == il_min][xl_col].min())]
            if len(corner1_traces) == 0:
                corner1_traces = df[df[il_col] == il_min].nsmallest(10, xl_col)
            c1_x = corner1_traces['_mid_x'].mean()
            c1_y = corner1_traces['_mid_y'].mean()
            c1_xl = int(corner1_traces[xl_col].mean())

            # Corner 2: IL max, XL min
            corner2_traces = df[(df[il_col] == il_max) & (df[xl_col] == df[df[il_col] == il_max][xl_col].min())]
            if len(corner2_traces) == 0:
                corner2_traces = df[df[il_col] == il_max].nsmallest(10, xl_col)
            c2_x = corner2_traces['_mid_x'].mean()
            c2_y = corner2_traces['_mid_y'].mean()
            c2_xl = int(corner2_traces[xl_col].mean())

            # Corner 3: IL min, XL max
            corner3_traces = df[(df[il_col] == il_min) & (df[xl_col] == df[df[il_col] == il_min][xl_col].max())]
            if len(corner3_traces) == 0:
                corner3_traces = df[df[il_col] == il_min].nlargest(10, xl_col)
            c3_x = corner3_traces['_mid_x'].mean()
            c3_y = corner3_traces['_mid_y'].mean()
            c3_xl = int(corner3_traces[xl_col].mean())

            # Calculate bin sizes from IL/XL differences and coordinate distances
            self.scan_status_label.setText("Calculating bin sizes...")
            QApplication.processEvents()

            # Inline bin size: distance along inline direction / number of inlines
            n_il = int(il_max - il_min)
            n_xl = int(xl_max - xl_min)

            # Distance from min IL to max IL (at constant XL)
            il_distance = np.sqrt((max_il_x - min_il_x)**2 + (max_il_y - min_il_y)**2)
            # Distance from min XL to max XL (at constant IL)
            xl_distance = np.sqrt((max_xl_x - min_xl_x)**2 + (max_xl_y - min_xl_y)**2)

            bin_il = il_distance / n_il if n_il > 0 else 25.0
            bin_xl = xl_distance / n_xl if n_xl > 0 else 25.0

            # Calculate grid azimuth (angle of inline direction from north/Y-axis)
            # Inline direction vector: from corner1 (IL min) to corner2 (IL max) at constant XL
            il_dx = c2_x - c1_x
            il_dy = c2_y - c1_y
            # Azimuth measured clockwise from north (Y-axis positive)
            # atan2(dx, dy) gives angle from Y-axis
            grid_azimuth_rad = math.atan2(il_dx, il_dy)
            grid_azimuth_deg = math.degrees(grid_azimuth_rad)
            # Normalize to 0-360
            if grid_azimuth_deg < 0:
                grid_azimuth_deg += 360.0

            # Store results
            self._scan_results = {
                'n_traces': n_traces,
                'il_col': il_col,  # Column used for inline
                'xl_col': xl_col,  # Column used for crossline
                'il_min': int(il_min), 'il_max': int(il_max), 'n_il': n_il + 1,
                'xl_min': int(xl_min), 'xl_max': int(xl_max), 'n_xl': n_xl + 1,
                'c1_x': c1_x, 'c1_y': c1_y,  # IL min, XL min corner
                'c2_x': c2_x, 'c2_y': c2_y,  # IL max, XL min corner
                'c3_x': c3_x, 'c3_y': c3_y,  # IL min, XL max corner
                'c4_x': c2_x + (c3_x - c1_x), 'c4_y': c2_y + (c3_y - c1_y),  # IL max, XL max (computed)
                'bin_il': bin_il,
                'bin_xl': bin_xl,
                'il_distance': il_distance,
                'xl_distance': xl_distance,
                'coord_scalar': coord_scalar,
                # Time axis parameters from metadata
                'dt_ms': dt_ms,
                'n_samples': n_samples,
                't_max_ms': t_max_ms,
                # Grid azimuth (inline direction from north)
                'grid_azimuth_deg': grid_azimuth_deg,
            }

            # Update display
            self._update_display()

            self.scan_status_label.setText(f"Scan complete: {n_traces:,} traces analyzed")
            self.scan_status_label.setStyleSheet("color: green;")

        except Exception as e:
            import traceback
            logger.error(f"Scan failed: {e}\n{traceback.format_exc()}")
            QMessageBox.warning(self, "Scan Error", f"Failed to scan input data:\n{e}")
            self.scan_status_label.setText(f"Scan failed: {e}")
            self.scan_status_label.setStyleSheet("color: red;")

    def _update_display(self):
        """Update display with scan results."""
        if not self._scan_results:
            return

        r = self._scan_results

        # Update corner labels
        self.min_il_label.setText(str(r['il_min']))
        self.min_xl_label.setText(str(r['xl_min']))
        self.min_x_label.setText(f"{r['c1_x']:.1f}")
        self.min_y_label.setText(f"{r['c1_y']:.1f}")

        self.max_il_label.setText(str(r['il_max']))
        self.max_xl_label.setText(str(r['xl_max']))
        self.max_x_label.setText(f"{r['c4_x']:.1f}")
        self.max_y_label.setText(f"{r['c4_y']:.1f}")

        # Update bin sizes
        self.bin_il_label.setText(f"{r['bin_il']:.2f} m")
        self.bin_xl_label.setText(f"{r['bin_xl']:.2f} m")

        # Update summary
        summary = f"INPUT DATA GRID SUMMARY:\n"
        summary += f"{'='*40}\n"
        summary += f"Total traces:     {r['n_traces']:,}\n"
        summary += f"Inline column:    {r.get('il_col', 'inline')}\n"
        summary += f"Crossline column: {r.get('xl_col', 'crossline')}\n"
        summary += f"Inline range:     {r['il_min']} - {r['il_max']} ({r['n_il']} lines)\n"
        summary += f"Crossline range:  {r['xl_min']} - {r['xl_max']} ({r['n_xl']} lines)\n"
        summary += f"IL extent:        {r['il_distance']:.1f} m\n"
        summary += f"XL extent:        {r['xl_distance']:.1f} m\n"
        summary += f"Inline bin:       {r['bin_il']:.2f} m\n"
        summary += f"Crossline bin:    {r['bin_xl']:.2f} m\n"
        summary += f"Grid azimuth:     {r.get('grid_azimuth_deg', 0):.1f}°\n"
        summary += f"\nTIME AXIS:\n"
        summary += f"Sample rate:      {r.get('dt_ms', 4.0):.2f} ms\n"
        summary += f"Num samples:      {r.get('n_samples', 1000)}\n"
        summary += f"Max time:         {r.get('t_max_ms', 4000):.1f} ms\n"
        if r.get('coord_scalar', 1.0) != 1.0:
            summary += f"Coord scalar:     {r['coord_scalar']:.6f}\n"

        self.summary_label.setText(summary)

    def get_scan_results(self):
        """Return scan results for use by other pages."""
        return self._scan_results

    def isComplete(self) -> bool:
        """Page is complete when scan has been performed."""
        return self._scan_results is not None


class ImageOutputPage(QWizardPage):
    """Page 5: Define output image grid parameters (user-defined, can differ from input grid)."""

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Image Output Parameters")
        self.setSubTitle("Define the output image grid for migration (bin sizes can differ from input)")

        # Store input grid info for reference
        self._input_bin_il = 25.0
        self._input_bin_xl = 25.0
        self._input_il_extent = 0.0
        self._input_xl_extent = 0.0
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._grid_azimuth_deg = 0.0
        self._il_min_input = 1
        self._xl_min_input = 1
        self._updating_from_bin_change = False  # Prevent recursive updates

        layout = QVBoxLayout(self)

        # Output bin sizes group (NEW)
        bin_group = QGroupBox("Output Bin Sizes")
        bin_layout = QGridLayout(bin_group)

        bin_layout.addWidget(QLabel("Inline Bin Size (m):"), 0, 0)
        self.output_bin_il_spin = QDoubleSpinBox()
        self.output_bin_il_spin.setRange(3.125, 200.0)
        self.output_bin_il_spin.setValue(25.0)
        self.output_bin_il_spin.setDecimals(3)
        self.output_bin_il_spin.setSingleStep(6.25)
        self.output_bin_il_spin.valueChanged.connect(self._on_bin_size_changed)
        bin_layout.addWidget(self.output_bin_il_spin, 0, 1)

        bin_layout.addWidget(QLabel("Crossline Bin Size (m):"), 0, 2)
        self.output_bin_xl_spin = QDoubleSpinBox()
        self.output_bin_xl_spin.setRange(3.125, 200.0)
        self.output_bin_xl_spin.setValue(25.0)
        self.output_bin_xl_spin.setDecimals(3)
        self.output_bin_xl_spin.setSingleStep(6.25)
        self.output_bin_xl_spin.valueChanged.connect(self._on_bin_size_changed)
        bin_layout.addWidget(self.output_bin_xl_spin, 0, 3)

        # Use input bin sizes button
        use_input_bins_btn = QPushButton("Use Input Bin Sizes")
        use_input_bins_btn.setToolTip("Set output bin sizes to match input data")
        use_input_bins_btn.clicked.connect(self._use_input_bin_sizes)
        bin_layout.addWidget(use_input_bins_btn, 1, 0, 1, 2)

        # Input bin size info label
        self.input_bin_info_label = QLabel("Input bins: - m x - m")
        self.input_bin_info_label.setStyleSheet("color: gray; font-style: italic;")
        bin_layout.addWidget(self.input_bin_info_label, 1, 2, 1, 2)

        layout.addWidget(bin_group)

        # IL/XL output range
        range_group = QGroupBox("Output Inline/Crossline Range")
        range_layout = QGridLayout(range_group)

        range_layout.addWidget(QLabel("Inline Start:"), 0, 0)
        self.il_start_spin = QSpinBox()
        self.il_start_spin.setRange(1, 1000000)
        self.il_start_spin.setValue(1)
        self.il_start_spin.valueChanged.connect(self._update_preview)
        range_layout.addWidget(self.il_start_spin, 0, 1)

        range_layout.addWidget(QLabel("Inline End:"), 0, 2)
        self.il_end_spin = QSpinBox()
        self.il_end_spin.setRange(1, 1000000)
        self.il_end_spin.setValue(100)
        self.il_end_spin.valueChanged.connect(self._update_preview)
        range_layout.addWidget(self.il_end_spin, 0, 3)

        range_layout.addWidget(QLabel("Inline Step:"), 0, 4)
        self.il_step_spin = QSpinBox()
        self.il_step_spin.setRange(1, 100)
        self.il_step_spin.setValue(1)
        self.il_step_spin.valueChanged.connect(self._update_preview)
        range_layout.addWidget(self.il_step_spin, 0, 5)

        range_layout.addWidget(QLabel("Crossline Start:"), 1, 0)
        self.xl_start_spin = QSpinBox()
        self.xl_start_spin.setRange(1, 1000000)
        self.xl_start_spin.setValue(1)
        self.xl_start_spin.valueChanged.connect(self._update_preview)
        range_layout.addWidget(self.xl_start_spin, 1, 1)

        range_layout.addWidget(QLabel("Crossline End:"), 1, 2)
        self.xl_end_spin = QSpinBox()
        self.xl_end_spin.setRange(1, 1000000)
        self.xl_end_spin.setValue(100)
        self.xl_end_spin.valueChanged.connect(self._update_preview)
        range_layout.addWidget(self.xl_end_spin, 1, 3)

        range_layout.addWidget(QLabel("Crossline Step:"), 1, 4)
        self.xl_step_spin = QSpinBox()
        self.xl_step_spin.setRange(1, 100)
        self.xl_step_spin.setValue(1)
        self.xl_step_spin.valueChanged.connect(self._update_preview)
        range_layout.addWidget(self.xl_step_spin, 1, 5)

        # Use full range button
        full_range_btn = QPushButton("Use Full Input Range")
        full_range_btn.setToolTip("Set IL/XL range to cover full input extent at current output bin size")
        full_range_btn.clicked.connect(self._use_full_range)
        range_layout.addWidget(full_range_btn, 2, 0, 1, 2)

        layout.addWidget(range_group)

        # Time parameters
        time_group = QGroupBox("Time Axis")
        time_layout = QGridLayout(time_group)

        time_layout.addWidget(QLabel("Start Time (ms):"), 0, 0)
        self.time_start_spin = QDoubleSpinBox()
        self.time_start_spin.setRange(0, 20000)
        self.time_start_spin.setValue(0)
        self.time_start_spin.valueChanged.connect(self._update_preview)
        time_layout.addWidget(self.time_start_spin, 0, 1)

        time_layout.addWidget(QLabel("End Time (ms):"), 0, 2)
        self.time_end_spin = QDoubleSpinBox()
        self.time_end_spin.setRange(100, 20000)
        self.time_end_spin.setValue(4000)
        self.time_end_spin.valueChanged.connect(self._update_preview)
        time_layout.addWidget(self.time_end_spin, 0, 3)

        time_layout.addWidget(QLabel("Sample Rate (ms):"), 1, 0)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.5, 16)
        self.dt_spin.setValue(4)
        self.dt_spin.setDecimals(2)
        self.dt_spin.valueChanged.connect(self._update_preview)
        time_layout.addWidget(self.dt_spin, 1, 1)

        # Use input time axis button
        use_input_time_btn = QPushButton("Use Input Time Axis")
        use_input_time_btn.setToolTip("Set time parameters from input data")
        use_input_time_btn.clicked.connect(self._use_input_time_axis)
        time_layout.addWidget(use_input_time_btn, 1, 2, 1, 2)

        layout.addWidget(time_group)

        # Output preview
        preview_group = QGroupBox("Output Grid Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self.preview_label.setFont(QFont("Courier", 10))
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        layout.addStretch()

    def initializePage(self):
        """Initialize from input grid scan results."""
        input_grid_page = self.wizard_ref.input_grid_page
        scan_results = input_grid_page.get_scan_results()

        if scan_results:
            # Store input grid info
            self._input_bin_il = scan_results['bin_il']
            self._input_bin_xl = scan_results['bin_xl']
            self._input_il_extent = scan_results['il_distance']
            self._input_xl_extent = scan_results['xl_distance']
            self._origin_x = scan_results['c1_x']
            self._origin_y = scan_results['c1_y']
            self._grid_azimuth_deg = scan_results.get('grid_azimuth_deg', 0.0)
            self._il_min_input = scan_results['il_min']
            self._xl_min_input = scan_results['xl_min']

            # Update input bin info label
            self.input_bin_info_label.setText(
                f"Input bins: {self._input_bin_il:.2f} m x {self._input_bin_xl:.2f} m"
            )

            # Set output bin sizes to match input (default)
            self.output_bin_il_spin.setValue(self._input_bin_il)
            self.output_bin_xl_spin.setValue(self._input_bin_xl)

            # Set IL/XL ranges from scan (will be recalculated if bin size changes)
            self.il_start_spin.setValue(scan_results['il_min'])
            self.il_end_spin.setValue(scan_results['il_max'])
            self.xl_start_spin.setValue(scan_results['xl_min'])
            self.xl_end_spin.setValue(scan_results['xl_max'])

            # Set time axis from input metadata
            self.time_start_spin.setValue(0)
            self.time_end_spin.setValue(scan_results.get('t_max_ms', 4000))
            self.dt_spin.setValue(scan_results.get('dt_ms', 4.0))

        self._update_preview()

    def _use_input_bin_sizes(self):
        """Set output bin sizes to match input data."""
        self.output_bin_il_spin.setValue(self._input_bin_il)
        self.output_bin_xl_spin.setValue(self._input_bin_xl)

    def _use_input_time_axis(self):
        """Set time parameters from input data."""
        input_grid_page = self.wizard_ref.input_grid_page
        scan_results = input_grid_page.get_scan_results()

        if scan_results:
            self.time_start_spin.setValue(0)
            self.time_end_spin.setValue(scan_results.get('t_max_ms', 4000))
            self.dt_spin.setValue(scan_results.get('dt_ms', 4.0))

    def _on_bin_size_changed(self):
        """Handle output bin size change - recalculate IL/XL range to maintain extent."""
        if self._updating_from_bin_change:
            return

        self._updating_from_bin_change = True
        try:
            output_bin_il = self.output_bin_il_spin.value()
            output_bin_xl = self.output_bin_xl_spin.value()

            # Calculate new IL/XL range to maintain same spatial extent
            if self._input_il_extent > 0 and output_bin_il > 0:
                n_output_il = max(1, int(round(self._input_il_extent / output_bin_il)))
                self.il_start_spin.setValue(1)  # Output starts at 1
                self.il_end_spin.setValue(n_output_il)

            if self._input_xl_extent > 0 and output_bin_xl > 0:
                n_output_xl = max(1, int(round(self._input_xl_extent / output_bin_xl)))
                self.xl_start_spin.setValue(1)  # Output starts at 1
                self.xl_end_spin.setValue(n_output_xl)

            self._update_preview()
        finally:
            self._updating_from_bin_change = False

    def _use_full_range(self):
        """Set output range to cover full input extent at current output bin size."""
        output_bin_il = self.output_bin_il_spin.value()
        output_bin_xl = self.output_bin_xl_spin.value()

        # Calculate IL/XL range to cover full input extent
        if self._input_il_extent > 0 and output_bin_il > 0:
            n_output_il = max(1, int(round(self._input_il_extent / output_bin_il)))
            self.il_start_spin.setValue(1)
            self.il_end_spin.setValue(n_output_il)
            self.il_step_spin.setValue(1)

        if self._input_xl_extent > 0 and output_bin_xl > 0:
            n_output_xl = max(1, int(round(self._input_xl_extent / output_bin_xl)))
            self.xl_start_spin.setValue(1)
            self.xl_end_spin.setValue(n_output_xl)
            self.xl_step_spin.setValue(1)

    def _update_preview(self):
        """Update output grid preview with extent comparison."""
        il_start = self.il_start_spin.value()
        il_end = self.il_end_spin.value()
        il_step = self.il_step_spin.value()
        xl_start = self.xl_start_spin.value()
        xl_end = self.xl_end_spin.value()
        xl_step = self.xl_step_spin.value()
        t_start = self.time_start_spin.value()
        t_end = self.time_end_spin.value()
        dt = self.dt_spin.value()
        output_bin_il = self.output_bin_il_spin.value()
        output_bin_xl = self.output_bin_xl_spin.value()

        n_il = max(1, (il_end - il_start) // il_step + 1)
        n_xl = max(1, (xl_end - xl_start) // xl_step + 1)
        n_time = max(1, int((t_end - t_start) / dt) + 1)

        # Calculate output extent
        output_il_extent = (n_il - 1) * output_bin_il
        output_xl_extent = (n_xl - 1) * output_bin_xl

        # Memory estimate
        bytes_per_volume = n_time * n_il * n_xl * 4  # float32
        memory_mb = bytes_per_volume / (1024 * 1024)
        memory_gb = bytes_per_volume / (1024 * 1024 * 1024)

        text = "OUTPUT IMAGE GRID:\n"
        text += f"{'='*40}\n"
        text += f"Inline:      {il_start} - {il_end} (step {il_step}) = {n_il} lines\n"
        text += f"Crossline:   {xl_start} - {xl_end} (step {xl_step}) = {n_xl} lines\n"
        text += f"Time:        {t_start:.0f} - {t_end:.0f} ms @ {dt:.2f} ms = {n_time} samples\n"
        text += f"\n"
        text += f"Output bin size:   {output_bin_il:.3f} m x {output_bin_xl:.3f} m\n"
        text += f"Output extent:     {output_il_extent/1000:.3f} km x {output_xl_extent/1000:.3f} km\n"
        text += f"Input extent:      {self._input_il_extent/1000:.3f} km x {self._input_xl_extent/1000:.3f} km\n"
        text += f"\n"
        text += f"Total output points: {n_time * n_il * n_xl:,}\n"

        if memory_gb >= 1:
            text += f"Memory per volume:   {memory_gb:.2f} GB\n"
        else:
            text += f"Memory per volume:   {memory_mb:.1f} MB\n"

        # Extent comparison and warnings
        warnings = []
        if self._input_il_extent > 0:
            il_ratio = output_il_extent / self._input_il_extent
            if abs(il_ratio - 1.0) > 0.01:
                if il_ratio < 1.0:
                    warnings.append(f"IL extent {100*(1-il_ratio):.1f}% smaller than input")
                else:
                    warnings.append(f"IL extent {100*(il_ratio-1):.1f}% larger than input")

        if self._input_xl_extent > 0:
            xl_ratio = output_xl_extent / self._input_xl_extent
            if abs(xl_ratio - 1.0) > 0.01:
                if xl_ratio < 1.0:
                    warnings.append(f"XL extent {100*(1-xl_ratio):.1f}% smaller than input")
                else:
                    warnings.append(f"XL extent {100*(xl_ratio-1):.1f}% larger than input")

        if warnings:
            text += f"\nNOTES:\n"
            for w in warnings:
                text += f"  - {w}\n"

        self.preview_label.setText(text)

    def get_output_params(self):
        """Get output parameters dictionary including bin sizes and grid info."""
        return {
            'il_start': self.il_start_spin.value(),
            'il_end': self.il_end_spin.value(),
            'il_step': self.il_step_spin.value(),
            'xl_start': self.xl_start_spin.value(),
            'xl_end': self.xl_end_spin.value(),
            'xl_step': self.xl_step_spin.value(),
            'time_start_ms': self.time_start_spin.value(),
            'time_end_ms': self.time_end_spin.value(),
            'dt_ms': self.dt_spin.value(),
            # Output bin sizes
            'output_bin_il': self.output_bin_il_spin.value(),
            'output_bin_xl': self.output_bin_xl_spin.value(),
            # Grid origin and azimuth (from input, for coordinate calculation)
            'x_origin': self._origin_x,
            'y_origin': self._origin_y,
            'grid_azimuth_deg': self._grid_azimuth_deg,
            # Input bin sizes for reference
            'input_bin_il': self._input_bin_il,
            'input_bin_xl': self._input_bin_xl,
        }


# Keep OutputGridPage as alias for backward compatibility
OutputGridPage = ImageOutputPage


class BinningConfigPage(QWizardPage):
    """Page 6: Configure offset/azimuth binning."""

    PRESETS = {
        'Custom': [],
        'Full Stack': [
            {'name': 'Full', 'offset_min': 0, 'offset_max': 99999, 'az_min': 0, 'az_max': 360}
        ],
        'Marine (6 offsets)': [
            {'name': f'Offset {i+1}', 'offset_min': i*500, 'offset_max': (i+1)*500, 'az_min': 0, 'az_max': 360}
            for i in range(6)
        ],
        'Land 3D (10 offsets)': [
            {'name': f'Offset {i+1}', 'offset_min': i*300, 'offset_max': (i+1)*300, 'az_min': 0, 'az_max': 360}
            for i in range(10)
        ],
        'Wide Azimuth OVT (16 bins)': [
            {'name': f'O{o+1}_A{a+1}', 'offset_min': o*750, 'offset_max': (o+1)*750,
             'az_min': a*90, 'az_max': (a+1)*90}
            for o in range(4) for a in range(4)
        ],
    }

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Binning Configuration")
        self.setSubTitle("Define offset and azimuth bins for migration output")

        layout = QVBoxLayout(self)

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS.keys()))
        self.preset_combo.setCurrentText('Full Stack')
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # Bin table
        self.bin_table = QTableWidget()
        self.bin_table.setColumnCount(6)
        self.bin_table.setHorizontalHeaderLabels([
            "Name", "Offset Min (m)", "Offset Max (m)", "Az Min (°)", "Az Max (°)", "Enabled"
        ])
        self.bin_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.bin_table)

        # Add/remove buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Bin")
        add_btn.clicked.connect(self._add_bin)
        btn_layout.addWidget(add_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_bin)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Summary
        self.bin_summary_label = QLabel()
        layout.addWidget(self.bin_summary_label)

        layout.addStretch()

        # Initialize with default preset
        self._on_preset_changed('Full Stack')

    def _on_preset_changed(self, preset_name: str):
        """Apply preset binning configuration."""
        bins = self.PRESETS.get(preset_name, [])

        self.bin_table.setRowCount(len(bins))

        for i, bin_def in enumerate(bins):
            self._set_bin_row(i, bin_def)

        self._update_summary()

    def _set_bin_row(self, row: int, bin_def: Dict):
        """Set values for a bin table row."""
        # Name
        self.bin_table.setItem(row, 0, QTableWidgetItem(bin_def.get('name', f'Bin {row+1}')))

        # Offset min
        offset_min_spin = QDoubleSpinBox()
        offset_min_spin.setRange(0, 99999)
        offset_min_spin.setValue(bin_def.get('offset_min', 0))
        self.bin_table.setCellWidget(row, 1, offset_min_spin)

        # Offset max
        offset_max_spin = QDoubleSpinBox()
        offset_max_spin.setRange(0, 99999)
        offset_max_spin.setValue(bin_def.get('offset_max', 99999))
        self.bin_table.setCellWidget(row, 2, offset_max_spin)

        # Az min
        az_min_spin = QDoubleSpinBox()
        az_min_spin.setRange(0, 360)
        az_min_spin.setValue(bin_def.get('az_min', 0))
        self.bin_table.setCellWidget(row, 3, az_min_spin)

        # Az max
        az_max_spin = QDoubleSpinBox()
        az_max_spin.setRange(0, 360)
        az_max_spin.setValue(bin_def.get('az_max', 360))
        self.bin_table.setCellWidget(row, 4, az_max_spin)

        # Enabled checkbox
        enabled_check = QCheckBox()
        enabled_check.setChecked(bin_def.get('enabled', True))
        self.bin_table.setCellWidget(row, 5, enabled_check)

    def _add_bin(self):
        """Add a new bin row."""
        row = self.bin_table.rowCount()
        self.bin_table.insertRow(row)
        self._set_bin_row(row, {
            'name': f'Bin {row+1}',
            'offset_min': 0,
            'offset_max': 3000,
            'az_min': 0,
            'az_max': 360,
            'enabled': True
        })
        self.preset_combo.setCurrentText('Custom')
        self._update_summary()

    def _remove_bin(self):
        """Remove selected bin row."""
        current_row = self.bin_table.currentRow()
        if current_row >= 0:
            self.bin_table.removeRow(current_row)
            self.preset_combo.setCurrentText('Custom')
            self._update_summary()

    def _update_summary(self):
        """Update bin count summary."""
        count = self.bin_table.rowCount()
        enabled = sum(1 for i in range(count)
                     if self.bin_table.cellWidget(i, 5) and
                     self.bin_table.cellWidget(i, 5).isChecked())
        self.bin_summary_label.setText(f"{enabled} of {count} bins enabled")

    def get_binning_table(self) -> List[Dict]:
        """Get current binning configuration."""
        bins = []
        for i in range(self.bin_table.rowCount()):
            name_item = self.bin_table.item(i, 0)
            offset_min_widget = self.bin_table.cellWidget(i, 1)
            offset_max_widget = self.bin_table.cellWidget(i, 2)
            az_min_widget = self.bin_table.cellWidget(i, 3)
            az_max_widget = self.bin_table.cellWidget(i, 4)
            enabled_widget = self.bin_table.cellWidget(i, 5)

            bins.append({
                'name': name_item.text() if name_item else f'Bin {i+1}',
                'offset_min': offset_min_widget.value() if offset_min_widget else 0,
                'offset_max': offset_max_widget.value() if offset_max_widget else 99999,
                'az_min': az_min_widget.value() if az_min_widget else 0,
                'az_max': az_max_widget.value() if az_max_widget else 360,
                'enabled': enabled_widget.isChecked() if enabled_widget else True,
            })
        return bins


class AdvancedOptionsPage(QWizardPage):
    """Page 6: Advanced migration options."""

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Advanced Options")
        self.setSubTitle("Configure advanced migration parameters")

        layout = QVBoxLayout(self)

        # Traveltime group
        tt_group = QGroupBox("Traveltime Calculation")
        tt_layout = QVBoxLayout(tt_group)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.tt_mode_combo = QComboBox()
        self.tt_mode_combo.addItems(["Straight Ray", "Curved Ray", "VTI Anisotropic"])
        self.tt_mode_combo.setToolTip(
            "Straight Ray: Fast, good for constant velocity\n"
            "Curved Ray: Handles velocity gradients\n"
            "VTI: For anisotropic media (requires eta/delta)"
        )
        mode_layout.addWidget(self.tt_mode_combo)
        mode_layout.addStretch()
        tt_layout.addLayout(mode_layout)

        layout.addWidget(tt_group)

        # Aperture group
        aperture_group = QGroupBox("Aperture Control")
        aperture_layout = QGridLayout(aperture_group)

        aperture_layout.addWidget(QLabel("Max Aperture (m):"), 0, 0)
        self.aperture_spin = QDoubleSpinBox()
        self.aperture_spin.setRange(100, 20000)
        self.aperture_spin.setValue(3000)
        self.aperture_spin.setSingleStep(100)
        self.aperture_spin.setToolTip("Maximum horizontal distance for trace contribution")
        aperture_layout.addWidget(self.aperture_spin, 0, 1)

        aperture_layout.addWidget(QLabel("Max Angle (°):"), 1, 0)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(10, 85)
        self.angle_spin.setValue(60)
        self.angle_spin.setToolTip("Maximum migration angle from vertical")
        aperture_layout.addWidget(self.angle_spin, 1, 1)

        layout.addWidget(aperture_group)

        # Processing group
        proc_group = QGroupBox("Processing Options")
        proc_layout = QVBoxLayout(proc_group)

        self.antialias_check = QCheckBox("Enable Antialiasing")
        self.antialias_check.setChecked(True)
        self.antialias_check.setToolTip("Apply frequency-dependent antialiasing to prevent aliasing artifacts")
        proc_layout.addWidget(self.antialias_check)

        self.checkpoint_check = QCheckBox("Enable Checkpointing")
        self.checkpoint_check.setChecked(True)
        self.checkpoint_check.setToolTip("Save progress periodically to allow job resume")
        proc_layout.addWidget(self.checkpoint_check)

        self.gpu_check = QCheckBox("Use GPU Acceleration")
        self.gpu_check.setChecked(True)
        self.gpu_check.setToolTip("Use CUDA GPU if available (much faster)")
        proc_layout.addWidget(self.gpu_check)

        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Parallel Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(4)
        self.workers_spin.setToolTip("Number of parallel processes for gather processing")
        workers_layout.addWidget(self.workers_spin)
        workers_layout.addStretch()
        proc_layout.addLayout(workers_layout)

        layout.addWidget(proc_group)

        # Performance tuning group (NEW)
        perf_group = QGroupBox("Performance Tuning")
        perf_layout = QGridLayout(perf_group)

        # Algorithm mode
        perf_layout.addWidget(QLabel("Migration Algorithm:"), 0, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Depth-Domain (Standard)",
            "Time-Domain (Fast)",
        ])
        self.algorithm_combo.setCurrentIndex(0)
        self.algorithm_combo.setToolTip(
            "Depth-Domain: Standard Kirchhoff with depth loop.\n"
            "  - Works with any velocity model (constant, gradient, or file)\n"
            "  - Most accurate, but slower\n\n"
            "Time-Domain: Direct time mapping using t_out = sqrt(t_in² + 4h²/v_rms²)\n"
            "  - ~50-100x faster than depth-domain\n"
            "  - For constant velocity: uses exact equation\n"
            "  - For gradient velocity: uses RMS velocity approximation\n"
            "  - For strong gradients, depth-domain may be more accurate"
        )
        perf_layout.addWidget(self.algorithm_combo, 0, 1)

        # Tile size
        perf_layout.addWidget(QLabel("Tile Size:"), 1, 0)
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(25, 1000)
        self.tile_size_spin.setValue(100)
        self.tile_size_spin.setSingleStep(25)
        self.tile_size_spin.setToolTip(
            "Number of output points per GPU tile.\n"
            "Larger = better GPU utilization, but more memory.\n"
            "Try 100-500 for best performance."
        )
        perf_layout.addWidget(self.tile_size_spin, 1, 1)

        # KD-tree option
        self.kdtree_check = QCheckBox("Use KD-tree Spatial Index")
        self.kdtree_check.setChecked(False)
        self.kdtree_check.setToolTip(
            "Build KD-tree for fast aperture queries.\n"
            "Recommended for datasets with >10K traces spread over large area."
        )
        perf_layout.addWidget(self.kdtree_check, 2, 0, 1, 2)

        # Time-dependent aperture
        self.time_dep_aperture_check = QCheckBox("Time-Dependent Aperture")
        self.time_dep_aperture_check.setChecked(True)
        self.time_dep_aperture_check.setToolTip(
            "Aperture grows with depth: aperture(t) = v*t/2*tan(angle)\n"
            "Reduces computation at shallow times (5-20x speedup)"
        )
        perf_layout.addWidget(self.time_dep_aperture_check, 3, 0, 1, 2)

        # Sample batch size (for time-domain mode)
        perf_layout.addWidget(QLabel("Sample Batch Size:"), 4, 0)
        self.sample_batch_spin = QSpinBox()
        self.sample_batch_spin.setRange(50, 500)
        self.sample_batch_spin.setValue(200)
        self.sample_batch_spin.setSingleStep(50)
        self.sample_batch_spin.setToolTip(
            "For time-domain mode: samples processed per batch.\n"
            "Larger = faster but more memory."
        )
        perf_layout.addWidget(self.sample_batch_spin, 4, 1)

        layout.addWidget(perf_group)

        layout.addStretch()


class ReviewLaunchPage(QWizardPage):
    """Page 7: Review configuration and launch job."""

    def __init__(self, wizard: PSTMWizard):
        super().__init__(wizard)
        self.wizard_ref = wizard
        self.setTitle("Review & Launch")
        self.setSubTitle("Review your configuration and start the migration job")

        layout = QVBoxLayout(self)

        # Configuration summary
        summary_group = QGroupBox("Configuration Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier", 9))
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)

        # Resource estimate
        resource_group = QGroupBox("Resource Estimate")
        resource_layout = QVBoxLayout(resource_group)
        self.resource_label = QLabel()
        self.resource_label.setFont(QFont("Courier", 9))
        resource_layout.addWidget(self.resource_label)
        layout.addWidget(resource_group)

        # Validation status
        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(80)
        validation_layout.addWidget(self.validation_text)
        layout.addWidget(validation_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Configuration...")
        save_btn.clicked.connect(self._save_config)
        btn_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Configuration...")
        load_btn.clicked.connect(self._load_config)
        btn_layout.addWidget(load_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def initializePage(self):
        """Called when page is shown - update all summaries."""
        self._update_summary()
        self._update_resource_estimate()
        self._update_validation()

    def _update_summary(self):
        """Update configuration summary."""
        w = self.wizard_ref

        text = "=" * 50 + "\n"
        text += "PSTM JOB CONFIGURATION\n"
        text += "=" * 50 + "\n\n"

        # Job setup
        text += "JOB SETUP\n"
        text += "-" * 30 + "\n"
        text += f"  Name:     {w.job_setup_page.name_edit.text()}\n"
        text += f"  Input:    {Path(w.job_setup_page.input_edit.text()).name if w.job_setup_page.input_edit.text() else 'Not set'}\n"
        text += f"  Output:   {w.job_setup_page.output_edit.text() or 'Not set'}\n\n"

        # Velocity
        text += "VELOCITY MODEL\n"
        text += "-" * 30 + "\n"
        if w.velocity_page.const_radio.isChecked():
            text += f"  Type:     Constant\n"
            text += f"  V0:       {w.velocity_page.v0_spin.value():.0f} m/s\n"
        elif w.velocity_page.gradient_radio.isChecked():
            text += f"  Type:     Linear Gradient\n"
            text += f"  V0:       {w.velocity_page.v0_spin.value():.0f} m/s\n"
            text += f"  Gradient: {w.velocity_page.gradient_spin.value():.3f} 1/s\n"
        else:
            text += f"  Type:     From File\n"
            text += f"  File:     {w.velocity_page.vel_file_edit.text()}\n"
        text += "\n"

        # Output grid - use image_output_page which has the actual widgets
        text += "OUTPUT GRID\n"
        text += "-" * 30 + "\n"
        img_page = w.image_output_page
        t_start = img_page.time_start_spin.value()
        t_end = img_page.time_end_spin.value()
        dt = img_page.dt_spin.value()
        il_start = img_page.il_start_spin.value()
        il_end = img_page.il_end_spin.value()
        il_step = img_page.il_step_spin.value()
        xl_start = img_page.xl_start_spin.value()
        xl_end = img_page.xl_end_spin.value()
        xl_step = img_page.xl_step_spin.value()
        n_time = int((t_end - t_start) / dt) + 1
        n_il = (il_end - il_start) // il_step + 1
        n_xl = (xl_end - xl_start) // xl_step + 1
        text += f"  Time:     {t_start:.0f} - {t_end:.0f} ms @ {dt:.1f} ms\n"
        text += f"  Inlines:  {il_start} - {il_end} (step {il_step})\n"
        text += f"  Xlines:   {xl_start} - {xl_end} (step {xl_step})\n"
        text += f"  Grid:     {n_time} x {n_il} x {n_xl}\n\n"

        # Binning
        text += "BINNING\n"
        text += "-" * 30 + "\n"
        text += f"  Preset:   {w.binning_page.preset_combo.currentText()}\n"
        bins = w.binning_page.get_binning_table()
        enabled_bins = [b for b in bins if b.get('enabled', True)]
        text += f"  Bins:     {len(enabled_bins)} enabled\n\n"

        # Advanced
        text += "ADVANCED OPTIONS\n"
        text += "-" * 30 + "\n"
        text += f"  Traveltime:    {w.advanced_page.tt_mode_combo.currentText()}\n"
        text += f"  Max Aperture:  {w.advanced_page.aperture_spin.value():.0f} m\n"
        text += f"  Max Angle:     {w.advanced_page.angle_spin.value():.0f}°\n"
        text += f"  Antialiasing:  {'Yes' if w.advanced_page.antialias_check.isChecked() else 'No'}\n"
        text += f"  Checkpointing: {'Yes' if w.advanced_page.checkpoint_check.isChecked() else 'No'}\n"
        text += f"  GPU:           {'Yes' if w.advanced_page.gpu_check.isChecked() else 'No'}\n"
        text += f"  Workers:       {w.advanced_page.workers_spin.value()}\n"

        self.summary_text.setText(text)

    def _update_resource_estimate(self):
        """Update resource estimate."""
        w = self.wizard_ref
        img_page = w.image_output_page

        # Calculate grid size
        t_start = img_page.time_start_spin.value()
        t_end = img_page.time_end_spin.value()
        dt = img_page.dt_spin.value()
        n_time = int((t_end - t_start) / dt) + 1
        n_il = (img_page.il_end_spin.value() - img_page.il_start_spin.value()) // img_page.il_step_spin.value() + 1
        n_xl = (img_page.xl_end_spin.value() - img_page.xl_start_spin.value()) // img_page.xl_step_spin.value() + 1

        bins = w.binning_page.get_binning_table()
        n_bins = len([b for b in bins if b.get('enabled', True)])

        # Memory estimate (float32 per volume + fold)
        bytes_per_volume = n_time * n_il * n_xl * 4
        total_memory = bytes_per_volume * n_bins * 2  # image + fold
        memory_gb = total_memory / (1024**3)

        # Output size estimate
        output_size_gb = bytes_per_volume * n_bins / (1024**3)

        text = f"Memory Required:    ~{memory_gb:.1f} GB\n"
        text += f"Output Size:        ~{output_size_gb:.1f} GB ({n_bins} volumes)\n"
        text += f"Estimated Time:     Depends on input size and hardware\n"

        self.resource_label.setText(text)

    def _update_validation(self):
        """Update validation status."""
        w = self.wizard_ref
        errors = []
        warnings = []

        # Check required fields
        if not w.job_setup_page.input_edit.text():
            errors.append("Input file not specified")
        elif not Path(w.job_setup_page.input_edit.text()).exists():
            errors.append("Input file does not exist")

        if not w.job_setup_page.output_edit.text():
            errors.append("Output directory not specified")
        elif not Path(w.job_setup_page.output_edit.text()).exists():
            warnings.append("Output directory does not exist (will be created)")

        # Check header mapping
        if not w.header_page.isComplete():
            errors.append("Required header mappings incomplete")

        # Check binning
        bins = w.binning_page.get_binning_table()
        enabled_bins = [b for b in bins if b.get('enabled', True)]
        if len(enabled_bins) == 0:
            errors.append("No bins enabled")

        # Format output
        if errors:
            text = "ERRORS (must fix):\n"
            for e in errors:
                text += f"  ✗ {e}\n"
        else:
            text = "✓ All required fields are valid\n"

        if warnings:
            text += "\nWARNINGS:\n"
            for w in warnings:
                text += f"  ⚠ {w}\n"

        self.validation_text.setText(text)

        if errors:
            self.validation_text.setStyleSheet("color: red;")
        elif warnings:
            self.validation_text.setStyleSheet("color: orange;")
        else:
            self.validation_text.setStyleSheet("color: green;")

    def _save_config(self):
        """Save configuration to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            import json
            self.wizard_ref._collect_all_config()
            with open(file_path, 'w') as f:
                json.dump(self.wizard_ref._config, f, indent=2)
            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{file_path}")

    def _load_config(self):
        """Load configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            import json
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                self.wizard_ref.set_config(config)
                # TODO: Update all page widgets from config
                QMessageBox.information(self, "Loaded", "Configuration loaded. Review settings on each page.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")

    def isComplete(self) -> bool:
        """Check if ready to finish."""
        w = self.wizard_ref

        # Basic validation
        if not w.job_setup_page.input_edit.text():
            return False
        if not w.job_setup_page.output_edit.text():
            return False
        if not w.header_page.isComplete():
            return False

        bins = w.binning_page.get_binning_table()
        enabled_bins = [b for b in bins if b.get('enabled', True)]
        if len(enabled_bins) == 0:
            return False

        return True
