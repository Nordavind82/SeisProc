"""
QC Stacking Dialog - Configure and execute QC stacking workflow

Multi-tab dialog for:
1. Line Selection - choose inline numbers for QC stacking
2. Velocity Configuration - load and preview velocity model
3. Stacking Parameters - NMO stretch mute, stack method, fold cutoff
4. Output - directory and naming

Usage:
    dialog = QCStackingDialog(dataset_path, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        config = dialog.get_config()
        # Execute stacking with config
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QTabWidget, QWidget, QTextEdit,
    QDialogButtonBox, QMessageBox, QProgressBar, QListWidget,
    QListWidgetItem, QSplitter, QFrame, QFormLayout, QRadioButton,
    QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QCStackingConfig:
    """
    Configuration for QC stacking workflow.

    Attributes:
        dataset_path: Path to source dataset (Zarr/Parquet)
        inline_numbers: List of inline numbers to stack
        velocity_file: Path to velocity file
        velocity_type: 'rms' or 'interval'
        time_unit: 'ms' or 's'
        velocity_unit: 'm/s', 'km/s', or 'ft/s'
        stretch_mute: NMO stretch mute factor
        stack_method: 'mean' or 'median'
        min_fold: Minimum fold cutoff
        output_dir: Output directory
        output_name: Output dataset name prefix

        # Header mapping for inline/xline
        inline_header: Header field name for inline number
        xline_header: Header field name for crossline number

        # Grid geometry for velocity interpolation
        inline_min: Minimum inline number in output grid
        inline_max: Maximum inline number in output grid
        xline_min: Minimum xline number in output grid
        xline_max: Maximum xline number in output grid
        dx: Inline bin size (spacing)
        dy: Xline bin size (spacing)
        interpolate_velocity: Whether to interpolate velocity to output grid
    """
    dataset_path: str = ""
    inline_numbers: List[int] = field(default_factory=list)
    velocity_file: str = ""
    velocity_type: str = "rms"
    time_unit: str = "ms"
    velocity_unit: str = "m/s"
    stretch_mute: float = 1.5
    stack_method: str = "mean"
    min_fold: int = 1
    output_dir: str = ""
    output_name: str = "qc_stack"

    # Header mapping
    inline_header: str = "INLINE_NO"
    xline_header: str = "XLINE_NO"

    # Grid geometry
    inline_min: Optional[int] = None
    inline_max: Optional[int] = None
    xline_min: Optional[int] = None
    xline_max: Optional[int] = None
    dx: float = 1.0
    dy: float = 1.0
    interpolate_velocity: bool = True

    # SEG-Y velocity byte mapping (for SEG-Y velocity files)
    use_custom_vel_bytes: bool = False
    vel_inline_byte: int = 189  # INLINE_3D
    vel_xline_byte: int = 193   # CROSSLINE_3D

    # Velocity spatial info (from velocity file preview, used for extrapolation)
    vel_inline_range: Optional[Tuple[int, int]] = None
    vel_xline_range: Optional[Tuple[int, int]] = None
    vel_cdp_range: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_path': self.dataset_path,
            'inline_numbers': self.inline_numbers,
            'velocity_file': self.velocity_file,
            'velocity_type': self.velocity_type,
            'time_unit': self.time_unit,
            'velocity_unit': self.velocity_unit,
            'stretch_mute': self.stretch_mute,
            'stack_method': self.stack_method,
            'min_fold': self.min_fold,
            'output_dir': self.output_dir,
            'output_name': self.output_name,
            'inline_header': self.inline_header,
            'xline_header': self.xline_header,
            'inline_min': self.inline_min,
            'inline_max': self.inline_max,
            'xline_min': self.xline_min,
            'xline_max': self.xline_max,
            'dx': self.dx,
            'dy': self.dy,
            'interpolate_velocity': self.interpolate_velocity,
            'use_custom_vel_bytes': self.use_custom_vel_bytes,
            'vel_inline_byte': self.vel_inline_byte,
            'vel_xline_byte': self.vel_xline_byte,
            'vel_inline_range': self.vel_inline_range,
            'vel_xline_range': self.vel_xline_range,
            'vel_cdp_range': self.vel_cdp_range,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QCStackingConfig':
        return cls(
            dataset_path=d.get('dataset_path', ''),
            inline_numbers=d.get('inline_numbers', []),
            velocity_file=d.get('velocity_file', ''),
            velocity_type=d.get('velocity_type', 'rms'),
            time_unit=d.get('time_unit', 'ms'),
            velocity_unit=d.get('velocity_unit', 'm/s'),
            stretch_mute=d.get('stretch_mute', 1.5),
            stack_method=d.get('stack_method', 'mean'),
            min_fold=d.get('min_fold', 1),
            output_dir=d.get('output_dir', ''),
            output_name=d.get('output_name', 'qc_stack'),
            inline_header=d.get('inline_header', 'INLINE_NO'),
            xline_header=d.get('xline_header', 'XLINE_NO'),
            inline_min=d.get('inline_min'),
            inline_max=d.get('inline_max'),
            xline_min=d.get('xline_min'),
            xline_max=d.get('xline_max'),
            dx=d.get('dx', 1.0),
            dy=d.get('dy', 1.0),
            interpolate_velocity=d.get('interpolate_velocity', True),
            use_custom_vel_bytes=d.get('use_custom_vel_bytes', False),
            vel_inline_byte=d.get('vel_inline_byte', 189),
            vel_xline_byte=d.get('vel_xline_byte', 193),
            vel_inline_range=d.get('vel_inline_range'),
            vel_xline_range=d.get('vel_xline_range'),
            vel_cdp_range=d.get('vel_cdp_range'),
        )

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration. Returns (is_valid, error_message)."""
        if not self.dataset_path:
            return False, "Dataset path not specified"
        if not Path(self.dataset_path).exists():
            return False, f"Dataset not found: {self.dataset_path}"
        if not self.inline_numbers:
            return False, "No inline numbers specified"
        if not self.velocity_file:
            return False, "Velocity file not specified"
        if not Path(self.velocity_file).exists():
            return False, f"Velocity file not found: {self.velocity_file}"
        if not self.output_dir:
            return False, "Output directory not specified"
        return True, ""


class QCStackingDialog(QDialog):
    """
    Dialog for configuring QC stacking workflow.

    Provides tabbed interface for:
    - Line selection
    - Velocity model configuration
    - Stacking parameters
    - Output configuration
    """

    config_ready = pyqtSignal(object)  # Emits QCStackingConfig when ready

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.dataset_path = dataset_path or ""
        self._dataset_info: Optional[Dict[str, Any]] = None
        self._velocity_info: Optional[Any] = None

        self._init_ui()
        self._connect_signals()

        if self.dataset_path:
            self._load_dataset_info()

    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("QC Stacking")
        self.setMinimumSize(700, 550)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self._create_line_selection_tab()
        self._create_velocity_tab()
        self._create_parameters_tab()
        self._create_output_tab()

        # Summary text
        self.summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(80)
        self.summary_text.setStyleSheet("background-color: #f5f5f5;")
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(self.summary_group)

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._update_summary()

    def _create_line_selection_tab(self):
        """Create line selection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Dataset info
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout(dataset_group)

        self.dataset_path_edit = QLineEdit(self.dataset_path)
        self.dataset_path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dataset)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.dataset_path_edit)
        path_layout.addWidget(browse_btn)
        dataset_layout.addRow("Dataset:", path_layout)

        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addRow("Info:", self.dataset_info_label)

        layout.addWidget(dataset_group)

        # Inline selection
        inline_group = QGroupBox("Inline Selection")
        inline_layout = QVBoxLayout(inline_group)

        # Input field for inlines
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Inlines:"))
        self.inline_edit = QLineEdit()
        self.inline_edit.setPlaceholderText("e.g., 100, 200, 300-350")
        self.inline_edit.setToolTip(
            "Enter inline numbers separated by commas.\n"
            "Use ranges with hyphen: 100-110\n"
            "Example: 100, 200, 300-350, 400"
        )
        input_layout.addWidget(self.inline_edit)

        parse_btn = QPushButton("Parse")
        parse_btn.clicked.connect(self._parse_inlines)
        input_layout.addWidget(parse_btn)

        inline_layout.addLayout(input_layout)

        # Parsed inlines display
        self.parsed_inlines_label = QLabel("Parsed: (none)")
        self.parsed_inlines_label.setStyleSheet("color: #666;")
        inline_layout.addWidget(self.parsed_inlines_label)

        # Quick selection buttons
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("Quick select:"))

        every_10_btn = QPushButton("Every 10th")
        every_10_btn.clicked.connect(lambda: self._quick_select(10))
        quick_layout.addWidget(every_10_btn)

        every_50_btn = QPushButton("Every 50th")
        every_50_btn.clicked.connect(lambda: self._quick_select(50))
        quick_layout.addWidget(every_50_btn)

        every_100_btn = QPushButton("Every 100th")
        every_100_btn.clicked.connect(lambda: self._quick_select(100))
        quick_layout.addWidget(every_100_btn)

        quick_layout.addStretch()
        inline_layout.addLayout(quick_layout)

        # Estimated traces
        self.trace_estimate_label = QLabel("Estimated traces: 0")
        inline_layout.addWidget(self.trace_estimate_label)

        layout.addWidget(inline_group)
        layout.addStretch()

        self.tab_widget.addTab(tab, "1. Inlines")

    def _create_velocity_tab(self):
        """Create velocity configuration tab with header mapping and grid geometry."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ═══════════════════════════════════════════════════════════════════════
        # Active Dataset Display (prominent at top)
        # ═══════════════════════════════════════════════════════════════════════
        dataset_group = QGroupBox("Active Dataset")
        dataset_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        dataset_layout = QFormLayout(dataset_group)

        self.velocity_tab_dataset_label = QLabel("No dataset loaded")
        self.velocity_tab_dataset_label.setWordWrap(True)
        self.velocity_tab_dataset_label.setStyleSheet(
            "QLabel { background-color: #e8f4e8; padding: 5px; border-radius: 3px; }"
        )
        dataset_layout.addRow("Dataset:", self.velocity_tab_dataset_label)

        layout.addWidget(dataset_group)

        # ═══════════════════════════════════════════════════════════════════════
        # Data Header Mapping (populated from dataset headers)
        # ═══════════════════════════════════════════════════════════════════════
        header_group = QGroupBox("Data Header Mapping")
        header_layout = QFormLayout(header_group)

        # Inline header - populated from dataset
        self.inline_header_combo = QComboBox()
        self.inline_header_combo.setToolTip(
            "Header field in seismic data containing inline numbers.\n"
            "Select from available headers in the dataset."
        )
        self.inline_header_combo.setPlaceholderText("Load dataset to see headers")
        header_layout.addRow("Inline Header:", self.inline_header_combo)

        # Xline header - populated from dataset
        self.xline_header_combo = QComboBox()
        self.xline_header_combo.setToolTip(
            "Header field in seismic data containing crossline numbers.\n"
            "Select from available headers in the dataset."
        )
        self.xline_header_combo.setPlaceholderText("Load dataset to see headers")
        header_layout.addRow("Xline Header:", self.xline_header_combo)

        layout.addWidget(header_group)

        # ═══════════════════════════════════════════════════════════════════════
        # SEG-Y Velocity Header Bytes (for reading inline/xline from velocity)
        # ═══════════════════════════════════════════════════════════════════════
        vel_bytes_group = QGroupBox("SEG-Y Velocity Header Bytes (optional)")
        vel_bytes_layout = QFormLayout(vel_bytes_group)

        # Inline byte position
        inline_byte_layout = QHBoxLayout()
        self.vel_inline_byte_spin = QSpinBox()
        self.vel_inline_byte_spin.setRange(0, 240)
        self.vel_inline_byte_spin.setValue(189)  # Standard INLINE_3D byte
        self.vel_inline_byte_spin.setToolTip("Byte position for inline in velocity SEG-Y")
        self.vel_inline_byte_spin.setFixedWidth(70)
        inline_byte_layout.addWidget(self.vel_inline_byte_spin)
        inline_byte_layout.addWidget(QLabel("(189=INLINE_3D, 9=FLDR)"))
        inline_byte_layout.addStretch()
        vel_bytes_layout.addRow("Inline Byte:", inline_byte_layout)

        # Xline byte position
        xline_byte_layout = QHBoxLayout()
        self.vel_xline_byte_spin = QSpinBox()
        self.vel_xline_byte_spin.setRange(0, 240)
        self.vel_xline_byte_spin.setValue(193)  # Standard CROSSLINE_3D byte
        self.vel_xline_byte_spin.setToolTip("Byte position for xline in velocity SEG-Y")
        self.vel_xline_byte_spin.setFixedWidth(70)
        xline_byte_layout.addWidget(self.vel_xline_byte_spin)
        xline_byte_layout.addWidget(QLabel("(193=XLINE_3D, 21=CDP)"))
        xline_byte_layout.addStretch()
        vel_bytes_layout.addRow("Xline Byte:", xline_byte_layout)

        # Use custom bytes checkbox - checked by default
        self.use_custom_vel_bytes = QCheckBox("Use custom byte positions for velocity SEG-Y")
        self.use_custom_vel_bytes.setChecked(True)  # Default to using custom bytes
        self.use_custom_vel_bytes.setToolTip(
            "Enable to use specified byte positions when reading velocity from SEG-Y files.\n"
            "Useful when velocity file uses different header positions than standard."
        )
        self.use_custom_vel_bytes.stateChanged.connect(self._on_vel_bytes_changed)
        vel_bytes_layout.addRow("", self.use_custom_vel_bytes)

        layout.addWidget(vel_bytes_group)

        # ═══════════════════════════════════════════════════════════════════════
        # Output Grid Geometry
        # ═══════════════════════════════════════════════════════════════════════
        grid_group = QGroupBox("Output Grid Geometry")
        grid_layout = QFormLayout(grid_group)

        # Inline range
        inline_range_layout = QHBoxLayout()
        self.inline_min_spin = QSpinBox()
        self.inline_min_spin.setRange(0, 999999)
        inline_range_layout.addWidget(self.inline_min_spin)
        inline_range_layout.addWidget(QLabel(" to "))
        self.inline_max_spin = QSpinBox()
        self.inline_max_spin.setRange(0, 999999)
        inline_range_layout.addWidget(self.inline_max_spin)
        inline_range_layout.addStretch()
        grid_layout.addRow("Inline Range:", inline_range_layout)

        # Xline range
        xline_range_layout = QHBoxLayout()
        self.xline_min_spin = QSpinBox()
        self.xline_min_spin.setRange(0, 999999)
        xline_range_layout.addWidget(self.xline_min_spin)
        xline_range_layout.addWidget(QLabel(" to "))
        self.xline_max_spin = QSpinBox()
        self.xline_max_spin.setRange(0, 999999)
        xline_range_layout.addWidget(self.xline_max_spin)
        xline_range_layout.addStretch()
        grid_layout.addRow("Xline Range:", xline_range_layout)

        # Bin sizes
        bin_layout = QHBoxLayout()
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 1000.0)
        self.dx_spin.setValue(1.0)
        self.dx_spin.setDecimals(1)
        bin_layout.addWidget(QLabel("dx:"))
        bin_layout.addWidget(self.dx_spin)
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 1000.0)
        self.dy_spin.setValue(1.0)
        self.dy_spin.setDecimals(1)
        bin_layout.addWidget(QLabel("dy:"))
        bin_layout.addWidget(self.dy_spin)
        bin_layout.addStretch()
        grid_layout.addRow("Bin Size:", bin_layout)

        # Grid info and detect button
        self.grid_info_label = QLabel("No dataset loaded")
        self.grid_info_label.setStyleSheet("color: #666;")
        self.grid_info_label.setWordWrap(True)
        grid_layout.addRow("Detected:", self.grid_info_label)

        # Detect from Dataset button
        self.detect_from_dataset_btn = QPushButton("Detect from Dataset")
        self.detect_from_dataset_btn.clicked.connect(self._detect_grid_geometry)
        self.detect_from_dataset_btn.setToolTip(
            "Automatically detect grid geometry from the active dataset"
        )
        grid_layout.addRow("", self.detect_from_dataset_btn)

        layout.addWidget(grid_group)

        # ═══════════════════════════════════════════════════════════════════════
        # Velocity File Selection
        # ═══════════════════════════════════════════════════════════════════════
        file_group = QGroupBox("Velocity File")
        file_layout = QFormLayout(file_group)

        self.velocity_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_velocity)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.velocity_path_edit)
        path_layout.addWidget(browse_btn)
        file_layout.addRow("File:", path_layout)

        # File type and units
        self.velocity_type_combo = QComboBox()
        self.velocity_type_combo.addItems(["RMS Velocity", "Interval Velocity"])
        file_layout.addRow("Velocity Type:", self.velocity_type_combo)

        units_layout = QHBoxLayout()
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(["ms", "s"])
        units_layout.addWidget(QLabel("Time:"))
        units_layout.addWidget(self.time_unit_combo)
        self.velocity_unit_combo = QComboBox()
        self.velocity_unit_combo.addItems(["m/s", "km/s", "ft/s"])
        units_layout.addWidget(QLabel("Velocity:"))
        units_layout.addWidget(self.velocity_unit_combo)
        units_layout.addStretch()
        file_layout.addRow("Units:", units_layout)

        layout.addWidget(file_group)

        # ═══════════════════════════════════════════════════════════════════════
        # Velocity Preview and Interpolation
        # ═══════════════════════════════════════════════════════════════════════
        preview_group = QGroupBox("Velocity Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.velocity_preview = QTextEdit()
        self.velocity_preview.setReadOnly(True)
        self.velocity_preview.setMaximumHeight(100)
        self.velocity_preview.setStyleSheet("font-family: monospace;")
        preview_layout.addWidget(self.velocity_preview)

        btn_layout = QHBoxLayout()
        preview_btn = QPushButton("Load Preview")
        preview_btn.clicked.connect(self._load_velocity_preview)
        btn_layout.addWidget(preview_btn)

        self.interpolate_velocity_check = QCheckBox("Interpolate to output grid")
        self.interpolate_velocity_check.setChecked(True)
        self.interpolate_velocity_check.setToolTip(
            "Interpolate velocity model to match output inline/xline grid."
        )
        btn_layout.addWidget(self.interpolate_velocity_check)
        btn_layout.addStretch()
        preview_layout.addLayout(btn_layout)

        layout.addWidget(preview_group)

        self.tab_widget.addTab(tab, "2. Velocity")

    def _create_parameters_tab(self):
        """Create stacking parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # NMO parameters
        nmo_group = QGroupBox("NMO Correction")
        nmo_layout = QFormLayout(nmo_group)

        self.stretch_mute_spin = QDoubleSpinBox()
        self.stretch_mute_spin.setRange(1.0, 5.0)
        self.stretch_mute_spin.setSingleStep(0.1)
        self.stretch_mute_spin.setValue(1.5)
        self.stretch_mute_spin.setToolTip(
            "Maximum allowed NMO stretch factor.\n"
            "Samples with higher stretch are muted.\n"
            "Typical values: 1.3 - 2.0"
        )
        nmo_layout.addRow("Stretch Mute Factor:", self.stretch_mute_spin)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["Linear", "Sinc"])
        nmo_layout.addRow("Interpolation:", self.interpolation_combo)

        layout.addWidget(nmo_group)

        # Stack parameters
        stack_group = QGroupBox("Stacking")
        stack_layout = QFormLayout(stack_group)

        self.stack_method_combo = QComboBox()
        self.stack_method_combo.addItems(["Mean", "Median"])
        self.stack_method_combo.setToolTip(
            "Mean: Standard averaging (faster)\n"
            "Median: Robust to outliers (slower)"
        )
        stack_layout.addRow("Stack Method:", self.stack_method_combo)

        self.min_fold_spin = QSpinBox()
        self.min_fold_spin.setRange(1, 100)
        self.min_fold_spin.setValue(1)
        self.min_fold_spin.setToolTip(
            "Minimum number of traces required.\n"
            "Samples with lower fold are zeroed."
        )
        stack_layout.addRow("Minimum Fold:", self.min_fold_spin)

        layout.addWidget(stack_group)
        layout.addStretch()

        self.tab_widget.addTab(tab, "3. Parameters")

    def _create_output_tab(self):
        """Create output configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout(output_group)

        # Output directory
        self.output_dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(browse_btn)
        output_layout.addRow("Output Directory:", dir_layout)

        # Output name
        self.output_name_edit = QLineEdit("qc_stack")
        output_layout.addRow("Output Name:", self.output_name_edit)

        layout.addWidget(output_group)

        # Output preview
        preview_group = QGroupBox("Output Files")
        preview_layout = QVBoxLayout(preview_group)

        self.output_preview = QLabel()
        self.output_preview.setWordWrap(True)
        self.output_preview.setStyleSheet("color: #666;")
        preview_layout.addWidget(self.output_preview)

        layout.addWidget(preview_group)
        layout.addStretch()

        self.tab_widget.addTab(tab, "4. Output")

    def _connect_signals(self):
        """Connect widget signals."""
        self.inline_edit.textChanged.connect(self._update_summary)
        self.velocity_path_edit.textChanged.connect(self._update_summary)
        self.output_dir_edit.textChanged.connect(self._update_output_preview)
        self.output_name_edit.textChanged.connect(self._update_output_preview)
        self.stretch_mute_spin.valueChanged.connect(self._update_summary)
        self.stack_method_combo.currentIndexChanged.connect(self._update_summary)

        # Re-detect grid geometry when header mapping changes
        self.inline_header_combo.currentTextChanged.connect(self._on_header_changed)
        self.xline_header_combo.currentTextChanged.connect(self._on_header_changed)

    def _on_header_changed(self):
        """Handle header combo box change - re-detect grid geometry."""
        if self._dataset_info:
            self._detect_grid_geometry()

    def _on_vel_bytes_changed(self):
        """Handle velocity byte position change - reload velocity preview."""
        if self.velocity_path_edit.text():
            self._load_velocity_preview()

    def _browse_dataset(self):
        """Browse for dataset directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory",
            str(Path(self.dataset_path).parent) if self.dataset_path else "",
        )
        if path:
            self.dataset_path = path
            self.dataset_path_edit.setText(path)
            self._load_dataset_info()

    def _browse_velocity(self):
        """Browse for velocity file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Velocity File",
            "",
            "Velocity Files (*.txt *.vel *.asc *.json *.sgy *.segy);;All Files (*)"
        )
        if path:
            self.velocity_path_edit.setText(path)
            self._load_velocity_preview()

    def _browse_output_dir(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_edit.text() or str(Path.home()),
        )
        if path:
            self.output_dir_edit.setText(path)
            self._update_output_preview()

    def _load_dataset_info(self):
        """Load and display dataset information, including QC config from import."""
        try:
            dataset_path = Path(self.dataset_path)

            # Try to load metadata
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)

                n_traces = metadata.get('n_traces', '?')
                n_samples = metadata.get('n_samples', '?')
                sort_key = metadata.get('sorted_by', 'unknown')

                # Check for QC config from SEG-Y import
                qc_config = metadata.get('qc_config', {})
                qc_inlines_from_import = qc_config.get('inline_numbers', [])

                # Initialize dataset info - always set it so grid detection works
                self._dataset_info = {
                    'n_traces': n_traces,
                    'n_samples': n_samples,
                    'sort_key': sort_key,
                    'available_headers': [],
                }

                # Get inline range from ensemble index if available
                ensemble_path = dataset_path / "ensemble_index.parquet"
                headers_path = dataset_path / "headers.parquet"
                inline_range = "unknown"

                # Try ensemble_index first for ensemble-level info
                if ensemble_path.exists():
                    import pandas as pd
                    df = pd.read_parquet(ensemble_path)
                    self._dataset_info['n_ensembles'] = len(df)
                    self._dataset_info['available_headers'] = list(df.columns)

                    # Detect inline column
                    inline_col = None
                    for col in ['INLINE_NO', 'inline_no', 'INLINE_3D', 'Inline3D', 'IL', 'FieldRecord', 'FLDR']:
                        if col in df.columns:
                            inline_col = col
                            break

                    if inline_col:
                        inline_range = f"{df[inline_col].min()} - {df[inline_col].max()}"
                        self._dataset_info['inline_col'] = inline_col
                        self._dataset_info['inline_min'] = int(df[inline_col].min())
                        self._dataset_info['inline_max'] = int(df[inline_col].max())

                    # Try to detect xline column
                    for col in ['XLINE_NO', 'CROSSLINE_3D', 'Crossline3D', 'XL', 'CDP', 'cdp']:
                        if col in df.columns:
                            self._dataset_info['xline_col'] = col
                            self._dataset_info['xline_min'] = int(df[col].min())
                            self._dataset_info['xline_max'] = int(df[col].max())
                            break

                # Also check headers.parquet for more complete column list
                if headers_path.exists():
                    import pandas as pd
                    # Just get column names without loading full data
                    df_headers = pd.read_parquet(headers_path, columns=None)
                    header_cols = list(df_headers.columns)
                    # Merge with existing headers
                    existing = set(self._dataset_info.get('available_headers', []))
                    for col in header_cols:
                        if col not in existing:
                            self._dataset_info['available_headers'].append(col)

                # Build info text
                info_text = f"Traces: {n_traces:,} | Samples: {n_samples}\n"
                info_text += f"Sort: {sort_key} | Inline range: {inline_range}"

                # Show QC config info if present
                if qc_inlines_from_import:
                    info_text += f"\nQC Inlines from import: {len(qc_inlines_from_import)} lines"

                self.dataset_info_label.setText(info_text)

                # Update velocity tab dataset label
                dataset_name = Path(self.dataset_path).name
                vel_tab_text = f"{dataset_name}\n{n_traces:,} traces | Inline: {inline_range}"
                self.velocity_tab_dataset_label.setText(vel_tab_text)
                self.velocity_tab_dataset_label.setStyleSheet(
                    "QLabel { background-color: #d4edda; padding: 5px; border-radius: 3px; font-weight: bold; }"
                )

                # Pre-populate QC inlines if they were specified during import
                # and the inline_edit is currently empty
                if qc_inlines_from_import and not self.inline_edit.text().strip():
                    inline_str = ", ".join(str(i) for i in qc_inlines_from_import)
                    self.inline_edit.setText(inline_str)
                    self._parse_inlines()
                    logger.info(f"Pre-populated {len(qc_inlines_from_import)} QC inlines from import config")

                # Populate header combos with detected headers
                self._populate_header_combos()

                # Auto-detect grid geometry
                self._detect_grid_geometry()

            else:
                self.dataset_info_label.setText("Metadata not found")
                self._dataset_info = None

        except Exception as e:
            self.dataset_info_label.setText(f"Error: {e}")
            logger.exception("Error loading dataset info")

    def _populate_header_combos(self):
        """Populate header combo boxes with available headers from dataset."""
        if not self._dataset_info:
            return

        available = self._dataset_info.get('available_headers', [])
        if not available:
            return

        # Clear existing items and populate from dataset
        self.inline_header_combo.clear()
        self.xline_header_combo.clear()

        # Add all available headers from dataset
        for header in sorted(available):
            self.inline_header_combo.addItem(header)
            self.xline_header_combo.addItem(header)

        # Select detected inline column or best guess
        inline_selected = False
        if 'inline_col' in self._dataset_info:
            idx = self.inline_header_combo.findText(self._dataset_info['inline_col'])
            if idx >= 0:
                self.inline_header_combo.setCurrentIndex(idx)
                inline_selected = True

        # If no inline detected, try common inline header names
        if not inline_selected:
            for name in ['INLINE_NO', 'inline_no', 'INLINE_3D', 'Inline3D', 'IL', 'FieldRecord', 'FLDR']:
                idx = self.inline_header_combo.findText(name)
                if idx >= 0:
                    self.inline_header_combo.setCurrentIndex(idx)
                    break

        # Select detected xline column or best guess
        xline_selected = False
        if 'xline_col' in self._dataset_info:
            idx = self.xline_header_combo.findText(self._dataset_info['xline_col'])
            if idx >= 0:
                self.xline_header_combo.setCurrentIndex(idx)
                xline_selected = True

        # If no xline detected, try common xline header names
        if not xline_selected:
            for name in ['XLINE_NO', 'CROSSLINE_3D', 'Crossline3D', 'XL', 'CDP', 'cdp', 'TraceNumber']:
                idx = self.xline_header_combo.findText(name)
                if idx >= 0:
                    self.xline_header_combo.setCurrentIndex(idx)
                    break

    def _detect_grid_geometry(self):
        """Detect and populate grid geometry from dataset."""
        if not self._dataset_info:
            self.grid_info_label.setText("No dataset loaded")
            return

        try:
            dataset_path = Path(self.dataset_path)

            # Try to get geometry from headers.parquet for more accurate ranges
            headers_path = dataset_path / "headers.parquet"
            ensemble_path = dataset_path / "ensemble_index.parquet"

            inline_col = self.inline_header_combo.currentText()
            xline_col = self.xline_header_combo.currentText()

            inline_min = inline_max = None
            xline_min = xline_max = None
            dx = dy = 1.0

            import pandas as pd

            # First, check if we already have detected values in _dataset_info
            if 'inline_min' in self._dataset_info and 'inline_max' in self._dataset_info:
                inline_min = self._dataset_info['inline_min']
                inline_max = self._dataset_info['inline_max']
            if 'xline_min' in self._dataset_info and 'xline_max' in self._dataset_info:
                xline_min = self._dataset_info['xline_min']
                xline_max = self._dataset_info['xline_max']

            # Try headers.parquet for more accurate detection using selected columns
            if headers_path.exists():
                # Read only the columns we need
                df_sample = pd.read_parquet(headers_path, columns=None)
                available_cols = list(df_sample.columns)

                cols_to_read = []
                if inline_col in available_cols:
                    cols_to_read.append(inline_col)
                if xline_col in available_cols:
                    cols_to_read.append(xline_col)

                if cols_to_read:
                    df = pd.read_parquet(headers_path, columns=cols_to_read)

                    if inline_col in df.columns:
                        inline_min = int(df[inline_col].min())
                        inline_max = int(df[inline_col].max())
                        # Estimate dx from unique sorted values
                        unique_inlines = np.sort(df[inline_col].unique())
                        if len(unique_inlines) > 1:
                            diffs = np.diff(unique_inlines)
                            dx = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

                    if xline_col in df.columns:
                        xline_min = int(df[xline_col].min())
                        xline_max = int(df[xline_col].max())
                        # Estimate dy from unique sorted values
                        unique_xlines = np.sort(df[xline_col].unique())
                        if len(unique_xlines) > 1:
                            diffs = np.diff(unique_xlines)
                            dy = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

            # Fallback to ensemble index for the selected columns
            elif ensemble_path.exists() and (inline_min is None or xline_min is None):
                df = pd.read_parquet(ensemble_path)
                if inline_min is None and inline_col in df.columns:
                    inline_min = int(df[inline_col].min())
                    inline_max = int(df[inline_col].max())
                if xline_min is None and xline_col in df.columns:
                    xline_min = int(df[xline_col].min())
                    xline_max = int(df[xline_col].max())

            # Update UI with detected or cached values
            if inline_min is not None:
                self.inline_min_spin.setValue(inline_min)
                self.inline_max_spin.setValue(inline_max)
            if xline_min is not None:
                self.xline_min_spin.setValue(xline_min)
                self.xline_max_spin.setValue(xline_max)

            self.dx_spin.setValue(dx)
            self.dy_spin.setValue(dy)

            # Update info label
            info_parts = []
            if inline_min is not None:
                info_parts.append(f"Inline ({inline_col}): {inline_min}-{inline_max} (dx={dx:.1f})")
            if xline_min is not None:
                info_parts.append(f"Xline ({xline_col}): {xline_min}-{xline_max} (dy={dy:.1f})")

            n_inlines = int((inline_max - inline_min) / dx + 1) if inline_min is not None else 0
            n_xlines = int((xline_max - xline_min) / dy + 1) if xline_min is not None else 0
            if n_inlines > 0 and n_xlines > 0:
                info_parts.append(f"Grid: {n_inlines} x {n_xlines} = {n_inlines * n_xlines:,} bins")

            if info_parts:
                self.grid_info_label.setText("\n".join(info_parts))
                self.grid_info_label.setStyleSheet("color: #333;")
            else:
                self.grid_info_label.setText(f"Could not detect geometry\nHeaders: {inline_col}, {xline_col} not found")
                self.grid_info_label.setStyleSheet("color: #c00;")

        except Exception as e:
            self.grid_info_label.setText(f"Error detecting geometry: {e}")
            self.grid_info_label.setStyleSheet("color: #c00;")
            logger.exception("Error detecting grid geometry")

    def _load_velocity_preview(self):
        """Load and display velocity file preview including inline/xline ranges."""
        try:
            from utils.velocity_io import preview_velocity_file, get_velocity_summary

            filepath = self.velocity_path_edit.text()
            if not filepath or not Path(filepath).exists():
                self.velocity_preview.setText("No file selected or file not found")
                return

            # Pass custom byte positions if SEG-Y and custom bytes enabled
            inline_byte = None
            xline_byte = None
            if self.use_custom_vel_bytes.isChecked():
                inline_byte = self.vel_inline_byte_spin.value()
                xline_byte = self.vel_xline_byte_spin.value()

            info = preview_velocity_file(
                filepath,
                inline_byte=inline_byte,
                xline_byte=xline_byte
            )
            self._velocity_info = info

            if info.is_valid:
                lines = [
                    f"Format: {info.format.value}",
                    f"Locations: {info.n_locations}",
                    f"Time samples: {info.n_time_samples}",
                    f"Time range: {info.time_range[0]:.3f} - {info.time_range[1]:.3f}",
                    f"Velocity range: {info.velocity_range[0]:.0f} - {info.velocity_range[1]:.0f}",
                ]
                # Show spatial ranges - inline/xline if available, otherwise CDP
                if info.inline_range:
                    lines.append(f"Inline range: {info.inline_range[0]} - {info.inline_range[1]}")
                if info.xline_range:
                    lines.append(f"Xline range: {info.xline_range[0]} - {info.xline_range[1]}")
                if info.cdp_range:
                    # Always show CDP range for reference
                    lines.append(f"CDP range: {info.cdp_range[0]} - {info.cdp_range[1]}")
                self.velocity_preview.setText("\n".join(lines))
            else:
                self.velocity_preview.setText(f"Error: {info.error_message}")

        except Exception as e:
            self.velocity_preview.setText(f"Error: {e}")
            logger.exception("Error loading velocity preview")

    def _parse_inlines(self):
        """Parse inline numbers from input field."""
        text = self.inline_edit.text().strip()
        if not text:
            self.parsed_inlines_label.setText("Parsed: (none)")
            return []

        inlines = []
        try:
            for part in text.split(','):
                part = part.strip()
                if '-' in part:
                    # Range
                    start, end = part.split('-')
                    inlines.extend(range(int(start), int(end) + 1))
                else:
                    inlines.append(int(part))

            inlines = sorted(set(inlines))

            if len(inlines) <= 10:
                display = ", ".join(str(i) for i in inlines)
            else:
                display = f"{inlines[0]}, {inlines[1]}, ... {inlines[-2]}, {inlines[-1]} ({len(inlines)} total)"

            self.parsed_inlines_label.setText(f"Parsed: {display}")
            self._update_trace_estimate(inlines)
            return inlines

        except Exception as e:
            self.parsed_inlines_label.setText(f"Parse error: {e}")
            return []

    def _quick_select(self, interval: int):
        """Quick select inlines at regular interval."""
        if not self._dataset_info:
            QMessageBox.warning(self, "Warning", "Load a dataset first")
            return

        min_il = self._dataset_info.get('inline_min', 0)
        max_il = self._dataset_info.get('inline_max', 100)

        inlines = list(range(min_il, max_il + 1, interval))
        self.inline_edit.setText(", ".join(str(i) for i in inlines))
        self._parse_inlines()

    def _update_trace_estimate(self, inlines: List[int]):
        """Update trace count estimate."""
        if not self._dataset_info:
            self.trace_estimate_label.setText("Estimated traces: unknown")
            return

        n_ensembles = self._dataset_info.get('n_ensembles', 0)
        inline_range = self._dataset_info.get('inline_max', 1) - self._dataset_info.get('inline_min', 0) + 1

        if inline_range > 0 and n_ensembles > 0:
            avg_per_inline = n_ensembles / inline_range
            estimated = int(len(inlines) * avg_per_inline)
            self.trace_estimate_label.setText(
                f"Estimated: ~{len(inlines)} inlines, ~{estimated:,} gathers"
            )
        else:
            self.trace_estimate_label.setText(f"Selected: {len(inlines)} inlines")

    def _update_output_preview(self):
        """Update output file preview."""
        output_dir = self.output_dir_edit.text()
        output_name = self.output_name_edit.text() or "qc_stack"

        if output_dir:
            output_path = Path(output_dir) / output_name
            self.output_preview.setText(
                f"Stack data: {output_path}.zarr\n"
                f"Metadata: {output_path}_metadata.json"
            )
        else:
            self.output_preview.setText("Select output directory")

    def _update_summary(self):
        """Update summary text."""
        inlines = self._parse_inlines()
        velocity_file = Path(self.velocity_path_edit.text()).name if self.velocity_path_edit.text() else "(none)"
        stretch = self.stretch_mute_spin.value()
        method = self.stack_method_combo.currentText()

        summary = [
            f"Dataset: {Path(self.dataset_path).name if self.dataset_path else '(none)'}",
            f"Inlines: {len(inlines)} selected",
            f"Velocity: {velocity_file}",
            f"NMO stretch mute: {stretch}, Stack: {method}",
        ]

        self.summary_text.setText("\n".join(summary))

    def _on_accept(self):
        """Handle OK button."""
        config = self.get_config()
        is_valid, error = config.validate()

        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error)
            return

        self.config_ready.emit(config)
        self.accept()

    def get_config(self) -> QCStackingConfig:
        """Get current configuration."""
        inlines = self._parse_inlines()

        # Get grid values, using None if spinbox is at 0 (unset)
        inline_min = self.inline_min_spin.value() if self.inline_min_spin.value() > 0 else None
        inline_max = self.inline_max_spin.value() if self.inline_max_spin.value() > 0 else None
        xline_min = self.xline_min_spin.value() if self.xline_min_spin.value() > 0 else None
        xline_max = self.xline_max_spin.value() if self.xline_max_spin.value() > 0 else None

        # Get velocity spatial info from preview if available
        vel_inline_range = None
        vel_xline_range = None
        vel_cdp_range = None
        if self._velocity_info and self._velocity_info.is_valid:
            vel_inline_range = self._velocity_info.inline_range
            vel_xline_range = self._velocity_info.xline_range
            vel_cdp_range = self._velocity_info.cdp_range

        return QCStackingConfig(
            dataset_path=self.dataset_path,
            inline_numbers=inlines,
            velocity_file=self.velocity_path_edit.text(),
            velocity_type='rms' if self.velocity_type_combo.currentIndex() == 0 else 'interval',
            time_unit=self.time_unit_combo.currentText(),
            velocity_unit=self.velocity_unit_combo.currentText(),
            stretch_mute=self.stretch_mute_spin.value(),
            stack_method=self.stack_method_combo.currentText().lower(),
            min_fold=self.min_fold_spin.value(),
            output_dir=self.output_dir_edit.text(),
            output_name=self.output_name_edit.text() or "qc_stack",
            # Header mapping
            inline_header=self.inline_header_combo.currentText(),
            xline_header=self.xline_header_combo.currentText(),
            # Grid geometry
            inline_min=inline_min,
            inline_max=inline_max,
            xline_min=xline_min,
            xline_max=xline_max,
            dx=self.dx_spin.value(),
            dy=self.dy_spin.value(),
            interpolate_velocity=self.interpolate_velocity_check.isChecked(),
            # SEG-Y velocity byte mapping
            use_custom_vel_bytes=self.use_custom_vel_bytes.isChecked(),
            vel_inline_byte=self.vel_inline_byte_spin.value(),
            vel_xline_byte=self.vel_xline_byte_spin.value(),
            # Velocity spatial info (for extrapolation)
            vel_inline_range=vel_inline_range,
            vel_xline_range=vel_xline_range,
            vel_cdp_range=vel_cdp_range,
        )

    def set_config(self, config: QCStackingConfig):
        """Set dialog from configuration."""
        self.dataset_path = config.dataset_path
        self.dataset_path_edit.setText(config.dataset_path)

        if config.inline_numbers:
            self.inline_edit.setText(", ".join(str(i) for i in config.inline_numbers))

        self.velocity_path_edit.setText(config.velocity_file)
        self.velocity_type_combo.setCurrentIndex(0 if config.velocity_type == 'rms' else 1)
        self.time_unit_combo.setCurrentText(config.time_unit)
        self.velocity_unit_combo.setCurrentText(config.velocity_unit)
        self.stretch_mute_spin.setValue(config.stretch_mute)
        self.stack_method_combo.setCurrentText(config.stack_method.capitalize())
        self.min_fold_spin.setValue(config.min_fold)
        self.output_dir_edit.setText(config.output_dir)
        self.output_name_edit.setText(config.output_name)

        # Header mapping
        self.inline_header_combo.setCurrentText(config.inline_header)
        self.xline_header_combo.setCurrentText(config.xline_header)

        # Grid geometry
        if config.inline_min is not None:
            self.inline_min_spin.setValue(config.inline_min)
        if config.inline_max is not None:
            self.inline_max_spin.setValue(config.inline_max)
        if config.xline_min is not None:
            self.xline_min_spin.setValue(config.xline_min)
        if config.xline_max is not None:
            self.xline_max_spin.setValue(config.xline_max)
        self.dx_spin.setValue(config.dx)
        self.dy_spin.setValue(config.dy)
        self.interpolate_velocity_check.setChecked(config.interpolate_velocity)

        # SEG-Y velocity byte mapping
        self.use_custom_vel_bytes.setChecked(config.use_custom_vel_bytes)
        self.vel_inline_byte_spin.setValue(config.vel_inline_byte)
        self.vel_xline_byte_spin.setValue(config.vel_xline_byte)

        self._load_dataset_info()
        self._parse_inlines()
        self._update_summary()
