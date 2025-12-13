"""
QC Batch Processing Dialog - Apply processing chain to selected gathers

Multi-tab dialog for:
1. Inline Selection - choose inline numbers for QC batch processing
2. Processing Chain - configure processor chain using ProcessingChainWidget
3. Velocity/NMO - optional NMO correction settings
4. Output - what to output (before, after, stacks, difference)

Usage:
    dialog = QCBatchDialog(dataset_path, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        config = dialog.get_config()
        # Execute batch processing with config
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import json

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QTabWidget, QWidget, QTextEdit,
    QDialogButtonBox, QMessageBox, QProgressBar, QFormLayout,
    QRadioButton, QButtonGroup, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal

from views.processing_chain_widget import ProcessingChainWidget

logger = logging.getLogger(__name__)


@dataclass
class QCBatchConfig:
    """
    Configuration for QC batch processing workflow.

    Attributes:
        dataset_path: Path to source dataset (Zarr/Parquet)
        inline_numbers: List of inline numbers to process
        processing_chain: List of processor configs
        apply_nmo: Whether to apply NMO before processing
        velocity_file: Path to velocity file (if apply_nmo)
        velocity_type: 'rms' or 'interval'
        stretch_mute: NMO stretch mute factor
        output_before_gathers: Output gathers before processing
        output_after_gathers: Output gathers after processing
        output_noise_gathers: Output noise gathers (before - after = input - processed)
        output_before_stack: Output stack of before gathers
        output_after_stack: Output stack of after gathers
        output_noise_stack: Output stack of noise gathers
        output_difference: Output difference (after_stack - before_stack)
        stack_method: 'mean' or 'median' for stacking
        output_dir: Output directory
        output_name: Output dataset name prefix
        mute_velocity: Mute velocity in m/s (0 = disabled)
        mute_top: Apply top mute
        mute_bottom: Apply bottom mute
        mute_taper: Taper samples for mute transition
        mute_target: Which data to apply mute to ('output', 'input', 'processed')
    """
    dataset_path: str = ""
    inline_numbers: List[int] = field(default_factory=list)
    processing_chain: List[Dict[str, Any]] = field(default_factory=list)

    # NMO settings
    apply_nmo: bool = False
    velocity_file: str = ""
    velocity_type: str = "rms"
    time_unit: str = "ms"
    velocity_unit: str = "m/s"
    stretch_mute: float = 1.5

    # Output options
    output_before_gathers: bool = False
    output_after_gathers: bool = True
    output_noise_gathers: bool = False
    output_before_stack: bool = True
    output_after_stack: bool = True
    output_noise_stack: bool = False
    output_difference: bool = True
    stack_method: str = "mean"
    min_fold: int = 1

    # Mute settings (optional, applied during processing)
    mute_velocity: float = 0.0  # 0 = disabled
    mute_top: bool = False
    mute_bottom: bool = False
    mute_taper: int = 20
    mute_target: str = "output"  # 'output', 'input', or 'processed'

    # Output location
    output_dir: str = ""
    output_name: str = "qc_batch"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_path': self.dataset_path,
            'inline_numbers': self.inline_numbers,
            'processing_chain': self.processing_chain,
            'apply_nmo': self.apply_nmo,
            'velocity_file': self.velocity_file,
            'velocity_type': self.velocity_type,
            'time_unit': self.time_unit,
            'velocity_unit': self.velocity_unit,
            'stretch_mute': self.stretch_mute,
            'output_before_gathers': self.output_before_gathers,
            'output_after_gathers': self.output_after_gathers,
            'output_noise_gathers': self.output_noise_gathers,
            'output_before_stack': self.output_before_stack,
            'output_after_stack': self.output_after_stack,
            'output_noise_stack': self.output_noise_stack,
            'output_difference': self.output_difference,
            'stack_method': self.stack_method,
            'min_fold': self.min_fold,
            'mute_velocity': self.mute_velocity,
            'mute_top': self.mute_top,
            'mute_bottom': self.mute_bottom,
            'mute_taper': self.mute_taper,
            'mute_target': self.mute_target,
            'output_dir': self.output_dir,
            'output_name': self.output_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QCBatchConfig':
        return cls(
            dataset_path=d.get('dataset_path', ''),
            inline_numbers=d.get('inline_numbers', []),
            processing_chain=d.get('processing_chain', []),
            apply_nmo=d.get('apply_nmo', False),
            velocity_file=d.get('velocity_file', ''),
            velocity_type=d.get('velocity_type', 'rms'),
            time_unit=d.get('time_unit', 'ms'),
            velocity_unit=d.get('velocity_unit', 'm/s'),
            stretch_mute=d.get('stretch_mute', 1.5),
            output_before_gathers=d.get('output_before_gathers', False),
            output_after_gathers=d.get('output_after_gathers', True),
            output_noise_gathers=d.get('output_noise_gathers', False),
            output_before_stack=d.get('output_before_stack', True),
            output_after_stack=d.get('output_after_stack', True),
            output_noise_stack=d.get('output_noise_stack', False),
            output_difference=d.get('output_difference', True),
            stack_method=d.get('stack_method', 'mean'),
            min_fold=d.get('min_fold', 1),
            mute_velocity=d.get('mute_velocity', 0.0),
            mute_top=d.get('mute_top', False),
            mute_bottom=d.get('mute_bottom', False),
            mute_taper=d.get('mute_taper', 20),
            mute_target=d.get('mute_target', 'output'),
            output_dir=d.get('output_dir', ''),
            output_name=d.get('output_name', 'qc_batch'),
        )

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration. Returns (is_valid, error_message)."""
        if not self.dataset_path:
            return False, "Dataset path not specified"
        if not Path(self.dataset_path).exists():
            return False, f"Dataset not found: {self.dataset_path}"
        if not self.inline_numbers:
            return False, "No inline numbers specified"
        if not self.processing_chain:
            return False, "No processors in chain"
        if self.apply_nmo:
            if not self.velocity_file:
                return False, "NMO enabled but no velocity file specified"
            if not Path(self.velocity_file).exists():
                return False, f"Velocity file not found: {self.velocity_file}"
        if not any([
            self.output_before_gathers,
            self.output_after_gathers,
            self.output_noise_gathers,
            self.output_before_stack,
            self.output_after_stack,
            self.output_noise_stack,
            self.output_difference
        ]):
            return False, "No outputs selected"
        if not self.output_dir:
            return False, "Output directory not specified"
        # Validate mute settings
        if (self.mute_top or self.mute_bottom) and self.mute_velocity <= 0:
            return False, "Mute enabled but no mute velocity specified"
        return True, ""

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'QCBatchConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


class QCBatchDialog(QDialog):
    """
    Dialog for configuring QC batch processing workflow.

    Provides tabbed interface for:
    - Line selection
    - Processing chain configuration
    - NMO/Velocity settings
    - Output configuration
    """

    config_ready = pyqtSignal(object)  # Emits QCBatchConfig when ready

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.dataset_path = dataset_path or ""
        self._dataset_info: Optional[Dict[str, Any]] = None

        self._init_ui()
        self._connect_signals()

        if self.dataset_path:
            self._load_dataset_info()

    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("QC Batch Processing")
        self.setMinimumSize(850, 650)
        self.resize(950, 700)

        layout = QVBoxLayout(self)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self._create_line_selection_tab()
        self._create_processing_tab()
        self._create_nmo_tab()
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

        # Button row
        button_layout = QHBoxLayout()

        # Save/Load config buttons
        save_btn = QPushButton("Save Config...")
        save_btn.clicked.connect(self._save_config)
        button_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Config...")
        load_btn.clicked.connect(self._load_config)
        button_layout.addWidget(load_btn)

        button_layout.addStretch()

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        button_layout.addWidget(self.button_box)

        layout.addLayout(button_layout)

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

    def _create_processing_tab(self):
        """Create processing chain configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Processing chain widget
        self.chain_widget = ProcessingChainWidget()
        layout.addWidget(self.chain_widget)

        # Chain summary
        summary_layout = QHBoxLayout()
        summary_layout.addWidget(QLabel("Chain:"))
        self.chain_summary_label = QLabel("No processors configured")
        self.chain_summary_label.setStyleSheet("color: #666;")
        summary_layout.addWidget(self.chain_summary_label)
        summary_layout.addStretch()
        layout.addLayout(summary_layout)

        self.tab_widget.addTab(tab, "2. Processing")

    def _create_nmo_tab(self):
        """Create NMO/Velocity configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # NMO enable
        nmo_group = QGroupBox("NMO Correction (Optional)")
        nmo_layout = QVBoxLayout(nmo_group)

        self.apply_nmo_check = QCheckBox("Apply NMO before processing")
        self.apply_nmo_check.setToolTip(
            "Apply Normal Moveout correction before processing chain.\n"
            "This flattens hyperbolic events for better filtering."
        )
        self.apply_nmo_check.stateChanged.connect(self._on_nmo_toggled)
        nmo_layout.addWidget(self.apply_nmo_check)

        # Velocity settings (initially disabled)
        self.velocity_widget = QWidget()
        velocity_layout = QFormLayout(self.velocity_widget)

        # Velocity file
        self.velocity_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_velocity)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.velocity_path_edit)
        path_layout.addWidget(browse_btn)
        velocity_layout.addRow("Velocity File:", path_layout)

        # Velocity type
        self.velocity_type_combo = QComboBox()
        self.velocity_type_combo.addItems(["RMS Velocity", "Interval Velocity"])
        velocity_layout.addRow("Velocity Type:", self.velocity_type_combo)

        # Units
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
        velocity_layout.addRow("Units:", units_layout)

        # Stretch mute
        self.stretch_mute_spin = QDoubleSpinBox()
        self.stretch_mute_spin.setRange(1.0, 5.0)
        self.stretch_mute_spin.setSingleStep(0.1)
        self.stretch_mute_spin.setValue(1.5)
        self.stretch_mute_spin.setToolTip(
            "Maximum allowed NMO stretch factor.\n"
            "Samples with higher stretch are muted."
        )
        velocity_layout.addRow("Stretch Mute Factor:", self.stretch_mute_spin)

        self.velocity_widget.setEnabled(False)
        nmo_layout.addWidget(self.velocity_widget)

        layout.addWidget(nmo_group)

        # Velocity preview
        preview_group = QGroupBox("Velocity Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.velocity_preview = QTextEdit()
        self.velocity_preview.setReadOnly(True)
        self.velocity_preview.setMaximumHeight(120)
        self.velocity_preview.setStyleSheet("font-family: monospace;")
        preview_layout.addWidget(self.velocity_preview)

        preview_btn = QPushButton("Load Preview")
        preview_btn.clicked.connect(self._load_velocity_preview)
        preview_layout.addWidget(preview_btn)

        layout.addWidget(preview_group)
        layout.addStretch()

        self.tab_widget.addTab(tab, "3. NMO/Velocity")

    def _create_output_tab(self):
        """Create output configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Output selection
        output_group = QGroupBox("Output Selection")
        output_layout = QVBoxLayout(output_group)

        # Gathers
        gathers_label = QLabel("<b>Gathers:</b>")
        output_layout.addWidget(gathers_label)

        gathers_layout = QHBoxLayout()
        self.output_before_gathers_check = QCheckBox("Before Processing")
        self.output_before_gathers_check.setToolTip("Output original gathers (optionally NMO corrected)")
        gathers_layout.addWidget(self.output_before_gathers_check)

        self.output_after_gathers_check = QCheckBox("After Processing")
        self.output_after_gathers_check.setChecked(True)
        self.output_after_gathers_check.setToolTip("Output processed gathers")
        gathers_layout.addWidget(self.output_after_gathers_check)

        self.output_noise_gathers_check = QCheckBox("Noise (Before - After)")
        self.output_noise_gathers_check.setToolTip("Output noise gathers (Input - Processed)")
        gathers_layout.addWidget(self.output_noise_gathers_check)
        gathers_layout.addStretch()
        output_layout.addLayout(gathers_layout)

        # Stacks
        stacks_label = QLabel("<b>Stacks:</b>")
        output_layout.addWidget(stacks_label)

        stacks_layout = QHBoxLayout()
        self.output_before_stack_check = QCheckBox("Before Stack")
        self.output_before_stack_check.setChecked(True)
        self.output_before_stack_check.setToolTip("Output stack of original gathers")
        stacks_layout.addWidget(self.output_before_stack_check)

        self.output_after_stack_check = QCheckBox("After Stack")
        self.output_after_stack_check.setChecked(True)
        self.output_after_stack_check.setToolTip("Output stack of processed gathers")
        stacks_layout.addWidget(self.output_after_stack_check)

        self.output_noise_stack_check = QCheckBox("Noise Stack")
        self.output_noise_stack_check.setToolTip("Output stack of noise gathers")
        stacks_layout.addWidget(self.output_noise_stack_check)

        self.output_difference_check = QCheckBox("Difference (After - Before)")
        self.output_difference_check.setChecked(True)
        self.output_difference_check.setToolTip("Output difference between stacks for QC")
        stacks_layout.addWidget(self.output_difference_check)
        stacks_layout.addStretch()
        output_layout.addLayout(stacks_layout)

        layout.addWidget(output_group)

        # Mute Options
        mute_group = QGroupBox("Mute Options (Applied During Processing)")
        mute_layout = QGridLayout(mute_group)

        self.mute_top_check = QCheckBox("Top Mute")
        self.mute_top_check.setToolTip("Zero samples before mute time: T = |offset| / velocity")
        self.mute_top_check.stateChanged.connect(self._on_mute_toggled)
        mute_layout.addWidget(self.mute_top_check, 0, 0)

        self.mute_bottom_check = QCheckBox("Bottom Mute")
        self.mute_bottom_check.setToolTip("Zero samples after mute time: T = |offset| / velocity")
        self.mute_bottom_check.stateChanged.connect(self._on_mute_toggled)
        mute_layout.addWidget(self.mute_bottom_check, 0, 1)

        mute_layout.addWidget(QLabel("Apply mute to:"), 1, 0)
        self.mute_target_combo = QComboBox()
        self.mute_target_combo.addItems([
            "Output (Noise = Before - After)",
            "Input (Before processing)",
            "Processed (After processing)"
        ])
        self.mute_target_combo.setToolTip(
            "Output: Mute applied to noise result\n"
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

        # Initially disable velocity/taper until mute is enabled
        self._mute_params_widget = QWidget()
        self.mute_velocity_spin.setEnabled(False)
        self.mute_taper_spin.setEnabled(False)
        self.mute_target_combo.setEnabled(False)

        layout.addWidget(mute_group)

        # Stacking parameters
        stack_group = QGroupBox("Stacking Parameters")
        stack_layout = QFormLayout(stack_group)

        self.stack_method_combo = QComboBox()
        self.stack_method_combo.addItems(["Mean", "Median"])
        stack_layout.addRow("Stack Method:", self.stack_method_combo)

        self.min_fold_spin = QSpinBox()
        self.min_fold_spin.setRange(1, 100)
        self.min_fold_spin.setValue(1)
        stack_layout.addRow("Minimum Fold:", self.min_fold_spin)

        layout.addWidget(stack_group)

        # Output location
        location_group = QGroupBox("Output Location")
        location_layout = QFormLayout(location_group)

        self.output_dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(browse_btn)
        location_layout.addRow("Output Directory:", dir_layout)

        self.output_name_edit = QLineEdit("qc_batch")
        location_layout.addRow("Output Prefix:", self.output_name_edit)

        layout.addWidget(location_group)

        # Output preview
        preview_group = QGroupBox("Output Files")
        preview_layout = QVBoxLayout(preview_group)

        self.output_preview = QLabel()
        self.output_preview.setWordWrap(True)
        self.output_preview.setStyleSheet("color: #666; font-size: 11px;")
        preview_layout.addWidget(self.output_preview)

        layout.addWidget(preview_group)
        layout.addStretch()

        self.tab_widget.addTab(tab, "4. Output")

    def _connect_signals(self):
        """Connect widget signals."""
        self.inline_edit.textChanged.connect(self._update_summary)
        self.chain_widget.chain_changed.connect(self._on_chain_changed)
        self.velocity_path_edit.textChanged.connect(self._update_summary)
        self.output_dir_edit.textChanged.connect(self._update_output_preview)
        self.output_name_edit.textChanged.connect(self._update_output_preview)
        self.output_before_gathers_check.stateChanged.connect(self._update_output_preview)
        self.output_after_gathers_check.stateChanged.connect(self._update_output_preview)
        self.output_noise_gathers_check.stateChanged.connect(self._update_output_preview)
        self.output_before_stack_check.stateChanged.connect(self._update_output_preview)
        self.output_after_stack_check.stateChanged.connect(self._update_output_preview)
        self.output_noise_stack_check.stateChanged.connect(self._update_output_preview)
        self.output_difference_check.stateChanged.connect(self._update_output_preview)

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
        """Load and display dataset information."""
        try:
            dataset_path = Path(self.dataset_path)

            # Try to load metadata
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

                n_traces = metadata.get('n_traces', '?')
                n_samples = metadata.get('n_samples', '?')
                sort_key = metadata.get('sorted_by', 'unknown')

                # Get inline range from ensemble index if available
                ensemble_path = dataset_path / "ensemble_index.parquet"
                inline_range = "unknown"
                if ensemble_path.exists():
                    import pandas as pd
                    df = pd.read_parquet(ensemble_path)
                    if 'INLINE_NO' in df.columns:
                        inline_range = f"{df['INLINE_NO'].min()} - {df['INLINE_NO'].max()}"
                        self._dataset_info = {
                            'inline_min': int(df['INLINE_NO'].min()),
                            'inline_max': int(df['INLINE_NO'].max()),
                            'n_ensembles': len(df),
                        }

                self.dataset_info_label.setText(
                    f"Traces: {n_traces:,} | Samples: {n_samples}\n"
                    f"Sort: {sort_key} | Inline range: {inline_range}"
                )
            else:
                self.dataset_info_label.setText("Metadata not found")
                self._dataset_info = None

        except Exception as e:
            self.dataset_info_label.setText(f"Error: {e}")
            logger.exception("Error loading dataset info")

    def _load_velocity_preview(self):
        """Load and display velocity file preview."""
        try:
            from utils.velocity_io import preview_velocity_file

            filepath = self.velocity_path_edit.text()
            if not filepath or not Path(filepath).exists():
                self.velocity_preview.setText("No file selected or file not found")
                return

            info = preview_velocity_file(filepath)

            if info.is_valid:
                lines = [
                    f"Format: {info.format.value}",
                    f"Locations: {info.n_locations}",
                    f"Time samples: {info.n_time_samples}",
                    f"Time range: {info.time_range[0]:.3f} - {info.time_range[1]:.3f}",
                    f"Velocity range: {info.velocity_range[0]:.0f} - {info.velocity_range[1]:.0f}",
                ]
                if info.cdp_range:
                    lines.append(f"CDP range: {info.cdp_range[0]} - {info.cdp_range[1]}")
                self.velocity_preview.setText("\n".join(lines))
            else:
                self.velocity_preview.setText(f"Error: {info.error_message}")

        except Exception as e:
            self.velocity_preview.setText(f"Error: {e}")
            logger.exception("Error loading velocity preview")

    def _on_nmo_toggled(self, state: int):
        """Handle NMO checkbox toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.velocity_widget.setEnabled(enabled)
        self._update_summary()

    def _on_mute_toggled(self, state: int = None):
        """Handle mute checkbox toggle."""
        enabled = self.mute_top_check.isChecked() or self.mute_bottom_check.isChecked()
        self.mute_velocity_spin.setEnabled(enabled)
        self.mute_taper_spin.setEnabled(enabled)
        self.mute_target_combo.setEnabled(enabled)
        self._update_output_preview()

    def _on_chain_changed(self):
        """Handle processing chain change."""
        summary = self.chain_widget.get_chain_summary()
        self.chain_summary_label.setText(summary)
        self._update_summary()

    def _parse_inlines(self) -> List[int]:
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
        output_name = self.output_name_edit.text() or "qc_batch"

        if not output_dir:
            self.output_preview.setText("Select output directory")
            return

        files = []
        base_path = Path(output_dir) / output_name

        if self.output_before_gathers_check.isChecked():
            files.append(f"Before gathers: {output_name}_before_gathers.zarr")
        if self.output_after_gathers_check.isChecked():
            files.append(f"After gathers: {output_name}_after_gathers.zarr")
        if self.output_noise_gathers_check.isChecked():
            files.append(f"Noise gathers: {output_name}_noise_gathers.zarr")
        if self.output_before_stack_check.isChecked():
            files.append(f"Before stack: {output_name}_before_stack.zarr")
        if self.output_after_stack_check.isChecked():
            files.append(f"After stack: {output_name}_after_stack.zarr")
        if self.output_noise_stack_check.isChecked():
            files.append(f"Noise stack: {output_name}_noise_stack.zarr")
        if self.output_difference_check.isChecked():
            files.append(f"Difference: {output_name}_difference.zarr")

        files.append(f"Metadata: {output_name}_metadata.json")

        self.output_preview.setText("\n".join(files) if files else "No outputs selected")

    def _update_summary(self):
        """Update summary text."""
        inlines = self._parse_inlines()
        chain_summary = self.chain_widget.get_chain_summary()
        nmo_str = "Yes" if self.apply_nmo_check.isChecked() else "No"

        outputs = []
        if self.output_before_stack_check.isChecked():
            outputs.append("Before Stack")
        if self.output_after_stack_check.isChecked():
            outputs.append("After Stack")
        if self.output_noise_stack_check.isChecked():
            outputs.append("Noise Stack")
        if self.output_difference_check.isChecked():
            outputs.append("Difference")

        mute_str = ""
        if self.mute_top_check.isChecked() or self.mute_bottom_check.isChecked():
            mute_parts = []
            if self.mute_top_check.isChecked():
                mute_parts.append("Top")
            if self.mute_bottom_check.isChecked():
                mute_parts.append("Bottom")
            mute_str = f" | Mute: {'+'.join(mute_parts)} @ {self.mute_velocity_spin.value():.0f} m/s"

        summary = [
            f"Dataset: {Path(self.dataset_path).name if self.dataset_path else '(none)'}",
            f"Inlines: {len(inlines)} selected",
            f"Chain: {chain_summary}",
            f"NMO: {nmo_str}{mute_str}",
            f"Outputs: {', '.join(outputs) if outputs else '(none)'}",
        ]

        self.summary_text.setText("\n".join(summary))

    def _save_config(self):
        """Save configuration to file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                config = self.get_config()
                config.save_to_file(path)
                QMessageBox.information(self, "Success", f"Configuration saved to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _load_config(self):
        """Load configuration from file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                config = QCBatchConfig.load_from_file(path)
                self.set_config(config)
                QMessageBox.information(self, "Success", "Configuration loaded")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load: {e}")

    def _on_accept(self):
        """Handle OK button."""
        config = self.get_config()
        is_valid, error = config.validate()

        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error)
            return

        self.config_ready.emit(config)
        self.accept()

    def get_config(self) -> QCBatchConfig:
        """Get current configuration."""
        inlines = self._parse_inlines()
        chain_config = self.chain_widget.get_chain_config()

        # Map mute target combo index to string
        mute_target_map = {0: 'output', 1: 'input', 2: 'processed'}
        mute_target = mute_target_map.get(self.mute_target_combo.currentIndex(), 'output')

        return QCBatchConfig(
            dataset_path=self.dataset_path,
            inline_numbers=inlines,
            processing_chain=chain_config,
            apply_nmo=self.apply_nmo_check.isChecked(),
            velocity_file=self.velocity_path_edit.text(),
            velocity_type='rms' if self.velocity_type_combo.currentIndex() == 0 else 'interval',
            time_unit=self.time_unit_combo.currentText(),
            velocity_unit=self.velocity_unit_combo.currentText(),
            stretch_mute=self.stretch_mute_spin.value(),
            output_before_gathers=self.output_before_gathers_check.isChecked(),
            output_after_gathers=self.output_after_gathers_check.isChecked(),
            output_noise_gathers=self.output_noise_gathers_check.isChecked(),
            output_before_stack=self.output_before_stack_check.isChecked(),
            output_after_stack=self.output_after_stack_check.isChecked(),
            output_noise_stack=self.output_noise_stack_check.isChecked(),
            output_difference=self.output_difference_check.isChecked(),
            stack_method=self.stack_method_combo.currentText().lower(),
            min_fold=self.min_fold_spin.value(),
            mute_velocity=self.mute_velocity_spin.value() if (self.mute_top_check.isChecked() or self.mute_bottom_check.isChecked()) else 0.0,
            mute_top=self.mute_top_check.isChecked(),
            mute_bottom=self.mute_bottom_check.isChecked(),
            mute_taper=self.mute_taper_spin.value(),
            mute_target=mute_target,
            output_dir=self.output_dir_edit.text(),
            output_name=self.output_name_edit.text() or "qc_batch",
        )

    def set_config(self, config: QCBatchConfig):
        """Set dialog from configuration."""
        self.dataset_path = config.dataset_path
        self.dataset_path_edit.setText(config.dataset_path)

        if config.inline_numbers:
            self.inline_edit.setText(", ".join(str(i) for i in config.inline_numbers))

        if config.processing_chain:
            self.chain_widget.set_chain_config(config.processing_chain)

        self.apply_nmo_check.setChecked(config.apply_nmo)
        self.velocity_path_edit.setText(config.velocity_file)
        self.velocity_type_combo.setCurrentIndex(0 if config.velocity_type == 'rms' else 1)
        self.time_unit_combo.setCurrentText(config.time_unit)
        self.velocity_unit_combo.setCurrentText(config.velocity_unit)
        self.stretch_mute_spin.setValue(config.stretch_mute)

        self.output_before_gathers_check.setChecked(config.output_before_gathers)
        self.output_after_gathers_check.setChecked(config.output_after_gathers)
        self.output_noise_gathers_check.setChecked(config.output_noise_gathers)
        self.output_before_stack_check.setChecked(config.output_before_stack)
        self.output_after_stack_check.setChecked(config.output_after_stack)
        self.output_noise_stack_check.setChecked(config.output_noise_stack)
        self.output_difference_check.setChecked(config.output_difference)

        self.stack_method_combo.setCurrentText(config.stack_method.capitalize())
        self.min_fold_spin.setValue(config.min_fold)

        # Set mute options
        self.mute_top_check.setChecked(config.mute_top)
        self.mute_bottom_check.setChecked(config.mute_bottom)
        self.mute_velocity_spin.setValue(config.mute_velocity if config.mute_velocity > 0 else 2500)
        self.mute_taper_spin.setValue(config.mute_taper)
        # Map mute target string to combo index
        mute_target_idx = {'output': 0, 'input': 1, 'processed': 2}.get(config.mute_target, 0)
        self.mute_target_combo.setCurrentIndex(mute_target_idx)
        self._on_mute_toggled()

        self.output_dir_edit.setText(config.output_dir)
        self.output_name_edit.setText(config.output_name)

        self._load_dataset_info()
        self._parse_inlines()
        self._on_chain_changed()
        self._update_output_preview()
        self._update_summary()
