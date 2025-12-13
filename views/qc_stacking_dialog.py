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
from typing import Optional, Dict, Any, List, Tuple
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
        """Create velocity configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection
        file_group = QGroupBox("Velocity File")
        file_layout = QFormLayout(file_group)

        self.velocity_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_velocity)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.velocity_path_edit)
        path_layout.addWidget(browse_btn)
        file_layout.addRow("File:", path_layout)

        # File type
        self.velocity_type_combo = QComboBox()
        self.velocity_type_combo.addItems(["RMS Velocity", "Interval Velocity"])
        file_layout.addRow("Velocity Type:", self.velocity_type_combo)

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

        file_layout.addRow("Units:", units_layout)

        layout.addWidget(file_group)

        # Preview
        preview_group = QGroupBox("Velocity Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.velocity_preview = QTextEdit()
        self.velocity_preview.setReadOnly(True)
        self.velocity_preview.setMaximumHeight(150)
        self.velocity_preview.setStyleSheet("font-family: monospace;")
        preview_layout.addWidget(self.velocity_preview)

        preview_btn = QPushButton("Load Preview")
        preview_btn.clicked.connect(self._load_velocity_preview)
        preview_layout.addWidget(preview_btn)

        layout.addWidget(preview_group)
        layout.addStretch()

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
                import json
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
            from utils.velocity_io import preview_velocity_file, get_velocity_summary

            filepath = self.velocity_path_edit.text()
            if not filepath or not Path(filepath).exists():
                self.velocity_preview.setText("No file selected or file not found")
                return

            info = preview_velocity_file(filepath)
            self._velocity_info = info

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

        self._load_dataset_info()
        self._parse_inlines()
        self._update_summary()
