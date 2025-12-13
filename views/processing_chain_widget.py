"""
Processing Chain Widget - Build and configure processor chains

Features:
- List of available processors
- Drag-drop ordering
- Parameter editor for each processor
- Chain preview/summary
- Save/load chain configurations

Usage:
    widget = ProcessingChainWidget()
    widget.chain_changed.connect(on_chain_changed)

    # Get configured chain
    processors = widget.get_processor_chain()
"""

import logging
from typing import Optional, List, Dict, Any, Type
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QListWidget, QListWidgetItem, QComboBox,
    QStackedWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QFrame, QSplitter, QMessageBox,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDrag, QIcon

from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


@dataclass
class ProcessorInfo:
    """Information about an available processor."""
    name: str
    display_name: str
    description: str
    processor_class: Type[BaseProcessor]
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_specs: List[Dict[str, Any]] = field(default_factory=list)


# Registry of available processors for the chain widget
CHAIN_PROCESSORS: List[ProcessorInfo] = []


def register_chain_processor(
    name: str,
    display_name: str,
    description: str,
    processor_class: Type[BaseProcessor],
    default_params: Dict[str, Any] = None,
    param_specs: List[Dict[str, Any]] = None
):
    """Register a processor for use in the chain widget."""
    CHAIN_PROCESSORS.append(ProcessorInfo(
        name=name,
        display_name=display_name,
        description=description,
        processor_class=processor_class,
        default_params=default_params or {},
        param_specs=param_specs or []
    ))


def _initialize_chain_processors():
    """Initialize default chain processors."""
    if CHAIN_PROCESSORS:
        return  # Already initialized

    from processors import (
        BandpassFilter, FKFilter, GainProcessor,
        TFDenoise, STFTDenoise, DWTDenoise
    )

    # Bandpass Filter
    register_chain_processor(
        name='bandpass',
        display_name='Bandpass Filter',
        description='Frequency domain bandpass filter',
        processor_class=BandpassFilter,
        default_params={'low_cut': 5.0, 'high_cut': 80.0, 'order': 4},
        param_specs=[
            {'name': 'low_cut', 'type': 'float', 'min': 0, 'max': 500, 'default': 5.0, 'label': 'Low Cut (Hz)'},
            {'name': 'high_cut', 'type': 'float', 'min': 1, 'max': 500, 'default': 80.0, 'label': 'High Cut (Hz)'},
            {'name': 'order', 'type': 'int', 'min': 1, 'max': 10, 'default': 4, 'label': 'Filter Order'},
        ]
    )

    # FK Filter
    register_chain_processor(
        name='fk_filter',
        display_name='FK Filter',
        description='Frequency-wavenumber domain filter',
        processor_class=FKFilter,
        default_params={'velocity_min': 1500, 'velocity_max': 6000, 'mode': 'pass'},
        param_specs=[
            {'name': 'velocity_min', 'type': 'float', 'min': 100, 'max': 10000, 'default': 1500, 'label': 'Min Velocity (m/s)'},
            {'name': 'velocity_max', 'type': 'float', 'min': 100, 'max': 10000, 'default': 6000, 'label': 'Max Velocity (m/s)'},
            {'name': 'mode', 'type': 'choice', 'choices': ['pass', 'reject'], 'default': 'pass', 'label': 'Mode'},
        ]
    )

    # Gain
    register_chain_processor(
        name='gain',
        display_name='Gain/AGC',
        description='Amplitude gain control',
        processor_class=GainProcessor,
        default_params={'gain_type': 'agc', 'window_ms': 500},
        param_specs=[
            {'name': 'gain_type', 'type': 'choice', 'choices': ['agc', 'spherical', 'exponential'], 'default': 'agc', 'label': 'Gain Type'},
            {'name': 'window_ms', 'type': 'float', 'min': 50, 'max': 2000, 'default': 500, 'label': 'Window (ms)'},
        ]
    )

    # TF Denoise
    register_chain_processor(
        name='tf_denoise',
        display_name='TF Denoise',
        description='Time-frequency domain denoising',
        processor_class=TFDenoise,
        default_params={'threshold': 3.0, 'method': 'soft'},
        param_specs=[
            {'name': 'threshold', 'type': 'float', 'min': 0.5, 'max': 10, 'default': 3.0, 'label': 'Threshold (σ)'},
            {'name': 'method', 'type': 'choice', 'choices': ['soft', 'hard'], 'default': 'soft', 'label': 'Threshold Method'},
        ]
    )

    # STFT Denoise
    register_chain_processor(
        name='stft_denoise',
        display_name='STFT Denoise',
        description='Short-time Fourier transform denoising',
        processor_class=STFTDenoise,
        default_params={'threshold': 3.0, 'window_size': 64},
        param_specs=[
            {'name': 'threshold', 'type': 'float', 'min': 0.5, 'max': 10, 'default': 3.0, 'label': 'Threshold (σ)'},
            {'name': 'window_size', 'type': 'int', 'min': 16, 'max': 256, 'default': 64, 'label': 'Window Size'},
        ]
    )


class ProcessorParamEditor(QWidget):
    """
    Parameter editor for a single processor.
    """

    params_changed = pyqtSignal()

    def __init__(self, processor_info: ProcessorInfo, parent=None):
        super().__init__(parent)
        self.processor_info = processor_info
        self._param_widgets: Dict[str, QWidget] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel(f"<b>{self.processor_info.display_name}</b>")
        layout.addRow(title)

        # Description
        desc = QLabel(self.processor_info.description)
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addRow(desc)

        # Parameters
        for spec in self.processor_info.param_specs:
            widget = self._create_param_widget(spec)
            self._param_widgets[spec['name']] = widget
            layout.addRow(spec['label'] + ":", widget)

    def _create_param_widget(self, spec: Dict[str, Any]) -> QWidget:
        """Create appropriate widget for parameter type."""
        param_type = spec.get('type', 'float')
        default = spec.get('default', 0)

        if param_type == 'int':
            widget = QSpinBox()
            widget.setRange(spec.get('min', 0), spec.get('max', 100))
            widget.setValue(int(default))
            widget.valueChanged.connect(self.params_changed.emit)

        elif param_type == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(spec.get('min', 0), spec.get('max', 100))
            widget.setDecimals(2)
            widget.setValue(float(default))
            widget.valueChanged.connect(self.params_changed.emit)

        elif param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(bool(default))
            widget.stateChanged.connect(self.params_changed.emit)

        elif param_type == 'choice':
            widget = QComboBox()
            widget.addItems(spec.get('choices', []))
            if default in spec.get('choices', []):
                widget.setCurrentText(str(default))
            widget.currentIndexChanged.connect(self.params_changed.emit)

        else:
            widget = QLineEdit()
            widget.setText(str(default))
            widget.textChanged.connect(self.params_changed.emit)

        return widget

    def get_params(self) -> Dict[str, Any]:
        """Get current parameter values."""
        params = {}
        for spec in self.processor_info.param_specs:
            name = spec['name']
            widget = self._param_widgets.get(name)
            if widget is None:
                continue

            if isinstance(widget, QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                params[name] = widget.text()

        return params

    def set_params(self, params: Dict[str, Any]):
        """Set parameter values."""
        for name, value in params.items():
            widget = self._param_widgets.get(name)
            if widget is None:
                continue

            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))


class ChainItemWidget(QFrame):
    """
    Widget representing a single processor in the chain.
    """

    remove_requested = pyqtSignal(object)  # self
    move_up_requested = pyqtSignal(object)
    move_down_requested = pyqtSignal(object)

    def __init__(self, processor_info: ProcessorInfo, index: int, parent=None):
        super().__init__(parent)
        self.processor_info = processor_info
        self.index = index

        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ChainItemWidget {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin: 2px;
            }
            ChainItemWidget:hover {
                background-color: #e8f0ff;
                border-color: #4a90d9;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Index label
        self.index_label = QLabel(f"{self.index + 1}.")
        self.index_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.index_label)

        # Name
        name_label = QLabel(self.processor_info.display_name)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        layout.addStretch()

        # Move buttons
        up_btn = QPushButton("▲")
        up_btn.setFixedSize(24, 24)
        up_btn.setToolTip("Move up")
        up_btn.clicked.connect(lambda: self.move_up_requested.emit(self))
        layout.addWidget(up_btn)

        down_btn = QPushButton("▼")
        down_btn.setFixedSize(24, 24)
        down_btn.setToolTip("Move down")
        down_btn.clicked.connect(lambda: self.move_down_requested.emit(self))
        layout.addWidget(down_btn)

        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setToolTip("Remove")
        remove_btn.setStyleSheet("color: red;")
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(remove_btn)

    def update_index(self, new_index: int):
        """Update displayed index."""
        self.index = new_index
        self.index_label.setText(f"{new_index + 1}.")


@dataclass
class ChainItem:
    """Item in the processing chain."""
    processor_info: ProcessorInfo
    params: Dict[str, Any] = field(default_factory=dict)
    widget: Optional[ChainItemWidget] = None
    editor: Optional[ProcessorParamEditor] = None


class ProcessingChainWidget(QWidget):
    """
    Widget for building and configuring a processing chain.

    Provides:
    - Available processors list
    - Chain list with drag-drop ordering
    - Parameter editor for selected processor
    - Chain summary
    """

    chain_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        _initialize_chain_processors()

        self._chain: List[ChainItem] = []
        self._selected_index: int = -1

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left: Available processors
        left_panel = QGroupBox("Available Processors")
        left_layout = QVBoxLayout(left_panel)

        self.available_list = QListWidget()
        for proc_info in CHAIN_PROCESSORS:
            item = QListWidgetItem(proc_info.display_name)
            item.setData(Qt.ItemDataRole.UserRole, proc_info)
            item.setToolTip(proc_info.description)
            self.available_list.addItem(item)
        left_layout.addWidget(self.available_list)

        add_btn = QPushButton("Add to Chain →")
        add_btn.clicked.connect(self._add_selected_processor)
        left_layout.addWidget(add_btn)

        left_panel.setMaximumWidth(200)
        layout.addWidget(left_panel)

        # Middle: Chain
        middle_panel = QGroupBox("Processing Chain")
        middle_layout = QVBoxLayout(middle_panel)

        # Chain list container
        self.chain_container = QWidget()
        self.chain_layout = QVBoxLayout(self.chain_container)
        self.chain_layout.setContentsMargins(0, 0, 0, 0)
        self.chain_layout.setSpacing(2)
        self.chain_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self.chain_container)
        scroll.setWidgetResizable(True)
        middle_layout.addWidget(scroll)

        # Clear button
        clear_btn = QPushButton("Clear Chain")
        clear_btn.clicked.connect(self._clear_chain)
        middle_layout.addWidget(clear_btn)

        layout.addWidget(middle_panel)

        # Right: Parameter editor
        right_panel = QGroupBox("Parameters")
        right_layout = QVBoxLayout(right_panel)

        self.param_stack = QStackedWidget()

        # Empty placeholder
        empty_label = QLabel("Select a processor to edit parameters")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_label.setStyleSheet("color: #888;")
        self.param_stack.addWidget(empty_label)

        right_layout.addWidget(self.param_stack)
        right_panel.setMinimumWidth(250)
        layout.addWidget(right_panel)

    def _add_selected_processor(self):
        """Add selected processor to chain."""
        current = self.available_list.currentItem()
        if current is None:
            return

        proc_info = current.data(Qt.ItemDataRole.UserRole)
        self._add_processor(proc_info)

    def _add_processor(self, proc_info: ProcessorInfo):
        """Add a processor to the chain."""
        index = len(self._chain)

        # Create chain item
        item = ChainItem(
            processor_info=proc_info,
            params=proc_info.default_params.copy()
        )

        # Create widget
        widget = ChainItemWidget(proc_info, index)
        widget.remove_requested.connect(self._on_remove_requested)
        widget.move_up_requested.connect(self._on_move_up)
        widget.move_down_requested.connect(self._on_move_down)
        widget.mousePressEvent = lambda e: self._select_item(index)
        item.widget = widget

        # Create parameter editor
        editor = ProcessorParamEditor(proc_info)
        editor.params_changed.connect(lambda: self._on_params_changed(item))
        item.editor = editor
        self.param_stack.addWidget(editor)

        # Add to chain
        self._chain.append(item)

        # Add widget to layout (before stretch)
        self.chain_layout.insertWidget(self.chain_layout.count() - 1, widget)

        # Select the new item
        self._select_item(index)

        self.chain_changed.emit()

    def _remove_processor(self, index: int):
        """Remove processor at index."""
        if index < 0 or index >= len(self._chain):
            return

        item = self._chain.pop(index)

        # Remove widget
        if item.widget:
            self.chain_layout.removeWidget(item.widget)
            item.widget.deleteLater()

        # Remove editor
        if item.editor:
            self.param_stack.removeWidget(item.editor)
            item.editor.deleteLater()

        # Update indices
        self._update_indices()

        # Update selection
        if self._selected_index >= len(self._chain):
            self._selected_index = len(self._chain) - 1
        if self._selected_index >= 0:
            self._select_item(self._selected_index)
        else:
            self.param_stack.setCurrentIndex(0)

        self.chain_changed.emit()

    def _move_processor(self, from_idx: int, to_idx: int):
        """Move processor from one index to another."""
        if from_idx < 0 or from_idx >= len(self._chain):
            return
        if to_idx < 0 or to_idx >= len(self._chain):
            return
        if from_idx == to_idx:
            return

        item = self._chain.pop(from_idx)
        self._chain.insert(to_idx, item)

        # Rebuild layout
        self._rebuild_chain_layout()

        # Update selection
        self._select_item(to_idx)

        self.chain_changed.emit()

    def _rebuild_chain_layout(self):
        """Rebuild the chain layout from current chain state."""
        # Remove all widgets
        while self.chain_layout.count() > 1:  # Keep stretch
            item = self.chain_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Re-add in order
        for i, chain_item in enumerate(self._chain):
            chain_item.widget.update_index(i)
            self.chain_layout.insertWidget(i, chain_item.widget)

    def _update_indices(self):
        """Update indices of all chain items."""
        for i, item in enumerate(self._chain):
            if item.widget:
                item.widget.update_index(i)

    def _select_item(self, index: int):
        """Select chain item at index."""
        self._selected_index = index

        if index < 0 or index >= len(self._chain):
            self.param_stack.setCurrentIndex(0)
            return

        item = self._chain[index]
        if item.editor:
            self.param_stack.setCurrentWidget(item.editor)

    def _on_remove_requested(self, widget: ChainItemWidget):
        """Handle remove request from widget."""
        for i, item in enumerate(self._chain):
            if item.widget is widget:
                self._remove_processor(i)
                break

    def _on_move_up(self, widget: ChainItemWidget):
        """Handle move up request."""
        for i, item in enumerate(self._chain):
            if item.widget is widget and i > 0:
                self._move_processor(i, i - 1)
                break

    def _on_move_down(self, widget: ChainItemWidget):
        """Handle move down request."""
        for i, item in enumerate(self._chain):
            if item.widget is widget and i < len(self._chain) - 1:
                self._move_processor(i, i + 1)
                break

    def _on_params_changed(self, item: ChainItem):
        """Handle parameter change for an item."""
        if item.editor:
            item.params = item.editor.get_params()
        self.chain_changed.emit()

    def _clear_chain(self):
        """Clear all processors from chain."""
        while self._chain:
            self._remove_processor(0)

    def get_processor_chain(self) -> List[BaseProcessor]:
        """
        Create processor instances from chain configuration.

        Returns:
            List of configured BaseProcessor instances
        """
        processors = []
        for item in self._chain:
            try:
                # Update params from editor
                if item.editor:
                    item.params = item.editor.get_params()

                # Create processor instance
                proc = item.processor_info.processor_class(**item.params)
                processors.append(proc)
            except Exception as e:
                logger.warning(f"Failed to create {item.processor_info.name}: {e}")

        return processors

    def get_chain_config(self) -> List[Dict[str, Any]]:
        """
        Get chain configuration as serializable dicts.

        Returns:
            List of processor configs
        """
        configs = []
        for item in self._chain:
            if item.editor:
                item.params = item.editor.get_params()

            configs.append({
                'name': item.processor_info.name,
                'display_name': item.processor_info.display_name,
                'params': item.params.copy()
            })
        return configs

    def set_chain_config(self, configs: List[Dict[str, Any]]):
        """
        Set chain from configuration.

        Args:
            configs: List of processor configs from get_chain_config()
        """
        self._clear_chain()

        for config in configs:
            name = config.get('name')
            params = config.get('params', {})

            # Find processor info
            proc_info = None
            for p in CHAIN_PROCESSORS:
                if p.name == name:
                    proc_info = p
                    break

            if proc_info:
                self._add_processor(proc_info)
                # Set params
                if self._chain and self._chain[-1].editor:
                    self._chain[-1].editor.set_params(params)
                    self._chain[-1].params = params

    def get_chain_summary(self) -> str:
        """Get human-readable chain summary."""
        if not self._chain:
            return "No processors configured"

        lines = [f"{i+1}. {item.processor_info.display_name}"
                 for i, item in enumerate(self._chain)]
        return " → ".join(lines)

    def is_empty(self) -> bool:
        """Check if chain is empty."""
        return len(self._chain) == 0
