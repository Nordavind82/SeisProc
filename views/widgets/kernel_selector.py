"""
Kernel Selector Widget

Provides UI components for selecting kernel backend (Python vs Metal C++).
Can be embedded in processor dialogs or settings.
"""

import logging
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette

logger = logging.getLogger(__name__)

# Try to import kernel backend
try:
    from processors.kernel_backend import (
        KernelBackend, get_backend_info, set_global_backend,
        get_global_backend, is_metal_available
    )
    KERNEL_BACKEND_AVAILABLE = True
except ImportError:
    KERNEL_BACKEND_AVAILABLE = False
    logger.debug("Kernel backend module not available")


class KernelBackendCombo(QComboBox):
    """
    Combo box for selecting kernel backend.

    Emits backendChanged signal when selection changes.
    """
    backendChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_items()
        self.currentTextChanged.connect(self._on_selection_changed)

    def _setup_items(self):
        """Setup combo box items."""
        self.addItem("Auto (Recommended)", "auto")
        self.addItem("Python (NumPy/PyWavelets)", "python")

        if KERNEL_BACKEND_AVAILABLE and is_metal_available():
            self.addItem("Metal C++ (GPU Accelerated)", "metal_cpp")
        else:
            # Add disabled Metal option with tooltip
            self.addItem("Metal C++ (Not Available)", "metal_cpp_disabled")
            # Disable the last item
            model = self.model()
            item = model.item(self.count() - 1)
            item.setEnabled(False)

    def _on_selection_changed(self, text: str):
        """Handle selection change."""
        backend = self.currentData()
        if backend and backend != "metal_cpp_disabled":
            self.backendChanged.emit(backend)

    def get_backend(self) -> Optional[str]:
        """Get current backend selection."""
        data = self.currentData()
        if data == "metal_cpp_disabled":
            return "auto"
        return data

    def set_backend(self, backend: str):
        """Set backend selection."""
        for i in range(self.count()):
            if self.itemData(i) == backend:
                self.setCurrentIndex(i)
                return
        # Default to auto
        self.setCurrentIndex(0)


class KernelSelectorWidget(QWidget):
    """
    Complete kernel selector widget with status display.

    Shows:
    - Backend selector combo
    - Current status (availability, device info)
    - Refresh button

    Emits backendChanged signal when backend selection changes.
    """
    backendChanged = pyqtSignal(str)

    def __init__(self, parent=None, show_status: bool = True, compact: bool = False):
        """
        Initialize kernel selector widget.

        Args:
            parent: Parent widget
            show_status: Show status information
            compact: Use compact layout
        """
        super().__init__(parent)
        self.show_status = show_status
        self.compact = compact
        self._setup_ui()
        self._update_status()

    def _setup_ui(self):
        """Setup widget UI."""
        if self.compact:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            label = QLabel("Kernel:")
            layout.addWidget(label)

            self.combo = KernelBackendCombo()
            self.combo.backendChanged.connect(self._on_backend_changed)
            layout.addWidget(self.combo)

            if self.show_status:
                self.status_label = QLabel()
                self.status_label.setStyleSheet("color: gray; font-size: 10px;")
                layout.addWidget(self.status_label)

            layout.addStretch()
        else:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            # Group box
            group = QGroupBox("Kernel Backend")
            group_layout = QVBoxLayout(group)

            # Backend selector row
            selector_layout = QHBoxLayout()
            selector_layout.addWidget(QLabel("Backend:"))

            self.combo = KernelBackendCombo()
            self.combo.backendChanged.connect(self._on_backend_changed)
            selector_layout.addWidget(self.combo, 1)

            refresh_btn = QPushButton("Refresh")
            refresh_btn.setToolTip("Check backend availability")
            refresh_btn.clicked.connect(self._update_status)
            selector_layout.addWidget(refresh_btn)

            group_layout.addLayout(selector_layout)

            if self.show_status:
                # Status display
                status_frame = QFrame()
                status_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
                status_layout = QVBoxLayout(status_frame)
                status_layout.setContentsMargins(8, 4, 8, 4)

                self.status_label = QLabel()
                self.status_label.setWordWrap(True)
                status_layout.addWidget(self.status_label)

                group_layout.addWidget(status_frame)

            layout.addWidget(group)

    def _on_backend_changed(self, backend: str):
        """Handle backend selection change."""
        if KERNEL_BACKEND_AVAILABLE:
            try:
                backend_enum = KernelBackend(backend)
                set_global_backend(backend_enum)
                logger.info(f"Kernel backend changed to: {backend}")
            except ValueError:
                pass

        self._update_status()
        self.backendChanged.emit(backend)

    def _update_status(self):
        """Update status display."""
        if not self.show_status:
            return

        if not KERNEL_BACKEND_AVAILABLE:
            self.status_label.setText("Kernel backend module not available")
            self.status_label.setStyleSheet("color: orange;")
            return

        info = get_backend_info()
        lines = []

        # Effective backend
        effective = info.get('effective_backend', 'unknown')
        if effective == 'metal_cpp':
            lines.append(f"Active: Metal C++ (GPU)")
            device = info.get('metal_device', 'Unknown')
            lines.append(f"Device: {device}")
            color = "green"
        elif effective == 'python':
            lines.append("Active: Python (CPU)")
            color = "blue"
        else:
            lines.append(f"Active: {effective}")
            color = "gray"

        # Availability
        metal_available = info.get('metal_cpp_available', False)
        if metal_available:
            lines.append("Metal C++: Available")
        else:
            lines.append("Metal C++: Not compiled")

        self.status_label.setText("\n".join(lines))
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def get_backend(self) -> str:
        """Get current backend selection."""
        return self.combo.get_backend()

    def set_backend(self, backend: str):
        """Set backend selection."""
        self.combo.set_backend(backend)

    def get_kernel_backend_enum(self):
        """Get KernelBackend enum value for current selection."""
        if not KERNEL_BACKEND_AVAILABLE:
            return None

        backend = self.get_backend()
        try:
            return KernelBackend(backend)
        except ValueError:
            return KernelBackend.AUTO


class ProcessorKernelMixin:
    """
    Mixin class for processor dialogs to add kernel selection.

    Usage:
        class MyProcessorDialog(QDialog, ProcessorKernelMixin):
            def __init__(self):
                super().__init__()
                # In your layout setup:
                self.add_kernel_selector(layout)

            def get_processor(self):
                params = {...}
                params['backend'] = self.get_kernel_backend()
                return MyProcessor(**params)
    """

    def add_kernel_selector(
        self,
        layout,
        compact: bool = True,
        show_status: bool = True
    ):
        """
        Add kernel selector to layout.

        Args:
            layout: QLayout to add selector to
            compact: Use compact layout
            show_status: Show status information
        """
        self._kernel_selector = KernelSelectorWidget(
            parent=self,
            show_status=show_status,
            compact=compact
        )
        layout.addWidget(self._kernel_selector)

    def get_kernel_backend(self) -> str:
        """Get selected kernel backend."""
        if hasattr(self, '_kernel_selector'):
            return self._kernel_selector.get_backend()
        return "auto"

    def get_kernel_backend_enum(self):
        """Get KernelBackend enum for selected backend."""
        if hasattr(self, '_kernel_selector'):
            return self._kernel_selector.get_kernel_backend_enum()
        return None


def create_kernel_info_label() -> QLabel:
    """
    Create a label showing kernel backend info.

    Useful for status bars or info panels.
    """
    label = QLabel()
    label.setToolTip("Kernel backend status")

    if not KERNEL_BACKEND_AVAILABLE:
        label.setText("Kernels: N/A")
        label.setStyleSheet("color: gray;")
        return label

    info = get_backend_info()
    effective = info.get('effective_backend', 'unknown')

    if effective == 'metal_cpp':
        device = info.get('metal_device', 'GPU')
        label.setText(f"Metal ({device})")
        label.setStyleSheet("color: green;")
    elif effective == 'python':
        label.setText("Python (CPU)")
        label.setStyleSheet("color: blue;")
    else:
        label.setText(f"Kernel: {effective}")

    return label
