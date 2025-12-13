"""
QC Stack Viewer - Compare before/after stacks with difference display

Features:
- Three synchronized panels: Before, After, Difference
- Flip mode for rapid A/B comparison
- Statistics panel with RMS difference, correlation
- Inline navigation
- Amplitude normalization options

Usage:
    viewer = QCStackViewerWindow()
    viewer.load_before_stack(before_path)
    viewer.load_after_stack(after_path)
    viewer.show()
"""

import numpy as np
import zarr
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QSlider, QSpinBox,
    QDockWidget, QFileDialog, QMessageBox, QSplitter,
    QStatusBar, QToolBar, QFrame, QFormLayout, QDoubleSpinBox,
    QButtonGroup, QRadioButton, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

import pyqtgraph as pg

logger = logging.getLogger(__name__)


@dataclass
class StackData:
    """Container for stack data."""
    data: np.ndarray  # (n_samples, n_traces)
    sample_interval: float  # seconds
    cdp_numbers: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    path: Optional[str] = None

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_traces(self) -> int:
        return self.data.shape[1]

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.sample_interval


class StackPanel(QWidget):
    """
    Single stack display panel using PyQtGraph.
    """

    def __init__(self, title: str = "Stack", parent=None):
        super().__init__(parent)
        self.title = title
        self._data: Optional[np.ndarray] = None
        self._clip_percent = 99

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Image view
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        # Set colormap (seismic-like)
        colors = [
            (0, 0, 139),    # dark blue
            (0, 0, 255),    # blue
            (255, 255, 255),  # white
            (255, 0, 0),    # red
            (139, 0, 0),    # dark red
        ]
        cmap = pg.ColorMap(pos=np.linspace(0, 1, len(colors)), color=colors)
        self.image_view.setColorMap(cmap)

        layout.addWidget(self.image_view)

    def set_data(self, data: np.ndarray, clip_percent: float = 99):
        """Set display data with automatic clipping."""
        self._data = data
        self._clip_percent = clip_percent

        if data is None or data.size == 0:
            self.image_view.clear()
            return

        # Clip for display
        clip_val = np.percentile(np.abs(data), clip_percent)
        if clip_val > 0:
            display_data = np.clip(data, -clip_val, clip_val)
        else:
            display_data = data

        # PyQtGraph expects (x, y) so transpose
        self.image_view.setImage(display_data.T, autoRange=True, autoLevels=True)

    def clear(self):
        """Clear display."""
        self._data = None
        self.image_view.clear()

    def get_view_box(self) -> pg.ViewBox:
        """Get view box for synchronization."""
        return self.image_view.getView()


class QCStackViewerWindow(QMainWindow):
    """
    Window for comparing QC stacks (before/after/difference).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._before_data: Optional[StackData] = None
        self._after_data: Optional[StackData] = None
        self._difference: Optional[np.ndarray] = None

        self._flip_mode = False
        self._showing_before = True
        self._sync_enabled = True

        self._init_ui()
        self._create_actions()
        self._connect_signals()

    def _init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("QC Stack Viewer")
        self.setMinimumSize(1000, 600)
        self.resize(1400, 800)

        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Toolbar
        self._create_toolbar()

        # Display mode controls
        mode_layout = QHBoxLayout()

        self.mode_group = QButtonGroup()
        self.side_by_side_radio = QRadioButton("Side by Side")
        self.before_only_radio = QRadioButton("Before Only")
        self.after_only_radio = QRadioButton("After Only")
        self.diff_only_radio = QRadioButton("Difference Only")
        self.flip_radio = QRadioButton("Flip Mode")

        self.mode_group.addButton(self.side_by_side_radio, 0)
        self.mode_group.addButton(self.before_only_radio, 1)
        self.mode_group.addButton(self.after_only_radio, 2)
        self.mode_group.addButton(self.diff_only_radio, 3)
        self.mode_group.addButton(self.flip_radio, 4)

        self.side_by_side_radio.setChecked(True)

        mode_layout.addWidget(QLabel("Display:"))
        mode_layout.addWidget(self.side_by_side_radio)
        mode_layout.addWidget(self.before_only_radio)
        mode_layout.addWidget(self.after_only_radio)
        mode_layout.addWidget(self.diff_only_radio)
        mode_layout.addWidget(self.flip_radio)
        mode_layout.addStretch()

        layout.addLayout(mode_layout)

        # Main splitter with three panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.before_panel = StackPanel("BEFORE")
        self.after_panel = StackPanel("AFTER")
        self.diff_panel = StackPanel("DIFFERENCE")

        self.splitter.addWidget(self.before_panel)
        self.splitter.addWidget(self.after_panel)
        self.splitter.addWidget(self.diff_panel)

        layout.addWidget(self.splitter, 1)

        # Statistics dock
        self._create_stats_dock()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Flip timer
        self._flip_timer = QTimer()
        self._flip_timer.timeout.connect(self._do_flip)

    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        # Load buttons
        load_before_action = QAction("Load Before...", self)
        load_before_action.triggered.connect(self._load_before)
        toolbar.addAction(load_before_action)

        load_after_action = QAction("Load After...", self)
        load_after_action.triggered.connect(self._load_after)
        toolbar.addAction(load_after_action)

        toolbar.addSeparator()

        # Clip control
        toolbar.addWidget(QLabel("Clip %:"))
        self.clip_spin = QSpinBox()
        self.clip_spin.setRange(90, 100)
        self.clip_spin.setValue(99)
        self.clip_spin.valueChanged.connect(self._update_display)
        toolbar.addWidget(self.clip_spin)

        toolbar.addSeparator()

        # Flip button
        self.flip_button = QPushButton("Flip (Space)")
        self.flip_button.clicked.connect(self._do_flip)
        toolbar.addWidget(self.flip_button)

    def _create_actions(self):
        """Create keyboard shortcuts."""
        # Space for flip
        flip_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        flip_shortcut.activated.connect(self._do_flip)

        # Number keys for display modes
        QShortcut(QKeySequence("1"), self).activated.connect(
            lambda: self.side_by_side_radio.setChecked(True))
        QShortcut(QKeySequence("2"), self).activated.connect(
            lambda: self.before_only_radio.setChecked(True))
        QShortcut(QKeySequence("3"), self).activated.connect(
            lambda: self.after_only_radio.setChecked(True))
        QShortcut(QKeySequence("4"), self).activated.connect(
            lambda: self.diff_only_radio.setChecked(True))
        QShortcut(QKeySequence("5"), self).activated.connect(
            lambda: self.flip_radio.setChecked(True))

    def _create_stats_dock(self):
        """Create statistics dock widget."""
        dock = QDockWidget("Statistics", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)

        stats_widget = QWidget()
        stats_layout = QFormLayout(stats_widget)

        self.rms_before_label = QLabel("-")
        self.rms_after_label = QLabel("-")
        self.rms_diff_label = QLabel("-")
        self.correlation_label = QLabel("-")
        self.max_diff_label = QLabel("-")
        self.snr_improvement_label = QLabel("-")

        stats_layout.addRow("RMS Before:", self.rms_before_label)
        stats_layout.addRow("RMS After:", self.rms_after_label)
        stats_layout.addRow("RMS Difference:", self.rms_diff_label)
        stats_layout.addRow("Correlation:", self.correlation_label)
        stats_layout.addRow("Max |Diff|:", self.max_diff_label)
        stats_layout.addRow("SNR Improvement:", self.snr_improvement_label)

        dock.setWidget(stats_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _connect_signals(self):
        """Connect signals."""
        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        # Sync view boxes
        self.before_panel.get_view_box().sigRangeChanged.connect(self._sync_views)
        self.after_panel.get_view_box().sigRangeChanged.connect(self._sync_views)
        self.diff_panel.get_view_box().sigRangeChanged.connect(self._sync_views)

    def _load_before(self):
        """Load before stack."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Before Stack",
            "",
            "Zarr (*.zarr);;All Files (*)"
        )
        if path:
            self.load_before_stack(path)

    def _load_after(self):
        """Load after stack."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load After Stack",
            "",
            "Zarr (*.zarr);;All Files (*)"
        )
        if path:
            self.load_after_stack(path)

    def load_before_stack(self, path: str):
        """Load before stack from file."""
        try:
            self._before_data = self._load_stack(path)
            self._compute_difference()
            self._update_display()
            self._update_statistics()
            self.status_bar.showMessage(f"Loaded before: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            logger.exception("Load before failed")

    def load_after_stack(self, path: str):
        """Load after stack from file."""
        try:
            self._after_data = self._load_stack(path)
            self._compute_difference()
            self._update_display()
            self._update_statistics()
            self.status_bar.showMessage(f"Loaded after: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            logger.exception("Load after failed")

    def _load_stack(self, path: str) -> StackData:
        """Load stack data from Zarr file."""
        path = Path(path)

        # Handle directory or file
        if path.is_dir():
            zarr_path = path
        else:
            zarr_path = path

        data = np.array(zarr.open(str(zarr_path), mode='r'))

        # Try to load metadata
        metadata = {}
        sample_interval = 0.004
        cdp_numbers = None

        # Look for metadata file
        meta_paths = [
            path.parent / f"{path.stem}_metadata.json",
            path.with_suffix('.json'),
            path.parent / "metadata.json",
        ]

        for meta_path in meta_paths:
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                sample_interval = metadata.get('sample_interval', 0.004)
                if 'cdp_numbers' in metadata:
                    cdp_numbers = np.array(metadata['cdp_numbers'])
                break

        return StackData(
            data=data,
            sample_interval=sample_interval,
            cdp_numbers=cdp_numbers,
            metadata=metadata,
            path=str(path)
        )

    def _compute_difference(self):
        """Compute difference between before and after."""
        if self._before_data is None or self._after_data is None:
            self._difference = None
            return

        # Check shapes match
        if self._before_data.data.shape != self._after_data.data.shape:
            logger.warning(
                f"Shape mismatch: before {self._before_data.data.shape} "
                f"vs after {self._after_data.data.shape}"
            )
            # Try to use common size
            min_samples = min(self._before_data.n_samples, self._after_data.n_samples)
            min_traces = min(self._before_data.n_traces, self._after_data.n_traces)

            before = self._before_data.data[:min_samples, :min_traces]
            after = self._after_data.data[:min_samples, :min_traces]
        else:
            before = self._before_data.data
            after = self._after_data.data

        self._difference = before - after

    def _update_display(self):
        """Update panel display based on mode."""
        clip = self.clip_spin.value()

        mode = self.mode_group.checkedId()

        if mode == 0:  # Side by side
            self.before_panel.setVisible(True)
            self.after_panel.setVisible(True)
            self.diff_panel.setVisible(True)
        elif mode == 1:  # Before only
            self.before_panel.setVisible(True)
            self.after_panel.setVisible(False)
            self.diff_panel.setVisible(False)
        elif mode == 2:  # After only
            self.before_panel.setVisible(False)
            self.after_panel.setVisible(True)
            self.diff_panel.setVisible(False)
        elif mode == 3:  # Diff only
            self.before_panel.setVisible(False)
            self.after_panel.setVisible(False)
            self.diff_panel.setVisible(True)
        elif mode == 4:  # Flip mode
            self.before_panel.setVisible(True)
            self.after_panel.setVisible(False)
            self.diff_panel.setVisible(False)
            self._flip_mode = True

        # Set data
        if self._before_data is not None:
            self.before_panel.set_data(self._before_data.data, clip)

        if self._after_data is not None:
            self.after_panel.set_data(self._after_data.data, clip)

        if self._difference is not None:
            self.diff_panel.set_data(self._difference, clip)

    def _on_mode_changed(self):
        """Handle display mode change."""
        mode = self.mode_group.checkedId()
        self._flip_mode = (mode == 4)
        self._update_display()

    def _do_flip(self):
        """Flip between before and after in flip mode."""
        if not self._flip_mode:
            return

        self._showing_before = not self._showing_before

        if self._showing_before:
            if self._before_data is not None:
                self.before_panel.set_data(self._before_data.data, self.clip_spin.value())
            self.before_panel.title = "BEFORE"
        else:
            if self._after_data is not None:
                self.before_panel.set_data(self._after_data.data, self.clip_spin.value())
            self.before_panel.title = "AFTER"

    def _sync_views(self, view_box):
        """Synchronize view ranges across panels."""
        if not self._sync_enabled:
            return

        self._sync_enabled = False

        rect = view_box.viewRect()

        for panel in [self.before_panel, self.after_panel, self.diff_panel]:
            if panel.get_view_box() is not view_box:
                panel.get_view_box().setRange(rect, padding=0)

        self._sync_enabled = True

    def _update_statistics(self):
        """Update statistics display."""
        if self._before_data is not None:
            rms_before = np.sqrt(np.mean(self._before_data.data**2))
            self.rms_before_label.setText(f"{rms_before:.2e}")
        else:
            self.rms_before_label.setText("-")

        if self._after_data is not None:
            rms_after = np.sqrt(np.mean(self._after_data.data**2))
            self.rms_after_label.setText(f"{rms_after:.2e}")
        else:
            self.rms_after_label.setText("-")

        if self._difference is not None:
            rms_diff = np.sqrt(np.mean(self._difference**2))
            max_diff = np.max(np.abs(self._difference))

            self.rms_diff_label.setText(f"{rms_diff:.2e}")
            self.max_diff_label.setText(f"{max_diff:.2e}")

            # Correlation
            if self._before_data is not None and self._after_data is not None:
                before_flat = self._before_data.data.flatten()
                after_flat = self._after_data.data.flatten()
                correlation = np.corrcoef(before_flat, after_flat)[0, 1]
                self.correlation_label.setText(f"{correlation:.4f}")

                # SNR improvement estimate (rough)
                signal_power = np.mean(self._after_data.data**2)
                noise_power = np.mean(self._difference**2)
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                    self.snr_improvement_label.setText(f"{snr_db:.1f} dB")
                else:
                    self.snr_improvement_label.setText("N/A")
        else:
            self.rms_diff_label.setText("-")
            self.correlation_label.setText("-")
            self.max_diff_label.setText("-")
            self.snr_improvement_label.setText("-")


# =============================================================================
# Convenience function
# =============================================================================

def show_qc_stack_viewer(before_path: Optional[str] = None, after_path: Optional[str] = None):
    """
    Show QC stack viewer with optional initial data.

    Args:
        before_path: Path to before stack
        after_path: Path to after stack

    Returns:
        QCStackViewerWindow instance
    """
    viewer = QCStackViewerWindow()

    if before_path:
        viewer.load_before_stack(before_path)
    if after_path:
        viewer.load_after_stack(after_path)

    viewer.show()
    return viewer
