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
    inline_numbers: Optional[list] = None  # List of inline numbers in the stack
    inline_ranges: Optional[Dict[int, Tuple[int, int]]] = None  # inline -> (start_idx, end_idx)
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

    def get_inline_data(self, inline_num: int) -> Optional[np.ndarray]:
        """Get data for a specific inline."""
        if self.inline_ranges and inline_num in self.inline_ranges:
            start, end = self.inline_ranges[inline_num]
            return self.data[:, start:end]
        return None


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
        self._qc_stack_data: Optional[StackData] = None  # Single QC stack data
        self._difference: Optional[np.ndarray] = None

        self._flip_mode = False
        self._showing_before = True
        self._sync_enabled = True
        self._current_inline_idx = 0

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

        # Load QC Stack button (primary action)
        load_qc_action = QAction("Load QC Stack...", self)
        load_qc_action.triggered.connect(self._load_qc_stack)
        toolbar.addAction(load_qc_action)

        toolbar.addSeparator()

        # Inline navigation
        toolbar.addWidget(QLabel("Inline:"))
        self.inline_combo = QComboBox()
        self.inline_combo.setMinimumWidth(100)
        self.inline_combo.currentIndexChanged.connect(self._on_inline_changed)
        toolbar.addWidget(self.inline_combo)

        self.prev_inline_btn = QPushButton("<")
        self.prev_inline_btn.setMaximumWidth(30)
        self.prev_inline_btn.clicked.connect(self._prev_inline)
        toolbar.addWidget(self.prev_inline_btn)

        self.next_inline_btn = QPushButton(">")
        self.next_inline_btn.setMaximumWidth(30)
        self.next_inline_btn.clicked.connect(self._next_inline)
        toolbar.addWidget(self.next_inline_btn)

        self.inline_info_label = QLabel("")
        toolbar.addWidget(self.inline_info_label)

        toolbar.addSeparator()

        # Load comparison buttons
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

        # Arrow keys for inline navigation
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self._prev_inline)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self._next_inline)

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
        """Load before stack (.zarr directory)."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Load Before Stack (.zarr directory)",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if path:
            self.load_before_stack(path)

    def _load_after(self):
        """Load after stack (.zarr directory)."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Load After Stack (.zarr directory)",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if path:
            self.load_after_stack(path)

    def _load_qc_stack(self):
        """Load QC stack data - select the .zarr directory or metadata JSON."""
        # Use directory dialog since zarr stores are directories
        path = QFileDialog.getExistingDirectory(
            self,
            "Load QC Stack (.zarr directory)",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if path:
            self.load_qc_stack(path)

    def load_qc_stack(self, path: str):
        """Load QC stack from file and populate inline navigation."""
        try:
            self._qc_stack_data = self._load_stack(path)

            # Populate inline combo
            self.inline_combo.blockSignals(True)
            self.inline_combo.clear()

            if self._qc_stack_data.inline_numbers:
                for inline in self._qc_stack_data.inline_numbers:
                    self.inline_combo.addItem(f"Inline {inline}", inline)
                self.inline_combo.setCurrentIndex(0)
                self._current_inline_idx = 0
            else:
                # No inline info - show all data
                self.inline_combo.addItem("All Data", -1)

            self.inline_combo.blockSignals(False)

            # Update info label
            n_inlines = len(self._qc_stack_data.inline_numbers) if self._qc_stack_data.inline_numbers else 1
            self.inline_info_label.setText(f"({n_inlines} inlines, {self._qc_stack_data.n_traces} CDPs)")

            # Display first inline
            self._display_current_inline()

            self.status_bar.showMessage(f"Loaded QC Stack: {Path(path).name}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load QC Stack: {e}")
            logger.exception("Load QC stack failed")

    def _on_inline_changed(self, index: int):
        """Handle inline selection change."""
        if index >= 0:
            self._current_inline_idx = index
            self._display_current_inline()

    def _prev_inline(self):
        """Go to previous inline."""
        if self.inline_combo.count() > 0:
            new_idx = max(0, self._current_inline_idx - 1)
            self.inline_combo.setCurrentIndex(new_idx)

    def _next_inline(self):
        """Go to next inline."""
        if self.inline_combo.count() > 0:
            new_idx = min(self.inline_combo.count() - 1, self._current_inline_idx + 1)
            self.inline_combo.setCurrentIndex(new_idx)

    def _display_current_inline(self):
        """Display the currently selected inline."""
        if self._qc_stack_data is None:
            return

        inline_num = self.inline_combo.currentData()

        if inline_num == -1 or inline_num is None:
            # Show all data
            display_data = self._qc_stack_data.data
        else:
            # Get data for specific inline
            display_data = self._qc_stack_data.get_inline_data(inline_num)
            if display_data is None:
                display_data = self._qc_stack_data.data

        # Display in the before panel (using it as main display for single stack)
        clip = self.clip_spin.value()
        self.before_panel.set_data(display_data, clip)

        # Hide other panels when viewing single stack
        self.before_panel.setVisible(True)
        self.after_panel.setVisible(False)
        self.diff_panel.setVisible(False)

        # Update statistics for current inline
        self._update_single_stack_stats(display_data)

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
        """Load stack data from Zarr directory."""
        path = Path(path)

        # Zarr stores are directories
        if not path.is_dir():
            raise ValueError(f"Expected zarr directory, got: {path}")

        zarr_path = path
        data = np.array(zarr.open(str(zarr_path), mode='r'))

        logger.info(f"Loaded zarr data shape: {data.shape} from {zarr_path}")

        # Try to load metadata
        metadata = {}
        sample_interval = 0.004
        cdp_numbers = None
        inline_numbers = None
        inline_ranges = None

        # Look for metadata file
        # For qc_stack_rd.zarr, metadata is qc_stack_rd_metadata.json in parent dir
        zarr_name = path.name  # e.g., "qc_stack_rd.zarr"
        base_name = zarr_name.replace('.zarr', '')  # e.g., "qc_stack_rd"

        meta_paths = [
            path.parent / f"{base_name}_metadata.json",  # Primary: same dir, _metadata suffix
            path.parent / f"{base_name}.json",           # Alternative: same dir, .json
            path / "metadata.json",                       # Inside zarr dir
            path.parent / "metadata.json",               # Generic in parent
        ]

        logger.info(f"Looking for metadata in: {[str(p) for p in meta_paths]}")

        for meta_path in meta_paths:
            if meta_path.exists():
                logger.info(f"Found metadata at: {meta_path}")
                with open(meta_path) as f:
                    metadata = json.load(f)
                sample_interval = metadata.get('sample_interval', 0.004)
                if 'cdp_numbers' in metadata:
                    cdp_numbers = np.array(metadata['cdp_numbers'])
                if 'inline_numbers' in metadata:
                    inline_numbers = metadata['inline_numbers']
                # Read inline_ranges directly from metadata if available
                if 'inline_ranges' in metadata:
                    # Convert string keys back to int
                    inline_ranges = {
                        int(k): tuple(v)
                        for k, v in metadata['inline_ranges'].items()
                    }
                    logger.info(f"Loaded inline_ranges: {inline_ranges}")
                break
        else:
            logger.warning(f"No metadata file found for {zarr_path}")

        # Fallback: Build inline ranges from cdp_numbers if not in metadata
        if inline_ranges is None and inline_numbers and cdp_numbers is not None and len(cdp_numbers) > 0:
            logger.info("Building inline_ranges from cdp_numbers (fallback)")
            inline_ranges = {}
            # CDPs are stored inline by inline, need to find boundaries
            # Simple approach: divide evenly
            cdps_per_inline = len(cdp_numbers) // len(inline_numbers)
            start_idx = 0

            for i, inline in enumerate(inline_numbers):
                if i < len(inline_numbers) - 1:
                    end_idx = start_idx + cdps_per_inline
                else:
                    end_idx = len(cdp_numbers)

                inline_ranges[inline] = (start_idx, end_idx)
                start_idx = end_idx

        return StackData(
            data=data,
            sample_interval=sample_interval,
            cdp_numbers=cdp_numbers,
            inline_numbers=inline_numbers,
            inline_ranges=inline_ranges,
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

    def _update_single_stack_stats(self, data: np.ndarray):
        """Update statistics for single stack display."""
        if data is None or data.size == 0:
            return

        rms = np.sqrt(np.mean(data**2))
        max_val = np.max(np.abs(data))

        self.rms_before_label.setText(f"{rms:.2e}")
        self.rms_after_label.setText("-")
        self.rms_diff_label.setText("-")
        self.correlation_label.setText("-")
        self.max_diff_label.setText(f"Max: {max_val:.2e}")
        self.snr_improvement_label.setText(f"Traces: {data.shape[1]}")


# =============================================================================
# Convenience function
# =============================================================================

def show_qc_stack_viewer(
    qc_stack_path: Optional[str] = None,
    before_path: Optional[str] = None,
    after_path: Optional[str] = None
):
    """
    Show QC stack viewer with optional initial data.

    Args:
        qc_stack_path: Path to QC stack (for single stack viewing with inline navigation)
        before_path: Path to before stack (for comparison mode)
        after_path: Path to after stack (for comparison mode)

    Returns:
        QCStackViewerWindow instance
    """
    viewer = QCStackViewerWindow()

    if qc_stack_path:
        viewer.load_qc_stack(qc_stack_path)
    elif before_path or after_path:
        if before_path:
            viewer.load_before_stack(before_path)
        if after_path:
            viewer.load_after_stack(after_path)

    viewer.show()
    return viewer
