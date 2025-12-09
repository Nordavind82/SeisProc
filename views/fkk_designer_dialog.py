"""
3D FKK Filter Designer Dialog

Interactive dialog for designing 3D FKK (Frequency-Wavenumber-Wavenumber) filters.
Shows slice-based visualization of input data, filtered output, and FKK spectrum.

UI Layout:
- Input Data: Time slice (X-Y) + Inline slice (T-X)
- Filtered Output: Same slices + difference view
- FKK Spectrum: Kx-Ky slice + F-Kx slice with velocity cone overlay
- Controls: Velocity range, azimuth, taper, mode
"""
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSlider, QDoubleSpinBox, QComboBox, QSpinBox,
    QRadioButton, QButtonGroup, QWidget, QSplitter, QCheckBox,
    QStatusBar, QProgressBar, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from typing import Optional, Tuple, Dict
import pyqtgraph as pg
import logging

from models.seismic_volume import SeismicVolume
from models.fkk_config import FKKConfig, FKK_PRESETS
from processors.fkk_filter_gpu import FKKFilterGPU, get_fkk_filter

logger = logging.getLogger(__name__)


class FKKDesignerDialog(QDialog):
    """
    3D FKK Filter Designer with slice-based visualization.

    Provides:
    - Synchronized slice viewers for input/output volumes
    - FKK spectrum visualization with filter overlay
    - Interactive parameter adjustment
    - Real-time preview with debouncing
    """

    filter_applied = pyqtSignal(object, object)  # (filtered_volume, config)

    def __init__(self, volume: SeismicVolume, parent=None):
        """
        Initialize FKK Designer dialog.

        Args:
            volume: SeismicVolume to design filter on
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("3D FKK Filter Designer")
        self.resize(1400, 900)

        self.volume = volume
        self.filtered_volume: Optional[SeismicVolume] = None
        self.spectrum: Optional[np.ndarray] = None
        self.spectrum_axes: Optional[Dict] = None

        # Current slice indices
        self.t_idx = volume.nt // 2
        self.y_idx = volume.ny // 2
        self.f_idx = 10  # Frequency slice index
        self.ky_idx = volume.ny // 2  # ky slice index (center = 0)

        # Filter processor
        self.processor = get_fkk_filter(prefer_gpu=True)

        # Current configuration
        self.config = FKKConfig()

        # Preview timer for debouncing
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._do_apply_filter)

        # View mode
        self._show_difference = False

        self._init_ui()
        self._connect_signals()
        self._compute_spectrum()
        self._update_all_views()

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Data views (Input + Output)
        left_panel = self._create_data_panel()
        main_splitter.addWidget(left_panel)

        # Right: Spectrum + Controls
        right_panel = self._create_spectrum_panel()
        main_splitter.addWidget(right_panel)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_splitter, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self._update_status()
        layout.addWidget(self.status_bar)

        # Bottom buttons
        button_layout = self._create_buttons()
        layout.addLayout(button_layout)

    def _create_header(self) -> QWidget:
        """Create header with volume info."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)

        nt, nx, ny = self.volume.shape
        info_text = (
            f"Volume: {nx}×{ny}×{nt} samples | "
            f"dt={self.volume.dt*1000:.1f}ms, dx={self.volume.dx:.1f}m, dy={self.volume.dy:.1f}m | "
            f"Size: {self.volume.memory_mb():.1f} MB"
        )

        label = QLabel(info_text)
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)
        layout.addStretch()

        return widget

    def _create_data_panel(self) -> QWidget:
        """Create left panel with input and output slice views."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter for input/output
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Input data group
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout(input_group)

        # Time slice view (X-Y at fixed T)
        input_layout.addWidget(QLabel("Time Slice (X-Y)"))
        self.input_time_view = pg.ImageView()
        self.input_time_view.ui.roiBtn.hide()
        self.input_time_view.ui.menuBtn.hide()
        input_layout.addWidget(self.input_time_view, stretch=1)

        # Time slice slider
        time_slider_layout = QHBoxLayout()
        time_slider_layout.addWidget(QLabel("t:"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, self.volume.nt - 1)
        self.time_slider.setValue(self.t_idx)
        time_slider_layout.addWidget(self.time_slider)
        self.time_label = QLabel(f"{self.t_idx * self.volume.dt * 1000:.0f} ms")
        self.time_label.setMinimumWidth(60)
        time_slider_layout.addWidget(self.time_label)
        input_layout.addLayout(time_slider_layout)

        # Inline slice view (T-X at fixed Y)
        input_layout.addWidget(QLabel("Inline (T-X)"))
        self.input_inline_view = pg.ImageView()
        self.input_inline_view.ui.roiBtn.hide()
        self.input_inline_view.ui.menuBtn.hide()
        input_layout.addWidget(self.input_inline_view, stretch=1)

        # Inline slider
        inline_slider_layout = QHBoxLayout()
        inline_slider_layout.addWidget(QLabel("Y:"))
        self.inline_slider = QSlider(Qt.Orientation.Horizontal)
        self.inline_slider.setRange(0, self.volume.ny - 1)
        self.inline_slider.setValue(self.y_idx)
        inline_slider_layout.addWidget(self.inline_slider)
        self.inline_label = QLabel(f"{self.y_idx}")
        self.inline_label.setMinimumWidth(40)
        inline_slider_layout.addWidget(self.inline_label)
        input_layout.addLayout(inline_slider_layout)

        splitter.addWidget(input_group)

        # Output data group
        output_group = QGroupBox("Filtered Output")
        output_layout = QVBoxLayout(output_group)

        # Output time slice
        self.output_time_view = pg.ImageView()
        self.output_time_view.ui.roiBtn.hide()
        self.output_time_view.ui.menuBtn.hide()
        output_layout.addWidget(self.output_time_view, stretch=1)

        # Output inline slice
        self.output_inline_view = pg.ImageView()
        self.output_inline_view.ui.roiBtn.hide()
        self.output_inline_view.ui.menuBtn.hide()
        output_layout.addWidget(self.output_inline_view, stretch=1)

        # View mode toggle
        view_mode_layout = QHBoxLayout()
        view_mode_layout.addWidget(QLabel("View:"))
        self.view_mode_group = QButtonGroup()

        self.view_filtered_btn = QRadioButton("Filtered")
        self.view_filtered_btn.setChecked(True)
        self.view_mode_group.addButton(self.view_filtered_btn, 0)
        view_mode_layout.addWidget(self.view_filtered_btn)

        self.view_diff_btn = QRadioButton("Difference")
        self.view_mode_group.addButton(self.view_diff_btn, 1)
        view_mode_layout.addWidget(self.view_diff_btn)

        view_mode_layout.addStretch()
        output_layout.addLayout(view_mode_layout)

        splitter.addWidget(output_group)

        layout.addWidget(splitter)
        return widget

    def _create_spectrum_panel(self) -> QWidget:
        """Create right panel with spectrum views and controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter for spectrum/controls
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Spectrum views group
        spectrum_group = QGroupBox("FKK Spectrum")
        spectrum_layout = QVBoxLayout(spectrum_group)

        # Kx-Ky slice at frequency f
        spectrum_layout.addWidget(QLabel("Kx-Ky @ frequency"))
        self.kxky_view = pg.ImageView()
        self.kxky_view.ui.roiBtn.hide()
        self.kxky_view.ui.menuBtn.hide()
        spectrum_layout.addWidget(self.kxky_view, stretch=1)

        # Frequency slider
        freq_slider_layout = QHBoxLayout()
        freq_slider_layout.addWidget(QLabel("f:"))
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(1, self.volume.nt // 2)
        self.freq_slider.setValue(self.f_idx)
        freq_slider_layout.addWidget(self.freq_slider)
        self.freq_label = QLabel("-- Hz")
        self.freq_label.setMinimumWidth(60)
        freq_slider_layout.addWidget(self.freq_label)
        spectrum_layout.addLayout(freq_slider_layout)

        # F-Kx slice at ky
        spectrum_layout.addWidget(QLabel("F-Kx @ ky"))
        self.fkx_view = pg.ImageView()
        self.fkx_view.ui.roiBtn.hide()
        self.fkx_view.ui.menuBtn.hide()
        spectrum_layout.addWidget(self.fkx_view, stretch=1)

        # ky slider
        ky_slider_layout = QHBoxLayout()
        ky_slider_layout.addWidget(QLabel("ky:"))
        self.ky_slider = QSlider(Qt.Orientation.Horizontal)
        self.ky_slider.setRange(0, self.volume.ny - 1)
        self.ky_slider.setValue(self.ky_idx)
        ky_slider_layout.addWidget(self.ky_slider)
        self.ky_label = QLabel("0")
        self.ky_label.setMinimumWidth(60)
        ky_slider_layout.addWidget(self.ky_label)
        spectrum_layout.addLayout(ky_slider_layout)

        splitter.addWidget(spectrum_group)

        # Filter controls group
        controls_group = QGroupBox("Filter Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_group = QButtonGroup()

        self.mode_reject = QRadioButton("Reject")
        self.mode_reject.setChecked(True)
        self.mode_group.addButton(self.mode_reject, 0)
        mode_layout.addWidget(self.mode_reject)

        self.mode_pass = QRadioButton("Pass")
        self.mode_group.addButton(self.mode_pass, 1)
        mode_layout.addWidget(self.mode_pass)

        mode_layout.addStretch()
        controls_layout.addLayout(mode_layout)

        # V_min
        vmin_layout = QHBoxLayout()
        vmin_layout.addWidget(QLabel("V min:"))
        self.v_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_min_slider.setRange(50, 5000)
        self.v_min_slider.setValue(int(self.config.v_min))
        vmin_layout.addWidget(self.v_min_slider)
        self.v_min_spin = QDoubleSpinBox()
        self.v_min_spin.setRange(50, 5000)
        self.v_min_spin.setValue(self.config.v_min)
        self.v_min_spin.setSuffix(" m/s")
        self.v_min_spin.setMinimumWidth(100)
        vmin_layout.addWidget(self.v_min_spin)
        controls_layout.addLayout(vmin_layout)

        # V_max
        vmax_layout = QHBoxLayout()
        vmax_layout.addWidget(QLabel("V max:"))
        self.v_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_max_slider.setRange(100, 10000)
        self.v_max_slider.setValue(int(self.config.v_max))
        vmax_layout.addWidget(self.v_max_slider)
        self.v_max_spin = QDoubleSpinBox()
        self.v_max_spin.setRange(100, 10000)
        self.v_max_spin.setValue(self.config.v_max)
        self.v_max_spin.setSuffix(" m/s")
        self.v_max_spin.setMinimumWidth(100)
        vmax_layout.addWidget(self.v_max_spin)
        controls_layout.addLayout(vmax_layout)

        # Azimuth range
        az_layout = QHBoxLayout()
        az_layout.addWidget(QLabel("Azimuth:"))
        self.az_min_spin = QDoubleSpinBox()
        self.az_min_spin.setRange(0, 360)
        self.az_min_spin.setValue(0)
        self.az_min_spin.setSuffix("°")
        az_layout.addWidget(self.az_min_spin)
        az_layout.addWidget(QLabel("to"))
        self.az_max_spin = QDoubleSpinBox()
        self.az_max_spin.setRange(0, 360)
        self.az_max_spin.setValue(360)
        self.az_max_spin.setSuffix("°")
        az_layout.addWidget(self.az_max_spin)
        az_layout.addStretch()
        controls_layout.addLayout(az_layout)

        # Taper width
        taper_layout = QHBoxLayout()
        taper_layout.addWidget(QLabel("Taper:"))
        self.taper_slider = QSlider(Qt.Orientation.Horizontal)
        self.taper_slider.setRange(1, 50)  # 0.01 to 0.50
        self.taper_slider.setValue(int(self.config.taper_width * 100))
        taper_layout.addWidget(self.taper_slider)
        self.taper_spin = QDoubleSpinBox()
        self.taper_spin.setRange(0.01, 0.50)
        self.taper_spin.setValue(self.config.taper_width)
        self.taper_spin.setSingleStep(0.01)
        self.taper_spin.setMinimumWidth(80)
        taper_layout.addWidget(self.taper_spin)
        controls_layout.addLayout(taper_layout)

        # === Quality Improvements Section (v2.0) ===
        controls_layout.addWidget(QLabel(""))  # Spacer

        # AGC controls
        agc_layout = QHBoxLayout()
        self.agc_checkbox = QCheckBox("Apply AGC")
        self.agc_checkbox.setChecked(self.config.apply_agc)
        self.agc_checkbox.setToolTip("Apply AGC before filtering and remove after using same scalars")
        agc_layout.addWidget(self.agc_checkbox)
        agc_layout.addWidget(QLabel("Window:"))
        self.agc_window_spin = QDoubleSpinBox()
        self.agc_window_spin.setRange(50, 5000)
        self.agc_window_spin.setValue(self.config.agc_window_ms)
        self.agc_window_spin.setSuffix(" ms")
        self.agc_window_spin.setMinimumWidth(80)
        agc_layout.addWidget(self.agc_window_spin)
        agc_layout.addWidget(QLabel("Max Gain:"))
        self.agc_gain_spin = QDoubleSpinBox()
        self.agc_gain_spin.setRange(1, 100)
        self.agc_gain_spin.setValue(self.config.agc_max_gain)
        self.agc_gain_spin.setMinimumWidth(60)
        agc_layout.addWidget(self.agc_gain_spin)
        controls_layout.addLayout(agc_layout)

        # Frequency band selection
        freq_band_layout = QHBoxLayout()
        freq_band_layout.addWidget(QLabel("Freq Band:"))
        self.f_min_spin = QDoubleSpinBox()
        self.f_min_spin.setRange(0, 500)
        self.f_min_spin.setValue(self.config.f_min if self.config.f_min is not None else 0)
        self.f_min_spin.setSuffix(" Hz")
        self.f_min_spin.setSpecialValueText("0 Hz")
        self.f_min_spin.setToolTip("Minimum frequency for filter action (0 = DC)")
        freq_band_layout.addWidget(self.f_min_spin)
        freq_band_layout.addWidget(QLabel("to"))
        nyquist = 0.5 / self.volume.dt
        self.f_max_spin = QDoubleSpinBox()
        self.f_max_spin.setRange(1, nyquist)
        self.f_max_spin.setValue(self.config.f_max if self.config.f_max is not None else nyquist)
        self.f_max_spin.setSuffix(" Hz")
        self.f_max_spin.setToolTip("Maximum frequency for filter action (Nyquist = {:.0f} Hz)".format(nyquist))
        freq_band_layout.addWidget(self.f_max_spin)
        freq_band_layout.addStretch()
        controls_layout.addLayout(freq_band_layout)

        # Temporal tapering (ms on top/bottom)
        temporal_taper_layout = QHBoxLayout()
        temporal_taper_layout.addWidget(QLabel("Temporal Taper:"))
        temporal_taper_layout.addWidget(QLabel("Top:"))
        self.taper_top_spin = QDoubleSpinBox()
        self.taper_top_spin.setRange(0, 1000)
        self.taper_top_spin.setValue(self.config.taper_ms_top)
        self.taper_top_spin.setSuffix(" ms")
        self.taper_top_spin.setToolTip("Taper length at top of traces")
        temporal_taper_layout.addWidget(self.taper_top_spin)
        temporal_taper_layout.addWidget(QLabel("Bottom:"))
        self.taper_bottom_spin = QDoubleSpinBox()
        self.taper_bottom_spin.setRange(0, 1000)
        self.taper_bottom_spin.setValue(self.config.taper_ms_bottom)
        self.taper_bottom_spin.setSuffix(" ms")
        self.taper_bottom_spin.setToolTip("Taper length at bottom of traces")
        temporal_taper_layout.addWidget(self.taper_bottom_spin)
        temporal_taper_layout.addStretch()
        controls_layout.addLayout(temporal_taper_layout)

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", None)
        for name in FKK_PRESETS.keys():
            self.preset_combo.addItem(name, name)
        preset_layout.addWidget(self.preset_combo)
        controls_layout.addLayout(preset_layout)

        # Action buttons
        action_layout = QHBoxLayout()
        self.compute_btn = QPushButton("Compute Spectrum")
        action_layout.addWidget(self.compute_btn)
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setStyleSheet("font-weight: bold;")
        action_layout.addWidget(self.apply_btn)
        controls_layout.addLayout(action_layout)

        splitter.addWidget(controls_group)

        # Set splitter proportions
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        return widget

    def _create_buttons(self) -> QHBoxLayout:
        """Create bottom button row."""
        layout = QHBoxLayout()

        self.export_btn = QPushButton("Export Result")
        layout.addWidget(self.export_btn)

        layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        layout.addWidget(self.cancel_btn)

        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.accept_btn)

        return layout

    def _connect_signals(self):
        """Connect UI signals to slots."""
        # Slice navigation
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.inline_slider.valueChanged.connect(self._on_inline_changed)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        self.ky_slider.valueChanged.connect(self._on_ky_changed)

        # View mode
        self.view_mode_group.buttonClicked.connect(self._on_view_mode_changed)

        # Filter parameters
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        self.v_min_slider.valueChanged.connect(self._on_v_min_slider_changed)
        self.v_min_spin.valueChanged.connect(self._on_v_min_spin_changed)
        self.v_max_slider.valueChanged.connect(self._on_v_max_slider_changed)
        self.v_max_spin.valueChanged.connect(self._on_v_max_spin_changed)
        self.az_min_spin.valueChanged.connect(self._on_az_changed)
        self.az_max_spin.valueChanged.connect(self._on_az_changed)
        self.taper_slider.valueChanged.connect(self._on_taper_slider_changed)
        self.taper_spin.valueChanged.connect(self._on_taper_spin_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        # Quality improvement controls (v2.0)
        self.agc_checkbox.stateChanged.connect(self._on_agc_changed)
        self.agc_window_spin.valueChanged.connect(self._on_agc_window_changed)
        self.agc_gain_spin.valueChanged.connect(self._on_agc_gain_changed)
        self.f_min_spin.valueChanged.connect(self._on_freq_band_changed)
        self.f_max_spin.valueChanged.connect(self._on_freq_band_changed)
        self.taper_top_spin.valueChanged.connect(self._on_temporal_taper_changed)
        self.taper_bottom_spin.valueChanged.connect(self._on_temporal_taper_changed)

        # Actions
        self.compute_btn.clicked.connect(self._compute_spectrum)
        self.apply_btn.clicked.connect(self._request_apply_filter)
        self.export_btn.clicked.connect(self._export_result)
        self.cancel_btn.clicked.connect(self.reject)
        self.accept_btn.clicked.connect(self._accept_result)

    # =========================================================================
    # Slice Navigation
    # =========================================================================

    def _on_time_changed(self, value: int):
        """Handle time slice slider change."""
        self.t_idx = value
        self.time_label.setText(f"{value * self.volume.dt * 1000:.0f} ms")
        self._update_data_views()

    def _on_inline_changed(self, value: int):
        """Handle inline slider change."""
        self.y_idx = value
        self.inline_label.setText(f"{value}")
        self._update_data_views()

    def _on_freq_changed(self, value: int):
        """Handle frequency slider change."""
        self.f_idx = value
        if self.spectrum_axes:
            freq = self.spectrum_axes['f_axis'][value]
            self.freq_label.setText(f"{freq:.1f} Hz")
        self._update_spectrum_views()

    def _on_ky_changed(self, value: int):
        """Handle ky slider change."""
        self.ky_idx = value
        if self.spectrum_axes:
            ky = self.spectrum_axes['ky_axis'][value]
            self.ky_label.setText(f"{ky:.4f}")
        self._update_spectrum_views()

    def _on_view_mode_changed(self, button):
        """Handle view mode toggle."""
        self._show_difference = (button == self.view_diff_btn)
        self._update_output_views()

    # =========================================================================
    # Filter Parameter Changes
    # =========================================================================

    def _on_mode_changed(self, button):
        """Handle mode change."""
        self.config.mode = 'reject' if button == self.mode_reject else 'pass'
        self._request_apply_filter()

    def _on_v_min_slider_changed(self, value: int):
        """Handle v_min slider change."""
        self.v_min_spin.blockSignals(True)
        self.v_min_spin.setValue(value)
        self.v_min_spin.blockSignals(False)
        self.config.v_min = float(value)
        self._request_apply_filter()

    def _on_v_min_spin_changed(self, value: float):
        """Handle v_min spin change."""
        self.v_min_slider.blockSignals(True)
        self.v_min_slider.setValue(int(value))
        self.v_min_slider.blockSignals(False)
        self.config.v_min = value
        self._request_apply_filter()

    def _on_v_max_slider_changed(self, value: int):
        """Handle v_max slider change."""
        self.v_max_spin.blockSignals(True)
        self.v_max_spin.setValue(value)
        self.v_max_spin.blockSignals(False)
        self.config.v_max = float(value)
        self._request_apply_filter()

    def _on_v_max_spin_changed(self, value: float):
        """Handle v_max spin change."""
        self.v_max_slider.blockSignals(True)
        self.v_max_slider.setValue(int(value))
        self.v_max_slider.blockSignals(False)
        self.config.v_max = value
        self._request_apply_filter()

    def _on_az_changed(self, value: float):
        """Handle azimuth change."""
        self.config.azimuth_min = self.az_min_spin.value()
        self.config.azimuth_max = self.az_max_spin.value()
        self._request_apply_filter()

    def _on_taper_slider_changed(self, value: int):
        """Handle taper slider change."""
        taper = value / 100.0
        self.taper_spin.blockSignals(True)
        self.taper_spin.setValue(taper)
        self.taper_spin.blockSignals(False)
        self.config.taper_width = taper
        self._request_apply_filter()

    def _on_taper_spin_changed(self, value: float):
        """Handle taper spin change."""
        self.taper_slider.blockSignals(True)
        self.taper_slider.setValue(int(value * 100))
        self.taper_slider.blockSignals(False)
        self.config.taper_width = value
        self._request_apply_filter()

    def _on_agc_changed(self, state: int):
        """Handle AGC checkbox change."""
        self.config.apply_agc = (state == Qt.CheckState.Checked.value)
        self._request_apply_filter()

    def _on_agc_window_changed(self, value: float):
        """Handle AGC window change."""
        self.config.agc_window_ms = value
        if self.config.apply_agc:
            self._request_apply_filter()

    def _on_agc_gain_changed(self, value: float):
        """Handle AGC max gain change."""
        self.config.agc_max_gain = value
        if self.config.apply_agc:
            self._request_apply_filter()

    def _on_freq_band_changed(self, value: float):
        """Handle frequency band change."""
        f_min = self.f_min_spin.value()
        f_max = self.f_max_spin.value()
        self.config.f_min = f_min if f_min > 0 else None
        self.config.f_max = f_max if f_max < (0.5 / self.volume.dt) else None
        self._request_apply_filter()

    def _on_temporal_taper_changed(self, value: float):
        """Handle temporal taper change."""
        self.config.taper_ms_top = self.taper_top_spin.value()
        self.config.taper_ms_bottom = self.taper_bottom_spin.value()
        self._request_apply_filter()

    def _on_preset_changed(self, index: int):
        """Handle preset selection."""
        name = self.preset_combo.currentData()
        if name and name in FKK_PRESETS:
            preset = FKK_PRESETS[name]
            self._set_config(preset)
            self._request_apply_filter()

    def _set_config(self, config: FKKConfig):
        """Set all UI elements from config."""
        self.config = config.copy()

        # Block signals during update
        self.v_min_slider.blockSignals(True)
        self.v_min_spin.blockSignals(True)
        self.v_max_slider.blockSignals(True)
        self.v_max_spin.blockSignals(True)
        self.taper_slider.blockSignals(True)
        self.taper_spin.blockSignals(True)
        self.az_min_spin.blockSignals(True)
        self.az_max_spin.blockSignals(True)
        self.agc_checkbox.blockSignals(True)
        self.agc_window_spin.blockSignals(True)
        self.agc_gain_spin.blockSignals(True)
        self.f_min_spin.blockSignals(True)
        self.f_max_spin.blockSignals(True)
        self.taper_top_spin.blockSignals(True)
        self.taper_bottom_spin.blockSignals(True)

        # Core parameters
        self.v_min_slider.setValue(int(config.v_min))
        self.v_min_spin.setValue(config.v_min)
        self.v_max_slider.setValue(int(config.v_max))
        self.v_max_spin.setValue(config.v_max)
        self.taper_slider.setValue(int(config.taper_width * 100))
        self.taper_spin.setValue(config.taper_width)
        self.az_min_spin.setValue(config.azimuth_min)
        self.az_max_spin.setValue(config.azimuth_max)

        if config.mode == 'reject':
            self.mode_reject.setChecked(True)
        else:
            self.mode_pass.setChecked(True)

        # Quality improvement parameters (v2.0)
        self.agc_checkbox.setChecked(config.apply_agc)
        self.agc_window_spin.setValue(config.agc_window_ms)
        self.agc_gain_spin.setValue(config.agc_max_gain)

        nyquist = 0.5 / self.volume.dt
        self.f_min_spin.setValue(config.f_min if config.f_min is not None else 0)
        self.f_max_spin.setValue(config.f_max if config.f_max is not None else nyquist)

        self.taper_top_spin.setValue(config.taper_ms_top)
        self.taper_bottom_spin.setValue(config.taper_ms_bottom)

        # Unblock signals
        self.v_min_slider.blockSignals(False)
        self.v_min_spin.blockSignals(False)
        self.v_max_slider.blockSignals(False)
        self.v_max_spin.blockSignals(False)
        self.taper_slider.blockSignals(False)
        self.taper_spin.blockSignals(False)
        self.az_min_spin.blockSignals(False)
        self.az_max_spin.blockSignals(False)
        self.agc_checkbox.blockSignals(False)
        self.agc_window_spin.blockSignals(False)
        self.agc_gain_spin.blockSignals(False)
        self.f_min_spin.blockSignals(False)
        self.f_max_spin.blockSignals(False)
        self.taper_top_spin.blockSignals(False)
        self.taper_bottom_spin.blockSignals(False)

    # =========================================================================
    # Processing
    # =========================================================================

    def _compute_spectrum(self):
        """Compute FKK spectrum."""
        self.status_bar.showMessage("Computing FKK spectrum...")
        try:
            self.spectrum = self.processor.compute_spectrum(self.volume)
            self.spectrum_axes = self.processor._compute_axes(self.volume)

            # Update frequency label
            if self.spectrum_axes:
                freq = self.spectrum_axes['f_axis'][self.f_idx]
                self.freq_label.setText(f"{freq:.1f} Hz")
                ky = self.spectrum_axes['ky_axis'][self.ky_idx]
                self.ky_label.setText(f"{ky:.4f}")

            self._update_spectrum_views()
            self.status_bar.showMessage("Spectrum computed", 3000)
        except Exception as e:
            logger.error(f"Spectrum computation failed: {e}")
            QMessageBox.warning(self, "Error", f"Spectrum computation failed:\n{e}")
            self.status_bar.showMessage("Spectrum computation failed")

    def _request_apply_filter(self):
        """Request filter application with debouncing."""
        self._preview_timer.start(150)  # 150ms debounce

    def _do_apply_filter(self):
        """Apply filter to volume."""
        self.status_bar.showMessage("Applying filter...")
        try:
            self.filtered_volume = self.processor.apply_filter(
                self.volume, self.config, use_cached_spectrum=True
            )
            self._update_output_views()
            self._update_spectrum_views()
            self.status_bar.showMessage(f"Filter applied: {self.config.get_summary()}", 3000)
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            self.status_bar.showMessage(f"Filter failed: {e}")

    # =========================================================================
    # View Updates
    # =========================================================================

    def _update_all_views(self):
        """Update all views."""
        self._update_data_views()
        self._update_spectrum_views()

    def _update_data_views(self):
        """Update input and output data views."""
        # Input views
        time_slice = self.volume.time_slice(self.t_idx)
        inline_slice = self.volume.inline_slice(self.y_idx)

        self.input_time_view.setImage(time_slice.T, autoLevels=True)
        self.input_inline_view.setImage(inline_slice.T, autoLevels=True)

        self._update_output_views()

    def _update_output_views(self):
        """Update output data views."""
        if self.filtered_volume is None:
            return

        time_slice_filt = self.filtered_volume.time_slice(self.t_idx)
        inline_slice_filt = self.filtered_volume.inline_slice(self.y_idx)

        if self._show_difference:
            # Show difference (input - filtered = rejected noise)
            time_slice_in = self.volume.time_slice(self.t_idx)
            inline_slice_in = self.volume.inline_slice(self.y_idx)
            time_slice_show = time_slice_in - time_slice_filt
            inline_slice_show = inline_slice_in - inline_slice_filt
        else:
            time_slice_show = time_slice_filt
            inline_slice_show = inline_slice_filt

        self.output_time_view.setImage(time_slice_show.T, autoLevels=True)
        self.output_inline_view.setImage(inline_slice_show.T, autoLevels=True)

    def _update_spectrum_views(self):
        """Update FKK spectrum views with filter overlay."""
        if self.spectrum is None:
            return

        # Kx-Ky slice at frequency f_idx
        kxky_slice = self.spectrum[self.f_idx, :, :]
        # Use log scale for display
        kxky_log = np.log10(kxky_slice + 1e-10)
        self.kxky_view.setImage(kxky_log.T, autoLevels=True)

        # F-Kx slice at ky_idx
        fkx_slice = self.spectrum[:, :, self.ky_idx]
        fkx_log = np.log10(fkx_slice + 1e-10)
        self.fkx_view.setImage(fkx_log.T, autoLevels=True)

        # TODO: Add velocity cone overlay

    def _update_status(self):
        """Update status bar."""
        status = self.processor.get_status() if hasattr(self.processor, 'get_status') else {}
        gpu_name = status.get('device_name', 'Unknown')
        mem_alloc = status.get('memory_allocated_mb')
        mem_total = status.get('memory_total_mb')

        if mem_alloc and mem_total:
            status_text = f"GPU: {gpu_name} | Memory: {mem_alloc:.0f}/{mem_total:.0f} MB"
        else:
            status_text = f"Device: {gpu_name}"

        self.status_bar.showMessage(status_text)

    # =========================================================================
    # Actions
    # =========================================================================

    def _export_result(self):
        """Export filtered result."""
        if self.filtered_volume is None:
            QMessageBox.warning(self, "No Result", "Apply filter first before exporting.")
            return

        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Filtered Volume", "", "NumPy (*.npy);;All Files (*)"
        )
        if path:
            np.save(path, self.filtered_volume.data)
            self.status_bar.showMessage(f"Exported to {path}", 5000)

    def _accept_result(self):
        """Accept and emit result."""
        if self.filtered_volume is None:
            self._do_apply_filter()

        if self.filtered_volume is not None:
            self.filter_applied.emit(self.filtered_volume, self.config)
            self.accept()
        else:
            QMessageBox.warning(self, "No Result", "Failed to apply filter.")

    def get_filtered_volume(self) -> Optional[SeismicVolume]:
        """Get the filtered volume result."""
        return self.filtered_volume

    def get_config(self) -> FKKConfig:
        """Get current filter configuration."""
        return self.config.copy()

    def closeEvent(self, event):
        """Clean up on close."""
        self._preview_timer.stop()
        if hasattr(self.processor, 'clear_cache'):
            self.processor.clear_cache()
        super().closeEvent(event)
