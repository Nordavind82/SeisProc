"""
3D FKK Filter Designer Dialog

Interactive dialog for designing 3D FKK (Frequency-Wavenumber-Wavenumber) filters.
Redesigned layout with:
- Narrow left panel: Filter controls
- Large right area: 4 quadrant display (XX, TX slices + KK, FK spectra)
- Toggle buttons to show filtered/unfiltered data
- Colormap selection and dB mode for spectrum
"""
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSlider, QDoubleSpinBox, QComboBox, QSpinBox,
    QRadioButton, QButtonGroup, QWidget, QSplitter, QCheckBox,
    QStatusBar, QMessageBox, QScrollArea, QFrame, QGridLayout,
    QSizePolicy, QToolButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from typing import Optional, Tuple, Dict
import pyqtgraph as pg
import logging

from models.seismic_volume import SeismicVolume
from models.fkk_config import FKKConfig, FKK_PRESETS
from processors.fkk_filter_gpu import FKKFilterGPU, get_fkk_filter

logger = logging.getLogger(__name__)


# ============================================================================
# Colormap definitions
# ============================================================================

def get_colormap_by_name(name: str):
    """Get colormap by name."""
    from pyqtgraph import ColorMap

    colormaps = {
        'seismic_bwr': {  # Blue-White-Red (classic seismic)
            'positions': [0.0, 0.5, 1.0],
            'colors': [(0, 0, 255), (255, 255, 255), (255, 0, 0)]
        },
        'seismic_gray': {  # Black-White-Black (variable density)
            'positions': [0.0, 0.5, 1.0],
            'colors': [(0, 0, 0), (255, 255, 255), (0, 0, 0)]
        },
        'seismic_rwb': {  # Red-White-Blue (inverted)
            'positions': [0.0, 0.5, 1.0],
            'colors': [(255, 0, 0), (255, 255, 255), (0, 0, 255)]
        },
        'viridis': {  # Perceptually uniform
            'positions': [0.0, 0.25, 0.5, 0.75, 1.0],
            'colors': [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
        },
        'plasma': {  # High contrast
            'positions': [0.0, 0.25, 0.5, 0.75, 1.0],
            'colors': [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)]
        },
        'inferno': {  # Good for spectrum
            'positions': [0.0, 0.25, 0.5, 0.75, 1.0],
            'colors': [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)]
        },
        'hot': {  # Black-Red-Yellow-White
            'positions': [0.0, 0.33, 0.66, 1.0],
            'colors': [(0, 0, 0), (230, 0, 0), (255, 210, 0), (255, 255, 255)]
        },
        'jet': {  # Rainbow (traditional)
            'positions': [0.0, 0.25, 0.5, 0.75, 1.0],
            'colors': [(0, 0, 128), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)]
        },
    }

    if name not in colormaps:
        name = 'seismic_bwr'

    cm = colormaps[name]
    return ColorMap(cm['positions'], cm['colors'])


def get_seismic_colormap():
    """Create seismic colormap (blue-white-red)."""
    return get_colormap_by_name('seismic_bwr')


def get_spectrum_colormap():
    """Create spectrum colormap (inferno - good for dB display)."""
    return get_colormap_by_name('inferno')


# Available colormaps for UI
SEISMIC_COLORMAPS = ['seismic_bwr', 'seismic_gray', 'seismic_rwb', 'viridis']
SPECTRUM_COLORMAPS = ['inferno', 'hot', 'plasma', 'viridis', 'jet']


class SeismicImageView(pg.PlotWidget):
    """Custom seismic display widget with proper colormap and aspect ratio."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)

        self.setBackground('k')  # Black background
        self.setTitle(title, color='w', size='10pt')

        # Create image item
        self.img_item = pg.ImageItem()
        self.addItem(self.img_item)

        # Current colormap names
        self._seismic_cmap_name = 'seismic_bwr'
        self._spectrum_cmap_name = 'inferno'

        # Set seismic colormap
        self.seismic_cmap = get_seismic_colormap()
        self.spectrum_cmap = get_spectrum_colormap()
        self.img_item.setLookupTable(self.seismic_cmap.getLookupTable())

        # Axis labels
        self.setLabel('left', '', color='w')
        self.setLabel('bottom', '', color='w')

        # Invert Y axis for seismic (time increases downward)
        self.invertY(True)

        self._data = None
        self._raw_data = None  # Store raw data before dB conversion
        self._is_spectrum = False
        self._use_db = True  # Use dB scale for spectrum by default
        self._db_range = 60  # dB dynamic range

    def setColormap(self, name: str, is_spectrum: bool = False):
        """Change colormap by name."""
        cmap = get_colormap_by_name(name)
        if is_spectrum:
            self._spectrum_cmap_name = name
            self.spectrum_cmap = cmap
        else:
            self._seismic_cmap_name = name
            self.seismic_cmap = cmap

        # Update current display if data exists
        if self._data is not None:
            if self._is_spectrum:
                self.img_item.setLookupTable(self.spectrum_cmap.getLookupTable())
            else:
                self.img_item.setLookupTable(self.seismic_cmap.getLookupTable())

    def setDbMode(self, use_db: bool, db_range: float = 60):
        """Set dB mode for spectrum display."""
        self._use_db = use_db
        self._db_range = db_range
        # Refresh if spectrum data exists
        if self._is_spectrum and self._raw_data is not None:
            self._update_spectrum_display()

    def _update_spectrum_display(self):
        """Update spectrum display with current dB settings."""
        if self._raw_data is None:
            return

        if self._use_db:
            # Convert to dB: 20*log10(amplitude)
            # Avoid log of zero
            data_safe = np.maximum(self._raw_data, 1e-20)
            data_db = 20 * np.log10(data_safe)
            # Normalize to dB range from max
            vmax = np.nanmax(data_db)
            vmin = vmax - self._db_range
            self._data = np.clip(data_db, vmin, vmax)
            self.img_item.setImage(self._data.T)
            self.img_item.setLevels([vmin, vmax])
        else:
            # Linear scale
            self._data = self._raw_data
            self.img_item.setImage(self._data.T)
            vmax = np.nanmax(self._data)
            vmin = 0
            self.img_item.setLevels([vmin, vmax])

    def setData(self, data: np.ndarray, is_spectrum: bool = False,
                autoLevels: bool = True, xlabel: str = '', ylabel: str = ''):
        """Set image data with proper scaling."""
        if data is None:
            return

        self._is_spectrum = is_spectrum

        # Choose colormap
        if is_spectrum:
            self.img_item.setLookupTable(self.spectrum_cmap.getLookupTable())
            self.invertY(False)
            # Store raw data for dB conversion
            self._raw_data = data.copy()
            self._update_spectrum_display()
        else:
            self.img_item.setLookupTable(self.seismic_cmap.getLookupTable())
            self.invertY(True)
            self._data = data
            self._raw_data = None
            self.img_item.setImage(data.T)

            if autoLevels:
                # Symmetric around zero for seismic
                vmax = np.nanpercentile(np.abs(data), 99)
                vmin = -vmax
                self.img_item.setLevels([vmin, vmax])

        # Set labels
        self.setLabel('bottom', xlabel, color='w')
        self.setLabel('left', ylabel, color='w')

    def getData(self) -> Optional[np.ndarray]:
        """Get current data."""
        return self._data

    def getRawData(self) -> Optional[np.ndarray]:
        """Get raw data (before dB conversion)."""
        return self._raw_data if self._raw_data is not None else self._data


class FKKDesignerDialog(QDialog):
    """
    3D FKK Filter Designer with improved layout.

    Layout:
    - Left: Narrow controls panel (scrollable)
    - Right: 4 large quadrants (XX, TX, KK, FK views)
    - Toggle buttons above each quadrant pair
    """

    filter_applied = pyqtSignal(object, object)  # (filtered_volume, config)

    def __init__(self, volume: SeismicVolume, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D FKK Filter Designer")
        self.resize(1600, 1000)

        self.volume = volume
        self.filtered_volume: Optional[SeismicVolume] = None
        self.spectrum: Optional[np.ndarray] = None
        self.filtered_spectrum: Optional[np.ndarray] = None
        self.spectrum_axes: Optional[Dict] = None

        # Current slice indices
        self.t_idx = volume.nt // 2
        self.y_idx = volume.ny // 2
        self.f_idx = min(10, volume.nt // 4)
        self.ky_idx = volume.ny // 2

        # Filter processor
        self.processor = get_fkk_filter(prefer_gpu=True)

        # Current configuration
        self.config = FKKConfig()

        # Preview timer for debouncing
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._do_apply_filter)

        # View state
        self._show_filtered_data = True
        self._show_filtered_spectrum = True

        self._init_ui()
        self._connect_signals()
        self._compute_spectrum()
        self._update_all_views()

    def _init_ui(self):
        """Initialize user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Left: Controls panel (narrow, scrollable)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(280)
        controls_scroll.setMaximumWidth(320)

        controls_widget = self._create_controls_panel()
        controls_scroll.setWidget(controls_widget)
        layout.addWidget(controls_scroll)

        # Right: Display area (4 quadrants)
        display_widget = self._create_display_panel()
        layout.addWidget(display_widget, stretch=1)

    def _create_controls_panel(self) -> QWidget:
        """Create narrow controls panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Header info
        info_label = QLabel(
            f"<b>Volume:</b> {self.volume.nx}x{self.volume.ny}x{self.volume.nt}<br>"
            f"<b>dt:</b> {self.volume.dt*1000:.1f}ms | "
            f"<b>dx:</b> {self.volume.dx:.1f}m | "
            f"<b>dy:</b> {self.volume.dy:.1f}m"
        )
        info_label.setStyleSheet("font-size: 9pt; color: #888;")
        layout.addWidget(info_label)

        # Slice navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)

        # Time slice
        t_layout = QHBoxLayout()
        t_layout.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, self.volume.nt - 1)
        self.time_slider.setValue(self.t_idx)
        t_layout.addWidget(self.time_slider)
        self.time_label = QLabel(f"{self.t_idx * self.volume.dt * 1000:.0f}ms")
        self.time_label.setMinimumWidth(50)
        t_layout.addWidget(self.time_label)
        nav_layout.addLayout(t_layout)

        # Y (inline) slice
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.inline_slider = QSlider(Qt.Orientation.Horizontal)
        self.inline_slider.setRange(0, self.volume.ny - 1)
        self.inline_slider.setValue(self.y_idx)
        y_layout.addWidget(self.inline_slider)
        self.inline_label = QLabel(f"{self.y_idx}")
        self.inline_label.setMinimumWidth(30)
        y_layout.addWidget(self.inline_label)
        nav_layout.addLayout(y_layout)

        # Frequency slice
        f_layout = QHBoxLayout()
        f_layout.addWidget(QLabel("Freq:"))
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(1, self.volume.nt // 2)
        self.freq_slider.setValue(self.f_idx)
        f_layout.addWidget(self.freq_slider)
        self.freq_label = QLabel("-- Hz")
        self.freq_label.setMinimumWidth(50)
        f_layout.addWidget(self.freq_label)
        nav_layout.addLayout(f_layout)

        # ky slice
        ky_layout = QHBoxLayout()
        ky_layout.addWidget(QLabel("ky:"))
        self.ky_slider = QSlider(Qt.Orientation.Horizontal)
        self.ky_slider.setRange(0, self.volume.ny - 1)
        self.ky_slider.setValue(self.ky_idx)
        ky_layout.addWidget(self.ky_slider)
        self.ky_label = QLabel("0")
        self.ky_label.setMinimumWidth(50)
        ky_layout.addWidget(self.ky_label)
        nav_layout.addLayout(ky_layout)

        layout.addWidget(nav_group)

        # Filter parameters
        filter_group = QGroupBox("Velocity Cone")
        filter_layout = QVBoxLayout(filter_group)

        # Mode
        mode_layout = QHBoxLayout()
        self.mode_reject = QRadioButton("Reject")
        self.mode_reject.setChecked(True)
        self.mode_pass = QRadioButton("Pass")
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.mode_reject, 0)
        self.mode_group.addButton(self.mode_pass, 1)
        mode_layout.addWidget(self.mode_reject)
        mode_layout.addWidget(self.mode_pass)
        mode_layout.addStretch()
        filter_layout.addLayout(mode_layout)

        # V min
        vmin_layout = QHBoxLayout()
        vmin_layout.addWidget(QLabel("V min:"))
        self.v_min_spin = QSpinBox()
        self.v_min_spin.setRange(50, 5000)
        self.v_min_spin.setValue(int(self.config.v_min))
        self.v_min_spin.setSuffix(" m/s")
        vmin_layout.addWidget(self.v_min_spin)
        filter_layout.addLayout(vmin_layout)

        # V max
        vmax_layout = QHBoxLayout()
        vmax_layout.addWidget(QLabel("V max:"))
        self.v_max_spin = QSpinBox()
        self.v_max_spin.setRange(100, 10000)
        self.v_max_spin.setValue(int(self.config.v_max))
        self.v_max_spin.setSuffix(" m/s")
        vmax_layout.addWidget(self.v_max_spin)
        filter_layout.addLayout(vmax_layout)

        # Taper
        taper_layout = QHBoxLayout()
        taper_layout.addWidget(QLabel("Taper:"))
        self.taper_spin = QDoubleSpinBox()
        self.taper_spin.setRange(0.01, 0.50)
        self.taper_spin.setValue(self.config.taper_width)
        self.taper_spin.setSingleStep(0.05)
        taper_layout.addWidget(self.taper_spin)
        filter_layout.addLayout(taper_layout)

        # Azimuth
        az_layout = QHBoxLayout()
        az_layout.addWidget(QLabel("Az:"))
        self.az_min_spin = QSpinBox()
        self.az_min_spin.setRange(0, 360)
        self.az_min_spin.setValue(0)
        self.az_min_spin.setSuffix("°")
        az_layout.addWidget(self.az_min_spin)
        az_layout.addWidget(QLabel("-"))
        self.az_max_spin = QSpinBox()
        self.az_max_spin.setRange(0, 360)
        self.az_max_spin.setValue(360)
        self.az_max_spin.setSuffix("°")
        az_layout.addWidget(self.az_max_spin)
        filter_layout.addLayout(az_layout)

        layout.addWidget(filter_group)

        # Advanced options (collapsible)
        adv_group = QGroupBox("Advanced")
        adv_layout = QVBoxLayout(adv_group)

        # AGC
        agc_layout = QHBoxLayout()
        self.agc_checkbox = QCheckBox("AGC")
        self.agc_checkbox.setChecked(self.config.apply_agc)
        agc_layout.addWidget(self.agc_checkbox)
        self.agc_window_spin = QSpinBox()
        self.agc_window_spin.setRange(50, 5000)
        self.agc_window_spin.setValue(int(self.config.agc_window_ms))
        self.agc_window_spin.setSuffix("ms")
        agc_layout.addWidget(self.agc_window_spin)
        adv_layout.addLayout(agc_layout)

        # Freq band
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("f:"))
        self.f_min_spin = QSpinBox()
        self.f_min_spin.setRange(0, 500)
        self.f_min_spin.setValue(0)
        self.f_min_spin.setSuffix("Hz")
        freq_layout.addWidget(self.f_min_spin)
        freq_layout.addWidget(QLabel("-"))
        nyquist = int(0.5 / self.volume.dt)
        self.f_max_spin = QSpinBox()
        self.f_max_spin.setRange(1, nyquist)
        self.f_max_spin.setValue(nyquist)
        self.f_max_spin.setSuffix("Hz")
        freq_layout.addWidget(self.f_max_spin)
        adv_layout.addLayout(freq_layout)

        # Temporal taper
        ttaper_layout = QHBoxLayout()
        ttaper_layout.addWidget(QLabel("T-taper:"))
        self.taper_top_spin = QSpinBox()
        self.taper_top_spin.setRange(0, 500)
        self.taper_top_spin.setValue(int(self.config.taper_ms_top))
        self.taper_top_spin.setSuffix("ms")
        ttaper_layout.addWidget(self.taper_top_spin)
        adv_layout.addLayout(ttaper_layout)

        # Edge padding
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("Edge pad:"))
        self.pad_traces_spin = QSpinBox()
        self.pad_traces_spin.setRange(0, 50)
        self.pad_traces_spin.setValue(self.config.pad_traces_x)
        edge_layout.addWidget(self.pad_traces_spin)
        adv_layout.addLayout(edge_layout)

        layout.addWidget(adv_group)

        # Display options
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        # Seismic colormap
        seis_cmap_layout = QHBoxLayout()
        seis_cmap_layout.addWidget(QLabel("Seismic:"))
        self.seismic_cmap_combo = QComboBox()
        for cmap in SEISMIC_COLORMAPS:
            self.seismic_cmap_combo.addItem(cmap.replace('_', ' ').title(), cmap)
        seis_cmap_layout.addWidget(self.seismic_cmap_combo)
        display_layout.addLayout(seis_cmap_layout)

        # Spectrum colormap
        spec_cmap_layout = QHBoxLayout()
        spec_cmap_layout.addWidget(QLabel("Spectrum:"))
        self.spectrum_cmap_combo = QComboBox()
        for cmap in SPECTRUM_COLORMAPS:
            self.spectrum_cmap_combo.addItem(cmap.replace('_', ' ').title(), cmap)
        spec_cmap_layout.addWidget(self.spectrum_cmap_combo)
        display_layout.addLayout(spec_cmap_layout)

        # dB mode for spectrum
        db_layout = QHBoxLayout()
        self.db_checkbox = QCheckBox("dB scale")
        self.db_checkbox.setChecked(True)
        self.db_checkbox.setToolTip("Display spectrum in dB (20*log10)")
        db_layout.addWidget(self.db_checkbox)
        db_layout.addWidget(QLabel("Range:"))
        self.db_range_spin = QSpinBox()
        self.db_range_spin.setRange(20, 120)
        self.db_range_spin.setValue(60)
        self.db_range_spin.setSuffix(" dB")
        self.db_range_spin.setToolTip("Dynamic range for dB display")
        db_layout.addWidget(self.db_range_spin)
        display_layout.addLayout(db_layout)

        layout.addWidget(display_group)

        # Preset
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", None)
        for name in FKK_PRESETS.keys():
            self.preset_combo.addItem(name, name)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # Action buttons
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        layout.addWidget(self.apply_btn)

        # Separator
        layout.addWidget(self._create_separator())

        # Accept/Cancel
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.cancel_btn)
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self.accept_btn)
        layout.addLayout(btn_layout)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        return widget

    def _create_separator(self) -> QFrame:
        """Create horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _create_display_panel(self) -> QWidget:
        """Create the 4-quadrant display panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Top row: Data views with toggle
        data_header = QWidget()
        data_header_layout = QHBoxLayout(data_header)
        data_header_layout.setContentsMargins(0, 0, 0, 0)

        data_label = QLabel("<b>SEISMIC DATA</b>")
        data_label.setStyleSheet("color: #fff; font-size: 11pt;")
        data_header_layout.addWidget(data_label)
        data_header_layout.addStretch()

        self.data_toggle_btn = QPushButton("Show: Filtered")
        self.data_toggle_btn.setCheckable(True)
        self.data_toggle_btn.setChecked(True)
        self.data_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
            }
        """)
        data_header_layout.addWidget(self.data_toggle_btn)

        self.diff_btn = QPushButton("Difference")
        self.diff_btn.setCheckable(True)
        self.diff_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #FF9800;
                color: white;
            }
        """)
        data_header_layout.addWidget(self.diff_btn)
        layout.addWidget(data_header)

        # Data views row (XX and TX)
        data_splitter = QSplitter(Qt.Orientation.Horizontal)

        # XX view (time slice X-Y)
        self.xx_view = SeismicImageView("Time Slice (X-Y)")
        data_splitter.addWidget(self.xx_view)

        # TX view (inline T-X)
        self.tx_view = SeismicImageView("Inline (T-X)")
        data_splitter.addWidget(self.tx_view)

        data_splitter.setStretchFactor(0, 1)
        data_splitter.setStretchFactor(1, 1)
        layout.addWidget(data_splitter, stretch=1)

        # Bottom row: Spectrum views with toggle
        spec_header = QWidget()
        spec_header_layout = QHBoxLayout(spec_header)
        spec_header_layout.setContentsMargins(0, 0, 0, 0)

        spec_label = QLabel("<b>FKK SPECTRUM</b>")
        spec_label.setStyleSheet("color: #fff; font-size: 11pt;")
        spec_header_layout.addWidget(spec_label)
        spec_header_layout.addStretch()

        self.spec_toggle_btn = QPushButton("Show: Filtered")
        self.spec_toggle_btn.setCheckable(True)
        self.spec_toggle_btn.setChecked(True)
        self.spec_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
            }
        """)
        spec_header_layout.addWidget(self.spec_toggle_btn)

        self.show_mask_btn = QPushButton("Show Mask")
        self.show_mask_btn.setCheckable(True)
        self.show_mask_btn.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #9C27B0;
                color: white;
            }
        """)
        spec_header_layout.addWidget(self.show_mask_btn)
        layout.addWidget(spec_header)

        # Spectrum views row (KK and FK)
        spec_splitter = QSplitter(Qt.Orientation.Horizontal)

        # KK view (Kx-Ky at frequency)
        self.kk_view = SeismicImageView("Kx-Ky @ frequency")
        spec_splitter.addWidget(self.kk_view)

        # FK view (F-Kx at ky)
        self.fk_view = SeismicImageView("F-Kx @ ky")
        spec_splitter.addWidget(self.fk_view)

        spec_splitter.setStretchFactor(0, 1)
        spec_splitter.setStretchFactor(1, 1)
        layout.addWidget(spec_splitter, stretch=1)

        return widget

    def _connect_signals(self):
        """Connect UI signals."""
        # Navigation
        self.time_slider.valueChanged.connect(self._on_time_changed)
        self.inline_slider.valueChanged.connect(self._on_inline_changed)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        self.ky_slider.valueChanged.connect(self._on_ky_changed)

        # Filter parameters
        self.mode_group.buttonClicked.connect(self._on_param_changed)
        self.v_min_spin.valueChanged.connect(self._on_param_changed)
        self.v_max_spin.valueChanged.connect(self._on_param_changed)
        self.taper_spin.valueChanged.connect(self._on_param_changed)
        self.az_min_spin.valueChanged.connect(self._on_param_changed)
        self.az_max_spin.valueChanged.connect(self._on_param_changed)
        self.agc_checkbox.stateChanged.connect(self._on_param_changed)
        self.agc_window_spin.valueChanged.connect(self._on_param_changed)
        self.f_min_spin.valueChanged.connect(self._on_param_changed)
        self.f_max_spin.valueChanged.connect(self._on_param_changed)
        self.taper_top_spin.valueChanged.connect(self._on_param_changed)
        self.pad_traces_spin.valueChanged.connect(self._on_param_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        # Toggle buttons
        self.data_toggle_btn.clicked.connect(self._on_data_toggle)
        self.diff_btn.clicked.connect(self._on_diff_toggle)
        self.spec_toggle_btn.clicked.connect(self._on_spec_toggle)
        self.show_mask_btn.clicked.connect(self._on_mask_toggle)

        # Display options
        self.seismic_cmap_combo.currentIndexChanged.connect(self._on_seismic_cmap_changed)
        self.spectrum_cmap_combo.currentIndexChanged.connect(self._on_spectrum_cmap_changed)
        self.db_checkbox.stateChanged.connect(self._on_db_mode_changed)
        self.db_range_spin.valueChanged.connect(self._on_db_range_changed)

        # Actions
        self.apply_btn.clicked.connect(self._request_apply_filter)
        self.cancel_btn.clicked.connect(self.reject)
        self.accept_btn.clicked.connect(self._accept_result)

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_time_changed(self, value: int):
        self.t_idx = value
        self.time_label.setText(f"{value * self.volume.dt * 1000:.0f}ms")
        self._update_data_views()

    def _on_inline_changed(self, value: int):
        self.y_idx = value
        self.inline_label.setText(f"{value}")
        self._update_data_views()

    def _on_freq_changed(self, value: int):
        self.f_idx = value
        if self.spectrum_axes:
            freq = self.spectrum_axes['f_axis'][value] if value < len(self.spectrum_axes['f_axis']) else 0
            self.freq_label.setText(f"{freq:.1f}Hz")
        self._update_spectrum_views()

    def _on_ky_changed(self, value: int):
        self.ky_idx = value
        if self.spectrum_axes:
            ky = self.spectrum_axes['ky_axis'][value] if value < len(self.spectrum_axes['ky_axis']) else 0
            self.ky_label.setText(f"{ky:.3f}")
        self._update_spectrum_views()

    def _on_param_changed(self, *args):
        """Update config from UI and request filter."""
        self._update_config_from_ui()
        self._request_apply_filter()

    def _update_config_from_ui(self):
        """Update config from current UI values."""
        self.config.mode = 'reject' if self.mode_reject.isChecked() else 'pass'
        self.config.v_min = float(self.v_min_spin.value())
        self.config.v_max = float(self.v_max_spin.value())
        self.config.taper_width = self.taper_spin.value()
        self.config.azimuth_min = float(self.az_min_spin.value())
        self.config.azimuth_max = float(self.az_max_spin.value())
        self.config.apply_agc = self.agc_checkbox.isChecked()
        self.config.agc_window_ms = float(self.agc_window_spin.value())
        self.config.f_min = float(self.f_min_spin.value()) if self.f_min_spin.value() > 0 else None
        self.config.f_max = float(self.f_max_spin.value()) if self.f_max_spin.value() < int(0.5/self.volume.dt) else None
        self.config.taper_ms_top = float(self.taper_top_spin.value())
        self.config.taper_ms_bottom = float(self.taper_top_spin.value())
        self.config.pad_traces_x = self.pad_traces_spin.value()
        self.config.pad_traces_y = self.pad_traces_spin.value()

    def _on_preset_changed(self, index: int):
        name = self.preset_combo.currentData()
        if name and name in FKK_PRESETS:
            preset = FKK_PRESETS[name]
            self._set_ui_from_config(preset)
            self._request_apply_filter()

    def _set_ui_from_config(self, config: FKKConfig):
        """Update UI from config."""
        self.v_min_spin.blockSignals(True)
        self.v_max_spin.blockSignals(True)
        self.taper_spin.blockSignals(True)

        self.v_min_spin.setValue(int(config.v_min))
        self.v_max_spin.setValue(int(config.v_max))
        self.taper_spin.setValue(config.taper_width)

        if config.mode == 'reject':
            self.mode_reject.setChecked(True)
        else:
            self.mode_pass.setChecked(True)

        self.v_min_spin.blockSignals(False)
        self.v_max_spin.blockSignals(False)
        self.taper_spin.blockSignals(False)

        self.config = config.copy()

    def _on_data_toggle(self):
        """Toggle between filtered and original data."""
        self._show_filtered_data = self.data_toggle_btn.isChecked()
        self.data_toggle_btn.setText("Show: Filtered" if self._show_filtered_data else "Show: Original")
        self._update_data_views()

    def _on_diff_toggle(self):
        """Toggle difference view."""
        self._update_data_views()

    def _on_spec_toggle(self):
        """Toggle between filtered and original spectrum."""
        self._show_filtered_spectrum = self.spec_toggle_btn.isChecked()
        self.spec_toggle_btn.setText("Show: Filtered" if self._show_filtered_spectrum else "Show: Original")
        self._update_spectrum_views()

    def _on_mask_toggle(self):
        """Toggle mask overlay."""
        self._update_spectrum_views()

    def _on_seismic_cmap_changed(self, index: int):
        """Change seismic colormap."""
        cmap_name = self.seismic_cmap_combo.currentData()
        if cmap_name:
            self.xx_view.setColormap(cmap_name, is_spectrum=False)
            self.tx_view.setColormap(cmap_name, is_spectrum=False)
            self._update_data_views()

    def _on_spectrum_cmap_changed(self, index: int):
        """Change spectrum colormap."""
        cmap_name = self.spectrum_cmap_combo.currentData()
        if cmap_name:
            self.kk_view.setColormap(cmap_name, is_spectrum=True)
            self.fk_view.setColormap(cmap_name, is_spectrum=True)
            self._update_spectrum_views()

    def _on_db_mode_changed(self, state: int):
        """Toggle dB mode for spectrum display."""
        use_db = state == Qt.CheckState.Checked.value
        db_range = self.db_range_spin.value()
        self.kk_view.setDbMode(use_db, db_range)
        self.fk_view.setDbMode(use_db, db_range)
        self._update_spectrum_views()

    def _on_db_range_changed(self, value: int):
        """Change dB dynamic range."""
        if self.db_checkbox.isChecked():
            self.kk_view.setDbMode(True, value)
            self.fk_view.setDbMode(True, value)
            self._update_spectrum_views()

    # =========================================================================
    # Processing
    # =========================================================================

    def _compute_spectrum(self):
        """Compute FKK spectrum."""
        self.status_label.setText("Computing spectrum...")
        try:
            self.spectrum = self.processor.compute_spectrum(self.volume)
            self.spectrum_axes = self.processor._compute_axes(self.volume)

            # Update frequency label
            if self.spectrum_axes and self.f_idx < len(self.spectrum_axes['f_axis']):
                freq = self.spectrum_axes['f_axis'][self.f_idx]
                self.freq_label.setText(f"{freq:.1f}Hz")

            self._update_spectrum_views()
            self.status_label.setText("Spectrum ready")
        except Exception as e:
            logger.error(f"Spectrum computation failed: {e}")
            self.status_label.setText(f"Error: {e}")

    def _request_apply_filter(self):
        """Request filter with debouncing."""
        self._preview_timer.start(200)

    def _do_apply_filter(self):
        """Apply filter."""
        self.status_label.setText("Applying filter...")
        try:
            self.filtered_volume = self.processor.apply_filter(self.volume, self.config)

            # Compute filtered spectrum
            if self.filtered_volume is not None:
                self.filtered_spectrum = self.processor.compute_spectrum(self.filtered_volume)

            self._update_all_views()
            self.status_label.setText(f"Filter: {self.config.get_summary()}")
        except Exception as e:
            logger.error(f"Filter failed: {e}")
            self.status_label.setText(f"Filter error: {e}")

    # =========================================================================
    # View Updates
    # =========================================================================

    def _update_all_views(self):
        self._update_data_views()
        self._update_spectrum_views()

    def _update_data_views(self):
        """Update XX and TX data views."""
        show_diff = self.diff_btn.isChecked()

        # Get slices
        input_xx = self.volume.time_slice(self.t_idx)
        input_tx = self.volume.inline_slice(self.y_idx)

        if self.filtered_volume is not None and (self._show_filtered_data or show_diff):
            filt_xx = self.filtered_volume.time_slice(self.t_idx)
            filt_tx = self.filtered_volume.inline_slice(self.y_idx)

            if show_diff:
                # Show difference (rejected noise)
                display_xx = input_xx - filt_xx
                display_tx = input_tx - filt_tx
            elif self._show_filtered_data:
                display_xx = filt_xx
                display_tx = filt_tx
            else:
                display_xx = input_xx
                display_tx = input_tx
        else:
            display_xx = input_xx
            display_tx = input_tx

        # Update views
        self.xx_view.setData(display_xx, is_spectrum=False, xlabel='X', ylabel='Y')
        self.tx_view.setData(display_tx, is_spectrum=False, xlabel='X', ylabel='Time')

    def _update_spectrum_views(self):
        """Update KK and FK spectrum views."""
        if self.spectrum is None:
            return

        # Choose which spectrum to show
        if self._show_filtered_spectrum and self.filtered_spectrum is not None:
            spec = self.filtered_spectrum
        else:
            spec = self.spectrum

        # Safe index bounds
        f_idx = min(self.f_idx, spec.shape[0] - 1)
        ky_idx = min(self.ky_idx, spec.shape[2] - 1)

        # KK view (Kx-Ky at frequency) - pass raw amplitude, dB conversion in view
        kk_slice = spec[f_idx, :, :]

        # FK view (F-Kx at ky) - pass raw amplitude, dB conversion in view
        fk_slice = spec[:, :, ky_idx]

        # Pass raw amplitude data - SeismicImageView handles dB conversion
        self.kk_view.setData(kk_slice, is_spectrum=True, xlabel='Kx', ylabel='Ky')
        self.fk_view.setData(fk_slice, is_spectrum=True, xlabel='Kx', ylabel='Freq')

    # =========================================================================
    # Actions
    # =========================================================================

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
        return self.filtered_volume

    def get_config(self) -> FKKConfig:
        return self.config.copy()

    def closeEvent(self, event):
        self._preview_timer.stop()
        if hasattr(self.processor, 'clear_cache'):
            self.processor.clear_cache()
        super().closeEvent(event)
