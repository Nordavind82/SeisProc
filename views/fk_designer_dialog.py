"""
FK Filter Designer Dialog

Interactive dialog for designing FK filters in Design mode.
Allows user to visualize FK spectrum, adjust parameters, preview results,
and save configurations.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QLineEdit, QGroupBox, QCheckBox,
    QSplitter, QWidget, QMessageBox, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Optional, List
import pyqtgraph as pg

from models.seismic_data import SeismicData
from models.fk_config import FKFilterConfig, FKConfigManager, SubGather
from processors.fk_filter import FKFilter, get_fk_filter_presets
from processors.agc import apply_agc_to_gather, remove_agc
from utils.subgather_detector import (
    detect_subgathers, extract_subgather_traces,
    calculate_subgather_trace_spacing, get_available_boundary_headers,
    validate_subgather_boundaries
)
from utils.trace_spacing import (
    calculate_trace_spacing_with_stats,
    calculate_subgather_trace_spacing_with_stats,
    format_spacing_stats,
    TraceSpacingStats,
    analyze_offset_step_uniformity,
    OffsetStepAnalysis
)
from utils.unit_conversion import (
    UnitConverter, format_distance, format_velocity,
    get_velocity_range_for_units, get_taper_range_for_units
)
from datetime import datetime

# Debug flag - set to True for verbose output during development
FK_DEBUG = False


class FKDesignerDialog(QDialog):
    """
    FK Filter Designer dialog for interactive filter design.

    Provides:
    - FK spectrum visualization
    - Interactive parameter adjustment
    - Side-by-side preview (Input | Filtered | Rejected)
    - Preset selection
    - Configuration saving
    """

    config_saved = pyqtSignal(FKFilterConfig)  # Emitted when config is saved

    def __init__(
        self,
        gather_data: SeismicData,
        gather_index: int,
        trace_spacing: float,
        gather_headers: Optional[pd.DataFrame] = None,
        parent=None
    ):
        """
        Initialize FK Designer dialog.

        Args:
            gather_data: Seismic gather to design filter on
            gather_index: Index of current gather
            trace_spacing: Spatial distance between traces (m)
            gather_headers: Optional DataFrame with trace headers (for sub-gathers)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("FK Filter Designer")
        self.resize(1400, 900)

        self.gather_data = gather_data
        self.gather_index = gather_index
        self.trace_spacing = trace_spacing
        self.gather_headers = gather_headers

        # DEBUG: Initialization
        if FK_DEBUG:
            print("\n" + "="*80)
            print("FK DESIGNER INITIALIZATION DEBUG")
            print("="*80)
            print(f"Data shape: {gather_data.traces.shape}")
            print(f"Coordinate units: {gather_data.coordinate_units}")
            print(f"Unit symbol: {gather_data.unit_symbol}")
            print(f"Initial trace_spacing param: {trace_spacing}")
            print(f"Has headers: {gather_headers is not None}")
            if gather_headers is not None:
                print(f"  Headers shape: {gather_headers.shape}")
                if 'GroupX' in gather_headers.columns:
                    gx = gather_headers['GroupX'].values[:5]
                    print(f"  First 5 GroupX: {gx}")
                    if len(gather_headers) > 1:
                        spacing_calc = abs(gather_headers['GroupX'].values[1] - gather_headers['GroupX'].values[0])
                        print(f"  GroupX[1] - GroupX[0]: {spacing_calc}")

        # Current filter parameters
        self.filter_type = 'velocity'  # 'velocity' or 'dip'
        self.v_min = 2000.0
        self.v_max = 6000.0
        self.v_min_enabled = True
        self.v_max_enabled = True
        self.dip_min = -0.01  # s/m (negative dip)
        self.dip_max = 0.01   # s/m (positive dip)
        self.dip_min_enabled = True
        self.dip_max_enabled = True
        self.taper_width = 300.0
        self.mode = 'pass'

        # Sub-gather state
        self.use_subgathers = False
        self.boundary_header: Optional[str] = None
        self.subgathers: List[SubGather] = []
        self.current_subgather_index = 0
        self.current_subgather: Optional[SubGather] = None

        # AGC state
        self.apply_agc = False
        self.agc_window_ms = 500.0
        self.agc_scale_factors: Optional[np.ndarray] = None
        self.preview_with_agc = False  # For preview toggle

        # FK display options
        self.fk_smoothing = 0  # 0-5 smoothing level
        self.fk_gain = 1.0     # 0.0001-10000 gain factor (logarithmic)
        self.fk_colormap = "Hot"  # Colormap name
        self.fk_show_filtered = False  # Show filtered FK spectrum instead of input
        self.fk_contrast_low = 1    # percentile for low clip
        self.fk_contrast_high = 99  # percentile for high clip
        self.interactive_boundaries = False  # Enable draggable velocity lines

        # Draggable velocity lines (created when enabled)
        self.vmin_line_pos = None
        self.vmin_line_neg = None
        self.vmax_line_pos = None
        self.vmax_line_neg = None

        # Working data (current sub-gather or full gather)
        self.working_data: SeismicData = gather_data  # Initially full gather
        self.working_trace_spacing: float = trace_spacing

        # Processed results (computed on demand)
        self.filtered_data: Optional[SeismicData] = None
        self.rejected_data: Optional[SeismicData] = None
        self.fk_spectrum: Optional[np.ndarray] = None
        self.freqs: Optional[np.ndarray] = None
        self.wavenumbers: Optional[np.ndarray] = None

        # FK spectrum cache for memoization
        # Cache key is hash of (data_id, trace_spacing, agc_state)
        self._fk_cache_key: Optional[str] = None
        self._cached_fk_spectrum: Optional[np.ndarray] = None
        self._cached_freqs: Optional[np.ndarray] = None
        self._cached_wavenumbers: Optional[np.ndarray] = None

        # Auto-update flag
        self.auto_update = True

        # Gap detection analysis
        self.offset_analysis: Optional[OffsetStepAnalysis] = None

        self._init_ui()
        self._check_offset_gaps()
        self._compute_fk_spectrum()
        self._apply_filter()
        self._update_displays()

        # Note: Units are now taken from data's coordinate_units, not app settings

    def _get_velocity_label(self, velocity: float) -> str:
        """Format velocity with native units."""
        unit = self.working_data.unit_symbol
        return f"{velocity:.0f} {unit}/s"

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Main splitter (horizontal: left controls | right displays)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(6)  # Wider handle for easier dragging

        # Style the splitter handle to make it more visible
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #c0c0c0;
                border: 1px solid #808080;
            }
            QSplitter::handle:hover {
                background-color: #a0a0ff;
            }
            QSplitter::handle:pressed {
                background-color: #8080ff;
            }
        """)

        # Left panel: Controls (scrollable)
        controls_panel = self._create_controls_panel()
        main_splitter.addWidget(controls_panel)

        # Right panel: FK spectrum + Preview
        display_panel = self._create_display_panel()
        main_splitter.addWidget(display_panel)

        # Set splitter sizes (20% controls, 80% displays for better viewing)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 8)

        # Set minimum width for controls panel
        main_splitter.setCollapsible(0, False)  # Don't allow collapsing controls

        # Store splitter for potential saving/restoring of state
        self.main_splitter = main_splitter

        layout.addWidget(main_splitter)

        # Bottom buttons
        bottom_buttons = self._create_bottom_buttons()
        layout.addLayout(bottom_buttons)

        self.setLayout(layout)

    def _create_header(self) -> QWidget:
        """Create header with gather info."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        n_samples, n_traces = self.gather_data.traces.shape
        label = QLabel(
            f"Gather {self.gather_index}: "
            f"{n_traces} traces × {n_samples} samples"
        )
        label.setStyleSheet("font-weight: bold; font-size: 11pt;")

        layout.addWidget(label)
        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def _create_controls_panel(self) -> QWidget:
        """Create left panel with filter controls (scrollable)."""
        from PyQt6.QtWidgets import QScrollArea

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Content widget
        content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # FK Display Options (moved from right panel)
        fk_display_group = self._create_fk_display_controls()
        layout.addWidget(fk_display_group)

        # Preset selection
        preset_group = self._create_preset_group()
        layout.addWidget(preset_group)

        # Sub-gather controls (if headers available)
        if self.gather_headers is not None:
            subgather_group = self._create_subgather_group()
            layout.addWidget(subgather_group)

        # AGC controls
        agc_group = self._create_agc_group()
        layout.addWidget(agc_group)

        # Parameter controls
        param_group = self._create_parameter_group()
        layout.addWidget(param_group)

        # Quality metrics
        metrics_group = self._create_metrics_group()
        layout.addWidget(metrics_group)

        layout.addStretch()

        content.setLayout(layout)
        scroll.setWidget(content)

        return scroll

    def _create_preset_group(self) -> QGroupBox:
        """Create preset selection group."""
        group = QGroupBox("Quick Presets")
        layout = QVBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", None)

        # Get presets with correct velocity units based on data's coordinate system
        presets = get_fk_filter_presets(self.gather_data.coordinate_units)
        for name, params in presets.items():
            self.preset_combo.addItem(name, params)

        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)

        layout.addWidget(QLabel("Select Preset:"))
        layout.addWidget(self.preset_combo)

        group.setLayout(layout)
        return group

    def _create_subgather_group(self) -> QGroupBox:
        """Create sub-gather controls group."""
        group = QGroupBox("Sub-Gathers (Optional)")
        layout = QVBoxLayout()

        # Enable checkbox
        self.subgather_checkbox = QCheckBox("Split gather by header changes")
        self.subgather_checkbox.stateChanged.connect(self._on_subgather_enabled_changed)
        layout.addWidget(self.subgather_checkbox)

        # Container for controls (hidden initially)
        self.subgather_controls = QWidget()
        sg_layout = QVBoxLayout()
        sg_layout.setContentsMargins(10, 0, 0, 0)

        # Boundary header selection
        sg_layout.addWidget(QLabel("Boundary Header:"))
        self.boundary_header_combo = QComboBox()

        # Populate with available headers
        if self.gather_headers is not None:
            available_headers = get_available_boundary_headers(self.gather_headers)
            self.boundary_header_combo.addItems(available_headers)

        self.boundary_header_combo.currentTextChanged.connect(self._on_boundary_header_changed)
        sg_layout.addWidget(self.boundary_header_combo)

        # Sub-gather info label
        self.subgather_info_label = QLabel("Detected: 0 sub-gathers")
        self.subgather_info_label.setStyleSheet("color: #666; font-size: 9pt;")
        sg_layout.addWidget(self.subgather_info_label)

        # Navigation controls
        nav_layout = QHBoxLayout()

        self.subgather_current_label = QLabel("Current: -")
        nav_layout.addWidget(self.subgather_current_label)

        nav_layout.addStretch()

        self.prev_subgather_btn = QPushButton("◄ Prev")
        self.prev_subgather_btn.clicked.connect(self._on_prev_subgather)
        nav_layout.addWidget(self.prev_subgather_btn)

        self.next_subgather_btn = QPushButton("Next ►")
        self.next_subgather_btn.clicked.connect(self._on_next_subgather)
        nav_layout.addWidget(self.next_subgather_btn)

        sg_layout.addLayout(nav_layout)

        self.subgather_controls.setLayout(sg_layout)
        self.subgather_controls.hide()
        layout.addWidget(self.subgather_controls)

        group.setLayout(layout)
        return group

    def _create_agc_group(self) -> QGroupBox:
        """Create AGC controls group."""
        group = QGroupBox("AGC Pre-Conditioning (Optional)")
        layout = QVBoxLayout()

        # Enable checkbox
        self.agc_checkbox = QCheckBox("Apply AGC before FK filtering")
        self.agc_checkbox.stateChanged.connect(self._on_agc_enabled_changed)
        layout.addWidget(self.agc_checkbox)

        # Container for controls (hidden initially)
        self.agc_controls = QWidget()
        agc_layout = QVBoxLayout()
        agc_layout.setContentsMargins(10, 0, 0, 0)

        # AGC window length
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("AGC Window (ms):"))

        self.agc_window_spin = QSpinBox()
        self.agc_window_spin.setRange(50, 2000)
        self.agc_window_spin.setValue(500)
        self.agc_window_spin.setSingleStep(50)
        self.agc_window_spin.valueChanged.connect(self._on_agc_window_changed)
        window_layout.addWidget(self.agc_window_spin)

        agc_layout.addLayout(window_layout)

        # Info label
        info_label = QLabel(
            "ℹ AGC equalizes amplitudes for better FK filtering,\n"
            "then is removed from output"
        )
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        info_label.setWordWrap(True)
        agc_layout.addWidget(info_label)

        # Preview toggle
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("FK Preview:"))

        self.agc_preview_group = QButtonGroup()

        self.agc_preview_without = QRadioButton("Without AGC")
        self.agc_preview_without.setChecked(True)
        self.agc_preview_group.addButton(self.agc_preview_without, 0)
        preview_layout.addWidget(self.agc_preview_without)

        self.agc_preview_with = QRadioButton("With AGC")
        self.agc_preview_group.addButton(self.agc_preview_with, 1)
        preview_layout.addWidget(self.agc_preview_with)

        self.agc_preview_group.buttonClicked.connect(self._on_agc_preview_changed)

        agc_layout.addLayout(preview_layout)

        self.agc_controls.setLayout(agc_layout)
        self.agc_controls.hide()
        layout.addWidget(self.agc_controls)

        group.setLayout(layout)
        return group

    def _create_parameter_group(self) -> QGroupBox:
        """Create parameter adjustment group."""
        group = QGroupBox("Filter Parameters")
        layout = QVBoxLayout()

        # Filter type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Filter Type:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItem("Velocity", 'velocity')
        self.filter_type_combo.addItem("Dip", 'dip')
        # Manual mode removed - not yet implemented
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        type_layout.addWidget(self.filter_type_combo)
        layout.addLayout(type_layout)

        # Mode selection (pass/reject)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Pass (keep)", 'pass')
        self.mode_combo.addItem("Reject (remove)", 'reject')
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # === VELOCITY PARAMETERS ===
        self.velocity_widget = QWidget()
        velocity_layout = QVBoxLayout()
        velocity_layout.setContentsMargins(0, 0, 0, 0)

        # V_min with enable checkbox
        vmin_header = QHBoxLayout()
        self.v_min_enable = QCheckBox("Min Velocity:")
        self.v_min_enable.setChecked(self.v_min_enabled)
        self.v_min_enable.stateChanged.connect(self._on_v_min_enable_changed)
        vmin_header.addWidget(self.v_min_enable)
        velocity_layout.addLayout(vmin_header)

        self.v_min_slider = QSlider(Qt.Orientation.Horizontal)
        # Get velocity range based on coordinate units
        v_range = get_velocity_range_for_units(self.gather_data.coordinate_units)
        self.v_min_slider.setRange(v_range[0], v_range[1])
        self.v_min_slider.setValue(int(self.v_min))
        self.v_min_slider.setToolTip(
            "Minimum velocity boundary (m/s or ft/s).\n"
            "In 'Pass' mode: Energy with apparent velocity below this is rejected.\n"
            "In 'Reject' mode: Energy with apparent velocity below this is kept."
        )
        self.v_min_slider.valueChanged.connect(self._on_v_min_changed)

        self.v_min_label = QLabel(self._get_velocity_label(self.v_min))
        vmin_layout = QHBoxLayout()
        vmin_layout.addWidget(self.v_min_slider)
        vmin_layout.addWidget(self.v_min_label)
        velocity_layout.addLayout(vmin_layout)

        # V_max with enable checkbox
        vmax_header = QHBoxLayout()
        self.v_max_enable = QCheckBox("Max Velocity:")
        self.v_max_enable.setChecked(self.v_max_enabled)
        self.v_max_enable.stateChanged.connect(self._on_v_max_enable_changed)
        vmax_header.addWidget(self.v_max_enable)
        velocity_layout.addLayout(vmax_header)

        self.v_max_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_max_slider.setRange(v_range[0], v_range[1])  # Use same range as v_min
        self.v_max_slider.setValue(int(self.v_max))
        self.v_max_slider.setToolTip(
            "Maximum velocity boundary (m/s or ft/s).\n"
            "In 'Pass' mode: Energy with apparent velocity above this is rejected.\n"
            "In 'Reject' mode: Energy with apparent velocity above this is kept."
        )
        self.v_max_slider.valueChanged.connect(self._on_v_max_changed)

        self.v_max_label = QLabel(self._get_velocity_label(self.v_max))
        vmax_layout = QHBoxLayout()
        vmax_layout.addWidget(self.v_max_slider)
        vmax_layout.addWidget(self.v_max_label)
        velocity_layout.addLayout(vmax_layout)

        self.velocity_widget.setLayout(velocity_layout)
        layout.addWidget(self.velocity_widget)

        # === DIP PARAMETERS ===
        self.dip_widget = QWidget()
        dip_layout = QVBoxLayout()
        dip_layout.setContentsMargins(0, 0, 0, 0)

        # Info label with correct unit
        dip_unit = self.gather_data.unit_symbol
        dip_info = QLabel(f"Dip in s/{dip_unit} (dt/dx)")
        dip_info.setStyleSheet("QLabel { font-size: 10px; color: gray; }")
        dip_layout.addWidget(dip_info)

        # Negative dip (dip_min) with enable checkbox
        dipmin_header = QHBoxLayout()
        self.dip_min_enable = QCheckBox("Negative Dip:")
        self.dip_min_enable.setChecked(self.dip_min_enabled)
        self.dip_min_enable.stateChanged.connect(self._on_dip_min_enable_changed)
        dipmin_header.addWidget(self.dip_min_enable)
        dip_layout.addLayout(dipmin_header)

        # Use spinbox for dip (finer control needed)
        self.dip_min_spin = QDoubleSpinBox()
        self.dip_min_spin.setRange(-1.0, 0.0)
        self.dip_min_spin.setValue(self.dip_min)
        self.dip_min_spin.setSingleStep(0.001)
        self.dip_min_spin.setDecimals(4)
        self.dip_min_spin.setSuffix(f" s/{dip_unit}")
        self.dip_min_spin.setToolTip(
            f"Negative dip boundary (dt/dx in s/{dip_unit}).\n"
            "Defines the steepest left-dipping events to filter.\n"
            "More negative = steeper left-dip."
        )
        self.dip_min_spin.valueChanged.connect(self._on_dip_min_changed)
        dip_layout.addWidget(self.dip_min_spin)

        # Positive dip (dip_max) with enable checkbox
        dipmax_header = QHBoxLayout()
        self.dip_max_enable = QCheckBox("Positive Dip:")
        self.dip_max_enable.setChecked(self.dip_max_enabled)
        self.dip_max_enable.stateChanged.connect(self._on_dip_max_enable_changed)
        dipmax_header.addWidget(self.dip_max_enable)
        dip_layout.addLayout(dipmax_header)

        self.dip_max_spin = QDoubleSpinBox()
        self.dip_max_spin.setRange(0.0, 1.0)
        self.dip_max_spin.setValue(self.dip_max)
        self.dip_max_spin.setSingleStep(0.001)
        self.dip_max_spin.setDecimals(4)
        self.dip_max_spin.setSuffix(f" s/{dip_unit}")
        self.dip_max_spin.setToolTip(
            f"Positive dip boundary (dt/dx in s/{dip_unit}).\n"
            "Defines the steepest right-dipping events to filter.\n"
            "More positive = steeper right-dip."
        )
        self.dip_max_spin.valueChanged.connect(self._on_dip_max_changed)
        dip_layout.addWidget(self.dip_max_spin)

        self.dip_widget.setLayout(dip_layout)
        layout.addWidget(self.dip_widget)
        self.dip_widget.setVisible(False)  # Hidden by default

        # Manual mode widget removed - not yet implemented

        # === COMMON PARAMETERS ===
        # Taper width slider
        layout.addWidget(QLabel("Taper Width:"))
        self.taper_slider = QSlider(Qt.Orientation.Horizontal)
        taper_range = get_taper_range_for_units(self.gather_data.coordinate_units)
        self.taper_slider.setRange(taper_range[0], taper_range[1])
        self.taper_slider.setValue(int(self.taper_width))
        self.taper_slider.setToolTip(
            "Width of the transition zone at velocity boundaries.\n"
            "Larger values create smoother filter edges, reducing ringing.\n"
            "0 = sharp cutoff, higher values = gradual transition."
        )
        self.taper_slider.valueChanged.connect(self._on_taper_changed)

        self.taper_label = QLabel(self._get_velocity_label(self.taper_width))
        taper_layout = QHBoxLayout()
        taper_layout.addWidget(self.taper_slider)
        taper_layout.addWidget(self.taper_label)
        layout.addLayout(taper_layout)

        # Auto-update checkbox
        self.auto_update_check = QCheckBox("Preview Live (auto-update on change)")
        self.auto_update_check.setChecked(True)
        self.auto_update_check.stateChanged.connect(self._on_auto_update_changed)
        layout.addWidget(self.auto_update_check)

        # Manual update button (for when auto-update is off)
        self.update_button = QPushButton("Update Preview")
        self.update_button.clicked.connect(self._apply_filter)
        self.update_button.setEnabled(False)
        layout.addWidget(self.update_button)

        group.setLayout(layout)
        return group

    def _create_metrics_group(self) -> QGroupBox:
        """Create quality metrics display group."""
        group = QGroupBox("Quality Metrics")
        layout = QVBoxLayout()

        self.energy_preserved_label = QLabel("Energy preserved: -- %")
        self.energy_rejected_label = QLabel("Energy rejected: -- %")

        layout.addWidget(self.energy_preserved_label)
        layout.addWidget(self.energy_rejected_label)

        group.setLayout(layout)
        return group

    def _create_display_panel(self) -> QWidget:
        """Create right panel with FK spectrum and preview (resizable)."""
        # Vertical splitter for FK spectrum and previews
        display_splitter = QSplitter(Qt.Orientation.Vertical)
        display_splitter.setHandleWidth(6)  # Wider handle for easier dragging

        # Style the splitter handle to make it more visible
        display_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #c0c0c0;
                border: 1px solid #808080;
            }
            QSplitter::handle:hover {
                background-color: #a0a0ff;
            }
            QSplitter::handle:pressed {
                background-color: #8080ff;
            }
        """)

        # FK Spectrum (top)
        fk_group = self._create_fk_spectrum_group()
        display_splitter.addWidget(fk_group)

        # Side-by-side preview (bottom)
        preview_group = self._create_preview_group()
        display_splitter.addWidget(preview_group)

        # Set splitter sizes (60% FK plot, 40% previews)
        display_splitter.setStretchFactor(0, 6)
        display_splitter.setStretchFactor(1, 4)

        # Store splitter reference
        self.display_splitter = display_splitter

        return display_splitter

    def _create_fk_display_controls(self) -> QGroupBox:
        """Create FK display option controls (vertical layout for left panel)."""
        group = QGroupBox("FK Display Options")
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Smoothing
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothing:"))
        self.fk_smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.fk_smoothing_slider.setRange(0, 5)
        self.fk_smoothing_slider.setValue(0)
        self.fk_smoothing_slider.valueChanged.connect(self._on_fk_display_changed)
        smooth_layout.addWidget(self.fk_smoothing_slider)
        self.fk_smoothing_label = QLabel("Off")
        self.fk_smoothing_label.setMinimumWidth(50)
        smooth_layout.addWidget(self.fk_smoothing_label)
        layout.addLayout(smooth_layout)

        # Gain (logarithmic slider: 0.0001 to 10000)
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.fk_gain_slider = QSlider(Qt.Orientation.Horizontal)
        # Log scale: -400 to 400 → 10^(-4) to 10^(4) = 0.0001 to 10000
        self.fk_gain_slider.setRange(-400, 400)
        self.fk_gain_slider.setValue(0)  # 10^0 = 1.0
        self.fk_gain_slider.valueChanged.connect(self._on_fk_display_changed)
        gain_layout.addWidget(self.fk_gain_slider)
        self.fk_gain_label = QLabel("1.0x")
        self.fk_gain_label.setMinimumWidth(50)
        gain_layout.addWidget(self.fk_gain_label)
        layout.addLayout(gain_layout)

        # Colormap selection
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.fk_colormap_combo = QComboBox()
        self.fk_colormap_combo.addItems(["Hot", "Viridis", "Gray", "Seismic", "Jet", "Turbo"])
        self.fk_colormap_combo.setCurrentText("Hot")
        self.fk_colormap_combo.currentTextChanged.connect(self._on_fk_display_changed)
        cmap_layout.addWidget(self.fk_colormap_combo)
        layout.addLayout(cmap_layout)

        # Checkboxes
        self.fk_show_filtered_check = QCheckBox("Show Filtered FK")
        self.fk_show_filtered_check.setToolTip("Display filtered FK spectrum instead of input")
        self.fk_show_filtered_check.stateChanged.connect(self._on_show_filtered_fk_changed)
        layout.addWidget(self.fk_show_filtered_check)

        self.fk_interactive_check = QCheckBox("Interactive Boundaries")
        self.fk_interactive_check.stateChanged.connect(self._on_interactive_boundaries_changed)
        layout.addWidget(self.fk_interactive_check)

        # Reset button
        reset_btn = QPushButton("Reset Display")
        reset_btn.clicked.connect(self._on_reset_fk_display)
        layout.addWidget(reset_btn)

        group.setLayout(layout)
        return group

    def _create_fk_spectrum_group(self) -> QGroupBox:
        """Create FK spectrum visualization group."""
        group = QGroupBox("FK Spectrum")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Placeholder for FK spectrum plot
        self.fk_plot = pg.PlotWidget()
        self.fk_plot.setLabel('left', 'Frequency', units='Hz')
        unit = self.gather_data.unit_symbol
        self.fk_plot.setLabel('bottom', 'Wavenumber', units=f'cycles/{unit}')
        self.fk_plot.setTitle("FK Spectrum (Log Amplitude)")

        layout.addWidget(self.fk_plot)

        # Compact trace spacing info (single line with info button)
        spacing_layout = QHBoxLayout()
        spacing_layout.setContentsMargins(0, 0, 0, 0)

        self.trace_spacing_label = QLabel("Trace Spacing: Calculating...")
        self.trace_spacing_label.setStyleSheet("QLabel { font-size: 11px; }")
        spacing_layout.addWidget(self.trace_spacing_label)

        # Info button to show details
        self.spacing_info_btn = QPushButton("ⓘ Details")
        self.spacing_info_btn.setMaximumWidth(70)
        self.spacing_info_btn.setStyleSheet("QPushButton { font-size: 10px; padding: 2px; }")
        self.spacing_info_btn.clicked.connect(self._show_spacing_details)
        spacing_layout.addWidget(self.spacing_info_btn)

        spacing_layout.addStretch()

        layout.addLayout(spacing_layout)

        group.setLayout(layout)
        return group

    def _create_preview_group(self) -> QGroupBox:
        """Create side-by-side preview group."""
        group = QGroupBox("Side-by-Side Preview")
        layout = QHBoxLayout()

        # Three plots: Input | Filtered | Rejected
        self.input_plot = self._create_gather_plot("Input")
        self.filtered_plot = self._create_gather_plot("Filtered")
        self.rejected_plot = self._create_gather_plot("Rejected")

        layout.addWidget(self.input_plot)
        layout.addWidget(self.filtered_plot)
        layout.addWidget(self.rejected_plot)

        group.setLayout(layout)
        return group

    def _create_gather_plot(self, title: str) -> pg.PlotWidget:
        """Create a gather plot widget."""
        plot = pg.PlotWidget()
        plot.setTitle(title)
        plot.setLabel('left', 'Time', units='ms')
        plot.setLabel('bottom', 'Trace')
        return plot

    def _create_bottom_buttons(self) -> QHBoxLayout:
        """Create bottom button bar."""
        layout = QHBoxLayout()

        # Config name input
        layout.addWidget(QLabel("Configuration Name:"))
        self.config_name_edit = QLineEdit()
        self.config_name_edit.setPlaceholderText("e.g., Ground_Roll_Removal_v1")
        layout.addWidget(self.config_name_edit)

        layout.addStretch()

        # Buttons
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self._on_save_config)
        layout.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

        return layout

    # Event handlers

    def _on_preset_selected(self, index: int):
        """Handle preset selection."""
        params = self.preset_combo.currentData()
        if params is None:
            return  # Custom selected

        # Update parameters from preset
        self.v_min = params['v_min']
        self.v_max = params['v_max']
        self.taper_width = params['taper_width']
        self.mode = params['mode']

        # Update UI controls
        self.v_min_slider.setValue(int(self.v_min))
        self.v_max_slider.setValue(int(self.v_max))
        self.taper_slider.setValue(int(self.taper_width))

        mode_index = 0 if self.mode == 'pass' else 1
        self.mode_combo.setCurrentIndex(mode_index)

        # Update config name suggestion
        preset_name = self.preset_combo.currentText()
        self.config_name_edit.setText(preset_name)

        # Apply filter
        if self.auto_update:
            self._apply_filter()

    def _on_filter_type_changed(self, index: int):
        """Handle filter type change (velocity/dip)."""
        self.filter_type = self.filter_type_combo.currentData()

        # Show/hide appropriate parameter widgets
        self.velocity_widget.setVisible(self.filter_type == 'velocity')
        self.dip_widget.setVisible(self.filter_type == 'dip')

        # Mark as custom
        self.preset_combo.setCurrentIndex(0)

        if self.auto_update:
            self._compute_fk_spectrum()
            self._apply_filter()

    def _on_mode_changed(self, index: int):
        """Handle mode change."""
        self.mode = self.mode_combo.currentData()
        if self.auto_update:
            self._apply_filter()

    def _on_v_min_enable_changed(self, state):
        """Handle v_min enable checkbox change."""
        self.v_min_enabled = (state == Qt.CheckState.Checked.value)
        self.v_min_slider.setEnabled(self.v_min_enabled)
        self.preset_combo.setCurrentIndex(0)
        if self.auto_update:
            self._apply_filter()

    def _on_v_max_enable_changed(self, state):
        """Handle v_max enable checkbox change."""
        self.v_max_enabled = (state == Qt.CheckState.Checked.value)
        self.v_max_slider.setEnabled(self.v_max_enabled)
        self.preset_combo.setCurrentIndex(0)
        if self.auto_update:
            self._apply_filter()

    def _on_v_min_changed(self, value: int):
        """Handle v_min slider change."""
        self.v_min = float(value)
        self.v_min_label.setText(self._get_velocity_label(self.v_min))

        # Only enforce constraint if both limits are enabled
        if self.v_min_enabled and self.v_max_enabled:
            if self.v_min >= self.v_max:
                self.v_max = self.v_min + 100
                self.v_max_slider.setValue(int(self.v_max))

        # Mark as custom
        self.preset_combo.setCurrentIndex(0)

        if self.auto_update:
            self._apply_filter()

    def _on_v_max_changed(self, value: int):
        """Handle v_max slider change."""
        self.v_max = float(value)
        self.v_max_label.setText(self._get_velocity_label(self.v_max))

        # Only enforce constraint if both limits are enabled
        if self.v_min_enabled and self.v_max_enabled:
            if self.v_max <= self.v_min:
                self.v_min = self.v_max - 100
                self.v_min_slider.setValue(int(self.v_min))

        # Mark as custom
        self.preset_combo.setCurrentIndex(0)

        if self.auto_update:
            self._apply_filter()

    def _on_dip_min_enable_changed(self, state):
        """Handle dip_min enable checkbox change."""
        self.dip_min_enabled = (state == Qt.CheckState.Checked.value)
        self.dip_min_spin.setEnabled(self.dip_min_enabled)
        self.preset_combo.setCurrentIndex(0)
        if self.auto_update:
            self._apply_filter()

    def _on_dip_max_enable_changed(self, state):
        """Handle dip_max enable checkbox change."""
        self.dip_max_enabled = (state == Qt.CheckState.Checked.value)
        self.dip_max_spin.setEnabled(self.dip_max_enabled)
        self.preset_combo.setCurrentIndex(0)
        if self.auto_update:
            self._apply_filter()

    def _on_dip_min_changed(self, value: float):
        """Handle dip_min change."""
        self.dip_min = value
        self.preset_combo.setCurrentIndex(0)
        if self.auto_update:
            self._apply_filter()

    def _on_dip_max_changed(self, value: float):
        """Handle dip_max change."""
        self.dip_max = value
        self.preset_combo.setCurrentIndex(0)
        if self.auto_update:
            self._apply_filter()

    def _on_taper_changed(self, value: int):
        """Handle taper slider change."""
        self.taper_width = float(value)
        self.taper_label.setText(self._get_velocity_label(self.taper_width))

        # Mark as custom
        self.preset_combo.setCurrentIndex(0)

        if self.auto_update:
            self._apply_filter()

    def _on_auto_update_changed(self, state: int):
        """Handle auto-update checkbox change."""
        self.auto_update = (state == Qt.CheckState.Checked.value)
        self.update_button.setEnabled(not self.auto_update)

    # Sub-gather event handlers

    def _on_subgather_enabled_changed(self, state: int):
        """Handle sub-gather enable checkbox change."""
        self.use_subgathers = (state == Qt.CheckState.Checked.value)

        if self.use_subgathers:
            self.subgather_controls.show()
            # Detect sub-gathers
            self._detect_subgathers()
        else:
            self.subgather_controls.hide()
            # Reset to full gather
            self.subgathers = []
            self.current_subgather = None
            self.working_data = self.gather_data
            self.working_trace_spacing = self.trace_spacing
            # Recompute
            if self.auto_update:
                self._compute_fk_spectrum()
                self._apply_filter()

    def _on_boundary_header_changed(self, header: str):
        """Handle boundary header selection change."""
        if self.use_subgathers and header:
            self.boundary_header = header
            self._detect_subgathers()

    def _detect_subgathers(self):
        """Detect sub-gathers based on current boundary header."""
        if not self.boundary_header or self.gather_headers is None:
            return

        try:
            self.subgathers = detect_subgathers(
                self.gather_headers,
                self.boundary_header,
                min_traces=8
            )

            # Validate
            is_valid, warnings = validate_subgather_boundaries(self.subgathers)

            if warnings and FK_DEBUG:
                for warning in warnings:
                    print(f"Sub-gather warning: {warning}")

            # Update info label
            self.subgather_info_label.setText(
                f"Detected: {len(self.subgathers)} sub-gathers"
            )

            # Set to first sub-gather
            if self.subgathers:
                self.current_subgather_index = 0
                self._update_current_subgather()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Sub-Gather Detection Failed",
                f"Failed to detect sub-gathers:\n{str(e)}"
            )
            self.use_subgathers = False
            self.subgather_checkbox.setChecked(False)

    def _update_current_subgather(self):
        """Update working data to current sub-gather."""
        if not self.subgathers or self.current_subgather_index >= len(self.subgathers):
            return

        self.current_subgather = self.subgathers[self.current_subgather_index]

        # Extract sub-gather traces
        subgather_traces = extract_subgather_traces(
            self.gather_data.traces,
            self.current_subgather
        )

        # Create SeismicData for sub-gather
        self.working_data = SeismicData(
            traces=subgather_traces,
            sample_rate=self.gather_data.sample_rate,
            metadata={'description': f'Sub-gather {self.current_subgather.description}'}
        )

        # Calculate trace spacing for this sub-gather using enhanced calculation
        # This will properly handle coordinates and SEGY scalars
        stats = calculate_subgather_trace_spacing_with_stats(
            self.gather_headers,
            self.current_subgather.start_trace,
            self.current_subgather.end_trace,
            default_spacing=self.trace_spacing
        )
        self.working_trace_spacing = stats.spacing

        # Update label with spacing info
        unit = self.working_data.unit_symbol
        self.subgather_current_label.setText(
            f"Current: {self.current_subgather_index + 1}/{len(self.subgathers)} "
            f"({self.current_subgather.description}) - Spacing: {stats.spacing:.1f}{unit}"
        )

        # Recompute FK and filter (will also update trace spacing display)
        if self.auto_update:
            self._compute_fk_spectrum()
            self._apply_filter()
            self._update_displays()  # This will update trace spacing display

    def _on_prev_subgather(self):
        """Navigate to previous sub-gather."""
        if self.subgathers and self.current_subgather_index > 0:
            self.current_subgather_index -= 1
            self._update_current_subgather()

    def _on_next_subgather(self):
        """Navigate to next sub-gather."""
        if self.subgathers and self.current_subgather_index < len(self.subgathers) - 1:
            self.current_subgather_index += 1
            self._update_current_subgather()

    # AGC event handlers

    def _on_agc_enabled_changed(self, state: int):
        """Handle AGC enable checkbox change."""
        self.apply_agc = (state == Qt.CheckState.Checked.value)

        if self.apply_agc:
            self.agc_controls.show()
        else:
            self.agc_controls.hide()
            self.preview_with_agc = False

        # Recompute if enabled
        if self.auto_update:
            self._compute_fk_spectrum()
            self._apply_filter()

    def _on_agc_window_changed(self, value: int):
        """Handle AGC window spinbox change."""
        self.agc_window_ms = float(value)

        if self.auto_update and self.apply_agc:
            self._compute_fk_spectrum()
            self._apply_filter()

    def _on_agc_preview_changed(self):
        """Handle AGC preview radio button change."""
        self.preview_with_agc = self.agc_preview_with.isChecked()

        # Recompute FK spectrum with/without AGC
        if self.auto_update:
            self._compute_fk_spectrum()

    def _on_fk_display_changed(self):
        """Handle FK display option changes (smoothing, gain, colormap)."""
        # Update smoothing value and label
        self.fk_smoothing = self.fk_smoothing_slider.value()
        smoothing_labels = ["Off", "Light", "Medium", "Heavy", "Very Heavy", "Max"]
        self.fk_smoothing_label.setText(smoothing_labels[self.fk_smoothing])

        # Update gain value and label (logarithmic scale)
        # Slider: -400 to 400 → gain: 10^(-4) to 10^(4) = 0.0001 to 10000
        slider_value = self.fk_gain_slider.value()
        log_gain = slider_value / 100.0  # -4.0 to 4.0
        self.fk_gain = 10.0 ** log_gain

        # Format label nicely
        if self.fk_gain >= 10:
            self.fk_gain_label.setText(f"{self.fk_gain:.0f}x")
        elif self.fk_gain >= 1:
            self.fk_gain_label.setText(f"{self.fk_gain:.1f}x")
        elif self.fk_gain >= 0.01:
            self.fk_gain_label.setText(f"{self.fk_gain:.2f}x")
        else:
            self.fk_gain_label.setText(f"{self.fk_gain:.4f}x")

        # Update colormap
        self.fk_colormap = self.fk_colormap_combo.currentText()

        # Refresh FK spectrum plot with new display options
        self._update_fk_spectrum_plot()

    def _on_interactive_boundaries_changed(self, state: int):
        """Handle interactive boundaries checkbox change."""
        self.interactive_boundaries = (state == Qt.CheckState.Checked.value)

        # Refresh FK spectrum plot to add/remove draggable lines
        self._update_fk_spectrum_plot()

    def _on_show_filtered_fk_changed(self, state: int):
        """Handle show filtered FK checkbox change."""
        self.fk_show_filtered = (state == Qt.CheckState.Checked.value)

        # Recompute FK spectrum from filtered or input data
        self._compute_fk_spectrum()

        # Update FK spectrum plot with new data
        self._update_fk_spectrum_plot()

    def _on_reset_fk_display(self):
        """Reset FK display options to defaults."""
        # Reset to default values
        self.fk_smoothing = 0
        self.fk_gain = 1.0
        self.fk_colormap = "Hot"
        self.fk_show_filtered = False
        self.interactive_boundaries = False

        # Update UI controls
        self.fk_smoothing_slider.setValue(0)
        self.fk_gain_slider.setValue(0)  # 10^0 = 1.0
        self.fk_colormap_combo.setCurrentText("Hot")
        self.fk_show_filtered_check.setChecked(False)
        self.fk_interactive_check.setChecked(False)

        # Update labels
        self.fk_smoothing_label.setText("Off")
        self.fk_gain_label.setText("1.0x")

        # Recompute and refresh FK spectrum plot
        self._compute_fk_spectrum()
        self._update_fk_spectrum_plot()

    def _on_save_config(self):
        """Handle save configuration button."""
        config_name = self.config_name_edit.text().strip()
        if not config_name:
            QMessageBox.warning(
                self,
                "No Name",
                "Please enter a name for this configuration."
            )
            return

        # Create configuration
        config = FKFilterConfig(
            name=config_name,
            filter_type='velocity_fan',
            v_min=self.v_min,
            v_max=self.v_max,
            taper_width=self.taper_width,
            mode=self.mode,
            created=datetime.now().isoformat(),
            created_on_gather=self.gather_index,
            description=f"Created on gather {self.gather_index}",
            use_subgathers=self.use_subgathers,
            boundary_header=self.boundary_header if self.use_subgathers else None,
            apply_agc=self.apply_agc,
            agc_window_ms=self.agc_window_ms,
            coordinate_units=self.gather_data.coordinate_units
        )

        # Save via config manager
        manager = FKConfigManager()
        filepath = manager.save_config(config)

        # Emit signal
        self.config_saved.emit(config)

        # Notify user
        QMessageBox.information(
            self,
            "Configuration Saved",
            f"FK filter configuration '{config_name}' has been saved.\n\n"
            f"You can now use it in Apply mode to filter all gathers."
        )

    # Processing methods

    def _get_fk_cache_key(self, traces: np.ndarray) -> str:
        """
        Generate cache key for FK spectrum memoization.

        Args:
            traces: Input trace data

        Returns:
            Unique cache key string
        """
        import hashlib
        # Create hash from data shape, a sample of values, and relevant parameters
        data_hash = hashlib.md5(
            traces.tobytes()[:10000] +  # First ~10KB of data
            str(traces.shape).encode() +
            str(self.working_trace_spacing).encode() +
            str(self.working_data.sample_rate).encode() +
            str(self.preview_with_agc and self.apply_agc).encode() +
            str(self.agc_window_ms if self.apply_agc else 0).encode()
        ).hexdigest()
        return data_hash

    def _invalidate_fk_cache(self):
        """
        Invalidate the FK spectrum cache.

        Call this when the underlying data changes and a fresh FFT is required.
        """
        self._fk_cache_key = None
        self._cached_fk_spectrum = None
        self._cached_freqs = None
        self._cached_wavenumbers = None

    def _compute_fk_spectrum(self):
        """
        Compute FK spectrum of working data (with optional AGC for preview and filtered display).

        Uses memoization to cache FFT results - only recomputes when input data changes.
        Filter parameters can change without requiring spectrum recomputation.
        """
        # Choose data source: filtered or input
        if self.fk_show_filtered and self.filtered_data is not None:
            # Show FK spectrum of filtered data - always recompute (no caching for filtered)
            traces = self.filtered_data.traces
            use_cache = False
        else:
            # Show FK spectrum of input data (default)
            traces = self.working_data.traces

            # Apply AGC if preview toggle is enabled (only for input data)
            if self.preview_with_agc and self.apply_agc:
                # Convert sample rate from milliseconds to Hz
                sample_rate_hz = 1000.0 / self.working_data.sample_rate
                traces, _ = apply_agc_to_gather(
                    traces,
                    sample_rate_hz,
                    self.agc_window_ms
                )

            use_cache = True

        # Check if we can use cached spectrum
        if use_cache:
            cache_key = self._get_fk_cache_key(traces)
            if cache_key == self._fk_cache_key and self._cached_fk_spectrum is not None:
                # Use cached results
                self.fk_spectrum = self._cached_fk_spectrum
                self.freqs = self._cached_freqs
                self.wavenumbers = self._cached_wavenumbers
                if FK_DEBUG:
                    print(f"[FK Cache HIT] Using cached spectrum (key: {cache_key[:8]}...)")
                return

        # Create processor for FK computation (only need it for compute_fk_spectrum method)
        processor = FKFilter(
            filter_type=self.filter_type,
            v_min=self.v_min,
            v_max=self.v_max,
            v_min_enabled=self.v_min_enabled,
            v_max_enabled=self.v_max_enabled,
            dip_min=self.dip_min,
            dip_max=self.dip_max,
            dip_min_enabled=self.dip_min_enabled,
            dip_max_enabled=self.dip_max_enabled,
            taper_width=self.taper_width,
            mode=self.mode,
            trace_spacing=self.working_trace_spacing,
            coordinate_units=self.gather_data.coordinate_units
        )

        # Compute FK spectrum
        # Convert sample rate from milliseconds to Hz
        sample_rate_hz = 1000.0 / self.working_data.sample_rate

        # DEBUG: FK spectrum calculation
        if FK_DEBUG:
            print("\n" + "="*80)
            print("FK SPECTRUM COMPUTATION DEBUG")
            print("="*80)
            print(f"Traces shape: {traces.shape}")
            print(f"Sample rate: {sample_rate_hz} Hz")
            print(f"Working trace spacing: {self.working_trace_spacing} {self.working_data.unit_symbol}")

        self.fk_spectrum, self.freqs, self.wavenumbers = processor.compute_fk_spectrum(
            traces,
            sample_rate_hz,
            self.working_trace_spacing
        )

        # Cache results for input data
        if use_cache:
            self._fk_cache_key = cache_key
            self._cached_fk_spectrum = self.fk_spectrum
            self._cached_freqs = self.freqs
            self._cached_wavenumbers = self.wavenumbers
            if FK_DEBUG:
                print(f"[FK Cache MISS] Computed and cached new spectrum (key: {cache_key[:8]}...)")

        # DEBUG: FK spectrum results
        if FK_DEBUG:
            print(f"\nFK Spectrum results:")
            print(f"  Frequencies: {self.freqs.min():.2f} to {self.freqs.max():.2f} Hz")
            print(f"  Wavenumbers: {self.wavenumbers.min():.8f} to {self.wavenumbers.max():.8f} cycles/{self.working_data.unit_symbol}")
            print(f"  Wavenumbers (milli): {self.wavenumbers.min()*1000:.5f} to {self.wavenumbers.max()*1000:.5f} mcycles/{self.working_data.unit_symbol}")
            k_nyquist = 1.0 / (2.0 * self.working_trace_spacing)
            print(f"  Expected Nyquist: {k_nyquist:.8f} cycles/{self.working_data.unit_symbol}")
            print(f"  Expected Nyquist (milli): {k_nyquist*1000:.5f} mcycles/{self.working_data.unit_symbol}")

    def _apply_filter(self):
        """Apply current filter parameters and update results (with optional AGC)."""
        try:
            # Get input data
            input_data = self.working_data
            input_traces = input_data.traces

            # Apply AGC if enabled
            agc_scale_factors = None
            if self.apply_agc:
                # Convert sample rate from milliseconds to Hz
                sample_rate_hz = 1000.0 / input_data.sample_rate
                input_traces, agc_scale_factors = apply_agc_to_gather(
                    input_traces,
                    sample_rate_hz,
                    self.agc_window_ms
                )

                # Create AGC-applied SeismicData
                input_data = SeismicData(
                    traces=input_traces,
                    sample_rate=input_data.sample_rate,
                    metadata={'description': 'AGC-applied'}
                )

            # Create processor with current parameters
            processor = FKFilter(
                filter_type=self.filter_type,
                v_min=self.v_min,
                v_max=self.v_max,
                v_min_enabled=self.v_min_enabled,
                v_max_enabled=self.v_max_enabled,
                dip_min=self.dip_min,
                dip_max=self.dip_max,
                dip_min_enabled=self.dip_min_enabled,
                dip_max_enabled=self.dip_max_enabled,
                taper_width=self.taper_width,
                mode=self.mode,
                trace_spacing=self.working_trace_spacing,
                coordinate_units=self.gather_data.coordinate_units
            )

            # Apply FK filter
            filtered_data = processor.process(input_data)

            # Remove AGC if it was applied
            if self.apply_agc and agc_scale_factors is not None:
                filtered_traces = remove_agc(filtered_data.traces, agc_scale_factors)
                self.filtered_data = SeismicData(
                    traces=filtered_traces,
                    sample_rate=filtered_data.sample_rate,
                    metadata={'description': 'FK filtered (AGC removed)'}
                )
            else:
                self.filtered_data = filtered_data

            # Compute rejected (difference from original working data)
            rejected_traces = self.working_data.traces - self.filtered_data.traces
            self.rejected_data = SeismicData(
                traces=rejected_traces,
                sample_rate=self.working_data.sample_rate,
                metadata={'description': 'Rejected by FK filter'}
            )

            # Update displays
            self._update_displays()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Filter Error",
                f"Error applying FK filter:\n{str(e)}"
            )

    def _update_displays(self):
        """Update all visualization displays."""
        self._update_trace_spacing_display()
        self._update_fk_spectrum_plot()
        self._update_preview_plots()
        self._update_metrics()

    def _update_fk_spectrum_plot(self):
        """Update FK spectrum visualization."""
        if self.fk_spectrum is None:
            return

        # Save current view range to preserve zoom
        view_range = self.fk_plot.viewRange()

        self.fk_plot.clear()

        # Compute log amplitude of FK spectrum
        fk_amp = np.abs(self.fk_spectrum)
        fk_amp_db = 20 * np.log10(fk_amp + 1e-10)  # Add epsilon to avoid log(0)

        # Shift to center zero frequency/wavenumber
        fk_amp_db_shifted = np.fft.fftshift(fk_amp_db)
        freqs_shifted = np.fft.fftshift(self.freqs)
        k_shifted = np.fft.fftshift(self.wavenumbers)

        # Display FK spectrum as image
        # Only show positive frequencies (seismic convention)
        pos_freq_idx = freqs_shifted >= 0
        fk_display = fk_amp_db_shifted[pos_freq_idx, :]
        freqs_display = freqs_shifted[pos_freq_idx]

        # Apply display options: smoothing and gain
        if self.fk_smoothing > 0:
            # Apply Gaussian smoothing (sigma = smoothing level * 0.5)
            sigma = self.fk_smoothing * 0.5
            fk_display = gaussian_filter(fk_display, sigma=sigma)

        # Compute reference levels BEFORE gain (for proper gain effect)
        vmin_base = np.percentile(fk_display, 1)
        vmax_base = np.percentile(fk_display, 99)

        # Apply gain multiplication to dB values
        fk_display = fk_display * self.fk_gain

        # Use adaptive level strategy based on gain:
        # For normal gains (0.1 - 10x): use base levels → gain changes brightness
        # For extreme gains: interpolate with actual levels → prevent clipping/saturation
        if 0.1 <= self.fk_gain <= 10.0:
            # Normal range: use fixed base levels (gain works as brightness control)
            vmin = vmin_base
            vmax = vmax_base
        else:
            # Extreme gain: blend between base and actual to prevent issues
            # This prevents "foggy" appearance at very low gain and saturation at very high gain
            vmin_actual = np.percentile(fk_display, 1)
            vmax_actual = np.percentile(fk_display, 99)

            # Interpolation factor: 0 at gain=1, increases for extreme gains
            if self.fk_gain < 0.1:
                # Very low gain: blend more toward actual levels
                blend = min(1.0, (0.1 - self.fk_gain) / 0.09)  # 0 at 0.1, 1 at 0.01
            else:
                # Very high gain: blend more toward actual levels
                blend = min(1.0, (self.fk_gain - 10.0) / 90.0)  # 0 at 10, 1 at 100

            vmin = vmin_base * (1 - blend) + vmin_actual * blend
            vmax = vmax_base * (1 - blend) + vmax_actual * blend

        # Create image item
        img_item = pg.ImageItem()
        img_item.setImage(fk_display.T, autoLevels=False)
        img_item.setLevels([vmin, vmax])

        # Set position and scale to match frequency/wavenumber axes
        df = freqs_display[1] - freqs_display[0] if len(freqs_display) > 1 else 1
        dk = k_shifted[1] - k_shifted[0] if len(k_shifted) > 1 else 1
        img_item.setRect(
            k_shifted[0], freqs_display[0],
            len(k_shifted) * dk, len(freqs_display) * df
        )

        # Apply selected colormap
        lut = self._create_colormap(self.fk_colormap)
        img_item.setLookupTable(lut)

        self.fk_plot.addItem(img_item)

        # Draw filter boundary lines based on filter type
        if self.filter_type == 'velocity':
            self._draw_velocity_lines(freqs_display, k_shifted)
        elif self.filter_type == 'dip':
            self._draw_dip_lines(freqs_display, k_shifted)
        # Manual mode doesn't draw automatic lines

        # Restore view range if it was saved (preserve zoom), otherwise use full range
        if view_range is not None and view_range != [[0, 1], [0, 1]]:
            # Restore saved zoom
            self.fk_plot.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        else:
            # Initial plot: set to full data range
            self.fk_plot.setXRange(k_shifted.min(), k_shifted.max())
            self.fk_plot.setYRange(0, freqs_display.max())

    def _draw_velocity_lines(self, freqs: np.ndarray, wavenumbers: np.ndarray):
        """Draw velocity lines on FK spectrum."""
        unit = self.working_data.unit_symbol

        # DEBUG: Velocity line drawing
        if FK_DEBUG:
            print("\n" + "="*80)
            print("VELOCITY LINE DRAWING DEBUG")
            print("="*80)
            print(f"Units: {unit}")
            print(f"v_min: {self.v_min} {unit}/s (enabled: {self.v_min_enabled})")
            print(f"v_max: {self.v_max} {unit}/s (enabled: {self.v_max_enabled})")
            print(f"taper_width: {self.taper_width} {unit}/s")
            print(f"Wavenumber range: {wavenumbers.min():.8f} to {wavenumbers.max():.8f} cycles/{unit}")
            print(f"Frequency range: {freqs.min():.2f} to {freqs.max():.2f} Hz")

        # Velocity lines: f = v * k (for positive and negative slopes)
        # Create dense k array for smooth line drawing (500 points)
        k_min, k_max = wavenumbers.min(), wavenumbers.max()
        k_display = np.linspace(k_min, k_max, 500)
        f_max = freqs.max()

        if FK_DEBUG:
            print(f"\nLine drawing setup:")
            print(f"  k_display: {len(k_display)} points from {k_min:.8f} to {k_max:.8f}")
            print(f"  f_max: {f_max:.2f} Hz")

        # Define velocities to draw (only if enabled)
        velocities = []

        # Add taper zone boundaries (if enabled and taper exists)
        if self.taper_width > 0:
            if self.v_min_enabled:
                v1 = self.v_min - self.taper_width
                if v1 > 0:
                    velocities.append((v1, 'gray', '--'))
            if self.v_max_enabled:
                v4 = self.v_max + self.taper_width
                if v4 > 0:
                    velocities.append((v4, 'gray', '--'))

        # Add main velocity boundaries (if enabled)
        if self.v_min_enabled:
            velocities.append((self.v_min, 'yellow', '-'))
        if self.v_max_enabled:
            velocities.append((self.v_max, 'yellow', '-'))

        # Draw each velocity line (both positive and negative k)
        # Use thicker lines when interactive boundaries is enabled
        line_width = 4 if self.interactive_boundaries else 2

        for v, color, style in velocities:
            if v <= 0:
                continue

            # DEBUG: Line calculation
            if FK_DEBUG:
                print(f"\n  Drawing line for v={v:.0f} {unit}/s ({color} {style}):")

            # Positive k: f = v * k
            k_pos = k_display[k_display >= 0]
            if len(k_pos) > 0:
                f_pos = v * k_pos
                f_pos_clipped = np.clip(f_pos, 0, f_max)
                if FK_DEBUG:
                    print(f"    Positive k: {len(k_pos)} points, range [{k_pos[0]:.8f}, {k_pos[-1]:.8f}] cycles/{unit}")
                    print(f"    f = v*k: range [{f_pos[0]:.3f}, {f_pos[-1]:.3f}] Hz (before clip)")
                    print(f"    f (clipped): range [{f_pos_clipped[0]:.3f}, {f_pos_clipped[-1]:.3f}] Hz")
                    print(f"    -> This line should go from (k={k_pos[0]:.6f}, f={f_pos_clipped[0]:.1f}) to (k={k_pos[-1]:.6f}, f={f_pos_clipped[-1]:.1f})")
                pen = pg.mkPen(color, width=line_width, style={'--': Qt.PenStyle.DashLine, '-': Qt.PenStyle.SolidLine}[style])
                self.fk_plot.plot(k_pos, f_pos_clipped, pen=pen)

            # Negative k: f = v * |k|
            k_neg = k_display[k_display <= 0]
            if len(k_neg) > 0:
                f_neg = v * np.abs(k_neg)
                f_neg = np.clip(f_neg, 0, f_max)
                if FK_DEBUG:
                    print(f"    Negative k: {len(k_neg)} points, range [{k_neg[0]:.8f}, {k_neg[-1]:.8f}] cycles/{unit}")
                    print(f"    -> Mirror line from (k={k_neg[0]:.6f}, f={f_neg[0]:.1f}) to (k={k_neg[-1]:.6f}, f={f_neg[-1]:.1f})")
                pen = pg.mkPen(color, width=line_width, style={'--': Qt.PenStyle.DashLine, '-': Qt.PenStyle.SolidLine}[style])
                self.fk_plot.plot(k_neg, f_neg, pen=pen)

        # Add legend text
        legend_text = f"Mode: {self.mode.capitalize()} (Velocity)\n"
        if self.v_min_enabled:
            legend_text += f"v_min: {self._get_velocity_label(self.v_min)} (yellow solid)\n"
        else:
            legend_text += "v_min: disabled\n"
        if self.v_max_enabled:
            legend_text += f"v_max: {self._get_velocity_label(self.v_max)} (yellow solid)"
        else:
            legend_text += "v_max: disabled"
        if self.taper_width > 0:
            legend_text += f"\nTaper zones (gray dashed)"
        if self.interactive_boundaries:
            legend_text += f"\n[Interactive Mode: Boundaries Highlighted]"

        text_item = pg.TextItem(legend_text, color='white', anchor=(0, 1))
        text_item.setPos(wavenumbers.min(), f_max)
        self.fk_plot.addItem(text_item)

    def _draw_dip_lines(self, freqs: np.ndarray, wavenumbers: np.ndarray):
        """Draw dip lines on FK spectrum."""
        # Dip lines: dip = k/f, or f = k/dip
        # For constant dip, the line passes through origin with slope = 1/dip
        k_max = wavenumbers.max()
        k_min = wavenumbers.min()
        f_max = freqs.max()

        # Define dips to draw (based on enabled limits)
        dips = []

        # Add taper zone boundaries (if enabled and taper exists)
        if self.taper_width > 0:
            if self.dip_min_enabled:
                d1 = self.dip_min - self.taper_width  # More negative
                dips.append((d1, 'gray', '--', 'min taper'))
            if self.dip_max_enabled:
                d4 = self.dip_max + self.taper_width  # More positive
                dips.append((d4, 'gray', '--', 'max taper'))

        # Add main dip boundaries (if enabled)
        if self.dip_min_enabled:
            dips.append((self.dip_min, 'yellow', '-', 'min'))
        if self.dip_max_enabled:
            dips.append((self.dip_max, 'yellow', '-', 'max'))

        # Use thicker lines when interactive boundaries is enabled
        line_width = 4 if self.interactive_boundaries else 2

        # Draw each dip line
        for dip, color, style, label in dips:
            if dip == 0:
                # Zero dip is a horizontal line at k=0 (not useful to display)
                continue

            # Dip line: f = k/dip
            # For negative dip (left-dipping): line has negative slope (passes through origin)
            # For positive dip (right-dipping): line has positive slope

            # Create line from origin through the FK plane
            # Line equation: f = k/dip
            # We want to draw from (k_min, k_min/dip) to (k_max, k_max/dip)
            # But clip to visible frequency range [0, f_max]

            # Sample k values across the visible range
            k_vals = np.linspace(k_min, k_max, 100)
            f_vals = k_vals / dip

            # Clip to positive frequencies
            mask = (f_vals >= 0) & (f_vals <= f_max)
            k_vals_clipped = k_vals[mask]
            f_vals_clipped = f_vals[mask]

            if len(k_vals_clipped) > 1:
                pen = pg.mkPen(color, width=line_width,
                              style={'--': Qt.PenStyle.DashLine,
                                    '-': Qt.PenStyle.SolidLine}[style])
                self.fk_plot.plot(k_vals_clipped, f_vals_clipped, pen=pen)

        # Add legend text
        legend_text = f"Mode: {self.mode.capitalize()} (Dip)\n"
        if self.dip_min_enabled:
            legend_text += f"dip_min: {self.dip_min:.4f} s/m (yellow solid)\n"
        else:
            legend_text += "dip_min: disabled\n"
        if self.dip_max_enabled:
            legend_text += f"dip_max: {self.dip_max:.4f} s/m (yellow solid)"
        else:
            legend_text += "dip_max: disabled"
        if self.taper_width > 0:
            legend_text += f"\nTaper zones (gray dashed)"
        if self.interactive_boundaries:
            legend_text += f"\n[Interactive Mode: Boundaries Highlighted]"

        text_item = pg.TextItem(legend_text, color='white', anchor=(0, 1))
        text_item.setPos(wavenumbers.min(), f_max)
        self.fk_plot.addItem(text_item)

    def _create_colormap(self, name: str):
        """
        Create colormap lookup table for FK spectrum.

        Args:
            name: Colormap name ("Hot", "Viridis", "Gray", "Seismic", "Jet", "Turbo")

        Returns:
            256x4 RGBA lookup table
        """
        # Define colormap control points
        if name == "Hot":
            positions = np.array([0.0, 0.33, 0.66, 1.0])
            colors = np.array([
                [0, 0, 0, 255],        # Black
                [255, 0, 0, 255],      # Red
                [255, 255, 0, 255],    # Yellow
                [255, 255, 255, 255]   # White
            ], dtype=np.float32)

        elif name == "Viridis":
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [68, 1, 84, 255],      # Dark purple
                [59, 82, 139, 255],    # Blue
                [33, 145, 140, 255],   # Teal
                [94, 201, 98, 255],    # Green
                [253, 231, 37, 255]    # Yellow
            ], dtype=np.float32)

        elif name == "Gray":
            positions = np.array([0.0, 1.0])
            colors = np.array([
                [0, 0, 0, 255],        # Black
                [255, 255, 255, 255]   # White
            ], dtype=np.float32)

        elif name == "Seismic":
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [0, 0, 128, 255],      # Dark blue
                [0, 128, 255, 255],    # Light blue
                [255, 255, 255, 255],  # White
                [255, 128, 0, 255],    # Orange
                [128, 0, 0, 255]       # Dark red
            ], dtype=np.float32)

        elif name == "Jet":
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [0, 0, 128, 255],      # Dark blue
                [0, 255, 255, 255],    # Cyan
                [255, 255, 0, 255],    # Yellow
                [255, 128, 0, 255],    # Orange
                [128, 0, 0, 255]       # Dark red
            ], dtype=np.float32)

        elif name == "Turbo":
            positions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            colors = np.array([
                [48, 18, 59, 255],     # Dark purple
                [0, 145, 222, 255],    # Blue
                [57, 255, 186, 255],   # Cyan
                [255, 255, 84, 255],   # Yellow
                [255, 92, 0, 255],     # Orange
                [122, 4, 3, 255]       # Dark red
            ], dtype=np.float32)

        else:
            # Default to Hot
            positions = np.array([0.0, 0.33, 0.66, 1.0])
            colors = np.array([
                [0, 0, 0, 255],
                [255, 0, 0, 255],
                [255, 255, 0, 255],
                [255, 255, 255, 255]
            ], dtype=np.float32)

        # Generate 256-entry lookup table
        lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(256):
            pos = i / 255.0
            idx = np.searchsorted(positions, pos)
            if idx == 0:
                lut[i] = colors[0].astype(np.uint8)
            elif idx >= len(positions):
                lut[i] = colors[-1].astype(np.uint8)
            else:
                pos_low = positions[idx - 1]
                pos_high = positions[idx]
                color_low = colors[idx - 1]
                color_high = colors[idx]
                t = (pos - pos_low) / (pos_high - pos_low)
                lut[i] = (color_low * (1 - t) + color_high * t).astype(np.uint8)

        return lut

    def _update_preview_plots(self):
        """Update side-by-side gather previews."""
        if self.filtered_data is None or self.rejected_data is None:
            return

        # Clear existing plots
        self.input_plot.clear()
        self.filtered_plot.clear()
        self.rejected_plot.clear()

        # Create seismic colormap
        lut = self._create_seismic_colormap()

        # Compute amplitude range for consistent scaling
        all_data = np.concatenate([
            self.gather_data.traces.flatten(),
            self.filtered_data.traces.flatten(),
            self.rejected_data.traces.flatten()
        ])
        vmin = np.percentile(all_data, 1)
        vmax = np.percentile(all_data, 99)

        # Time axis (convert to ms)
        n_samples = self.gather_data.traces.shape[0]
        dt = 1000.0 / self.gather_data.sample_rate  # Convert to ms
        time_ms = np.arange(n_samples) * dt

        # Trace numbers
        n_traces = self.gather_data.traces.shape[1]
        trace_nums = np.arange(n_traces)

        # Plot input gather
        self._plot_gather(
            self.input_plot,
            self.gather_data.traces,
            time_ms,
            trace_nums,
            lut,
            vmin,
            vmax,
            "Input"
        )

        # Plot filtered gather
        self._plot_gather(
            self.filtered_plot,
            self.filtered_data.traces,
            time_ms,
            trace_nums,
            lut,
            vmin,
            vmax,
            "Filtered"
        )

        # Plot rejected gather
        self._plot_gather(
            self.rejected_plot,
            self.rejected_data.traces,
            time_ms,
            trace_nums,
            lut,
            vmin,
            vmax,
            "Rejected"
        )

    def _plot_gather(
        self,
        plot_widget: pg.PlotWidget,
        data: np.ndarray,
        time_ms: np.ndarray,
        trace_nums: np.ndarray,
        lut: np.ndarray,
        vmin: float,
        vmax: float,
        title: str
    ):
        """Plot a single gather."""
        # Create image item
        img_item = pg.ImageItem()
        img_item.setImage(data.T, autoLevels=False, levels=(vmin, vmax))

        # Set position and scale
        dt = time_ms[1] - time_ms[0] if len(time_ms) > 1 else 1
        dx = 1  # Trace spacing
        img_item.setRect(
            trace_nums[0], time_ms[0],
            len(trace_nums) * dx, len(time_ms) * dt
        )

        # Apply colormap
        img_item.setLookupTable(lut)

        # Add to plot
        plot_widget.addItem(img_item)

        # Configure plot
        plot_widget.setTitle(title)
        plot_widget.setLabel('left', 'Time', units='ms')
        plot_widget.setLabel('bottom', 'Trace')
        plot_widget.invertY(True)  # Time increases downward

    def _create_seismic_colormap(self):
        """Create seismic colormap (blue-white-red) for gather display."""
        positions = np.array([0.0, 0.45, 0.50, 0.55, 1.0])
        colors = np.array([
            [0, 0, 255, 255],        # Blue
            [135, 206, 250, 255],    # Light blue
            [245, 245, 245, 255],    # White
            [255, 160, 122, 255],    # Light red
            [255, 0, 0, 255]         # Red
        ], dtype=np.float32)

        # Generate 256-entry lookup table
        lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(256):
            pos = i / 255.0
            idx = np.searchsorted(positions, pos)
            if idx == 0:
                lut[i] = colors[0].astype(np.uint8)
            elif idx >= len(positions):
                lut[i] = colors[-1].astype(np.uint8)
            else:
                pos_low = positions[idx - 1]
                pos_high = positions[idx]
                color_low = colors[idx - 1]
                color_high = colors[idx]
                t = (pos - pos_low) / (pos_high - pos_low)
                lut[i] = (color_low * (1 - t) + color_high * t).astype(np.uint8)

        return lut

    def _update_metrics(self):
        """Update quality metrics display."""
        if self.filtered_data is None or self.rejected_data is None:
            return

        # Calculate energy
        input_energy = np.sum(self.gather_data.traces**2)
        filtered_energy = np.sum(self.filtered_data.traces**2)
        rejected_energy = np.sum(self.rejected_data.traces**2)

        if input_energy > 0:
            preserved_pct = (filtered_energy / input_energy) * 100
            rejected_pct = (rejected_energy / input_energy) * 100
        else:
            preserved_pct = 0
            rejected_pct = 0

        self.energy_preserved_label.setText(f"Energy preserved: {preserved_pct:.1f}%")
        self.energy_rejected_label.setText(f"Energy rejected: {rejected_pct:.1f}%")

    def _calculate_trace_spacing_stats(self) -> TraceSpacingStats:
        """
        Calculate trace spacing statistics for current working data.

        Returns:
            TraceSpacingStats object with spacing and statistics
        """
        if FK_DEBUG:
            print("\n" + "="*80)
            print("TRACE SPACING CALCULATION DEBUG")
            print("="*80)
            print(f"Has headers: {self.gather_headers is not None}")
            print(f"Use subgathers: {self.use_subgathers}")

        if self.gather_headers is None:
            # No headers available, use default
            from utils.trace_spacing import TraceSpacingStats
            return TraceSpacingStats(
                spacing=self.trace_spacing,
                mean=self.trace_spacing,
                std=0.0,
                min_spacing=self.trace_spacing,
                max_spacing=self.trace_spacing,
                n_spacings=0,
                coordinate_source='provided',
                scalar_applied=1.0,
                coordinates_raw=np.array([]),
                coordinates_scaled=np.array([]),
                spacings_all=np.array([])
            )

        if self.use_subgathers and self.current_subgather is not None:
            if FK_DEBUG:
                print(f"Calculating for sub-gather {self.current_subgather_index + 1}")
            # Calculate for current sub-gather
            stats = calculate_subgather_trace_spacing_with_stats(
                self.gather_headers,
                self.current_subgather.start_trace,
                self.current_subgather.end_trace,
                default_spacing=self.trace_spacing
            )
        else:
            if FK_DEBUG:
                print("Calculating for full gather")
            # Calculate for full gather
            stats = calculate_trace_spacing_with_stats(
                self.gather_headers,
                default_spacing=self.trace_spacing
            )

        if FK_DEBUG:
            print(f"\nResult:")
            print(f"  Spacing: {stats.spacing:.2f}")
            print(f"  Source: {stats.coordinate_source}")
            print(f"  Scalar: {stats.scalar_applied}")
            print(f"  n_spacings: {stats.n_spacings}")
            if len(stats.spacings_all) > 0:
                print(f"  First 5 spacings: {stats.spacings_all[:5]}")

        return stats

    def _update_trace_spacing_display(self):
        """Update trace spacing information display."""
        stats = self._calculate_trace_spacing_stats()

        # Store stats for details dialog
        self.trace_spacing_stats = stats

        # Update compact label (single line)
        unit = self.working_data.unit_symbol
        compact_text = f"Spacing: {stats.spacing:.1f} {unit} (from {stats.coordinate_source})"
        self.trace_spacing_label.setText(compact_text)

        # Set tooltip with brief info
        tooltip = f"Trace Spacing: {stats.spacing:.2f} {unit}\nSource: {stats.coordinate_source}"
        if stats.coordinate_source not in ['default', 'd3', 'provided']:
            tooltip += f"\nScalar: {stats.scalar_applied}\nClick 'Details' for full statistics"
        self.trace_spacing_label.setToolTip(tooltip)

        # Update working trace spacing
        self.working_trace_spacing = stats.spacing

    def _show_spacing_details(self):
        """Show detailed trace spacing statistics in a message box."""
        if not hasattr(self, 'trace_spacing_stats'):
            QMessageBox.information(self, "Trace Spacing", "No spacing information available yet.")
            return

        # Format detailed statistics
        details = format_spacing_stats(self.trace_spacing_stats)

        # Show in message box
        QMessageBox.information(
            self,
            "Trace Spacing Details",
            details
        )

    def _check_offset_gaps(self):
        """Check for gaps in offset steps and warn if sub-gathers should be used."""
        if self.gather_headers is None:
            return

        # Analyze offset step uniformity
        self.offset_analysis = analyze_offset_step_uniformity(self.gather_headers)

        if self.offset_analysis is None:
            return

        # Show warning if gaps detected and sub-gathers not enabled
        if self.offset_analysis.has_gaps and not self.use_subgathers:
            cv = self.offset_analysis.step_cv
            n_gaps = self.offset_analysis.n_gaps

            msg = (
                f"⚠️ Offset Step Analysis\n\n"
                f"Detected {n_gaps} large gaps in offset progression.\n"
                f"Offset step CV: {cv:.1f}%\n\n"
                f"This indicates multiple sub-gathers mixed together.\n"
                f"FK filtering requires uniform offset steps.\n\n"
            )

            if self.offset_analysis.suggested_headers:
                msg += "Suggested boundary headers:\n"
                for header in self.offset_analysis.suggested_headers[:3]:
                    msg += f"  • {header}\n"
                msg += "\nEnable 'Split gather by header changes'?"
            else:
                msg += "Enable 'Split gather by header changes' to fix this."

            reply = QMessageBox.question(
                self,
                "Sub-Gather Splitting Recommended",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.subgather_checkbox.setChecked(True)
                # Auto-select first suggested header if available
                if self.offset_analysis.suggested_headers:
                    suggested = self.offset_analysis.suggested_headers[0]
                    idx = self.boundary_header_combo.findText(suggested)
                    if idx >= 0:
                        self.boundary_header_combo.setCurrentIndex(idx)
