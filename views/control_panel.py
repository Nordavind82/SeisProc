"""
Control panel - user interface for processing parameters and display controls.
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QLabel, QDoubleSpinBox, QSlider,
                              QSpinBox, QComboBox, QListWidget, QAbstractItemView,
                              QCheckBox, QScrollArea, QRadioButton, QButtonGroup,
                              QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
import sys
from processors.bandpass_filter import BandpassFilter
from models.fk_config import FKConfigManager, FKFilterConfig
from models.fkk_config import FKKConfig, FKK_PRESETS

# Try to import GPU modules
try:
    from processors.gpu.device_manager import get_device_manager
    from processors.tf_denoise_gpu import TFDenoiseGPU
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ControlPanel(QWidget):
    """
    Control panel for processing parameters and display settings.

    Signals:
        process_requested: Emitted when user requests processing with parameters
        gain_changed: Emitted when gain slider changes
        clip_changed: Emitted when clip percentile changes
        zoom_in_requested: Emitted when zoom in button clicked
        zoom_out_requested: Emitted when zoom out button clicked
        reset_view_requested: Emitted when reset view button clicked
    """

    process_requested = pyqtSignal(object)  # Emits processor object
    amplitude_range_changed = pyqtSignal(float, float)  # min, max
    colormap_changed = pyqtSignal(str)  # colormap name
    interpolation_changed = pyqtSignal(str)  # interpolation mode
    sort_keys_changed = pyqtSignal(list)  # list of header names for sorting
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    reset_view_requested = pyqtSignal()
    fk_design_requested = pyqtSignal()  # Request to open FK designer
    fk_config_selected = pyqtSignal(str)  # FK config name selected for applying
    fkk_design_requested = pyqtSignal()  # Request to open 3D FKK designer
    fkk_apply_requested = pyqtSignal(object)  # FKKConfig for applying

    def __init__(self, nyquist_freq: float = 250.0, parent=None):
        super().__init__(parent)
        self.nyquist_freq = nyquist_freq

        # Initialize GPU device manager if available
        if GPU_AVAILABLE:
            try:
                self.device_manager = get_device_manager()
                self.gpu_available = self.device_manager.is_gpu_available()
            except Exception as e:
                print(f"Warning: GPU initialization failed: {e}")
                self.device_manager = None
                self.gpu_available = False
        else:
            self.device_manager = None
            self.gpu_available = False

        # Initialize FK config manager
        self.fk_config_manager = FKConfigManager()

        self._init_ui()

    def _init_ui(self):
        """Initialize user interface."""
        # Create main layout for the panel
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        # Create a widget to hold all controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # Algorithm selection
        controls_layout.addWidget(self._create_algorithm_selector())

        # Processing controls (dynamic based on algorithm)
        self.bandpass_group = self._create_bandpass_group()
        self.tfdenoise_group = self._create_tfdenoise_group()
        self.fk_filter_group = self._create_fk_filter_group()
        self.fkk_filter_group = self._create_fkk_filter_group()
        controls_layout.addWidget(self.bandpass_group)
        controls_layout.addWidget(self.tfdenoise_group)
        controls_layout.addWidget(self.fk_filter_group)
        controls_layout.addWidget(self.fkk_filter_group)
        self.tfdenoise_group.hide()  # Initially hidden
        self.fk_filter_group.hide()  # Initially hidden
        self.fkk_filter_group.hide()  # Initially hidden

        # Display controls
        controls_layout.addWidget(self._create_display_group())

        # Sort controls
        controls_layout.addWidget(self._create_sort_group())

        # View controls
        controls_layout.addWidget(self._create_view_group())

        # Stretch to push everything to top
        controls_layout.addStretch()

        controls_widget.setLayout(controls_layout)
        scroll_area.setWidget(controls_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setMaximumWidth(300)

    def _create_algorithm_selector(self) -> QGroupBox:
        """Create algorithm selection group."""
        group = QGroupBox("Algorithm Selection")
        layout = QVBoxLayout()

        # Algorithm dropdown
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "Bandpass Filter",
            "TF-Denoise (S-Transform)",
            "FK Filter",
            "3D FKK Filter"
        ])
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(self.algorithm_combo)
        layout.addLayout(algo_layout)

        # GPU acceleration controls
        if GPU_AVAILABLE:
            gpu_layout = QVBoxLayout()

            # GPU checkbox
            self.gpu_checkbox = QCheckBox("Use GPU Acceleration")
            self.gpu_checkbox.setChecked(self.gpu_available)
            self.gpu_checkbox.setEnabled(self.gpu_available)
            self.gpu_checkbox.stateChanged.connect(self._on_gpu_checkbox_changed)
            gpu_layout.addWidget(self.gpu_checkbox)

            # GPU status label
            if self.gpu_available:
                gpu_name = self.device_manager.get_device_name()
                status_text = f"🟢 {gpu_name}"
            else:
                status_text = "🟡 GPU not available"

            self.gpu_status_label = QLabel(status_text)
            self.gpu_status_label.setStyleSheet("font-size: 9pt; color: gray;")
            gpu_layout.addWidget(self.gpu_status_label)

            layout.addLayout(gpu_layout)
        else:
            self.gpu_checkbox = None
            self.gpu_status_label = None

        group.setLayout(layout)
        return group

    def _create_bandpass_group(self) -> QGroupBox:
        """Create bandpass filter parameters group."""
        group = QGroupBox("Bandpass Filter Parameters")
        layout = QVBoxLayout()

        # Low frequency
        low_layout = QHBoxLayout()
        low_layout.addWidget(QLabel("Low Freq (Hz):"))
        self.low_freq_spin = QDoubleSpinBox()
        self.low_freq_spin.setRange(1.0, self.nyquist_freq - 1)
        self.low_freq_spin.setValue(10.0)
        self.low_freq_spin.setSingleStep(1.0)
        self.low_freq_spin.setDecimals(1)
        low_layout.addWidget(self.low_freq_spin)
        layout.addLayout(low_layout)

        # High frequency
        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("High Freq (Hz):"))
        self.high_freq_spin = QDoubleSpinBox()
        self.high_freq_spin.setRange(2.0, self.nyquist_freq - 0.1)
        self.high_freq_spin.setValue(80.0)
        self.high_freq_spin.setSingleStep(1.0)
        self.high_freq_spin.setDecimals(1)
        high_layout.addWidget(self.high_freq_spin)
        layout.addLayout(high_layout)

        # Filter order
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Filter Order:"))
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(4)
        order_layout.addWidget(self.order_spin)
        layout.addLayout(order_layout)

        # Nyquist info
        nyquist_label = QLabel(f"Nyquist: {self.nyquist_freq:.1f} Hz")
        nyquist_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(nyquist_label)

        # Apply button
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_btn)

        group.setLayout(layout)
        return group

    def _create_tfdenoise_group(self) -> QGroupBox:
        """Create TF-Denoise parameters group."""
        group = QGroupBox("TF-Denoise Parameters")
        layout = QVBoxLayout()

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.tfdenoise_preset_combo = QComboBox()
        self.tfdenoise_preset_combo.addItems([
            "Low-Freq Noise",
            "White Noise",
            "High-Freq Noise",
            "Conservative",
            "Custom"
        ])
        self.tfdenoise_preset_combo.currentIndexChanged.connect(self._on_tfdenoise_preset_changed)
        preset_layout.addWidget(self.tfdenoise_preset_combo)
        layout.addLayout(preset_layout)

        # Spatial aperture
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Spatial Aperture:"))
        self.aperture_spin = QSpinBox()
        self.aperture_spin.setRange(3, 21)
        self.aperture_spin.setValue(7)
        self.aperture_spin.setSingleStep(2)  # Odd numbers only
        self.aperture_spin.setToolTip("Number of traces to use for spatial denoising (odd numbers)")
        aperture_layout.addWidget(self.aperture_spin)
        layout.addLayout(aperture_layout)

        # Frequency range
        fmin_layout = QHBoxLayout()
        fmin_layout.addWidget(QLabel("Min Freq (Hz):"))
        self.tfdenoise_fmin_spin = QDoubleSpinBox()
        self.tfdenoise_fmin_spin.setRange(1.0, self.nyquist_freq - 1)
        self.tfdenoise_fmin_spin.setValue(5.0)
        self.tfdenoise_fmin_spin.setDecimals(1)
        fmin_layout.addWidget(self.tfdenoise_fmin_spin)
        layout.addLayout(fmin_layout)

        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Freq (Hz):"))
        self.tfdenoise_fmax_spin = QDoubleSpinBox()
        self.tfdenoise_fmax_spin.setRange(2.0, self.nyquist_freq - 0.1)
        self.tfdenoise_fmax_spin.setValue(100.0)
        self.tfdenoise_fmax_spin.setDecimals(1)
        fmax_layout.addWidget(self.tfdenoise_fmax_spin)
        layout.addLayout(fmax_layout)

        # MAD threshold multiplier
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("Threshold (k):"))
        self.threshold_k_spin = QDoubleSpinBox()
        self.threshold_k_spin.setRange(0.5, 10.0)
        self.threshold_k_spin.setValue(3.0)
        self.threshold_k_spin.setSingleStep(0.5)
        self.threshold_k_spin.setDecimals(1)
        self.threshold_k_spin.setToolTip("MAD threshold multiplier (higher = more aggressive)")
        k_layout.addWidget(self.threshold_k_spin)
        layout.addLayout(k_layout)

        # Threshold type
        threshold_type_layout = QHBoxLayout()
        threshold_type_layout.addWidget(QLabel("Threshold Type:"))
        self.threshold_type_combo = QComboBox()
        self.threshold_type_combo.addItems([
            "Soft",
            "Garrote"
        ])
        threshold_type_layout.addWidget(self.threshold_type_combo)
        layout.addLayout(threshold_type_layout)

        # Transform type
        transform_layout = QHBoxLayout()
        transform_layout.addWidget(QLabel("Transform:"))
        self.transform_type_combo = QComboBox()
        self.transform_type_combo.addItems([
            "S-Transform",
            "STFT"
        ])
        transform_layout.addWidget(self.transform_type_combo)
        layout.addLayout(transform_layout)

        # Threshold mode (NEW)
        threshold_mode_layout = QHBoxLayout()
        threshold_mode_layout.addWidget(QLabel("Noise Removal:"))
        self.threshold_mode_combo = QComboBox()
        self.threshold_mode_combo.addItems([
            "Adaptive (Recommended)",
            "Hard (Full Removal)",
            "Scaled (Progressive)",
            "Soft (Legacy)"
        ])
        self.threshold_mode_combo.setToolTip(
            "Adaptive: Hard for severe outliers, scaled for moderate (recommended)\n"
            "Hard: Full removal for all outliers\n"
            "Scaled: Progressive removal based on severity\n"
            "Soft: Legacy partial removal"
        )
        threshold_mode_layout.addWidget(self.threshold_mode_combo)
        layout.addLayout(threshold_mode_layout)

        # Low-amplitude protection (NEW)
        self.low_amp_protection_checkbox = QCheckBox("Low-amplitude protection")
        self.low_amp_protection_checkbox.setChecked(True)
        self.low_amp_protection_checkbox.setToolTip(
            "Prevent inflation of low-amplitude samples\n"
            "(isolated signals won't be boosted toward median)"
        )
        layout.addWidget(self.low_amp_protection_checkbox)

        # Apply button
        self.tfdenoise_apply_btn = QPushButton("Apply TF-Denoise")
        self.tfdenoise_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.tfdenoise_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.tfdenoise_apply_btn)

        group.setLayout(layout)
        return group

    def _create_fk_filter_group(self) -> QGroupBox:
        """Create FK filter group with Design/Apply modes."""
        group = QGroupBox("FK Filter")
        layout = QVBoxLayout()

        # Mode selection (Design/Apply)
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()

        self.fk_mode_design = QRadioButton("Design (create new filter)")
        self.fk_mode_apply = QRadioButton("Apply (use saved filter)")
        self.fk_mode_design.setChecked(True)

        self.fk_mode_group = QButtonGroup()
        self.fk_mode_group.addButton(self.fk_mode_design, 0)
        self.fk_mode_group.addButton(self.fk_mode_apply, 1)
        self.fk_mode_group.buttonClicked.connect(self._on_fk_mode_changed)

        mode_layout.addWidget(self.fk_mode_design)
        mode_layout.addWidget(self.fk_mode_apply)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Design mode controls
        self.fk_design_widget = QWidget()
        design_layout = QVBoxLayout()
        design_layout.setContentsMargins(0, 0, 0, 0)

        design_info = QLabel("Design FK filter parameters on current gather")
        design_info.setStyleSheet("color: #666; font-size: 9pt;")
        design_info.setWordWrap(True)
        design_layout.addWidget(design_info)

        self.fk_design_btn = QPushButton("Open FK Filter Designer...")
        self.fk_design_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.fk_design_btn.clicked.connect(self._on_fk_design_clicked)
        design_layout.addWidget(self.fk_design_btn)

        self.fk_design_widget.setLayout(design_layout)
        layout.addWidget(self.fk_design_widget)

        # Apply mode controls
        self.fk_apply_widget = QWidget()
        apply_layout = QVBoxLayout()
        apply_layout.setContentsMargins(0, 0, 0, 0)

        apply_info = QLabel("Select saved configuration to apply to all gathers")
        apply_info.setStyleSheet("color: #666; font-size: 9pt;")
        apply_info.setWordWrap(True)
        apply_layout.addWidget(apply_info)

        # List of saved configurations
        self.fk_config_list = QListWidget()
        self.fk_config_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.fk_config_list.itemSelectionChanged.connect(self._on_fk_config_selection_changed)
        apply_layout.addWidget(self.fk_config_list)

        # Refresh button
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self._refresh_fk_config_list)
        apply_layout.addWidget(refresh_btn)

        # Management buttons
        mgmt_layout = QHBoxLayout()

        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_fk_delete_clicked)
        mgmt_layout.addWidget(delete_btn)

        apply_layout.addLayout(mgmt_layout)

        # Apply button
        self.fk_apply_btn = QPushButton("Apply Selected Config")
        self.fk_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.fk_apply_btn.setEnabled(False)
        self.fk_apply_btn.clicked.connect(self._on_fk_apply_clicked)
        apply_layout.addWidget(self.fk_apply_btn)

        self.fk_apply_widget.setLayout(apply_layout)
        layout.addWidget(self.fk_apply_widget)
        self.fk_apply_widget.hide()  # Initially hidden

        group.setLayout(layout)

        # Load initial config list
        self._refresh_fk_config_list()

        return group

    def _create_fkk_filter_group(self) -> QGroupBox:
        """Create 3D FKK filter group with Design/Apply modes."""
        group = QGroupBox("3D FKK Filter")
        layout = QVBoxLayout()

        # Mode selection (Design/Apply)
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()

        self.fkk_mode_design = QRadioButton("Design (build volume & create filter)")
        self.fkk_mode_apply = QRadioButton("Apply (use saved preset)")
        self.fkk_mode_design.setChecked(True)

        self.fkk_mode_group = QButtonGroup()
        self.fkk_mode_group.addButton(self.fkk_mode_design, 0)
        self.fkk_mode_group.addButton(self.fkk_mode_apply, 1)
        self.fkk_mode_group.buttonClicked.connect(self._on_fkk_mode_changed)

        mode_layout.addWidget(self.fkk_mode_design)
        mode_layout.addWidget(self.fkk_mode_apply)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Design mode controls
        self.fkk_design_widget = QWidget()
        design_layout = QVBoxLayout()
        design_layout.setContentsMargins(0, 0, 0, 0)

        design_info = QLabel(
            "Build 3D volume from current data using\n"
            "selected headers for inline/crossline axes,\n"
            "then design velocity cone filter."
        )
        design_info.setStyleSheet("color: #666; font-size: 9pt;")
        design_info.setWordWrap(True)
        design_layout.addWidget(design_info)

        self.fkk_design_btn = QPushButton("Open 3D FKK Designer...")
        self.fkk_design_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #6A1B9A;
            }
        """)
        self.fkk_design_btn.clicked.connect(self._on_fkk_design_clicked)
        design_layout.addWidget(self.fkk_design_btn)

        self.fkk_design_widget.setLayout(design_layout)
        layout.addWidget(self.fkk_design_widget)

        # Apply mode controls
        self.fkk_apply_widget = QWidget()
        apply_layout = QVBoxLayout()
        apply_layout.setContentsMargins(0, 0, 0, 0)

        apply_info = QLabel("Select preset and configure parameters")
        apply_info.setStyleSheet("color: #666; font-size: 9pt;")
        apply_info.setWordWrap(True)
        apply_layout.addWidget(apply_info)

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.fkk_preset_combo = QComboBox()
        self.fkk_preset_combo.addItem("Custom", None)
        for name in FKK_PRESETS.keys():
            self.fkk_preset_combo.addItem(name, name)
        self.fkk_preset_combo.currentIndexChanged.connect(self._on_fkk_preset_changed)
        preset_layout.addWidget(self.fkk_preset_combo)
        apply_layout.addLayout(preset_layout)

        # V_min
        vmin_layout = QHBoxLayout()
        vmin_layout.addWidget(QLabel("V min:"))
        self.fkk_vmin_spin = QDoubleSpinBox()
        self.fkk_vmin_spin.setRange(50, 10000)
        self.fkk_vmin_spin.setValue(200)
        self.fkk_vmin_spin.setSuffix(" m/s")
        vmin_layout.addWidget(self.fkk_vmin_spin)
        apply_layout.addLayout(vmin_layout)

        # V_max
        vmax_layout = QHBoxLayout()
        vmax_layout.addWidget(QLabel("V max:"))
        self.fkk_vmax_spin = QDoubleSpinBox()
        self.fkk_vmax_spin.setRange(100, 20000)
        self.fkk_vmax_spin.setValue(1500)
        self.fkk_vmax_spin.setSuffix(" m/s")
        vmax_layout.addWidget(self.fkk_vmax_spin)
        apply_layout.addLayout(vmax_layout)

        # Mode
        mode_filter_layout = QHBoxLayout()
        mode_filter_layout.addWidget(QLabel("Filter:"))
        self.fkk_filter_mode_combo = QComboBox()
        self.fkk_filter_mode_combo.addItem("Reject (remove)", "reject")
        self.fkk_filter_mode_combo.addItem("Pass (keep)", "pass")
        mode_filter_layout.addWidget(self.fkk_filter_mode_combo)
        apply_layout.addLayout(mode_filter_layout)

        # AGC checkbox
        agc_layout = QHBoxLayout()
        self.fkk_agc_checkbox = QCheckBox("Apply AGC")
        self.fkk_agc_checkbox.setToolTip("Normalize amplitudes before filtering")
        agc_layout.addWidget(self.fkk_agc_checkbox)
        agc_layout.addWidget(QLabel("Window:"))
        self.fkk_agc_window_spin = QDoubleSpinBox()
        self.fkk_agc_window_spin.setRange(50, 2000)
        self.fkk_agc_window_spin.setValue(500)
        self.fkk_agc_window_spin.setSuffix(" ms")
        agc_layout.addWidget(self.fkk_agc_window_spin)
        apply_layout.addLayout(agc_layout)

        # Frequency band
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Freq:"))
        self.fkk_fmin_spin = QDoubleSpinBox()
        self.fkk_fmin_spin.setRange(0, 500)
        self.fkk_fmin_spin.setValue(0)
        self.fkk_fmin_spin.setSuffix(" Hz")
        freq_layout.addWidget(self.fkk_fmin_spin)
        freq_layout.addWidget(QLabel("-"))
        self.fkk_fmax_spin = QDoubleSpinBox()
        self.fkk_fmax_spin.setRange(1, 500)
        self.fkk_fmax_spin.setValue(self.nyquist_freq)
        self.fkk_fmax_spin.setSuffix(" Hz")
        freq_layout.addWidget(self.fkk_fmax_spin)
        apply_layout.addLayout(freq_layout)

        # Apply button
        self.fkk_apply_btn = QPushButton("Apply FKK Filter")
        self.fkk_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.fkk_apply_btn.clicked.connect(self._on_fkk_apply_clicked)
        apply_layout.addWidget(self.fkk_apply_btn)

        self.fkk_apply_widget.setLayout(apply_layout)
        layout.addWidget(self.fkk_apply_widget)
        self.fkk_apply_widget.hide()  # Initially hidden

        group.setLayout(layout)
        return group

    def _on_fkk_mode_changed(self, button):
        """Handle FKK mode radio button change."""
        if button == self.fkk_mode_design:
            self.fkk_design_widget.show()
            self.fkk_apply_widget.hide()
        else:
            self.fkk_design_widget.hide()
            self.fkk_apply_widget.show()

    def _on_fkk_design_clicked(self):
        """Handle FKK design button click."""
        self.fkk_design_requested.emit()

    def _on_fkk_preset_changed(self, index):
        """Handle FKK preset selection."""
        preset_name = self.fkk_preset_combo.currentData()
        if preset_name and preset_name in FKK_PRESETS:
            preset = FKK_PRESETS[preset_name]
            self.fkk_vmin_spin.setValue(preset.v_min)
            self.fkk_vmax_spin.setValue(preset.v_max)
            mode_idx = 0 if preset.mode == 'reject' else 1
            self.fkk_filter_mode_combo.setCurrentIndex(mode_idx)

    def _on_fkk_apply_clicked(self):
        """Handle FKK apply button click."""
        f_min = self.fkk_fmin_spin.value()
        f_max = self.fkk_fmax_spin.value()

        config = FKKConfig(
            v_min=self.fkk_vmin_spin.value(),
            v_max=self.fkk_vmax_spin.value(),
            mode=self.fkk_filter_mode_combo.currentData(),
            taper_width=0.1,
            apply_agc=self.fkk_agc_checkbox.isChecked(),
            agc_window_ms=self.fkk_agc_window_spin.value(),
            f_min=f_min if f_min > 0 else None,
            f_max=f_max if f_max < self.nyquist_freq else None
        )
        self.fkk_apply_requested.emit(config)

    def _create_display_group(self) -> QGroupBox:
        """Create display controls group."""
        group = QGroupBox("Display Controls")
        layout = QVBoxLayout()

        # Colormap selector
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Seismic (RWB)",
            "Grayscale",
            "Viridis",
            "Plasma",
            "Inferno",
            "Jet"
        ])
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        colormap_layout.addWidget(self.colormap_combo)
        layout.addLayout(colormap_layout)

        # Interpolation selector
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Interpolation:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems([
            "Smooth (Bilinear)",
            "Very Smooth (Bicubic)",
            "Sharp (Nearest)"
        ])
        self.interp_combo.setCurrentIndex(0)  # Default to Bilinear
        self.interp_combo.setToolTip(
            "Smooth: Bilinear interpolation (default)\n"
            "Very Smooth: Bicubic interpolation\n"
            "Sharp: Nearest-neighbor (no interpolation)"
        )
        self.interp_combo.currentIndexChanged.connect(self._on_interpolation_changed)
        interp_layout.addWidget(self.interp_combo)
        layout.addLayout(interp_layout)

        # Min amplitude
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Amp:"))
        self.min_amp_spin = QDoubleSpinBox()
        self.min_amp_spin.setRange(-1e6, 1e6)
        self.min_amp_spin.setValue(-1.0)
        self.min_amp_spin.setDecimals(4)
        self.min_amp_spin.setSingleStep(0.1)
        self.min_amp_spin.valueChanged.connect(self._on_amplitude_changed)
        min_layout.addWidget(self.min_amp_spin)
        layout.addLayout(min_layout)

        # Max amplitude
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Amp:"))
        self.max_amp_spin = QDoubleSpinBox()
        self.max_amp_spin.setRange(-1e6, 1e6)
        self.max_amp_spin.setValue(1.0)
        self.max_amp_spin.setDecimals(4)
        self.max_amp_spin.setSingleStep(0.1)
        self.max_amp_spin.valueChanged.connect(self._on_amplitude_changed)
        max_layout.addWidget(self.max_amp_spin)
        layout.addLayout(max_layout)

        # Auto-scale button
        auto_btn = QPushButton("Auto Scale from Data")
        auto_btn.clicked.connect(self._request_auto_scale)
        layout.addWidget(auto_btn)

        group.setLayout(layout)
        return group

    def _create_sort_group(self) -> QGroupBox:
        """Create in-gather sorting controls."""
        group = QGroupBox("In-Gather Sort")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Select headers to sort traces within each gather:")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(info_label)

        # Sort keys list (multi-select)
        self.sort_keys_list = QListWidget()
        self.sort_keys_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.sort_keys_list.setMaximumHeight(120)
        self.sort_keys_list.setToolTip(
            "Select one or more headers to sort by.\n"
            "Order of selection determines sort priority:\n"
            "First selected = primary sort key\n"
            "Second selected = secondary sort key, etc."
        )
        layout.addWidget(self.sort_keys_list)

        # Buttons
        btn_layout = QHBoxLayout()

        apply_sort_btn = QPushButton("Apply Sort")
        apply_sort_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        apply_sort_btn.clicked.connect(self._on_apply_sort)
        btn_layout.addWidget(apply_sort_btn)

        clear_sort_btn = QPushButton("Clear Sort")
        clear_sort_btn.clicked.connect(self._on_clear_sort)
        btn_layout.addWidget(clear_sort_btn)

        layout.addLayout(btn_layout)

        # Current sort display
        self.current_sort_label = QLabel("Sort: None")
        self.current_sort_label.setStyleSheet("color: #666; font-size: 9pt; font-weight: bold;")
        self.current_sort_label.setWordWrap(True)
        layout.addWidget(self.current_sort_label)

        group.setLayout(layout)
        return group

    def _create_view_group(self) -> QGroupBox:
        """Create view control buttons."""
        group = QGroupBox("View Controls")
        layout = QVBoxLayout()

        # Zoom buttons
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in_requested.emit)
        zoom_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out_requested.emit)
        zoom_layout.addWidget(zoom_out_btn)
        layout.addLayout(zoom_layout)

        # Reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view_requested.emit)
        layout.addWidget(reset_btn)

        group.setLayout(layout)
        return group

    def _on_algorithm_changed(self, index: int):
        """Handle algorithm selection change."""
        # Hide all algorithm groups first
        self.bandpass_group.hide()
        self.tfdenoise_group.hide()
        self.fk_filter_group.hide()
        self.fkk_filter_group.hide()

        if index == 0:  # Bandpass Filter
            self.bandpass_group.show()
            # Disable GPU for bandpass filter
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 1:  # TF-Denoise
            self.tfdenoise_group.show()
            # Enable GPU for TF-Denoise if available
            if self.gpu_checkbox is not None and self.gpu_available:
                self.gpu_checkbox.setEnabled(True)
        elif index == 2:  # FK Filter
            self.fk_filter_group.show()
            # Disable GPU for FK filter
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 3:  # 3D FKK Filter
            self.fkk_filter_group.show()
            # Enable GPU for 3D FKK filter if available
            if self.gpu_checkbox is not None and self.gpu_available:
                self.gpu_checkbox.setEnabled(True)

    def _on_gpu_checkbox_changed(self, state):
        """Handle GPU checkbox state change."""
        if self.gpu_status_label is not None:
            if state == Qt.CheckState.Checked.value:
                gpu_name = self.device_manager.get_device_name()
                self.gpu_status_label.setText(f"🟢 {gpu_name} (Enabled)")
            else:
                self.gpu_status_label.setText(f"🟡 GPU Disabled (Using CPU)")

    def _on_tfdenoise_preset_changed(self, index: int):
        """Handle TF-Denoise preset change."""
        # Preset configurations
        presets = {
            0: {"aperture": 11, "fmin": 5.0, "fmax": 30.0, "k": 2.5},  # Low-Freq Noise
            1: {"aperture": 15, "fmin": 5.0, "fmax": 150.0, "k": 3.5},  # White Noise
            2: {"aperture": 7, "fmin": 50.0, "fmax": 150.0, "k": 3.0},  # High-Freq Noise
            3: {"aperture": 9, "fmin": 10.0, "fmax": 100.0, "k": 2.0},  # Conservative
        }

        if index < 4:  # Not Custom
            preset = presets[index]
            self.aperture_spin.blockSignals(True)
            self.tfdenoise_fmin_spin.blockSignals(True)
            self.tfdenoise_fmax_spin.blockSignals(True)
            self.threshold_k_spin.blockSignals(True)

            self.aperture_spin.setValue(preset["aperture"])
            self.tfdenoise_fmin_spin.setValue(preset["fmin"])
            self.tfdenoise_fmax_spin.setValue(preset["fmax"])
            self.threshold_k_spin.setValue(preset["k"])

            self.aperture_spin.blockSignals(False)
            self.tfdenoise_fmin_spin.blockSignals(False)
            self.tfdenoise_fmax_spin.blockSignals(False)
            self.threshold_k_spin.blockSignals(False)

    def _on_apply_clicked(self):
        """Handle apply button click."""
        try:
            # Check which algorithm is selected
            if self.algorithm_combo.currentIndex() == 0:  # Bandpass Filter
                processor = BandpassFilter(
                    low_freq=self.low_freq_spin.value(),
                    high_freq=self.high_freq_spin.value(),
                    order=self.order_spin.value()
                )
            else:  # TF-Denoise
                # Check if GPU should be used
                use_gpu = (
                    GPU_AVAILABLE and
                    self.gpu_checkbox is not None and
                    self.gpu_checkbox.isChecked()
                )

                # Get threshold mode from combo box
                threshold_mode_map = {
                    0: 'adaptive',  # Adaptive (Recommended)
                    1: 'hard',      # Hard (Full Removal)
                    2: 'scaled',    # Scaled (Progressive)
                    3: 'soft'       # Soft (Legacy)
                }
                threshold_mode = threshold_mode_map.get(
                    self.threshold_mode_combo.currentIndex(), 'adaptive'
                )

                # Get low-amplitude protection setting
                low_amp_protection = self.low_amp_protection_checkbox.isChecked()

                if use_gpu:
                    # Use GPU-accelerated version
                    processor = TFDenoiseGPU(
                        aperture=self.aperture_spin.value(),
                        fmin=self.tfdenoise_fmin_spin.value(),
                        fmax=self.tfdenoise_fmax_spin.value(),
                        threshold_k=self.threshold_k_spin.value(),
                        threshold_type=self.threshold_type_combo.currentText().lower(),
                        threshold_mode=threshold_mode,
                        transform_type=self.transform_type_combo.currentText().lower().replace("-", ""),
                        use_gpu='auto',
                        low_amp_protection=low_amp_protection,
                        device_manager=self.device_manager
                    )
                    print(f"✓ Using GPU-accelerated TF-Denoise: {self.device_manager.get_device_name()}")
                    print(f"  Threshold mode: {threshold_mode}, Low-amp protection: {low_amp_protection}")
                else:
                    # Use CPU version
                    from processors.tf_denoise import TFDenoise
                    processor = TFDenoise(
                        aperture=self.aperture_spin.value(),
                        fmin=self.tfdenoise_fmin_spin.value(),
                        fmax=self.tfdenoise_fmax_spin.value(),
                        threshold_k=self.threshold_k_spin.value(),
                        threshold_type=self.threshold_type_combo.currentText().lower(),
                        threshold_mode=threshold_mode,
                        transform_type=self.transform_type_combo.currentText().lower().replace("-", ""),
                        low_amp_protection=low_amp_protection
                    )
                    print(f"✓ Using CPU TF-Denoise")
                    print(f"  Threshold mode: {threshold_mode}, Low-amp protection: {low_amp_protection}")

            self.process_requested.emit(processor)
        except ValueError as e:
            # Could show error dialog here
            print(f"Error creating processor: {e}")
        except Exception as e:
            print(f"Error: {e}")

    def _on_amplitude_changed(self):
        """Handle amplitude spinbox changes."""
        min_amp = self.min_amp_spin.value()
        max_amp = self.max_amp_spin.value()

        if max_amp > min_amp:
            self.amplitude_range_changed.emit(min_amp, max_amp)

    def _on_colormap_changed(self, index: int):
        """Handle colormap selection change."""
        colormap_map = {
            0: 'seismic',
            1: 'grayscale',
            2: 'viridis',
            3: 'plasma',
            4: 'inferno',
            5: 'jet'
        }
        colormap = colormap_map.get(index, 'seismic')
        self.colormap_changed.emit(colormap)

    def _on_interpolation_changed(self, index: int):
        """Handle interpolation selection change."""
        interp_map = {
            0: 'bilinear',   # Smooth (Bilinear)
            1: 'bicubic',    # Very Smooth (Bicubic)
            2: 'nearest'     # Sharp (Nearest)
        }
        interpolation = interp_map.get(index, 'bilinear')
        self.interpolation_changed.emit(interpolation)

    def _on_apply_sort(self):
        """Handle apply sort button click."""
        # Get selected items in order
        selected_items = self.sort_keys_list.selectedItems()

        if not selected_items:
            # No selection - clear sort
            self._on_clear_sort()
            return

        # Get sort keys in selection order
        sort_keys = [item.text() for item in selected_items]

        # Update display
        sort_text = "Sort: " + " → ".join(sort_keys)
        self.current_sort_label.setText(sort_text)

        # Emit signal
        self.sort_keys_changed.emit(sort_keys)

    def _on_clear_sort(self):
        """Handle clear sort button click."""
        # Clear selection
        self.sort_keys_list.clearSelection()

        # Update display
        self.current_sort_label.setText("Sort: None")

        # Emit signal with empty list
        self.sort_keys_changed.emit([])

    def _request_auto_scale(self):
        """Request auto-scaling (will be connected to main window)."""
        # Emit signal to request auto-scale
        # Main window will calculate from data and call set_amplitude_range
        pass

    def update_nyquist(self, nyquist_freq: float):
        """Update Nyquist frequency based on loaded data."""
        self.nyquist_freq = nyquist_freq
        self.high_freq_spin.setMaximum(nyquist_freq - 0.1)
        self.low_freq_spin.setMaximum(nyquist_freq - 1.0)

        # Update display
        group = self.findChild(QGroupBox, "Bandpass Filter")
        if group:
            for widget in group.findChildren(QLabel):
                if "Nyquist" in widget.text():
                    widget.setText(f"Nyquist: {nyquist_freq:.1f} Hz")

    def set_amplitude_range(self, min_amp: float, max_amp: float):
        """Set amplitude range values (e.g., from auto-scale)."""
        self.min_amp_spin.blockSignals(True)
        self.max_amp_spin.blockSignals(True)

        self.min_amp_spin.setValue(min_amp)
        self.max_amp_spin.setValue(max_amp)

        self.min_amp_spin.blockSignals(False)
        self.max_amp_spin.blockSignals(False)

        # Emit change
        self.amplitude_range_changed.emit(min_amp, max_amp)

    def set_available_sort_headers(self, headers: list):
        """
        Populate the sort keys list with available headers.

        Args:
            headers: List of header names that can be used for sorting
        """
        self.sort_keys_list.clear()
        self.sort_keys_list.addItems(headers)

    def update_sort_display(self, sort_keys: list):
        """
        Update the current sort display label.

        Args:
            sort_keys: List of active sort keys
        """
        if sort_keys:
            sort_text = "Sort: " + " → ".join(sort_keys)
        else:
            sort_text = "Sort: None"

        self.current_sort_label.setText(sort_text)

    # FK Filter event handlers

    def _on_fk_mode_changed(self):
        """Handle FK mode (Design/Apply) change."""
        if self.fk_mode_design.isChecked():
            self.fk_design_widget.show()
            self.fk_apply_widget.hide()
        else:
            self.fk_design_widget.hide()
            self.fk_apply_widget.show()
            self._refresh_fk_config_list()

    def _on_fk_design_clicked(self):
        """Handle FK design button click."""
        self.fk_design_requested.emit()

    def _on_fk_apply_clicked(self):
        """Handle FK apply button click."""
        selected_items = self.fk_config_list.selectedItems()
        if selected_items:
            config_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
            self.fk_config_selected.emit(config_name)

    def _on_fk_config_selection_changed(self):
        """Handle FK config list selection change."""
        has_selection = len(self.fk_config_list.selectedItems()) > 0
        self.fk_apply_btn.setEnabled(has_selection)

    def _on_fk_delete_clicked(self):
        """Handle FK delete button click."""
        from PyQt6.QtWidgets import QMessageBox

        selected_items = self.fk_config_list.selectedItems()
        if not selected_items:
            return

        config_name = selected_items[0].data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Delete Configuration",
            f"Are you sure you want to delete '{config_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.fk_config_manager.delete_config(config_name)
            self._refresh_fk_config_list()

    def _refresh_fk_config_list(self):
        """Refresh the FK configuration list."""
        self.fk_config_list.clear()

        configs = self.fk_config_manager.get_all_configs()
        for config in configs:
            item = QListWidgetItem(f"{config.name}\n{config.get_summary()}")
            item.setData(Qt.ItemDataRole.UserRole, config.name)
            self.fk_config_list.addItem(item)

    def refresh_fk_configs(self):
        """Public method to refresh FK configs (called after saving new config)."""
        self._refresh_fk_config_list()
