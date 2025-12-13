"""
Control panel - user interface for processing parameters and display controls.
"""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QLabel, QDoubleSpinBox, QSlider,
                              QSpinBox, QComboBox, QListWidget, QAbstractItemView,
                              QCheckBox, QScrollArea, QRadioButton, QButtonGroup,
                              QListWidgetItem, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
import sys
from processors.bandpass_filter import BandpassFilter
from processors.dwt_denoise import DWTDenoise, PYWT_AVAILABLE
from processors.gabor_denoise import GaborDenoise
from processors.emd_denoise import EMDDenoise, PYEMD_AVAILABLE
from processors.stft_denoise import STFTDenoise
from processors.stockwell_denoise import StockwellDenoise
from processors.omp_denoise import OMPDenoise
from processors.deconvolution import DeconvolutionProcessor, DeconConfig
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
    # PSTM signals
    pstm_apply_requested = pyqtSignal(float, float, float)  # velocity, aperture, max_angle
    pstm_wizard_requested = pyqtSignal()  # Request to open PSTM wizard
    # Mute signals
    mute_apply_requested = pyqtSignal(object)  # MuteConfig or None

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
        self.stockwell_group = self._create_stockwell_group()
        self.stft_group = self._create_stft_group()
        self.dwtdenoise_group = self._create_dwtdenoise_group()
        self.gabor_group = self._create_gabor_group()
        self.emd_group = self._create_emd_group()
        self.omp_group = self._create_omp_group()
        self.denoise3d_group = self._create_denoise3d_group()
        self.fk_filter_group = self._create_fk_filter_group()
        self.fkk_filter_group = self._create_fkk_filter_group()
        self.pstm_group = self._create_pstm_group()
        controls_layout.addWidget(self.bandpass_group)
        controls_layout.addWidget(self.stockwell_group)
        controls_layout.addWidget(self.stft_group)
        controls_layout.addWidget(self.dwtdenoise_group)
        controls_layout.addWidget(self.gabor_group)
        controls_layout.addWidget(self.emd_group)
        controls_layout.addWidget(self.omp_group)
        controls_layout.addWidget(self.denoise3d_group)
        controls_layout.addWidget(self.fk_filter_group)
        controls_layout.addWidget(self.fkk_filter_group)
        controls_layout.addWidget(self.pstm_group)
        self.mute_group = self._create_mute_group()
        controls_layout.addWidget(self.mute_group)
        self.deconvolution_group = self._create_deconvolution_group()
        controls_layout.addWidget(self.deconvolution_group)
        self.deconvolution_group.hide()  # Initially hidden
        self.stockwell_group.hide()  # Initially hidden
        self.stft_group.hide()  # Initially hidden
        self.dwtdenoise_group.hide()  # Initially hidden
        self.gabor_group.hide()  # Initially hidden
        self.emd_group.hide()  # Initially hidden
        self.omp_group.hide()  # Initially hidden
        self.denoise3d_group.hide()  # Initially hidden
        self.fk_filter_group.hide()  # Initially hidden
        self.fkk_filter_group.hide()  # Initially hidden
        self.pstm_group.hide()  # Initially hidden

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
            "Stockwell Transform (ST)",
            "STFT Denoise",
            "DWT-Denoise (Wavelet)",
            "Gabor Transform",
            "EMD Decomposition",
            "OMP Sparse Denoise",
            "3D Spatial Denoise",
            "FK Filter",
            "3D FKK Filter",
            "Kirchhoff PSTM",
            "Deconvolution"
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

    def _create_stockwell_group(self) -> QGroupBox:
        """Create Stockwell Transform (S-Transform) parameters group."""
        group = QGroupBox("Stockwell Transform (ST) Parameters")
        layout = QVBoxLayout()

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.stockwell_preset_combo = QComboBox()
        self.stockwell_preset_combo.addItems([
            "Low-Freq Noise",
            "White Noise",
            "High-Freq Noise",
            "Conservative",
            "Custom"
        ])
        self.stockwell_preset_combo.currentIndexChanged.connect(self._on_stockwell_preset_changed)
        preset_layout.addWidget(self.stockwell_preset_combo)
        layout.addLayout(preset_layout)

        # Spatial aperture
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Spatial Aperture:"))
        self.stockwell_aperture_spin = QSpinBox()
        self.stockwell_aperture_spin.setRange(3, 21)
        self.stockwell_aperture_spin.setValue(7)
        self.stockwell_aperture_spin.setSingleStep(2)
        self.stockwell_aperture_spin.setToolTip("Number of traces to use for spatial denoising (odd numbers)")
        aperture_layout.addWidget(self.stockwell_aperture_spin)
        layout.addLayout(aperture_layout)

        # Frequency range
        fmin_layout = QHBoxLayout()
        fmin_layout.addWidget(QLabel("Min Freq (Hz):"))
        self.stockwell_fmin_spin = QDoubleSpinBox()
        self.stockwell_fmin_spin.setRange(1.0, self.nyquist_freq - 1)
        self.stockwell_fmin_spin.setValue(5.0)
        self.stockwell_fmin_spin.setDecimals(1)
        fmin_layout.addWidget(self.stockwell_fmin_spin)
        layout.addLayout(fmin_layout)

        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Freq (Hz):"))
        self.stockwell_fmax_spin = QDoubleSpinBox()
        self.stockwell_fmax_spin.setRange(2.0, self.nyquist_freq - 0.1)
        self.stockwell_fmax_spin.setValue(100.0)
        self.stockwell_fmax_spin.setDecimals(1)
        fmax_layout.addWidget(self.stockwell_fmax_spin)
        layout.addLayout(fmax_layout)

        # MAD threshold multiplier
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("Threshold (k):"))
        self.stockwell_threshold_k_spin = QDoubleSpinBox()
        self.stockwell_threshold_k_spin.setRange(0.5, 10.0)
        self.stockwell_threshold_k_spin.setValue(3.0)
        self.stockwell_threshold_k_spin.setSingleStep(0.5)
        self.stockwell_threshold_k_spin.setDecimals(1)
        self.stockwell_threshold_k_spin.setToolTip("MAD threshold multiplier (higher = more aggressive)")
        k_layout.addWidget(self.stockwell_threshold_k_spin)
        layout.addLayout(k_layout)

        # Threshold mode
        threshold_mode_layout = QHBoxLayout()
        threshold_mode_layout.addWidget(QLabel("Noise Removal:"))
        self.stockwell_threshold_mode_combo = QComboBox()
        self.stockwell_threshold_mode_combo.addItems([
            "Adaptive (Recommended)",
            "Hard (Full Removal)",
            "Scaled (Progressive)",
            "Soft (Legacy)"
        ])
        self.stockwell_threshold_mode_combo.setToolTip(
            "Adaptive: Hard for severe outliers, scaled for moderate (recommended)\n"
            "Hard: Full removal for all outliers\n"
            "Scaled: Progressive removal based on severity\n"
            "Soft: Legacy partial removal"
        )
        threshold_mode_layout.addWidget(self.stockwell_threshold_mode_combo)
        layout.addLayout(threshold_mode_layout)

        # Low-amplitude protection
        self.stockwell_low_amp_checkbox = QCheckBox("Low-amplitude protection")
        self.stockwell_low_amp_checkbox.setChecked(True)
        self.stockwell_low_amp_checkbox.setToolTip(
            "Prevent inflation of low-amplitude samples\n"
            "(isolated signals won't be boosted toward median)"
        )
        layout.addWidget(self.stockwell_low_amp_checkbox)

        # Apply button
        self.stockwell_apply_btn = QPushButton("Apply Stockwell Denoise")
        self.stockwell_apply_btn.setStyleSheet("""
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
        self.stockwell_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.stockwell_apply_btn)

        group.setLayout(layout)
        return group

    def _create_stft_group(self) -> QGroupBox:
        """Create STFT Denoise parameters group."""
        group = QGroupBox("STFT Denoise Parameters")
        layout = QVBoxLayout()

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.stft_preset_combo = QComboBox()
        self.stft_preset_combo.addItems([
            "Low-Freq Noise",
            "White Noise",
            "High-Freq Noise",
            "Conservative",
            "Custom"
        ])
        self.stft_preset_combo.currentIndexChanged.connect(self._on_stft_preset_changed)
        preset_layout.addWidget(self.stft_preset_combo)
        layout.addLayout(preset_layout)

        # Spatial aperture
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Spatial Aperture:"))
        self.stft_aperture_spin = QSpinBox()
        self.stft_aperture_spin.setRange(3, 21)
        self.stft_aperture_spin.setValue(7)
        self.stft_aperture_spin.setSingleStep(2)
        self.stft_aperture_spin.setToolTip("Number of traces to use for spatial denoising (odd numbers)")
        aperture_layout.addWidget(self.stft_aperture_spin)
        layout.addLayout(aperture_layout)

        # Window size (nperseg) - STFT specific
        nperseg_layout = QHBoxLayout()
        nperseg_layout.addWidget(QLabel("Window Size:"))
        self.stft_nperseg_spin = QSpinBox()
        self.stft_nperseg_spin.setRange(16, 256)
        self.stft_nperseg_spin.setValue(64)
        self.stft_nperseg_spin.setSingleStep(16)
        self.stft_nperseg_spin.setToolTip("STFT window size (samples). Larger = better freq resolution, worse time resolution")
        nperseg_layout.addWidget(self.stft_nperseg_spin)
        layout.addLayout(nperseg_layout)

        # Frequency range
        fmin_layout = QHBoxLayout()
        fmin_layout.addWidget(QLabel("Min Freq (Hz):"))
        self.stft_fmin_spin = QDoubleSpinBox()
        self.stft_fmin_spin.setRange(1.0, self.nyquist_freq - 1)
        self.stft_fmin_spin.setValue(5.0)
        self.stft_fmin_spin.setDecimals(1)
        fmin_layout.addWidget(self.stft_fmin_spin)
        layout.addLayout(fmin_layout)

        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Freq (Hz):"))
        self.stft_fmax_spin = QDoubleSpinBox()
        self.stft_fmax_spin.setRange(2.0, self.nyquist_freq - 0.1)
        self.stft_fmax_spin.setValue(100.0)
        self.stft_fmax_spin.setDecimals(1)
        fmax_layout.addWidget(self.stft_fmax_spin)
        layout.addLayout(fmax_layout)

        # MAD threshold multiplier
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("Threshold (k):"))
        self.stft_threshold_k_spin = QDoubleSpinBox()
        self.stft_threshold_k_spin.setRange(0.5, 10.0)
        self.stft_threshold_k_spin.setValue(3.0)
        self.stft_threshold_k_spin.setSingleStep(0.5)
        self.stft_threshold_k_spin.setDecimals(1)
        self.stft_threshold_k_spin.setToolTip("MAD threshold multiplier (higher = more aggressive)")
        k_layout.addWidget(self.stft_threshold_k_spin)
        layout.addLayout(k_layout)

        # Threshold mode
        threshold_mode_layout = QHBoxLayout()
        threshold_mode_layout.addWidget(QLabel("Noise Removal:"))
        self.stft_threshold_mode_combo = QComboBox()
        self.stft_threshold_mode_combo.addItems([
            "Adaptive (Recommended)",
            "Hard (Full Removal)",
            "Scaled (Progressive)",
            "Soft (Legacy)"
        ])
        self.stft_threshold_mode_combo.setToolTip(
            "Adaptive: Hard for severe outliers, scaled for moderate (recommended)\n"
            "Hard: Full removal for all outliers\n"
            "Scaled: Progressive removal based on severity\n"
            "Soft: Legacy partial removal"
        )
        threshold_mode_layout.addWidget(self.stft_threshold_mode_combo)
        layout.addLayout(threshold_mode_layout)

        # Low-amplitude protection
        self.stft_low_amp_checkbox = QCheckBox("Low-amplitude protection")
        self.stft_low_amp_checkbox.setChecked(True)
        self.stft_low_amp_checkbox.setToolTip(
            "Prevent inflation of low-amplitude samples\n"
            "(isolated signals won't be boosted toward median)"
        )
        layout.addWidget(self.stft_low_amp_checkbox)

        # Apply button
        self.stft_apply_btn = QPushButton("Apply STFT Denoise")
        self.stft_apply_btn.setStyleSheet("""
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
        self.stft_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.stft_apply_btn)

        group.setLayout(layout)
        return group

    def _create_dwtdenoise_group(self) -> QGroupBox:
        """Create DWT-Denoise parameters group."""
        group = QGroupBox("DWT-Denoise Parameters")
        layout = QVBoxLayout()

        # PyWavelets availability check
        if not PYWT_AVAILABLE:
            warning_label = QLabel("PyWavelets not installed.\nInstall with: pip install PyWavelets")
            warning_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(warning_label)
            group.setLayout(layout)
            return group

        # Wavelet selection
        wavelet_layout = QHBoxLayout()
        wavelet_layout.addWidget(QLabel("Wavelet:"))
        self.dwt_wavelet_combo = QComboBox()
        self.dwt_wavelet_combo.addItems([
            "db4 (Daubechies-4)",
            "db8 (Daubechies-8)",
            "sym4 (Symlet-4)",
            "sym8 (Symlet-8)",
            "coif4 (Coiflet-4)",
            "bior3.5 (Biorthogonal)"
        ])
        self.dwt_wavelet_combo.setToolTip(
            "db4/db8: Good general purpose (recommended)\n"
            "sym4/sym8: More symmetric wavelets\n"
            "coif4: Nearly symmetric, good for smooth signals\n"
            "bior3.5: Linear phase, good for sharp edges"
        )
        wavelet_layout.addWidget(self.dwt_wavelet_combo)
        layout.addLayout(wavelet_layout)

        # Transform type
        transform_layout = QHBoxLayout()
        transform_layout.addWidget(QLabel("Transform:"))
        self.dwt_transform_combo = QComboBox()
        self.dwt_transform_combo.addItems([
            "DWT (Fast)",
            "SWT (Translation-Invariant)",
            "DWT-Spatial (with Aperture)",
            "WPT (Wavelet Packets)",
            "WPT-Spatial (with Aperture)"
        ])
        self.dwt_transform_combo.setToolTip(
            "DWT: Fast standard wavelet transform (5-10x faster)\n"
            "SWT: Stationary WT, avoids shift artifacts (slower)\n"
            "DWT-Spatial: Uses spatial aperture for robust thresholding\n"
            "WPT: Wavelet Packets - full tree, better frequency resolution\n"
            "WPT-Spatial: WPT with spatial aperture processing"
        )
        self.dwt_transform_combo.currentIndexChanged.connect(self._on_dwt_transform_changed)
        transform_layout.addWidget(self.dwt_transform_combo)
        layout.addLayout(transform_layout)

        # Decomposition level
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Level:"))
        self.dwt_level_spin = QSpinBox()
        self.dwt_level_spin.setRange(1, 10)
        self.dwt_level_spin.setValue(5)
        self.dwt_level_spin.setToolTip("Decomposition level (higher = more scales)")
        level_layout.addWidget(self.dwt_level_spin)
        layout.addLayout(level_layout)

        # MAD threshold multiplier
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("Threshold (k):"))
        self.dwt_threshold_k_spin = QDoubleSpinBox()
        self.dwt_threshold_k_spin.setRange(0.5, 10.0)
        self.dwt_threshold_k_spin.setValue(2.5)
        self.dwt_threshold_k_spin.setSingleStep(0.5)
        self.dwt_threshold_k_spin.setDecimals(1)
        self.dwt_threshold_k_spin.setToolTip("MAD threshold multiplier (lower = more aggressive)")
        k_layout.addWidget(self.dwt_threshold_k_spin)
        layout.addLayout(k_layout)

        # Threshold mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Threshold Mode:"))
        self.dwt_threshold_mode_combo = QComboBox()
        self.dwt_threshold_mode_combo.addItems([
            "Soft (Wavelet Shrinkage)",
            "Hard (Keep/Zero)"
        ])
        self.dwt_threshold_mode_combo.setToolTip(
            "Soft: Shrinks coefficients toward zero (smoother)\n"
            "Hard: Keeps or zeros coefficients (preserves edges)"
        )
        mode_layout.addWidget(self.dwt_threshold_mode_combo)
        layout.addLayout(mode_layout)

        # Spatial aperture (only for spatial modes)
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Spatial Aperture:"))
        self.dwt_aperture_spin = QSpinBox()
        self.dwt_aperture_spin.setRange(3, 21)
        self.dwt_aperture_spin.setValue(7)
        self.dwt_aperture_spin.setSingleStep(2)
        self.dwt_aperture_spin.setToolTip("Number of traces for spatial MAD estimation (odd)")
        self.dwt_aperture_spin.setEnabled(False)  # Disabled by default
        aperture_layout.addWidget(self.dwt_aperture_spin)
        layout.addLayout(aperture_layout)
        self.dwt_aperture_layout = aperture_layout

        # Best-basis selection (only for WPT modes)
        self.dwt_best_basis_checkbox = QCheckBox("Best-basis selection (Shannon entropy)")
        self.dwt_best_basis_checkbox.setChecked(False)
        self.dwt_best_basis_checkbox.setEnabled(False)  # Disabled by default
        self.dwt_best_basis_checkbox.setToolTip(
            "Use Shannon entropy to find optimal decomposition tree.\n"
            "Can improve denoising for non-stationary signals."
        )
        layout.addWidget(self.dwt_best_basis_checkbox)

        # Performance info
        perf_label = QLabel("5-10x faster than STFT with comparable quality")
        perf_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        layout.addWidget(perf_label)

        # Apply button
        self.dwtdenoise_apply_btn = QPushButton("Apply DWT-Denoise")
        self.dwtdenoise_apply_btn.setStyleSheet("""
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
                background-color: #1565C0;
            }
        """)
        self.dwtdenoise_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.dwtdenoise_apply_btn)

        group.setLayout(layout)
        return group

    def _on_dwt_transform_changed(self, index: int):
        """Handle DWT transform type change."""
        # Enable aperture for spatial modes (index 2=DWT-Spatial, 4=WPT-Spatial)
        self.dwt_aperture_spin.setEnabled(index in [2, 4])
        # Enable best-basis for WPT modes (index 3=WPT, 4=WPT-Spatial)
        self.dwt_best_basis_checkbox.setEnabled(index in [3, 4])

    def _create_gabor_group(self) -> QGroupBox:
        """Create Gabor Transform Denoise parameters group."""
        group = QGroupBox("Gabor Transform Parameters")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("STFT with Gaussian windows for optimal TF localization")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Window size
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window Size:"))
        self.gabor_window_spin = QSpinBox()
        self.gabor_window_spin.setRange(16, 512)
        self.gabor_window_spin.setValue(64)
        self.gabor_window_spin.setSingleStep(16)
        self.gabor_window_spin.setToolTip("Gabor window size in samples (power of 2 recommended)")
        window_layout.addWidget(self.gabor_window_spin)
        layout.addLayout(window_layout)

        # Sigma (Gaussian width)
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.gabor_sigma_spin = QDoubleSpinBox()
        self.gabor_sigma_spin.setRange(0.0, 100.0)
        self.gabor_sigma_spin.setValue(0.0)
        self.gabor_sigma_spin.setSingleStep(1.0)
        self.gabor_sigma_spin.setDecimals(1)
        self.gabor_sigma_spin.setSpecialValueText("Auto")
        self.gabor_sigma_spin.setToolTip(
            "Gaussian window standard deviation.\n"
            "0 = Auto (window_size/6 for optimal TF trade-off)\n"
            "Lower = better frequency resolution\n"
            "Higher = better time resolution"
        )
        sigma_layout.addWidget(self.gabor_sigma_spin)
        layout.addLayout(sigma_layout)

        # Overlap percentage
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap %:"))
        self.gabor_overlap_spin = QSpinBox()
        self.gabor_overlap_spin.setRange(25, 90)
        self.gabor_overlap_spin.setValue(75)
        self.gabor_overlap_spin.setSingleStep(5)
        self.gabor_overlap_spin.setToolTip("Window overlap percentage (75% typical)")
        overlap_layout.addWidget(self.gabor_overlap_spin)
        layout.addLayout(overlap_layout)

        # Spatial aperture
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Spatial Aperture:"))
        self.gabor_aperture_spin = QSpinBox()
        self.gabor_aperture_spin.setRange(3, 21)
        self.gabor_aperture_spin.setValue(7)
        self.gabor_aperture_spin.setSingleStep(2)
        self.gabor_aperture_spin.setToolTip("Number of traces for spatial MAD estimation (odd)")
        aperture_layout.addWidget(self.gabor_aperture_spin)
        layout.addLayout(aperture_layout)

        # Frequency range
        fmin_layout = QHBoxLayout()
        fmin_layout.addWidget(QLabel("Min Freq (Hz):"))
        self.gabor_fmin_spin = QDoubleSpinBox()
        self.gabor_fmin_spin.setRange(1.0, self.nyquist_freq - 1)
        self.gabor_fmin_spin.setValue(5.0)
        self.gabor_fmin_spin.setDecimals(1)
        fmin_layout.addWidget(self.gabor_fmin_spin)
        layout.addLayout(fmin_layout)

        fmax_layout = QHBoxLayout()
        fmax_layout.addWidget(QLabel("Max Freq (Hz):"))
        self.gabor_fmax_spin = QDoubleSpinBox()
        self.gabor_fmax_spin.setRange(2.0, self.nyquist_freq - 0.1)
        self.gabor_fmax_spin.setValue(100.0)
        self.gabor_fmax_spin.setDecimals(1)
        fmax_layout.addWidget(self.gabor_fmax_spin)
        layout.addLayout(fmax_layout)

        # MAD threshold multiplier
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("Threshold (k):"))
        self.gabor_threshold_k_spin = QDoubleSpinBox()
        self.gabor_threshold_k_spin.setRange(0.5, 10.0)
        self.gabor_threshold_k_spin.setValue(3.0)
        self.gabor_threshold_k_spin.setSingleStep(0.5)
        self.gabor_threshold_k_spin.setDecimals(1)
        self.gabor_threshold_k_spin.setToolTip("MAD threshold multiplier (higher = more aggressive)")
        k_layout.addWidget(self.gabor_threshold_k_spin)
        layout.addLayout(k_layout)

        # Threshold mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Threshold Mode:"))
        self.gabor_threshold_mode_combo = QComboBox()
        self.gabor_threshold_mode_combo.addItems([
            "Soft (Shrinkage)",
            "Hard (Keep/Zero)"
        ])
        self.gabor_threshold_mode_combo.setToolTip(
            "Soft: Shrinks coefficients toward median (smoother)\n"
            "Hard: Keeps or zeros coefficients (preserves edges)"
        )
        mode_layout.addWidget(self.gabor_threshold_mode_combo)
        layout.addLayout(mode_layout)

        # Low-amplitude protection
        self.gabor_low_amp_checkbox = QCheckBox("Low-amplitude protection")
        self.gabor_low_amp_checkbox.setChecked(True)
        self.gabor_low_amp_checkbox.setToolTip("Prevent inflation of low-amplitude samples")
        layout.addWidget(self.gabor_low_amp_checkbox)

        # Apply button
        self.gabor_apply_btn = QPushButton("Apply Gabor Denoise")
        self.gabor_apply_btn.setStyleSheet("""
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
        self.gabor_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.gabor_apply_btn)

        group.setLayout(layout)
        return group

    def _create_emd_group(self) -> QGroupBox:
        """Create EMD Decomposition parameters group."""
        group = QGroupBox("EMD Decomposition Parameters")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Adaptive decomposition into Intrinsic Mode Functions")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.emd_method_combo = QComboBox()
        self.emd_method_combo.addItems([
            "EMD (Fast)",
            "EEMD (Robust)",
            "CEEMDAN (Best Quality)"
        ])
        self.emd_method_combo.setCurrentIndex(1)  # Default to EEMD
        self.emd_method_combo.setToolTip(
            "EMD: Fast but may have mode mixing\n"
            "EEMD: Ensemble EMD, noise-assisted (robust)\n"
            "CEEMDAN: Complete EEMD (best quality, slowest)"
        )
        self.emd_method_combo.currentIndexChanged.connect(self._on_emd_method_changed)
        method_layout.addWidget(self.emd_method_combo)
        layout.addLayout(method_layout)

        # IMF removal strategy
        remove_layout = QHBoxLayout()
        remove_layout.addWidget(QLabel("Remove IMFs:"))
        self.emd_remove_combo = QComboBox()
        self.emd_remove_combo.addItems([
            "First (High-freq noise)",
            "First 2",
            "First 3",
            "Last (Low-freq trend)",
            "Last 2",
            "Custom"
        ])
        self.emd_remove_combo.setToolTip(
            "IMF 0 = highest frequency (often noise)\n"
            "Last IMF = lowest frequency (trend/DC)"
        )
        remove_layout.addWidget(self.emd_remove_combo)
        layout.addLayout(remove_layout)

        # Custom IMF indices (hidden by default)
        self.emd_custom_layout = QHBoxLayout()
        self.emd_custom_layout.addWidget(QLabel("IMF indices:"))
        self.emd_custom_edit = QSpinBox()
        self.emd_custom_edit.setRange(0, 10)
        self.emd_custom_edit.setValue(0)
        self.emd_custom_edit.setToolTip("Enter IMF indices to remove (0-indexed)")
        self.emd_custom_layout.addWidget(self.emd_custom_edit)
        self.emd_custom_widget = QWidget()
        self.emd_custom_widget.setLayout(self.emd_custom_layout)
        self.emd_custom_widget.hide()
        layout.addWidget(self.emd_custom_widget)
        self.emd_remove_combo.currentIndexChanged.connect(self._on_emd_remove_changed)

        # Ensemble size (for EEMD/CEEMDAN)
        ensemble_layout = QHBoxLayout()
        ensemble_layout.addWidget(QLabel("Ensemble size:"))
        self.emd_ensemble_spin = QSpinBox()
        self.emd_ensemble_spin.setRange(10, 500)
        self.emd_ensemble_spin.setValue(100)
        self.emd_ensemble_spin.setSingleStep(10)
        self.emd_ensemble_spin.setToolTip("Number of noise realizations (EEMD/CEEMDAN only)")
        ensemble_layout.addWidget(self.emd_ensemble_spin)
        layout.addLayout(ensemble_layout)

        # Noise amplitude
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise amplitude:"))
        self.emd_noise_spin = QDoubleSpinBox()
        self.emd_noise_spin.setRange(0.01, 1.0)
        self.emd_noise_spin.setValue(0.2)
        self.emd_noise_spin.setSingleStep(0.05)
        self.emd_noise_spin.setDecimals(2)
        self.emd_noise_spin.setToolTip("Added noise amplitude as fraction of signal std")
        noise_layout.addWidget(self.emd_noise_spin)
        layout.addLayout(noise_layout)

        # Warning for slow methods
        self.emd_warning_label = QLabel("Note: EEMD/CEEMDAN can be slow for large data")
        self.emd_warning_label.setStyleSheet("color: #FF9800; font-size: 9pt;")
        layout.addWidget(self.emd_warning_label)

        # Apply button
        self.emd_apply_btn = QPushButton("Apply EMD Denoise")
        self.emd_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:pressed {
                background-color: #D84315;
            }
        """)
        self.emd_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.emd_apply_btn)

        group.setLayout(layout)
        return group

    def _on_emd_method_changed(self, index: int):
        """Handle EMD method selection change."""
        # Enable ensemble/noise controls only for EEMD/CEEMDAN
        ensemble_enabled = index > 0
        self.emd_ensemble_spin.setEnabled(ensemble_enabled)
        self.emd_noise_spin.setEnabled(ensemble_enabled)
        self.emd_warning_label.setVisible(ensemble_enabled)

    def _on_emd_remove_changed(self, index: int):
        """Handle IMF removal strategy change."""
        # Show custom input only for "Custom" option
        self.emd_custom_widget.setVisible(index == 5)

    def _create_omp_group(self) -> QGroupBox:
        """Create OMP Sparse Denoise parameters group."""
        group = QGroupBox("OMP Sparse Denoise Parameters")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Sparse representation denoising using Orthogonal Matching Pursuit")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Dictionary type selection
        dict_layout = QHBoxLayout()
        dict_layout.addWidget(QLabel("Dictionary:"))
        self.omp_dict_combo = QComboBox()
        self.omp_dict_combo.addItems([
            "DCT (Fast, smooth signals)",
            "DFT (Oscillatory)",
            "Gabor (Time-frequency)",
            "Wavelet (Seismic)",
            "Hybrid (Multi-basis)"
        ])
        self.omp_dict_combo.setCurrentIndex(3)  # Default to Wavelet for seismic
        self.omp_dict_combo.setToolTip(
            "DCT: Discrete Cosine Transform - fast, good for smooth signals\n"
            "DFT: Discrete Fourier Transform - periodic/oscillatory signals\n"
            "Gabor: Localized in time and frequency - transient events\n"
            "Wavelet: Ricker wavelets - best for seismic reflections\n"
            "Hybrid: Combination of all bases - most flexible"
        )
        dict_layout.addWidget(self.omp_dict_combo)
        layout.addLayout(dict_layout)

        # Patch size
        patch_layout = QHBoxLayout()
        patch_layout.addWidget(QLabel("Patch size:"))
        self.omp_patch_spin = QSpinBox()
        self.omp_patch_spin.setRange(16, 256)
        self.omp_patch_spin.setValue(64)
        self.omp_patch_spin.setSingleStep(16)
        self.omp_patch_spin.setToolTip("Size of signal patches for sparse coding (samples)")
        patch_layout.addWidget(self.omp_patch_spin)
        layout.addLayout(patch_layout)

        # Sparsity level
        sparsity_layout = QHBoxLayout()
        sparsity_layout.addWidget(QLabel("Sparsity (atoms):"))
        self.omp_sparsity_spin = QSpinBox()
        self.omp_sparsity_spin.setRange(1, 50)
        self.omp_sparsity_spin.setValue(8)
        self.omp_sparsity_spin.setToolTip(
            "Maximum atoms per patch (lower = more denoising, higher = preserve detail)"
        )
        sparsity_layout.addWidget(self.omp_sparsity_spin)
        layout.addLayout(sparsity_layout)

        # Residual tolerance
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Residual tol:"))
        self.omp_tol_spin = QDoubleSpinBox()
        self.omp_tol_spin.setRange(0.01, 0.5)
        self.omp_tol_spin.setValue(0.1)
        self.omp_tol_spin.setSingleStep(0.01)
        self.omp_tol_spin.setDecimals(2)
        self.omp_tol_spin.setToolTip(
            "Stop OMP when residual < tolerance × signal norm\n"
            "Lower = more atoms used, Higher = earlier stopping"
        )
        tol_layout.addWidget(self.omp_tol_spin)
        layout.addLayout(tol_layout)

        # Overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Patch overlap:"))
        self.omp_overlap_spin = QDoubleSpinBox()
        self.omp_overlap_spin.setRange(0.0, 0.9)
        self.omp_overlap_spin.setValue(0.5)
        self.omp_overlap_spin.setSingleStep(0.1)
        self.omp_overlap_spin.setDecimals(1)
        self.omp_overlap_spin.setToolTip("Overlap between adjacent patches (0-0.9)")
        overlap_layout.addWidget(self.omp_overlap_spin)
        layout.addLayout(overlap_layout)

        # Spatial aperture
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Spatial aperture:"))
        self.omp_aperture_spin = QSpinBox()
        self.omp_aperture_spin.setRange(1, 15)
        self.omp_aperture_spin.setValue(1)
        self.omp_aperture_spin.setSingleStep(2)
        self.omp_aperture_spin.setToolTip(
            "Number of neighboring traces for joint processing\n"
            "1 = single trace, 3+ = use neighbors (must be odd)"
        )
        aperture_layout.addWidget(self.omp_aperture_spin)
        layout.addLayout(aperture_layout)

        # Processing mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.omp_mode_combo = QComboBox()
        self.omp_mode_combo.addItems(["Patch", "Spatial", "Adaptive"])
        self.omp_mode_combo.setCurrentIndex(2)  # Default to Adaptive
        self.omp_mode_combo.setToolTip(
            "Patch: Process traces independently (fastest)\n"
            "Spatial: Joint sparse coding across aperture\n"
            "Adaptive: Time-varying sparsity based on local SNR (recommended)"
        )
        mode_layout.addWidget(self.omp_mode_combo)
        layout.addLayout(mode_layout)

        # Noise estimation method
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise estimation:"))
        self.omp_noise_combo = QComboBox()
        self.omp_noise_combo.addItems([
            "None (fixed params)",
            "MAD-Diff (robust)",
            "MAD-Residual",
            "Wavelet"
        ])
        self.omp_noise_combo.setCurrentIndex(1)  # Default to MAD-Diff
        self.omp_noise_combo.setToolTip(
            "None: Use fixed sparsity/tolerance\n"
            "MAD-Diff: Median Absolute Deviation of trace differences (robust, recommended)\n"
            "MAD-Residual: MAD after subtracting local mean\n"
            "Wavelet: Wavelet-based noise estimation"
        )
        noise_layout.addWidget(self.omp_noise_combo)
        layout.addLayout(noise_layout)

        # Adaptive sparsity checkbox
        self.omp_adaptive_checkbox = QCheckBox("Adaptive sparsity (vary with local SNR)")
        self.omp_adaptive_checkbox.setChecked(True)
        self.omp_adaptive_checkbox.setToolTip(
            "When enabled, uses fewer atoms in noisy regions (more aggressive)\n"
            "and more atoms in clean regions (preserve detail)"
        )
        layout.addWidget(self.omp_adaptive_checkbox)

        # Performance note
        perf_label = QLabel("Uses Numba + spatial statistics for robust denoising")
        perf_label.setStyleSheet("color: #4CAF50; font-size: 9pt;")
        layout.addWidget(perf_label)

        # Apply button
        self.omp_apply_btn = QPushButton("Apply OMP Denoise")
        self.omp_apply_btn.setStyleSheet("""
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
        self.omp_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.omp_apply_btn)

        group.setLayout(layout)
        return group

    def _create_denoise3d_group(self) -> QGroupBox:
        """Create 3D Spatial Denoise parameters group."""
        group = QGroupBox("3D Spatial Denoise Parameters")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            "Build 3D volume from 2D gather,\n"
            "then apply DWT denoising with 3D spatial MAD.\n"
            "Handles multi-fold bins properly."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Volume building mode
        mode_group = QGroupBox("Volume Building Mode")
        mode_layout = QVBoxLayout()

        self.denoise3d_coord_radio = QRadioButton("Coordinate-based (CDP_X/CDP_Y + bin size)")
        self.denoise3d_header_radio = QRadioButton("Header-based (inline/xline keys)")
        self.denoise3d_coord_radio.setChecked(True)
        self.denoise3d_coord_radio.setToolTip(
            "Use CDP coordinates with user-specified bin size.\n"
            "Properly handles multiple traces per bin."
        )
        self.denoise3d_header_radio.setToolTip(
            "Legacy mode: use header keys for inline/xline.\n"
            "Last trace wins for duplicate positions."
        )

        self.denoise3d_mode_btn_group = QButtonGroup()
        self.denoise3d_mode_btn_group.addButton(self.denoise3d_coord_radio, 0)
        self.denoise3d_mode_btn_group.addButton(self.denoise3d_header_radio, 1)
        self.denoise3d_mode_btn_group.buttonClicked.connect(self._on_denoise3d_mode_changed)

        mode_layout.addWidget(self.denoise3d_coord_radio)
        mode_layout.addWidget(self.denoise3d_header_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Coordinate mode parameters
        self.denoise3d_coord_widget = QWidget()
        coord_layout = QFormLayout()
        coord_layout.setContentsMargins(0, 0, 0, 0)

        self.denoise3d_bin_x_spin = QDoubleSpinBox()
        self.denoise3d_bin_x_spin.setRange(1.0, 500.0)
        self.denoise3d_bin_x_spin.setValue(25.0)
        self.denoise3d_bin_x_spin.setSuffix(" m")
        self.denoise3d_bin_x_spin.setToolTip("Bin size in X (inline) direction")
        coord_layout.addRow("Bin Size X:", self.denoise3d_bin_x_spin)

        self.denoise3d_bin_y_spin = QDoubleSpinBox()
        self.denoise3d_bin_y_spin.setRange(1.0, 500.0)
        self.denoise3d_bin_y_spin.setValue(25.0)
        self.denoise3d_bin_y_spin.setSuffix(" m")
        self.denoise3d_bin_y_spin.setToolTip("Bin size in Y (crossline) direction")
        coord_layout.addRow("Bin Size Y:", self.denoise3d_bin_y_spin)

        self.denoise3d_coord_widget.setLayout(coord_layout)
        layout.addWidget(self.denoise3d_coord_widget)

        # Header mode parameters
        self.denoise3d_header_widget = QWidget()
        header_layout = QFormLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.denoise3d_inline_combo = QComboBox()
        self.denoise3d_inline_combo.setToolTip(
            "Header for inline (X) axis.\n"
            "For shot gather: field_record, sin, source_x\n"
            "For CDP: CDP, inline"
        )
        self.denoise3d_inline_combo.addItems(["(load dataset first)"])
        header_layout.addRow("Inline Key:", self.denoise3d_inline_combo)

        self.denoise3d_xline_combo = QComboBox()
        self.denoise3d_xline_combo.setToolTip(
            "Header for crossline (Y) axis.\n"
            "For shot gather: trace_number, rec_sloc, receiver_x\n"
            "For CDP: crossline"
        )
        self.denoise3d_xline_combo.addItems(["(load dataset first)"])
        header_layout.addRow("Crossline Key:", self.denoise3d_xline_combo)

        self.denoise3d_header_widget.setLayout(header_layout)
        self.denoise3d_header_widget.hide()  # Hidden by default (coord mode)
        layout.addWidget(self.denoise3d_header_widget)

        # Status label
        self.denoise3d_status_label = QLabel("")
        self.denoise3d_status_label.setStyleSheet("color: #888; font-size: 10px;")
        self.denoise3d_status_label.setWordWrap(True)
        layout.addWidget(self.denoise3d_status_label)

        # Connect combo changes to update status
        self.denoise3d_inline_combo.currentIndexChanged.connect(self._update_denoise3d_status)
        self.denoise3d_xline_combo.currentIndexChanged.connect(self._update_denoise3d_status)

        # Multi-fold reconstruction method
        recon_group = QGroupBox("Multi-Fold Reconstruction")
        recon_layout = QFormLayout()

        self.denoise3d_recon_combo = QComboBox()
        self.denoise3d_recon_combo.addItems([
            "noise_subtract (fast)",
            "residual_preserve (medium)",
            "multi_pass (accurate)"
        ])
        self.denoise3d_recon_combo.setCurrentIndex(0)
        self.denoise3d_recon_combo.setToolTip(
            "How to handle multiple traces per bin:\n"
            "- noise_subtract: Fast, compute common noise model\n"
            "- residual_preserve: Store per-trace residuals\n"
            "- multi_pass: Filter each trace individually (N passes)"
        )
        recon_layout.addRow("Method:", self.denoise3d_recon_combo)

        recon_group.setLayout(recon_layout)
        layout.addWidget(recon_group)

        # Aperture controls
        aperture_group = QGroupBox("3D Spatial Aperture")
        aperture_layout = QFormLayout()

        self.denoise3d_ap_inline_spin = QSpinBox()
        self.denoise3d_ap_inline_spin.setRange(1, 15)
        self.denoise3d_ap_inline_spin.setSingleStep(2)
        self.denoise3d_ap_inline_spin.setValue(3)
        self.denoise3d_ap_inline_spin.setToolTip("Aperture in inline direction (odd)")
        aperture_layout.addRow("Inline Aperture:", self.denoise3d_ap_inline_spin)

        self.denoise3d_ap_xline_spin = QSpinBox()
        self.denoise3d_ap_xline_spin.setRange(1, 15)
        self.denoise3d_ap_xline_spin.setSingleStep(2)
        self.denoise3d_ap_xline_spin.setValue(3)
        self.denoise3d_ap_xline_spin.setToolTip("Aperture in crossline direction (odd)")
        aperture_layout.addRow("Crossline Aperture:", self.denoise3d_ap_xline_spin)

        aperture_group.setLayout(aperture_layout)
        layout.addWidget(aperture_group)

        # DWT parameters
        dwt_group = QGroupBox("DWT Parameters")
        dwt_layout = QFormLayout()

        self.denoise3d_wavelet_combo = QComboBox()
        self.denoise3d_wavelet_combo.addItems([
            "db4", "db6", "db8", "sym4", "sym6", "coif2", "bior2.2"
        ])
        self.denoise3d_wavelet_combo.setCurrentIndex(0)
        self.denoise3d_wavelet_combo.setToolTip("Wavelet for DWT decomposition")
        dwt_layout.addRow("Wavelet:", self.denoise3d_wavelet_combo)

        self.denoise3d_k_spin = QDoubleSpinBox()
        self.denoise3d_k_spin.setRange(1.0, 10.0)
        self.denoise3d_k_spin.setSingleStep(0.5)
        self.denoise3d_k_spin.setValue(3.0)
        self.denoise3d_k_spin.setToolTip("MAD threshold multiplier (k)")
        dwt_layout.addRow("Threshold k:", self.denoise3d_k_spin)

        self.denoise3d_mode_combo = QComboBox()
        self.denoise3d_mode_combo.addItems(["soft", "hard"])
        self.denoise3d_mode_combo.setCurrentIndex(0)
        self.denoise3d_mode_combo.setToolTip("Thresholding mode")
        dwt_layout.addRow("Threshold Mode:", self.denoise3d_mode_combo)

        dwt_group.setLayout(dwt_layout)
        layout.addWidget(dwt_group)

        # Apply button
        self.denoise3d_apply_btn = QPushButton("Apply 3D Denoise")
        self.denoise3d_apply_btn.setStyleSheet("""
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
        """)
        self.denoise3d_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.denoise3d_apply_btn)

        group.setLayout(layout)
        return group

    def _on_denoise3d_mode_changed(self, button):
        """Handle 3D denoise volume mode change."""
        use_coordinates = self.denoise3d_coord_radio.isChecked()
        self.denoise3d_coord_widget.setVisible(use_coordinates)
        self.denoise3d_header_widget.setVisible(not use_coordinates)

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
        self.fkk_mode_apply = QRadioButton("Apply (direct parameters)")
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

        # Apply mode controls - all parameters directly in layout (no scroll)
        self.fkk_apply_widget = QWidget()
        apply_layout = QVBoxLayout()
        apply_layout.setContentsMargins(0, 0, 0, 0)
        apply_layout.setSpacing(4)

        # === Preset Selection ===
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.fkk_preset_combo = QComboBox()
        self.fkk_preset_combo.addItem("Custom", None)
        for name in FKK_PRESETS.keys():
            self.fkk_preset_combo.addItem(name, name)
        self.fkk_preset_combo.currentIndexChanged.connect(self._on_fkk_preset_changed)
        preset_layout.addWidget(self.fkk_preset_combo)
        apply_layout.addLayout(preset_layout)

        # === Core Velocity Parameters ===
        apply_layout.addWidget(QLabel("<b>Velocity Cone</b>"))

        # V_min / V_max
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("V:"))
        self.fkk_vmin_spin = QDoubleSpinBox()
        self.fkk_vmin_spin.setRange(50, 10000)
        self.fkk_vmin_spin.setValue(200)
        self.fkk_vmin_spin.setSuffix(" m/s")
        vel_layout.addWidget(self.fkk_vmin_spin)
        vel_layout.addWidget(QLabel("-"))
        self.fkk_vmax_spin = QDoubleSpinBox()
        self.fkk_vmax_spin.setRange(100, 20000)
        self.fkk_vmax_spin.setValue(1500)
        self.fkk_vmax_spin.setSuffix(" m/s")
        vel_layout.addWidget(self.fkk_vmax_spin)
        apply_layout.addLayout(vel_layout)

        # Azimuth range
        az_layout = QHBoxLayout()
        az_layout.addWidget(QLabel("Az:"))
        self.fkk_azmin_spin = QDoubleSpinBox()
        self.fkk_azmin_spin.setRange(0, 360)
        self.fkk_azmin_spin.setValue(0)
        self.fkk_azmin_spin.setSuffix("°")
        az_layout.addWidget(self.fkk_azmin_spin)
        az_layout.addWidget(QLabel("-"))
        self.fkk_azmax_spin = QDoubleSpinBox()
        self.fkk_azmax_spin.setRange(0, 360)
        self.fkk_azmax_spin.setValue(360)
        self.fkk_azmax_spin.setSuffix("°")
        az_layout.addWidget(self.fkk_azmax_spin)
        apply_layout.addLayout(az_layout)

        # Taper width and Mode
        taper_mode_layout = QHBoxLayout()
        taper_mode_layout.addWidget(QLabel("Taper:"))
        self.fkk_taper_spin = QDoubleSpinBox()
        self.fkk_taper_spin.setRange(0.01, 0.5)
        self.fkk_taper_spin.setSingleStep(0.05)
        self.fkk_taper_spin.setValue(0.1)
        taper_mode_layout.addWidget(self.fkk_taper_spin)
        taper_mode_layout.addWidget(QLabel("Mode:"))
        self.fkk_filter_mode_combo = QComboBox()
        self.fkk_filter_mode_combo.addItem("Reject", "reject")
        self.fkk_filter_mode_combo.addItem("Pass", "pass")
        taper_mode_layout.addWidget(self.fkk_filter_mode_combo)
        apply_layout.addLayout(taper_mode_layout)

        # === Frequency Band ===
        apply_layout.addWidget(QLabel("<b>Frequency Band</b>"))
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("F:"))
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

        # === AGC Controls ===
        apply_layout.addWidget(QLabel("<b>AGC</b>"))
        agc_layout = QHBoxLayout()
        self.fkk_agc_checkbox = QCheckBox("Enable")
        self.fkk_agc_checkbox.setToolTip("Apply AGC before filtering, remove after")
        agc_layout.addWidget(self.fkk_agc_checkbox)
        agc_layout.addWidget(QLabel("Win:"))
        self.fkk_agc_window_spin = QDoubleSpinBox()
        self.fkk_agc_window_spin.setRange(50, 5000)
        self.fkk_agc_window_spin.setValue(500)
        self.fkk_agc_window_spin.setSuffix(" ms")
        agc_layout.addWidget(self.fkk_agc_window_spin)
        agc_layout.addWidget(QLabel("Gain:"))
        self.fkk_agc_gain_spin = QDoubleSpinBox()
        self.fkk_agc_gain_spin.setRange(1, 100)
        self.fkk_agc_gain_spin.setValue(10)
        agc_layout.addWidget(self.fkk_agc_gain_spin)
        apply_layout.addLayout(agc_layout)

        # === Temporal Taper ===
        apply_layout.addWidget(QLabel("<b>Temporal Taper</b>"))
        temp_taper_layout = QHBoxLayout()
        temp_taper_layout.addWidget(QLabel("Top:"))
        self.fkk_taper_top_spin = QDoubleSpinBox()
        self.fkk_taper_top_spin.setRange(0, 1000)
        self.fkk_taper_top_spin.setValue(0)
        self.fkk_taper_top_spin.setSuffix(" ms")
        temp_taper_layout.addWidget(self.fkk_taper_top_spin)
        temp_taper_layout.addWidget(QLabel("Bot:"))
        self.fkk_taper_bottom_spin = QDoubleSpinBox()
        self.fkk_taper_bottom_spin.setRange(0, 1000)
        self.fkk_taper_bottom_spin.setValue(0)
        self.fkk_taper_bottom_spin.setSuffix(" ms")
        temp_taper_layout.addWidget(self.fkk_taper_bottom_spin)
        apply_layout.addLayout(temp_taper_layout)

        # === Temporal Pad-Copy ===
        apply_layout.addWidget(QLabel("<b>Temporal Pad-Copy</b>"))
        temp_pad_info = QLabel("(reduces top edge artifacts from first breaks)")
        temp_pad_info.setStyleSheet("color: #666; font-size: 9pt;")
        apply_layout.addWidget(temp_pad_info)

        temp_pad_layout = QHBoxLayout()
        temp_pad_layout.addWidget(QLabel("Top:"))
        self.fkk_pad_time_top_spin = QDoubleSpinBox()
        self.fkk_pad_time_top_spin.setRange(0, 500)
        self.fkk_pad_time_top_spin.setValue(0)
        self.fkk_pad_time_top_spin.setSuffix(" ms")
        self.fkk_pad_time_top_spin.setToolTip("Time to pad at top (0=disabled). Try 50-200ms.")
        temp_pad_layout.addWidget(self.fkk_pad_time_top_spin)
        temp_pad_layout.addWidget(QLabel("Bot:"))
        self.fkk_pad_time_bottom_spin = QDoubleSpinBox()
        self.fkk_pad_time_bottom_spin.setRange(0, 500)
        self.fkk_pad_time_bottom_spin.setValue(0)
        self.fkk_pad_time_bottom_spin.setSuffix(" ms")
        self.fkk_pad_time_bottom_spin.setToolTip("Time to pad at bottom (0=disabled)")
        temp_pad_layout.addWidget(self.fkk_pad_time_bottom_spin)
        apply_layout.addLayout(temp_pad_layout)

        # === Spatial Edge Handling ===
        apply_layout.addWidget(QLabel("<b>Edge Handling</b>"))

        # Edge method
        edge_method_layout = QHBoxLayout()
        edge_method_layout.addWidget(QLabel("Method:"))
        self.fkk_edge_method_combo = QComboBox()
        self.fkk_edge_method_combo.addItem("None", "none")
        self.fkk_edge_method_combo.addItem("Pad Copy", "pad_copy")
        self.fkk_edge_method_combo.setCurrentIndex(1)  # Default to pad_copy
        self.fkk_edge_method_combo.setToolTip(
            "None: No edge treatment (may have artifacts)\n"
            "Pad Copy: Pad with copies of edge traces, taper only padded zone"
        )
        edge_method_layout.addWidget(self.fkk_edge_method_combo)
        apply_layout.addLayout(edge_method_layout)

        # Pad traces label with guidance
        pad_info = QLabel("Edge Pad (increase if artifacts):")
        pad_info.setToolTip(
            "Number of traces to pad on each side.\n"
            "Larger values = better artifact suppression.\n"
            "Try 20-50 if edge artifacts persist."
        )
        apply_layout.addWidget(pad_info)

        # Pad traces X/Y with larger range
        pad_traces_layout = QHBoxLayout()
        pad_traces_layout.addWidget(QLabel("X:"))
        self.fkk_pad_traces_x_spin = QSpinBox()
        self.fkk_pad_traces_x_spin.setRange(0, 100)
        self.fkk_pad_traces_x_spin.setValue(0)
        self.fkk_pad_traces_x_spin.setSpecialValueText("Auto")
        self.fkk_pad_traces_x_spin.setToolTip(
            "Traces to pad in X direction.\n"
            "0 = Auto (~10%, min 3, max 20)\n"
            "Try 20-50 if artifacts persist."
        )
        pad_traces_layout.addWidget(self.fkk_pad_traces_x_spin)
        pad_traces_layout.addWidget(QLabel("Y:"))
        self.fkk_pad_traces_y_spin = QSpinBox()
        self.fkk_pad_traces_y_spin.setRange(0, 100)
        self.fkk_pad_traces_y_spin.setValue(0)
        self.fkk_pad_traces_y_spin.setSpecialValueText("Auto")
        self.fkk_pad_traces_y_spin.setToolTip(
            "Traces to pad in Y direction.\n"
            "0 = Auto (~10%, min 3, max 20)\n"
            "Try 20-50 if artifacts persist."
        )
        pad_traces_layout.addWidget(self.fkk_pad_traces_y_spin)
        apply_layout.addLayout(pad_traces_layout)

        # Padding factor (FFT padding)
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("FFT Pad:"))
        self.fkk_padding_factor_spin = QDoubleSpinBox()
        self.fkk_padding_factor_spin.setRange(1.0, 4.0)
        self.fkk_padding_factor_spin.setSingleStep(0.5)
        self.fkk_padding_factor_spin.setValue(1.0)
        self.fkk_padding_factor_spin.setSuffix("x")
        self.fkk_padding_factor_spin.setToolTip("Extra FFT padding multiplier")
        padding_layout.addWidget(self.fkk_padding_factor_spin)
        apply_layout.addLayout(padding_layout)

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
            # Core velocity parameters
            self.fkk_vmin_spin.setValue(preset.v_min)
            self.fkk_vmax_spin.setValue(preset.v_max)
            self.fkk_azmin_spin.setValue(preset.azimuth_min)
            self.fkk_azmax_spin.setValue(preset.azimuth_max)
            self.fkk_taper_spin.setValue(preset.taper_width)
            mode_idx = 0 if preset.mode == 'reject' else 1
            self.fkk_filter_mode_combo.setCurrentIndex(mode_idx)
            # AGC
            self.fkk_agc_checkbox.setChecked(preset.apply_agc)
            self.fkk_agc_window_spin.setValue(preset.agc_window_ms)
            self.fkk_agc_gain_spin.setValue(preset.agc_max_gain)
            # Frequency band
            self.fkk_fmin_spin.setValue(preset.f_min if preset.f_min is not None else 0)
            self.fkk_fmax_spin.setValue(preset.f_max if preset.f_max is not None else self.nyquist_freq)
            # Temporal taper
            self.fkk_taper_top_spin.setValue(preset.taper_ms_top)
            self.fkk_taper_bottom_spin.setValue(preset.taper_ms_bottom)
            # Temporal pad-copy
            self.fkk_pad_time_top_spin.setValue(preset.pad_time_top_ms)
            self.fkk_pad_time_bottom_spin.setValue(preset.pad_time_bottom_ms)
            # Edge handling
            edge_idx = self.fkk_edge_method_combo.findData(preset.edge_method)
            if edge_idx >= 0:
                self.fkk_edge_method_combo.setCurrentIndex(edge_idx)
            self.fkk_pad_traces_x_spin.setValue(preset.pad_traces_x)
            self.fkk_pad_traces_y_spin.setValue(preset.pad_traces_y)
            self.fkk_padding_factor_spin.setValue(preset.padding_factor)

    def _on_fkk_apply_clicked(self):
        """Handle FKK apply button click."""
        f_min = self.fkk_fmin_spin.value()
        f_max = self.fkk_fmax_spin.value()

        config = FKKConfig(
            # Core velocity parameters
            v_min=self.fkk_vmin_spin.value(),
            v_max=self.fkk_vmax_spin.value(),
            azimuth_min=self.fkk_azmin_spin.value(),
            azimuth_max=self.fkk_azmax_spin.value(),
            taper_width=self.fkk_taper_spin.value(),
            mode=self.fkk_filter_mode_combo.currentData(),
            # AGC
            apply_agc=self.fkk_agc_checkbox.isChecked(),
            agc_window_ms=self.fkk_agc_window_spin.value(),
            agc_max_gain=self.fkk_agc_gain_spin.value(),
            # Frequency band
            f_min=f_min if f_min > 0 else None,
            f_max=f_max if f_max < self.nyquist_freq else None,
            # Temporal taper
            taper_ms_top=self.fkk_taper_top_spin.value(),
            taper_ms_bottom=self.fkk_taper_bottom_spin.value(),
            # Temporal pad-copy
            pad_time_top_ms=self.fkk_pad_time_top_spin.value(),
            pad_time_bottom_ms=self.fkk_pad_time_bottom_spin.value(),
            # Edge handling
            edge_method=self.fkk_edge_method_combo.currentData(),
            pad_traces_x=self.fkk_pad_traces_x_spin.value(),
            pad_traces_y=self.fkk_pad_traces_y_spin.value(),
            padding_factor=self.fkk_padding_factor_spin.value()
        )
        self.fkk_apply_requested.emit(config)

    def set_fkk_config(self, config: FKKConfig):
        """Set FKK parameters from config (e.g., from designer)."""
        # Core velocity parameters
        self.fkk_vmin_spin.setValue(config.v_min)
        self.fkk_vmax_spin.setValue(config.v_max)
        self.fkk_azmin_spin.setValue(config.azimuth_min)
        self.fkk_azmax_spin.setValue(config.azimuth_max)
        self.fkk_taper_spin.setValue(config.taper_width)
        mode_idx = 0 if config.mode == 'reject' else 1
        self.fkk_filter_mode_combo.setCurrentIndex(mode_idx)
        # AGC
        self.fkk_agc_checkbox.setChecked(config.apply_agc)
        self.fkk_agc_window_spin.setValue(config.agc_window_ms)
        self.fkk_agc_gain_spin.setValue(config.agc_max_gain)
        # Frequency band
        self.fkk_fmin_spin.setValue(config.f_min if config.f_min is not None else 0)
        self.fkk_fmax_spin.setValue(config.f_max if config.f_max is not None else self.nyquist_freq)
        # Temporal taper
        self.fkk_taper_top_spin.setValue(config.taper_ms_top)
        self.fkk_taper_bottom_spin.setValue(config.taper_ms_bottom)
        # Temporal pad-copy
        self.fkk_pad_time_top_spin.setValue(config.pad_time_top_ms)
        self.fkk_pad_time_bottom_spin.setValue(config.pad_time_bottom_ms)
        # Edge handling
        edge_idx = self.fkk_edge_method_combo.findData(config.edge_method)
        if edge_idx >= 0:
            self.fkk_edge_method_combo.setCurrentIndex(edge_idx)
        self.fkk_pad_traces_x_spin.setValue(config.pad_traces_x)
        self.fkk_pad_traces_y_spin.setValue(config.pad_traces_y)
        self.fkk_padding_factor_spin.setValue(config.padding_factor)
        # Set preset to Custom
        self.fkk_preset_combo.setCurrentIndex(0)

    def _create_pstm_group(self) -> QGroupBox:
        """Create Kirchhoff PSTM parameters group."""
        group = QGroupBox("Kirchhoff PSTM Parameters")
        layout = QVBoxLayout()

        # Velocity
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("Velocity (m/s):"))
        self.pstm_velocity_spin = QDoubleSpinBox()
        self.pstm_velocity_spin.setRange(500, 8000)
        self.pstm_velocity_spin.setValue(2500)
        self.pstm_velocity_spin.setSingleStep(100)
        self.pstm_velocity_spin.setDecimals(0)
        self.pstm_velocity_spin.setToolTip("Constant migration velocity in m/s")
        vel_layout.addWidget(self.pstm_velocity_spin)
        layout.addLayout(vel_layout)

        # Aperture
        aperture_layout = QHBoxLayout()
        aperture_layout.addWidget(QLabel("Aperture (m):"))
        self.pstm_aperture_spin = QDoubleSpinBox()
        self.pstm_aperture_spin.setRange(100, 20000)
        self.pstm_aperture_spin.setValue(3000)
        self.pstm_aperture_spin.setSingleStep(100)
        self.pstm_aperture_spin.setDecimals(0)
        self.pstm_aperture_spin.setToolTip("Maximum migration aperture in meters")
        aperture_layout.addWidget(self.pstm_aperture_spin)
        layout.addLayout(aperture_layout)

        # Max angle
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Max Angle (deg):"))
        self.pstm_angle_spin = QDoubleSpinBox()
        self.pstm_angle_spin.setRange(10, 85)
        self.pstm_angle_spin.setValue(60)
        self.pstm_angle_spin.setSingleStep(5)
        self.pstm_angle_spin.setDecimals(0)
        self.pstm_angle_spin.setToolTip("Maximum migration angle from vertical")
        angle_layout.addWidget(self.pstm_angle_spin)
        layout.addLayout(angle_layout)

        # Info label
        info_label = QLabel("Applies migration to current gather")
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(info_label)

        # Apply button
        self.pstm_apply_btn = QPushButton("Apply PSTM")
        self.pstm_apply_btn.setStyleSheet("""
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
                background-color: #1565C0;
            }
        """)
        self.pstm_apply_btn.clicked.connect(self._on_pstm_apply)
        layout.addWidget(self.pstm_apply_btn)

        # Wizard button
        self.pstm_wizard_btn = QPushButton("Open PSTM Wizard...")
        self.pstm_wizard_btn.setToolTip("Configure and run full dataset migration")
        self.pstm_wizard_btn.clicked.connect(lambda: self.pstm_wizard_requested.emit())
        layout.addWidget(self.pstm_wizard_btn)

        group.setLayout(layout)
        return group

    def _on_pstm_apply(self):
        """Handle PSTM apply button click."""
        self.pstm_apply_requested.emit(
            self.pstm_velocity_spin.value(),
            self.pstm_aperture_spin.value(),
            self.pstm_angle_spin.value()
        )

    def _create_mute_group(self) -> QGroupBox:
        """Create mute controls group for top/bottom mute based on velocity."""
        group = QGroupBox("Mute Controls")
        layout = QVBoxLayout()

        # Top mute checkbox
        self.mute_top_checkbox = QCheckBox("Top Mute")
        self.mute_top_checkbox.setToolTip("Zero samples before mute time (T = offset/velocity)")
        self.mute_top_checkbox.stateChanged.connect(self._on_mute_checkbox_changed)
        layout.addWidget(self.mute_top_checkbox)

        # Bottom mute checkbox
        self.mute_bottom_checkbox = QCheckBox("Bottom Mute")
        self.mute_bottom_checkbox.setToolTip("Zero samples after mute time (T = offset/velocity)")
        self.mute_bottom_checkbox.stateChanged.connect(self._on_mute_checkbox_changed)
        layout.addWidget(self.mute_bottom_checkbox)

        # Velocity
        vel_layout = QHBoxLayout()
        vel_layout.addWidget(QLabel("Velocity (m/s):"))
        self.mute_velocity_spin = QDoubleSpinBox()
        self.mute_velocity_spin.setRange(500, 8000)
        self.mute_velocity_spin.setValue(2500)
        self.mute_velocity_spin.setSingleStep(100)
        self.mute_velocity_spin.setDecimals(0)
        self.mute_velocity_spin.setToolTip("Mute velocity: T_mute = offset / velocity")
        vel_layout.addWidget(self.mute_velocity_spin)
        layout.addLayout(vel_layout)

        # Taper samples
        taper_layout = QHBoxLayout()
        taper_layout.addWidget(QLabel("Taper (samples):"))
        self.mute_taper_spin = QSpinBox()
        self.mute_taper_spin.setRange(0, 100)
        self.mute_taper_spin.setValue(20)
        self.mute_taper_spin.setToolTip("Number of samples for cosine taper at mute boundary")
        taper_layout.addWidget(self.mute_taper_spin)
        layout.addLayout(taper_layout)

        # Info label
        info_label = QLabel("Linear mute: T = offset / velocity")
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(info_label)

        # Apply button
        self.mute_apply_btn = QPushButton("Apply Mute")
        self.mute_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.mute_apply_btn.setEnabled(False)  # Disabled until mute type selected
        self.mute_apply_btn.clicked.connect(self._on_mute_apply)
        layout.addWidget(self.mute_apply_btn)

        # Clear mute button
        self.mute_clear_btn = QPushButton("Clear Mute")
        self.mute_clear_btn.clicked.connect(self._on_mute_clear)
        layout.addWidget(self.mute_clear_btn)

        group.setLayout(layout)
        return group

    def _on_mute_checkbox_changed(self):
        """Enable/disable apply button based on mute type selection."""
        has_mute = self.mute_top_checkbox.isChecked() or self.mute_bottom_checkbox.isChecked()
        self.mute_apply_btn.setEnabled(has_mute)

    def _on_mute_apply(self):
        """Handle mute apply button click."""
        from processors.mute_processor import MuteConfig

        top_mute = self.mute_top_checkbox.isChecked()
        bottom_mute = self.mute_bottom_checkbox.isChecked()

        if not top_mute and not bottom_mute:
            return  # Nothing to apply

        config = MuteConfig(
            velocity=self.mute_velocity_spin.value(),
            top_mute=top_mute,
            bottom_mute=bottom_mute,
            taper_samples=self.mute_taper_spin.value()
        )
        self.mute_apply_requested.emit(config)

    def _on_mute_clear(self):
        """Handle mute clear button click."""
        self.mute_top_checkbox.setChecked(False)
        self.mute_bottom_checkbox.setChecked(False)
        self.mute_apply_requested.emit(None)

    def _create_deconvolution_group(self) -> QGroupBox:
        """Create deconvolution parameters group."""
        group = QGroupBox("Deconvolution Parameters")
        layout = QVBoxLayout()

        # Mode selection (Spiking / Predictive)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.decon_mode_combo = QComboBox()
        self.decon_mode_combo.addItems(["Spiking", "Predictive"])
        self.decon_mode_combo.setToolTip(
            "Spiking: Compress wavelet to spike (whitening)\n"
            "Predictive: Attenuate multiples with known period"
        )
        self.decon_mode_combo.currentIndexChanged.connect(self._on_decon_mode_changed)
        mode_layout.addWidget(self.decon_mode_combo)
        layout.addLayout(mode_layout)

        # Top Window section header
        top_header = QLabel("─── Top Window ───")
        top_header.setStyleSheet("color: #666; font-size: 9pt; font-weight: bold;")
        layout.addWidget(top_header)

        # Time top (start time at zero offset)
        ttop_layout = QHBoxLayout()
        ttop_layout.addWidget(QLabel("T0 Top (ms):"))
        self.decon_time_top_spin = QDoubleSpinBox()
        self.decon_time_top_spin.setRange(0, 5000)
        self.decon_time_top_spin.setValue(100)
        self.decon_time_top_spin.setSingleStep(50)
        self.decon_time_top_spin.setDecimals(0)
        self.decon_time_top_spin.setToolTip("Top window start time at zero offset (ms)")
        ttop_layout.addWidget(self.decon_time_top_spin)
        layout.addLayout(ttop_layout)

        # Velocity top
        vtop_layout = QHBoxLayout()
        vtop_layout.addWidget(QLabel("V Top (m/s):"))
        self.decon_velocity_top_spin = QDoubleSpinBox()
        self.decon_velocity_top_spin.setRange(500, 10000)
        self.decon_velocity_top_spin.setValue(3500)
        self.decon_velocity_top_spin.setSingleStep(100)
        self.decon_velocity_top_spin.setDecimals(0)
        self.decon_velocity_top_spin.setToolTip("Top window moveout velocity (m/s)")
        vtop_layout.addWidget(self.decon_velocity_top_spin)
        layout.addLayout(vtop_layout)

        # Bottom Window section header
        bot_header = QLabel("─── Bottom Window ───")
        bot_header.setStyleSheet("color: #666; font-size: 9pt; font-weight: bold;")
        layout.addWidget(bot_header)

        # Time bottom (start time at zero offset)
        tbot_layout = QHBoxLayout()
        tbot_layout.addWidget(QLabel("T0 Bot (ms):"))
        self.decon_time_bottom_spin = QDoubleSpinBox()
        self.decon_time_bottom_spin.setRange(100, 10000)
        self.decon_time_bottom_spin.setValue(500)
        self.decon_time_bottom_spin.setSingleStep(100)
        self.decon_time_bottom_spin.setDecimals(0)
        self.decon_time_bottom_spin.setToolTip("Bottom window start time at zero offset (ms)")
        tbot_layout.addWidget(self.decon_time_bottom_spin)
        layout.addLayout(tbot_layout)

        # Velocity bottom
        vbot_layout = QHBoxLayout()
        vbot_layout.addWidget(QLabel("V Bot (m/s):"))
        self.decon_velocity_bottom_spin = QDoubleSpinBox()
        self.decon_velocity_bottom_spin.setRange(300, 8000)
        self.decon_velocity_bottom_spin.setValue(1500)
        self.decon_velocity_bottom_spin.setSingleStep(100)
        self.decon_velocity_bottom_spin.setDecimals(0)
        self.decon_velocity_bottom_spin.setToolTip("Bottom window moveout velocity (m/s)")
        vbot_layout.addWidget(self.decon_velocity_bottom_spin)
        layout.addLayout(vbot_layout)

        # Filter parameters header
        filter_header = QLabel("─── Filter ───")
        filter_header.setStyleSheet("color: #666; font-size: 9pt; font-weight: bold;")
        layout.addWidget(filter_header)

        # Filter length
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Length (ms):"))
        self.decon_filter_length_spin = QDoubleSpinBox()
        self.decon_filter_length_spin.setRange(20, 500)
        self.decon_filter_length_spin.setValue(160)
        self.decon_filter_length_spin.setSingleStep(20)
        self.decon_filter_length_spin.setDecimals(0)
        self.decon_filter_length_spin.setToolTip("Length of the deconvolution operator")
        filter_layout.addWidget(self.decon_filter_length_spin)
        layout.addLayout(filter_layout)

        # White noise percentage
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("White Noise (%):"))
        self.decon_white_noise_spin = QDoubleSpinBox()
        self.decon_white_noise_spin.setRange(0.01, 20)
        self.decon_white_noise_spin.setValue(1.0)
        self.decon_white_noise_spin.setSingleStep(0.5)
        self.decon_white_noise_spin.setDecimals(2)
        self.decon_white_noise_spin.setToolTip(
            "Pre-whitening percentage for stability\n"
            "Typical values: 0.1-5%"
        )
        noise_layout.addWidget(self.decon_white_noise_spin)
        layout.addLayout(noise_layout)

        # Prediction distance (for predictive mode)
        pred_layout = QHBoxLayout()
        pred_layout.addWidget(QLabel("Prediction (ms):"))
        self.decon_prediction_spin = QDoubleSpinBox()
        self.decon_prediction_spin.setRange(4, 1000)
        self.decon_prediction_spin.setValue(100)
        self.decon_prediction_spin.setSingleStep(10)
        self.decon_prediction_spin.setDecimals(0)
        self.decon_prediction_spin.setToolTip(
            "Prediction distance for predictive deconvolution\n"
            "Set to multiple period (e.g., water bottom 2-way time)"
        )
        self.decon_prediction_spin.setEnabled(False)  # Disabled for spiking mode
        pred_layout.addWidget(self.decon_prediction_spin)
        self.decon_prediction_label = pred_layout.itemAt(0).widget()
        layout.addLayout(pred_layout)

        # Info label
        info_label = QLabel("T(x) = T0 + (offset/V) × 1000")
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(info_label)

        # Apply button
        self.decon_apply_btn = QPushButton("Apply Deconvolution")
        self.decon_apply_btn.setStyleSheet("""
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
        self.decon_apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.decon_apply_btn)

        group.setLayout(layout)
        return group

    def _on_decon_mode_changed(self, index: int):
        """Handle deconvolution mode change."""
        is_predictive = index == 1
        self.decon_prediction_spin.setEnabled(is_predictive)
        # Update prediction label style to show enabled/disabled
        if hasattr(self, 'decon_prediction_label'):
            if is_predictive:
                self.decon_prediction_label.setStyleSheet("")
            else:
                self.decon_prediction_label.setStyleSheet("color: #999;")

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
        self.stockwell_group.hide()
        self.stft_group.hide()
        self.dwtdenoise_group.hide()
        self.gabor_group.hide()
        self.emd_group.hide()
        self.omp_group.hide()
        self.denoise3d_group.hide()
        self.fk_filter_group.hide()
        self.fkk_filter_group.hide()
        self.pstm_group.hide()
        self.deconvolution_group.hide()

        if index == 0:  # Bandpass Filter
            self.bandpass_group.show()
            # Disable GPU for bandpass filter
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 1:  # Stockwell Transform (ST)
            self.stockwell_group.show()
            # Enable GPU for Stockwell if available
            if self.gpu_checkbox is not None and self.gpu_available:
                self.gpu_checkbox.setEnabled(True)
        elif index == 2:  # STFT Denoise
            self.stft_group.show()
            # Enable GPU for STFT if available
            if self.gpu_checkbox is not None and self.gpu_available:
                self.gpu_checkbox.setEnabled(True)
        elif index == 3:  # DWT-Denoise
            self.dwtdenoise_group.show()
            # Disable GPU for DWT (CPU-only for now)
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 4:  # Gabor Transform
            self.gabor_group.show()
            # Disable GPU for Gabor (CPU-only for now)
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 5:  # EMD Decomposition
            self.emd_group.show()
            # Disable GPU for EMD (CPU-only)
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 6:  # OMP Sparse Denoise
            self.omp_group.show()
            # Disable GPU for OMP (CPU-only with Numba)
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 7:  # 3D Spatial Denoise
            self.denoise3d_group.show()
            # Disable GPU for 3D Denoise (CPU-only)
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 8:  # FK Filter
            self.fk_filter_group.show()
            # Disable GPU for FK filter
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)
        elif index == 9:  # 3D FKK Filter
            self.fkk_filter_group.show()
            # Enable GPU for 3D FKK filter if available
            if self.gpu_checkbox is not None and self.gpu_available:
                self.gpu_checkbox.setEnabled(True)
        elif index == 10:  # Kirchhoff PSTM
            self.pstm_group.show()
            # Enable GPU for PSTM if available
            if self.gpu_checkbox is not None and self.gpu_available:
                self.gpu_checkbox.setEnabled(True)
        elif index == 11:  # Deconvolution
            self.deconvolution_group.show()
            # Disable GPU for deconvolution (CPU-only)
            if self.gpu_checkbox is not None:
                self.gpu_checkbox.setEnabled(False)

    def _on_gpu_checkbox_changed(self, state):
        """Handle GPU checkbox state change."""
        if self.gpu_status_label is not None:
            if state == Qt.CheckState.Checked.value:
                gpu_name = self.device_manager.get_device_name()
                self.gpu_status_label.setText(f"🟢 {gpu_name} (Enabled)")
            else:
                self.gpu_status_label.setText(f"🟡 GPU Disabled (Using CPU)")

    def _on_stockwell_preset_changed(self, index: int):
        """Handle Stockwell Transform preset change."""
        # Preset configurations
        presets = {
            0: {"aperture": 11, "fmin": 5.0, "fmax": 30.0, "k": 2.5},  # Low-Freq Noise
            1: {"aperture": 15, "fmin": 5.0, "fmax": 150.0, "k": 3.5},  # White Noise
            2: {"aperture": 7, "fmin": 50.0, "fmax": 150.0, "k": 3.0},  # High-Freq Noise
            3: {"aperture": 9, "fmin": 10.0, "fmax": 100.0, "k": 2.0},  # Conservative
        }

        if index < 4:  # Not Custom
            preset = presets[index]
            self.stockwell_aperture_spin.blockSignals(True)
            self.stockwell_fmin_spin.blockSignals(True)
            self.stockwell_fmax_spin.blockSignals(True)
            self.stockwell_threshold_k_spin.blockSignals(True)

            self.stockwell_aperture_spin.setValue(preset["aperture"])
            self.stockwell_fmin_spin.setValue(preset["fmin"])
            self.stockwell_fmax_spin.setValue(preset["fmax"])
            self.stockwell_threshold_k_spin.setValue(preset["k"])

            self.stockwell_aperture_spin.blockSignals(False)
            self.stockwell_fmin_spin.blockSignals(False)
            self.stockwell_fmax_spin.blockSignals(False)
            self.stockwell_threshold_k_spin.blockSignals(False)

    def _on_stft_preset_changed(self, index: int):
        """Handle STFT Denoise preset change."""
        # Preset configurations
        presets = {
            0: {"aperture": 11, "fmin": 5.0, "fmax": 30.0, "k": 2.5, "nperseg": 64},  # Low-Freq Noise
            1: {"aperture": 15, "fmin": 5.0, "fmax": 150.0, "k": 3.5, "nperseg": 64},  # White Noise
            2: {"aperture": 7, "fmin": 50.0, "fmax": 150.0, "k": 3.0, "nperseg": 32},  # High-Freq Noise
            3: {"aperture": 9, "fmin": 10.0, "fmax": 100.0, "k": 2.0, "nperseg": 64},  # Conservative
        }

        if index < 4:  # Not Custom
            preset = presets[index]
            self.stft_aperture_spin.blockSignals(True)
            self.stft_fmin_spin.blockSignals(True)
            self.stft_fmax_spin.blockSignals(True)
            self.stft_threshold_k_spin.blockSignals(True)
            self.stft_nperseg_spin.blockSignals(True)

            self.stft_aperture_spin.setValue(preset["aperture"])
            self.stft_fmin_spin.setValue(preset["fmin"])
            self.stft_fmax_spin.setValue(preset["fmax"])
            self.stft_threshold_k_spin.setValue(preset["k"])
            self.stft_nperseg_spin.setValue(preset["nperseg"])

            self.stft_aperture_spin.blockSignals(False)
            self.stft_fmin_spin.blockSignals(False)
            self.stft_fmax_spin.blockSignals(False)
            self.stft_threshold_k_spin.blockSignals(False)
            self.stft_nperseg_spin.blockSignals(False)

    def _on_apply_clicked(self):
        """Handle apply button click."""
        try:
            algo_index = self.algorithm_combo.currentIndex()

            # Check which algorithm is selected
            if algo_index == 0:  # Bandpass Filter
                processor = BandpassFilter(
                    low_freq=self.low_freq_spin.value(),
                    high_freq=self.high_freq_spin.value(),
                    order=self.order_spin.value()
                )
                print(f"✓ Using Bandpass Filter")

            elif algo_index == 1:  # Stockwell Transform (ST)
                # Check if GPU should be used
                use_gpu = (
                    GPU_AVAILABLE and
                    self.gpu_checkbox is not None and
                    self.gpu_checkbox.isChecked()
                )

                # Get threshold mode from combo box
                threshold_mode_map = {
                    0: 'adaptive',
                    1: 'hard',
                    2: 'scaled',
                    3: 'soft'
                }
                threshold_mode = threshold_mode_map.get(
                    self.stockwell_threshold_mode_combo.currentIndex(), 'adaptive'
                )

                # Get low-amplitude protection setting
                low_amp_protection = self.stockwell_low_amp_checkbox.isChecked()

                if use_gpu:
                    # Use GPU-accelerated version (S-Transform)
                    processor = TFDenoiseGPU(
                        aperture=self.stockwell_aperture_spin.value(),
                        fmin=self.stockwell_fmin_spin.value(),
                        fmax=self.stockwell_fmax_spin.value(),
                        threshold_k=self.stockwell_threshold_k_spin.value(),
                        threshold_type='soft',
                        threshold_mode=threshold_mode,
                        transform_type='stransform',
                        use_gpu='auto',
                        low_amp_protection=low_amp_protection,
                        device_manager=self.device_manager
                    )
                    print(f"✓ Using GPU Stockwell (S-Transform): {self.device_manager.get_device_name()}")
                else:
                    # Use CPU Stockwell processor
                    processor = StockwellDenoise(
                        aperture=self.stockwell_aperture_spin.value(),
                        fmin=self.stockwell_fmin_spin.value(),
                        fmax=self.stockwell_fmax_spin.value(),
                        threshold_k=self.stockwell_threshold_k_spin.value(),
                        threshold_mode=threshold_mode,
                        low_amp_protection=low_amp_protection
                    )
                    print(f"✓ Using CPU Stockwell (S-Transform) Denoise")
                print(f"  Threshold mode: {threshold_mode}, Low-amp protection: {low_amp_protection}")

            elif algo_index == 2:  # STFT Denoise
                # Check if GPU should be used
                use_gpu = (
                    GPU_AVAILABLE and
                    self.gpu_checkbox is not None and
                    self.gpu_checkbox.isChecked()
                )

                # Get threshold mode from combo box
                threshold_mode_map = {
                    0: 'adaptive',
                    1: 'hard',
                    2: 'scaled',
                    3: 'soft'
                }
                threshold_mode = threshold_mode_map.get(
                    self.stft_threshold_mode_combo.currentIndex(), 'adaptive'
                )

                # Get low-amplitude protection setting
                low_amp_protection = self.stft_low_amp_checkbox.isChecked()

                if use_gpu:
                    # Use GPU-accelerated version (STFT)
                    processor = TFDenoiseGPU(
                        aperture=self.stft_aperture_spin.value(),
                        fmin=self.stft_fmin_spin.value(),
                        fmax=self.stft_fmax_spin.value(),
                        threshold_k=self.stft_threshold_k_spin.value(),
                        threshold_type='soft',
                        threshold_mode=threshold_mode,
                        transform_type='stft',
                        use_gpu='auto',
                        low_amp_protection=low_amp_protection,
                        device_manager=self.device_manager
                    )
                    print(f"✓ Using GPU STFT Denoise: {self.device_manager.get_device_name()}")
                else:
                    # Use CPU STFT processor
                    processor = STFTDenoise(
                        aperture=self.stft_aperture_spin.value(),
                        fmin=self.stft_fmin_spin.value(),
                        fmax=self.stft_fmax_spin.value(),
                        nperseg=self.stft_nperseg_spin.value(),
                        threshold_k=self.stft_threshold_k_spin.value(),
                        threshold_mode=threshold_mode,
                        low_amp_protection=low_amp_protection
                    )
                    print(f"✓ Using CPU STFT Denoise (window={self.stft_nperseg_spin.value()})")
                print(f"  Threshold mode: {threshold_mode}, Low-amp protection: {low_amp_protection}")

            elif algo_index == 3:  # DWT-Denoise
                # Get wavelet name from combo box text
                wavelet_map = {
                    0: 'db4',
                    1: 'db8',
                    2: 'sym4',
                    3: 'sym8',
                    4: 'coif4',
                    5: 'bior3.5'
                }
                wavelet = wavelet_map.get(self.dwt_wavelet_combo.currentIndex(), 'db4')

                # Get transform type
                transform_map = {
                    0: 'dwt',
                    1: 'swt',
                    2: 'dwt_spatial',
                    3: 'wpt',
                    4: 'wpt_spatial'
                }
                transform_type = transform_map.get(self.dwt_transform_combo.currentIndex(), 'dwt')

                # Get threshold mode
                threshold_mode = 'soft' if self.dwt_threshold_mode_combo.currentIndex() == 0 else 'hard'

                # Get best-basis setting (only for WPT modes)
                best_basis = (transform_type in ['wpt', 'wpt_spatial'] and
                             self.dwt_best_basis_checkbox.isChecked())

                processor = DWTDenoise(
                    wavelet=wavelet,
                    level=self.dwt_level_spin.value(),
                    threshold_k=self.dwt_threshold_k_spin.value(),
                    threshold_mode=threshold_mode,
                    transform_type=transform_type,
                    aperture=self.dwt_aperture_spin.value(),
                    best_basis=best_basis
                )
                print(f"✓ Using DWT-Denoise ({transform_type.upper()})")
                extra_info = ", best-basis" if best_basis else ""
                print(f"  Wavelet: {wavelet}, Level: {self.dwt_level_spin.value()}, k={self.dwt_threshold_k_spin.value()}{extra_info}")

            elif algo_index == 4:  # Gabor Transform
                # Get sigma (0 = auto/None)
                sigma_value = self.gabor_sigma_spin.value()
                sigma = sigma_value if sigma_value > 0 else None

                # Get threshold mode
                threshold_mode = 'soft' if self.gabor_threshold_mode_combo.currentIndex() == 0 else 'hard'

                processor = GaborDenoise(
                    aperture=self.gabor_aperture_spin.value(),
                    fmin=self.gabor_fmin_spin.value(),
                    fmax=self.gabor_fmax_spin.value(),
                    threshold_k=self.gabor_threshold_k_spin.value(),
                    threshold_mode=threshold_mode,
                    window_size=self.gabor_window_spin.value(),
                    sigma=sigma,
                    overlap_percent=float(self.gabor_overlap_spin.value()),
                    low_amp_protection=self.gabor_low_amp_checkbox.isChecked()
                )
                sigma_str = f"sigma={sigma:.1f}" if sigma else "sigma=auto"
                print(f"✓ Using Gabor Transform Denoise")
                print(f"  Window: {self.gabor_window_spin.value()}, {sigma_str}, Overlap: {self.gabor_overlap_spin.value()}%")
                print(f"  Freq: {self.gabor_fmin_spin.value()}-{self.gabor_fmax_spin.value()}Hz, k={self.gabor_threshold_k_spin.value()}")

            elif algo_index == 5:  # EMD Decomposition
                # Get method
                method_map = {0: 'emd', 1: 'eemd', 2: 'ceemdan'}
                method = method_map.get(self.emd_method_combo.currentIndex(), 'eemd')

                # Get removal strategy
                remove_map = {
                    0: 'first',
                    1: 'first_2',
                    2: 'first_3',
                    3: 'last',
                    4: 'last_2',
                    5: [self.emd_custom_edit.value()]  # Custom
                }
                remove_imfs = remove_map.get(self.emd_remove_combo.currentIndex(), 'first')

                processor = EMDDenoise(
                    method=method,
                    remove_imfs=remove_imfs,
                    ensemble_size=self.emd_ensemble_spin.value(),
                    noise_amplitude=self.emd_noise_spin.value()
                )
                print(f"✓ Using EMD Decomposition ({method.upper()})")
                print(f"  Remove: {remove_imfs}, Ensemble: {self.emd_ensemble_spin.value()}")

            elif algo_index == 6:  # OMP Sparse Denoise
                # Get dictionary type
                dict_map = {
                    0: 'dct',
                    1: 'dft',
                    2: 'gabor',
                    3: 'wavelet',
                    4: 'hybrid'
                }
                dictionary_type = dict_map.get(self.omp_dict_combo.currentIndex(), 'wavelet')

                # Get processing mode
                mode_map = {0: 'patch', 1: 'spatial', 2: 'adaptive'}
                denoise_mode = mode_map.get(self.omp_mode_combo.currentIndex(), 'adaptive')

                # Get noise estimation method
                noise_map = {0: 'none', 1: 'mad_diff', 2: 'mad_residual', 3: 'wavelet'}
                noise_estimation = noise_map.get(self.omp_noise_combo.currentIndex(), 'mad_diff')

                # Get adaptive sparsity setting
                adaptive_sparsity = self.omp_adaptive_checkbox.isChecked()

                # Get aperture (ensure odd)
                aperture = self.omp_aperture_spin.value()
                if aperture % 2 == 0:
                    aperture += 1  # Make it odd

                processor = OMPDenoise(
                    patch_size=self.omp_patch_spin.value(),
                    overlap=self.omp_overlap_spin.value(),
                    sparsity=self.omp_sparsity_spin.value(),
                    residual_tol=self.omp_tol_spin.value(),
                    dictionary_type=dictionary_type,
                    aperture=aperture,
                    denoise_mode=denoise_mode,
                    noise_estimation=noise_estimation,
                    adaptive_sparsity=adaptive_sparsity
                )
                print(f"✓ Using OMP Sparse Denoise ({dictionary_type.upper()} dictionary)")
                print(f"  Patch: {self.omp_patch_spin.value()}, Sparsity: {self.omp_sparsity_spin.value()} atoms")
                print(f"  Mode: {denoise_mode}, Noise: {noise_estimation}, Adaptive: {adaptive_sparsity}")
                print(f"  Aperture: {aperture}")

            elif algo_index == 7:  # 3D Spatial Denoise
                from processors.denoise_3d import Denoise3D

                # Get volume building mode
                use_coordinates = self.denoise3d_coord_radio.isChecked()

                # Get apertures (ensure odd)
                ap_inline = self.denoise3d_ap_inline_spin.value()
                if ap_inline > 1 and ap_inline % 2 == 0:
                    ap_inline += 1

                ap_xline = self.denoise3d_ap_xline_spin.value()
                if ap_xline > 1 and ap_xline % 2 == 0:
                    ap_xline += 1

                # Get DWT parameters
                wavelet = self.denoise3d_wavelet_combo.currentText()
                threshold_k = self.denoise3d_k_spin.value()
                threshold_mode = self.denoise3d_mode_combo.currentText()

                # Get reconstruction method
                recon_text = self.denoise3d_recon_combo.currentText()
                reconstruction_method = recon_text.split(' ')[0]  # Extract method name

                if use_coordinates:
                    # Coordinate-based mode
                    bin_size_x = self.denoise3d_bin_x_spin.value()
                    bin_size_y = self.denoise3d_bin_y_spin.value()

                    processor = Denoise3D(
                        use_coordinates=True,
                        bin_size_x=bin_size_x,
                        bin_size_y=bin_size_y,
                        reconstruction_method=reconstruction_method,
                        aperture_inline=ap_inline,
                        aperture_xline=ap_xline,
                        wavelet=wavelet,
                        threshold_k=threshold_k,
                        threshold_mode=threshold_mode
                    )
                    print(f"✓ Using 3D Spatial Denoise (coordinate mode)")
                    print(f"  Bin size: {bin_size_x}×{bin_size_y}m")
                    print(f"  Reconstruction: {reconstruction_method}")
                else:
                    # Header-based mode (legacy)
                    inline_key = self.denoise3d_inline_combo.currentData()
                    xline_key = self.denoise3d_xline_combo.currentData()

                    if not inline_key or not xline_key:
                        print("Error: Please select valid header keys for 3D volume")
                        return

                    if inline_key == xline_key:
                        print("Error: Inline and Crossline keys must be different")
                        return

                    processor = Denoise3D(
                        use_coordinates=False,
                        inline_key=inline_key,
                        xline_key=xline_key,
                        reconstruction_method=reconstruction_method,
                        aperture_inline=ap_inline,
                        aperture_xline=ap_xline,
                        wavelet=wavelet,
                        threshold_k=threshold_k,
                        threshold_mode=threshold_mode
                    )
                    print(f"✓ Using 3D Spatial Denoise (header mode)")
                    print(f"  Volume: {inline_key} × {xline_key}")

                print(f"  Aperture: {ap_inline}×{ap_xline}, Wavelet: {wavelet}")
                print(f"  k={threshold_k}, mode={threshold_mode}")

            elif algo_index == 11:  # Deconvolution
                # Get deconvolution mode
                mode = 'spiking' if self.decon_mode_combo.currentIndex() == 0 else 'predictive'

                # Get prediction distance for predictive mode
                prediction_distance_ms = 0.0
                if mode == 'predictive':
                    prediction_distance_ms = self.decon_prediction_spin.value()

                config = DeconConfig(
                    mode=mode,
                    time_top_ms=self.decon_time_top_spin.value(),
                    time_bottom_ms=self.decon_time_bottom_spin.value(),
                    velocity_top=self.decon_velocity_top_spin.value(),
                    velocity_bottom=self.decon_velocity_bottom_spin.value(),
                    filter_length_ms=self.decon_filter_length_spin.value(),
                    white_noise_percent=self.decon_white_noise_spin.value(),
                    prediction_distance_ms=prediction_distance_ms,
                )

                processor = DeconvolutionProcessor(config=config)
                print(f"✓ Using {mode.capitalize()} Deconvolution")
                print(f"  Top: T0={config.time_top_ms:.0f}ms, V={config.velocity_top:.0f} m/s")
                print(f"  Bot: T0={config.time_bottom_ms:.0f}ms, V={config.velocity_bottom:.0f} m/s")
                print(f"  Filter: {config.filter_length_ms:.0f}ms, White noise: {config.white_noise_percent:.1f}%")
                if mode == 'predictive':
                    print(f"  Prediction distance: {prediction_distance_ms:.0f}ms")

            else:
                # Other algorithms not handled here (FK, FKK, PSTM have their own handlers)
                return

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

    def set_available_3d_headers(self, headers_with_counts: list):
        """
        Populate the 3D denoise header dropdowns with available headers.

        Args:
            headers_with_counts: List of tuples (header_name, unique_count)
                                 sorted by relevance/unique count
        """
        # Store for status updates
        self._denoise3d_header_counts = {h: c for h, c in headers_with_counts}

        # Block signals during update
        self.denoise3d_inline_combo.blockSignals(True)
        self.denoise3d_xline_combo.blockSignals(True)

        self.denoise3d_inline_combo.clear()
        self.denoise3d_xline_combo.clear()

        if not headers_with_counts:
            self.denoise3d_inline_combo.addItem("(no headers available)")
            self.denoise3d_xline_combo.addItem("(no headers available)")
            self.denoise3d_status_label.setText("Load a dataset with headers")
            self.denoise3d_inline_combo.blockSignals(False)
            self.denoise3d_xline_combo.blockSignals(False)
            return

        # Add headers with unique count info
        for header, count in headers_with_counts:
            display = f"{header} ({count} unique)"
            self.denoise3d_inline_combo.addItem(display, header)
            self.denoise3d_xline_combo.addItem(display, header)

        # Set smart defaults for inline (source-related headers)
        inline_defaults = ['sin', 's_line', 'field_record', 'FFID', 'source_x',
                          'SourceLine', 'CDP', 'inline']
        for default in inline_defaults:
            idx = self.denoise3d_inline_combo.findData(default)
            if idx >= 0:
                self.denoise3d_inline_combo.setCurrentIndex(idx)
                break

        # Set smart defaults for crossline (receiver-related headers)
        xline_defaults = ['rec_sloc', 'trace_number', 'Channel', 'receiver_x',
                         'ReceiverStation', 'crossline', 'trace_number_cdp']
        for default in xline_defaults:
            idx = self.denoise3d_xline_combo.findData(default)
            if idx >= 0:
                self.denoise3d_xline_combo.setCurrentIndex(idx)
                break

        # If same selection, pick different for crossline
        if self.denoise3d_xline_combo.currentIndex() == self.denoise3d_inline_combo.currentIndex():
            for i in range(self.denoise3d_xline_combo.count()):
                if i != self.denoise3d_inline_combo.currentIndex():
                    self.denoise3d_xline_combo.setCurrentIndex(i)
                    break

        self.denoise3d_inline_combo.blockSignals(False)
        self.denoise3d_xline_combo.blockSignals(False)

        # Update status
        self._update_denoise3d_status()

    def _update_denoise3d_status(self):
        """Update the 3D denoise status label with volume size estimate."""
        inline_key = self.denoise3d_inline_combo.currentData()
        xline_key = self.denoise3d_xline_combo.currentData()

        if not hasattr(self, '_denoise3d_header_counts'):
            return

        counts = self._denoise3d_header_counts

        if inline_key and xline_key and inline_key in counts and xline_key in counts:
            n_inline = counts[inline_key]
            n_xline = counts[xline_key]
            grid_size = n_inline * n_xline

            if inline_key == xline_key:
                self.denoise3d_status_label.setText(
                    "⚠ Inline and Crossline must be different!"
                )
                self.denoise3d_status_label.setStyleSheet("color: red; font-size: 10px;")
            else:
                self.denoise3d_status_label.setText(
                    f"Volume grid: {n_inline} × {n_xline} = {grid_size:,} positions"
                )
                self.denoise3d_status_label.setStyleSheet("color: #888; font-size: 10px;")
        else:
            self.denoise3d_status_label.setText("")

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
