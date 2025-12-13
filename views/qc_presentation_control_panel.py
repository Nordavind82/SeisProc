"""
QC Presentation Control Panel - Band management, gain controls, and export.
"""
from dataclasses import dataclass
from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup,
    QSlider, QScrollArea, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal


@dataclass
class FrequencyBand:
    """Definition of a frequency band for filtering."""
    low_freq: float
    high_freq: float
    gain: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.low_freq:.0f}-{self.high_freq:.0f} Hz"


@dataclass
class SpectralWindow:
    """Rectangular window for spectral analysis."""
    time_start: float = 0.0
    time_end: float = 1000.0
    trace_start: int = 0
    trace_end: int = 100


class QCPresentationControlPanel(QWidget):
    """
    Control panel for QC Presentation Tool.

    Sections:
    - Band Definition Table
    - Gain Controls (per band)
    - Spectral Window Definition
    - Mode Selection (Setup/Presentation)
    - Export Controls
    """

    # Signals
    bands_changed = pyqtSignal(list)              # List[FrequencyBand]
    gain_changed = pyqtSignal(int, float)         # band_index, gain_value
    fullband_gain_changed = pyqtSignal(float)     # fullband gain
    spectral_window_changed = pyqtSignal(object)  # SpectralWindow
    mode_changed = pyqtSignal(str)                # "setup" or "presentation"
    export_requested = pyqtSignal()
    draw_roi_requested = pyqtSignal()

    def __init__(self, nyquist_freq: float = 250.0, parent=None):
        super().__init__(parent)
        self.nyquist_freq = nyquist_freq
        self.max_duration = 5000.0  # ms
        self.max_traces = 1000

        # Internal state
        self.bands: List[FrequencyBand] = []
        self.gain_sliders = {}  # band_index -> slider
        self.gain_spinboxes = {}  # band_index -> spinbox

        self._init_ui()
        self._load_default_bands()

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Create scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setSpacing(10)

        # Band definition group
        band_group = self._create_band_group()
        scroll_layout.addWidget(band_group)

        # Gain controls group
        gain_group = self._create_gain_group()
        scroll_layout.addWidget(gain_group)

        # Spectral window group
        window_group = self._create_window_group()
        scroll_layout.addWidget(window_group)

        # Mode selection group
        mode_group = self._create_mode_group()
        scroll_layout.addWidget(mode_group)

        # Export group
        export_group = self._create_export_group()
        scroll_layout.addWidget(export_group)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)

        layout.addWidget(scroll)
        self.setLayout(layout)

    def _create_band_group(self) -> QGroupBox:
        """Create band definition section."""
        group = QGroupBox("Frequency Bands")
        layout = QVBoxLayout()

        # Band table
        self.band_table = QTableWidget()
        self.band_table.setColumnCount(2)
        self.band_table.setHorizontalHeaderLabels(["Low (Hz)", "High (Hz)"])
        self.band_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.band_table.setMinimumHeight(120)
        self.band_table.setMaximumHeight(150)
        layout.addWidget(self.band_table)

        # Buttons
        btn_layout = QHBoxLayout()

        self.add_band_btn = QPushButton("Add Band")
        self.add_band_btn.clicked.connect(self._on_add_band)
        btn_layout.addWidget(self.add_band_btn)

        self.remove_band_btn = QPushButton("Remove")
        self.remove_band_btn.clicked.connect(self._on_remove_band)
        btn_layout.addWidget(self.remove_band_btn)

        layout.addLayout(btn_layout)

        # Apply changes button
        self.apply_bands_btn = QPushButton("Apply Bands")
        self.apply_bands_btn.clicked.connect(self._on_apply_bands)
        self.apply_bands_btn.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.apply_bands_btn)

        group.setLayout(layout)
        return group

    def _create_gain_group(self) -> QGroupBox:
        """Create gain controls section."""
        group = QGroupBox("Gain Controls")
        layout = QVBoxLayout()

        # Fullband gain
        fullband_layout = QHBoxLayout()
        fullband_layout.addWidget(QLabel("Full Band:"))

        self.fullband_slider = QSlider(Qt.Orientation.Horizontal)
        self.fullband_slider.setRange(1, 100)  # 0.1x to 10x
        self.fullband_slider.setValue(10)  # 1.0x
        self.fullband_slider.valueChanged.connect(self._on_fullband_slider_changed)
        fullband_layout.addWidget(self.fullband_slider)

        self.fullband_spinbox = QDoubleSpinBox()
        self.fullband_spinbox.setRange(0.1, 10.0)
        self.fullband_spinbox.setValue(1.0)
        self.fullband_spinbox.setSingleStep(0.1)
        self.fullband_spinbox.setDecimals(1)
        self.fullband_spinbox.valueChanged.connect(self._on_fullband_spinbox_changed)
        self.fullband_spinbox.setFixedWidth(60)
        fullband_layout.addWidget(self.fullband_spinbox)

        layout.addLayout(fullband_layout)

        # Container for per-band gain controls
        self.gains_container = QWidget()
        self.gains_layout = QVBoxLayout()
        self.gains_layout.setSpacing(5)
        self.gains_container.setLayout(self.gains_layout)
        layout.addWidget(self.gains_container)

        group.setLayout(layout)
        return group

    def _create_window_group(self) -> QGroupBox:
        """Create spectral window section."""
        group = QGroupBox("Spectral Window")
        layout = QVBoxLayout()

        # Time range
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time (ms):"))

        self.time_start_spin = QDoubleSpinBox()
        self.time_start_spin.setRange(0, 10000)
        self.time_start_spin.setValue(0)
        self.time_start_spin.setDecimals(0)
        self.time_start_spin.valueChanged.connect(self._on_window_changed)
        time_layout.addWidget(self.time_start_spin)

        time_layout.addWidget(QLabel("-"))

        self.time_end_spin = QDoubleSpinBox()
        self.time_end_spin.setRange(0, 10000)
        self.time_end_spin.setValue(1000)
        self.time_end_spin.setDecimals(0)
        self.time_end_spin.valueChanged.connect(self._on_window_changed)
        time_layout.addWidget(self.time_end_spin)

        layout.addLayout(time_layout)

        # Trace range
        trace_layout = QHBoxLayout()
        trace_layout.addWidget(QLabel("Traces:"))

        self.trace_start_spin = QSpinBox()
        self.trace_start_spin.setRange(0, 10000)
        self.trace_start_spin.setValue(0)
        self.trace_start_spin.valueChanged.connect(self._on_window_changed)
        trace_layout.addWidget(self.trace_start_spin)

        trace_layout.addWidget(QLabel("-"))

        self.trace_end_spin = QSpinBox()
        self.trace_end_spin.setRange(0, 10000)
        self.trace_end_spin.setValue(100)
        self.trace_end_spin.valueChanged.connect(self._on_window_changed)
        trace_layout.addWidget(self.trace_end_spin)

        layout.addLayout(trace_layout)

        # Draw button
        self.draw_roi_btn = QPushButton("Draw on Seismic")
        self.draw_roi_btn.clicked.connect(self._on_draw_roi)
        layout.addWidget(self.draw_roi_btn)

        group.setLayout(layout)
        return group

    def _create_mode_group(self) -> QGroupBox:
        """Create mode selection section."""
        group = QGroupBox("Mode")
        layout = QVBoxLayout()

        self.mode_group = QButtonGroup(self)

        self.setup_radio = QRadioButton("Setup Mode")
        self.setup_radio.setChecked(True)
        self.setup_radio.setToolTip("Adjust gains and preview bands")
        self.mode_group.addButton(self.setup_radio)
        layout.addWidget(self.setup_radio)

        self.presentation_radio = QRadioButton("Presentation Mode")
        self.presentation_radio.setToolTip("Lock gains and start presentation")
        self.mode_group.addButton(self.presentation_radio)
        layout.addWidget(self.presentation_radio)

        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        # Start presentation button
        self.start_presentation_btn = QPushButton("Start Presentation")
        self.start_presentation_btn.clicked.connect(self._start_presentation)
        self.start_presentation_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.start_presentation_btn)

        group.setLayout(layout)
        return group

    def _create_export_group(self) -> QGroupBox:
        """Create export section."""
        group = QGroupBox("Export")
        layout = QVBoxLayout()

        self.export_btn = QPushButton("To Slides (PowerPoint)")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.export_btn)

        group.setLayout(layout)
        return group

    def _load_default_bands(self):
        """Load default frequency bands."""
        default_bands = [
            (2, 4),
            (4, 8),
            (10, 20),
            (20, 40),
        ]

        # Populate table
        self.band_table.setRowCount(len(default_bands))
        for i, (low, high) in enumerate(default_bands):
            low_item = QTableWidgetItem(str(low))
            high_item = QTableWidgetItem(str(high))
            low_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            high_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.band_table.setItem(i, 0, low_item)
            self.band_table.setItem(i, 1, high_item)

        # Apply bands
        self._on_apply_bands()

    def _on_add_band(self):
        """Add new band row to table."""
        row = self.band_table.rowCount()
        self.band_table.insertRow(row)

        # Default values
        low_item = QTableWidgetItem("5")
        high_item = QTableWidgetItem("15")
        low_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        high_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.band_table.setItem(row, 0, low_item)
        self.band_table.setItem(row, 1, high_item)

    def _on_remove_band(self):
        """Remove selected band from table."""
        current_row = self.band_table.currentRow()
        if current_row >= 0:
            self.band_table.removeRow(current_row)

    def _on_apply_bands(self):
        """Apply band configuration from table."""
        self.bands = []

        for row in range(self.band_table.rowCount()):
            low_item = self.band_table.item(row, 0)
            high_item = self.band_table.item(row, 1)

            if low_item and high_item:
                try:
                    low_freq = float(low_item.text())
                    high_freq = float(high_item.text())

                    if low_freq > 0 and high_freq > low_freq and high_freq < self.nyquist_freq:
                        self.bands.append(FrequencyBand(
                            low_freq=low_freq,
                            high_freq=high_freq,
                            gain=1.0
                        ))
                except ValueError:
                    pass

        # Update gain controls
        self._update_gain_controls()

        # Emit signal
        self.bands_changed.emit(self.bands)

    def _update_gain_controls(self):
        """Update per-band gain controls."""
        # Clear existing controls
        while self.gains_layout.count():
            child = self.gains_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.gain_sliders = {}
        self.gain_spinboxes = {}

        # Create control for each band
        for i, band in enumerate(self.bands):
            row = QHBoxLayout()

            label = QLabel(f"{band.name}:")
            label.setFixedWidth(80)
            row.addWidget(label)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 100)  # 0.1x to 10x
            slider.setValue(10)  # 1.0x
            slider.valueChanged.connect(lambda v, idx=i: self._on_band_slider_changed(idx, v))
            row.addWidget(slider)
            self.gain_sliders[i] = slider

            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.1, 10.0)
            spinbox.setValue(1.0)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(1)
            spinbox.valueChanged.connect(lambda v, idx=i: self._on_band_spinbox_changed(idx, v))
            spinbox.setFixedWidth(60)
            row.addWidget(spinbox)
            self.gain_spinboxes[i] = spinbox

            row_widget = QWidget()
            row_widget.setLayout(row)
            self.gains_layout.addWidget(row_widget)

    def _on_fullband_slider_changed(self, value: int):
        """Handle fullband slider change."""
        gain = value / 10.0
        self.fullband_spinbox.blockSignals(True)
        self.fullband_spinbox.setValue(gain)
        self.fullband_spinbox.blockSignals(False)
        self.fullband_gain_changed.emit(gain)

    def _on_fullband_spinbox_changed(self, value: float):
        """Handle fullband spinbox change."""
        self.fullband_slider.blockSignals(True)
        self.fullband_slider.setValue(int(value * 10))
        self.fullband_slider.blockSignals(False)
        self.fullband_gain_changed.emit(value)

    def _on_band_slider_changed(self, band_index: int, value: int):
        """Handle band-specific slider change."""
        gain = value / 10.0

        if band_index in self.gain_spinboxes:
            self.gain_spinboxes[band_index].blockSignals(True)
            self.gain_spinboxes[band_index].setValue(gain)
            self.gain_spinboxes[band_index].blockSignals(False)

        if 0 <= band_index < len(self.bands):
            self.bands[band_index].gain = gain
            self.gain_changed.emit(band_index, gain)

    def _on_band_spinbox_changed(self, band_index: int, value: float):
        """Handle band-specific spinbox change."""
        if band_index in self.gain_sliders:
            self.gain_sliders[band_index].blockSignals(True)
            self.gain_sliders[band_index].setValue(int(value * 10))
            self.gain_sliders[band_index].blockSignals(False)

        if 0 <= band_index < len(self.bands):
            self.bands[band_index].gain = value
            self.gain_changed.emit(band_index, value)

    def _on_window_changed(self):
        """Handle spectral window change."""
        window = SpectralWindow(
            time_start=self.time_start_spin.value(),
            time_end=self.time_end_spin.value(),
            trace_start=self.trace_start_spin.value(),
            trace_end=self.trace_end_spin.value()
        )
        self.spectral_window_changed.emit(window)

    def _on_draw_roi(self):
        """Request to draw ROI on seismic."""
        self.draw_roi_requested.emit()

    def _on_mode_changed(self, button):
        """Handle mode change."""
        if self.setup_radio.isChecked():
            self._enable_controls(True)
            self.mode_changed.emit("setup")
        else:
            self._enable_controls(False)
            self.mode_changed.emit("presentation")

    def _start_presentation(self):
        """Start presentation mode."""
        self.presentation_radio.setChecked(True)
        self._enable_controls(False)
        self.mode_changed.emit("presentation")

    def _enable_controls(self, enabled: bool):
        """Enable or disable gain controls."""
        self.fullband_slider.setEnabled(enabled)
        self.fullband_spinbox.setEnabled(enabled)

        for slider in self.gain_sliders.values():
            slider.setEnabled(enabled)

        for spinbox in self.gain_spinboxes.values():
            spinbox.setEnabled(enabled)

        self.band_table.setEnabled(enabled)
        self.add_band_btn.setEnabled(enabled)
        self.remove_band_btn.setEnabled(enabled)
        self.apply_bands_btn.setEnabled(enabled)

    def _on_export(self):
        """Request export to PowerPoint."""
        self.export_requested.emit()

    def set_spectral_window(self, time_start: float, time_end: float,
                            trace_start: int, trace_end: int):
        """Set spectral window values (called from ROI)."""
        self.time_start_spin.blockSignals(True)
        self.time_end_spin.blockSignals(True)
        self.trace_start_spin.blockSignals(True)
        self.trace_end_spin.blockSignals(True)

        self.time_start_spin.setValue(time_start)
        self.time_end_spin.setValue(time_end)
        self.trace_start_spin.setValue(trace_start)
        self.trace_end_spin.setValue(trace_end)

        self.time_start_spin.blockSignals(False)
        self.time_end_spin.blockSignals(False)
        self.trace_start_spin.blockSignals(False)
        self.trace_end_spin.blockSignals(False)

    def set_data_info(self, nyquist_freq: float, duration: float, n_traces: int):
        """Update control ranges based on data."""
        self.nyquist_freq = nyquist_freq
        self.max_duration = duration
        self.max_traces = n_traces

        # Update spinbox ranges
        self.time_end_spin.setMaximum(duration)
        self.time_end_spin.setValue(duration)

        self.trace_end_spin.setMaximum(n_traces - 1)
        self.trace_end_spin.setValue(n_traces - 1)

    def get_bands(self) -> List[FrequencyBand]:
        """Get current band configuration."""
        return self.bands.copy()

    def get_spectral_window(self) -> SpectralWindow:
        """Get current spectral window."""
        return SpectralWindow(
            time_start=self.time_start_spin.value(),
            time_end=self.time_end_spin.value(),
            trace_start=self.trace_start_spin.value(),
            trace_end=self.trace_end_spin.value()
        )
