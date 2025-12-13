"""
Interactive Spectral Analysis (ISA) window.
Displays seismic data with interactive spectrum viewing.
"""
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QSpinBox, QPushButton, QGroupBox,
                              QComboBox, QCheckBox, QSplitter, QDoubleSpinBox, QRadioButton)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from models.seismic_data import SeismicData
from models.viewport_state import ViewportState
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from processors.spectral_analyzer import SpectralAnalyzer


class ISAWindow(QMainWindow):
    """
    Interactive Spectral Analysis window.

    Layout:
    - Top: Seismic data viewer (clickable to select traces)
    - Bottom: Amplitude/Phase spectrum display
    - Right: Control panel for spectrum settings

    Features:
    - Amplitude and phase spectrum display
    - Compare input vs processed data spectra
    - Single trace or average spectrum
    - Time window selection
    """

    def __init__(self, data: SeismicData, viewport_state: ViewportState = None,
                 processed_data: SeismicData = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Spectral Analysis (ISA)")
        self.setGeometry(150, 150, 1400, 900)

        self.data = data
        self.processed_data = processed_data  # Optional processed data for comparison
        self.spectral_analyzer = SpectralAnalyzer(data.sample_rate)

        # Use shared viewport state if provided, otherwise create new one
        self.viewport_state = viewport_state if viewport_state is not None else ViewportState()

        # Current state
        self.current_trace_idx = 0
        self.show_average = False

        # Trace range for spectral estimation
        self.use_trace_range = False
        self.trace_range_start = 0
        self.trace_range_end = data.n_traces - 1

        # Frequency range for display
        self.freq_min = 0
        self.freq_max = data.nyquist_freq

        # Spectrum display mode
        self.spectrum_y_scale = 'db'  # 'db' (dB scale) or 'linear' (linear amplitude)
        self.spectrum_x_scale = 'linear'  # 'linear' or 'log' (for frequency axis)
        self.show_phase = False  # Whether to show phase spectrum
        self.compare_spectra = False  # Whether to compare input vs processed

        # Time window for spectrum analysis
        self.use_time_window = False
        self.time_window_start = 0.0  # ms
        self.time_window_end = data.duration  # ms

        self._init_ui()
        self._update_spectrum()

    def _init_ui(self):
        """Initialize user interface."""
        central = QWidget()
        main_layout = QHBoxLayout()

        # Left: Seismic viewer and spectrum display
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter for seismic viewer and spectrum
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Seismic data viewer (uses shared viewport_state)
        self.seismic_viewer = SeismicViewerPyQtGraph("Seismic Data (Click trace to view spectrum)", self.viewport_state)
        self.seismic_viewer.set_data(self.data)

        # Connect click event to trace selection
        self.seismic_viewer.trace_clicked.connect(self._on_trace_clicked)

        # Connect viewport state changes for synchronization with main window
        self.viewport_state.amplitude_range_changed.connect(self._on_viewport_amplitude_changed)
        self.viewport_state.colormap_changed.connect(self._on_viewport_colormap_changed)

        splitter.addWidget(self.seismic_viewer)

        # Spectrum display
        spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout()
        spectrum_layout.setContentsMargins(5, 5, 5, 5)

        # Create matplotlib figure for spectrum (no toolbar)
        self.spectrum_figure = Figure(figsize=(10, 4), facecolor='white')
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)

        # No matplotlib toolbar - removed as requested
        spectrum_layout.addWidget(self.spectrum_canvas)

        spectrum_widget.setLayout(spectrum_layout)
        splitter.addWidget(spectrum_widget)

        # Set initial sizes (seismic viewer 60%, spectrum 40%)
        splitter.setSizes([600, 400])

        left_layout.addWidget(splitter)
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel, stretch=1)

        # Right: Control panel
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Initialize viewport to show all data
        self.viewport_state.reset_to_data(self.data.duration, self.data.n_traces - 1)

    def _create_control_panel(self) -> QWidget:
        """Create control panel for spectrum settings."""
        panel = QWidget()
        panel.setMaximumWidth(320)
        layout = QVBoxLayout()

        # Trace selection group
        trace_group = QGroupBox("Trace Selection")
        trace_layout = QVBoxLayout()

        # Trace number selector
        trace_selector_layout = QHBoxLayout()
        trace_selector_layout.addWidget(QLabel("Trace:"))
        self.trace_spinbox = QSpinBox()
        self.trace_spinbox.setMinimum(0)
        self.trace_spinbox.setMaximum(self.data.n_traces - 1)
        self.trace_spinbox.setValue(0)
        self.trace_spinbox.valueChanged.connect(self._on_trace_changed)
        trace_selector_layout.addWidget(self.trace_spinbox)
        trace_layout.addLayout(trace_selector_layout)

        # Average spectrum checkbox
        self.average_checkbox = QCheckBox("Show average spectrum (all traces)")
        self.average_checkbox.stateChanged.connect(self._on_average_changed)
        trace_layout.addWidget(self.average_checkbox)

        # Trace range selection
        self.trace_range_checkbox = QCheckBox("Use trace range for estimation")
        self.trace_range_checkbox.setToolTip("Compute spectrum using a range of traces")
        self.trace_range_checkbox.stateChanged.connect(self._on_trace_range_changed)
        trace_layout.addWidget(self.trace_range_checkbox)

        # Start trace for range
        trace_start_layout = QHBoxLayout()
        trace_start_layout.addWidget(QLabel("Start Trace:"))
        self.trace_start_spinbox = QSpinBox()
        self.trace_start_spinbox.setMinimum(0)
        self.trace_start_spinbox.setMaximum(self.data.n_traces - 1)
        self.trace_start_spinbox.setValue(0)
        self.trace_start_spinbox.valueChanged.connect(self._on_trace_range_values_changed)
        self.trace_start_spinbox.setEnabled(False)
        trace_start_layout.addWidget(self.trace_start_spinbox)
        trace_layout.addLayout(trace_start_layout)

        # End trace for range
        trace_end_layout = QHBoxLayout()
        trace_end_layout.addWidget(QLabel("End Trace:"))
        self.trace_end_spinbox = QSpinBox()
        self.trace_end_spinbox.setMinimum(0)
        self.trace_end_spinbox.setMaximum(self.data.n_traces - 1)
        self.trace_end_spinbox.setValue(self.data.n_traces - 1)
        self.trace_end_spinbox.valueChanged.connect(self._on_trace_range_values_changed)
        self.trace_end_spinbox.setEnabled(False)
        trace_end_layout.addWidget(self.trace_end_spinbox)
        trace_layout.addLayout(trace_end_layout)

        trace_group.setLayout(trace_layout)
        layout.addWidget(trace_group)

        # Time window group
        time_window_group = QGroupBox("Time Window")
        time_window_layout = QVBoxLayout()

        # Enable time window checkbox
        self.time_window_checkbox = QCheckBox("Use time window for analysis")
        self.time_window_checkbox.stateChanged.connect(self._on_time_window_changed)
        time_window_layout.addWidget(self.time_window_checkbox)

        # Start time
        time_start_layout = QHBoxLayout()
        time_start_layout.addWidget(QLabel("Start (ms):"))
        self.time_start_spinbox = QDoubleSpinBox()
        self.time_start_spinbox.setMinimum(0.0)
        self.time_start_spinbox.setMaximum(self.data.duration)
        self.time_start_spinbox.setValue(0.0)
        self.time_start_spinbox.setSingleStep(10.0)
        self.time_start_spinbox.setDecimals(1)
        self.time_start_spinbox.valueChanged.connect(self._on_time_window_values_changed)
        self.time_start_spinbox.setEnabled(False)
        time_start_layout.addWidget(self.time_start_spinbox)
        time_window_layout.addLayout(time_start_layout)

        # End time
        time_end_layout = QHBoxLayout()
        time_end_layout.addWidget(QLabel("End (ms):"))
        self.time_end_spinbox = QDoubleSpinBox()
        self.time_end_spinbox.setMinimum(0.0)
        self.time_end_spinbox.setMaximum(self.data.duration)
        self.time_end_spinbox.setValue(self.data.duration)
        self.time_end_spinbox.setSingleStep(10.0)
        self.time_end_spinbox.setDecimals(1)
        self.time_end_spinbox.valueChanged.connect(self._on_time_window_values_changed)
        self.time_end_spinbox.setEnabled(False)
        time_end_layout.addWidget(self.time_end_spinbox)
        time_window_layout.addLayout(time_end_layout)

        time_window_group.setLayout(time_window_layout)
        layout.addWidget(time_window_group)

        # Spectrum display group
        spectrum_display_group = QGroupBox("Spectrum Display")
        spectrum_display_layout = QVBoxLayout()

        # Y-axis scale selection
        y_scale_label = QLabel("Y-axis (Amplitude):")
        spectrum_display_layout.addWidget(y_scale_label)

        self.db_radio = QRadioButton("dB scale (20*log10)")
        self.db_radio.setChecked(True)
        self.db_radio.toggled.connect(self._on_y_scale_changed)
        spectrum_display_layout.addWidget(self.db_radio)

        self.linear_radio = QRadioButton("Linear amplitude")
        spectrum_display_layout.addWidget(self.linear_radio)

        # X-axis scale selection
        spectrum_display_layout.addWidget(QLabel(""))  # Spacer
        x_scale_label = QLabel("X-axis (Frequency):")
        spectrum_display_layout.addWidget(x_scale_label)

        self.freq_linear_radio = QRadioButton("Linear frequency")
        self.freq_linear_radio.setChecked(True)
        self.freq_linear_radio.toggled.connect(self._on_x_scale_changed)
        spectrum_display_layout.addWidget(self.freq_linear_radio)

        self.freq_log_radio = QRadioButton("Log frequency")
        spectrum_display_layout.addWidget(self.freq_log_radio)

        # Phase spectrum option
        spectrum_display_layout.addWidget(QLabel(""))  # Spacer
        self.show_phase_checkbox = QCheckBox("Show phase spectrum")
        self.show_phase_checkbox.setToolTip("Display phase spectrum below amplitude spectrum")
        self.show_phase_checkbox.stateChanged.connect(self._on_show_phase_changed)
        spectrum_display_layout.addWidget(self.show_phase_checkbox)

        # Compare input vs processed
        self.compare_checkbox = QCheckBox("Compare with processed")
        self.compare_checkbox.setToolTip("Overlay processed data spectrum for comparison")
        self.compare_checkbox.setEnabled(self.processed_data is not None)
        self.compare_checkbox.stateChanged.connect(self._on_compare_changed)
        spectrum_display_layout.addWidget(self.compare_checkbox)

        spectrum_display_group.setLayout(spectrum_display_layout)
        layout.addWidget(spectrum_display_group)

        # Colormap selection for data
        colormap_group = QGroupBox("Data Colormap")
        colormap_layout = QVBoxLayout()

        colormap_selector_layout = QHBoxLayout()
        colormap_selector_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['seismic', 'grayscale', 'viridis', 'plasma', 'jet', 'inferno'])
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        colormap_selector_layout.addWidget(self.colormap_combo)
        colormap_layout.addLayout(colormap_selector_layout)

        colormap_group.setLayout(colormap_layout)
        layout.addWidget(colormap_group)

        # Frequency range group
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QVBoxLayout()

        # Min frequency
        freq_min_layout = QHBoxLayout()
        freq_min_layout.addWidget(QLabel("Min (Hz):"))
        self.freq_min_spinbox = QSpinBox()
        self.freq_min_spinbox.setMinimum(0)
        self.freq_min_spinbox.setMaximum(int(self.data.nyquist_freq))
        self.freq_min_spinbox.setValue(0)
        self.freq_min_spinbox.valueChanged.connect(self._on_freq_range_changed)
        freq_min_layout.addWidget(self.freq_min_spinbox)
        freq_layout.addLayout(freq_min_layout)

        # Max frequency
        freq_max_layout = QHBoxLayout()
        freq_max_layout.addWidget(QLabel("Max (Hz):"))
        self.freq_max_spinbox = QSpinBox()
        self.freq_max_spinbox.setMinimum(1)
        self.freq_max_spinbox.setMaximum(int(self.data.nyquist_freq))
        self.freq_max_spinbox.setValue(int(self.data.nyquist_freq))
        self.freq_max_spinbox.valueChanged.connect(self._on_freq_range_changed)
        freq_max_layout.addWidget(self.freq_max_spinbox)
        freq_layout.addLayout(freq_max_layout)

        # Reset button
        reset_btn = QPushButton("Reset Range")
        reset_btn.clicked.connect(self._reset_freq_range)
        freq_layout.addWidget(reset_btn)

        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)

        # Info display
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout()

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self._update_info_label()
        info_layout.addWidget(self.info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def _on_trace_clicked(self, trace_idx: int):
        """Handle trace click in seismic viewer."""
        self.current_trace_idx = trace_idx
        self.trace_spinbox.setValue(trace_idx)
        self._update_spectrum()

    def _on_trace_changed(self, trace_idx: int):
        """Handle trace selection change."""
        self.current_trace_idx = trace_idx
        if not self.show_average:
            self._update_spectrum()

    def _on_average_changed(self, state):
        """Handle average spectrum checkbox change."""
        self.show_average = (state == Qt.CheckState.Checked.value)
        # Disable trace range when showing average of all traces
        if self.show_average:
            self.trace_range_checkbox.setEnabled(False)
        else:
            self.trace_range_checkbox.setEnabled(True)
        self._update_spectrum()

    def _on_trace_range_changed(self, state):
        """Handle trace range checkbox change."""
        self.use_trace_range = (state == Qt.CheckState.Checked.value)
        self.trace_start_spinbox.setEnabled(self.use_trace_range)
        self.trace_end_spinbox.setEnabled(self.use_trace_range)
        self._update_spectrum()

    def _on_trace_range_values_changed(self):
        """Handle trace range start/end value changes."""
        if self.use_trace_range:
            self.trace_range_start = self.trace_start_spinbox.value()
            self.trace_range_end = self.trace_end_spinbox.value()
            # Ensure start <= end
            if self.trace_range_start > self.trace_range_end:
                self.trace_range_start, self.trace_range_end = self.trace_range_end, self.trace_range_start
                self.trace_start_spinbox.blockSignals(True)
                self.trace_end_spinbox.blockSignals(True)
                self.trace_start_spinbox.setValue(self.trace_range_start)
                self.trace_end_spinbox.setValue(self.trace_range_end)
                self.trace_start_spinbox.blockSignals(False)
                self.trace_end_spinbox.blockSignals(False)
            self._update_spectrum()

    def _on_time_window_changed(self, state):
        """Handle time window checkbox change."""
        self.use_time_window = (state == Qt.CheckState.Checked.value)
        self.time_start_spinbox.setEnabled(self.use_time_window)
        self.time_end_spinbox.setEnabled(self.use_time_window)
        self._update_spectrum()

    def _on_time_window_values_changed(self):
        """Handle time window value changes."""
        if self.use_time_window:
            self.time_window_start = self.time_start_spinbox.value()
            self.time_window_end = self.time_end_spinbox.value()
            self._update_spectrum()

    def _on_y_scale_changed(self):
        """Handle Y-axis scale change (dB/linear)."""
        self.spectrum_y_scale = 'db' if self.db_radio.isChecked() else 'linear'
        self._update_spectrum()

    def _on_x_scale_changed(self):
        """Handle X-axis scale change (linear/log frequency)."""
        self.spectrum_x_scale = 'linear' if self.freq_linear_radio.isChecked() else 'log'
        self._update_spectrum()

    def _on_show_phase_changed(self, state):
        """Handle show phase checkbox change."""
        self.show_phase = (state == Qt.CheckState.Checked.value)
        self._update_spectrum()

    def _on_compare_changed(self, state):
        """Handle compare checkbox change."""
        self.compare_spectra = (state == Qt.CheckState.Checked.value)
        self._update_spectrum()

    def set_processed_data(self, processed_data: SeismicData):
        """
        Set processed data for comparison.

        Args:
            processed_data: Processed seismic data to compare with input
        """
        self.processed_data = processed_data
        self.compare_checkbox.setEnabled(processed_data is not None)
        if processed_data is None:
            self.compare_checkbox.setChecked(False)
            self.compare_spectra = False
        self._update_spectrum()

    def _on_colormap_changed(self, colormap_name: str):
        """Handle colormap change from dropdown."""
        self.viewport_state.set_colormap(colormap_name)

    def _on_viewport_colormap_changed(self, colormap_name: str):
        """Handle colormap change from viewport state (main window)."""
        # Update combo box to match without triggering signal
        self.colormap_combo.blockSignals(True)
        index = self.colormap_combo.findText(colormap_name)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
        self.colormap_combo.blockSignals(False)

    def _on_viewport_amplitude_changed(self, min_amp: float, max_amp: float):
        """Handle amplitude range change from viewport state (main window)."""
        # Amplitude range is already applied to viewer through viewport_state
        # No additional action needed - just for consistency
        pass

    def _on_freq_range_changed(self):
        """Handle frequency range change."""
        self.freq_min = self.freq_min_spinbox.value()
        self.freq_max = self.freq_max_spinbox.value()
        self._update_spectrum()

    def _reset_freq_range(self):
        """Reset frequency range to full range."""
        self.freq_min_spinbox.setValue(0)
        self.freq_max_spinbox.setValue(int(self.data.nyquist_freq))

    def _update_spectrum(self):
        """Update spectrum display with amplitude, phase, and comparison support."""
        # Clear figure and recreate axes based on display options
        self.spectrum_figure.clear()

        if self.show_phase:
            # Two subplots: amplitude and phase
            self.spectrum_ax = self.spectrum_figure.add_subplot(211)
            self.phase_ax = self.spectrum_figure.add_subplot(212)
        else:
            self.spectrum_ax = self.spectrum_figure.add_subplot(111)
            self.phase_ax = None

        # Extract time window if enabled
        if self.use_time_window:
            start_idx = int(self.time_window_start / self.data.sample_rate)
            end_idx = int(self.time_window_end / self.data.sample_rate)
            start_idx = max(0, start_idx)
            end_idx = min(self.data.n_samples, end_idx)
            if start_idx >= end_idx:
                start_idx = 0
                end_idx = self.data.n_samples
            windowed_traces = self.data.traces[start_idx:end_idx, :]
            window_info = f" [{self.time_window_start:.1f}-{self.time_window_end:.1f} ms]"

            # Also window processed data if comparing
            if self.processed_data is not None and self.compare_spectra:
                windowed_processed = self.processed_data.traces[start_idx:end_idx, :]
            else:
                windowed_processed = None
        else:
            windowed_traces = self.data.traces
            window_info = ""
            windowed_processed = self.processed_data.traces if (
                self.processed_data is not None and self.compare_spectra
            ) else None

        # Compute spectra for input data
        if self.show_average:
            if self.show_phase:
                frequencies, amplitudes_db, phase_deg = self.spectral_analyzer.compute_average_spectrum_with_phase(windowed_traces)
            else:
                frequencies, amplitudes_db = self.spectral_analyzer.compute_average_spectrum(windowed_traces)
                phase_deg = None
            title = f"Average Spectrum (All Traces){window_info}"
        elif self.use_trace_range:
            # Use trace range for spectral estimation
            range_start = max(0, min(self.trace_range_start, windowed_traces.shape[1] - 1))
            range_end = max(0, min(self.trace_range_end, windowed_traces.shape[1] - 1))
            if range_start > range_end:
                range_start, range_end = range_end, range_start
            range_traces = windowed_traces[:, range_start:range_end + 1]
            trace_range_info = f" (Traces {range_start}-{range_end})"
            if self.show_phase:
                frequencies, amplitudes_db, phase_deg = self.spectral_analyzer.compute_average_spectrum_with_phase(range_traces)
            else:
                frequencies, amplitudes_db = self.spectral_analyzer.compute_average_spectrum(range_traces)
                phase_deg = None
            title = f"Average Spectrum{trace_range_info}{window_info}"
        else:
            trace = windowed_traces[:, self.current_trace_idx]
            if self.show_phase:
                frequencies, amplitudes_db, phase_deg = self.spectral_analyzer.compute_spectrum_with_phase(trace)
            else:
                frequencies, amplitudes_db = self.spectral_analyzer.compute_spectrum(trace)
                phase_deg = None
            title = f"Spectrum - Trace {self.current_trace_idx}{window_info}"

        # Compute spectra for processed data if comparing
        proc_amps_db = None
        proc_phase_deg = None
        if windowed_processed is not None:
            if self.show_average:
                if self.show_phase:
                    _, proc_amps_db, proc_phase_deg = self.spectral_analyzer.compute_average_spectrum_with_phase(windowed_processed)
                else:
                    _, proc_amps_db = self.spectral_analyzer.compute_average_spectrum(windowed_processed)
            elif self.use_trace_range:
                # Use same trace range for processed data
                range_start = max(0, min(self.trace_range_start, windowed_processed.shape[1] - 1))
                range_end = max(0, min(self.trace_range_end, windowed_processed.shape[1] - 1))
                if range_start > range_end:
                    range_start, range_end = range_end, range_start
                range_processed = windowed_processed[:, range_start:range_end + 1]
                if self.show_phase:
                    _, proc_amps_db, proc_phase_deg = self.spectral_analyzer.compute_average_spectrum_with_phase(range_processed)
                else:
                    _, proc_amps_db = self.spectral_analyzer.compute_average_spectrum(range_processed)
            else:
                proc_trace = windowed_processed[:, self.current_trace_idx]
                if self.show_phase:
                    _, proc_amps_db, proc_phase_deg = self.spectral_analyzer.compute_spectrum_with_phase(proc_trace)
                else:
                    _, proc_amps_db = self.spectral_analyzer.compute_spectrum(proc_trace)

        # Filter by frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        plot_freqs = frequencies[mask]
        plot_amps_db = amplitudes_db[mask]
        plot_phase = phase_deg[mask] if phase_deg is not None else None
        plot_proc_amps = proc_amps_db[mask] if proc_amps_db is not None else None
        plot_proc_phase = proc_phase_deg[mask] if proc_phase_deg is not None else None

        # Prepare Y-axis data based on scale selection
        if self.spectrum_y_scale == 'db':
            y_data = plot_amps_db
            y_label = 'Amplitude (dB)'
            y_proc = plot_proc_amps
        else:
            y_data = 10 ** (plot_amps_db / 20.0)
            y_label = 'Amplitude (Linear)'
            y_proc = 10 ** (plot_proc_amps / 20.0) if plot_proc_amps is not None else None

        # Plot amplitude spectrum
        self.spectrum_ax.plot(plot_freqs, y_data, 'b-', linewidth=1.0, label='Input')

        if y_proc is not None:
            self.spectrum_ax.plot(plot_freqs, y_proc, 'r-', linewidth=1.0, alpha=0.8, label='Processed')

        self.spectrum_ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        self.spectrum_ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
        self.spectrum_ax.set_title(f"Amplitude {title}", fontsize=11, fontweight='bold')
        self.spectrum_ax.grid(True, alpha=0.3, linestyle='--')

        # X-axis scale
        if self.spectrum_x_scale == 'log':
            self.spectrum_ax.set_xscale('log')
            if self.freq_min == 0 and len(plot_freqs) > 1:
                self.spectrum_ax.set_xlim(left=max(0.1, plot_freqs[1]))
        else:
            self.spectrum_ax.set_xscale('linear')

        # Find and mark dominant frequency
        if len(plot_freqs) > 0:
            dominant_freq = self.spectral_analyzer.find_dominant_frequency(plot_freqs, plot_amps_db)
            dominant_idx = np.argmax(plot_amps_db)
            dominant_y = y_data[dominant_idx]
            self.spectrum_ax.plot(dominant_freq, dominant_y, 'bo', markersize=8,
                                 label=f'Peak: {dominant_freq:.1f} Hz')

        if y_proc is not None or len(plot_freqs) > 0:
            self.spectrum_ax.legend(loc='upper right')

        # Plot phase spectrum if enabled
        if self.phase_ax is not None and plot_phase is not None:
            self.phase_ax.plot(plot_freqs, plot_phase, 'b-', linewidth=1.0, label='Input')

            if plot_proc_phase is not None:
                self.phase_ax.plot(plot_freqs, plot_proc_phase, 'r-', linewidth=1.0, alpha=0.8, label='Processed')
                self.phase_ax.legend(loc='upper right')

            self.phase_ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
            self.phase_ax.set_ylabel('Phase (degrees)', fontsize=10, fontweight='bold')
            self.phase_ax.set_title(f"Phase {title}", fontsize=11, fontweight='bold')
            self.phase_ax.grid(True, alpha=0.3, linestyle='--')
            self.phase_ax.set_ylim(-180, 180)

            if self.spectrum_x_scale == 'log':
                self.phase_ax.set_xscale('log')
                if self.freq_min == 0 and len(plot_freqs) > 1:
                    self.phase_ax.set_xlim(left=max(0.1, plot_freqs[1]))

        self.spectrum_figure.tight_layout()
        self.spectrum_canvas.draw()
        self._update_info_label()

    def _update_info_label(self):
        """Update information label."""
        info_text = (
            f"<b>Data Info:</b><br>"
            f"Traces: {self.data.n_traces}<br>"
            f"Samples: {self.data.n_samples}<br>"
            f"Duration: {self.data.duration:.1f} ms<br>"
            f"Sample Rate: {self.data.sample_rate:.2f} ms<br>"
            f"Nyquist: {self.data.nyquist_freq:.1f} Hz<br>"
        )

        if self.show_average:
            info_text += f"<br><b>Mode:</b> All traces average"
        elif self.use_trace_range:
            n_traces_in_range = self.trace_range_end - self.trace_range_start + 1
            info_text += f"<br><b>Trace Range:</b> {self.trace_range_start}-{self.trace_range_end}"
            info_text += f"<br><b>Traces Used:</b> {n_traces_in_range}"
        else:
            info_text += f"<br><b>Current Trace:</b> {self.current_trace_idx}"

        self.info_label.setText(info_text)
