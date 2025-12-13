"""
QC Presentation Tool - Interactive multi-band seismic QC viewer with PowerPoint export.

Features:
- Multi-band filtering (user-defined frequency bands)
- Flip sequence: Before/After/Difference for each band
- Global spectra at end showing all 3 versions overlaid
- Per-band gain control with setup mode
- Zoom persistence across flips
- Interactive spectral window selection
- PowerPoint export with one image per slide
"""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStatusBar, QComboBox, QMessageBox, QFileDialog, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from models.seismic_data import SeismicData
from models.viewport_state import ViewportState
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from processors.bandpass_filter import BandpassFilter
from processors.spectral_analyzer import SpectralAnalyzer


class ViewType(Enum):
    """Types of views in the flip sequence."""
    BEFORE = "Before"
    AFTER = "After"
    DIFFERENCE = "Difference"
    SPECTRUM_WHOLE = "Spectrum (Whole Gather)"
    SPECTRUM_WINDOW = "Spectrum (Window)"


@dataclass
class FrequencyBand:
    """Definition of a frequency band for filtering."""
    low_freq: float      # Hz
    high_freq: float     # Hz
    gain: float = 1.0    # Display gain multiplier
    name: str = ""       # Auto-generated if empty

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.low_freq:.0f}-{self.high_freq:.0f} Hz"


@dataclass
class SpectralWindow:
    """Rectangular window for spectral analysis."""
    time_start: float = 0.0     # ms
    time_end: float = 1000.0    # ms
    trace_start: int = 0        # Index
    trace_end: int = 100        # Index


@dataclass
class PresentationState:
    """Tracks current position in flip sequence."""
    current_sequence_index: int = 0
    is_setup_mode: bool = True
    gains_locked: bool = False


class QCPresentationWindow(QMainWindow):
    """
    QC Presentation Tool for multi-band seismic analysis.

    Features:
    - Multiple frequency band filtering (Before/After/Diff for each)
    - Full bandwidth included as first band
    - LMB/RMB flip navigation
    - Per-band gain control
    - Persistent zoom across flips
    - Spectral display with interactive window selection
    - PowerPoint export
    """

    # Signals
    export_completed = pyqtSignal(str)  # Path to saved file

    def __init__(self, viewport_state: ViewportState, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QC Presentation Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Shared viewport state for synchronized views
        self.viewport_state = viewport_state

        # Data sources
        self.input_data: Optional[SeismicData] = None
        self.processed_data: Optional[SeismicData] = None

        # Configuration
        self.bands: List[FrequencyBand] = []
        self.fullband_gain: float = 1.0
        self.filter_order: int = 4
        self.spectral_window: Optional[SpectralWindow] = None

        # Precomputed band data: band_index -> (before, after, difference)
        # Index -1 is fullband
        self.band_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # State
        self.state = PresentationState()
        self.flip_sequence: List[Tuple[int, ViewType]] = []  # (band_index, view_type)

        # Saved viewport for zoom persistence
        self._saved_limits = None

        # UI Components
        self.viewer: Optional[SeismicViewerPyQtGraph] = None
        self.spectrum_figure: Optional[Figure] = None
        self.spectrum_canvas: Optional[FigureCanvas] = None
        self.spectrum_ax = None
        self.view_label: Optional[QLabel] = None
        self.selection_roi: Optional[pg.RectROI] = None

        # Create UI
        self._init_ui()
        self._setup_shortcuts()

        # Connect to viewport state changes
        self.viewport_state.colormap_changed.connect(self._on_viewport_colormap_changed)

        # Show initial message
        self.statusBar().showMessage("LMB: Forward | RMB: Backward | Space: Toggle | Setup mode active")

    def _init_ui(self):
        """Initialize user interface."""
        central = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Left panel: Viewer + spectrum
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Top bar with view label and colormap
        top_bar = self._create_top_bar()
        left_layout.addWidget(top_bar)

        # Create stacked widget area for viewer/spectrum
        self.viewer_container = QWidget()
        viewer_layout = QVBoxLayout()
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        # Seismic viewer
        self.viewer = SeismicViewerPyQtGraph("QC Presentation", self.viewport_state)
        self.viewer.mousePressEvent = self._on_viewer_click
        viewer_layout.addWidget(self.viewer)

        # Spectrum canvas (hidden initially)
        self.spectrum_figure = Figure(figsize=(10, 6), facecolor='white')
        self.spectrum_canvas = FigureCanvas(self.spectrum_figure)
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        self.spectrum_canvas.hide()
        viewer_layout.addWidget(self.spectrum_canvas)

        self.viewer_container.setLayout(viewer_layout)
        left_layout.addWidget(self.viewer_container, stretch=1)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel: Control panel
        from views.qc_presentation_control_panel import QCPresentationControlPanel
        self.control_panel = QCPresentationControlPanel(parent=self)
        self.control_panel.setMaximumWidth(320)

        # Connect control panel signals
        self.control_panel.bands_changed.connect(self._on_bands_changed)
        self.control_panel.gain_changed.connect(self._on_gain_changed)
        self.control_panel.fullband_gain_changed.connect(self._on_fullband_gain_changed)
        self.control_panel.spectral_window_changed.connect(self._on_spectral_window_changed)
        self.control_panel.mode_changed.connect(self._on_mode_changed)
        self.control_panel.export_requested.connect(self._on_export_requested)
        self.control_panel.draw_roi_requested.connect(self._enable_rectangle_selection)

        main_layout.addWidget(self.control_panel)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _create_top_bar(self) -> QWidget:
        """Create top bar with view label and colormap selector."""
        top_bar = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(5, 5, 5, 5)

        # Current view label
        self.view_label = QLabel("Full Band : Before")
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        self.view_label.setFont(font)
        self._update_view_label_style("#2196F3")  # Blue for Before
        top_layout.addWidget(self.view_label, stretch=1)

        # Colormap selector
        colormap_label = QLabel("Colormap:")
        colormap_label.setStyleSheet("font-weight: bold; padding: 5px;")
        top_layout.addWidget(colormap_label)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Seismic (RWB)", "Grayscale", "Viridis", "Plasma", "Inferno", "Jet"
        ])
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        top_layout.addWidget(self.colormap_combo)

        top_bar.setLayout(top_layout)
        return top_bar

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Space to toggle forward
        space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        space_shortcut.activated.connect(self._cycle_forward)

        # Left/Right arrows
        left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        left_shortcut.activated.connect(self._cycle_backward)

        right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        right_shortcut.activated.connect(self._cycle_forward)

    def set_data(self, input_data: SeismicData, processed_data: Optional[SeismicData] = None):
        """
        Set input and processed data.

        Args:
            input_data: Input seismic data (before processing)
            processed_data: Processed seismic data (after processing)
        """
        self.input_data = input_data
        self.processed_data = processed_data if processed_data is not None else input_data

        # Update control panel with data info
        if hasattr(self, 'control_panel'):
            self.control_panel.set_data_info(
                nyquist_freq=input_data.nyquist_freq,
                duration=input_data.duration,
                n_traces=input_data.n_traces
            )

        # Initialize spectral window
        self.spectral_window = SpectralWindow(
            time_start=0.0,
            time_end=input_data.duration,
            trace_start=0,
            trace_end=input_data.n_traces - 1
        )

        # Build flip sequence and precompute data
        self._build_flip_sequence()
        self._precompute_all_bands()

        # Display first view
        self._update_display()

    def _build_flip_sequence(self):
        """Build the complete flip sequence."""
        self.flip_sequence = []

        # Fullband (index -1): Before, After, Difference
        self.flip_sequence.append((-1, ViewType.BEFORE))
        self.flip_sequence.append((-1, ViewType.AFTER))
        self.flip_sequence.append((-1, ViewType.DIFFERENCE))

        # Each user-defined band
        for i in range(len(self.bands)):
            self.flip_sequence.append((i, ViewType.BEFORE))
            self.flip_sequence.append((i, ViewType.AFTER))
            self.flip_sequence.append((i, ViewType.DIFFERENCE))

        # Spectral displays at end
        self.flip_sequence.append((None, ViewType.SPECTRUM_WHOLE))
        self.flip_sequence.append((None, ViewType.SPECTRUM_WINDOW))

    def _precompute_all_bands(self):
        """Compute filtered data for all bands upfront."""
        if self.input_data is None:
            return

        self.band_data = {}

        # Fullband (index -1)
        before = self.input_data.traces.copy()
        after = self.processed_data.traces.copy()
        diff = before - after
        self.band_data[-1] = (before, after, diff)

        # Each frequency band
        for i, band in enumerate(self.bands):
            try:
                bp_filter = BandpassFilter(
                    low_freq=band.low_freq,
                    high_freq=band.high_freq,
                    order=self.filter_order
                )

                before_filtered = bp_filter.process(self.input_data).traces
                after_filtered = bp_filter.process(self.processed_data).traces
                diff_filtered = before_filtered - after_filtered

                self.band_data[i] = (before_filtered, after_filtered, diff_filtered)
            except Exception as e:
                print(f"Warning: Failed to filter band {band.name}: {e}")
                # Use unfiltered data as fallback
                self.band_data[i] = (
                    self.input_data.traces.copy(),
                    self.processed_data.traces.copy(),
                    self.input_data.traces - self.processed_data.traces
                )

    def _on_viewer_click(self, event):
        """Handle mouse clicks on viewer for flipping."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._cycle_forward()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._cycle_backward()
            event.accept()
        else:
            super(type(self.viewer), self.viewer).mousePressEvent(event)

    def _cycle_forward(self):
        """Cycle to next view in sequence."""
        if not self.flip_sequence:
            return

        self.state.current_sequence_index = (
            self.state.current_sequence_index + 1
        ) % len(self.flip_sequence)

        self._update_display()

    def _cycle_backward(self):
        """Cycle to previous view in sequence."""
        if not self.flip_sequence:
            return

        self.state.current_sequence_index = (
            self.state.current_sequence_index - 1
        ) % len(self.flip_sequence)

        self._update_display()

    def _update_display(self):
        """Update display with current view, preserving zoom."""
        if not self.flip_sequence or self.input_data is None:
            return

        # Save current viewport before changing data
        self._save_viewport()

        band_idx, view_type = self.flip_sequence[self.state.current_sequence_index]

        # Handle spectral views
        if view_type in (ViewType.SPECTRUM_WHOLE, ViewType.SPECTRUM_WINDOW):
            self._show_spectral_display(view_type == ViewType.SPECTRUM_WINDOW)
            self._update_view_label_for_spectrum(view_type)
            return

        # Show seismic viewer, hide spectrum
        self.viewer.show()
        self.spectrum_canvas.hide()

        # Get data for current band and view
        if band_idx not in self.band_data:
            return

        before, after, diff = self.band_data[band_idx]

        if view_type == ViewType.BEFORE:
            data = before
        elif view_type == ViewType.AFTER:
            data = after
        else:  # DIFFERENCE
            data = diff

        # Get gain for this band
        gain = self._get_gain_for_band(band_idx)

        # Compute amplitude range based on data and gain
        base_amp = self._compute_base_amplitude(data)
        scaled_amp = base_amp / gain

        # Update amplitude range (this affects display scaling)
        self.viewport_state.blockSignals(True)
        try:
            self.viewport_state.set_amplitude_range(-scaled_amp, scaled_amp)
        except ValueError:
            pass  # Ignore if invalid range
        self.viewport_state.blockSignals(False)

        # Create SeismicData for display
        display_data = SeismicData(
            traces=data,
            sample_rate=self.input_data.sample_rate,
            headers=self.input_data.headers,
            metadata=self.input_data.metadata.copy()
        )

        # Update viewer
        self.viewer.set_data(display_data)

        # Restore viewport (zoom persistence)
        self._restore_viewport()

        # Update view label
        self._update_view_label_for_seismic(band_idx, view_type)

        # Status message
        band_name = "Full Band" if band_idx == -1 else self.bands[band_idx].name
        self.statusBar().showMessage(
            f"{band_name} : {view_type.value} (Gain: {gain:.1f}x)",
            3000
        )

    def _get_gain_for_band(self, band_idx: int) -> float:
        """Get gain for a specific band index."""
        if band_idx == -1:
            return self.fullband_gain
        elif 0 <= band_idx < len(self.bands):
            return self.bands[band_idx].gain
        return 1.0

    def _compute_base_amplitude(self, data: np.ndarray) -> float:
        """Compute base amplitude for display scaling."""
        # Use 99th percentile of absolute values
        percentile = np.percentile(np.abs(data), 99)
        return max(percentile, 1e-10)  # Avoid zero

    def _save_viewport(self):
        """Save current viewport limits."""
        limits = self.viewport_state.limits
        self._saved_limits = (
            limits.time_min, limits.time_max,
            limits.trace_min, limits.trace_max
        )

    def _restore_viewport(self):
        """Restore saved viewport limits."""
        if self._saved_limits is not None:
            self.viewport_state.set_limits(*self._saved_limits)

    def _show_spectral_display(self, use_window: bool):
        """Display spectra with Before/After/Diff overlaid."""
        # Hide seismic viewer, show spectrum
        self.viewer.hide()
        self.spectrum_canvas.show()

        if self.input_data is None:
            return

        # Get data region
        if use_window and self.spectral_window:
            win = self.spectral_window
            # Convert time to sample indices
            t_start_idx = int(win.time_start / self.input_data.sample_rate)
            t_end_idx = int(win.time_end / self.input_data.sample_rate)
            t_start_idx = max(0, t_start_idx)
            t_end_idx = min(self.input_data.n_samples, t_end_idx)

            tr_start = max(0, win.trace_start)
            tr_end = min(self.input_data.n_traces - 1, win.trace_end)

            before_region = self.input_data.traces[t_start_idx:t_end_idx, tr_start:tr_end+1]
            after_region = self.processed_data.traces[t_start_idx:t_end_idx, tr_start:tr_end+1]
        else:
            before_region = self.input_data.traces
            after_region = self.processed_data.traces

        diff_region = before_region - after_region

        # Compute spectra
        analyzer = SpectralAnalyzer(self.input_data.sample_rate)

        try:
            freq_before, amp_before = analyzer.compute_average_spectrum(before_region)
            freq_after, amp_after = analyzer.compute_average_spectrum(after_region)
            freq_diff, amp_diff = analyzer.compute_average_spectrum(diff_region)
        except Exception as e:
            print(f"Error computing spectrum: {e}")
            return

        # Plot overlaid spectra
        self.spectrum_ax.clear()
        self.spectrum_ax.plot(freq_before, amp_before, 'b-', label='Before', linewidth=1.5)
        self.spectrum_ax.plot(freq_after, amp_after, 'g-', label='After', linewidth=1.5)
        self.spectrum_ax.plot(freq_diff, amp_diff, 'r-', label='Difference', linewidth=1.5, alpha=0.7)

        self.spectrum_ax.set_xlabel('Frequency (Hz)', fontsize=11)
        self.spectrum_ax.set_ylabel('Amplitude (dB)', fontsize=11)

        title = 'Spectral Comparison'
        if use_window:
            title += f' (Window: T={self.spectral_window.time_start:.0f}-{self.spectral_window.time_end:.0f}ms)'
        else:
            title += ' (Whole Gather)'
        self.spectrum_ax.set_title(title, fontsize=12)

        self.spectrum_ax.legend(loc='upper right')
        self.spectrum_ax.grid(True, alpha=0.3)
        self.spectrum_ax.set_xlim(0, self.input_data.nyquist_freq)

        self.spectrum_figure.tight_layout()
        self.spectrum_canvas.draw()

    def _update_view_label_for_seismic(self, band_idx: int, view_type: ViewType):
        """Update view label for seismic display."""
        band_name = "Full Band" if band_idx == -1 else self.bands[band_idx].name
        text = f"{band_name} : {view_type.value}"

        # Color coding
        colors = {
            ViewType.BEFORE: "#2196F3",     # Blue
            ViewType.AFTER: "#4CAF50",      # Green
            ViewType.DIFFERENCE: "#FF9800"  # Orange
        }
        color = colors.get(view_type, "#2196F3")

        self.view_label.setText(text)
        self._update_view_label_style(color)

    def _update_view_label_for_spectrum(self, view_type: ViewType):
        """Update view label for spectrum display."""
        text = view_type.value
        color = "#9C27B0"  # Purple for spectrum

        self.view_label.setText(text)
        self._update_view_label_style(color)

    def _update_view_label_style(self, color: str):
        """Update view label background color."""
        self.view_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 5px;
            }}
        """)

    def _enable_rectangle_selection(self):
        """Enable interactive rectangle drawing on seismic viewer."""
        if self.selection_roi is not None:
            # Remove existing ROI
            try:
                self.viewer.plot_item.removeItem(self.selection_roi)
            except:
                pass

        # Create ROI with initial position
        initial_pos = [0, 0]
        initial_size = [
            self.input_data.n_traces // 2 if self.input_data else 50,
            self.input_data.duration / 2 if self.input_data else 500
        ]

        self.selection_roi = pg.RectROI(
            initial_pos, initial_size,
            pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen('y', width=3),
            handlePen=pg.mkPen('y', width=2),
            handleHoverPen=pg.mkPen('y', width=3)
        )

        # Add handles for resizing
        self.selection_roi.addScaleHandle([0, 0], [1, 1])
        self.selection_roi.addScaleHandle([1, 1], [0, 0])
        self.selection_roi.addScaleHandle([0, 1], [1, 0])
        self.selection_roi.addScaleHandle([1, 0], [0, 1])

        self.viewer.plot_item.addItem(self.selection_roi)
        self.selection_roi.sigRegionChangeFinished.connect(self._on_roi_changed)

        self.statusBar().showMessage("Draw rectangle on seismic to define spectral window", 5000)

    def _on_roi_changed(self, roi):
        """Update spectral window from ROI."""
        pos = roi.pos()
        size = roi.size()

        # ROI coordinates: x = trace, y = time
        trace_start = int(pos.x())
        trace_end = int(pos.x() + size.x())
        time_start = pos.y()
        time_end = pos.y() + size.y()

        # Update spectral window
        self.spectral_window = SpectralWindow(
            time_start=time_start,
            time_end=time_end,
            trace_start=trace_start,
            trace_end=trace_end
        )

        # Update control panel spinboxes
        if hasattr(self, 'control_panel'):
            self.control_panel.set_spectral_window(
                time_start, time_end, trace_start, trace_end
            )

    # === Control Panel Signal Handlers ===

    def _on_bands_changed(self, bands: List[FrequencyBand]):
        """Handle band configuration changes."""
        self.bands = bands
        self._build_flip_sequence()

        if self.input_data is not None:
            self._precompute_all_bands()

        # Reset to first view
        self.state.current_sequence_index = 0
        self._update_display()

    def _on_gain_changed(self, band_index: int, gain: float):
        """Handle gain change for a specific band."""
        if 0 <= band_index < len(self.bands):
            self.bands[band_index].gain = gain

            # If currently viewing this band, update display
            if self.flip_sequence:
                current_band, _ = self.flip_sequence[self.state.current_sequence_index]
                if current_band == band_index:
                    self._update_display()

    def _on_fullband_gain_changed(self, gain: float):
        """Handle fullband gain change."""
        self.fullband_gain = gain

        # If currently viewing fullband, update display
        if self.flip_sequence:
            current_band, view_type = self.flip_sequence[self.state.current_sequence_index]
            if current_band == -1 and view_type not in (ViewType.SPECTRUM_WHOLE, ViewType.SPECTRUM_WINDOW):
                self._update_display()

    def _on_spectral_window_changed(self, window: SpectralWindow):
        """Handle spectral window change from spinboxes."""
        self.spectral_window = window

        # Update ROI if visible
        if self.selection_roi is not None:
            self.selection_roi.blockSignals(True)
            self.selection_roi.setPos([window.trace_start, window.time_start])
            self.selection_roi.setSize([
                window.trace_end - window.trace_start,
                window.time_end - window.time_start
            ])
            self.selection_roi.blockSignals(False)

        # If showing windowed spectrum, update
        if self.flip_sequence:
            _, view_type = self.flip_sequence[self.state.current_sequence_index]
            if view_type == ViewType.SPECTRUM_WINDOW:
                self._show_spectral_display(True)

    def _on_mode_changed(self, mode: str):
        """Handle mode change (setup vs presentation)."""
        if mode == "presentation":
            self.state.is_setup_mode = False
            self.state.gains_locked = True
            self.statusBar().showMessage("Presentation mode - gains locked", 3000)
        else:
            self.state.is_setup_mode = True
            self.state.gains_locked = False
            self.statusBar().showMessage("Setup mode - adjust gains as needed", 3000)

    def _on_export_requested(self):
        """Handle export to PowerPoint request."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PowerPoint Presentation",
            "qc_presentation.pptx",
            "PowerPoint Files (*.pptx)"
        )

        if file_path:
            self._export_to_pptx(file_path)

    def _export_to_pptx(self, output_path: str):
        """Export all views to PowerPoint presentation."""
        try:
            from utils.pptx_export import PPTXExporter

            exporter = PPTXExporter(output_path)

            # Save current state
            saved_index = self.state.current_sequence_index

            # Iterate through all views
            for i, (band_idx, view_type) in enumerate(self.flip_sequence):
                self.state.current_sequence_index = i

                if view_type in (ViewType.SPECTRUM_WHOLE, ViewType.SPECTRUM_WINDOW):
                    # Export spectrum
                    use_window = (view_type == ViewType.SPECTRUM_WINDOW)
                    self._show_spectral_display(use_window)

                    title = view_type.value
                    subtitle = self._get_export_subtitle(band_idx, view_type)

                    exporter.add_spectrum_slide(
                        self.spectrum_figure,
                        title=title,
                        subtitle=subtitle
                    )
                else:
                    # Export seismic image
                    self._update_display()

                    band_name = "Full Band" if band_idx == -1 else self.bands[band_idx].name
                    title = f"{band_name} : {view_type.value}"
                    subtitle = self._get_export_subtitle(band_idx, view_type)

                    # Get current display data
                    if band_idx in self.band_data:
                        before, after, diff = self.band_data[band_idx]
                        if view_type == ViewType.BEFORE:
                            data = before
                        elif view_type == ViewType.AFTER:
                            data = after
                        else:
                            data = diff

                        exporter.add_seismic_slide(
                            data,
                            title=title,
                            subtitle=subtitle,
                            sample_rate=self.input_data.sample_rate
                        )

            # Save presentation
            exporter.save()

            # Restore state
            self.state.current_sequence_index = saved_index
            self._update_display()

            self.statusBar().showMessage(f"Exported to: {output_path}", 5000)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Presentation saved to:\n{output_path}"
            )

            self.export_completed.emit(output_path)

        except ImportError:
            QMessageBox.critical(
                self,
                "Export Error",
                "python-pptx library not installed.\n"
                "Install with: pip install python-pptx"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export presentation:\n{str(e)}"
            )

    def _get_export_subtitle(self, band_idx: int, view_type: ViewType) -> str:
        """Generate subtitle for export slide."""
        parts = []

        if band_idx is not None and band_idx >= 0 and band_idx < len(self.bands):
            band = self.bands[band_idx]
            parts.append(f"Band: {band.low_freq}-{band.high_freq} Hz")
            parts.append(f"Gain: {band.gain:.1f}x")
        elif band_idx == -1:
            parts.append("Full Bandwidth")
            parts.append(f"Gain: {self.fullband_gain:.1f}x")

        if self.input_data:
            parts.append(f"Traces: {self.input_data.n_traces}")
            parts.append(f"Duration: {self.input_data.duration:.0f} ms")

        return " | ".join(parts)

    def _on_colormap_changed(self, index: int):
        """Handle colormap selection change."""
        colormap_map = {
            0: 'seismic', 1: 'grayscale', 2: 'viridis',
            3: 'plasma', 4: 'inferno', 5: 'jet'
        }
        colormap = colormap_map.get(index, 'seismic')
        self.viewport_state.set_colormap(colormap)

    def _on_viewport_colormap_changed(self, colormap: str):
        """Handle colormap change from viewport state."""
        index_map = {
            'seismic': 0, 'grayscale': 1, 'viridis': 2,
            'plasma': 3, 'inferno': 4, 'jet': 5
        }
        index = index_map.get(colormap, 0)

        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentIndex(index)
        self.colormap_combo.blockSignals(False)

    def go_to_band(self, band_idx: int, view_type: ViewType = ViewType.BEFORE):
        """Navigate directly to a specific band and view type."""
        for i, (b_idx, v_type) in enumerate(self.flip_sequence):
            if b_idx == band_idx and v_type == view_type:
                self.state.current_sequence_index = i
                self._update_display()
                return

    def clear_data(self):
        """Clear all data."""
        self.input_data = None
        self.processed_data = None
        self.band_data = {}
        self.flip_sequence = []
        self.state = PresentationState()
        self.viewer.clear()
