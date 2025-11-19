"""
Seismic viewer widget - displays seismic data with proper scaling.
Uses matplotlib for professional seismic visualization.
"""
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys
from models.seismic_data import SeismicData
from models.viewport_state import ViewportState, ViewportLimits


class SeismicViewer(QWidget):
    """
    Professional seismic data viewer widget.

    Displays seismic data as variable density (color) plot with proper
    amplitude scaling and axis labels.
    """

    def __init__(self, title: str, viewport_state: ViewportState, parent=None):
        super().__init__(parent)
        self.title = title
        self.viewport_state = viewport_state
        self.data = None
        self._current_image = None

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Create custom toolbar (minimal controls)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Connect to viewport state changes
        self.viewport_state.limits_changed.connect(self._on_limits_changed)
        self.viewport_state.gain_changed.connect(self._on_gain_changed)
        self.viewport_state.clip_percentile_changed.connect(self._on_clip_changed)

        # Initialize empty plot
        self._setup_empty_plot()

    def _setup_empty_plot(self):
        """Set up initial empty plot."""
        self.ax.clear()
        self.ax.set_xlabel('Trace Number', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Time (ms)', fontsize=10, fontweight='bold')
        self.ax.set_title(self.title, fontsize=12, fontweight='bold', pad=10)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.invert_yaxis()  # Time increases downward (standard seismic convention)
        self.canvas.draw()

    def set_data(self, data: SeismicData):
        """
        Set seismic data to display.

        Args:
            data: SeismicData object to visualize
        """
        self.data = data
        self._update_display()

    def _update_display(self):
        """Update the display with current data and viewport state."""
        if self.data is None:
            return

        self.ax.clear()

        # Get current viewport limits and gain
        limits = self.viewport_state.limits
        gain = self.viewport_state.gain
        clip_percentile = self.viewport_state.clip_percentile

        # Apply gain to data
        display_data = self.data.traces * gain

        # Calculate clip value based on percentile
        clip_value = np.percentile(np.abs(display_data), clip_percentile)

        # Create color mesh plot (variable density display)
        time_axis = self.data.get_time_axis()
        trace_axis = self.data.get_trace_axis()

        # Use professional seismic colormap (seismic or RdBu)
        self._current_image = self.ax.imshow(
            display_data,
            aspect='auto',
            cmap='seismic',  # Red-White-Blue, standard for seismic
            vmin=-clip_value,
            vmax=clip_value,
            extent=[trace_axis[0], trace_axis[-1], time_axis[-1], time_axis[0]],
            interpolation='bilinear'
        )

        # Set viewport limits
        self.ax.set_xlim(limits.trace_min, limits.trace_max)
        self.ax.set_ylim(limits.time_max, limits.time_min)  # Inverted for seismic convention

        # Labels and styling
        self.ax.set_xlabel('Trace Number', fontsize=10, fontweight='bold')
        self.ax.set_ylabel('Time (ms)', fontsize=10, fontweight='bold')
        self.ax.set_title(self.title, fontsize=12, fontweight='bold', pad=10)
        self.ax.grid(True, alpha=0.3, linestyle='--', color='gray', linewidth=0.5)

        # Add colorbar if not present
        if not hasattr(self, 'colorbar') or self.colorbar is None:
            self.colorbar = self.figure.colorbar(
                self._current_image,
                ax=self.ax,
                label='Amplitude',
                fraction=0.046,
                pad=0.04
            )
        else:
            self.colorbar.update_normal(self._current_image)

        self.canvas.draw()

    def _on_limits_changed(self, limits: ViewportLimits):
        """Handle viewport limits change."""
        if self.data is not None:
            self._update_display()

    def _on_gain_changed(self, gain: float):
        """Handle gain change."""
        if self.data is not None:
            self._update_display()

    def _on_clip_changed(self, percentile: float):
        """Handle clip percentile change."""
        if self.data is not None:
            self._update_display()

    def clear(self):
        """Clear the display."""
        self.data = None
        self._current_image = None
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self._setup_empty_plot()
