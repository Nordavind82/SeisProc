"""
Viewport state manager - handles synchronized zoom, pan, and display settings.
This is the single source of truth for all three panels.
"""
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass
from typing import Optional


@dataclass
class ViewportLimits:
    """Current viewport limits."""
    time_min: float = 0.0
    time_max: float = 1000.0
    trace_min: float = 0.0
    trace_max: float = 100.0

    def copy(self) -> 'ViewportLimits':
        """Create a copy of these limits."""
        return ViewportLimits(
            time_min=self.time_min,
            time_max=self.time_max,
            trace_min=self.trace_min,
            trace_max=self.trace_max
        )


class ViewportState(QObject):
    """
    Manages synchronized viewport state across all panels.
    Uses signals to notify views of changes.
    """

    # Signals for state changes
    limits_changed = pyqtSignal(ViewportLimits)
    amplitude_range_changed = pyqtSignal(float, float)  # min_amplitude, max_amplitude
    colormap_changed = pyqtSignal(str)  # colormap name
    interpolation_changed = pyqtSignal(str)  # interpolation mode

    def __init__(self):
        super().__init__()
        self._limits = ViewportLimits()
        self._min_amplitude = -1.0
        self._max_amplitude = 1.0
        self._colormap = 'seismic'  # Default colormap
        self._interpolation = 'bilinear'  # Default interpolation mode

    @property
    def limits(self) -> ViewportLimits:
        """Current viewport limits."""
        return self._limits.copy()

    @property
    def min_amplitude(self) -> float:
        """Current minimum amplitude for display."""
        return self._min_amplitude

    @property
    def max_amplitude(self) -> float:
        """Current maximum amplitude for display."""
        return self._max_amplitude

    @property
    def colormap(self) -> str:
        """Current colormap name."""
        return self._colormap

    @property
    def interpolation(self) -> str:
        """Current interpolation mode."""
        return self._interpolation

    def set_limits(self, time_min: float, time_max: float,
                   trace_min: float, trace_max: float):
        """
        Set new viewport limits.

        Args:
            time_min, time_max: Time range in milliseconds
            trace_min, trace_max: Trace range (can be fractional for smooth zoom)
        """
        self._limits = ViewportLimits(
            time_min=time_min,
            time_max=time_max,
            trace_min=trace_min,
            trace_max=trace_max
        )
        self.limits_changed.emit(self._limits.copy())

    def set_amplitude_range(self, min_amp: float, max_amp: float):
        """
        Set amplitude range for display.

        Args:
            min_amp: Minimum amplitude value
            max_amp: Maximum amplitude value
        """
        if max_amp <= min_amp:
            raise ValueError(f"Max amplitude ({max_amp}) must be > min amplitude ({min_amp})")

        self._min_amplitude = min_amp
        self._max_amplitude = max_amp
        self.amplitude_range_changed.emit(min_amp, max_amp)

    def set_colormap(self, colormap: str):
        """
        Set colormap for display.

        Args:
            colormap: Colormap name ('seismic', 'grayscale', 'viridis', etc.)
        """
        self._colormap = colormap
        self.colormap_changed.emit(colormap)

    def set_interpolation(self, interpolation: str):
        """
        Set interpolation mode for display.

        Args:
            interpolation: Interpolation mode ('bilinear', 'bicubic', 'nearest')
        """
        valid_modes = ['bilinear', 'bicubic', 'nearest']
        if interpolation not in valid_modes:
            raise ValueError(f"Invalid interpolation mode: {interpolation}. Must be one of {valid_modes}")

        self._interpolation = interpolation
        self.interpolation_changed.emit(interpolation)

    def zoom_in(self, factor: float = 0.5, center_time: Optional[float] = None,
                center_trace: Optional[float] = None):
        """
        Zoom in by reducing viewport size.

        Args:
            factor: Zoom factor (0.5 = zoom to 50% of current view)
            center_time: Center time for zoom (default: current center)
            center_trace: Center trace for zoom (default: current center)
        """
        if center_time is None:
            center_time = (self._limits.time_min + self._limits.time_max) / 2
        if center_trace is None:
            center_trace = (self._limits.trace_min + self._limits.trace_max) / 2

        time_range = (self._limits.time_max - self._limits.time_min) * factor
        trace_range = (self._limits.trace_max - self._limits.trace_min) * factor

        self.set_limits(
            center_time - time_range / 2,
            center_time + time_range / 2,
            center_trace - trace_range / 2,
            center_trace + trace_range / 2
        )

    def zoom_out(self, factor: float = 2.0):
        """
        Zoom out by increasing viewport size.

        Args:
            factor: Zoom factor (2.0 = zoom to 200% of current view)
        """
        self.zoom_in(factor=factor)

    def pan(self, delta_time: float, delta_trace: float):
        """
        Pan the viewport.

        Args:
            delta_time: Change in time (milliseconds)
            delta_trace: Change in trace number
        """
        self.set_limits(
            self._limits.time_min + delta_time,
            self._limits.time_max + delta_time,
            self._limits.trace_min + delta_trace,
            self._limits.trace_max + delta_trace
        )

    def reset_to_data(self, data_time_max: float, data_trace_max: float):
        """
        Reset viewport to show all data.

        Args:
            data_time_max: Maximum time in data (milliseconds)
            data_trace_max: Maximum trace number in data
        """
        self.set_limits(0.0, data_time_max, 0.0, data_trace_max)
