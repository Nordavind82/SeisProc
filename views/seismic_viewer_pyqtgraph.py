"""
PyQtGraph-based seismic viewer - high-performance visualization with full mouse control.

Features:
- Fast OpenGL-accelerated rendering
- Mouse wheel zoom (2D and axis-specific)
- Box zoom selection
- Mouse pan (drag)
- Synchronized views
- Professional seismic colormap
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView, GraphicsLayoutWidget
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QToolBar
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QPainter
import sys
from models.seismic_data import SeismicData
from models.lazy_seismic_data import LazySeismicData
from models.viewport_state import ViewportState, ViewportLimits
from utils.window_cache import WindowCache


class SeismicViewerPyQtGraph(QWidget):
    """
    High-performance seismic viewer using PyQtGraph.

    Mouse Controls:
    - Mouse Wheel: Zoom in/out (both axes)
    - Ctrl + Mouse Wheel: Zoom X-axis only
    - Shift + Mouse Wheel: Zoom Y-axis only
    - Left Mouse Drag: Pan
    - Right Mouse Drag: Box zoom
    - Middle Mouse: Reset view
    """

    def __init__(self, title: str, viewport_state: ViewportState, parent=None):
        super().__init__(parent)
        self.title = title
        self.viewport_state = viewport_state
        self.data = None

        # Lazy data loading support
        self.lazy_data = None
        self._lazy_data_id = None  # Unique ID for current dataset
        self._cached_window = None  # Cached window data (for backward compat)
        self._cached_bounds = None  # (time_start, time_end, trace_start, trace_end)
        self._hysteresis_threshold = 0.25  # Reload if viewport moves > 25% outside cached window
        self._window_padding = 0.10  # Add 10% padding when loading windows

        # Multi-window cache with LRU eviction
        self._window_cache = WindowCache(max_windows=5, max_memory_mb=500)

        # Create UI
        self._init_ui()

        # Connect to viewport state changes
        self.viewport_state.limits_changed.connect(self._on_limits_changed)
        self.viewport_state.amplitude_range_changed.connect(self._on_amplitude_range_changed)
        self.viewport_state.colormap_changed.connect(self._on_colormap_changed)
        self.viewport_state.interpolation_changed.connect(self._on_interpolation_changed)

        # Track current display parameters
        self._current_min_amp = -1.0
        self._current_max_amp = 1.0
        self._current_colormap = 'seismic'
        self._current_interpolation = 'bilinear'

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Create graphics layout widget
        self.graphics_widget = GraphicsLayoutWidget()
        self.graphics_widget.setBackground('w')  # White background

        # Enable high-quality rendering on the graphics view
        # This enables smooth interpolation for all items in the view
        self.graphics_widget.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )

        # Create plot item with axes
        self.plot_item = self.graphics_widget.addPlot(
            title=self.title,
            labels={'left': 'Time (ms)', 'bottom': 'Trace Number'}
        )

        # Configure plot appearance
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.setMenuEnabled(False)  # Disable context menu (we'll use our own controls)

        # Invert Y-axis (seismic convention: time increases downward)
        self.plot_item.invertY(True)

        # Create image item for displaying seismic data
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)

        # Set up seismic colormap (red-white-blue)
        self._setup_colormap()

        # Configure mouse interaction modes
        self._setup_mouse_controls()

        layout.addWidget(self.graphics_widget)
        self.setLayout(layout)

    def _create_toolbar(self) -> QWidget:
        """Create toolbar with zoom mode controls."""
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 2, 5, 2)

        # Title label
        title_label = QLabel(f"<b>{self.title}</b>")
        toolbar_layout.addWidget(title_label)

        toolbar_layout.addStretch()

        # Zoom mode selector
        toolbar_layout.addWidget(QLabel("Zoom Mode:"))
        self.zoom_mode_combo = QComboBox()
        self.zoom_mode_combo.addItems([
            "Both Axes (2D)",
            "X-Axis Only (Traces)",
            "Y-Axis Only (Time)",
            "Pan Mode",
            "Box Zoom"
        ])
        self.zoom_mode_combo.currentIndexChanged.connect(self._on_zoom_mode_changed)
        toolbar_layout.addWidget(self.zoom_mode_combo)

        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view_local)
        toolbar_layout.addWidget(reset_btn)

        toolbar_widget.setLayout(toolbar_layout)
        return toolbar_widget

    def _setup_colormap(self):
        """Set up initial colormap."""
        self._apply_colormap('seismic')

    def _apply_colormap(self, colormap_name: str):
        """Apply a colormap by name with 256-entry LUT for smooth gradients."""
        # Define colormap key positions and colors
        if colormap_name == 'seismic':
            # Red-White-Blue seismic colormap with enhanced zero-crossing detail
            positions = np.array([0.0, 0.45, 0.50, 0.55, 1.0])
            colors = np.array([
                [0, 0, 255, 255],      # Blue (negative)
                [135, 206, 250, 255],  # Light blue (near zero negative)
                [245, 245, 245, 255],  # Light gray (zero)
                [255, 160, 122, 255],  # Light red (near zero positive)
                [255, 0, 0, 255]       # Red (positive)
            ], dtype=np.float32)

        elif colormap_name == 'grayscale':
            # Grayscale colormap
            positions = np.array([0.0, 1.0])
            colors = np.array([
                [0, 0, 0, 255],      # Black
                [255, 255, 255, 255] # White
            ], dtype=np.float32)

        elif colormap_name == 'viridis':
            # Viridis colormap
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [68, 1, 84, 255],
                [59, 82, 139, 255],
                [33, 145, 140, 255],
                [94, 201, 98, 255],
                [253, 231, 37, 255]
            ], dtype=np.float32)

        elif colormap_name == 'plasma':
            # Plasma colormap
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [13, 8, 135, 255],
                [126, 3, 168, 255],
                [204, 71, 120, 255],
                [248, 149, 64, 255],
                [240, 249, 33, 255]
            ], dtype=np.float32)

        elif colormap_name == 'inferno':
            # Inferno colormap
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [0, 0, 4, 255],
                [87, 16, 110, 255],
                [188, 55, 84, 255],
                [249, 142, 9, 255],
                [252, 255, 164, 255]
            ], dtype=np.float32)

        elif colormap_name == 'jet':
            # Jet colormap
            positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            colors = np.array([
                [0, 0, 127, 255],
                [0, 0, 255, 255],
                [0, 255, 255, 255],
                [255, 255, 0, 255],
                [255, 0, 0, 255]
            ], dtype=np.float32)

        else:
            # Default to seismic
            positions = np.array([0.0, 0.45, 0.50, 0.55, 1.0])
            colors = np.array([
                [0, 0, 255, 255],
                [135, 206, 250, 255],
                [245, 245, 245, 255],
                [255, 160, 122, 255],
                [255, 0, 0, 255]
            ], dtype=np.float32)

        # Generate 256-entry lookup table for smooth gradients
        lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(256):
            # Normalized position [0, 1]
            pos = i / 255.0

            # Find surrounding positions
            idx = np.searchsorted(positions, pos)
            if idx == 0:
                lut[i] = colors[0].astype(np.uint8)
            elif idx >= len(positions):
                lut[i] = colors[-1].astype(np.uint8)
            else:
                # Linear interpolation between positions
                pos_low = positions[idx - 1]
                pos_high = positions[idx]
                color_low = colors[idx - 1]
                color_high = colors[idx]

                # Interpolation factor
                t = (pos - pos_low) / (pos_high - pos_low)

                # Interpolate RGBA
                interpolated = color_low * (1 - t) + color_high * t
                lut[i] = interpolated.astype(np.uint8)

        # Apply lookup table to image item
        self.image_item.setLookupTable(lut)
        self._current_colormap = colormap_name

    def _setup_mouse_controls(self):
        """Configure mouse interaction modes."""
        # Get view box
        self.view_box = self.plot_item.getViewBox()

        # Enable mouse interaction
        self.view_box.setMouseEnabled(x=True, y=True)

        # Set default mouse mode (both axes)
        self.view_box.setMouseMode(pg.ViewBox.RectMode)

        # Connect range change signal for synchronization
        self.view_box.sigRangeChanged.connect(self._on_view_range_changed)

    def _on_zoom_mode_changed(self, index: int):
        """Handle zoom mode change."""
        if index == 0:  # Both axes
            self.view_box.setMouseEnabled(x=True, y=True)
            self.view_box.setMouseMode(pg.ViewBox.RectMode)
        elif index == 1:  # X-axis only
            self.view_box.setMouseEnabled(x=True, y=False)
            self.view_box.setMouseMode(pg.ViewBox.RectMode)
        elif index == 2:  # Y-axis only
            self.view_box.setMouseEnabled(x=False, y=True)
            self.view_box.setMouseMode(pg.ViewBox.RectMode)
        elif index == 3:  # Pan mode
            self.view_box.setMouseEnabled(x=True, y=True)
            self.view_box.setMouseMode(pg.ViewBox.PanMode)
        elif index == 4:  # Box zoom
            self.view_box.setMouseEnabled(x=True, y=True)
            self.view_box.setMouseMode(pg.ViewBox.RectMode)

    def _on_view_range_changed(self):
        """Handle view range change from mouse interaction."""
        # Handle lazy data - load visible window on demand
        if self.lazy_data is not None:
            self._load_visible_window()
            return

        if self.data is None:
            return

        # Get current view range
        view_range = self.view_box.viewRange()
        x_range = view_range[0]  # [min, max] for X (traces)
        y_range = view_range[1]  # [min, max] for Y (time)

        # Convert to our coordinate system
        time_axis = self.data.get_time_axis()
        trace_axis = self.data.get_trace_axis()

        # Map view coordinates to data coordinates
        trace_min = max(0, x_range[0])
        trace_max = min(len(trace_axis) - 1, x_range[1])
        time_min = max(0, y_range[0])
        time_max = min(time_axis[-1], y_range[1])

        # Update viewport state (this will sync other views)
        # Temporarily disconnect to avoid recursion
        self.viewport_state.limits_changed.disconnect(self._on_limits_changed)
        self.viewport_state.set_limits(time_min, time_max, trace_min, trace_max)
        self.viewport_state.limits_changed.connect(self._on_limits_changed)

    def set_data(self, data: SeismicData):
        """
        Set seismic data to display.

        Args:
            data: SeismicData object to visualize
        """
        self.data = data
        self.lazy_data = None
        self._lazy_data_id = None
        self._cached_window = None
        self._cached_bounds = None
        self._window_cache.clear()  # Clear cache when switching to regular data
        self._update_display()

    def set_lazy_data(self, lazy_data: LazySeismicData):
        """
        Set lazy seismic data for memory-efficient loading.

        Args:
            lazy_data: LazySeismicData object providing on-demand data access
        """
        self.lazy_data = lazy_data
        self.data = None
        self._cached_window = None
        self._cached_bounds = None

        # Generate unique ID for this dataset (using id() which gives memory address)
        self._lazy_data_id = id(lazy_data)

        # Clear window cache when new dataset loaded
        self._window_cache.clear()

        # Load initial window
        self._load_visible_window()

    def _load_visible_window(self):
        """
        Load only the visible portion of data with padding for smooth panning.

        This method implements windowed loading with:
        - 10% padding on each side for smooth panning
        - Hysteresis: only reload if viewport moves > 25% outside cached window
        - Caching of loaded window to avoid redundant loads
        """
        if self.lazy_data is None:
            return

        # Get current viewport range
        view_range = self.view_box.viewRange()
        x_range = view_range[0]  # [min, max] for X (traces)
        y_range = view_range[1]  # [min, max] for Y (time)

        # Convert to data coordinates with bounds checking
        trace_min = max(0, int(x_range[0]))
        trace_max = min(self.lazy_data.n_traces, int(x_range[1]) + 1)
        time_min = max(0, y_range[0])
        time_max = min(self.lazy_data.duration, y_range[1])

        # Check if we need to reload (hysteresis)
        if self._cached_bounds is not None:
            cached_time_start, cached_time_end, cached_trace_start, cached_trace_end = self._cached_bounds

            # Calculate how much of the viewport is outside the cached window
            time_range = time_max - time_min
            trace_range = trace_max - trace_min

            time_outside = 0
            if time_min < cached_time_start:
                time_outside = cached_time_start - time_min
            elif time_max > cached_time_end:
                time_outside = time_max - cached_time_end

            trace_outside = 0
            if trace_min < cached_trace_start:
                trace_outside = cached_trace_start - trace_min
            elif trace_max > cached_trace_end:
                trace_outside = trace_max - cached_trace_end

            # Check if we're still within hysteresis threshold
            time_ratio = time_outside / time_range if time_range > 0 else 0
            trace_ratio = trace_outside / trace_range if trace_range > 0 else 0

            if time_ratio < self._hysteresis_threshold and trace_ratio < self._hysteresis_threshold:
                # Still within cached window, no reload needed
                self._update_lazy_display()
                return

        # Calculate padded bounds for loading
        time_range = time_max - time_min
        trace_range = trace_max - trace_min

        time_padding = time_range * self._window_padding
        trace_padding = int(trace_range * self._window_padding)

        load_time_start = max(0, time_min - time_padding)
        load_time_end = min(self.lazy_data.duration, time_max + time_padding)
        load_trace_start = max(0, trace_min - trace_padding)
        load_trace_end = min(self.lazy_data.n_traces, trace_max + trace_padding)

        # Create cache key (includes dataset ID to handle multiple datasets)
        cache_key = (self._lazy_data_id, load_time_start, load_time_end, load_trace_start, load_trace_end)

        # Check cache first
        window_data = self._window_cache.get(cache_key)

        if window_data is None:
            # Cache miss - load from Zarr
            window_data = self.lazy_data.get_window(
                load_time_start, load_time_end,
                load_trace_start, load_trace_end
            )

            # Add to cache
            self._window_cache.put(cache_key, window_data)

        # Cache the loaded window (for backward compat with single-window logic)
        self._cached_window = window_data
        self._cached_bounds = (load_time_start, load_time_end, load_trace_start, load_trace_end)

        # Update display
        self._update_lazy_display()

    def _update_lazy_display(self):
        """Update display using cached window data."""
        if self._cached_window is None or self._cached_bounds is None:
            return

        # Get amplitude range
        min_amp = self.viewport_state.min_amplitude
        max_amp = self.viewport_state.max_amplitude

        # Update image with cached window
        self.image_item.setImage(
            self._cached_window.T,  # Transpose for correct orientation
            autoLevels=False,
            levels=(min_amp, max_amp)
        )

        # Set the position and scale of the image to match the cached bounds
        cached_time_start, cached_time_end, cached_trace_start, cached_trace_end = self._cached_bounds

        # Calculate position (top-left corner) and scale
        self.image_item.setRect(
            cached_trace_start,  # X position
            cached_time_start,   # Y position
            cached_trace_end - cached_trace_start,  # Width
            cached_time_end - cached_time_start    # Height
        )

        # Apply interpolation mode
        if self._current_interpolation == 'nearest':
            self.graphics_widget.setRenderHints(
                QPainter.RenderHint.Antialiasing |
                QPainter.RenderHint.TextAntialiasing
            )
        else:
            self.graphics_widget.setRenderHints(
                QPainter.RenderHint.Antialiasing |
                QPainter.RenderHint.SmoothPixmapTransform |
                QPainter.RenderHint.TextAntialiasing
            )

    def _update_display(self):
        """Update the display with current data and viewport state."""
        if self.data is None:
            return

        # Get current viewport limits and amplitude range
        limits = self.viewport_state.limits
        min_amp = self.viewport_state.min_amplitude
        max_amp = self.viewport_state.max_amplitude

        # Data without modification
        display_data = self.data.traces

        # Update image
        # PyQtGraph expects data as (width, height), so transpose
        self.image_item.setImage(
            display_data.T,  # Transpose for correct orientation
            autoLevels=False,
            levels=(min_amp, max_amp)
        )

        # Apply interpolation mode by toggling view's rendering hints
        # This controls how Qt's graphics system renders the image
        if self._current_interpolation == 'nearest':
            # Nearest neighbor - sharp, blocky pixels (disable smooth transform)
            self.graphics_widget.setRenderHints(
                QPainter.RenderHint.Antialiasing |
                QPainter.RenderHint.TextAntialiasing
            )
        else:
            # Bilinear or Bicubic - smooth interpolation
            # Qt doesn't distinguish between bilinear/bicubic, uses high-quality transform
            self.graphics_widget.setRenderHints(
                QPainter.RenderHint.Antialiasing |
                QPainter.RenderHint.SmoothPixmapTransform |
                QPainter.RenderHint.TextAntialiasing
            )

            # For "very smooth" mode, also enable high-quality antialiasing
            if self._current_interpolation == 'bicubic':
                self.graphics_widget.setRenderHint(
                    QPainter.RenderHint.HighQualityAntialiasing, True
                )

        # Set the coordinate system transform
        # Map image coordinates to real coordinates (time, trace)
        time_axis = self.data.get_time_axis()
        trace_axis = self.data.get_trace_axis()

        # Set transform: scale and translate to match data coordinates
        self.image_item.setRect(
            trace_axis[0],           # x (trace min)
            time_axis[0],            # y (time min)
            trace_axis[-1] - trace_axis[0],  # width (trace range)
            time_axis[-1] - time_axis[0]     # height (time range)
        )

        # Set view limits
        self.view_box.setRange(
            xRange=(limits.trace_min, limits.trace_max),
            yRange=(limits.time_min, limits.time_max),
            padding=0
        )

        # Store current parameters
        self._current_min_amp = min_amp
        self._current_max_amp = max_amp

    def _on_limits_changed(self, limits: ViewportLimits):
        """Handle viewport limits change from external source."""
        if self.data is not None:
            # Update view without triggering range change signal
            self.view_box.blockSignals(True)
            self.view_box.setRange(
                xRange=(limits.trace_min, limits.trace_max),
                yRange=(limits.time_min, limits.time_max),
                padding=0
            )
            self.view_box.blockSignals(False)

    def _on_amplitude_range_changed(self, min_amp: float, max_amp: float):
        """Handle amplitude range change."""
        if min_amp != self._current_min_amp or max_amp != self._current_max_amp:
            if self.lazy_data is not None:
                self._update_lazy_display()
            elif self.data is not None:
                self._update_display()

    def _on_colormap_changed(self, colormap: str):
        """Handle colormap change."""
        if colormap != self._current_colormap:
            self._apply_colormap(colormap)
            if self.lazy_data is not None:
                self._update_lazy_display()
            elif self.data is not None:
                self._update_display()

    def _on_interpolation_changed(self, interpolation: str):
        """Handle interpolation mode change."""
        if interpolation != self._current_interpolation:
            self._current_interpolation = interpolation
            if self.lazy_data is not None:
                self._update_lazy_display()
            elif self.data is not None:
                self._update_display()

    def _reset_view_local(self):
        """Reset view to show all data."""
        if self.lazy_data is not None:
            self.viewport_state.set_limits(
                0, self.lazy_data.duration,
                0, self.lazy_data.n_traces - 1
            )
        elif self.data is not None:
            time_axis = self.data.get_time_axis()
            trace_axis = self.data.get_trace_axis()
            self.viewport_state.set_limits(
                time_axis[0], time_axis[-1],
                trace_axis[0], trace_axis[-1]
            )

    def clear(self):
        """Clear the display."""
        self.data = None
        self.lazy_data = None
        self._lazy_data_id = None
        self._cached_window = None
        self._cached_bounds = None
        self._window_cache.clear()
        self.image_item.clear()

    def get_cache_stats(self):
        """
        Get window cache statistics.

        Returns:
            Dictionary with cache statistics (hits, misses, hit_rate, memory_usage, etc.)
            Returns None if lazy data is not being used
        """
        if self.lazy_data is None:
            return None

        return self._window_cache.get_stats()

    def wheelEvent(self, event):
        """
        Handle mouse wheel for zooming.

        - Wheel only: Zoom both axes
        - Ctrl + Wheel: Zoom X-axis only
        - Shift + Wheel: Zoom Y-axis only
        """
        modifiers = event.modifiers()

        if modifiers == Qt.KeyboardModifier.ControlModifier:
            # X-axis zoom only
            self.zoom_mode_combo.setCurrentIndex(1)
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            # Y-axis zoom only
            self.zoom_mode_combo.setCurrentIndex(2)
        else:
            # Both axes zoom
            self.zoom_mode_combo.setCurrentIndex(0)

        # Let the widget handle the actual zoom
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for custom interactions."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Middle click: reset view
            self._reset_view_local()
            event.accept()
        else:
            super().mousePressEvent(event)
