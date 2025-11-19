"""
Flip window - single viewer that cycles through Input/Processed/Difference on mouse clicks.

LMB (Left Mouse Button): Input → Processed → Difference → Input (forward)
RMB (Right Mouse Button): Input → Difference → Processed → Input (backward)
"""
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QStatusBar, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import sys
from models.seismic_data import SeismicData
from models.viewport_state import ViewportState
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph


class FlipWindow(QMainWindow):
    """
    Flip window that cycles through different data views on mouse clicks.

    Features:
    - Single seismic viewer
    - LMB: Forward cycle (Input → Processed → Difference)
    - RMB: Backward cycle (Difference → Processed → Input)
    - Synchronized zoom/pan with main window
    - Status bar shows current view
    """

    def __init__(self, viewport_state: ViewportState, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flip View")
        self.setGeometry(150, 150, 1000, 800)

        # Shared viewport state for synchronized views
        self.viewport_state = viewport_state

        # Data sources (will be updated from main window)
        self.input_data = None
        self.processed_data = None
        self.difference_data = None

        # Current view state
        self.view_modes = ['Input', 'Processed', 'Difference']
        self.current_view_index = 0  # Start with Input

        # Create UI
        self._init_ui()

        # Connect to viewport state colormap changes
        self.viewport_state.colormap_changed.connect(self._on_viewport_colormap_changed)

        # Set initial colormap from viewport state
        self._sync_colormap_from_viewport()

        # Show initial message
        self.statusBar().showMessage("Click LMB to cycle forward, RMB to cycle backward")

    def _init_ui(self):
        """Initialize user interface."""
        # Central widget
        central = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Top bar with view label and colormap selector
        top_bar = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(5, 5, 5, 5)

        # Current view label
        self.view_label = QLabel()
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        self.view_label.setFont(font)
        self.view_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        top_layout.addWidget(self.view_label, stretch=1)

        # Colormap selector
        colormap_label = QLabel("Colormap:")
        colormap_label.setStyleSheet("font-weight: bold; padding: 5px;")
        top_layout.addWidget(colormap_label)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "Seismic (RWB)",
            "Grayscale",
            "Viridis",
            "Plasma",
            "Inferno",
            "Jet"
        ])
        self.colormap_combo.setToolTip("Select colormap (synchronized with main window)")
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        top_layout.addWidget(self.colormap_combo)

        top_bar.setLayout(top_layout)
        layout.addWidget(top_bar)

        # Single seismic viewer
        self.viewer = SeismicViewerPyQtGraph("Flip View", self.viewport_state)

        # Enable mouse click events on the viewer
        self.viewer.mousePressEvent = self._on_viewer_click

        layout.addWidget(self.viewer)

        central.setLayout(layout)
        self.setCentralWidget(central)

        # Update label
        self._update_view_label()

    def set_data(self, input_data: SeismicData, processed_data: SeismicData,
                 difference_data: SeismicData):
        """
        Set data sources for flipping.

        Args:
            input_data: Input seismic data
            processed_data: Processed seismic data
            difference_data: Difference seismic data
        """
        self.input_data = input_data
        self.processed_data = processed_data
        self.difference_data = difference_data

        # Display current view
        self._update_display()

    def _on_viewer_click(self, event):
        """Handle mouse clicks on viewer for flipping."""
        if event.button() == Qt.MouseButton.LeftButton:
            # LMB: Forward cycle (Input → Processed → Difference → Input)
            self._cycle_forward()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            # RMB: Backward cycle (Difference → Processed → Input → Difference)
            self._cycle_backward()
            event.accept()
        else:
            # Let parent handle other buttons
            super(SeismicViewerPyQtGraph, self.viewer).mousePressEvent(event)

    def _cycle_forward(self):
        """Cycle to next view (Input → Processed → Difference → Input)."""
        self.current_view_index = (self.current_view_index + 1) % len(self.view_modes)
        self._update_display()
        self._update_view_label()

        # Status message
        self.statusBar().showMessage(
            f"Cycled forward to: {self.view_modes[self.current_view_index]}",
            2000
        )

    def _cycle_backward(self):
        """Cycle to previous view (Difference → Processed → Input → Difference)."""
        self.current_view_index = (self.current_view_index - 1) % len(self.view_modes)
        self._update_display()
        self._update_view_label()

        # Status message
        self.statusBar().showMessage(
            f"Cycled backward to: {self.view_modes[self.current_view_index]}",
            2000
        )

    def _update_display(self):
        """Update viewer with current data."""
        current_view = self.view_modes[self.current_view_index]

        if current_view == 'Input' and self.input_data is not None:
            self.viewer.set_data(self.input_data)
        elif current_view == 'Processed' and self.processed_data is not None:
            self.viewer.set_data(self.processed_data)
        elif current_view == 'Difference' and self.difference_data is not None:
            self.viewer.set_data(self.difference_data)
        else:
            # No data available for this view
            self.viewer.clear()

    def _update_view_label(self):
        """Update the view label to show current mode."""
        current_view = self.view_modes[self.current_view_index]

        # Color coding
        if current_view == 'Input':
            color = "#2196F3"  # Blue
        elif current_view == 'Processed':
            color = "#4CAF50"  # Green
        else:  # Difference
            color = "#FF9800"  # Orange

        self.view_label.setText(f"📊 Current View: {current_view}")
        self.view_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin: 5px;
            }}
        """)

    def clear_data(self):
        """Clear all data."""
        self.input_data = None
        self.processed_data = None
        self.difference_data = None
        self.viewer.clear()
        self.current_view_index = 0
        self._update_view_label()

    def _on_colormap_changed(self, index: int):
        """Handle colormap selection change in flip window."""
        colormap_map = {
            0: 'seismic',
            1: 'grayscale',
            2: 'viridis',
            3: 'plasma',
            4: 'inferno',
            5: 'jet'
        }
        colormap = colormap_map.get(index, 'seismic')

        # Update viewport state (this will sync with main window)
        self.viewport_state.set_colormap(colormap)

    def _on_viewport_colormap_changed(self, colormap: str):
        """Handle colormap change from viewport state (main window)."""
        # Update combo box to match (without triggering signal)
        self._sync_colormap_from_viewport()

    def _sync_colormap_from_viewport(self):
        """Sync colormap combo box with viewport state."""
        colormap = self.viewport_state.colormap

        index_map = {
            'seismic': 0,
            'grayscale': 1,
            'viridis': 2,
            'plasma': 3,
            'inferno': 4,
            'jet': 5
        }

        index = index_map.get(colormap, 0)

        # Block signals to avoid loop
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentIndex(index)
        self.colormap_combo.blockSignals(False)
