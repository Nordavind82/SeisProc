"""
Gather navigation panel - UI controls for navigating through seismic gathers.
"""
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                              QLabel, QSpinBox, QGroupBox, QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import sys
from models.gather_navigator import GatherNavigator


class GatherNavigationPanel(QWidget):
    """
    Navigation panel for multi-gather seismic data.

    Provides:
    - Previous/Next buttons
    - Jump to specific gather
    - Current gather information display
    - Progress indicator
    """

    # Signals
    gather_navigation_requested = pyqtSignal(str)  # 'prev', 'next', or 'goto'
    auto_process_changed = pyqtSignal(bool)  # auto-process enabled/disabled

    def __init__(self, gather_navigator: GatherNavigator, parent=None):
        super().__init__(parent)
        self.gather_navigator = gather_navigator

        # Connect to navigator signals
        self.gather_navigator.gather_changed.connect(self._on_gather_changed)
        self.gather_navigator.navigation_state_changed.connect(self._on_navigation_state_changed)

        self._init_ui()
        self._update_display()

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Gather info display
        info_group = QGroupBox("Current Gather")
        info_layout = QVBoxLayout()

        # Gather description
        self.gather_desc_label = QLabel("No data loaded")
        self.gather_desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.gather_desc_label.setFont(font)
        info_layout.addWidget(self.gather_desc_label)

        # Gather details
        self.gather_details_label = QLabel("")
        self.gather_details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gather_details_label.setStyleSheet("color: #666;")
        info_layout.addWidget(self.gather_details_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()

        # Auto-process checkbox
        self.auto_process_checkbox = QCheckBox("Auto-apply processing on navigate")
        self.auto_process_checkbox.setToolTip(
            "When enabled, automatically applies the last processing\n"
            "when navigating to next/previous gather"
        )
        self.auto_process_checkbox.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #2E7D32;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.auto_process_checkbox.stateChanged.connect(self._on_auto_process_changed)
        nav_layout.addWidget(self.auto_process_checkbox)

        # Previous/Next buttons row
        button_row = QHBoxLayout()

        self.prev_button = QPushButton("◄ Previous")
        self.prev_button.clicked.connect(self._on_previous_clicked)
        self.prev_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:disabled {
                color: #999;
            }
        """)
        button_row.addWidget(self.prev_button)

        self.next_button = QPushButton("Next ►")
        self.next_button.clicked.connect(self._on_next_clicked)
        self.next_button.setStyleSheet("""
            QPushButton {
                padding: 8px 15px;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton:disabled {
                color: #999;
            }
        """)
        button_row.addWidget(self.next_button)

        nav_layout.addLayout(button_row)

        # Jump to gather row
        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Go to gather:"))

        self.gather_spinbox = QSpinBox()
        self.gather_spinbox.setMinimum(1)
        self.gather_spinbox.setMaximum(1)
        self.gather_spinbox.setValue(1)
        self.gather_spinbox.valueChanged.connect(self._on_goto_requested)
        jump_row.addWidget(self.gather_spinbox)

        self.total_gathers_label = QLabel("/ 1")
        jump_row.addWidget(self.total_gathers_label)

        jump_row.addStretch()

        nav_layout.addLayout(jump_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        nav_layout.addWidget(self.progress_bar)

        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("No data loaded")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #444;")
        stats_layout.addWidget(self.stats_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()

        self.setLayout(layout)
        self.setMaximumWidth(280)

    def _on_previous_clicked(self):
        """Handle previous button click."""
        if self.gather_navigator.previous_gather():
            self.gather_navigation_requested.emit('prev')

    def _on_next_clicked(self):
        """Handle next button click."""
        if self.gather_navigator.next_gather():
            self.gather_navigation_requested.emit('next')

    def _on_goto_requested(self, value: int):
        """Handle goto spinbox change."""
        # Convert from 1-based to 0-based
        gather_id = value - 1
        if self.gather_navigator.goto_gather(gather_id):
            self.gather_navigation_requested.emit('goto')

    def _on_auto_process_changed(self, state: int):
        """Handle auto-process checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.auto_process_changed.emit(enabled)

    def _on_gather_changed(self, gather_id: int, gather_info: dict):
        """Handle gather change from navigator."""
        # Update spinbox without triggering signal
        self.gather_spinbox.blockSignals(True)
        self.gather_spinbox.setValue(gather_id + 1)  # 1-based display
        self.gather_spinbox.blockSignals(False)

        # Update gather info display
        self.gather_desc_label.setText(gather_info['description'])

        # Update details
        details = f"Gather {gather_id + 1} of {self.gather_navigator.n_gathers}"
        if 'n_traces' in gather_info:
            details += f" • {gather_info['n_traces']} traces"
        if 'start_trace' in gather_info:
            details += f" • Traces {gather_info['start_trace']}-{gather_info['end_trace']}"

        self.gather_details_label.setText(details)

        # Update progress bar
        self.progress_bar.setMaximum(self.gather_navigator.n_gathers)
        self.progress_bar.setValue(gather_id + 1)

    def _on_navigation_state_changed(self, can_prev: bool, can_next: bool):
        """Handle navigation state change."""
        self.prev_button.setEnabled(can_prev)
        self.next_button.setEnabled(can_next)

    def _update_display(self):
        """Update display with current state."""
        if not self.gather_navigator.has_gathers():
            self._set_disabled_state()
            return

        # Update spinbox range
        self.gather_spinbox.setMaximum(self.gather_navigator.n_gathers)
        self.total_gathers_label.setText(f"/ {self.gather_navigator.n_gathers}")

        # Update statistics
        stats = self.gather_navigator.get_statistics()
        stats_text = self._format_statistics(stats)
        self.stats_label.setText(stats_text)

        # Enable controls
        self.setEnabled(True)

    def _set_disabled_state(self):
        """Set UI to disabled state for single gather mode."""
        self.gather_desc_label.setText("Single gather mode")
        self.gather_details_label.setText("All traces displayed")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.gather_spinbox.setEnabled(False)
        self.stats_label.setText("No ensemble information available")

    def _format_statistics(self, stats: dict) -> str:
        """Format statistics for display."""
        if stats['mode'] == 'single':
            return f"Total traces: {stats['total_traces']}"

        lines = [
            f"<b>Total gathers:</b> {stats['n_gathers']}",
            f"<b>Total traces:</b> {stats['total_traces']}",
        ]

        if 'ensemble_keys' in stats and stats['ensemble_keys']:
            keys_str = ", ".join(stats['ensemble_keys'])
            lines.append(f"<b>Ensemble keys:</b> {keys_str}")

        if 'traces_per_gather' in stats:
            tpg = stats['traces_per_gather']
            lines.append(
                f"<b>Traces/gather:</b> {tpg['min']}-{tpg['max']} "
                f"(avg: {tpg['mean']:.1f})"
            )

        return "<br>".join(lines)

    def update_statistics(self):
        """Refresh statistics display."""
        if self.gather_navigator.has_gathers():
            stats = self.gather_navigator.get_statistics()
            stats_text = self._format_statistics(stats)
            self.stats_label.setText(stats_text)

    def is_auto_process_enabled(self) -> bool:
        """Check if auto-processing is enabled."""
        return self.auto_process_checkbox.isChecked()

    def set_auto_process_enabled(self, enabled: bool):
        """Set auto-processing enabled state."""
        self.auto_process_checkbox.setChecked(enabled)
