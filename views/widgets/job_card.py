"""
Job Card Widget

Displays a single job with detailed progress statistics and action buttons.
Shows gathers, traces, workers, throughput, and ETA - all statistics
previously shown in the legacy QProgressDialog.

Features:
- UI throttling (500ms) to prevent blinking
- EMA smoothing for rate display
- Compute kernel indicator (Metal/CPU)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import time


class JobCardWidget(QFrame):
    """
    Card widget displaying a single job with full status and controls.

    Shows all processing statistics:
    - Phase
    - Gathers: current / total
    - Traces: current / total
    - Workers: active count
    - Rate: traces/sec
    - ETA: remaining time

    Signals
    -------
    cancel_requested : UUID
        Emitted when cancel button clicked
    pause_requested : UUID
        Emitted when pause button clicked
    resume_requested : UUID
        Emitted when resume button clicked
    details_requested : UUID
        Emitted when details button clicked
    """

    cancel_requested = pyqtSignal(object)  # UUID
    pause_requested = pyqtSignal(object)  # UUID
    resume_requested = pyqtSignal(object)  # UUID
    details_requested = pyqtSignal(object)  # UUID

    # State colors
    STATE_COLORS = {
        "CREATED": "#6c757d",    # Gray
        "QUEUED": "#17a2b8",     # Cyan
        "RUNNING": "#007bff",    # Blue
        "PAUSED": "#ffc107",     # Yellow
        "CANCELLING": "#fd7e14", # Orange
        "CANCELLED": "#6c757d",  # Gray
        "COMPLETED": "#28a745",  # Green
        "FAILED": "#dc3545",     # Red
        "TIMEOUT": "#dc3545",    # Red
    }

    # Throttling constants
    UI_UPDATE_INTERVAL_MS = 250  # Minimum ms between UI updates (was 500ms)

    def __init__(self, job_id: UUID, name: str, job_type: str, parent=None):
        super().__init__(parent)
        self._job_id = job_id
        self._name = name
        self._job_type = job_type
        self._state = "CREATED"
        self._progress = 0.0
        self._message = ""
        self._started_at: Optional[datetime] = None
        self._eta_seconds: Optional[float] = None

        # Detailed statistics
        self._current_gathers = 0
        self._total_gathers = 0
        self._current_traces = 0
        self._total_traces = 0
        self._active_workers = 0
        self._initial_workers = 0  # Track initial worker count
        self._traces_per_sec = 0.0
        self._phase = ""
        self._compute_kernel = ""  # Metal, CPU, etc.
        self._rate_trend = "stable"

        # I/O metrics
        self._io_rate_mbps = 0.0
        self._io_rate_trend = ""
        self._io_stall_detected = False

        # UI throttling state
        self._last_ui_update = 0.0
        self._pending_update = False
        self._pending_data: Dict[str, Any] = {}

        # EMA smoothing for rate (alpha = 0.2 for smooth display)
        self._smoothed_rate = 0.0
        self._ema_alpha = 0.2

        # Throttle timer for deferred updates
        self._throttle_timer = QTimer(self)
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.timeout.connect(self._apply_pending_update)

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Set up the widget UI with detailed statistics."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(160)
        self.setMaximumHeight(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Top row: name, type, state
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Job name (bold, bright white)
        self._name_label = QLabel(self._name)
        font = self._name_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self._name_label.setFont(font)
        self._name_label.setStyleSheet("color: #ffffff;")  # Bright white
        top_row.addWidget(self._name_label)

        # Job type (light grey)
        self._type_label = QLabel(f"({self._job_type})")
        self._type_label.setStyleSheet("color: #aaa;")  # Light grey
        top_row.addWidget(self._type_label)

        top_row.addStretch()

        # State badge
        self._state_label = QLabel(self._state)
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_label.setMinimumWidth(80)
        top_row.addWidget(self._state_label)

        layout.addLayout(top_row)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(20)
        self._progress_bar.setFormat("%p%")
        layout.addWidget(self._progress_bar)

        # Statistics grid (2 rows x 3 columns)
        stats_grid = QGridLayout()
        stats_grid.setSpacing(12)
        stats_grid.setContentsMargins(0, 4, 0, 4)

        # Create stat labels with bright, visible colors for dark backgrounds
        label_style = "font-size: 11px; color: #aaa;"  # Light grey for labels
        value_style = "font-size: 11px; font-weight: bold; color: #4ade80;"  # Bright green for values

        # Row 1: Phase, Gathers, Traces
        stats_grid.addWidget(self._create_stat_label("Phase:", label_style), 0, 0)
        self._phase_value = QLabel("Initializing...")
        self._phase_value.setStyleSheet("font-size: 11px; font-weight: bold; color: #fbbf24;")  # Amber for phase
        stats_grid.addWidget(self._phase_value, 0, 1)

        stats_grid.addWidget(self._create_stat_label("Gathers:", label_style), 0, 2)
        self._gathers_value = QLabel("0 / 0")
        self._gathers_value.setStyleSheet(value_style)  # Green
        stats_grid.addWidget(self._gathers_value, 0, 3)

        stats_grid.addWidget(self._create_stat_label("Traces:", label_style), 0, 4)
        self._traces_value = QLabel("0 / 0")
        self._traces_value.setStyleSheet(value_style)  # Green
        stats_grid.addWidget(self._traces_value, 0, 5)

        # Row 2: Workers, Rate (with trend), ETA
        stats_grid.addWidget(self._create_stat_label("Workers:", label_style), 1, 0)
        self._workers_value = QLabel("0 active")
        self._workers_value.setStyleSheet("font-size: 11px; font-weight: bold; color: #60a5fa;")  # Blue
        stats_grid.addWidget(self._workers_value, 1, 1)

        stats_grid.addWidget(self._create_stat_label("Rate:", label_style), 1, 2)
        # Rate display with trend indicator container
        rate_container = QHBoxLayout()
        rate_container.setSpacing(4)
        rate_container.setContentsMargins(0, 0, 0, 0)
        self._rate_value = QLabel("0 traces/s")
        self._rate_value.setStyleSheet("font-size: 11px; font-weight: bold; color: #4ade80;")  # Bright green
        rate_container.addWidget(self._rate_value)
        self._rate_trend_label = QLabel("")  # Trend indicator (▲ ▼ ─)
        self._rate_trend_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        rate_container.addWidget(self._rate_trend_label)
        rate_container.addStretch()
        rate_widget = QWidget()
        rate_widget.setLayout(rate_container)
        stats_grid.addWidget(rate_widget, 1, 3)

        stats_grid.addWidget(self._create_stat_label("ETA:", label_style), 1, 4)
        self._eta_value = QLabel("calculating...")
        self._eta_value.setStyleSheet("font-size: 11px; font-weight: bold; color: #22d3ee;")  # Cyan
        stats_grid.addWidget(self._eta_value, 1, 5)

        # Row 3: I/O Rate, Kernel info
        stats_grid.addWidget(self._create_stat_label("I/O:", label_style), 2, 0)
        # I/O rate with trend indicator
        io_container = QHBoxLayout()
        io_container.setSpacing(4)
        io_container.setContentsMargins(0, 0, 0, 0)
        self._io_rate_value = QLabel("--")
        self._io_rate_value.setStyleSheet("font-size: 11px; font-weight: bold; color: #a78bfa;")  # Purple
        io_container.addWidget(self._io_rate_value)
        self._io_trend_label = QLabel("")  # I/O trend indicator
        self._io_trend_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        io_container.addWidget(self._io_trend_label)
        self._io_stall_label = QLabel("")  # Stall warning
        self._io_stall_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #f87171;")
        io_container.addWidget(self._io_stall_label)
        io_container.addStretch()
        io_widget = QWidget()
        io_widget.setLayout(io_container)
        stats_grid.addWidget(io_widget, 2, 1)

        stats_grid.addWidget(self._create_stat_label("Kernel:", label_style), 2, 2)
        self._kernel_value = QLabel("detecting...")
        self._kernel_value.setStyleSheet("font-size: 11px; font-weight: bold; color: #f472b6;")  # Pink
        stats_grid.addWidget(self._kernel_value, 2, 3, 1, 3)  # Span 3 columns

        layout.addLayout(stats_grid)

        # Bottom row: action buttons
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        bottom_row.addStretch()

        # Action buttons
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedWidth(70)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        bottom_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(70)
        self._cancel_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; border: none; border-radius: 4px; padding: 4px; }"
            "QPushButton:hover { background-color: #c82333; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        bottom_row.addWidget(self._cancel_btn)

        layout.addLayout(bottom_row)

    def _create_stat_label(self, text: str, style: str) -> QLabel:
        """Create a styled statistic label."""
        label = QLabel(text)
        label.setStyleSheet(style)
        return label

    def _apply_style(self):
        """Apply styling based on current state."""
        color = self.STATE_COLORS.get(self._state, "#6c757d")
        self._state_label.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"border-radius: 4px; padding: 2px 8px; font-size: 11px;"
        )

        # Update button states
        is_running = self._state == "RUNNING"
        is_paused = self._state == "PAUSED"
        can_cancel = self._state in ("CREATED", "QUEUED", "RUNNING", "PAUSED")

        self._pause_btn.setEnabled(is_running or is_paused)
        self._pause_btn.setText("Resume" if is_paused else "Pause")
        self._cancel_btn.setEnabled(can_cancel)

        # Progress bar color
        if self._state == "COMPLETED":
            self._progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #28a745; }"
            )
        elif self._state in ("FAILED", "TIMEOUT"):
            self._progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #dc3545; }"
            )
        elif self._state == "CANCELLED":
            self._progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #6c757d; }"
            )
        else:
            self._progress_bar.setStyleSheet("")

    @property
    def job_id(self) -> UUID:
        """Get job ID."""
        return self._job_id

    def update_state(self, state: str):
        """Update job state."""
        self._state = state
        self._state_label.setText(state)
        self._apply_style()

    def update_progress(
        self,
        percent: float,
        message: str = "",
        eta_seconds: Optional[float] = None,
        **kwargs
    ):
        """
        Update job progress with detailed statistics.

        Uses UI throttling (500ms) to prevent blinking and EMA smoothing
        for rate/ETA to provide stable, readable values.

        Parameters
        ----------
        percent : float
            Overall progress percentage (0-100)
        message : str
            Status message (phase)
        eta_seconds : float, optional
            Estimated time remaining in seconds
        **kwargs : dict
            Additional statistics:
            - current_gathers, total_gathers
            - current_traces, total_traces
            - active_workers
            - traces_per_sec
            - phase
            - compute_kernel
        """
        # Store the latest data
        self._pending_data = {
            'percent': percent,
            'message': message,
            'eta_seconds': eta_seconds,
            **kwargs
        }

        # Check if we should update now or defer
        now = time.time() * 1000  # ms
        time_since_last = now - self._last_ui_update

        if time_since_last >= self.UI_UPDATE_INTERVAL_MS:
            # Enough time has passed, update immediately
            self._apply_pending_update()
        else:
            # Schedule deferred update if not already scheduled
            if not self._throttle_timer.isActive():
                remaining = int(self.UI_UPDATE_INTERVAL_MS - time_since_last)
                self._throttle_timer.start(max(50, remaining))

    def _apply_pending_update(self):
        """Apply the pending update to the UI (called by timer or directly)."""
        if not self._pending_data:
            return

        self._last_ui_update = time.time() * 1000
        data = self._pending_data

        percent = data.get('percent', self._progress)
        message = data.get('message', self._message)
        eta_seconds = data.get('eta_seconds', self._eta_seconds)

        self._progress = percent
        self._message = message
        self._eta_seconds = eta_seconds

        # Update progress bar
        self._progress_bar.setValue(int(percent))

        # Update phase
        phase = data.get('phase', message)
        if phase:
            self._phase = phase
            self._phase_value.setText(phase)

        # Update gathers - only update if values are provided (not None)
        current_gathers = data.get('current_gathers')
        total_gathers = data.get('total_gathers')
        if current_gathers is not None or total_gathers is not None:
            # Use provided value or fall back to current
            self._current_gathers = current_gathers if current_gathers is not None else self._current_gathers
            self._total_gathers = total_gathers if total_gathers is not None else self._total_gathers
            self._gathers_value.setText(f"{self._current_gathers:,} / {self._total_gathers:,}")

        # Update traces - only update if values are provided (not None)
        current_traces = data.get('current_traces')
        total_traces = data.get('total_traces')
        if current_traces is not None or total_traces is not None:
            # Use provided value or fall back to current
            self._current_traces = current_traces if current_traces is not None else self._current_traces
            self._total_traces = total_traces if total_traces is not None else self._total_traces
            self._traces_value.setText(f"{self._current_traces:,} / {self._total_traces:,}")

        # Update workers with initial count tracking
        active_workers = data.get('active_workers', self._active_workers)
        if active_workers is not None:
            # Track initial worker count
            if self._initial_workers == 0 and active_workers > 0:
                self._initial_workers = active_workers
            self._active_workers = active_workers

            # Show workers with context if some have completed
            if self._initial_workers > 0 and active_workers < self._initial_workers:
                completed_workers = self._initial_workers - active_workers
                self._workers_value.setText(f"{active_workers}/{self._initial_workers}")
                self._workers_value.setToolTip(
                    f"{active_workers} workers active ({completed_workers} completed their segments)\n"
                    "Workers complete at different times due to segment size variations."
                )
            else:
                self._workers_value.setText(f"{active_workers} active")
                self._workers_value.setToolTip("Active parallel workers processing data")

        # Update rate with EMA smoothing
        raw_rate = data.get('traces_per_sec', 0)
        if raw_rate is not None and raw_rate > 0:
            if self._smoothed_rate == 0:
                self._smoothed_rate = raw_rate  # Initialize
            else:
                # Exponential moving average
                self._smoothed_rate = (
                    self._ema_alpha * raw_rate +
                    (1 - self._ema_alpha) * self._smoothed_rate
                )
            self._traces_per_sec = self._smoothed_rate
            self._rate_value.setText(f"{self._smoothed_rate:,.0f} traces/s")

        # Update rate trend indicator with helpful tooltip
        rate_trend = data.get('rate_trend', '')
        if rate_trend:
            self._rate_trend = rate_trend
            if rate_trend == "increasing":
                self._rate_trend_label.setText("▲")
                self._rate_trend_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #4ade80;")  # Green
                self._rate_trend_label.setToolTip("Rate increasing - good performance")
            elif rate_trend == "decreasing":
                self._rate_trend_label.setText("▼")
                self._rate_trend_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #f87171;")  # Red
                self._rate_trend_label.setToolTip(
                    "Rate decreasing - normal as workers complete their segments\n"
                    "Performance scales with active worker count."
                )
            else:  # stable
                self._rate_trend_label.setText("─")
                self._rate_trend_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #9ca3af;")  # Gray
                self._rate_trend_label.setToolTip("Rate stable")

        # Update ETA (use smoothed rate for more stable estimate)
        if eta_seconds is not None and eta_seconds > 0 and eta_seconds < float('inf'):
            # Apply light smoothing to ETA as well
            if self._eta_seconds and self._eta_seconds > 0:
                smoothed_eta = 0.3 * eta_seconds + 0.7 * self._eta_seconds
            else:
                smoothed_eta = eta_seconds
            self._eta_seconds = smoothed_eta

            if smoothed_eta < 60:
                eta_str = f"{int(smoothed_eta)}s"
            elif smoothed_eta < 3600:
                eta_str = f"{int(smoothed_eta / 60)}m {int(smoothed_eta % 60)}s"
            else:
                eta_str = f"{int(smoothed_eta / 3600)}h {int((smoothed_eta % 3600) / 60)}m"
            self._eta_value.setText(eta_str)

            # Add informative tooltip about ETA calculation
            if self._rate_trend == "decreasing":
                self._eta_value.setToolTip(
                    f"Estimated time remaining: {eta_str}\n"
                    "ETA accounts for rate decay as workers complete."
                )
            else:
                self._eta_value.setToolTip(f"Estimated time remaining: {eta_str}")
        elif eta_seconds == 0 or self._progress >= 100:
            self._eta_value.setText("Complete")
            self._eta_value.setToolTip("Operation complete")

        # Update compute kernel
        compute_kernel = data.get('compute_kernel', self._compute_kernel)
        if compute_kernel:
            self._compute_kernel = compute_kernel
            self._kernel_value.setText(compute_kernel)

        # Update I/O metrics
        io_rate_mbps = data.get('io_rate_mbps', 0.0)
        io_rate_trend = data.get('io_rate_trend', '')
        io_stall = data.get('io_stall_detected', False)

        if io_rate_mbps and io_rate_mbps > 0:
            self._io_rate_mbps = io_rate_mbps
            self._io_rate_value.setText(f"{io_rate_mbps:.0f} MB/s")

            # Set appropriate tooltip based on rate
            if io_rate_mbps > 500:
                self._io_rate_value.setToolTip(f"Write rate: {io_rate_mbps:.0f} MB/s (excellent)")
            elif io_rate_mbps > 200:
                self._io_rate_value.setToolTip(f"Write rate: {io_rate_mbps:.0f} MB/s (good)")
            else:
                self._io_rate_value.setToolTip(
                    f"Write rate: {io_rate_mbps:.0f} MB/s (limited)\n"
                    "Consider using faster storage (internal SSD)"
                )

        if io_rate_trend:
            self._io_rate_trend = io_rate_trend
            if io_rate_trend == "▲":
                self._io_trend_label.setText("▲")
                self._io_trend_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #4ade80;")
                self._io_trend_label.setToolTip("I/O rate increasing")
            elif io_rate_trend == "▼":
                self._io_trend_label.setText("▼")
                self._io_trend_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #fbbf24;")
                self._io_trend_label.setToolTip("I/O rate decreasing - possible storage saturation")
            else:
                self._io_trend_label.setText("─")
                self._io_trend_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #9ca3af;")
                self._io_trend_label.setToolTip("I/O rate stable")

        # Update I/O stall warning
        self._io_stall_detected = io_stall
        if io_stall:
            self._io_stall_label.setText("⚠ STALL")
            self._io_stall_label.setToolTip(
                "I/O stall detected - storage cannot keep up with write rate.\n"
                "This typically happens with external drives or network storage.\n"
                "Consider using internal SSD for better performance."
            )
        else:
            self._io_stall_label.setText("")
            self._io_stall_label.setToolTip("")

    def _on_pause_clicked(self):
        """Handle pause/resume button click."""
        if self._state == "PAUSED":
            self.resume_requested.emit(self._job_id)
        else:
            self.pause_requested.emit(self._job_id)

    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        self.cancel_requested.emit(self._job_id)
