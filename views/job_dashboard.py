"""
Job Dashboard Widget

Central widget for monitoring and controlling all jobs in the system.
Displays active jobs, queued jobs, and recent job history.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QScrollArea, QFrame,
    QSplitter, QGroupBox, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from typing import Dict, List, Optional, Any
from uuid import UUID

from .widgets.job_card import JobCardWidget
from .widgets.job_queue_widget import JobQueueWidget


class JobDashboardWidget(QWidget):
    """
    Dashboard for monitoring and controlling all jobs.

    Provides:
    - Active jobs view with progress and controls
    - Queued jobs with priority management
    - Job history (completed/failed/cancelled)
    - Resource monitoring integration

    Signals
    -------
    cancel_job_requested : UUID
        Emitted when user requests to cancel a job
    pause_job_requested : UUID
        Emitted when user requests to pause a job
    resume_job_requested : UUID
        Emitted when user requests to resume a job
    cancel_all_requested : None
        Emitted when user requests to cancel all jobs
    """

    cancel_job_requested = pyqtSignal(object)  # UUID
    pause_job_requested = pyqtSignal(object)  # UUID
    resume_job_requested = pyqtSignal(object)  # UUID
    cancel_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._active_jobs: Dict[UUID, JobCardWidget] = {}
        self._history: List[Dict[str, Any]] = []
        self._max_history = 50

        # Queue for progress updates that arrive before card is created
        # This handles the race condition where progress signals arrive before job_started
        self._pending_progress: Dict[UUID, List[Dict[str, Any]]] = {}

        self._setup_ui()
        self._setup_refresh_timer()

    def _setup_ui(self):
        """Set up the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header with title and global controls
        header = self._create_header()
        layout.addLayout(header)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Active jobs
        active_panel = self._create_active_panel()
        splitter.addWidget(active_panel)

        # Right: Queue and history tabs
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([600, 300])
        layout.addWidget(splitter)

        # Status bar
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        layout.addWidget(self._status_label)

    def _create_header(self) -> QHBoxLayout:
        """Create header with title and controls."""
        header = QHBoxLayout()

        title = QLabel("Job Dashboard")
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)

        header.addStretch()

        # Active count badge
        self._active_count = QLabel("0 active")
        self._active_count.setStyleSheet(
            "background-color: #007bff; color: white; "
            "border-radius: 10px; padding: 4px 12px;"
        )
        header.addWidget(self._active_count)

        # Queued count badge
        self._queued_count = QLabel("0 queued")
        self._queued_count.setStyleSheet(
            "background-color: #17a2b8; color: white; "
            "border-radius: 10px; padding: 4px 12px;"
        )
        header.addWidget(self._queued_count)

        # Cancel all button
        self._cancel_all_btn = QPushButton("Cancel All")
        self._cancel_all_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; "
            "border: none; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #c82333; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self._cancel_all_btn.setEnabled(False)
        self._cancel_all_btn.clicked.connect(self._on_cancel_all)
        header.addWidget(self._cancel_all_btn)

        return header

    def _create_active_panel(self) -> QFrame:
        """Create panel showing active jobs."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        label = QLabel("Active Jobs")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        header.addWidget(label)
        header.addStretch()
        layout.addLayout(header)

        # Scrollable container for job cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._active_container = QWidget()
        self._active_layout = QVBoxLayout(self._active_container)
        self._active_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._active_layout.setSpacing(8)

        # Placeholder when empty
        self._empty_label = QLabel("No active jobs")
        self._empty_label.setStyleSheet("color: #6c757d;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._active_layout.addWidget(self._empty_label)

        scroll.setWidget(self._active_container)
        layout.addWidget(scroll)

        return panel

    def _create_right_panel(self) -> QTabWidget:
        """Create right panel with queue and history tabs."""
        tabs = QTabWidget()

        # Queue tab
        self._queue_widget = JobQueueWidget()
        self._queue_widget.cancel_requested.connect(self._on_cancel_job)
        tabs.addTab(self._queue_widget, "Queue")

        # History tab
        history_widget = self._create_history_widget()
        tabs.addTab(history_widget, "History")

        return tabs

    def _create_history_widget(self) -> QFrame:
        """Create history display widget."""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # History list (simplified)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self._history_container = QWidget()
        self._history_layout = QVBoxLayout(self._history_container)
        self._history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._history_layout.setSpacing(4)

        self._history_empty_label = QLabel("No job history")
        self._history_empty_label.setStyleSheet("color: #6c757d;")
        self._history_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._history_layout.addWidget(self._history_empty_label)

        scroll.setWidget(self._history_container)
        layout.addWidget(scroll)

        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        layout.addWidget(clear_btn)

        return panel

    def _setup_refresh_timer(self):
        """Set up timer for periodic UI refresh."""
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._update_status)
        self._refresh_timer.start(1000)  # 1 second refresh

    # Public API

    @pyqtSlot(object, dict)
    def on_job_started(self, job_id: UUID, job_info: Dict[str, Any]):
        """Handle job started event."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DASHBOARD] on_job_started: job_id={job_id}, name={job_info.get('name', 'Unknown')}")

        # Check if job is already active (prevent duplicate cards from multiple signals)
        if job_id in self._active_jobs:
            # Job already has a card, just update its state
            self._active_jobs[job_id].update_state("RUNNING")
            return

        # Hide empty label
        self._empty_label.hide()

        # Remove from queue if present (job transitioning from queued to running)
        self._queue_widget.remove_job(job_id)

        # Create job card
        card = JobCardWidget(
            job_id=job_id,
            name=job_info.get("name", "Unknown"),
            job_type=job_info.get("job_type", "Unknown"),
        )
        card.cancel_requested.connect(self._on_cancel_job)
        card.pause_requested.connect(self._on_pause_job)
        card.resume_requested.connect(self._on_resume_job)

        self._active_jobs[job_id] = card
        self._active_layout.insertWidget(0, card)

        card.update_state("RUNNING")
        self._update_counts()

        # Replay any queued progress updates that arrived before the card was created
        if job_id in self._pending_progress:
            pending = self._pending_progress.pop(job_id)
            if pending:
                # Apply the most recent progress update
                self.on_progress_updated(job_id, pending[-1])

    @pyqtSlot(object, dict)
    def on_progress_updated(self, job_id: UUID, progress: Dict[str, Any]):
        """Handle progress update event with detailed statistics."""
        import logging
        logger = logging.getLogger(__name__)

        card = self._active_jobs.get(job_id)
        if not card:
            # Queue progress for later - card may not be created yet (race condition)
            if job_id not in self._pending_progress:
                self._pending_progress[job_id] = []
            # Only keep last 5 pending updates to avoid memory bloat
            if len(self._pending_progress[job_id]) < 5:
                self._pending_progress[job_id].append(progress)
            return

        # Only pass values that exist in progress dict to avoid resetting to 0
        # Use None as sentinel to indicate "no update" for optional fields
        card.update_progress(
            percent=progress.get("percent", 0),
            message=progress.get("message", ""),
            eta_seconds=progress.get("eta_seconds"),
            # Extended statistics for detailed display
            phase=progress.get("phase", ""),
            current_gathers=progress.get("current_gathers"),  # None if missing
            total_gathers=progress.get("total_gathers"),      # None if missing
            current_traces=progress.get("current_traces"),    # None if missing
            total_traces=progress.get("total_traces"),        # None if missing
            active_workers=progress.get("active_workers"),    # None if missing
            traces_per_sec=progress.get("traces_per_sec", 0),
            # Compute kernel info (Metal/CPU)
            compute_kernel=progress.get("compute_kernel", ""),
        )

    @pyqtSlot(object, str)
    def on_job_state_changed(self, job_id: UUID, state: str):
        """Handle job state change event."""
        card = self._active_jobs.get(job_id)
        if card:
            card.update_state(state)

            # Move to history if terminal
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                self._move_to_history(job_id, state)

    @pyqtSlot(object, dict)
    def on_job_queued(self, job_id: UUID, job_info: Dict[str, Any]):
        """Handle job queued event."""
        # Skip if job is already active (prevent stale queue entries)
        if job_id in self._active_jobs:
            return

        self._queue_widget.add_job(
            job_id=job_id,
            name=job_info.get("name", "Unknown"),
            job_type=job_info.get("job_type", "Unknown"),
            priority=job_info.get("priority", 5),
        )
        self._update_counts()

    @pyqtSlot(object)
    def on_job_dequeued(self, job_id: UUID):
        """Handle job removed from queue (started or cancelled)."""
        self._queue_widget.remove_job(job_id)
        self._update_counts()

    def active_jobs_count(self) -> int:
        """Get count of active jobs."""
        return len(self._active_jobs)

    def get_active_job_names(self) -> List[str]:
        """Get names of all active jobs."""
        return [card._name for card in self._active_jobs.values()]

    def get_job_progress(self, job_id: UUID) -> Optional[float]:
        """Get progress of a specific job."""
        card = self._active_jobs.get(job_id)
        if card:
            return card._progress
        return None

    # Private methods

    def _move_to_history(self, job_id: UUID, final_state: str):
        """Move job from active to history."""
        card = self._active_jobs.pop(job_id, None)
        if card:
            # Remove from layout
            self._active_layout.removeWidget(card)
            card.deleteLater()

            # Add to history
            self._add_to_history({
                "job_id": job_id,
                "name": card._name,
                "job_type": card._job_type,
                "state": final_state,
                "progress": card._progress,
            })

        # Show empty label if no active jobs
        if not self._active_jobs:
            self._empty_label.show()

        self._update_counts()

    def _add_to_history(self, job_info: Dict[str, Any]):
        """Add job to history."""
        self._history_empty_label.hide()

        # Create history entry label
        state = job_info.get("state", "Unknown")
        name = job_info.get("name", "Unknown")
        color = JobCardWidget.STATE_COLORS.get(state, "#6c757d")

        entry = QLabel(f"● {name} - {state}")
        entry.setStyleSheet(f"color: {color};")
        self._history_layout.insertWidget(0, entry)

        self._history.insert(0, job_info)

        # Trim history
        while len(self._history) > self._max_history:
            self._history.pop()
            if self._history_layout.count() > self._max_history + 1:
                item = self._history_layout.takeAt(self._max_history)
                if item.widget():
                    item.widget().deleteLater()

    def _clear_history(self):
        """Clear job history."""
        self._history.clear()
        # Remove all widgets except empty label
        while self._history_layout.count() > 1:
            item = self._history_layout.takeAt(0)
            if item.widget() and item.widget() != self._history_empty_label:
                item.widget().deleteLater()
        self._history_empty_label.show()

    def _update_counts(self):
        """Update count badges."""
        active = len(self._active_jobs)
        queued = self._queue_widget._list_widget.count()

        self._active_count.setText(f"{active} active")
        self._queued_count.setText(f"{queued} queued")

        # Enable/disable cancel all
        self._cancel_all_btn.setEnabled(active > 0 or queued > 0)

    def _update_status(self):
        """Update status bar."""
        active = len(self._active_jobs)
        if active > 0:
            self._status_label.setText(f"Processing {active} job(s)...")
        else:
            self._status_label.setText("Ready")

    # Signal handlers

    def _on_cancel_job(self, job_id: UUID):
        """Handle cancel request for a job."""
        self.cancel_job_requested.emit(job_id)

    def _on_pause_job(self, job_id: UUID):
        """Handle pause request for a job."""
        self.pause_job_requested.emit(job_id)

    def _on_resume_job(self, job_id: UUID):
        """Handle resume request for a job."""
        self.resume_job_requested.emit(job_id)

    def _on_cancel_all(self):
        """Handle cancel all request."""
        reply = QMessageBox.question(
            self,
            "Cancel All Jobs",
            "Are you sure you want to cancel all running and queued jobs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.cancel_all_requested.emit()
