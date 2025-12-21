"""
Job Queue Widget

Displays a list of queued jobs with reordering capability.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import List, Dict, Any
from uuid import UUID


class JobQueueWidget(QFrame):
    """
    Widget displaying queued jobs with priority controls.

    Signals
    -------
    job_selected : UUID
        Emitted when a job is selected
    priority_up_requested : UUID
        Emitted when move up button clicked
    priority_down_requested : UUID
        Emitted when move down button clicked
    cancel_requested : UUID
        Emitted when cancel button clicked
    """

    job_selected = pyqtSignal(object)  # UUID
    priority_up_requested = pyqtSignal(object)  # UUID
    priority_down_requested = pyqtSignal(object)  # UUID
    cancel_requested = pyqtSignal(object)  # UUID

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: Dict[UUID, Dict[str, Any]] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("Queued Jobs")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)

        self._count_label = QLabel("(0)")
        self._count_label.setStyleSheet("color: #6c757d;")
        header.addWidget(self._count_label)
        header.addStretch()

        layout.addLayout(header)

        # Job list
        self._list_widget = QListWidget()
        self._list_widget.setAlternatingRowColors(True)
        self._list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list_widget)

        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(4)

        self._up_btn = QPushButton("Move Up")
        self._up_btn.setEnabled(False)
        self._up_btn.clicked.connect(self._on_move_up)
        controls.addWidget(self._up_btn)

        self._down_btn = QPushButton("Move Down")
        self._down_btn.setEnabled(False)
        self._down_btn.clicked.connect(self._on_move_down)
        controls.addWidget(self._down_btn)

        controls.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        controls.addWidget(self._cancel_btn)

        layout.addLayout(controls)

    def add_job(self, job_id: UUID, name: str, job_type: str, priority: int = 5):
        """Add a job to the queue."""
        self._jobs[job_id] = {
            "name": name,
            "job_type": job_type,
            "priority": priority,
        }

        item = QListWidgetItem(f"{name} ({job_type})")
        item.setData(Qt.ItemDataRole.UserRole, job_id)
        self._list_widget.addItem(item)

        self._update_count()

    def remove_job(self, job_id: UUID):
        """Remove a job from the queue."""
        if job_id in self._jobs:
            del self._jobs[job_id]

        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == job_id:
                self._list_widget.takeItem(i)
                break

        self._update_count()

    def clear(self):
        """Clear all jobs from the queue."""
        self._jobs.clear()
        self._list_widget.clear()
        self._update_count()

    def get_job_ids(self) -> List[UUID]:
        """Get list of job IDs in queue order."""
        return [
            self._list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list_widget.count())
        ]

    def _update_count(self):
        """Update the count label."""
        count = self._list_widget.count()
        self._count_label.setText(f"({count})")

    def _on_selection_changed(self):
        """Handle selection change."""
        selected = self._list_widget.currentItem()
        has_selection = selected is not None
        row = self._list_widget.currentRow()

        self._up_btn.setEnabled(has_selection and row > 0)
        self._down_btn.setEnabled(
            has_selection and row < self._list_widget.count() - 1
        )
        self._cancel_btn.setEnabled(has_selection)

        if has_selection:
            job_id = selected.data(Qt.ItemDataRole.UserRole)
            self.job_selected.emit(job_id)

    def _on_move_up(self):
        """Handle move up button click."""
        current = self._list_widget.currentItem()
        if current:
            job_id = current.data(Qt.ItemDataRole.UserRole)
            self.priority_up_requested.emit(job_id)

    def _on_move_down(self):
        """Handle move down button click."""
        current = self._list_widget.currentItem()
        if current:
            job_id = current.data(Qt.ItemDataRole.UserRole)
            self.priority_down_requested.emit(job_id)

    def _on_cancel(self):
        """Handle cancel button click."""
        current = self._list_widget.currentItem()
        if current:
            job_id = current.data(Qt.ItemDataRole.UserRole)
            self.cancel_requested.emit(job_id)
