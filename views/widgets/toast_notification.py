"""
Toast Notification Widget

Provides non-intrusive toast-style notifications that appear
and auto-dismiss, with support for different severity levels.
"""

import logging
from typing import Optional, List
from enum import Enum, auto

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

logger = logging.getLogger(__name__)


class ToastType(Enum):
    """Toast notification types."""
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()


class Toast(QFrame):
    """
    A single toast notification.

    Displays a message with an icon and optional close button.
    Auto-dismisses after a timeout or can be manually closed.

    Signals
    -------
    closed : None
        Emitted when toast is closed
    """

    closed = pyqtSignal()

    # Colors for different toast types
    COLORS = {
        ToastType.INFO: ("#17a2b8", "#d1ecf1"),     # Cyan
        ToastType.SUCCESS: ("#28a745", "#d4edda"),  # Green
        ToastType.WARNING: ("#ffc107", "#fff3cd"),  # Yellow
        ToastType.ERROR: ("#dc3545", "#f8d7da"),    # Red
    }

    # Icons for different types
    ICONS = {
        ToastType.INFO: "i",
        ToastType.SUCCESS: "✓",
        ToastType.WARNING: "!",
        ToastType.ERROR: "✕",
    }

    def __init__(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        title: Optional[str] = None,
        duration_ms: int = 5000,
        closable: bool = True,
        parent=None,
    ):
        """
        Initialize toast notification.

        Parameters
        ----------
        message : str
            Toast message
        toast_type : ToastType
            Type of toast (affects styling)
        title : str, optional
            Optional title
        duration_ms : int
            Auto-dismiss duration (0 = never)
        closable : bool
            Show close button
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        self._message = message
        self._type = toast_type
        self._title = title
        self._duration = duration_ms
        self._closable = closable

        self._setup_ui()
        self._apply_style()

        if duration_ms > 0:
            self._setup_timer()

    def _setup_ui(self):
        """Set up the toast UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(350)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Icon
        icon_label = QLabel(self.ICONS.get(self._type, "i"))
        icon_label.setFixedSize(24, 24)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"""
            background-color: {self.COLORS[self._type][0]};
            color: white;
            border-radius: 12px;
            font-weight: bold;
            font-size: 14px;
        """)
        layout.addWidget(icon_label)

        # Content
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)

        # Title (optional)
        if self._title:
            title_label = QLabel(self._title)
            title_label.setStyleSheet(f"font-weight: bold; color: {self.COLORS[self._type][0]};")
            content_layout.addWidget(title_label)

        # Message
        message_label = QLabel(self._message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #333333;")
        content_layout.addWidget(message_label)

        layout.addLayout(content_layout, 1)

        # Close button
        if self._closable:
            close_btn = QPushButton("×")
            close_btn.setFixedSize(20, 20)
            close_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #666666;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #333333;
                }
            """)
            close_btn.clicked.connect(self.close_toast)
            layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignTop)

    def _apply_style(self):
        """Apply styling based on toast type."""
        bg_color = self.COLORS[self._type][1]
        border_color = self.COLORS[self._type][0]

        self.setStyleSheet(f"""
            Toast {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-left: 4px solid {border_color};
                border-radius: 4px;
            }}
        """)

    def _setup_timer(self):
        """Set up auto-dismiss timer."""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close_toast)
        self._timer.start(self._duration)

    def close_toast(self):
        """Close the toast with animation."""
        # Fade out animation
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(200)
        self._fade_animation.setStartValue(1.0)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.finished.connect(self._on_fade_complete)
        self._fade_animation.start()

    def _on_fade_complete(self):
        """Handle fade animation complete."""
        self.closed.emit()
        self.deleteLater()


class ToastManager(QWidget):
    """
    Manager for displaying toast notifications.

    Handles positioning and stacking of multiple toasts.

    Usage
    -----
    >>> manager = ToastManager(parent_window)
    >>> manager.show_info("File saved successfully")
    >>> manager.show_error("Failed to load file", title="Error")
    """

    # Spacing between toasts
    TOAST_SPACING = 10

    def __init__(self, parent=None):
        """
        Initialize toast manager.

        Parameters
        ----------
        parent : QWidget
            Parent widget (typically main window)
        """
        super().__init__(parent)

        self._toasts: List[Toast] = []
        self._position = "top-right"  # Default position
        self._margin = 20

        # Make this widget invisible and non-interactive
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

    def set_position(self, position: str):
        """
        Set toast position.

        Parameters
        ----------
        position : str
            Position: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
        """
        self._position = position

    def show_toast(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        title: Optional[str] = None,
        duration_ms: int = 5000,
    ):
        """
        Show a toast notification.

        Parameters
        ----------
        message : str
            Toast message
        toast_type : ToastType
            Type of toast
        title : str, optional
            Optional title
        duration_ms : int
            Auto-dismiss duration (0 = never)
        """
        toast = Toast(
            message=message,
            toast_type=toast_type,
            title=title,
            duration_ms=duration_ms,
            parent=self.parent(),
        )

        toast.closed.connect(lambda: self._remove_toast(toast))

        self._toasts.append(toast)
        self._position_toasts()

        toast.show()
        toast.raise_()

    def show_info(self, message: str, title: Optional[str] = None, duration_ms: int = 5000):
        """Show an info toast."""
        self.show_toast(message, ToastType.INFO, title, duration_ms)

    def show_success(self, message: str, title: Optional[str] = None, duration_ms: int = 4000):
        """Show a success toast."""
        self.show_toast(message, ToastType.SUCCESS, title, duration_ms)

    def show_warning(self, message: str, title: Optional[str] = None, duration_ms: int = 6000):
        """Show a warning toast."""
        self.show_toast(message, ToastType.WARNING, title, duration_ms)

    def show_error(self, message: str, title: Optional[str] = None, duration_ms: int = 8000):
        """Show an error toast."""
        self.show_toast(message, ToastType.ERROR, title, duration_ms)

    def _remove_toast(self, toast: Toast):
        """Remove a toast from the manager."""
        if toast in self._toasts:
            self._toasts.remove(toast)
            self._position_toasts()

    def _position_toasts(self):
        """Position all active toasts."""
        if not self.parent():
            return

        parent = self.parent()
        parent_rect = parent.rect()

        # Calculate starting position
        if "right" in self._position:
            x = parent_rect.width() - 350 - self._margin
        else:
            x = self._margin

        if "top" in self._position:
            start_y = self._margin
            direction = 1
        else:
            start_y = parent_rect.height() - self._margin
            direction = -1

        # Position each toast
        current_y = start_y
        for toast in self._toasts:
            toast_height = toast.sizeHint().height()

            if direction < 0:
                current_y -= toast_height

            toast.move(x, current_y)

            if direction > 0:
                current_y += toast_height + self.TOAST_SPACING
            else:
                current_y -= self.TOAST_SPACING

    def clear_all(self):
        """Clear all toasts."""
        for toast in list(self._toasts):
            toast.close_toast()

    def get_active_count(self) -> int:
        """Get number of active toasts."""
        return len(self._toasts)


class AlertToastBridge:
    """
    Bridge between AlertManager and ToastManager.

    Automatically displays toast notifications for new alerts.

    Usage
    -----
    >>> bridge = AlertToastBridge(toast_manager)
    >>> bridge.connect_to_alert_manager(alert_manager)
    """

    def __init__(self, toast_manager: ToastManager):
        """
        Initialize bridge.

        Parameters
        ----------
        toast_manager : ToastManager
            Toast manager to use for notifications
        """
        self._toast_manager = toast_manager

    def connect_to_alert_manager(self, alert_manager):
        """
        Connect to an alert manager.

        Parameters
        ----------
        alert_manager : AlertManager
            Alert manager to connect to
        """
        alert_manager.register_callback(self._on_alert)

    def _on_alert(self, alert):
        """Handle new alert."""
        from utils.ray_orchestration.alert_manager import AlertSeverity

        # Map alert severity to toast type
        severity_map = {
            AlertSeverity.INFO: ToastType.INFO,
            AlertSeverity.WARNING: ToastType.WARNING,
            AlertSeverity.ERROR: ToastType.ERROR,
            AlertSeverity.CRITICAL: ToastType.ERROR,
        }

        toast_type = severity_map.get(alert.severity, ToastType.INFO)

        # Adjust duration based on severity
        duration_map = {
            AlertSeverity.INFO: 4000,
            AlertSeverity.WARNING: 6000,
            AlertSeverity.ERROR: 8000,
            AlertSeverity.CRITICAL: 10000,
        }

        duration = duration_map.get(alert.severity, 5000)

        self._toast_manager.show_toast(
            message=alert.message,
            toast_type=toast_type,
            title=alert.title,
            duration_ms=duration,
        )
