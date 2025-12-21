"""
Main Window Integration Module

Provides integration of Ray job management and resource monitoring
widgets into the main application window.
"""

import logging
from typing import Optional, Dict, Any
from uuid import UUID

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QStatusBar, QWidget, QHBoxLayout, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QObject

from .job_dashboard import JobDashboardWidget
from .widgets.resource_monitor_widget import (
    CompactResourceMonitorWidget,
    ResourceMonitorWidget,
)
from .widgets.job_analytics_widget import JobAnalyticsWidget
from .widgets.toast_notification import (
    ToastManager,
    ToastType,
    AlertToastBridge,
)

logger = logging.getLogger(__name__)


class MainWindowJobIntegration(QObject):
    """
    Mixin/helper class to integrate job management into a QMainWindow.

    Adds:
    - Job Dashboard as a dock widget
    - Resource Monitor as a dock widget
    - Compact resource monitor in status bar

    Usage
    -----
    >>> class MainWindow(QMainWindow, MainWindowJobIntegration):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.setup_job_integration()

    Or as composition:
    >>> class MainWindow(QMainWindow):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self._job_integration = MainWindowJobIntegration(self)
    ...         self._job_integration.setup()
    """

    def __init__(self, main_window: QMainWindow):
        """
        Initialize job integration for a main window.

        Parameters
        ----------
        main_window : QMainWindow
            The main window to integrate with
        """
        super().__init__(parent=main_window)
        self._window = main_window
        self._job_dashboard: Optional[JobDashboardWidget] = None
        self._job_dock: Optional[QDockWidget] = None
        self._resource_dock: Optional[QDockWidget] = None
        self._analytics_dock: Optional[QDockWidget] = None
        self._resource_monitor: Optional[ResourceMonitorWidget] = None
        self._compact_monitor: Optional[CompactResourceMonitorWidget] = None
        self._job_analytics: Optional[JobAnalyticsWidget] = None
        self._toast_manager: Optional[ToastManager] = None
        self._alert_toast_bridge: Optional[AlertToastBridge] = None
        self._job_manager = None
        self._signal_bridge = None
        self._alert_manager = None

    def setup(
        self,
        job_manager=None,
        signal_bridge=None,
        alert_manager=None,
        show_job_dock: bool = False,
        show_resource_dock: bool = False,
        show_analytics_dock: bool = False,
    ):
        """
        Set up job integration components.

        Parameters
        ----------
        job_manager : JobManager, optional
            JobManager instance for job control
        signal_bridge : JobManagerBridge, optional
            Signal bridge for Qt updates
        alert_manager : AlertManager, optional
            Alert manager for job alerts and notifications
        show_job_dock : bool
            Whether to show job dashboard dock initially
        show_resource_dock : bool
            Whether to show resource monitor dock initially
        show_analytics_dock : bool
            Whether to show job analytics dock initially
        """
        self._job_manager = job_manager
        self._signal_bridge = signal_bridge
        self._alert_manager = alert_manager

        # Create dock widgets
        self._create_job_dashboard_dock()
        self._create_resource_monitor_dock()
        self._create_analytics_dock()

        # Add compact monitor to status bar
        self._add_status_bar_monitor()

        # Create toast notification system
        self._create_toast_system()

        # Connect signals if bridge provided
        if self._signal_bridge:
            self._connect_bridge_signals()
            # Start the bridge to enable JobManager callbacks and polling
            if hasattr(self._signal_bridge, 'start'):
                self._signal_bridge.start()

        # Connect alert manager to toast notifications
        if self._alert_manager and self._alert_toast_bridge:
            self._alert_toast_bridge.connect_to_alert_manager(self._alert_manager)

        # Set initial visibility
        if self._job_dock:
            self._job_dock.setVisible(show_job_dock)
        if self._resource_dock:
            self._resource_dock.setVisible(show_resource_dock)
        if self._analytics_dock:
            self._analytics_dock.setVisible(show_analytics_dock)

        logger.info("Job integration setup complete")

    def _create_job_dashboard_dock(self):
        """Create the job dashboard dock widget."""
        self._job_dashboard = JobDashboardWidget()

        self._job_dock = QDockWidget("Job Dashboard", self._window)
        self._job_dock.setWidget(self._job_dashboard)
        self._job_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Connect job control signals
        # Note: cancel_job_requested is connected in MainWindow._on_cancel_job_requested
        # which has the unified handler that routes to SEGY workers or batch jobs
        self._job_dashboard.pause_job_requested.connect(self._on_pause_job)
        self._job_dashboard.resume_job_requested.connect(self._on_resume_job)
        self._job_dashboard.cancel_all_requested.connect(self._on_cancel_all)

        # Add dock to main window (bottom right by default)
        self._window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self._job_dock
        )

    def _create_resource_monitor_dock(self):
        """Create the resource monitor dock widget."""
        self._resource_monitor = ResourceMonitorWidget()

        self._resource_dock = QDockWidget("Resource Monitor", self._window)
        self._resource_dock.setWidget(self._resource_monitor)
        self._resource_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Connect alert signals
        self._resource_monitor.alert_triggered.connect(self._on_resource_alert)

        # Add dock (tabbed with job dashboard)
        self._window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self._resource_dock
        )

        # Tabify with job dashboard if both in same area
        if self._job_dock:
            self._window.tabifyDockWidget(self._job_dock, self._resource_dock)

    def _add_status_bar_monitor(self):
        """Add compact resource monitor to status bar."""
        self._compact_monitor = CompactResourceMonitorWidget()

        # Get or create status bar
        status_bar = self._window.statusBar()
        if status_bar:
            # Add as permanent widget on the right
            status_bar.addPermanentWidget(self._compact_monitor)

    def _create_analytics_dock(self):
        """Create the job analytics dock widget."""
        self._job_analytics = JobAnalyticsWidget()

        self._analytics_dock = QDockWidget("Job Analytics", self._window)
        self._analytics_dock.setWidget(self._job_analytics)
        self._analytics_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Add dock (tabbed with other docks)
        self._window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self._analytics_dock
        )

        # Tabify with resource dock if available
        if self._resource_dock:
            self._window.tabifyDockWidget(self._resource_dock, self._analytics_dock)

    def _create_toast_system(self):
        """Create toast notification system with alert bridge."""
        self._toast_manager = ToastManager(self._window)
        self._alert_toast_bridge = AlertToastBridge(self._toast_manager)

        # Also connect resource alerts to toast notifications
        if self._resource_monitor:
            self._resource_monitor.alert_triggered.connect(self._on_resource_alert_toast)

    @pyqtSlot(object)
    def _on_resource_alert_toast(self, alert):
        """Show toast for resource alert."""
        if self._toast_manager:
            # Map resource alert severity to toast type
            severity = getattr(alert, 'severity', 'warning')
            if severity == 'critical':
                self._toast_manager.show_error(
                    alert.message,
                    title="Resource Alert"
                )
            else:
                self._toast_manager.show_warning(
                    alert.message,
                    title="Resource Warning"
                )

    def _connect_bridge_signals(self):
        """Connect signal bridge to dashboard.

        The bridge signals emit:
        - job_queued(UUID, dict) -> dashboard.on_job_queued(job_id, job_info)
        - job_started(UUID, dict) -> dashboard.on_job_started(job_id, job_info)
        - job_progress(UUID, dict) -> dashboard.on_progress_updated(job_id, progress)
        - job_state_changed(UUID, str) -> dashboard.on_job_state_changed(job_id, state)
        """
        if not self._signal_bridge or not self._job_dashboard:
            logger.warning("[INTEGRATION] Cannot connect signals: bridge or dashboard is None")
            return

        # Get the signals emitter from the bridge
        signals = self._signal_bridge.signals if hasattr(self._signal_bridge, 'signals') else self._signal_bridge

        logger.info(f"[INTEGRATION] Connecting bridge signals to dashboard")
        logger.info(f"[INTEGRATION]   bridge={self._signal_bridge}, signals={signals}, dashboard={self._job_dashboard}")

        # Direct connections where signatures match
        signals.job_queued.connect(self._job_dashboard.on_job_queued)
        signals.job_started.connect(self._job_dashboard.on_job_started)
        signals.job_progress.connect(self._job_dashboard.on_progress_updated)
        signals.job_state_changed.connect(self._job_dashboard.on_job_state_changed)

        logger.info("[INTEGRATION] Signal connections established")

        # job_completed/job_failed/job_cancelled need state change forwarding
        signals.job_completed.connect(self._on_job_completed)
        signals.job_failed.connect(self._on_job_failed)
        signals.job_cancelled.connect(self._on_job_cancelled)

    # Signal handlers - adapt bridge signals to dashboard slots

    @pyqtSlot(object, dict)
    def _on_job_completed(self, job_id: UUID, result: Dict[str, Any]):
        """Handle job completed signal - forward state change to dashboard."""
        if self._job_dashboard:
            self._job_dashboard.on_job_state_changed(job_id, "COMPLETED")

    @pyqtSlot(object, dict)
    def _on_job_failed(self, job_id: UUID, error_info: Dict[str, Any]):
        """Handle job failed signal - forward state change to dashboard."""
        if self._job_dashboard:
            self._job_dashboard.on_job_state_changed(job_id, "FAILED")

    @pyqtSlot(object)
    def _on_job_cancelled(self, job_id: UUID):
        """Handle job cancelled signal - forward state change to dashboard."""
        if self._job_dashboard:
            self._job_dashboard.on_job_state_changed(job_id, "CANCELLED")

    @pyqtSlot(object)
    def _on_cancel_job(self, job_id: UUID):
        """Handle cancel job request."""
        if self._job_manager:
            try:
                self._job_manager.cancel_job(job_id)
                logger.info(f"Cancelled job {job_id}")
            except Exception as e:
                logger.error(f"Failed to cancel job {job_id}: {e}")

    @pyqtSlot(object)
    def _on_pause_job(self, job_id: UUID):
        """Handle pause job request."""
        if self._job_manager:
            try:
                self._job_manager.pause_job(job_id)
                logger.info(f"Paused job {job_id}")
            except Exception as e:
                logger.error(f"Failed to pause job {job_id}: {e}")

    @pyqtSlot(object)
    def _on_resume_job(self, job_id: UUID):
        """Handle resume job request."""
        if self._job_manager:
            try:
                self._job_manager.resume_job(job_id)
                logger.info(f"Resumed job {job_id}")
            except Exception as e:
                logger.error(f"Failed to resume job {job_id}: {e}")

    @pyqtSlot()
    def _on_cancel_all(self):
        """Handle cancel all request."""
        if self._job_manager:
            try:
                self._job_manager.cancel_all()
                logger.info("Cancelled all jobs")
            except Exception as e:
                logger.error(f"Failed to cancel all jobs: {e}")

    @pyqtSlot(object)
    def _on_resource_alert(self, alert):
        """Handle resource alert."""
        # Could show notification, update status bar, etc.
        logger.warning(f"Resource alert: {alert.message}")

    # Public API
    def show_job_dashboard(self):
        """Show the job dashboard dock."""
        if self._job_dock:
            self._job_dock.show()
            self._job_dock.raise_()

    def show_resource_monitor(self):
        """Show the resource monitor dock."""
        if self._resource_dock:
            self._resource_dock.show()
            self._resource_dock.raise_()

    def show_job_analytics(self):
        """Show the job analytics dock."""
        if self._analytics_dock:
            self._analytics_dock.show()
            self._analytics_dock.raise_()

    def get_job_dashboard(self) -> Optional[JobDashboardWidget]:
        """Get the job dashboard widget."""
        return self._job_dashboard

    def get_resource_monitor(self) -> Optional[ResourceMonitorWidget]:
        """Get the resource monitor widget."""
        return self._resource_monitor

    def get_compact_monitor(self) -> Optional[CompactResourceMonitorWidget]:
        """Get the compact resource monitor widget."""
        return self._compact_monitor

    def get_job_analytics(self) -> Optional[JobAnalyticsWidget]:
        """Get the job analytics widget."""
        return self._job_analytics

    def get_toast_manager(self) -> Optional[ToastManager]:
        """Get the toast notification manager."""
        return self._toast_manager

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
            Type of toast (INFO, SUCCESS, WARNING, ERROR)
        title : str, optional
            Optional title
        duration_ms : int
            Auto-dismiss duration in milliseconds
        """
        if self._toast_manager:
            self._toast_manager.show_toast(message, toast_type, title, duration_ms)

    def cleanup(self):
        """Clean up integration resources."""
        # Stop signal bridge
        if self._signal_bridge and hasattr(self._signal_bridge, 'stop'):
            self._signal_bridge.stop()
        if self._resource_monitor:
            self._resource_monitor.cleanup()
        if self._compact_monitor:
            self._compact_monitor.cleanup()
        if self._job_analytics:
            self._job_analytics.cleanup()
        if self._toast_manager:
            self._toast_manager.clear_all()


def add_view_menu_actions(view_menu, integration: MainWindowJobIntegration):
    """
    Add job and resource monitor actions to a View menu.

    Parameters
    ----------
    view_menu : QMenu
        The View menu to add actions to
    integration : MainWindowJobIntegration
        The integration instance
    """
    from PyQt6.QtGui import QAction

    view_menu.addSeparator()

    # Job Dashboard action
    job_action = QAction("&Job Dashboard", view_menu)
    job_action.setShortcut("Ctrl+J")
    job_action.setToolTip("Show job dashboard with active and queued jobs")
    job_action.triggered.connect(integration.show_job_dashboard)
    view_menu.addAction(job_action)

    # Resource Monitor action
    resource_action = QAction("&Resource Monitor", view_menu)
    resource_action.setShortcut("Ctrl+Shift+R")
    resource_action.setToolTip("Show CPU, memory, and GPU usage")
    resource_action.triggered.connect(integration.show_resource_monitor)
    view_menu.addAction(resource_action)

    # Job Analytics action
    analytics_action = QAction("Job &Analytics", view_menu)
    analytics_action.setShortcut("Ctrl+Shift+A")
    analytics_action.setToolTip("Show job statistics and analytics")
    analytics_action.triggered.connect(integration.show_job_analytics)
    view_menu.addAction(analytics_action)
