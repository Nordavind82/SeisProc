"""
Custom widgets for SeisProc views.
"""

from .kernel_selector import KernelSelectorWidget, KernelBackendCombo
from .job_card import JobCardWidget
from .job_queue_widget import JobQueueWidget
from .resource_monitor_widget import (
    ResourceMonitorWidget,
    CompactResourceMonitorWidget,
    ResourceGauge,
    AlertIndicator,
)
from .job_analytics_widget import (
    JobAnalyticsWidget,
    StatCard,
    SimpleBarChart,
    SimplePieChart,
)
from .toast_notification import (
    Toast,
    ToastType,
    ToastManager,
    AlertToastBridge,
)

__all__ = [
    'KernelSelectorWidget',
    'KernelBackendCombo',
    'JobCardWidget',
    'JobQueueWidget',
    'ResourceMonitorWidget',
    'CompactResourceMonitorWidget',
    'ResourceGauge',
    'AlertIndicator',
    'JobAnalyticsWidget',
    'StatCard',
    'SimpleBarChart',
    'SimplePieChart',
    'Toast',
    'ToastType',
    'ToastManager',
    'AlertToastBridge',
]
