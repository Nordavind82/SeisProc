"""
Resource Monitor for Job Processing

Monitors system resources during job execution and provides
alerts when resources are constrained.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
from uuid import UUID

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    @property
    def memory_total_mb(self) -> float:
        """Total memory in MB."""
        return self.memory_used_mb + self.memory_available_mb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'cpu_count': self.cpu_count,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'memory_percent': self.memory_percent,
            'disk_read_mb': self.disk_read_mb,
            'disk_write_mb': self.disk_write_mb,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'gpu_memory_total_mb': self.gpu_memory_total_mb,
            'gpu_utilization': self.gpu_utilization,
        }


@dataclass
class ResourceAlert:
    """Alert for resource constraint."""
    timestamp: datetime
    resource_type: str  # 'memory', 'cpu', 'disk', 'gpu'
    severity: str  # 'warning', 'critical'
    message: str
    current_value: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'resource_type': self.resource_type,
            'severity': self.severity,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
        }


@dataclass
class ResourceThresholds:
    """Thresholds for resource alerts."""
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    cpu_warning_percent: float = 90.0
    cpu_critical_percent: float = 99.0
    gpu_memory_warning_percent: float = 85.0
    gpu_memory_critical_percent: float = 95.0


class ResourceMonitor:
    """
    Monitors system resources during job execution.

    Provides:
    - Periodic resource snapshots
    - Alerts when thresholds are exceeded
    - Historical resource usage tracking
    - GPU monitoring when available

    Usage
    -----
    >>> monitor = ResourceMonitor()
    >>> monitor.start()
    >>>
    >>> # Get current snapshot
    >>> snapshot = monitor.get_current_snapshot()
    >>> print(f"Memory: {snapshot.memory_percent:.1f}%")
    >>>
    >>> # Get alerts
    >>> alerts = monitor.get_alerts()
    >>>
    >>> monitor.stop()
    """

    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        sample_interval: float = 2.0,
        history_size: int = 300,  # Keep 10 minutes at 2s interval
        alert_callback: Optional[Callable[[ResourceAlert], None]] = None,
    ):
        """
        Initialize resource monitor.

        Parameters
        ----------
        thresholds : ResourceThresholds, optional
            Alert thresholds
        sample_interval : float
            Sampling interval in seconds
        history_size : int
            Number of samples to keep in history
        alert_callback : callable, optional
            Callback for alerts
        """
        self._thresholds = thresholds or ResourceThresholds()
        self._sample_interval = sample_interval
        self._history_size = history_size
        self._alert_callback = alert_callback

        self._history: List[ResourceSnapshot] = []
        self._alerts: List[ResourceAlert] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Disk I/O tracking
        self._last_disk_io = None

        # GPU monitoring
        self._gpu_available = self._check_gpu_available()

    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def start(self):
        """Start resource monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitor started")

    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Resource monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                snapshot = self._collect_snapshot()
                self._add_snapshot(snapshot)
                self._check_thresholds(snapshot)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")

            time.sleep(self._sample_interval)

    def _collect_snapshot(self) -> ResourceSnapshot:
        """Collect current resource snapshot."""
        if not PSUTIL_AVAILABLE:
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                cpu_count=1,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                memory_percent=0.0,
            )

        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count() or 1

        # Memory
        mem = psutil.virtual_memory()
        memory_used_mb = mem.used / (1024 * 1024)
        memory_available_mb = mem.available / (1024 * 1024)
        memory_percent = mem.percent

        # Disk I/O
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io:
                elapsed = self._sample_interval
                disk_read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024) / elapsed
                disk_write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024) / elapsed
            self._last_disk_io = disk_io
        except Exception:
            pass

        # GPU
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_utilization = None

        if self._gpu_available:
            try:
                import torch
                gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                # Note: torch doesn't provide utilization directly
            except Exception:
                pass

        return ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            memory_percent=memory_percent,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
        )

    def _add_snapshot(self, snapshot: ResourceSnapshot):
        """Add snapshot to history."""
        with self._lock:
            self._history.append(snapshot)
            # Trim history
            while len(self._history) > self._history_size:
                self._history.pop(0)

    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check thresholds and generate alerts."""
        # Memory
        if snapshot.memory_percent >= self._thresholds.memory_critical_percent:
            self._add_alert(ResourceAlert(
                timestamp=snapshot.timestamp,
                resource_type='memory',
                severity='critical',
                message=f"Memory usage critical: {snapshot.memory_percent:.1f}%",
                current_value=snapshot.memory_percent,
                threshold=self._thresholds.memory_critical_percent,
            ))
        elif snapshot.memory_percent >= self._thresholds.memory_warning_percent:
            self._add_alert(ResourceAlert(
                timestamp=snapshot.timestamp,
                resource_type='memory',
                severity='warning',
                message=f"Memory usage high: {snapshot.memory_percent:.1f}%",
                current_value=snapshot.memory_percent,
                threshold=self._thresholds.memory_warning_percent,
            ))

        # CPU
        if snapshot.cpu_percent >= self._thresholds.cpu_critical_percent:
            self._add_alert(ResourceAlert(
                timestamp=snapshot.timestamp,
                resource_type='cpu',
                severity='critical',
                message=f"CPU usage critical: {snapshot.cpu_percent:.1f}%",
                current_value=snapshot.cpu_percent,
                threshold=self._thresholds.cpu_critical_percent,
            ))
        elif snapshot.cpu_percent >= self._thresholds.cpu_warning_percent:
            self._add_alert(ResourceAlert(
                timestamp=snapshot.timestamp,
                resource_type='cpu',
                severity='warning',
                message=f"CPU usage high: {snapshot.cpu_percent:.1f}%",
                current_value=snapshot.cpu_percent,
                threshold=self._thresholds.cpu_warning_percent,
            ))

        # GPU
        if snapshot.gpu_memory_used_mb and snapshot.gpu_memory_total_mb:
            gpu_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
            if gpu_percent >= self._thresholds.gpu_memory_critical_percent:
                self._add_alert(ResourceAlert(
                    timestamp=snapshot.timestamp,
                    resource_type='gpu',
                    severity='critical',
                    message=f"GPU memory critical: {gpu_percent:.1f}%",
                    current_value=gpu_percent,
                    threshold=self._thresholds.gpu_memory_critical_percent,
                ))
            elif gpu_percent >= self._thresholds.gpu_memory_warning_percent:
                self._add_alert(ResourceAlert(
                    timestamp=snapshot.timestamp,
                    resource_type='gpu',
                    severity='warning',
                    message=f"GPU memory high: {gpu_percent:.1f}%",
                    current_value=gpu_percent,
                    threshold=self._thresholds.gpu_memory_warning_percent,
                ))

    def _add_alert(self, alert: ResourceAlert):
        """Add alert and call callback."""
        with self._lock:
            self._alerts.append(alert)
            # Keep last 100 alerts
            while len(self._alerts) > 100:
                self._alerts.pop(0)

        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return self._collect_snapshot()

    def get_history(self, last_n: Optional[int] = None) -> List[ResourceSnapshot]:
        """Get resource history."""
        with self._lock:
            if last_n:
                return self._history[-last_n:]
            return list(self._history)

    def get_alerts(self, last_n: Optional[int] = None) -> List[ResourceAlert]:
        """Get alerts."""
        with self._lock:
            if last_n:
                return self._alerts[-last_n:]
            return list(self._alerts)

    def clear_alerts(self):
        """Clear alert history."""
        with self._lock:
            self._alerts.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        with self._lock:
            if not self._history:
                return {}

            cpu_values = [s.cpu_percent for s in self._history]
            memory_values = [s.memory_percent for s in self._history]

            return {
                'samples': len(self._history),
                'sample_interval': self._sample_interval,
                'cpu': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0,
                },
                'memory': {
                    'current': memory_values[-1] if memory_values else 0,
                    'avg': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0,
                },
                'alerts': {
                    'total': len(self._alerts),
                    'warnings': sum(1 for a in self._alerts if a.severity == 'warning'),
                    'critical': sum(1 for a in self._alerts if a.severity == 'critical'),
                },
            }


# Global monitor instance
_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor(
    thresholds: Optional[ResourceThresholds] = None,
) -> ResourceMonitor:
    """Get the global resource monitor."""
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor(thresholds=thresholds)
    return _monitor


def start_monitoring():
    """Start the global resource monitor."""
    monitor = get_resource_monitor()
    monitor.start()


def stop_monitoring():
    """Stop the global resource monitor."""
    global _monitor
    if _monitor:
        _monitor.stop()


def get_current_resources() -> ResourceSnapshot:
    """Get current resource snapshot."""
    monitor = get_resource_monitor()
    return monitor.get_current_snapshot()
