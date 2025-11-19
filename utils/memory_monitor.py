"""
Memory usage monitoring module.

Tracks application memory usage in real-time with configurable thresholds
and background monitoring thread.
"""
import psutil
import threading
import time
from typing import Optional, Callable
from PyQt6.QtCore import QObject, pyqtSignal


class MemoryMonitor(QObject):
    """
    Monitors application memory usage with background thread.

    Provides real-time memory statistics and emits signals when
    usage exceeds configurable thresholds. Platform-independent
    using psutil library.

    Signals:
        threshold_exceeded: Emitted when memory usage exceeds threshold
                          Args: (current_bytes, threshold_bytes, percentage)
        memory_updated: Emitted periodically with current usage
                       Args: (current_bytes, available_bytes, percentage)
    """

    # Qt signals for memory events
    # Using 'qint64' to handle large memory values (> 2GB)
    threshold_exceeded = pyqtSignal('qint64', 'qint64', float)  # current, threshold, percentage
    memory_updated = pyqtSignal('qint64', 'qint64', float)      # current, available, percentage

    def __init__(self,
                 update_interval: float = 2.0,
                 threshold_bytes: Optional[int] = None,
                 threshold_percentage: Optional[float] = None):
        """
        Initialize memory monitor.

        Args:
            update_interval: Update frequency in seconds (default 2.0)
            threshold_bytes: Memory threshold in bytes (optional)
            threshold_percentage: Memory threshold as percentage of total RAM (optional)
                                If both thresholds provided, uses threshold_bytes
        """
        super().__init__()

        self.update_interval = update_interval
        self.threshold_bytes = threshold_bytes
        self.threshold_percentage = threshold_percentage

        # Get process handle
        self._process = psutil.Process()

        # Thread control
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # State tracking
        self._threshold_exceeded_flag = False
        self._last_usage = 0
        self._last_available = 0
        self._last_percentage = 0.0

        # Start monitoring thread
        self._start_monitoring()

    def get_current_usage(self) -> int:
        """
        Get current application memory usage in bytes.

        Returns RSS (Resident Set Size) - actual physical memory used.

        Returns:
            Memory usage in bytes
        """
        with self._lock:
            try:
                mem_info = self._process.memory_info()
                return mem_info.rss  # Resident Set Size (physical memory)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0

    def get_available_memory(self) -> int:
        """
        Get system available RAM in bytes.

        Returns:
            Available memory in bytes
        """
        try:
            mem = psutil.virtual_memory()
            return mem.available
        except Exception:
            return 0

    def get_total_memory(self) -> int:
        """
        Get total system RAM in bytes.

        Returns:
            Total memory in bytes
        """
        try:
            mem = psutil.virtual_memory()
            return mem.total
        except Exception:
            return 0

    def get_usage_percentage(self) -> float:
        """
        Get application memory usage as percentage of total system RAM.

        Returns:
            Percentage (0.0 to 100.0)
        """
        current = self.get_current_usage()
        total = self.get_total_memory()

        if total == 0:
            return 0.0

        return (current / total) * 100.0

    def get_statistics(self) -> dict:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with keys:
                - current_bytes: Current app memory usage
                - available_bytes: System available memory
                - total_bytes: Total system memory
                - usage_percentage: App usage as % of total
                - system_usage_percentage: System-wide memory usage %
                - current_mb: Current usage in MB
                - available_mb: Available memory in MB
                - total_mb: Total memory in MB
        """
        current = self.get_current_usage()
        available = self.get_available_memory()
        total = self.get_total_memory()
        usage_pct = self.get_usage_percentage()

        system_mem = psutil.virtual_memory()
        system_usage_pct = system_mem.percent

        return {
            'current_bytes': current,
            'available_bytes': available,
            'total_bytes': total,
            'usage_percentage': usage_pct,
            'system_usage_percentage': system_usage_pct,
            'current_mb': current / (1024 * 1024),
            'available_mb': available / (1024 * 1024),
            'total_mb': total / (1024 * 1024)
        }

    def set_threshold_bytes(self, threshold: int) -> None:
        """
        Set memory threshold in bytes.

        Args:
            threshold: Memory threshold in bytes
        """
        with self._lock:
            self.threshold_bytes = threshold
            self._threshold_exceeded_flag = False

    def set_threshold_percentage(self, percentage: float) -> None:
        """
        Set memory threshold as percentage of total RAM.

        Args:
            percentage: Threshold percentage (0.0 to 100.0)
        """
        with self._lock:
            self.threshold_percentage = percentage
            self._threshold_exceeded_flag = False

    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()

    def _monitor_worker(self) -> None:
        """Background worker thread that monitors memory usage."""
        while not self._stop_event.is_set():
            try:
                # Get current memory stats
                current = self.get_current_usage()
                available = self.get_available_memory()
                total = self.get_total_memory()
                percentage = (current / total * 100.0) if total > 0 else 0.0

                # Store for fast access
                with self._lock:
                    self._last_usage = current
                    self._last_available = available
                    self._last_percentage = percentage

                # Emit update signal
                self.memory_updated.emit(current, available, percentage)

                # Check threshold
                threshold = self._get_effective_threshold()
                if threshold is not None and current > threshold:
                    with self._lock:
                        # Only emit once when threshold first exceeded
                        if not self._threshold_exceeded_flag:
                            self._threshold_exceeded_flag = True
                            self.threshold_exceeded.emit(current, threshold, percentage)
                else:
                    with self._lock:
                        # Reset flag when back below threshold
                        self._threshold_exceeded_flag = False

            except Exception as e:
                # Silently handle errors in monitoring thread
                pass

            # Sleep with interruptible wait
            self._stop_event.wait(self.update_interval)

    def _get_effective_threshold(self) -> Optional[int]:
        """
        Get effective threshold in bytes.

        Returns:
            Threshold in bytes or None if no threshold set
        """
        with self._lock:
            if self.threshold_bytes is not None:
                return self.threshold_bytes

            if self.threshold_percentage is not None:
                total = self.get_total_memory()
                return int(total * self.threshold_percentage / 100.0)

            return None

    def stop(self) -> None:
        """
        Stop monitoring and clean up thread.

        Should be called before destroying the monitor.
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def is_monitoring(self) -> bool:
        """
        Check if monitoring thread is running.

        Returns:
            True if monitoring active
        """
        return (self._monitor_thread is not None and
                self._monitor_thread.is_alive())

    def get_cached_statistics(self) -> dict:
        """
        Get last cached memory statistics without querying system.

        Faster than get_statistics() as it uses cached values.

        Returns:
            Dictionary with cached statistics
        """
        with self._lock:
            total = self.get_total_memory()
            return {
                'current_bytes': self._last_usage,
                'available_bytes': self._last_available,
                'total_bytes': total,
                'usage_percentage': self._last_percentage,
                'current_mb': self._last_usage / (1024 * 1024),
                'available_mb': self._last_available / (1024 * 1024),
                'total_mb': total / (1024 * 1024)
            }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        bytes_value: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    if bytes_value < 1024:
        return f"{bytes_value} B"
    elif bytes_value < 1024 ** 2:
        return f"{bytes_value / 1024:.1f} KB"
    elif bytes_value < 1024 ** 3:
        return f"{bytes_value / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_value / (1024 ** 3):.2f} GB"
