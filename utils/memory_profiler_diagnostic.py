"""
Memory profiler diagnostic tool for identifying memory collapse in sorting pipeline.

Provides checkpoints to track memory usage at each stage of the processing pipeline.
Use this to identify exact location of memory spikes.

Usage:
    from utils.memory_profiler_diagnostic import MemoryProfiler, memory_checkpoint

    profiler = MemoryProfiler("sorting_pipeline")

    with memory_checkpoint(profiler, "load_headers"):
        headers_df = pd.read_parquet(path)

    with memory_checkpoint(profiler, "compute_mapping"):
        mapping = compute_global_mapping(...)

    profiler.print_report()
"""

import gc
import time
import psutil
import tracemalloc
import logging
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryCheckpoint:
    """Single memory measurement checkpoint."""
    name: str
    timestamp: float
    rss_bytes: int
    available_bytes: int
    tracemalloc_current: int
    tracemalloc_peak: int
    delta_rss: int = 0
    duration: float = 0.0


@dataclass
class MemoryProfiler:
    """
    Memory profiler for tracking memory usage across pipeline stages.

    Attributes:
        name: Profiler name for identification
        checkpoints: List of memory checkpoints
        start_rss: RSS at profiler start
        warnings: List of warning messages
    """
    name: str
    checkpoints: List[MemoryCheckpoint] = field(default_factory=list)
    start_rss: int = 0
    start_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    _tracemalloc_started: bool = False

    def __post_init__(self):
        """Initialize profiler state."""
        self.start_time = time.time()
        self.start_rss = self._get_rss()

        # Start tracemalloc if not already running
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True

    def _get_rss(self) -> int:
        """Get current RSS (Resident Set Size) in bytes."""
        try:
            return psutil.Process().memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _get_available(self) -> int:
        """Get available system memory in bytes."""
        try:
            return psutil.virtual_memory().available
        except Exception:
            return 0

    def checkpoint(self, name: str) -> MemoryCheckpoint:
        """
        Record a memory checkpoint.

        Args:
            name: Checkpoint name/description

        Returns:
            MemoryCheckpoint with current memory state
        """
        current_rss = self._get_rss()
        available = self._get_available()

        # Get tracemalloc stats
        try:
            current, peak = tracemalloc.get_traced_memory()
        except Exception:
            current, peak = 0, 0

        # Calculate delta from last checkpoint or start
        if self.checkpoints:
            delta_rss = current_rss - self.checkpoints[-1].rss_bytes
        else:
            delta_rss = current_rss - self.start_rss

        cp = MemoryCheckpoint(
            name=name,
            timestamp=time.time(),
            rss_bytes=current_rss,
            available_bytes=available,
            tracemalloc_current=current,
            tracemalloc_peak=peak,
            delta_rss=delta_rss
        )

        self.checkpoints.append(cp)

        # Warn on large allocations
        delta_mb = delta_rss / (1024 * 1024)
        if delta_mb > 500:  # > 500 MB allocation
            warning = f"Large allocation at '{name}': {delta_mb:.1f} MB"
            self.warnings.append(warning)
            logger.warning(warning)

        # Warn on low available memory
        available_mb = available / (1024 * 1024)
        if available_mb < 1000:  # < 1 GB available
            warning = f"Low memory at '{name}': {available_mb:.0f} MB available"
            self.warnings.append(warning)
            logger.warning(warning)

        return cp

    def get_peak_checkpoint(self) -> Optional[MemoryCheckpoint]:
        """Get checkpoint with highest RSS usage."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda cp: cp.rss_bytes)

    def get_largest_allocation(self) -> Optional[MemoryCheckpoint]:
        """Get checkpoint with largest memory delta."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda cp: cp.delta_rss)

    def print_report(self) -> str:
        """Print formatted memory report."""
        lines = [
            f"\n{'='*60}",
            f"MEMORY PROFILE: {self.name}",
            f"{'='*60}",
            f"Start RSS: {self.start_rss / (1024**2):.1f} MB",
            f"Duration: {time.time() - self.start_time:.2f}s",
            f"\nCheckpoints:",
            f"{'-'*60}"
        ]

        for i, cp in enumerate(self.checkpoints):
            delta_sign = "+" if cp.delta_rss >= 0 else ""
            lines.append(
                f"  [{i+1}] {cp.name:30s} | "
                f"RSS: {cp.rss_bytes / (1024**2):8.1f} MB | "
                f"Delta: {delta_sign}{cp.delta_rss / (1024**2):7.1f} MB | "
                f"Avail: {cp.available_bytes / (1024**2):8.0f} MB"
            )

        # Summary
        if self.checkpoints:
            peak = self.get_peak_checkpoint()
            largest = self.get_largest_allocation()

            lines.extend([
                f"{'-'*60}",
                f"PEAK RSS: {peak.rss_bytes / (1024**2):.1f} MB at '{peak.name}'",
                f"LARGEST ALLOC: {largest.delta_rss / (1024**2):.1f} MB at '{largest.name}'"
            ])

        # Warnings
        if self.warnings:
            lines.extend([
                f"\nWARNINGS ({len(self.warnings)}):",
                *[f"  - {w}" for w in self.warnings]
            ])

        lines.append(f"{'='*60}\n")

        report = "\n".join(lines)
        print(report)
        return report

    def to_dict(self) -> Dict[str, Any]:
        """Export profile data as dictionary."""
        return {
            'name': self.name,
            'start_rss_bytes': self.start_rss,
            'duration_seconds': time.time() - self.start_time,
            'checkpoints': [
                {
                    'name': cp.name,
                    'rss_mb': cp.rss_bytes / (1024**2),
                    'delta_mb': cp.delta_rss / (1024**2),
                    'available_mb': cp.available_bytes / (1024**2)
                }
                for cp in self.checkpoints
            ],
            'warnings': self.warnings
        }

    def cleanup(self):
        """Stop tracemalloc if we started it."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False


@contextmanager
def memory_checkpoint(profiler: MemoryProfiler, name: str, gc_before: bool = False):
    """
    Context manager for memory checkpointing.

    Args:
        profiler: MemoryProfiler instance
        name: Checkpoint name
        gc_before: Run garbage collection before checkpoint

    Usage:
        with memory_checkpoint(profiler, "load_data"):
            data = load_large_data()
    """
    if gc_before:
        gc.collect()

    start_cp = profiler.checkpoint(f"{name}_start")
    start_time = time.time()

    try:
        yield profiler
    finally:
        end_cp = profiler.checkpoint(f"{name}_end")
        end_cp.duration = time.time() - start_time


def estimate_sorting_memory(
    n_traces: int,
    n_header_columns: int = 50,
    avg_column_bytes: int = 8
) -> Dict[str, float]:
    """
    Estimate memory requirements for sorting operation.

    Args:
        n_traces: Number of traces
        n_header_columns: Number of header columns
        avg_column_bytes: Average bytes per column value

    Returns:
        Dictionary with memory estimates in MB
    """
    # Global mapping array: int64 per trace
    mapping_mb = (n_traces * 8) / (1024**2)

    # Headers DataFrame: columns × traces × avg_bytes
    headers_mb = (n_header_columns * n_traces * avg_column_bytes) / (1024**2)

    # Sorted headers copy (during reorder)
    sorted_copy_mb = headers_mb

    # Sort indices per gather (assuming avg 100 traces/gather)
    n_gathers = n_traces // 100
    sort_indices_mb = (n_gathers * 100 * 8) / (1024**2)

    # Total peak (headers + mapping + sorted copy)
    peak_mb = headers_mb + mapping_mb + sorted_copy_mb

    return {
        'mapping_array_mb': mapping_mb,
        'headers_df_mb': headers_mb,
        'sorted_copy_mb': sorted_copy_mb,
        'sort_indices_mb': sort_indices_mb,
        'estimated_peak_mb': peak_mb,
        'recommended_available_mb': peak_mb * 1.5  # 50% safety margin
    }


def check_memory_budget(
    n_traces: int,
    sorting_enabled: bool = True,
    safety_factor: float = 0.7
) -> tuple:
    """
    Pre-flight memory check before processing.

    Args:
        n_traces: Number of traces to process
        sorting_enabled: Whether sorting will be performed
        safety_factor: Fraction of available memory to use (default 70%)

    Returns:
        Tuple of (is_safe, available_mb, required_mb, message)
    """
    available_mb = psutil.virtual_memory().available / (1024**2)

    if sorting_enabled:
        estimates = estimate_sorting_memory(n_traces)
        required_mb = estimates['estimated_peak_mb']
    else:
        # Without sorting: just processing memory
        required_mb = (n_traces * 8) / (1024**2) * 2  # Rough estimate

    safe_available = available_mb * safety_factor
    is_safe = required_mb < safe_available

    if is_safe:
        message = f"Memory OK: {required_mb:.0f} MB required, {available_mb:.0f} MB available"
    else:
        message = (
            f"MEMORY WARNING: Estimated {required_mb:.0f} MB required, "
            f"only {available_mb:.0f} MB available. "
            f"Consider: (1) Disable sorting, (2) Process smaller batches, "
            f"(3) Close other applications."
        )

    return is_safe, available_mb, required_mb, message


class SortingMemoryGuard:
    """
    Memory guard that monitors and prevents OOM during sorting.

    Usage:
        guard = SortingMemoryGuard(n_traces=1_000_000)
        guard.check_can_proceed()  # Raises MemoryError if unsafe

        # During processing
        guard.check_threshold("after_headers_load")
    """

    def __init__(
        self,
        n_traces: int,
        threshold_percentage: float = 85.0,
        abort_percentage: float = 95.0
    ):
        """
        Initialize memory guard.

        Args:
            n_traces: Number of traces being processed
            threshold_percentage: Warn when system memory usage exceeds this
            abort_percentage: Abort when system memory usage exceeds this
        """
        self.n_traces = n_traces
        self.threshold_percentage = threshold_percentage
        self.abort_percentage = abort_percentage
        self.start_rss = psutil.Process().memory_info().rss

    def get_system_memory_percent(self) -> float:
        """Get current system memory usage percentage."""
        return psutil.virtual_memory().percent

    def get_process_memory_mb(self) -> float:
        """Get current process memory in MB."""
        return psutil.Process().memory_info().rss / (1024**2)

    def check_can_proceed(self) -> None:
        """
        Pre-flight check. Raises MemoryError if insufficient memory.
        """
        is_safe, available_mb, required_mb, message = check_memory_budget(
            self.n_traces, sorting_enabled=True
        )

        if not is_safe:
            raise MemoryError(message)

        logger.info(message)

    def check_threshold(self, stage: str) -> None:
        """
        Check memory during processing. Raises MemoryError if critical.

        Args:
            stage: Current processing stage name
        """
        mem_percent = self.get_system_memory_percent()
        process_mb = self.get_process_memory_mb()

        if mem_percent >= self.abort_percentage:
            gc.collect()  # Emergency GC
            mem_percent = self.get_system_memory_percent()

            if mem_percent >= self.abort_percentage:
                raise MemoryError(
                    f"Memory critical at '{stage}': {mem_percent:.1f}% system usage, "
                    f"process using {process_mb:.0f} MB. Aborting to prevent crash."
                )

        elif mem_percent >= self.threshold_percentage:
            logger.warning(
                f"Memory high at '{stage}': {mem_percent:.1f}% system usage, "
                f"process using {process_mb:.0f} MB"
            )
