"""
SEGY Import/Export Worker Threads

QThread workers for running SEGY I/O operations in the background with
full integration into the centralized job dashboard.

These workers follow the same pattern as QCBatchWorker for consistency:
- Submit job to JobManager
- Emit queued/started/progress/completed signals via QtBridge
- All progress shown in JobDashboardWidget, no built-in dialogs
"""

import logging
import time
from pathlib import Path
from typing import Optional, Any, Dict
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

from models.job import Job, JobType
from models.job_config import JobConfig
from models.job_progress import ProgressUpdate
from .job_manager import get_job_manager
from .qt_bridge import get_job_bridge
from .segy_job_adapter import SEGYImportJobAdapter, SEGYExportJobAdapter
from .cancellation import CancellationError

logger = logging.getLogger(__name__)


@dataclass
class SEGYImportResult:
    """Result from SEGY import operation."""
    success: bool
    output_dir: str
    traces_path: str
    headers_path: str
    n_traces: int
    elapsed_time: float
    error: Optional[str] = None


@dataclass
class SEGYExportResult:
    """Result from SEGY export operation."""
    success: bool
    output_path: str
    n_traces: int
    elapsed_time: float
    error: Optional[str] = None


class SEGYImportWorker(QThread):
    """
    Worker thread for running SEGY import in background.

    Integrates with centralized job dashboard - no built-in progress dialogs.

    Usage:
        from utils.segy_import.multiprocess_import.coordinator import ImportConfig

        config = ImportConfig(segy_path=..., output_dir=..., header_mapping=...)
        worker = SEGYImportWorker(config)
        worker.finished_with_result.connect(on_complete)
        worker.error_occurred.connect(on_error)
        worker.start()

        # Progress shown in JobDashboardWidget automatically
    """

    # Signals for external listeners
    progress_updated = pyqtSignal(object)  # ImportProgress
    finished_with_result = pyqtSignal(object)  # SEGYImportResult
    error_occurred = pyqtSignal(str)

    def __init__(self, import_config: Any, job_name: Optional[str] = None, parent=None):
        """
        Initialize import worker.

        Parameters
        ----------
        import_config : ImportConfig
            SEGY import configuration from multiprocess_import.coordinator
        job_name : str, optional
            Human-readable job name (defaults to filename)
        parent : QObject, optional
            Parent object
        """
        super().__init__(parent)
        self._import_config = import_config
        self._job_name = job_name or Path(import_config.segy_path).name
        self._adapter: Optional[SEGYImportJobAdapter] = None
        self._coordinator = None  # Store coordinator for cancellation
        self._job: Optional[Job] = None
        self._job_manager = get_job_manager()
        self._qt_bridge = get_job_bridge()
        self._start_time: Optional[float] = None
        self._cancelled = False

    @property
    def job(self) -> Optional[Job]:
        """Get the associated job."""
        return self._job

    @property
    def job_id(self):
        """Get job ID."""
        return self._job.id if self._job else None

    def run(self):
        """Execute SEGY import in thread."""
        self._start_time = time.time()

        try:
            # Create adapter - minimal overhead
            t0 = time.time()
            self._adapter = SEGYImportJobAdapter(
                self._import_config,
                job_name=self._job_name
            )
            logger.debug(f"Adapter creation: {time.time() - t0:.3f}s")

            # Submit job (creates job in JobManager)
            t0 = time.time()
            self._job = self._adapter.submit()
            logger.debug(f"Job submission: {time.time() - t0:.3f}s")

            # Emit queued signal for Dashboard
            # Note: submit_job doesn't trigger callbacks, so we emit manually
            if self._qt_bridge:
                try:
                    self._qt_bridge.signals.emit_job_queued(self._job)
                except Exception:
                    pass

            # Start job - this triggers on_job_started callback via JobManager
            # which emits job_started signal through the bridge automatically
            self._job_manager.start_job(self._job.id)

            # Run import with progress callback
            logger.info(f"Starting coordinator.run() at {time.time() - self._start_time:.1f}s")
            result = self._run_import()
            logger.info(f"coordinator.run() completed in {time.time() - self._start_time:.1f}s")

            if result.success:
                # Success - complete_job triggers on_job_completed callback
                # which emits signals through the bridge automatically
                elapsed = time.time() - self._start_time if self._start_time else 0
                self._job_manager.complete_job(
                    self._job.id,
                    result={
                        'n_traces': result.n_traces,
                        'traces_path': result.traces_path,
                        'headers_path': result.headers_path,
                        'elapsed_time': elapsed,
                    },
                )

                import_result = SEGYImportResult(
                    success=True,
                    output_dir=result.output_dir,
                    traces_path=result.traces_path,
                    headers_path=result.headers_path,
                    n_traces=result.n_traces,
                    elapsed_time=elapsed,
                )
                self.finished_with_result.emit(import_result)
            else:
                # Failure - fail_job/cancel_job triggers callbacks
                # which emit signals through the bridge automatically
                error_msg = result.error or "Import failed"
                if "cancelled" in error_msg.lower():
                    self._job_manager.cancel_job(self._job.id)
                    self._job_manager.finalize_cancellation(self._job.id)
                else:
                    self._job_manager.fail_job(self._job.id, error=error_msg)

                self.error_occurred.emit(error_msg)

        except CancellationError:
            # Handle cancellation - finalize triggers callback
            if self._job:
                self._job_manager.finalize_cancellation(self._job.id)
            self.error_occurred.emit("Import cancelled by user")

        except Exception as e:
            logger.exception("SEGY import worker error")
            if self._job:
                # fail_job triggers on_job_failed callback
                self._job_manager.fail_job(self._job.id, error=str(e))
            self.error_occurred.emit(str(e))

    def _run_import(self) -> Any:
        """Run the actual import and handle progress."""
        from utils.segy_import.multiprocess_import.coordinator import (
            ParallelImportCoordinator,
            ImportProgress,
        )

        # Create coordinator and store for cancellation
        self._coordinator = ParallelImportCoordinator(self._import_config)

        # Throttle progress updates to reduce signal emission overhead
        # But ensure updates happen often enough to show responsiveness
        last_emit_time = 0.0
        MIN_EMIT_INTERVAL = 0.5  # 2 Hz max update rate
        last_traces = 0

        # Rate history for trend tracking
        rate_history = []  # List of (elapsed, traces, rate)
        RATE_HISTORY_SIZE = 5

        # Progress callback that emits to dashboard with throttling
        def on_progress(progress: ImportProgress):
            nonlocal last_emit_time, last_traces, rate_history

            current_time = progress.elapsed_time
            time_passed = (current_time - last_emit_time) >= MIN_EMIT_INTERVAL

            # Always emit on phase change
            phase_changed = not hasattr(on_progress, '_last_phase') or on_progress._last_phase != progress.phase
            on_progress._last_phase = progress.phase

            # Emit if: phase changed OR enough time passed (regardless of trace count)
            # This ensures UI stays responsive even during worker initialization
            if not (phase_changed or time_passed):
                return

            last_emit_time = current_time
            prev_traces = last_traces
            last_traces = progress.current_traces

            # Calculate current rate and track history
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            percent = (progress.current_traces / progress.total_traces * 100) if progress.total_traces > 0 else 0
            avg_rate = progress.current_traces / progress.elapsed_time if progress.elapsed_time > 0 else 0

            # Calculate instant rate (traces since last update)
            instant_rate = (progress.current_traces - prev_traces) / MIN_EMIT_INTERVAL if prev_traces > 0 else avg_rate

            # Track rate history for trend analysis
            rate_history.append((progress.elapsed_time, progress.current_traces, instant_rate))
            if len(rate_history) > RATE_HISTORY_SIZE:
                rate_history.pop(0)

            # Determine rate trend
            rate_trend = "stable"
            if len(rate_history) >= 3:
                recent_rates = [r[2] for r in rate_history[-3:]]
                older_rates = [r[2] for r in rate_history[:2]] if len(rate_history) >= 3 else recent_rates
                recent_avg = sum(recent_rates) / len(recent_rates)
                older_avg = sum(older_rates) / len(older_rates) if older_rates else recent_avg
                if recent_avg > older_avg * 1.1:
                    rate_trend = "increasing"
                elif recent_avg < older_avg * 0.85:
                    rate_trend = "decreasing"

            # Build worker progress summary for log
            worker_summary = ""
            if progress.worker_progress:
                active_workers = [
                    f"W{wid}:{done:,}"
                    for wid, done in sorted(progress.worker_progress.items())
                    if done < (progress.total_traces // len(progress.worker_progress))  # Only show incomplete
                ]
                if active_workers and len(active_workers) <= 5:
                    worker_summary = f" [{', '.join(active_workers)}]"

            # I/O metrics from coordinator
            io_rate_mbps = getattr(progress, 'io_rate_mbps', 0.0)
            io_rate_trend = getattr(progress, 'io_rate_trend', '')
            io_stall = getattr(progress, 'io_stall_detected', False)
            io_status = f", I/O={io_rate_mbps:.0f} MB/s {io_rate_trend}" if io_rate_mbps > 0 else ""
            stall_warning = " ⚠️ I/O STALL" if io_stall else ""

            # Debug log progress updates with timestamp, rate info, I/O metrics, and trend
            logger.debug(
                f"[{timestamp}] Import progress: phase={progress.phase}, "
                f"traces={progress.current_traces:,}/{progress.total_traces:,} ({percent:.1f}%), "
                f"workers={progress.active_workers}, rate={avg_rate:,.0f} tr/s ({rate_trend}){io_status}"
                f", elapsed={progress.elapsed_time:.1f}s, eta={progress.eta_seconds:.1f}s{stall_warning}{worker_summary}"
            )

            # Emit to local listeners (lightweight)
            self.progress_updated.emit(progress)

            # Emit to dashboard via bridge
            if self._qt_bridge and self._job:
                try:
                    # Calculate segment (gather-like) progress from worker progress
                    # Each segment represents a portion of the file being processed
                    total_segments = len(progress.worker_progress) if progress.worker_progress else 0
                    completed_segments = sum(
                        1 for wid, done in progress.worker_progress.items()
                        if done >= (progress.total_traces // max(1, total_segments)) * 0.99
                    ) if progress.worker_progress else 0

                    self._qt_bridge.signals.job_progress.emit(
                        self._job.id,
                        {
                            'percent': percent,
                            'message': f"Phase: {progress.phase}",
                            'phase': progress.phase,
                            'current_traces': progress.current_traces,
                            'total_traces': progress.total_traces,
                            'current_gathers': completed_segments,
                            'total_gathers': total_segments,
                            'active_workers': progress.active_workers,
                            'traces_per_sec': avg_rate,
                            'instant_rate': instant_rate,
                            'rate_trend': rate_trend,
                            'eta_seconds': progress.eta_seconds,
                            'worker_progress': progress.worker_progress,
                            # I/O metrics for storage performance monitoring
                            'io_rate_mbps': io_rate_mbps,
                            'io_rate_trend': io_rate_trend,
                            'io_stall_detected': io_stall,
                        }
                    )
                except Exception:
                    pass

        # Run full import pipeline
        return self._coordinator.run(progress_callback=on_progress)

    def cancel(self):
        """Request cancellation of the import."""
        self._cancelled = True
        if self._coordinator:
            self._coordinator.cancel()
        if self._job:
            self._job_manager.cancel_job(self._job.id)


class SEGYExportWorker(QThread):
    """
    Worker thread for running SEGY export in background.

    Integrates with centralized job dashboard - no built-in progress dialogs.

    Usage:
        worker = SEGYExportWorker(
            traces_path='/path/to/traces.zarr',
            headers_path='/path/to/headers.parquet',
            output_path='/path/to/output.sgy',
            original_segy_path='/path/to/original.sgy',
        )
        worker.finished_with_result.connect(on_complete)
        worker.error_occurred.connect(on_error)
        worker.start()

        # Progress shown in JobDashboardWidget automatically
    """

    # Signals for external listeners
    progress_updated = pyqtSignal(object)  # dict with progress info
    finished_with_result = pyqtSignal(object)  # SEGYExportResult
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        traces_path: str,
        headers_path: str,
        output_path: str,
        original_segy_path: str,
        chunk_size: int = 5000,
        headers_df: Any = None,
        job_name: Optional[str] = None,
        parent=None
    ):
        """
        Initialize export worker.

        Parameters
        ----------
        traces_path : str
            Path to traces Zarr array
        headers_path : str
            Path to headers parquet file
        output_path : str
            Path for output SEGY file
        original_segy_path : str
            Path to original SEGY file (for header template)
        chunk_size : int
            Number of traces per chunk (default 5000)
        headers_df : DataFrame, optional
            Sorted headers DataFrame (if sorting was applied)
        job_name : str, optional
            Human-readable job name
        parent : QObject, optional
            Parent object
        """
        super().__init__(parent)
        self._traces_path = traces_path
        self._headers_path = headers_path
        self._output_path = output_path
        self._original_segy_path = original_segy_path
        self._chunk_size = chunk_size
        self._headers_df = headers_df
        self._job_name = job_name or Path(output_path).name
        self._job: Optional[Job] = None
        self._job_manager = get_job_manager()
        self._qt_bridge = get_job_bridge()
        self._start_time: Optional[float] = None
        self._cancelled = False

    @property
    def job(self) -> Optional[Job]:
        """Get the associated job."""
        return self._job

    @property
    def job_id(self):
        """Get job ID."""
        return self._job.id if self._job else None

    def run(self):
        """Execute SEGY export in thread."""
        self._start_time = time.time()

        try:
            # Estimate file size for job config
            try:
                import zarr
                z = zarr.open(self._traces_path, 'r')
                n_samples, n_traces = z.shape
                # Estimate: 4 bytes per sample + 240 bytes header per trace
                estimated_size_mb = int((n_samples * 4 + 240) * n_traces / (1024 * 1024))
            except Exception:
                n_traces = 0
                estimated_size_mb = 100

            # Submit job to JobManager
            self._job = self._job_manager.submit_job(
                name=f"Export: {self._job_name}",
                job_type=JobType.SEGY_EXPORT,
                config=JobConfig.for_segy_import(file_size_mb=estimated_size_mb),
                custom_config={
                    'traces_path': self._traces_path,
                    'headers_path': self._headers_path,
                    'output_path': self._output_path,
                    'original_segy_path': self._original_segy_path,
                },
            )

            # Emit queued signal (submit_job doesn't trigger callbacks)
            if self._qt_bridge:
                try:
                    self._qt_bridge.signals.emit_job_queued(self._job)
                except Exception:
                    pass

            # Start job - triggers on_job_started callback via JobManager
            self._job_manager.start_job(self._job.id)

            # Run export
            self._run_export()

            # Success - complete_job triggers on_job_completed callback
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._job_manager.complete_job(
                self._job.id,
                result={
                    'output_path': self._output_path,
                    'n_traces': n_traces,
                    'elapsed_time': elapsed,
                },
            )

            export_result = SEGYExportResult(
                success=True,
                output_path=self._output_path,
                n_traces=n_traces,
                elapsed_time=elapsed,
            )
            self.finished_with_result.emit(export_result)

        except Exception as e:
            logger.exception("SEGY export worker error")
            error_msg = str(e)

            if self._job:
                if self._cancelled or "cancelled" in error_msg.lower():
                    self._job_manager.cancel_job(self._job.id)
                    self._job_manager.finalize_cancellation(self._job.id)
                else:
                    # fail_job triggers on_job_failed callback
                    self._job_manager.fail_job(self._job.id, error=error_msg)

            self.error_occurred.emit(error_msg)

    def _run_export(self):
        """Run the actual export with progress reporting."""
        from utils.segy_import.segy_export import export_from_zarr_chunked

        # Throttle progress updates to reduce overhead
        last_emit_time = [0.0]  # Use list for nonlocal in nested scope
        MIN_EMIT_INTERVAL = 0.5  # 2 Hz max

        def on_progress(current: int, total: int, time_remaining: float):
            # Check cancellation
            if self._cancelled:
                raise InterruptedError("Export cancelled by user")

            # Throttle: skip if not enough time passed
            elapsed = time.time() - self._start_time if self._start_time else 0.001
            time_passed = (elapsed - last_emit_time[0]) >= MIN_EMIT_INTERVAL

            if not time_passed:
                return

            last_emit_time[0] = elapsed

            # Calculate rate
            traces_per_sec = current / elapsed if elapsed > 0 else 0

            # Emit to local listeners
            progress_info = {
                'current_traces': current,
                'total_traces': total,
                'traces_per_sec': traces_per_sec,
                'eta_seconds': time_remaining,
            }
            self.progress_updated.emit(progress_info)

            # Emit to dashboard
            if self._qt_bridge and self._job:
                try:
                    self._qt_bridge.signals.job_progress.emit(
                        self._job.id,
                        {
                            'percent': (current / total * 100) if total > 0 else 0,
                            'message': 'Exporting traces',
                            'phase': 'exporting',
                            'current_traces': current,
                            'total_traces': total,
                            'active_workers': 1,
                            'traces_per_sec': traces_per_sec,
                            'eta_seconds': time_remaining,
                        }
                    )
                except Exception:
                    pass

        export_from_zarr_chunked(
            output_path=self._output_path,
            original_segy_path=self._original_segy_path,
            processed_zarr_path=self._traces_path,
            chunk_size=self._chunk_size,
            progress_callback=on_progress,
            headers_df=self._headers_df,
        )

    def cancel(self):
        """Request cancellation of the export."""
        self._cancelled = True
        if self._job:
            self._job_manager.cancel_job(self._job.id)


@dataclass
class ParallelExportResult:
    """Result from parallel SEGY export operation."""
    success: bool
    output_path: str
    n_traces: int
    elapsed_time: float
    error: Optional[str] = None


class ParallelExportWorker(QThread):
    """
    Worker thread for running parallel multi-process SEGY export in background.

    Uses ParallelExportCoordinator for high-performance multi-stage export
    with centralized job dashboard progress tracking (no built-in QProgressDialogs).

    Usage:
        from utils.parallel_export import ExportConfig

        config = ExportConfig(...)
        worker = ParallelExportWorker(config)
        worker.finished_with_result.connect(on_complete)
        worker.error_occurred.connect(on_error)
        worker.start()

        # Progress shown in JobDashboardWidget automatically
    """

    # Signals for external listeners
    progress_updated = pyqtSignal(object)  # ExportProgress
    finished_with_result = pyqtSignal(object)  # ParallelExportResult
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        export_config: Any,
        n_traces: int = 0,
        estimated_size_mb: int = 100,
        job_name: Optional[str] = None,
        parent=None
    ):
        """
        Initialize parallel export worker.

        Parameters
        ----------
        export_config : ExportConfig
            Parallel export configuration from utils.parallel_export
        n_traces : int
            Number of traces (for progress tracking)
        estimated_size_mb : int
            Estimated output file size in MB
        job_name : str, optional
            Human-readable job name
        parent : QObject, optional
            Parent object
        """
        super().__init__(parent)
        self._export_config = export_config
        self._n_traces = n_traces
        self._estimated_size_mb = estimated_size_mb
        self._job_name = job_name or Path(export_config.output_path).name
        self._coordinator = None
        self._job: Optional[Job] = None
        self._job_manager = get_job_manager()
        self._qt_bridge = get_job_bridge()
        self._start_time: Optional[float] = None
        self._cancelled = False

    @property
    def job(self) -> Optional[Job]:
        """Get the associated job."""
        return self._job

    @property
    def job_id(self):
        """Get job ID."""
        return self._job.id if self._job else None

    def run(self):
        """Execute parallel SEGY export in thread."""
        from utils.parallel_export import ParallelExportCoordinator

        self._start_time = time.time()

        try:
            # Create coordinator
            self._coordinator = ParallelExportCoordinator(self._export_config)

            # Submit job to JobManager
            self._job = self._job_manager.submit_job(
                name=f"Export: {self._job_name}",
                job_type=JobType.SEGY_EXPORT,
                config=JobConfig.for_segy_import(file_size_mb=self._estimated_size_mb),
                custom_config={
                    'output_path': self._export_config.output_path,
                    'n_traces': self._n_traces,
                    'export_type': getattr(self._export_config, 'export_type', 'processed'),
                },
            )

            # Emit queued signal (submit_job doesn't trigger callbacks)
            if self._qt_bridge:
                try:
                    self._qt_bridge.signals.emit_job_queued(self._job)
                except Exception:
                    pass

            # Start job - triggers on_job_started callback via JobManager
            self._job_manager.start_job(self._job.id)

            # Run all export stages
            self._run_export_stages()

            # Success - complete_job triggers on_job_completed callback
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._job_manager.complete_job(
                self._job.id,
                result={
                    'output_path': self._export_config.output_path,
                    'n_traces': self._n_traces,
                    'elapsed_time': elapsed,
                },
            )

            export_result = ParallelExportResult(
                success=True,
                output_path=self._export_config.output_path,
                n_traces=self._n_traces,
                elapsed_time=elapsed,
            )
            self.finished_with_result.emit(export_result)

        except InterruptedError:
            # Cancelled - cancel/finalize triggers callbacks
            if self._job:
                self._job_manager.cancel_job(self._job.id)
                self._job_manager.finalize_cancellation(self._job.id)
            self.error_occurred.emit("Export cancelled by user")

        except Exception as e:
            logger.exception("Parallel export worker error")
            error_msg = str(e)

            if self._job:
                if self._cancelled:
                    self._job_manager.cancel_job(self._job.id)
                    self._job_manager.finalize_cancellation(self._job.id)
                else:
                    # fail_job triggers on_job_failed callback
                    self._job_manager.fail_job(self._job.id, error=error_msg)

            self.error_occurred.emit(error_msg)

    def _run_export_stages(self):
        """Run all parallel export stages with progress reporting."""
        # Throttle state for Stage 1
        last_emit_time = [0.0]
        last_phase = [None]
        MIN_EMIT_INTERVAL = 0.5  # 2 Hz max

        # Stage 1: Parallel Export
        def on_export_progress(prog):
            # Check cancellation
            if self._cancelled:
                self._coordinator.cancel()
                raise InterruptedError("Export cancelled by user")

            # Throttle: emit on phase change or time passed
            time_passed = (prog.elapsed_time - last_emit_time[0]) >= MIN_EMIT_INTERVAL
            phase_changed = last_phase[0] != prog.phase

            if not (phase_changed or time_passed):
                return

            last_emit_time[0] = prog.elapsed_time
            last_phase[0] = prog.phase

            # Emit to local listeners
            self.progress_updated.emit(prog)

            # Emit to dashboard
            if self._qt_bridge and self._job:
                try:
                    rate = prog.current_traces / prog.elapsed_time if prog.elapsed_time > 0 else 0
                    self._qt_bridge.signals.job_progress.emit(
                        self._job.id,
                        {
                            'percent': (prog.current_traces / prog.total_traces * 100)
                                       if prog.total_traces > 0 else 0,
                            'message': f'Phase: {prog.phase}',
                            'phase': prog.phase,
                            'current_traces': prog.current_traces,
                            'total_traces': prog.total_traces,
                            'active_workers': prog.active_workers,
                            'traces_per_sec': rate,
                            'eta_seconds': prog.eta_seconds,
                        }
                    )
                except Exception:
                    pass

        stage_result = self._coordinator.run_parallel_export(progress_callback=on_export_progress)

        if self._cancelled or self._coordinator.was_cancelled:
            raise InterruptedError("Export cancelled by user")

        if not stage_result.success:
            raise RuntimeError(stage_result.error or "Export stage failed")

        # Throttle state for Stage 2
        last_merge_emit = [0.0]

        # Stage 2: Merge segments
        def on_merge_progress(bytes_done, total_bytes):
            if self._cancelled:
                raise InterruptedError("Export cancelled by user")

            # Throttle merge progress - emit if enough time passed
            current_time = time.time()
            time_passed = (current_time - last_merge_emit[0]) >= MIN_EMIT_INTERVAL

            if not time_passed:
                return

            last_merge_emit[0] = current_time

            if self._qt_bridge and self._job:
                try:
                    percent = (bytes_done / total_bytes * 100) if total_bytes > 0 else 0
                    self._qt_bridge.signals.job_progress.emit(
                        self._job.id,
                        {
                            'percent': percent,
                            'message': 'Merging segments',
                            'phase': 'merging',
                            'current_traces': 0,
                            'total_traces': self._n_traces,
                            'active_workers': 1,
                            'traces_per_sec': 0,
                            'eta_seconds': 0,
                        }
                    )
                except Exception:
                    pass

        # run_merge returns int (total traces), raises on error
        self._coordinator.run_merge(
            stage_result,
            progress_callback=on_merge_progress
        )

        # Stage 3: Cleanup (typically fast, no throttling needed)
        def on_cleanup_progress(files_done, total_files):
            if self._qt_bridge and self._job:
                try:
                    percent = (files_done / total_files * 100) if total_files > 0 else 0
                    self._qt_bridge.signals.job_progress.emit(
                        self._job.id,
                        {
                            'percent': 90 + percent * 0.1,  # 90-100%
                            'message': 'Cleaning up',
                            'phase': 'cleanup',
                            'current_traces': self._n_traces,
                            'total_traces': self._n_traces,
                            'active_workers': 1,
                            'traces_per_sec': 0,
                            'eta_seconds': 0,
                        }
                    )
                except Exception:
                    pass

        self._coordinator.run_cleanup(stage_result, progress_callback=on_cleanup_progress)

    def cancel(self):
        """Request cancellation of the export."""
        self._cancelled = True
        if self._coordinator:
            self._coordinator.cancel()
        if self._job:
            self._job_manager.cancel_job(self._job.id)
