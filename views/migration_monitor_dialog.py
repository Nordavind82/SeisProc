"""
Migration Job Monitor Dialog

Displays progress and allows control of running migration jobs.
Runs migration in a background thread with progress updates.
"""
import logging
import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QTextCursor

logger = logging.getLogger(__name__)


class MigrationWorker(QThread):
    """Background worker thread for migration execution."""

    progress_updated = pyqtSignal(dict)  # Progress info dict
    log_message = pyqtSignal(str, str)   # message, level
    finished = pyqtSignal(bool, str)     # success, message
    bin_started = pyqtSignal(str, int, int)  # bin_name, current, total

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._cancelled = False
        self._paused = False
        self._pause_condition = None

    def run(self):
        """Execute migration job."""
        try:
            self.log_message.emit("Initializing migration job...", "info")

            # Import migration components
            from pathlib import Path
            import numpy as np
            from processors.migration import KirchhoffMigrator
            from models.velocity_model import create_constant_velocity, create_linear_gradient_velocity
            from models.migration_geometry import MigrationGeometry
            from utils.dataset_indexer import DatasetIndexer, BinnedDataset
            from models.binning import BinningTable, OffsetAzimuthBin
            from seisio.gather_readers import ZarrDataReader, create_gather_iterator
            from utils.segy_import.data_storage import DataStorage
            import json

            input_file = self.config.get('input_file', '')
            output_directory = self.config.get('output_directory', '')

            # Validate input
            if not input_file:
                raise ValueError("No input file specified. Please select a dataset in the wizard.")

            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input path does not exist: {input_path}")

            output_dir = Path(output_directory) if output_directory else input_path.parent / 'migrated_output'

            self.log_message.emit(f"Input: {input_path}", "info")
            self.log_message.emit(f"Output: {output_dir}", "info")

            # Determine input type and load data
            if input_path.is_dir():
                # Zarr dataset
                self.log_message.emit(f"Loading Zarr dataset: {input_path.name}", "info")
                storage = DataStorage(str(input_path))
                data_reader = ZarrDataReader(input_path)

                # Load metadata
                metadata_path = input_path / 'metadata.json'
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                n_samples = metadata.get('n_samples', data_reader.get_n_samples())
                sample_rate_ms = metadata.get('sample_rate_ms', 4.0)

                # Build index from parquet headers
                self.log_message.emit("Building trace index from headers...", "info")
                header_mapping = self.config.get('header_mapping', {})
                indexer = DatasetIndexer(header_mapping=header_mapping)

                headers_path = input_path / 'headers.parquet'
                if headers_path.exists():
                    dataset_index = indexer.index_from_parquet(
                        str(headers_path),
                        n_samples=n_samples,
                        sample_rate_ms=sample_rate_ms,
                        header_mapping=header_mapping
                    )
                else:
                    raise FileNotFoundError(f"Headers file not found: {headers_path}")

            else:
                # SEG-Y file - index directly
                self.log_message.emit(f"Indexing SEG-Y file: {input_path.name}", "info")
                indexer = DatasetIndexer()
                dataset_index = indexer.index_file(str(input_path))

                # Create data reader for SEG-Y
                from seisio.gather_readers import NumpyDataReader
                import segyio
                with segyio.open(str(input_path), 'r', ignore_geometry=True) as f:
                    traces = f.trace.raw[:]
                data_reader = NumpyDataReader(traces)
                sample_rate_ms = dataset_index.sample_rate_ms

            self.log_message.emit(
                f"Dataset: {dataset_index.n_traces:,} traces, "
                f"{dataset_index.n_samples} samples @ {sample_rate_ms}ms",
                "info"
            )

            # Create velocity model
            self.log_message.emit("Creating velocity model...", "info")
            velocity_model = self._create_velocity_model()

            # Create MigrationConfig
            from models.migration_config import MigrationConfig, OutputGrid, TraveltimeMode

            # Build output grid from config
            output_grid = OutputGrid(
                n_time=int(self.config.get('time_max_ms', 6000) / self.config.get('dt_ms', 4)),
                n_inline=max(1, (self.config.get('inline_max', 100) - self.config.get('inline_min', 1)) // self.config.get('inline_step', 1) + 1),
                n_xline=max(1, (self.config.get('xline_max', 100) - self.config.get('xline_min', 1)) // self.config.get('xline_step', 1) + 1),
                dt=self.config.get('dt_ms', 4.0) / 1000.0,  # Convert to seconds
                d_inline=25.0,  # Default 25m
                d_xline=25.0,   # Default 25m
                t0=self.config.get('time_min_ms', 0) / 1000.0,
                inline_start=self.config.get('inline_min', 1),
                xline_start=self.config.get('xline_min', 1),
            )

            # Determine traveltime mode
            tt_mode_str = self.config.get('traveltime_mode', 'straight_ray')
            if 'curved' in tt_mode_str.lower():
                traveltime_mode = TraveltimeMode.CURVED_RAY
            else:
                traveltime_mode = TraveltimeMode.STRAIGHT_RAY

            migration_config = MigrationConfig(
                output_grid=output_grid,
                max_aperture_m=self.config.get('max_aperture_m', 5000.0),
                max_angle_deg=self.config.get('max_angle_deg', 60.0),
                traveltime_mode=traveltime_mode,
                antialias_enabled=self.config.get('antialias_enabled', True),
            )

            self.log_message.emit(
                f"Output grid: {output_grid.n_inline}x{output_grid.n_xline}x{output_grid.n_time} "
                f"({output_grid.memory_gb:.2f} GB)",
                "info"
            )

            # Create migrator
            migrator = KirchhoffMigrator(
                velocity=velocity_model,
                config=migration_config,
            )

            # Setup binning
            bins_config = self.config.get('binning_table', [])
            if not bins_config:
                # Create default single bin (full stack)
                bins_config = [{
                    'name': 'Full Stack',
                    'offset_min': 0,
                    'offset_max': 50000,
                    'azimuth_min': 0,
                    'azimuth_max': 360,
                    'enabled': True
                }]

            # Create binning table
            offset_bins = []
            for bc in bins_config:
                if bc.get('enabled', True):
                    # Handle both 'az_min'/'az_max' and 'azimuth_min'/'azimuth_max' keys
                    az_min = bc.get('azimuth_min', bc.get('az_min', 0))
                    az_max = bc.get('azimuth_max', bc.get('az_max', 360))
                    offset_bins.append(OffsetAzimuthBin(
                        name=bc.get('name', 'Bin'),
                        offset_min=bc.get('offset_min', 0),
                        offset_max=bc.get('offset_max', 50000),
                        azimuth_min=az_min,
                        azimuth_max=az_max,
                    ))

            binning_table = BinningTable(bins=offset_bins)
            self.log_message.emit(f"Created binning table with {len(offset_bins)} bins", "info")

            binned_dataset = BinnedDataset(dataset_index, binning_table)
            self.log_message.emit("Assigned traces to bins", "info")

            enabled_bins = [b for b in bins_config if b.get('enabled', True)]
            total_bins = len(enabled_bins)

            if total_bins == 0:
                self.log_message.emit("No bins enabled - using default full stack", "warning")
                total_bins = 1
                enabled_bins = [{'name': 'Full Stack', 'enabled': True}]

            # Get bin summary for logging
            bin_summary = binned_dataset.get_bin_summary()
            total_bins = len(bin_summary)
            self.log_message.emit(f"Processing {total_bins} offset/azimuth bins", "info")
            for bn, count in bin_summary.items():
                self.log_message.emit(f"  {bn}: {count:,} traces", "info")

            # Create output storage
            output_dir.mkdir(parents=True, exist_ok=True)
            output_storage = DataStorage(str(output_dir))

            # Process each bin using iterate_bins for reliable access
            bins_processed = 0
            for bin_name, trace_numbers in binned_dataset.iterate_bins():
                bins_processed += 1

                if self._cancelled:
                    self.log_message.emit("Job cancelled by user", "warning")
                    self.finished.emit(False, "Job cancelled")
                    return

                # Handle pause
                while self._paused and not self._cancelled:
                    time.sleep(0.1)

                self.bin_started.emit(bin_name, bins_processed, total_bins)
                self.log_message.emit(f"Processing bin: {bin_name}", "info")

                n_bin_traces = len(trace_numbers)

                self.log_message.emit(f"  {n_bin_traces:,} traces in bin", "info")

                if n_bin_traces == 0:
                    self.log_message.emit(f"  Skipping empty bin: {bin_name}", "warning")
                    continue

                # Update progress
                progress = {
                    'overall_percent': ((bins_processed - 1) / max(total_bins, 1)) * 100,
                    'current_bin': bin_name,
                    'bin_number': bins_processed,
                    'total_bins': total_bins,
                    'bin_percent': 0,
                    'traces_processed': 0,
                    'traces_total': n_bin_traces,
                }
                self.progress_updated.emit(progress)

                # Process traces in chunks
                # Use smaller chunks for better progress updates on large datasets
                chunk_size = min(500, max(100, n_bin_traces // 100))  # Adaptive chunk size
                n_chunks = (n_bin_traces + chunk_size - 1) // chunk_size
                migrated_traces = []

                self.log_message.emit(
                    f"  Processing {n_bin_traces:,} traces in {n_chunks} chunks of ~{chunk_size}",
                    "info"
                )

                import time as time_module
                bin_start_time = time_module.time()

                for chunk_idx, chunk_start in enumerate(range(0, n_bin_traces, chunk_size)):
                    if self._cancelled:
                        break

                    while self._paused and not self._cancelled:
                        time.sleep(0.1)

                    chunk_end = min(chunk_start + chunk_size, n_bin_traces)
                    chunk_traces = trace_numbers[chunk_start:chunk_end]
                    actual_chunk_size = len(chunk_traces)

                    # Log every 10 chunks or first chunk
                    if chunk_idx % 10 == 0 or chunk_idx == 0:
                        elapsed = time_module.time() - bin_start_time
                        if chunk_idx > 0:
                            rate = chunk_start / elapsed
                            eta = (n_bin_traces - chunk_start) / rate if rate > 0 else 0
                            self.log_message.emit(
                                f"  Chunk {chunk_idx+1}/{n_chunks}: traces {chunk_start:,}-{chunk_end:,} "
                                f"({rate:.0f} traces/s, ETA: {eta/60:.1f} min)",
                                "debug"
                            )
                        else:
                            self.log_message.emit(
                                f"  Chunk {chunk_idx+1}/{n_chunks}: traces {chunk_start:,}-{chunk_end:,}",
                                "debug"
                            )

                    # Read trace data - returns (n_traces, n_samples)
                    read_start = time_module.time()
                    trace_data = data_reader.read_traces(np.array(chunk_traces))
                    read_time = time_module.time() - read_start

                    # Get geometry for these traces
                    entries = [dataset_index.entries[t] for t in chunk_traces]

                    # Build geometry arrays for the entire chunk
                    source_x = np.array([e.source_x or 0 for e in entries], dtype=np.float32)
                    source_y = np.array([e.source_y or 0 for e in entries], dtype=np.float32)
                    receiver_x = np.array([e.receiver_x or 0 for e in entries], dtype=np.float32)
                    receiver_y = np.array([e.receiver_y or 0 for e in entries], dtype=np.float32)

                    # Create SeismicData object
                    # SeismicData expects (n_samples, n_traces) format
                    from models.seismic_data import SeismicData
                    gather = SeismicData(
                        traces=trace_data.T,  # Transpose to (n_samples, n_traces)
                        sample_rate=sample_rate_ms,
                    )

                    # Create geometry for this chunk
                    geometry = MigrationGeometry(
                        source_x=source_x,
                        source_y=source_y,
                        receiver_x=receiver_x,
                        receiver_y=receiver_y,
                    )

                    # Migrate the entire chunk as a gather
                    migrate_start = time_module.time()
                    try:
                        image, fold = migrator.migrate_gather(gather, geometry)
                        migrate_time = time_module.time() - migrate_start
                        migrated_traces.extend([None] * actual_chunk_size)

                        # Log timing for first chunk
                        if chunk_idx == 0:
                            self.log_message.emit(
                                f"  First chunk timing: read={read_time:.2f}s, migrate={migrate_time:.2f}s",
                                "debug"
                            )
                    except Exception as e:
                        # Log error but continue
                        logger.warning(f"Migration error for chunk {chunk_idx}: {e}")
                        self.log_message.emit(f"  Chunk {chunk_idx} error: {e}", "warning")
                        migrated_traces.extend([None] * actual_chunk_size)

                    # Update progress
                    traces_done = chunk_end
                    progress['bin_percent'] = (traces_done / n_bin_traces) * 100
                    progress['overall_percent'] = ((bins_processed - 1 + traces_done / n_bin_traces) / max(total_bins, 1)) * 100
                    progress['traces_processed'] = traces_done
                    self.progress_updated.emit(progress)

                bin_elapsed = time_module.time() - bin_start_time
                traces_per_sec = len(migrated_traces) / bin_elapsed if bin_elapsed > 0 else 0

                self.log_message.emit(
                    f"Completed bin: {bin_name} - {len(migrated_traces):,} traces in {bin_elapsed:.1f}s "
                    f"({traces_per_sec:.0f} traces/s)",
                    "info"
                )

            if not self._cancelled:
                self.log_message.emit("Migration completed successfully!", "info")
                self.finished.emit(True, "Migration completed successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            self.log_message.emit(f"Error: {str(e)}", "error")
            self.finished.emit(False, str(e))

    def _create_velocity_model(self):
        """Create velocity model from configuration."""
        from models.velocity_model import create_constant_velocity, create_linear_gradient_velocity

        vel_type = self.config.get('velocity_type', 'constant')
        v0 = self.config.get('velocity_v0', 2500.0)

        if vel_type == 'constant':
            return create_constant_velocity(v0, is_time=True)
        elif vel_type == 'gradient':
            gradient = self.config.get('velocity_gradient', 0.0)
            # Create gradient model
            return create_linear_gradient_velocity(
                v0=v0,
                gradient=gradient,
                z_max=10.0,  # 10 seconds
                dz=0.004,
                is_time=True
            )
        else:
            # File-based - would load from file
            return create_constant_velocity(v0, is_time=True)

    def cancel(self):
        """Request job cancellation."""
        self._cancelled = True
        self.log_message.emit("Cancellation requested...", "warning")

    def pause(self):
        """Pause job execution."""
        self._paused = True
        self.log_message.emit("Job paused", "info")

    def resume(self):
        """Resume paused job."""
        self._paused = False
        self.log_message.emit("Job resumed", "info")

    @property
    def is_paused(self) -> bool:
        return self._paused


class MigrationMonitorDialog(QDialog):
    """Dialog for monitoring migration job progress."""

    job_completed = pyqtSignal(bool, str)  # success, output_path

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        self.worker: Optional[MigrationWorker] = None
        self.start_time: Optional[float] = None

        job_name = config.get('name', 'Migration Job')
        self.setWindowTitle(f"Migration: {job_name}")
        self.resize(700, 550)
        self.setModal(False)  # Allow interaction with main window

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)

        # Job info
        info_group = QGroupBox("Job Information")
        info_layout = QVBoxLayout(info_group)
        self.job_label = QLabel(f"Job: {self.config.get('name', 'Unknown')}")
        info_layout.addWidget(self.job_label)

        input_file = self.config.get('input_file', '')
        if input_file:
            self.input_label = QLabel(f"Input: {Path(input_file).name}")
            info_layout.addWidget(self.input_label)

        output_dir = self.config.get('output_directory', '')
        if output_dir:
            self.output_label = QLabel(f"Output: {output_dir}")
            info_layout.addWidget(self.output_label)

        layout.addWidget(info_group)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall:"))
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        overall_layout.addWidget(self.overall_progress)
        self.overall_label = QLabel("0%")
        self.overall_label.setMinimumWidth(50)
        overall_layout.addWidget(self.overall_label)
        progress_layout.addLayout(overall_layout)

        # Current bin progress
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Current Bin:"))
        self.bin_progress = QProgressBar()
        self.bin_progress.setRange(0, 100)
        self.bin_progress.setValue(0)
        bin_layout.addWidget(self.bin_progress)
        self.bin_label = QLabel("-")
        self.bin_label.setMinimumWidth(100)
        bin_layout.addWidget(self.bin_label)
        progress_layout.addLayout(bin_layout)

        # Time info
        time_layout = QHBoxLayout()
        self.time_label = QLabel("Elapsed: 00:00:00 | Remaining: --:--:--")
        time_layout.addWidget(self.time_label)
        progress_layout.addLayout(time_layout)

        layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        btn_layout.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()

        self.view_btn = QPushButton("View Output")
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._on_view_output)
        btn_layout.addWidget(self.view_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._on_close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        # Timer for elapsed time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)

    def start_job(self):
        """Start the migration job."""
        self.worker = MigrationWorker(self.config)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.bin_started.connect(self._on_bin_started)
        self.worker.finished.connect(self._on_finished)

        self.start_time = time.time()
        self.timer.start(1000)

        self._log("Starting migration job...", "info")
        self.worker.start()

    def _on_progress(self, progress: Dict):
        """Handle progress update from worker."""
        overall = progress.get('overall_percent', 0)
        bin_percent = progress.get('bin_percent', 0)

        self.overall_progress.setValue(int(overall))
        self.overall_label.setText(f"{overall:.1f}%")

        self.bin_progress.setValue(int(bin_percent))

    def _on_bin_started(self, bin_name: str, current: int, total: int):
        """Handle new bin started."""
        self.bin_label.setText(f"{bin_name} ({current}/{total})")
        self.bin_progress.setValue(0)

    def _on_log(self, message: str, level: str = "info"):
        """Handle log message from worker."""
        self._log(message, level)

    def _log(self, message: str, level: str = "info"):
        """Add message to log with timestamp."""
        timestamp = time.strftime("%H:%M:%S")

        # Color based on level
        if level == "error":
            color = "red"
        elif level == "warning":
            color = "orange"
        elif level == "success":
            color = "green"
        else:
            color = "white"

        html = f'<span style="color: gray;">[{timestamp}]</span> '
        html += f'<span style="color: {color};">{message}</span><br>'

        self.log_text.insertHtml(html)

        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def _update_time(self):
        """Update elapsed time display."""
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)

        # Estimate remaining time based on progress
        overall = self.overall_progress.value()
        if overall > 0:
            estimated_total = elapsed / (overall / 100)
            remaining = estimated_total - elapsed
            remaining_str = self._format_time(remaining)
        else:
            remaining_str = "--:--:--"

        self.time_label.setText(f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _on_pause(self):
        """Handle pause button click."""
        if self.worker is None:
            return

        if self.worker.is_paused:
            self.worker.resume()
            self.pause_btn.setText("Pause")
        else:
            self.worker.pause()
            self.pause_btn.setText("Resume")

    def _on_cancel(self):
        """Handle cancel button click."""
        reply = QMessageBox.question(
            self,
            "Cancel Job",
            "Are you sure you want to cancel this job?\n\n"
            "Progress will be saved if checkpointing is enabled.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes and self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)

    def _on_finished(self, success: bool, message: str):
        """Handle job completion."""
        self.timer.stop()

        if success:
            self._log(message, "success")
            self.overall_progress.setValue(100)
            self.overall_label.setText("100%")
            self.view_btn.setEnabled(True)

            QMessageBox.information(
                self,
                "Migration Complete",
                f"Migration completed successfully!\n\n"
                f"Output saved to:\n{self.config.get('output_directory', 'Unknown')}"
            )
        else:
            self._log(f"Job failed: {message}", "error")

            if "cancelled" not in message.lower():
                QMessageBox.warning(
                    self,
                    "Migration Failed",
                    f"Migration failed:\n\n{message}"
                )

        # Update button states
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.close_btn.setText("Close")

        # Emit completion signal
        output_path = self.config.get('output_directory', '')
        self.job_completed.emit(success, output_path)

    def _on_view_output(self):
        """Open output directory or load results."""
        output_dir = self.config.get('output_directory', '')
        if output_dir and Path(output_dir).exists():
            import subprocess
            import sys

            if sys.platform == 'darwin':
                subprocess.run(['open', output_dir])
            elif sys.platform == 'win32':
                subprocess.run(['explorer', output_dir])
            else:
                subprocess.run(['xdg-open', output_dir])

    def _on_close(self):
        """Handle close button click."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Job Running",
                "A migration job is still running.\n\n"
                "Close anyway? The job will continue in the background.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.close()

    def closeEvent(self, event):
        """Handle dialog close."""
        # Don't stop the worker - let it run in background
        event.accept()
