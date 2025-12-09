"""
Migration Orchestrator

Manages bin-by-bin execution of migration jobs:
- Sequential bin processing (lower memory)
- Parallel bin processing (faster, higher memory)
- Progress reporting per bin and overall
- Checkpoint/resume capability
- Resource estimation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
import json
import time
import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from models.migration_job import MigrationJobConfig
from models.binning import BinningTable
from seisio.migration_output import MigrationOutputManager, create_output_manager
from seisio.gather_readers import (
    Gather,
    GatherIterator,
    CommonOffsetGatherIterator,
    DataReader,
    NumpyDataReader,
)
from utils.dataset_indexer import DatasetIndex, BinnedDataset

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing mode for migration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class BinProgress:
    """Progress information for a single bin."""
    bin_name: str
    status: str = "pending"  # pending, in_progress, completed, failed
    n_gathers_total: int = 0
    n_gathers_processed: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def progress_percent(self) -> float:
        """Progress percentage."""
        if self.n_gathers_total == 0:
            return 0.0
        return 100.0 * self.n_gathers_processed / self.n_gathers_total


@dataclass
class MigrationProgress:
    """Overall migration progress."""
    job_name: str
    n_bins_total: int = 0
    n_bins_completed: int = 0
    n_bins_failed: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    bin_progress: Dict[str, BinProgress] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def overall_percent(self) -> float:
        """Overall completion percentage."""
        if self.n_bins_total == 0:
            return 0.0
        return 100.0 * self.n_bins_completed / self.n_bins_total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_name': self.job_name,
            'n_bins_total': self.n_bins_total,
            'n_bins_completed': self.n_bins_completed,
            'n_bins_failed': self.n_bins_failed,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_seconds': self.elapsed_seconds,
            'overall_percent': self.overall_percent,
            'bin_progress': {
                name: {
                    'status': bp.status,
                    'n_gathers_processed': bp.n_gathers_processed,
                    'n_gathers_total': bp.n_gathers_total,
                    'progress_percent': bp.progress_percent,
                }
                for name, bp in self.bin_progress.items()
            }
        }


# Progress callback type
ProgressCallback = Callable[[MigrationProgress], None]


@dataclass
class MigrationOrchestrator:
    """
    Orchestrates bin-by-bin migration execution.

    Manages:
    - Bin processing order
    - Sequential or parallel execution
    - Progress tracking
    - Checkpointing
    - Resource management
    """
    job_config: MigrationJobConfig
    index: DatasetIndex
    data_reader: DataReader
    processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    max_workers: int = 1
    enable_checkpointing: bool = True
    checkpoint_dir: Optional[str] = None
    progress_callback: Optional[ProgressCallback] = None

    # Internal state
    _output_manager: Optional[MigrationOutputManager] = field(default=None, init=False)
    _binned_dataset: Optional[BinnedDataset] = field(default=None, init=False)
    _progress: Optional[MigrationProgress] = field(default=None, init=False)
    _migrator: Optional[Any] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize orchestrator."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(
                Path(self.job_config.output_directory) / ".checkpoints"
            )

    def setup(self):
        """
        Set up migration components.

        Call this before run() to prepare all components.
        """
        logger.info(f"Setting up migration job: {self.job_config.name}")

        # Get binning table
        binning_table = self.job_config.get_binning_table()

        # Create binned dataset
        self._binned_dataset = BinnedDataset(self.index, binning_table)

        # Create output manager
        self._output_manager = create_output_manager(self.job_config)
        self._output_manager.initialize()

        # Create migrator
        from processors.migration.kirchhoff_migrator import KirchhoffMigrator

        velocity = self.job_config.get_velocity_model()
        config = self.job_config.get_migration_config()

        self._migrator = KirchhoffMigrator(velocity, config)

        # Initialize progress tracking
        enabled_bins = [b.name for b in binning_table.bins if b.enabled]
        self._progress = MigrationProgress(
            job_name=self.job_config.name,
            n_bins_total=len(enabled_bins),
        )

        for bin_name in enabled_bins:
            traces = self._binned_dataset.get_bin_trace_numbers(bin_name)
            self._progress.bin_progress[bin_name] = BinProgress(
                bin_name=bin_name,
                n_gathers_total=len(traces),
            )

        # Create checkpoint directory
        if self.enable_checkpointing:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Setup complete: {len(enabled_bins)} bins to process")

    def run(self) -> MigrationProgress:
        """
        Run the migration job.

        Returns:
            MigrationProgress with final status
        """
        if self._progress is None:
            self.setup()

        logger.info(f"Starting migration: {self.job_config.name}")
        self._progress.start_time = time.time()

        try:
            # Load checkpoint if available
            if self.enable_checkpointing:
                self._load_checkpoint()

            # Get bins to process
            bins_to_process = [
                name for name, bp in self._progress.bin_progress.items()
                if bp.status != "completed"
            ]

            if self.processing_mode == ProcessingMode.SEQUENTIAL:
                self._run_sequential(bins_to_process)
            else:
                self._run_parallel(bins_to_process)

            # Finalize
            self._finalize()

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

        finally:
            self._progress.end_time = time.time()
            self._report_progress()

        return self._progress

    def _run_sequential(self, bin_names: List[str]):
        """Run bins sequentially."""
        for bin_name in bin_names:
            try:
                self._process_bin(bin_name)
            except Exception as e:
                self._handle_bin_error(bin_name, e)

            # Save checkpoint after each bin
            if self.enable_checkpointing:
                self._save_checkpoint()

    def _run_parallel(self, bin_names: List[str]):
        """Run bins in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_bin, name): name
                for name in bin_names
            }

            for future in as_completed(futures):
                bin_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._handle_bin_error(bin_name, e)

                # Save checkpoint
                if self.enable_checkpointing:
                    self._save_checkpoint()

    def _process_bin(self, bin_name: str):
        """
        Process a single bin.

        Args:
            bin_name: Name of bin to process
        """
        bp = self._progress.bin_progress[bin_name]
        bp.status = "in_progress"
        bp.start_time = time.time()

        logger.info(f"Processing bin: {bin_name}")
        self._report_progress()

        try:
            # Get traces for this bin
            trace_numbers = self._binned_dataset.get_bin_trace_numbers(bin_name)

            if len(trace_numbers) == 0:
                logger.warning(f"No traces in bin {bin_name}, skipping")
                bp.status = "completed"
                bp.end_time = time.time()
                self._progress.n_bins_completed += 1
                return

            # Create gather for this bin
            gather = self._create_gather(bin_name, trace_numbers)

            # Migrate the gather
            result = self._migrator.migrate_gather(gather)

            # Add to output
            self._output_manager.add_migrated_data(
                bin_name,
                result.migrated_volume,
                fold=result.fold_volume if hasattr(result, 'fold_volume') else None,
            )

            # Mark complete
            bp.status = "completed"
            bp.end_time = time.time()
            bp.n_gathers_processed = len(trace_numbers)
            self._progress.n_bins_completed += 1

            self._output_manager.finalize_bin(bin_name)

            logger.info(
                f"Completed bin {bin_name}: {len(trace_numbers)} traces "
                f"in {bp.elapsed_seconds:.1f}s"
            )

        except Exception as e:
            bp.status = "failed"
            bp.error_message = str(e)
            bp.end_time = time.time()
            self._progress.n_bins_failed += 1
            raise

        finally:
            self._report_progress()

    def _create_gather(self, bin_name: str, trace_numbers: List[int]) -> Gather:
        """Create gather from trace numbers."""
        trace_arr = np.array(trace_numbers, dtype=np.int32)

        # Read trace data
        data = self.data_reader.read_traces(trace_arr)

        # Extract geometry from index
        entries = [self.index.entries[t] for t in trace_numbers]

        offsets = np.array([e.offset for e in entries], dtype=np.float32)
        azimuths = np.array([e.azimuth for e in entries], dtype=np.float32)

        # Optional geometry
        source_x = None
        source_y = None
        receiver_x = None
        receiver_y = None

        if entries[0].source_x is not None:
            source_x = np.array([e.source_x for e in entries], dtype=np.float32)
            source_y = np.array([e.source_y for e in entries], dtype=np.float32)
            receiver_x = np.array([e.receiver_x for e in entries], dtype=np.float32)
            receiver_y = np.array([e.receiver_y for e in entries], dtype=np.float32)

        # Get bin info
        bin_obj = self._binned_dataset.binning_table.get_bin(bin_name)

        return Gather(
            gather_id=bin_name,
            trace_numbers=trace_arr,
            data=data,
            offsets=offsets,
            azimuths=azimuths,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            metadata={
                'bin_offset_center': (bin_obj.offset_min + bin_obj.offset_max) / 2,
                'bin_azimuth_center': (bin_obj.azimuth_min + bin_obj.azimuth_max) / 2,
            }
        )

    def _handle_bin_error(self, bin_name: str, error: Exception):
        """Handle error during bin processing."""
        logger.error(f"Error processing bin {bin_name}: {error}")
        bp = self._progress.bin_progress[bin_name]
        bp.status = "failed"
        bp.error_message = str(error)
        bp.end_time = time.time()
        self._progress.n_bins_failed += 1

    def _finalize(self):
        """Finalize migration after all bins processed."""
        logger.info("Finalizing migration...")

        # Normalize by fold if requested
        if self.job_config.create_fold_volume:
            self._output_manager.normalize_by_fold(min_fold=1)

        # Save outputs
        self._output_manager.save_all()

        # Clean up checkpoints
        if self.enable_checkpointing:
            self._clean_checkpoints()

        logger.info("Migration complete")

    def _report_progress(self):
        """Report progress to callback."""
        if self.progress_callback and self._progress:
            self.progress_callback(self._progress)

    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        checkpoint_file = Path(self.checkpoint_dir) / "checkpoint.json"

        checkpoint = {
            'progress': self._progress.to_dict(),
            'timestamp': time.time(),
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.debug(f"Saved checkpoint to {checkpoint_file}")

    def _load_checkpoint(self):
        """Load checkpoint from disk if available."""
        checkpoint_file = Path(self.checkpoint_dir) / "checkpoint.json"

        if not checkpoint_file.exists():
            return

        logger.info(f"Loading checkpoint from {checkpoint_file}")

        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        # Restore bin status
        saved_progress = checkpoint.get('progress', {})
        saved_bin_progress = saved_progress.get('bin_progress', {})

        for bin_name, saved_bp in saved_bin_progress.items():
            if bin_name in self._progress.bin_progress:
                bp = self._progress.bin_progress[bin_name]
                if saved_bp.get('status') == 'completed':
                    bp.status = 'completed'
                    bp.n_gathers_processed = saved_bp.get('n_gathers_processed', 0)

        # Update completed count
        self._progress.n_bins_completed = sum(
            1 for bp in self._progress.bin_progress.values()
            if bp.status == 'completed'
        )

        logger.info(
            f"Resumed from checkpoint: "
            f"{self._progress.n_bins_completed}/{self._progress.n_bins_total} bins complete"
        )

    def _clean_checkpoints(self):
        """Clean up checkpoint files after successful completion."""
        checkpoint_file = Path(self.checkpoint_dir) / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.debug("Cleaned up checkpoint file")

    def estimate_resources(self) -> Dict[str, Any]:
        """
        Estimate resource requirements for this job.

        Returns:
            Dictionary with memory, time estimates
        """
        if self._binned_dataset is None:
            self.setup()

        # Output volume memory
        grid = self.job_config.get_output_grid()
        volume_size_gb = (
            grid.n_time * grid.n_inline * grid.n_xline * 4  # float32
        ) / (1024**3)

        n_bins = len([b for b in self.job_config.get_binning_table().bins if b.enabled])
        output_memory_gb = volume_size_gb * n_bins

        if self.job_config.create_stack_volume:
            output_memory_gb += volume_size_gb

        # Trace data memory (estimate per gather)
        n_traces_per_bin = self.index.n_traces // max(1, n_bins)
        trace_memory_gb = (
            n_traces_per_bin * self.index.n_samples * 4
        ) / (1024**3)

        # Total estimate
        total_memory_gb = output_memory_gb + trace_memory_gb

        # Time estimate (very rough)
        # Assume ~1000 traces/second processing rate
        total_traces = self.index.n_traces
        estimated_time_s = total_traces / 1000

        return {
            'output_volume_gb': volume_size_gb,
            'total_output_memory_gb': output_memory_gb,
            'per_gather_memory_gb': trace_memory_gb,
            'estimated_total_memory_gb': total_memory_gb,
            'n_bins': n_bins,
            'n_traces': self.index.n_traces,
            'estimated_time_seconds': estimated_time_s,
            'estimated_time_minutes': estimated_time_s / 60,
        }

    def get_progress(self) -> Optional[MigrationProgress]:
        """Get current progress."""
        return self._progress

    def get_output_manager(self) -> Optional[MigrationOutputManager]:
        """Get output manager."""
        return self._output_manager
