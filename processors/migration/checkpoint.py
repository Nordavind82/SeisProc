"""
Enhanced Checkpoint/Resume System for Migration

Provides robust checkpointing for long-running migration jobs:
- Per-bin and per-trace progress tracking
- Intermediate volume saving
- Atomic checkpoint writes
- Job validation on resume
- Failure recovery
"""

import json
import hashlib
import shutil
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from enum import Enum
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


class CheckpointVersion:
    """Version of checkpoint format for compatibility."""
    CURRENT = "1.0"


class BinStatus(Enum):
    """Status of a bin in the migration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TraceProgress:
    """Progress within a single bin at trace level."""
    total_traces: int
    processed_traces: int = 0
    last_trace_number: Optional[int] = None

    @property
    def is_complete(self) -> bool:
        return self.processed_traces >= self.total_traces

    @property
    def progress_percent(self) -> float:
        if self.total_traces == 0:
            return 100.0
        return 100.0 * self.processed_traces / self.total_traces


@dataclass
class BinCheckpoint:
    """Checkpoint data for a single bin."""
    bin_name: str
    status: BinStatus = BinStatus.PENDING
    trace_progress: Optional[TraceProgress] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    output_file: Optional[str] = None
    volume_checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bin_name': self.bin_name,
            'status': self.status.value,
            'trace_progress': {
                'total_traces': self.trace_progress.total_traces,
                'processed_traces': self.trace_progress.processed_traces,
                'last_trace_number': self.trace_progress.last_trace_number,
            } if self.trace_progress else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'error_message': self.error_message,
            'output_file': self.output_file,
            'volume_checksum': self.volume_checksum,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BinCheckpoint':
        """Create from dictionary."""
        trace_progress = None
        if d.get('trace_progress'):
            tp = d['trace_progress']
            trace_progress = TraceProgress(
                total_traces=tp['total_traces'],
                processed_traces=tp['processed_traces'],
                last_trace_number=tp.get('last_trace_number'),
            )

        return cls(
            bin_name=d['bin_name'],
            status=BinStatus(d['status']),
            trace_progress=trace_progress,
            start_time=d.get('start_time'),
            end_time=d.get('end_time'),
            error_message=d.get('error_message'),
            output_file=d.get('output_file'),
            volume_checksum=d.get('volume_checksum'),
        )


@dataclass
class JobCheckpoint:
    """Complete checkpoint for a migration job."""
    job_id: str
    job_name: str
    version: str = CheckpointVersion.CURRENT
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Job configuration hash for validation
    config_hash: Optional[str] = None

    # Overall progress
    total_bins: int = 0
    completed_bins: int = 0
    failed_bins: int = 0

    # Per-bin checkpoints
    bins: Dict[str, BinCheckpoint] = field(default_factory=dict)

    # Job timing
    job_start_time: Optional[float] = None
    job_end_time: Optional[float] = None

    # Intermediate volume info
    intermediate_volume_path: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'job_id': self.job_id,
            'job_name': self.job_name,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'config_hash': self.config_hash,
            'total_bins': self.total_bins,
            'completed_bins': self.completed_bins,
            'failed_bins': self.failed_bins,
            'bins': {name: bc.to_dict() for name, bc in self.bins.items()},
            'job_start_time': self.job_start_time,
            'job_end_time': self.job_end_time,
            'intermediate_volume_path': self.intermediate_volume_path,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'JobCheckpoint':
        """Create from dictionary."""
        bins = {}
        for name, bc_dict in d.get('bins', {}).items():
            bins[name] = BinCheckpoint.from_dict(bc_dict)

        return cls(
            job_id=d['job_id'],
            job_name=d['job_name'],
            version=d.get('version', '1.0'),
            created_at=d.get('created_at', time.time()),
            updated_at=d.get('updated_at', time.time()),
            config_hash=d.get('config_hash'),
            total_bins=d.get('total_bins', 0),
            completed_bins=d.get('completed_bins', 0),
            failed_bins=d.get('failed_bins', 0),
            bins=bins,
            job_start_time=d.get('job_start_time'),
            job_end_time=d.get('job_end_time'),
            intermediate_volume_path=d.get('intermediate_volume_path'),
            metadata=d.get('metadata', {}),
        )

    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.completed_bins + self.failed_bins >= self.total_bins

    @property
    def progress_percent(self) -> float:
        """Overall progress percentage."""
        if self.total_bins == 0:
            return 0.0
        return 100.0 * self.completed_bins / self.total_bins

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time."""
        if self.job_start_time is None:
            return 0.0
        end = self.job_end_time or time.time()
        return end - self.job_start_time

    def get_pending_bins(self) -> List[str]:
        """Get names of bins that still need processing."""
        return [
            name for name, bc in self.bins.items()
            if bc.status in (BinStatus.PENDING, BinStatus.FAILED)
        ]

    def get_in_progress_bins(self) -> List[str]:
        """Get names of bins currently in progress."""
        return [
            name for name, bc in self.bins.items()
            if bc.status == BinStatus.IN_PROGRESS
        ]


class CheckpointManager:
    """
    Manages checkpoint creation, saving, and loading.

    Features:
    - Atomic writes (write to temp then rename)
    - Multiple checkpoint files (current + backup)
    - Validation on load
    - Automatic cleanup
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        job_id: str,
        keep_backups: int = 2,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            job_id: Unique identifier for this job
            keep_backups: Number of backup checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.job_id = job_id
        self.keep_backups = keep_backups

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self._checkpoint_file = self.checkpoint_dir / f"checkpoint_{job_id}.json"
        self._backup_pattern = self.checkpoint_dir / f"checkpoint_{job_id}.backup.{{n}}.json"
        self._lock_file = self.checkpoint_dir / f"checkpoint_{job_id}.lock"

        # Current checkpoint in memory
        self._checkpoint: Optional[JobCheckpoint] = None

    @property
    def checkpoint_file(self) -> Path:
        """Path to main checkpoint file."""
        return self._checkpoint_file

    def create_checkpoint(
        self,
        job_name: str,
        bin_names: List[str],
        bin_trace_counts: Dict[str, int],
        config_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JobCheckpoint:
        """
        Create a new checkpoint for a job.

        Args:
            job_name: Human-readable job name
            bin_names: Names of bins to process
            bin_trace_counts: Number of traces per bin
            config_hash: Hash of job configuration for validation
            metadata: Additional metadata to store

        Returns:
            New JobCheckpoint instance
        """
        self._checkpoint = JobCheckpoint(
            job_id=self.job_id,
            job_name=job_name,
            config_hash=config_hash,
            total_bins=len(bin_names),
            metadata=metadata or {},
        )

        # Initialize bin checkpoints
        for name in bin_names:
            n_traces = bin_trace_counts.get(name, 0)
            self._checkpoint.bins[name] = BinCheckpoint(
                bin_name=name,
                trace_progress=TraceProgress(total_traces=n_traces),
            )

        return self._checkpoint

    def get_checkpoint(self) -> Optional[JobCheckpoint]:
        """Get current checkpoint."""
        return self._checkpoint

    def start_job(self):
        """Mark job as started."""
        if self._checkpoint:
            self._checkpoint.job_start_time = time.time()
            self.save()

    def end_job(self):
        """Mark job as ended."""
        if self._checkpoint:
            self._checkpoint.job_end_time = time.time()
            self.save()

    def start_bin(self, bin_name: str):
        """
        Mark a bin as started.

        Args:
            bin_name: Name of bin being started
        """
        if self._checkpoint and bin_name in self._checkpoint.bins:
            bc = self._checkpoint.bins[bin_name]
            bc.status = BinStatus.IN_PROGRESS
            bc.start_time = time.time()
            bc.error_message = None
            self.save()

    def update_bin_progress(
        self,
        bin_name: str,
        processed_traces: int,
        last_trace_number: Optional[int] = None,
    ):
        """
        Update trace-level progress for a bin.

        Args:
            bin_name: Name of bin
            processed_traces: Number of traces processed so far
            last_trace_number: Last trace number processed
        """
        if self._checkpoint and bin_name in self._checkpoint.bins:
            bc = self._checkpoint.bins[bin_name]
            if bc.trace_progress:
                bc.trace_progress.processed_traces = processed_traces
                bc.trace_progress.last_trace_number = last_trace_number
            self._checkpoint.updated_at = time.time()
            # Don't save on every update - caller should batch saves

    def complete_bin(
        self,
        bin_name: str,
        output_file: Optional[str] = None,
        volume_checksum: Optional[str] = None,
    ):
        """
        Mark a bin as completed.

        Args:
            bin_name: Name of completed bin
            output_file: Path to output file
            volume_checksum: Checksum of output volume
        """
        if self._checkpoint and bin_name in self._checkpoint.bins:
            bc = self._checkpoint.bins[bin_name]
            bc.status = BinStatus.COMPLETED
            bc.end_time = time.time()
            bc.output_file = output_file
            bc.volume_checksum = volume_checksum
            if bc.trace_progress:
                bc.trace_progress.processed_traces = bc.trace_progress.total_traces

            self._checkpoint.completed_bins += 1
            self.save()

            logger.info(f"Bin {bin_name} completed and checkpointed")

    def fail_bin(self, bin_name: str, error_message: str):
        """
        Mark a bin as failed.

        Args:
            bin_name: Name of failed bin
            error_message: Error description
        """
        if self._checkpoint and bin_name in self._checkpoint.bins:
            bc = self._checkpoint.bins[bin_name]
            bc.status = BinStatus.FAILED
            bc.end_time = time.time()
            bc.error_message = error_message

            self._checkpoint.failed_bins += 1
            self.save()

            logger.warning(f"Bin {bin_name} failed: {error_message}")

    def skip_bin(self, bin_name: str, reason: str = "No traces"):
        """
        Mark a bin as skipped.

        Args:
            bin_name: Name of skipped bin
            reason: Reason for skipping
        """
        if self._checkpoint and bin_name in self._checkpoint.bins:
            bc = self._checkpoint.bins[bin_name]
            bc.status = BinStatus.SKIPPED
            bc.end_time = time.time()
            bc.error_message = reason

            self._checkpoint.completed_bins += 1
            self.save()

    def save(self):
        """
        Save checkpoint to disk atomically.

        Uses write-to-temp-then-rename pattern for atomicity.
        """
        if self._checkpoint is None:
            return

        self._checkpoint.updated_at = time.time()

        # Rotate backups
        self._rotate_backups()

        # Write to temp file
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.json',
            prefix='checkpoint_',
            dir=str(self.checkpoint_dir),
        )

        try:
            with open(temp_fd, 'w') as f:
                json.dump(self._checkpoint.to_dict(), f, indent=2)

            # Atomic rename
            shutil.move(temp_path, str(self._checkpoint_file))

            logger.debug(f"Checkpoint saved to {self._checkpoint_file}")

        except Exception as e:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def _rotate_backups(self):
        """Rotate backup checkpoint files."""
        if not self._checkpoint_file.exists():
            return

        # Shift existing backups
        for i in range(self.keep_backups - 1, 0, -1):
            src = self._get_backup_path(i - 1)
            dst = self._get_backup_path(i)
            if src.exists():
                shutil.move(str(src), str(dst))

        # Current becomes backup 0
        if self._checkpoint_file.exists():
            backup0 = self._get_backup_path(0)
            shutil.copy2(str(self._checkpoint_file), str(backup0))

    def _get_backup_path(self, n: int) -> Path:
        """Get path to backup file n."""
        return self.checkpoint_dir / f"checkpoint_{self.job_id}.backup.{n}.json"

    def load(self, validate_config_hash: Optional[str] = None) -> Optional[JobCheckpoint]:
        """
        Load checkpoint from disk.

        Args:
            validate_config_hash: If provided, validate that stored hash matches

        Returns:
            Loaded checkpoint or None if not found

        Raises:
            CheckpointValidationError: If validation fails
        """
        if not self._checkpoint_file.exists():
            logger.debug(f"No checkpoint file found at {self._checkpoint_file}")
            return None

        try:
            with open(self._checkpoint_file, 'r') as f:
                data = json.load(f)

            checkpoint = JobCheckpoint.from_dict(data)

            # Validate job ID
            if checkpoint.job_id != self.job_id:
                raise CheckpointValidationError(
                    f"Checkpoint job_id mismatch: expected {self.job_id}, "
                    f"got {checkpoint.job_id}"
                )

            # Validate config hash if provided
            if validate_config_hash and checkpoint.config_hash:
                if checkpoint.config_hash != validate_config_hash:
                    raise CheckpointValidationError(
                        "Job configuration has changed since checkpoint was created. "
                        "Either use the original configuration or delete the checkpoint."
                    )

            # Validate version compatibility
            if checkpoint.version != CheckpointVersion.CURRENT:
                logger.warning(
                    f"Checkpoint version {checkpoint.version} differs from "
                    f"current {CheckpointVersion.CURRENT}"
                )

            self._checkpoint = checkpoint

            logger.info(
                f"Loaded checkpoint: {checkpoint.completed_bins}/{checkpoint.total_bins} "
                f"bins complete"
            )

            return checkpoint

        except json.JSONDecodeError as e:
            logger.error(f"Corrupt checkpoint file: {e}")
            # Try loading from backup
            return self._load_from_backup(validate_config_hash)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _load_from_backup(
        self,
        validate_config_hash: Optional[str] = None,
    ) -> Optional[JobCheckpoint]:
        """Try to load from backup files."""
        for i in range(self.keep_backups):
            backup_path = self._get_backup_path(i)
            if backup_path.exists():
                logger.info(f"Trying backup checkpoint: {backup_path}")
                try:
                    with open(backup_path, 'r') as f:
                        data = json.load(f)
                    checkpoint = JobCheckpoint.from_dict(data)

                    if checkpoint.job_id == self.job_id:
                        self._checkpoint = checkpoint
                        # Restore as primary
                        self.save()
                        logger.info("Restored checkpoint from backup")
                        return checkpoint
                except:
                    continue

        return None

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self._checkpoint_file.exists()

    def delete(self):
        """Delete checkpoint and all backups."""
        if self._checkpoint_file.exists():
            self._checkpoint_file.unlink()

        for i in range(self.keep_backups):
            backup = self._get_backup_path(i)
            if backup.exists():
                backup.unlink()

        self._checkpoint = None
        logger.info("Checkpoint deleted")

    def get_resume_info(self) -> Dict[str, Any]:
        """
        Get information for resuming a job.

        Returns:
            Dictionary with resume information
        """
        if self._checkpoint is None:
            return {'can_resume': False, 'reason': 'No checkpoint loaded'}

        pending = self._checkpoint.get_pending_bins()
        in_progress = self._checkpoint.get_in_progress_bins()

        # In-progress bins need to restart (may have partial data)
        bins_to_process = pending + in_progress

        return {
            'can_resume': True,
            'completed_bins': self._checkpoint.completed_bins,
            'total_bins': self._checkpoint.total_bins,
            'bins_to_process': bins_to_process,
            'in_progress_bins': in_progress,
            'elapsed_seconds': self._checkpoint.elapsed_seconds,
            'last_update': self._checkpoint.updated_at,
        }


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""
    pass


class IntermediateVolumeSaver:
    """
    Saves intermediate migration volumes for long-running jobs.

    Allows recovery of partial results if job fails.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        job_id: str,
        volume_shape: tuple,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize intermediate volume saver.

        Args:
            output_dir: Directory for intermediate files
            job_id: Job identifier
            volume_shape: Shape of output volume (nz, nx, ny)
            dtype: Data type for volume
        """
        self.output_dir = Path(output_dir)
        self.job_id = job_id
        self.volume_shape = volume_shape
        self.dtype = dtype

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use memory-mapped file for large volumes
        self._mmap_path = self.output_dir / f"intermediate_{job_id}.npy"
        self._mmap: Optional[np.memmap] = None

    def initialize(self, fill_value: float = 0.0) -> np.memmap:
        """
        Initialize memory-mapped intermediate volume.

        Args:
            fill_value: Initial value to fill volume with

        Returns:
            Memory-mapped array
        """
        self._mmap = np.memmap(
            str(self._mmap_path),
            dtype=self.dtype,
            mode='w+',
            shape=self.volume_shape,
        )
        self._mmap.fill(fill_value)
        self._mmap.flush()

        logger.info(f"Initialized intermediate volume: {self._mmap_path}")
        return self._mmap

    def load_existing(self) -> Optional[np.memmap]:
        """
        Load existing intermediate volume.

        Returns:
            Memory-mapped array or None if not found
        """
        if not self._mmap_path.exists():
            return None

        self._mmap = np.memmap(
            str(self._mmap_path),
            dtype=self.dtype,
            mode='r+',
            shape=self.volume_shape,
        )

        logger.info(f"Loaded intermediate volume: {self._mmap_path}")
        return self._mmap

    def get_volume(self) -> Optional[np.memmap]:
        """Get current memory-mapped volume."""
        return self._mmap

    def update_region(
        self,
        data: np.ndarray,
        z_slice: slice,
        x_slice: slice,
        y_slice: slice,
    ):
        """
        Update a region of the intermediate volume.

        Args:
            data: Data to write
            z_slice: Z axis slice
            x_slice: X axis slice
            y_slice: Y axis slice
        """
        if self._mmap is not None:
            self._mmap[z_slice, x_slice, y_slice] = data

    def add_to_region(
        self,
        data: np.ndarray,
        z_slice: slice,
        x_slice: slice,
        y_slice: slice,
    ):
        """
        Add data to a region (for stacking).

        Args:
            data: Data to add
            z_slice: Z axis slice
            x_slice: X axis slice
            y_slice: Y axis slice
        """
        if self._mmap is not None:
            self._mmap[z_slice, x_slice, y_slice] += data

    def flush(self):
        """Flush changes to disk."""
        if self._mmap is not None:
            self._mmap.flush()

    def finalize(self, output_path: Union[str, Path]) -> str:
        """
        Finalize and move to final output location.

        Args:
            output_path: Final output path

        Returns:
            Path to final file
        """
        output_path = Path(output_path)

        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None

        # Move to final location
        shutil.move(str(self._mmap_path), str(output_path))

        logger.info(f"Finalized volume to: {output_path}")
        return str(output_path)

    def cleanup(self):
        """Remove intermediate files."""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

        if self._mmap_path.exists():
            self._mmap_path.unlink()

    def compute_checksum(self) -> str:
        """
        Compute checksum of current volume.

        Returns:
            MD5 hash of volume data
        """
        if self._mmap is None:
            return ""

        self._mmap.flush()

        # Compute checksum in chunks for large volumes
        hasher = hashlib.md5()
        chunk_size = 1024 * 1024 * 100  # 100 MB chunks

        flat = self._mmap.ravel()
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i + chunk_size]
            hasher.update(chunk.tobytes())

        return hasher.hexdigest()


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """
    Compute hash of job configuration for validation.

    Args:
        config_dict: Job configuration dictionary

    Returns:
        MD5 hash string
    """
    # Sort keys for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def create_checkpoint_manager(
    output_dir: Union[str, Path],
    job_name: str,
) -> CheckpointManager:
    """
    Factory function to create a checkpoint manager.

    Args:
        output_dir: Output directory for the job
        job_name: Name of the job

    Returns:
        Configured CheckpointManager
    """
    checkpoint_dir = Path(output_dir) / ".checkpoints"

    # Create job ID from name and timestamp
    job_id = hashlib.md5(
        f"{job_name}_{time.time()}".encode()
    ).hexdigest()[:12]

    return CheckpointManager(checkpoint_dir, job_id)


def resume_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    job_id: str,
    validate_config_hash: Optional[str] = None,
) -> Optional[CheckpointManager]:
    """
    Resume from an existing checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint
        job_id: Job identifier
        validate_config_hash: Config hash to validate against

    Returns:
        CheckpointManager with loaded checkpoint or None
    """
    manager = CheckpointManager(checkpoint_dir, job_id)

    if manager.exists():
        try:
            manager.load(validate_config_hash)
            return manager
        except CheckpointValidationError as e:
            logger.error(f"Cannot resume: {e}")
            return None

    return None


def find_resumable_jobs(
    output_dir: Union[str, Path],
) -> List[Dict[str, Any]]:
    """
    Find all resumable jobs in an output directory.

    Args:
        output_dir: Directory to search

    Returns:
        List of resumable job info dictionaries
    """
    checkpoint_dir = Path(output_dir) / ".checkpoints"

    if not checkpoint_dir.exists():
        return []

    jobs = []
    for checkpoint_file in checkpoint_dir.glob("checkpoint_*.json"):
        if ".backup." in checkpoint_file.name:
            continue

        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            checkpoint = JobCheckpoint.from_dict(data)

            if not checkpoint.is_complete:
                jobs.append({
                    'job_id': checkpoint.job_id,
                    'job_name': checkpoint.job_name,
                    'progress_percent': checkpoint.progress_percent,
                    'completed_bins': checkpoint.completed_bins,
                    'total_bins': checkpoint.total_bins,
                    'last_update': checkpoint.updated_at,
                    'checkpoint_file': str(checkpoint_file),
                })
        except:
            continue

    return jobs
