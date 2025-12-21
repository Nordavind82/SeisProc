"""
Checkpoint System for Job Recovery

Provides save/restore capability for long-running jobs.
Enables pause/resume and recovery from failures.
"""

import gzip
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from uuid import UUID
import pickle

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """
    Represents a saved checkpoint for a job.

    Attributes
    ----------
    job_id : UUID
        Job identifier
    checkpoint_id : int
        Sequential checkpoint number
    created_at : datetime
        When checkpoint was created
    phase : str
        Current processing phase
    items_completed : int
        Number of items completed
    items_total : int
        Total items to process
    state : dict
        Arbitrary state data (processor-specific)
    metadata : dict
        Additional metadata
    """

    job_id: UUID
    checkpoint_id: int
    created_at: datetime
    phase: str
    items_completed: int
    items_total: int
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": str(self.job_id),
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "phase": self.phase,
            "items_completed": self.items_completed,
            "items_total": self.items_total,
            "state": self.state,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Deserialize from dictionary."""
        return cls(
            job_id=UUID(data["job_id"]),
            checkpoint_id=data["checkpoint_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            phase=data["phase"],
            items_completed=data["items_completed"],
            items_total=data["items_total"],
            state=data.get("state", {}),
            metadata=data.get("metadata", {}),
        )

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.items_total == 0:
            return 0.0
        return (self.items_completed / self.items_total) * 100.0


class CheckpointManager:
    """
    Manages checkpoints for job recovery.

    Thread-safe singleton that handles:
    - Creating and saving checkpoints
    - Loading checkpoints for resume
    - Cleanup of old checkpoints
    - Atomic checkpoint writes

    Usage
    -----
    >>> manager = CheckpointManager(checkpoint_dir=".checkpoints")
    >>>
    >>> # Save checkpoint
    >>> checkpoint = manager.create_checkpoint(
    ...     job_id=job.id,
    ...     phase="processing",
    ...     items_completed=500,
    ...     items_total=1000,
    ...     state={"current_index": 500}
    ... )
    >>> manager.save_checkpoint(checkpoint)
    >>>
    >>> # Load latest checkpoint
    >>> restored = manager.load_latest_checkpoint(job.id)
    >>> if restored:
    ...     resume_from = restored.state["current_index"]
    """

    _instance: Optional['CheckpointManager'] = None
    _lock = threading.Lock()

    def __new__(cls, checkpoint_dir: str = ".checkpoints") -> 'CheckpointManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        if self._initialized:
            return

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_counts: Dict[UUID, int] = {}
        self._max_checkpoints = 3  # Keep last N checkpoints per job
        self._compress = True

        # Create checkpoint directory
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info(f"CheckpointManager initialized: {self._checkpoint_dir}")

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        return self._checkpoint_dir

    def create_checkpoint(
        self,
        job_id: UUID,
        phase: str,
        items_completed: int,
        items_total: int,
        state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint for a job.

        Parameters
        ----------
        job_id : UUID
            Job identifier
        phase : str
            Current processing phase
        items_completed : int
            Number of items completed
        items_total : int
            Total items to process
        state : dict, optional
            Processor-specific state data
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        Checkpoint
            The created checkpoint
        """
        # Get next checkpoint ID for this job
        with self._lock:
            checkpoint_id = self._checkpoint_counts.get(job_id, 0) + 1
            self._checkpoint_counts[job_id] = checkpoint_id

        return Checkpoint(
            job_id=job_id,
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(),
            phase=phase,
            items_completed=items_completed,
            items_total=items_total,
            state=state or {},
            metadata=metadata or {},
        )

    def save_checkpoint(self, checkpoint: Checkpoint) -> Path:
        """
        Save a checkpoint to disk.

        Uses atomic write (write to temp, then rename) to prevent corruption.

        Parameters
        ----------
        checkpoint : Checkpoint
            Checkpoint to save

        Returns
        -------
        Path
            Path to saved checkpoint file
        """
        job_dir = self._checkpoint_dir / str(checkpoint.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        filename = f"checkpoint_{checkpoint.checkpoint_id:05d}"
        if self._compress:
            filename += ".json.gz"
        else:
            filename += ".json"

        filepath = job_dir / filename
        temp_path = filepath.with_suffix(".tmp")

        try:
            # Write to temp file
            data = json.dumps(checkpoint.to_dict(), indent=2)

            if self._compress:
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    f.write(data)
            else:
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(data)

            # Atomic rename
            temp_path.rename(filepath)

            logger.debug(
                f"Checkpoint saved: {filepath} "
                f"({checkpoint.items_completed}/{checkpoint.items_total})"
            )

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(checkpoint.job_id)

            return filepath

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(self, filepath: Path) -> Optional[Checkpoint]:
        """
        Load a checkpoint from file.

        Parameters
        ----------
        filepath : Path
            Path to checkpoint file

        Returns
        -------
        Checkpoint or None
            Loaded checkpoint, or None if file doesn't exist
        """
        if not filepath.exists():
            return None

        try:
            if filepath.suffix == ".gz" or str(filepath).endswith(".json.gz"):
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

            return Checkpoint.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return None

    def load_latest_checkpoint(self, job_id: UUID) -> Optional[Checkpoint]:
        """
        Load the most recent checkpoint for a job.

        Parameters
        ----------
        job_id : UUID
            Job identifier

        Returns
        -------
        Checkpoint or None
            Latest checkpoint, or None if no checkpoints exist
        """
        job_dir = self._checkpoint_dir / str(job_id)

        if not job_dir.exists():
            return None

        # Find checkpoint files
        checkpoints = list(job_dir.glob("checkpoint_*.json*"))
        if not checkpoints:
            return None

        # Sort by checkpoint ID (in filename)
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1].split(".")[0]))

        # Load the latest
        return self.load_checkpoint(checkpoints[-1])

    def list_checkpoints(self, job_id: UUID) -> List[Path]:
        """
        List all checkpoint files for a job.

        Parameters
        ----------
        job_id : UUID
            Job identifier

        Returns
        -------
        list
            Paths to checkpoint files, sorted by ID
        """
        job_dir = self._checkpoint_dir / str(job_id)

        if not job_dir.exists():
            return []

        checkpoints = list(job_dir.glob("checkpoint_*.json*"))
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1].split(".")[0]))

        return checkpoints

    def delete_checkpoints(self, job_id: UUID) -> int:
        """
        Delete all checkpoints for a job.

        Parameters
        ----------
        job_id : UUID
            Job identifier

        Returns
        -------
        int
            Number of checkpoints deleted
        """
        job_dir = self._checkpoint_dir / str(job_id)

        if not job_dir.exists():
            return 0

        count = 0
        for checkpoint in job_dir.glob("checkpoint_*.json*"):
            checkpoint.unlink()
            count += 1

        # Remove directory if empty
        try:
            job_dir.rmdir()
        except OSError:
            pass  # Directory not empty

        # Reset counter
        with self._lock:
            self._checkpoint_counts.pop(job_id, None)

        logger.info(f"Deleted {count} checkpoints for job {job_id}")
        return count

    def _cleanup_old_checkpoints(self, job_id: UUID):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoints = self.list_checkpoints(job_id)

        while len(checkpoints) > self._max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.debug(f"Removed old checkpoint: {oldest}")

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get checkpoint disk usage statistics."""
        total_size = 0
        job_count = 0
        checkpoint_count = 0

        for job_dir in self._checkpoint_dir.iterdir():
            if job_dir.is_dir():
                job_count += 1
                for checkpoint in job_dir.glob("checkpoint_*.json*"):
                    checkpoint_count += 1
                    total_size += checkpoint.stat().st_size

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "job_count": job_count,
            "checkpoint_count": checkpoint_count,
            "checkpoint_dir": str(self._checkpoint_dir),
        }


# Convenience functions

_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(checkpoint_dir: str = ".checkpoints") -> CheckpointManager:
    """Get the global checkpoint manager."""
    global _manager
    if _manager is None:
        _manager = CheckpointManager(checkpoint_dir)
    return _manager


def save_checkpoint(
    job_id: UUID,
    phase: str,
    items_completed: int,
    items_total: int,
    state: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convenience function to create and save a checkpoint."""
    manager = get_checkpoint_manager()
    checkpoint = manager.create_checkpoint(
        job_id=job_id,
        phase=phase,
        items_completed=items_completed,
        items_total=items_total,
        state=state,
    )
    return manager.save_checkpoint(checkpoint)


def load_latest_checkpoint(job_id: UUID) -> Optional[Checkpoint]:
    """Convenience function to load the latest checkpoint."""
    manager = get_checkpoint_manager()
    return manager.load_latest_checkpoint(job_id)
