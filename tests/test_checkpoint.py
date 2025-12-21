"""
Tests for Checkpoint System

Tests checkpoint creation, saving, loading, and cleanup.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from uuid import uuid4


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        from utils.ray_orchestration.checkpoint import Checkpoint
        from datetime import datetime

        job_id = uuid4()
        checkpoint = Checkpoint(
            job_id=job_id,
            checkpoint_id=1,
            created_at=datetime.now(),
            phase="processing",
            items_completed=500,
            items_total=1000,
            state={"current_index": 500},
        )

        assert checkpoint.job_id == job_id
        assert checkpoint.checkpoint_id == 1
        assert checkpoint.items_completed == 500
        assert checkpoint.items_total == 1000
        assert checkpoint.progress_percent == 50.0

    def test_checkpoint_serialization(self):
        """Test checkpoint serialization and deserialization."""
        from utils.ray_orchestration.checkpoint import Checkpoint
        from datetime import datetime

        job_id = uuid4()
        original = Checkpoint(
            job_id=job_id,
            checkpoint_id=5,
            created_at=datetime.now(),
            phase="denoising",
            items_completed=750,
            items_total=1000,
            state={"wavelet": "db4", "level": 4},
            metadata={"worker_id": "worker-1"},
        )

        # Serialize
        data = original.to_dict()
        assert data["job_id"] == str(job_id)
        assert data["checkpoint_id"] == 5
        assert data["phase"] == "denoising"

        # Deserialize
        restored = Checkpoint.from_dict(data)
        assert restored.job_id == original.job_id
        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.state == original.state
        assert restored.metadata == original.metadata


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def setup_method(self):
        """Create temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()
        # Reset singleton
        from utils.ray_orchestration.checkpoint import CheckpointManager
        CheckpointManager._instance = None

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from utils.ray_orchestration.checkpoint import CheckpointManager
        CheckpointManager._instance = None

    def test_manager_creates_directory(self):
        """Test that manager creates checkpoint directory."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        assert checkpoint_dir.exists()

    def test_create_and_save_checkpoint(self):
        """Test creating and saving a checkpoint."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        manager = CheckpointManager(self.temp_dir)
        job_id = uuid4()

        checkpoint = manager.create_checkpoint(
            job_id=job_id,
            phase="processing",
            items_completed=100,
            items_total=1000,
            state={"index": 100},
        )

        filepath = manager.save_checkpoint(checkpoint)

        assert filepath.exists()
        assert "checkpoint_00001" in str(filepath)

    def test_load_checkpoint(self):
        """Test loading a saved checkpoint."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        manager = CheckpointManager(self.temp_dir)
        job_id = uuid4()

        # Save checkpoint
        checkpoint = manager.create_checkpoint(
            job_id=job_id,
            phase="importing",
            items_completed=500,
            items_total=2000,
            state={"file_offset": 1024000},
        )
        filepath = manager.save_checkpoint(checkpoint)

        # Load checkpoint
        loaded = manager.load_checkpoint(filepath)

        assert loaded is not None
        assert loaded.job_id == job_id
        assert loaded.items_completed == 500
        assert loaded.state["file_offset"] == 1024000

    def test_load_latest_checkpoint(self):
        """Test loading the most recent checkpoint."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        manager = CheckpointManager(self.temp_dir)
        job_id = uuid4()

        # Save multiple checkpoints
        for i in range(3):
            checkpoint = manager.create_checkpoint(
                job_id=job_id,
                phase="processing",
                items_completed=(i + 1) * 100,
                items_total=1000,
            )
            manager.save_checkpoint(checkpoint)

        # Load latest
        latest = manager.load_latest_checkpoint(job_id)

        assert latest is not None
        assert latest.checkpoint_id == 3
        assert latest.items_completed == 300

    def test_list_checkpoints(self):
        """Test listing checkpoints for a job."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        manager = CheckpointManager(self.temp_dir)
        job_id = uuid4()

        # Save multiple checkpoints
        for i in range(5):
            checkpoint = manager.create_checkpoint(
                job_id=job_id,
                phase="processing",
                items_completed=i * 100,
                items_total=1000,
            )
            manager.save_checkpoint(checkpoint)

        # List checkpoints (should be limited to 3 by default)
        checkpoints = manager.list_checkpoints(job_id)

        # Default max is 3, so oldest are deleted
        assert len(checkpoints) == 3

    def test_delete_checkpoints(self):
        """Test deleting checkpoints for a job."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        manager = CheckpointManager(self.temp_dir)
        job_id = uuid4()

        # Save checkpoints
        for i in range(3):
            checkpoint = manager.create_checkpoint(
                job_id=job_id,
                phase="processing",
                items_completed=i * 100,
                items_total=1000,
            )
            manager.save_checkpoint(checkpoint)

        # Delete all
        count = manager.delete_checkpoints(job_id)

        assert count == 3
        assert len(manager.list_checkpoints(job_id)) == 0

    def test_disk_usage(self):
        """Test getting disk usage statistics."""
        from utils.ray_orchestration.checkpoint import CheckpointManager

        manager = CheckpointManager(self.temp_dir)

        # Create some checkpoints
        for _ in range(2):
            job_id = uuid4()
            checkpoint = manager.create_checkpoint(
                job_id=job_id,
                phase="test",
                items_completed=50,
                items_total=100,
            )
            manager.save_checkpoint(checkpoint)

        usage = manager.get_disk_usage()

        assert usage["job_count"] == 2
        assert usage["checkpoint_count"] == 2
        assert usage["total_size_mb"] >= 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Create temporary checkpoint directory."""
        self.temp_dir = tempfile.mkdtemp()
        from utils.ray_orchestration.checkpoint import CheckpointManager
        CheckpointManager._instance = None
        # Reset global manager
        import utils.ray_orchestration.checkpoint as checkpoint_module
        checkpoint_module._manager = None

    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        from utils.ray_orchestration.checkpoint import CheckpointManager
        CheckpointManager._instance = None

    def test_save_and_load_checkpoint(self):
        """Test convenience functions for save and load."""
        from utils.ray_orchestration.checkpoint import (
            get_checkpoint_manager,
            save_checkpoint,
            load_latest_checkpoint,
        )

        # Initialize with temp dir
        manager = get_checkpoint_manager(self.temp_dir)

        job_id = uuid4()

        # Save
        filepath = save_checkpoint(
            job_id=job_id,
            phase="testing",
            items_completed=25,
            items_total=100,
            state={"test": True},
        )

        assert filepath.exists()

        # Load
        loaded = load_latest_checkpoint(job_id)

        assert loaded is not None
        assert loaded.items_completed == 25
        assert loaded.state["test"] is True
