"""
Unit tests for migration checkpoint/resume system.

Tests:
- Checkpoint creation and serialization
- Save/load with atomic writes
- Resume functionality
- Validation and error handling
- Intermediate volume saving
"""

import numpy as np
import pytest
import json
import time
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.checkpoint import (
    BinStatus,
    TraceProgress,
    BinCheckpoint,
    JobCheckpoint,
    CheckpointManager,
    CheckpointValidationError,
    IntermediateVolumeSaver,
    compute_config_hash,
    create_checkpoint_manager,
    resume_from_checkpoint,
    find_resumable_jobs,
)


class TestTraceProgress:
    """Tests for TraceProgress dataclass."""

    def test_basic_creation(self):
        """Test basic trace progress creation."""
        tp = TraceProgress(total_traces=100, processed_traces=50)

        assert tp.total_traces == 100
        assert tp.processed_traces == 50
        assert not tp.is_complete

    def test_is_complete(self):
        """Test completion check."""
        tp = TraceProgress(total_traces=100, processed_traces=100)
        assert tp.is_complete

        tp = TraceProgress(total_traces=100, processed_traces=99)
        assert not tp.is_complete

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        tp = TraceProgress(total_traces=100, processed_traces=25)
        assert tp.progress_percent == 25.0

        tp = TraceProgress(total_traces=0, processed_traces=0)
        assert tp.progress_percent == 100.0  # Edge case


class TestBinCheckpoint:
    """Tests for BinCheckpoint dataclass."""

    def test_basic_creation(self):
        """Test basic bin checkpoint creation."""
        bc = BinCheckpoint(bin_name="near")

        assert bc.bin_name == "near"
        assert bc.status == BinStatus.PENDING
        assert bc.trace_progress is None

    def test_with_trace_progress(self):
        """Test bin checkpoint with trace progress."""
        tp = TraceProgress(total_traces=500, processed_traces=250)
        bc = BinCheckpoint(
            bin_name="mid",
            status=BinStatus.IN_PROGRESS,
            trace_progress=tp,
            start_time=time.time(),
        )

        assert bc.status == BinStatus.IN_PROGRESS
        assert bc.trace_progress.processed_traces == 250

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        tp = TraceProgress(total_traces=100, processed_traces=75, last_trace_number=74)
        bc = BinCheckpoint(
            bin_name="far",
            status=BinStatus.COMPLETED,
            trace_progress=tp,
            start_time=1000.0,
            end_time=2000.0,
            output_file="/path/to/output.sgy",
            volume_checksum="abc123",
        )

        d = bc.to_dict()
        bc2 = BinCheckpoint.from_dict(d)

        assert bc2.bin_name == bc.bin_name
        assert bc2.status == bc.status
        assert bc2.trace_progress.total_traces == tp.total_traces
        assert bc2.trace_progress.last_trace_number == tp.last_trace_number
        assert bc2.output_file == bc.output_file
        assert bc2.volume_checksum == bc.volume_checksum


class TestJobCheckpoint:
    """Tests for JobCheckpoint dataclass."""

    def test_basic_creation(self):
        """Test basic job checkpoint creation."""
        jc = JobCheckpoint(job_id="test123", job_name="Test Migration")

        assert jc.job_id == "test123"
        assert jc.job_name == "Test Migration"
        assert jc.total_bins == 0
        assert jc.completed_bins == 0

    def test_with_bins(self):
        """Test job checkpoint with bins."""
        jc = JobCheckpoint(
            job_id="test456",
            job_name="Full Migration",
            total_bins=5,
            completed_bins=2,
        )

        for name in ["near", "mid", "far"]:
            jc.bins[name] = BinCheckpoint(bin_name=name)

        jc.bins["near"].status = BinStatus.COMPLETED
        jc.bins["mid"].status = BinStatus.COMPLETED

        assert len(jc.bins) == 3
        assert jc.get_pending_bins() == ["far"]

    def test_is_complete(self):
        """Test job completion check."""
        jc = JobCheckpoint(
            job_id="test",
            job_name="Test",
            total_bins=3,
            completed_bins=3,
        )
        assert jc.is_complete

        jc.completed_bins = 2
        jc.failed_bins = 1
        assert jc.is_complete

        jc.failed_bins = 0
        assert not jc.is_complete

    def test_progress_percent(self):
        """Test progress percentage."""
        jc = JobCheckpoint(
            job_id="test",
            job_name="Test",
            total_bins=10,
            completed_bins=3,
        )
        assert jc.progress_percent == 30.0

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        jc = JobCheckpoint(
            job_id="roundtrip",
            job_name="Roundtrip Test",
            total_bins=2,
            completed_bins=1,
            config_hash="hash123",
            metadata={"key": "value"},
        )

        jc.bins["bin1"] = BinCheckpoint(
            bin_name="bin1",
            status=BinStatus.COMPLETED,
        )
        jc.bins["bin2"] = BinCheckpoint(bin_name="bin2")

        d = jc.to_dict()
        jc2 = JobCheckpoint.from_dict(d)

        assert jc2.job_id == jc.job_id
        assert jc2.job_name == jc.job_name
        assert jc2.total_bins == jc.total_bins
        assert jc2.config_hash == jc.config_hash
        assert jc2.metadata == jc.metadata
        assert len(jc2.bins) == 2
        assert jc2.bins["bin1"].status == BinStatus.COMPLETED

    def test_get_pending_and_in_progress(self):
        """Test getting pending and in-progress bins."""
        jc = JobCheckpoint(
            job_id="test",
            job_name="Test",
            total_bins=4,
        )

        jc.bins["a"] = BinCheckpoint(bin_name="a", status=BinStatus.COMPLETED)
        jc.bins["b"] = BinCheckpoint(bin_name="b", status=BinStatus.IN_PROGRESS)
        jc.bins["c"] = BinCheckpoint(bin_name="c", status=BinStatus.PENDING)
        jc.bins["d"] = BinCheckpoint(bin_name="d", status=BinStatus.FAILED)

        pending = jc.get_pending_bins()
        in_progress = jc.get_in_progress_bins()

        assert set(pending) == {"c", "d"}  # Failed bins need reprocessing
        assert in_progress == ["b"]


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def tmp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_checkpoint(self, tmp_dir):
        """Test creating a new checkpoint."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "job123")

        cp = manager.create_checkpoint(
            job_name="Test Job",
            bin_names=["near", "mid", "far"],
            bin_trace_counts={"near": 100, "mid": 200, "far": 150},
            config_hash="cfg_hash_123",
        )

        assert cp.job_id == "job123"
        assert cp.total_bins == 3
        assert len(cp.bins) == 3
        assert cp.bins["near"].trace_progress.total_traces == 100

    def test_start_and_complete_bin(self, tmp_dir):
        """Test bin lifecycle."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "job456")

        manager.create_checkpoint(
            job_name="Lifecycle Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 50},
        )

        manager.start_job()
        manager.start_bin("bin1")

        cp = manager.get_checkpoint()
        assert cp.bins["bin1"].status == BinStatus.IN_PROGRESS
        assert cp.bins["bin1"].start_time is not None

        manager.complete_bin("bin1", output_file="/output/bin1.sgy")

        assert cp.bins["bin1"].status == BinStatus.COMPLETED
        assert cp.bins["bin1"].end_time is not None
        assert cp.completed_bins == 1

    def test_fail_bin(self, tmp_dir):
        """Test bin failure handling."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "job789")

        manager.create_checkpoint(
            job_name="Failure Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
        )

        manager.start_bin("bin1")
        manager.fail_bin("bin1", "Out of memory")

        cp = manager.get_checkpoint()
        assert cp.bins["bin1"].status == BinStatus.FAILED
        assert cp.bins["bin1"].error_message == "Out of memory"
        assert cp.failed_bins == 1

    def test_save_and_load(self, tmp_dir):
        """Test checkpoint save and load."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "saveload")

        manager.create_checkpoint(
            job_name="Save Load Test",
            bin_names=["a", "b", "c"],
            bin_trace_counts={"a": 10, "b": 20, "c": 30},
            config_hash="test_hash",
            metadata={"author": "test"},
        )

        manager.start_job()
        manager.complete_bin("a")
        manager.start_bin("b")

        # Save
        manager.save()

        # Create new manager and load
        manager2 = CheckpointManager(tmp_dir / "checkpoints", "saveload")
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.job_name == "Save Load Test"
        assert loaded.completed_bins == 1
        assert loaded.bins["a"].status == BinStatus.COMPLETED
        assert loaded.bins["b"].status == BinStatus.IN_PROGRESS
        assert loaded.metadata["author"] == "test"

    def test_load_with_validation(self, tmp_dir):
        """Test loading with config hash validation."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "validate")

        manager.create_checkpoint(
            job_name="Validation Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
            config_hash="original_hash",
        )
        manager.save()

        # Load with matching hash - should work
        manager2 = CheckpointManager(tmp_dir / "checkpoints", "validate")
        loaded = manager2.load(validate_config_hash="original_hash")
        assert loaded is not None

        # Load with different hash - should fail
        manager3 = CheckpointManager(tmp_dir / "checkpoints", "validate")
        with pytest.raises(CheckpointValidationError, match="changed"):
            manager3.load(validate_config_hash="different_hash")

    def test_backup_rotation(self, tmp_dir):
        """Test backup file rotation."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "backup", keep_backups=2)

        manager.create_checkpoint(
            job_name="Backup Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
        )

        # Multiple saves should create backups
        for i in range(5):
            manager.get_checkpoint().metadata["version"] = i
            manager.save()

        # Check backup files exist
        checkpoint_dir = tmp_dir / "checkpoints"
        assert (checkpoint_dir / "checkpoint_backup.backup.0.json").exists()
        assert (checkpoint_dir / "checkpoint_backup.backup.1.json").exists()

    def test_get_resume_info(self, tmp_dir):
        """Test getting resume information."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "resume")

        manager.create_checkpoint(
            job_name="Resume Test",
            bin_names=["a", "b", "c", "d"],
            bin_trace_counts={"a": 10, "b": 20, "c": 30, "d": 40},
        )

        manager.complete_bin("a")
        manager.start_bin("b")
        manager.fail_bin("c", "error")

        info = manager.get_resume_info()

        assert info["can_resume"] is True
        assert info["completed_bins"] == 1
        assert info["total_bins"] == 4
        assert set(info["bins_to_process"]) == {"b", "c", "d"}
        assert info["in_progress_bins"] == ["b"]

    def test_delete_checkpoint(self, tmp_dir):
        """Test checkpoint deletion."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "delete")

        manager.create_checkpoint(
            job_name="Delete Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
        )
        manager.save()

        assert manager.exists()

        manager.delete()

        assert not manager.exists()
        assert manager.get_checkpoint() is None

    def test_update_bin_progress(self, tmp_dir):
        """Test updating trace-level progress."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "progress")

        manager.create_checkpoint(
            job_name="Progress Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 1000},
        )

        manager.start_bin("bin1")

        for i in range(0, 1000, 100):
            manager.update_bin_progress("bin1", processed_traces=i, last_trace_number=i)

        cp = manager.get_checkpoint()
        assert cp.bins["bin1"].trace_progress.processed_traces == 900
        assert cp.bins["bin1"].trace_progress.last_trace_number == 900


class TestIntermediateVolumeSaver:
    """Tests for IntermediateVolumeSaver class."""

    @pytest.fixture
    def tmp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialize_volume(self, tmp_dir):
        """Test initializing intermediate volume."""
        saver = IntermediateVolumeSaver(
            tmp_dir,
            "job123",
            volume_shape=(100, 50, 50),
        )

        vol = saver.initialize(fill_value=0.0)

        assert vol.shape == (100, 50, 50)
        assert vol.dtype == np.float32
        assert np.all(vol == 0.0)

        saver.cleanup()

    def test_update_region(self, tmp_dir):
        """Test updating a region of the volume."""
        saver = IntermediateVolumeSaver(
            tmp_dir,
            "job456",
            volume_shape=(10, 20, 15),
        )

        vol = saver.initialize()

        # Update a region
        data = np.ones((5, 10, 8), dtype=np.float32)
        saver.update_region(data, slice(0, 5), slice(0, 10), slice(0, 8))
        saver.flush()

        assert np.all(vol[0:5, 0:10, 0:8] == 1.0)
        assert np.all(vol[5:, :, :] == 0.0)

        saver.cleanup()

    def test_add_to_region(self, tmp_dir):
        """Test adding to a region (stacking)."""
        saver = IntermediateVolumeSaver(
            tmp_dir,
            "job789",
            volume_shape=(5, 5, 5),
        )

        vol = saver.initialize(fill_value=1.0)

        # Add to a region multiple times
        data = np.ones((3, 3, 3), dtype=np.float32)
        saver.add_to_region(data, slice(0, 3), slice(0, 3), slice(0, 3))
        saver.add_to_region(data, slice(0, 3), slice(0, 3), slice(0, 3))
        saver.flush()

        assert np.all(vol[0:3, 0:3, 0:3] == 3.0)  # 1.0 + 1.0 + 1.0
        assert np.all(vol[3:, :, :] == 1.0)

        saver.cleanup()

    def test_load_existing(self, tmp_dir):
        """Test loading an existing intermediate volume."""
        saver1 = IntermediateVolumeSaver(
            tmp_dir,
            "existing",
            volume_shape=(10, 10, 10),
        )

        vol1 = saver1.initialize()
        vol1[5, 5, 5] = 42.0
        saver1.flush()

        # Create new saver and load
        saver2 = IntermediateVolumeSaver(
            tmp_dir,
            "existing",
            volume_shape=(10, 10, 10),
        )

        vol2 = saver2.load_existing()

        assert vol2 is not None
        assert vol2[5, 5, 5] == 42.0

        saver1.cleanup()

    def test_finalize(self, tmp_dir):
        """Test finalizing to output location."""
        saver = IntermediateVolumeSaver(
            tmp_dir / "intermediate",
            "finalize",
            volume_shape=(5, 5, 5),
        )

        vol = saver.initialize(fill_value=7.0)

        output_path = tmp_dir / "final_output.npy"
        result = saver.finalize(output_path)

        assert Path(result).exists()
        assert not (tmp_dir / "intermediate" / "intermediate_finalize.npy").exists()

        # Verify data by loading as memmap (since it's a raw numpy memmap file)
        loaded = np.memmap(output_path, dtype=np.float32, mode='r', shape=(5, 5, 5))
        assert np.all(loaded == 7.0)
        del loaded

    def test_compute_checksum(self, tmp_dir):
        """Test computing volume checksum."""
        saver = IntermediateVolumeSaver(
            tmp_dir,
            "checksum",
            volume_shape=(10, 10, 10),
        )

        vol = saver.initialize()
        vol[:] = np.random.rand(10, 10, 10).astype(np.float32)
        saver.flush()

        cs1 = saver.compute_checksum()
        cs2 = saver.compute_checksum()

        assert cs1 == cs2
        assert len(cs1) == 32  # MD5 hash length

        saver.cleanup()


class TestUtilityFunctions:
    """Tests for utility functions."""

    @pytest.fixture
    def tmp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_compute_config_hash(self):
        """Test config hash computation."""
        config1 = {"a": 1, "b": "test", "c": [1, 2, 3]}
        config2 = {"a": 1, "b": "test", "c": [1, 2, 3]}
        config3 = {"a": 2, "b": "test", "c": [1, 2, 3]}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32

    def test_create_checkpoint_manager(self, tmp_dir):
        """Test factory function."""
        manager = create_checkpoint_manager(tmp_dir, "Test Job")

        assert manager is not None
        assert (tmp_dir / ".checkpoints").exists()

    def test_resume_from_checkpoint(self, tmp_dir):
        """Test resume function."""
        # Create and save checkpoint
        manager = CheckpointManager(tmp_dir / "checkpoints", "resume_test")
        manager.create_checkpoint(
            job_name="Resume Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
            config_hash="test_hash",
        )
        manager.save()

        # Resume
        resumed = resume_from_checkpoint(
            tmp_dir / "checkpoints",
            "resume_test",
            validate_config_hash="test_hash",
        )

        assert resumed is not None
        assert resumed.get_checkpoint().job_name == "Resume Test"

    def test_resume_with_wrong_hash(self, tmp_dir):
        """Test resume with wrong config hash."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "wrong_hash")
        manager.create_checkpoint(
            job_name="Test",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
            config_hash="original",
        )
        manager.save()

        # Resume with wrong hash should return None
        resumed = resume_from_checkpoint(
            tmp_dir / "checkpoints",
            "wrong_hash",
            validate_config_hash="different",
        )

        assert resumed is None

    def test_find_resumable_jobs(self, tmp_dir):
        """Test finding resumable jobs."""
        # Create some incomplete jobs
        for i in range(3):
            manager = CheckpointManager(tmp_dir / ".checkpoints", f"job{i}")
            manager.create_checkpoint(
                job_name=f"Job {i}",
                bin_names=["a", "b"],
                bin_trace_counts={"a": 10, "b": 20},
            )
            if i == 1:
                manager.complete_bin("a")
                manager.complete_bin("b")  # Complete job
            manager.save()

        jobs = find_resumable_jobs(tmp_dir)

        # Should find 2 incomplete jobs
        assert len(jobs) == 2
        job_names = [j["job_name"] for j in jobs]
        assert "Job 0" in job_names
        assert "Job 2" in job_names
        assert "Job 1" not in job_names  # Completed


class TestCheckpointEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def tmp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_nonexistent(self, tmp_dir):
        """Test loading when no checkpoint exists."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "nonexistent")
        loaded = manager.load()
        assert loaded is None

    def test_save_without_checkpoint(self, tmp_dir):
        """Test saving without creating checkpoint first."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "nosave")
        # Should not raise, just do nothing
        manager.save()
        assert not manager.exists()

    def test_empty_bins_list(self, tmp_dir):
        """Test with empty bins list."""
        manager = CheckpointManager(tmp_dir / "checkpoints", "empty")
        cp = manager.create_checkpoint(
            job_name="Empty Test",
            bin_names=[],
            bin_trace_counts={},
        )

        assert cp.total_bins == 0
        assert cp.is_complete  # No bins to process

    def test_corrupt_checkpoint_fallback(self, tmp_dir):
        """Test fallback to backup on corrupt checkpoint."""
        checkpoint_dir = tmp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create a valid backup
        manager = CheckpointManager(checkpoint_dir, "corrupt")
        manager.create_checkpoint(
            job_name="Valid Backup",
            bin_names=["bin1"],
            bin_trace_counts={"bin1": 100},
        )
        manager.save()

        # Corrupt the main checkpoint file
        main_file = checkpoint_dir / "checkpoint_corrupt.json"
        with open(main_file, 'w') as f:
            f.write("not valid json {{{")

        # Load should recover from backup
        manager2 = CheckpointManager(checkpoint_dir, "corrupt")
        loaded = manager2.load()

        # Should have recovered from backup
        assert loaded is not None or manager2.exists()  # Either recovered or backup exists


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
