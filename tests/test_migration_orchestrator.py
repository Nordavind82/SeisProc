"""
Unit tests for migration orchestrator.

Tests:
- Orchestrator setup
- Sequential bin processing
- Progress tracking
- Checkpoint/resume
- Resource estimation
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.orchestrator import (
    MigrationOrchestrator,
    MigrationProgress,
    BinProgress,
    ProcessingMode,
)
from models.migration_job import MigrationJobConfig
from utils.dataset_indexer import DatasetIndex, DatasetIndexer, BinnedDataset
from seisio.gather_readers import NumpyDataReader, Gather


class TestBinProgress:
    """Tests for BinProgress."""

    def test_basic_creation(self):
        """Test basic progress creation."""
        bp = BinProgress(
            bin_name="test_bin",
            n_gathers_total=100,
        )

        assert bp.bin_name == "test_bin"
        assert bp.status == "pending"
        assert bp.n_gathers_processed == 0

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        bp = BinProgress(
            bin_name="test",
            n_gathers_total=100,
            n_gathers_processed=50,
        )

        assert bp.progress_percent == 50.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        import time

        bp = BinProgress(
            bin_name="test",
            start_time=time.time() - 10,  # Started 10 seconds ago
        )

        assert bp.elapsed_seconds >= 10.0


class TestMigrationProgress:
    """Tests for MigrationProgress."""

    def test_basic_creation(self):
        """Test basic progress creation."""
        progress = MigrationProgress(
            job_name="Test Job",
            n_bins_total=5,
        )

        assert progress.job_name == "Test Job"
        assert progress.n_bins_total == 5
        assert progress.n_bins_completed == 0

    def test_overall_percent(self):
        """Test overall percentage calculation."""
        progress = MigrationProgress(
            job_name="Test",
            n_bins_total=10,
            n_bins_completed=3,
        )

        assert progress.overall_percent == 30.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        progress = MigrationProgress(
            job_name="Test",
            n_bins_total=5,
            n_bins_completed=2,
        )
        progress.bin_progress['bin1'] = BinProgress(
            bin_name='bin1',
            status='completed',
            n_gathers_total=50,
            n_gathers_processed=50,
        )

        d = progress.to_dict()

        assert d['job_name'] == 'Test'
        assert d['n_bins_total'] == 5
        assert 'bin_progress' in d


class TestMigrationOrchestrator:
    """Tests for MigrationOrchestrator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_traces = 100
        n_samples = 200

        # Create geometry
        indexer = DatasetIndexer()
        offsets = np.random.uniform(0, 2000, n_traces)
        angles = np.random.uniform(0, 2 * np.pi, n_traces)

        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = offsets * np.sin(angles)
        receiver_y = offsets * np.cos(angles)

        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=n_samples,
            sample_rate_ms=4.0,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        # Create trace data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, reader

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories."""
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            f.write(b'dummy')
            input_file = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            yield input_file, output_dir

        Path(input_file).unlink()

    @pytest.fixture
    def job_config(self, temp_dirs):
        """Create sample job configuration."""
        input_file, output_dir = temp_dirs

        return MigrationJobConfig(
            name="Test Migration",
            input_file=input_file,
            output_directory=output_dir,
            binning_preset='full_stack',  # Single bin for simplicity
            velocity_v0=2500.0,
            time_min_ms=0.0,
            time_max_ms=800.0,  # Short for testing
            dt_ms=4.0,
            inline_min=1,
            inline_max=5,
            inline_step=1,
            xline_min=1,
            xline_max=5,
            xline_step=1,
            max_aperture_m=5000.0,
        )

    def test_orchestrator_creation(self, sample_data, job_config):
        """Test orchestrator creation."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
        )

        assert orchestrator.processing_mode == ProcessingMode.SEQUENTIAL
        assert orchestrator.enable_checkpointing is True

    def test_setup(self, sample_data, job_config):
        """Test orchestrator setup."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
        )

        orchestrator.setup()

        assert orchestrator._output_manager is not None
        assert orchestrator._binned_dataset is not None
        assert orchestrator._progress is not None
        assert orchestrator._migrator is not None

    def test_estimate_resources(self, sample_data, job_config):
        """Test resource estimation."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
        )

        estimates = orchestrator.estimate_resources()

        assert 'output_volume_gb' in estimates
        assert 'n_bins' in estimates
        assert 'n_traces' in estimates
        assert estimates['n_traces'] == 100

    def test_progress_tracking(self, sample_data, job_config):
        """Test progress is tracked during setup."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
        )

        orchestrator.setup()
        progress = orchestrator.get_progress()

        assert progress is not None
        assert progress.job_name == "Test Migration"
        assert progress.n_bins_total >= 1

    def test_progress_callback(self, sample_data, job_config):
        """Test progress callback is called."""
        index, reader = sample_data

        callback_calls = []

        def progress_callback(progress):
            callback_calls.append(progress.to_dict())

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
            progress_callback=progress_callback,
        )

        orchestrator.setup()

        # Callback should be called during run
        # For this test we just verify setup works with callback set
        assert orchestrator.progress_callback is not None

    def test_checkpoint_directory_creation(self, sample_data, job_config):
        """Test checkpoint directory is created."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
            enable_checkpointing=True,
        )

        orchestrator.setup()

        assert Path(orchestrator.checkpoint_dir).exists()

    def test_create_gather(self, sample_data, job_config):
        """Test gather creation from bin."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
        )

        orchestrator.setup()

        # Get first bin
        binning = job_config.get_binning_table()
        bin_name = binning.bins[0].name
        trace_numbers = orchestrator._binned_dataset.get_bin_trace_numbers(bin_name)

        # Create gather
        gather = orchestrator._create_gather(bin_name, trace_numbers)

        assert isinstance(gather, Gather)
        assert gather.gather_id == bin_name
        assert len(gather.offsets) == len(trace_numbers)

    def test_processing_mode_parallel(self, sample_data, job_config):
        """Test parallel processing mode setting."""
        index, reader = sample_data

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
            processing_mode=ProcessingMode.PARALLEL,
            max_workers=2,
        )

        assert orchestrator.processing_mode == ProcessingMode.PARALLEL
        assert orchestrator.max_workers == 2


class TestMigrationOrchestratorIntegration:
    """Integration tests for orchestrator with mocked migrator."""

    @pytest.fixture
    def mock_migrator_result(self):
        """Create mock migration result."""
        result = MagicMock()
        result.migrated_volume = np.zeros((201, 5, 5), dtype=np.float32)
        result.fold_volume = np.ones((201, 5, 5), dtype=np.int32)
        return result

    @pytest.fixture
    def sample_setup(self, tmp_path):
        """Create complete sample setup for integration test."""
        n_traces = 50
        n_samples = 100

        # Create index
        indexer = DatasetIndexer()
        offsets = np.random.uniform(0, 2000, n_traces)
        angles = np.random.uniform(0, 2 * np.pi, n_traces)

        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = offsets * np.sin(angles)
        receiver_y = offsets * np.cos(angles)

        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=n_samples,
            sample_rate_ms=4.0,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        # Create trace data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        # Create input file
        input_file = tmp_path / "input.sgy"
        input_file.write_bytes(b'dummy')

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create job config
        job_config = MigrationJobConfig(
            name="Integration Test",
            input_file=str(input_file),
            output_directory=str(output_dir),
            binning_preset='full_stack',
            velocity_v0=2500.0,
            time_min_ms=0.0,
            time_max_ms=800.0,
            dt_ms=4.0,
            inline_min=1,
            inline_max=5,
            xline_min=1,
            xline_max=5,
        )

        return index, reader, job_config

    def test_full_run_with_mock_migrator(self, sample_setup, mock_migrator_result):
        """Test full orchestrator run with mocked migrator."""
        index, reader, job_config = sample_setup

        orchestrator = MigrationOrchestrator(
            job_config=job_config,
            index=index,
            data_reader=reader,
            enable_checkpointing=False,  # Disable for test
        )

        orchestrator.setup()

        # Mock the migrator
        orchestrator._migrator = MagicMock()
        orchestrator._migrator.migrate_gather.return_value = mock_migrator_result

        # Run
        progress = orchestrator.run()

        # Verify completion
        assert progress.n_bins_completed >= 1
        assert progress.end_time is not None

        # Verify outputs were saved
        output_dir = Path(job_config.output_directory)
        assert any(output_dir.glob("*.npy"))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
