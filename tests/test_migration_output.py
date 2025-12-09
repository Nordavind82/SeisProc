"""
Unit tests for migration output manager.

Tests:
- Volume initialization
- Data accumulation
- Fold handling
- Stack generation
- File saving
- Header population
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from seisio.migration_output import (
    MigrationOutputManager,
    OutputVolumeInfo,
    create_output_manager,
)
from models.migration_config import OutputGrid
from models.binning import create_uniform_offset_binning


class TestMigrationOutputManager:
    """Tests for MigrationOutputManager."""

    @pytest.fixture
    def output_grid(self):
        """Create sample output grid."""
        return OutputGrid(
            n_time=100,
            n_inline=20,
            n_xline=20,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            inline_start=100,
            xline_start=200,
        )

    @pytest.fixture
    def binning_table(self):
        """Create sample binning table."""
        return create_uniform_offset_binning(0, 2000, 4)

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_initialization(self, output_grid, binning_table, temp_output_dir):
        """Test output manager initialization."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )

        manager.initialize()

        assert len(manager._volumes) == 4
        assert manager._stack_volume is not None
        assert manager._initialized

    def test_volume_shape(self, output_grid, binning_table, temp_output_dir):
        """Test output volume shape matches grid."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )

        manager.initialize()

        for volume in manager._volumes.values():
            assert volume.shape == (100, 20, 20)

    def test_add_migrated_data(self, output_grid, binning_table, temp_output_dir):
        """Test adding migrated data."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        # Create test data
        data = np.ones((100, 20, 20), dtype=np.float32) * 10.0
        bin_name = binning_table.bins[0].name

        manager.add_migrated_data(bin_name, data)

        # Check data was added
        assert np.allclose(manager._volumes[bin_name], 10.0)

        # Check stack was updated
        assert np.allclose(manager._stack_volume, 10.0)

    def test_accumulate_to_bin(self, output_grid, binning_table, temp_output_dir):
        """Test sample-by-sample accumulation."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
            create_fold=True,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name

        # Accumulate values
        for i in range(10):
            manager.accumulate_to_bin(bin_name, 50, 10, 10, 1.0)

        # Check accumulation
        assert manager._volumes[bin_name][50, 10, 10] == 10.0

        # Check fold
        assert manager._fold_volumes[bin_name][50, 10, 10] == 10

    def test_normalize_by_fold(self, output_grid, binning_table, temp_output_dir):
        """Test fold normalization."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
            create_fold=True,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name

        # Add data with fold
        for i in range(5):
            manager.accumulate_to_bin(bin_name, 50, 10, 10, 2.0)

        # Normalize
        manager.normalize_by_fold(min_fold=1)

        # Check normalized value (10.0 / 5 = 2.0)
        assert manager._volumes[bin_name][50, 10, 10] == 2.0

    def test_normalize_min_fold_threshold(self, output_grid, binning_table, temp_output_dir):
        """Test min fold threshold during normalization."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
            create_fold=True,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name

        # Add data with fold = 2 at one point
        manager.accumulate_to_bin(bin_name, 50, 10, 10, 4.0)
        manager.accumulate_to_bin(bin_name, 50, 10, 10, 4.0)

        # Add data with fold = 1 at another point
        manager.accumulate_to_bin(bin_name, 50, 11, 11, 4.0)

        # Normalize with min_fold=2
        manager.normalize_by_fold(min_fold=2)

        # Point with fold=2 should be normalized
        assert manager._volumes[bin_name][50, 10, 10] == 4.0

        # Point with fold=1 should be zeroed
        assert manager._volumes[bin_name][50, 11, 11] == 0.0

    def test_no_stack_option(self, output_grid, binning_table, temp_output_dir):
        """Test with stack creation disabled."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
            create_stack=False,
        )
        manager.initialize()

        assert manager._stack_volume is None
        assert 'stack' not in manager._volume_info

    def test_save_volumes_npy(self, output_grid, binning_table, temp_output_dir):
        """Test saving volumes as numpy."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
            output_format='npy',
        )
        manager.initialize()

        # Add some data
        bin_name = binning_table.bins[0].name
        data = np.ones((100, 20, 20), dtype=np.float32)
        manager.add_migrated_data(bin_name, data)

        # Save
        manager.save_all()

        # Check files exist
        for info in manager._volume_info.values():
            assert Path(info.filepath).exists()

    def test_get_volume(self, output_grid, binning_table, temp_output_dir):
        """Test getting volume by name."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name
        volume = manager.get_volume(bin_name)

        assert volume is not None
        assert volume.shape == (100, 20, 20)

        # Get stack volume
        stack = manager.get_volume('stack')
        assert stack is not None

    def test_get_output_headers(self, output_grid, binning_table, temp_output_dir):
        """Test header generation."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name
        headers = manager.get_output_headers(bin_name, 5, 10)

        assert headers['INLINE_3D'] == 105  # inline_start + idx
        assert headers['CROSSLINE_3D'] == 210  # xline_start + idx
        assert 'CDP_X' in headers
        assert 'CDP_Y' in headers
        assert 'OFFSET' in headers

    def test_get_summary(self, output_grid, binning_table, temp_output_dir):
        """Test summary generation."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        summary = manager.get_summary()

        assert summary['n_bins'] == 4
        assert summary['create_stack'] is True
        assert 'volumes' in summary

    def test_finalize_bin(self, output_grid, binning_table, temp_output_dir):
        """Test bin finalization."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name
        assert not manager._volume_info[bin_name].is_complete

        manager.finalize_bin(bin_name)

        assert manager._volume_info[bin_name].is_complete

    def test_not_initialized_error(self, output_grid, binning_table, temp_output_dir):
        """Test error when adding data before initialization."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )

        data = np.ones((100, 20, 20), dtype=np.float32)
        bin_name = binning_table.bins[0].name

        with pytest.raises(RuntimeError, match="not initialized"):
            manager.add_migrated_data(bin_name, data)

    def test_unknown_bin_error(self, output_grid, binning_table, temp_output_dir):
        """Test error for unknown bin name."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        data = np.ones((100, 20, 20), dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown bin"):
            manager.add_migrated_data("nonexistent_bin", data)


class TestOutputVolumeInfo:
    """Tests for OutputVolumeInfo."""

    def test_basic_creation(self):
        """Test basic info creation."""
        info = OutputVolumeInfo(
            name="test_bin",
            filepath="/output/test.npy",
            bin_name="test_bin",
            offset_center=500.0,
            azimuth_center=45.0,
        )

        assert info.name == "test_bin"
        assert info.offset_center == 500.0
        assert info.n_traces_migrated == 0
        assert not info.is_complete


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
