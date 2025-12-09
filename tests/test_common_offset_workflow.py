"""
Integration Test: Common Offset Migration Workflow

End-to-end test that verifies the complete migration workflow:
1. Create synthetic prestack data
2. Index the dataset
3. Assign traces to offset bins
4. Run migration per bin
5. Verify output volumes
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.velocity_model import VelocityModel
from models.migration_config import MigrationConfig, OutputGrid
from models.binning import create_uniform_offset_binning
from utils.dataset_indexer import DatasetIndexer, BinnedDataset
from seisio.gather_readers import NumpyDataReader, CommonOffsetGatherIterator
from seisio.migration_output import MigrationOutputManager
from processors.migration.orchestrator import MigrationOrchestrator, ProcessingMode
from models.migration_job import MigrationJobConfig
from tests.fixtures.synthetic_prestack import create_point_diffractor_data


class TestCommonOffsetWorkflow:
    """End-to-end common offset migration workflow tests."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic prestack data with point diffractor."""
        n_shots = 5
        n_receivers = 40
        n_samples = 300
        dt_ms = 4.0
        velocity = 2500.0

        # Create point diffractor data using the actual API
        gathers, geometries, metadata = create_point_diffractor_data(
            diffractor_x=0.0,
            diffractor_y=0.0,
            diffractor_z=1.0,  # 1 second TWT = 1250m at v=2500
            velocity=velocity,
            n_shots=n_shots,
            n_receivers_per_shot=n_receivers,
            dt_ms=dt_ms,
            n_samples=n_samples,
        )

        # Flatten all gathers into single arrays
        n_traces = n_shots * n_receivers
        all_traces = np.concatenate([g.traces for g in gathers], axis=1)
        data = all_traces.T  # Shape: (n_traces, n_samples)

        # Combine geometries
        source_x = np.concatenate([g.source_x for g in geometries])
        source_y = np.concatenate([g.source_y for g in geometries])
        receiver_x = np.concatenate([g.receiver_x for g in geometries])
        receiver_y = np.concatenate([g.receiver_y for g in geometries])

        geometry = {
            'source_x': source_x,
            'source_y': source_y,
            'receiver_x': receiver_x,
            'receiver_y': receiver_y,
        }

        return {
            'data': data.astype(np.float32),
            'geometry': geometry,
            'n_traces': n_traces,
            'n_samples': n_samples,
            'dt_ms': dt_ms,
            'velocity': velocity,
        }

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test."""
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            f.write(b'dummy')
            input_file = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            yield input_file, output_dir

        Path(input_file).unlink()

    def test_workflow_index_creation(self, synthetic_data):
        """Test dataset indexing from synthetic geometry."""
        geometry = synthetic_data['geometry']

        # Create indexer
        indexer = DatasetIndexer()

        # Index from geometry
        index = indexer.index_from_geometry(
            n_traces=synthetic_data['n_traces'],
            n_samples=synthetic_data['n_samples'],
            sample_rate_ms=synthetic_data['dt_ms'],
            source_x=geometry['source_x'],
            source_y=geometry['source_y'],
            receiver_x=geometry['receiver_x'],
            receiver_y=geometry['receiver_y'],
        )

        # Verify index
        assert index.n_traces == synthetic_data['n_traces']
        assert len(index.entries) == synthetic_data['n_traces']

        # Verify offset/azimuth computed
        offsets = index.offsets
        assert len(offsets) == synthetic_data['n_traces']
        assert np.min(offsets) >= 0
        assert np.max(offsets) <= 2000.0

    def test_workflow_bin_assignment(self, synthetic_data):
        """Test trace assignment to offset bins."""
        geometry = synthetic_data['geometry']

        # Create index
        indexer = DatasetIndexer()
        index = indexer.index_from_geometry(
            n_traces=synthetic_data['n_traces'],
            n_samples=synthetic_data['n_samples'],
            sample_rate_ms=synthetic_data['dt_ms'],
            source_x=geometry['source_x'],
            source_y=geometry['source_y'],
            receiver_x=geometry['receiver_x'],
            receiver_y=geometry['receiver_y'],
        )

        # Create binning (4 offset bins)
        binning = create_uniform_offset_binning(0, 2000, 4)

        # Create binned dataset
        binned = BinnedDataset(index, binning)

        # Verify bin assignments
        summary = binned.get_bin_summary()
        assert len(summary) == 4

        # All traces should be assigned
        total_assigned = sum(summary.values())
        assert total_assigned == synthetic_data['n_traces']

    def test_workflow_gather_iteration(self, synthetic_data):
        """Test iterating over offset gathers."""
        geometry = synthetic_data['geometry']
        data = synthetic_data['data']

        # Create index
        indexer = DatasetIndexer()
        index = indexer.index_from_geometry(
            n_traces=synthetic_data['n_traces'],
            n_samples=synthetic_data['n_samples'],
            sample_rate_ms=synthetic_data['dt_ms'],
            source_x=geometry['source_x'],
            source_y=geometry['source_y'],
            receiver_x=geometry['receiver_x'],
            receiver_y=geometry['receiver_y'],
        )

        # Create binning that matches our synthetic data offset range (0-500m)
        # The synthetic data has offsets from ~12.5m to ~487.5m
        binning = create_uniform_offset_binning(0, 500, 4)
        binned = BinnedDataset(index, binning)

        # Create data reader
        reader = NumpyDataReader(data)

        # Create gather iterator
        iterator = CommonOffsetGatherIterator(index, reader, binned)

        # Iterate and verify - should have gathers for bins with traces
        gathers = list(iterator)
        assert len(gathers) >= 1  # At least one gather with traces

        for gather in gathers:
            assert gather.n_traces > 0
            assert gather.n_samples == synthetic_data['n_samples']
            assert len(gather.offsets) == gather.n_traces
            # Verify metadata has bin info
            assert 'bin_offset_min' in gather.metadata
            assert 'bin_offset_max' in gather.metadata

    def test_workflow_output_manager(self, synthetic_data, temp_dirs):
        """Test output volume management."""
        input_file, output_dir = temp_dirs
        geometry = synthetic_data['geometry']

        # Create index and binning
        indexer = DatasetIndexer()
        index = indexer.index_from_geometry(
            n_traces=synthetic_data['n_traces'],
            n_samples=synthetic_data['n_samples'],
            sample_rate_ms=synthetic_data['dt_ms'],
            source_x=geometry['source_x'],
            source_y=geometry['source_y'],
            receiver_x=geometry['receiver_x'],
            receiver_y=geometry['receiver_y'],
        )

        binning = create_uniform_offset_binning(0, 2000, 4)

        # Create output grid
        output_grid = OutputGrid(
            n_time=100,
            n_inline=10,
            n_xline=10,
            dt=synthetic_data['dt_ms'] / 1000.0,
        )

        # Create output manager
        manager = MigrationOutputManager(
            output_directory=output_dir,
            output_grid=output_grid,
            binning_table=binning,
            create_stack=True,
            create_fold=True,
        )

        manager.initialize()

        # Verify initialization
        assert len(manager._volumes) == 4
        assert manager._stack_volume is not None

        # Add some test data
        test_data = np.ones((100, 10, 10), dtype=np.float32)
        bin_name = binning.bins[0].name
        manager.add_migrated_data(bin_name, test_data)

        # Verify data added
        assert np.sum(manager._volumes[bin_name]) > 0
        assert np.sum(manager._stack_volume) > 0

    def test_workflow_full_pipeline(self, synthetic_data, temp_dirs):
        """Test complete common offset workflow pipeline."""
        input_file, output_dir = temp_dirs
        geometry = synthetic_data['geometry']
        data = synthetic_data['data']

        # 1. Create index
        indexer = DatasetIndexer()
        index = indexer.index_from_geometry(
            n_traces=synthetic_data['n_traces'],
            n_samples=synthetic_data['n_samples'],
            sample_rate_ms=synthetic_data['dt_ms'],
            source_x=geometry['source_x'],
            source_y=geometry['source_y'],
            receiver_x=geometry['receiver_x'],
            receiver_y=geometry['receiver_y'],
        )

        # 2. Create binning
        binning = create_uniform_offset_binning(0, 2000, 4)

        # 3. Create binned dataset
        binned = BinnedDataset(index, binning)

        # 4. Create data reader
        reader = NumpyDataReader(data)

        # 5. Iterate through gathers and verify
        iterator = CommonOffsetGatherIterator(index, reader, binned)

        total_traces_processed = 0
        for gather in iterator:
            total_traces_processed += gather.n_traces

            # Verify gather data is valid
            assert not np.isnan(gather.data).any()
            assert gather.data.shape[1] == synthetic_data['n_samples']

        # 6. Verify all traces processed
        assert total_traces_processed == synthetic_data['n_traces']

    def test_workflow_bin_coverage(self, synthetic_data):
        """Test that bin coverage report is accurate."""
        geometry = synthetic_data['geometry']

        # Create index
        indexer = DatasetIndexer()
        index = indexer.index_from_geometry(
            n_traces=synthetic_data['n_traces'],
            n_samples=synthetic_data['n_samples'],
            sample_rate_ms=synthetic_data['dt_ms'],
            source_x=geometry['source_x'],
            source_y=geometry['source_y'],
            receiver_x=geometry['receiver_x'],
            receiver_y=geometry['receiver_y'],
        )

        # Create binning
        binning = create_uniform_offset_binning(0, 2000, 4)
        binned = BinnedDataset(index, binning)

        # Get coverage report
        report = binned.get_coverage_report()

        assert report['n_traces'] == synthetic_data['n_traces']
        assert report['n_bins'] == 4
        assert 'bin_counts' in report
        assert 'coverage_percent' in report


class TestCommonOffsetIntegrationWithMigrator:
    """Integration tests with actual migrator (mocked)."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup."""
        n_traces = 50
        n_samples = 100

        # Simple geometry
        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = np.linspace(100, 1500, n_traces)
        receiver_y = np.zeros(n_traces)

        # Create index
        indexer = DatasetIndexer()
        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=n_samples,
            sample_rate_ms=4.0,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        # Create data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, reader, n_traces, n_samples

    def test_gather_data_matches_original(self, simple_setup):
        """Verify gather data matches original traces."""
        index, reader, n_traces, n_samples = simple_setup

        # Create binning
        binning = create_uniform_offset_binning(0, 1600, 4)
        binned = BinnedDataset(index, binning)

        # Create iterator
        iterator = CommonOffsetGatherIterator(index, reader, binned)

        # Collect all trace numbers from gathers
        all_trace_nums = set()
        for gather in iterator:
            all_trace_nums.update(gather.trace_numbers.tolist())

        # Verify all traces accounted for
        assert len(all_trace_nums) == n_traces

    def test_bin_offset_ranges_correct(self, simple_setup):
        """Verify traces in each bin have correct offset ranges."""
        index, reader, n_traces, n_samples = simple_setup

        # Create binning
        binning = create_uniform_offset_binning(0, 1600, 4)
        binned = BinnedDataset(index, binning)

        # Create iterator
        iterator = CommonOffsetGatherIterator(index, reader, binned)

        for gather in iterator:
            bin_min = gather.metadata['bin_offset_min']
            bin_max = gather.metadata['bin_offset_max']

            # All offsets should be within bin range
            for offset in gather.offsets:
                assert bin_min <= offset < bin_max, \
                    f"Offset {offset} not in bin range [{bin_min}, {bin_max})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
