"""
Tests for parallel processing infrastructure.

Tests the processor serialization, gather partitioning, and parallel
processing coordinator functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import zarr
import json

from processors.bandpass_filter import BandpassFilter
from processors.gain_processor import GainProcessor
from processors.base_processor import BaseProcessor
from utils.parallel_processing import (
    ProcessingConfig,
    GatherSegment,
    ProcessingTask,
    GatherPartitioner,
    ParallelProcessingCoordinator,
    SortOptions
)


class TestProcessorSerialization:
    """Test processor serialization for multiprocess transfer."""

    def test_bandpass_filter_roundtrip(self):
        """Test BandpassFilter serialization roundtrip."""
        original = BandpassFilter(
            low_freq=5.0,
            high_freq=60.0,
            order=4
        )

        # Serialize
        config = original.to_dict()

        # Verify structure
        assert 'class_name' in config
        assert 'module' in config
        assert 'params' in config
        assert config['class_name'] == 'BandpassFilter'
        assert config['params']['low_freq'] == 5.0
        assert config['params']['high_freq'] == 60.0

        # Reconstruct
        reconstructed = BaseProcessor.from_dict(config)

        # Verify type and params
        assert isinstance(reconstructed, BandpassFilter)
        assert reconstructed.params['low_freq'] == original.params['low_freq']
        assert reconstructed.params['high_freq'] == original.params['high_freq']

    def test_gain_processor_roundtrip(self):
        """Test GainProcessor serialization roundtrip."""
        original = GainProcessor(
            gain_type='agc',
            window_ms=500.0
        )

        config = original.to_dict()
        reconstructed = BaseProcessor.from_dict(config)

        assert isinstance(reconstructed, GainProcessor)
        assert reconstructed.params['gain_type'] == 'agc'
        assert reconstructed.params['window_ms'] == 500.0


class TestGatherPartitioner:
    """Test gather partitioning for worker distribution."""

    def create_test_ensemble_df(self, n_gathers, traces_per_gather=100):
        """Create test ensemble DataFrame."""
        data = []
        trace_offset = 0

        for i in range(n_gathers):
            n_traces = traces_per_gather
            data.append({
                'ensemble_id': i,
                'start_trace': trace_offset,
                'end_trace': trace_offset + n_traces - 1,
                'n_traces': n_traces
            })
            trace_offset += n_traces

        return pd.DataFrame(data)

    def test_partition_simple(self):
        """Test basic partitioning with equal gather sizes."""
        ensemble_df = self.create_test_ensemble_df(10, traces_per_gather=100)
        partitioner = GatherPartitioner(ensemble_df, n_segments=2)

        segments = partitioner.partition()

        # Should have 2 segments
        assert len(segments) == 2

        # Together should cover all gathers
        total_gathers = sum(s.n_gathers for s in segments)
        assert total_gathers == 10

        # Together should cover all traces
        total_traces = sum(s.n_traces for s in segments)
        assert total_traces == 1000

    def test_partition_more_workers_than_gathers(self):
        """Test partitioning when workers > gathers."""
        ensemble_df = self.create_test_ensemble_df(3, traces_per_gather=100)
        partitioner = GatherPartitioner(ensemble_df, n_segments=10)

        segments = partitioner.partition()

        # Should have one segment per gather
        assert len(segments) == 3

    def test_partition_stats(self):
        """Test partition statistics."""
        ensemble_df = self.create_test_ensemble_df(20, traces_per_gather=50)
        partitioner = GatherPartitioner(ensemble_df, n_segments=4)

        segments = partitioner.partition()
        stats = partitioner.get_partition_stats(segments)

        assert stats['n_segments'] == 4
        assert stats['total_gathers'] == 20
        assert stats['total_traces'] == 1000

    def test_partition_uneven_gathers(self):
        """Test partitioning with variable gather sizes."""
        # Create gathers with varying sizes
        data = [
            {'ensemble_id': 0, 'start_trace': 0, 'end_trace': 99, 'n_traces': 100},
            {'ensemble_id': 1, 'start_trace': 100, 'end_trace': 599, 'n_traces': 500},  # Large
            {'ensemble_id': 2, 'start_trace': 600, 'end_trace': 649, 'n_traces': 50},
            {'ensemble_id': 3, 'start_trace': 650, 'end_trace': 749, 'n_traces': 100},
            {'ensemble_id': 4, 'start_trace': 750, 'end_trace': 999, 'n_traces': 250},
        ]
        ensemble_df = pd.DataFrame(data)
        partitioner = GatherPartitioner(ensemble_df, n_segments=2)

        segments = partitioner.partition()

        # Should balance by trace count, not gather count
        assert len(segments) == 2
        total_traces = sum(s.n_traces for s in segments)
        assert total_traces == 1000


class TestProcessingCoordinator:
    """Test the parallel processing coordinator."""

    @pytest.fixture
    def test_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def create_test_zarr_storage(self, storage_dir, n_traces=100, n_samples=500, n_gathers=10):
        """Create test Zarr storage with metadata."""
        traces_per_gather = n_traces // n_gathers

        # Create traces.zarr
        traces_path = storage_dir / 'traces.zarr'
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        zarr.save(str(traces_path), traces)

        # Create metadata.json
        metadata = {
            'n_traces': n_traces,
            'n_samples': n_samples,
            'sample_rate': 2.0,
            'file_format': 'zarr'
        }
        with open(storage_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Create ensemble_index.parquet
        ensemble_data = []
        trace_offset = 0
        for i in range(n_gathers):
            ensemble_data.append({
                'ensemble_id': i,
                'start_trace': trace_offset,
                'end_trace': trace_offset + traces_per_gather - 1,
                'n_traces': traces_per_gather
            })
            trace_offset += traces_per_gather

        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_df.to_parquet(storage_dir / 'ensemble_index.parquet')

        # Create headers.parquet (minimal)
        headers_data = {
            'TRACE_SEQUENCE_LINE': np.arange(n_traces),
            'CDP': np.repeat(np.arange(n_gathers), traces_per_gather)
        }
        headers_df = pd.DataFrame(headers_data)
        headers_df.to_parquet(storage_dir / 'headers.parquet')

        return traces

    def test_coordinator_initialization(self, test_storage_dir):
        """Test coordinator can be initialized."""
        self.create_test_zarr_storage(test_storage_dir)

        processor = BandpassFilter(low_freq=5.0, high_freq=60.0)
        output_dir = test_storage_dir / 'output'

        config = ProcessingConfig(
            input_storage_dir=str(test_storage_dir),
            output_storage_dir=str(output_dir),
            processor_config=processor.to_dict(),
            n_workers=2
        )

        coordinator = ParallelProcessingCoordinator(config)
        assert coordinator.n_workers == 2

    def test_coordinator_run_small_dataset(self, test_storage_dir):
        """Test processing a small dataset."""
        n_traces = 100
        n_samples = 200
        n_gathers = 5

        original_traces = self.create_test_zarr_storage(
            test_storage_dir,
            n_traces=n_traces,
            n_samples=n_samples,
            n_gathers=n_gathers
        )

        processor = GainProcessor(gain_type='normalize')
        output_dir = test_storage_dir / 'output'

        config = ProcessingConfig(
            input_storage_dir=str(test_storage_dir),
            output_storage_dir=str(output_dir),
            processor_config=processor.to_dict(),
            n_workers=2
        )

        coordinator = ParallelProcessingCoordinator(config)
        result = coordinator.run()

        # Check result
        assert result.success, f"Processing failed: {result.error}"
        assert result.n_traces == n_traces
        assert result.n_gathers == n_gathers
        assert result.n_samples == n_samples
        assert result.throughput_traces_per_sec > 0

        # Verify output exists
        output_zarr_path = Path(result.output_zarr_path)
        assert output_zarr_path.exists()

        # Verify output shape
        output_zarr = zarr.open(str(output_zarr_path), mode='r')
        assert output_zarr.shape == (n_samples, n_traces)


class TestSortOptions:
    """Test in-gather sorting configuration."""

    def test_sort_options_creation(self):
        """Test SortOptions dataclass creation."""
        options = SortOptions(
            enabled=True,
            sort_key='offset',
            ascending=True
        )

        assert options.enabled is True
        assert options.sort_key == 'offset'
        assert options.ascending is True

    def test_sort_options_with_secondary_key(self):
        """Test SortOptions with secondary sort key."""
        options = SortOptions(
            enabled=True,
            sort_key='CDP',
            ascending=True,
            secondary_key='offset',
            secondary_ascending=False
        )

        assert options.secondary_key == 'offset'
        assert options.secondary_ascending is False


class TestProcessingWithSorting:
    """Test processing with in-gather sorting enabled."""

    @pytest.fixture
    def test_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def create_test_zarr_storage_with_offsets(self, storage_dir, n_traces=100, n_samples=200, n_gathers=5):
        """Create test Zarr storage with offset headers for sorting tests."""
        traces_per_gather = n_traces // n_gathers

        # Create traces.zarr
        traces_path = storage_dir / 'traces.zarr'
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        zarr.save(str(traces_path), traces)

        # Create metadata.json
        metadata = {
            'n_traces': n_traces,
            'n_samples': n_samples,
            'sample_rate': 2.0,
            'file_format': 'zarr'
        }
        with open(storage_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Create ensemble_index.parquet
        ensemble_data = []
        trace_offset = 0
        for i in range(n_gathers):
            ensemble_data.append({
                'ensemble_id': i,
                'start_trace': trace_offset,
                'end_trace': trace_offset + traces_per_gather - 1,
                'n_traces': traces_per_gather
            })
            trace_offset += traces_per_gather

        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_df.to_parquet(storage_dir / 'ensemble_index.parquet')

        # Create headers.parquet with UNSORTED offsets within each gather
        headers_data = {
            'TRACE_SEQUENCE_LINE': np.arange(n_traces),
            'CDP': np.repeat(np.arange(n_gathers), traces_per_gather),
            'offset': []
        }

        # Create random offsets within each gather (not sorted)
        for g in range(n_gathers):
            # Generate random offsets for this gather
            offsets = np.random.permutation(traces_per_gather) * 100  # Random order
            headers_data['offset'].extend(offsets.tolist())

        headers_data['offset'] = np.array(headers_data['offset'])
        headers_df = pd.DataFrame(headers_data)
        headers_df.to_parquet(storage_dir / 'headers.parquet')

        return traces, headers_df

    def test_processing_with_sorting(self, test_storage_dir):
        """Test processing with in-gather sorting enabled."""
        n_traces = 100
        n_samples = 200
        n_gathers = 5
        traces_per_gather = n_traces // n_gathers

        original_traces, original_headers = self.create_test_zarr_storage_with_offsets(
            test_storage_dir,
            n_traces=n_traces,
            n_samples=n_samples,
            n_gathers=n_gathers
        )

        processor = GainProcessor(gain_type='normalize')
        output_dir = test_storage_dir / 'output'

        # Enable sorting by offset
        sort_options = SortOptions(
            enabled=True,
            sort_key='offset',
            ascending=True
        )

        config = ProcessingConfig(
            input_storage_dir=str(test_storage_dir),
            output_storage_dir=str(output_dir),
            processor_config=processor.to_dict(),
            n_workers=2,
            sort_options=sort_options
        )

        coordinator = ParallelProcessingCoordinator(config)
        result = coordinator.run()

        # Check result
        assert result.success, f"Processing failed: {result.error}"
        assert result.n_traces == n_traces

        # Verify output headers are sorted
        output_headers = pd.read_parquet(output_dir / 'headers.parquet')

        # Check that within each gather, offsets are sorted ascending
        for g in range(n_gathers):
            start = g * traces_per_gather
            end = start + traces_per_gather
            gather_offsets = output_headers.iloc[start:end]['offset'].values

            # Verify ascending order
            assert np.all(gather_offsets[:-1] <= gather_offsets[1:]), \
                f"Gather {g} offsets not sorted: {gather_offsets}"

        # Verify original_trace_index column exists
        assert 'original_trace_index' in output_headers.columns

    def test_processing_without_sorting(self, test_storage_dir):
        """Test processing without sorting preserves original order."""
        n_traces = 100
        n_samples = 200
        n_gathers = 5

        original_traces, original_headers = self.create_test_zarr_storage_with_offsets(
            test_storage_dir,
            n_traces=n_traces,
            n_samples=n_samples,
            n_gathers=n_gathers
        )

        processor = GainProcessor(gain_type='normalize')
        output_dir = test_storage_dir / 'output'

        # No sorting
        config = ProcessingConfig(
            input_storage_dir=str(test_storage_dir),
            output_storage_dir=str(output_dir),
            processor_config=processor.to_dict(),
            n_workers=2
        )

        coordinator = ParallelProcessingCoordinator(config)
        result = coordinator.run()

        assert result.success, f"Processing failed: {result.error}"

        # Verify output headers match original order
        output_headers = pd.read_parquet(output_dir / 'headers.parquet')

        # Offsets should be in same order as original
        np.testing.assert_array_equal(
            output_headers['offset'].values,
            original_headers['offset'].values
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
