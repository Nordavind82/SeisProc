"""
Unit tests for dataset indexer.

Tests:
- DatasetIndex creation and operations
- Index from geometry arrays
- Bin assignments
- Index persistence
- Validation
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_indexer import (
    TraceIndexEntry,
    DatasetIndex,
    DatasetIndexer,
    BinnedDataset,
    validate_index,
)
from models.binning import (
    BinningTable,
    OffsetAzimuthBin,
    create_uniform_offset_binning,
)


class TestTraceIndexEntry:
    """Tests for TraceIndexEntry."""

    def test_basic_creation(self):
        """Test basic entry creation."""
        entry = TraceIndexEntry(
            trace_number=0,
            file_position=0,
            offset=500.0,
            azimuth=45.0,
        )

        assert entry.trace_number == 0
        assert entry.offset == 500.0
        assert entry.azimuth == 45.0
        assert entry.inline is None
        assert entry.source_x is None

    def test_full_entry(self):
        """Test entry with all fields."""
        entry = TraceIndexEntry(
            trace_number=100,
            file_position=12345,
            offset=1000.0,
            azimuth=90.0,
            inline=200,
            xline=300,
            cdp_x=5000.0,
            cdp_y=6000.0,
            source_x=4500.0,
            source_y=5500.0,
            receiver_x=5500.0,
            receiver_y=6500.0,
        )

        assert entry.inline == 200
        assert entry.cdp_x == 5000.0
        assert entry.source_x == 4500.0


class TestDatasetIndex:
    """Tests for DatasetIndex."""

    @pytest.fixture
    def sample_index(self):
        """Create a sample index for testing."""
        entries = [
            TraceIndexEntry(
                trace_number=i,
                file_position=i * 1000,
                offset=100.0 + i * 100,  # 100, 200, 300, ...
                azimuth=(i * 30) % 360,  # 0, 30, 60, ...
                inline=i // 10,
                xline=i % 10,
            )
            for i in range(100)
        ]

        return DatasetIndex(
            filepath="/test/data.sgy",
            n_traces=100,
            n_samples=500,
            sample_rate_ms=4.0,
            entries=entries,
        )

    def test_basic_properties(self, sample_index):
        """Test basic index properties."""
        assert sample_index.n_traces == 100
        assert sample_index.n_samples == 500
        assert sample_index.sample_rate_ms == 4.0
        assert len(sample_index.entries) == 100

    def test_offsets_array(self, sample_index):
        """Test offsets array extraction."""
        offsets = sample_index.offsets
        assert len(offsets) == 100
        assert offsets[0] == 100.0
        assert offsets[9] == 1000.0

    def test_azimuths_array(self, sample_index):
        """Test azimuths array extraction."""
        azimuths = sample_index.azimuths
        assert len(azimuths) == 100
        assert azimuths[0] == 0.0
        assert azimuths[1] == 30.0

    def test_get_traces_in_offset_range(self, sample_index):
        """Test offset range query."""
        traces = sample_index.get_traces_in_offset_range(200.0, 500.0)

        # Traces with offset 200, 300, 400 (indices 1, 2, 3)
        assert len(traces) == 3
        assert 1 in traces
        assert 2 in traces
        assert 3 in traces

    def test_get_traces_for_bin(self, sample_index):
        """Test bin query."""
        traces = sample_index.get_traces_for_bin(
            offset_min=100.0,
            offset_max=500.0,
            azimuth_min=0.0,
            azimuth_max=60.0,
        )

        # Should include traces with offset 100-499 and azimuth 0-59
        assert len(traces) > 0

    def test_compute_statistics(self, sample_index):
        """Test statistics computation."""
        stats = sample_index.compute_statistics()

        assert stats['n_traces'] == 100
        assert stats['offset_min'] == 100.0
        assert stats['offset_max'] == 10000.0
        assert 'offset_mean' in stats
        assert 'has_inline_xline' in stats

    def test_serialization(self, sample_index):
        """Test to_dict and from_dict."""
        d = sample_index.to_dict()
        restored = DatasetIndex.from_dict(d)

        assert restored.n_traces == sample_index.n_traces
        assert restored.n_samples == sample_index.n_samples
        assert len(restored.entries) == len(sample_index.entries)
        assert restored.entries[0].offset == sample_index.entries[0].offset

    def test_save_load(self, sample_index):
        """Test save and load to file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            sample_index.save(filepath)
            restored = DatasetIndex.load(filepath)

            assert restored.n_traces == sample_index.n_traces
            assert len(restored.entries) == len(sample_index.entries)
        finally:
            Path(filepath).unlink()


class TestDatasetIndexer:
    """Tests for DatasetIndexer."""

    def test_indexer_creation(self):
        """Test indexer creation."""
        indexer = DatasetIndexer()
        assert indexer.compute_offset is True
        assert indexer.compute_azimuth is True

    def test_indexer_with_mapping(self):
        """Test indexer with header mapping."""
        mapping = {
            'source_x': 'ShotX',
            'source_y': 'ShotY',
        }
        indexer = DatasetIndexer(header_mapping=mapping)
        assert indexer.header_mapping == mapping

    def test_compute_offset_azimuth(self):
        """Test offset/azimuth computation from coordinates."""
        indexer = DatasetIndexer()

        # Simple case: receiver 1000m east of source
        offset, azimuth = indexer._compute_offset_azimuth(
            sx=0.0, sy=0.0,
            gx=1000.0, gy=0.0,
        )

        assert offset == pytest.approx(1000.0, abs=0.1)
        assert azimuth == pytest.approx(90.0, abs=0.1)  # East

    def test_compute_offset_azimuth_north(self):
        """Test azimuth for north direction."""
        indexer = DatasetIndexer()

        offset, azimuth = indexer._compute_offset_azimuth(
            sx=0.0, sy=0.0,
            gx=0.0, gy=1000.0,
        )

        assert azimuth == pytest.approx(0.0, abs=0.1)  # North

    def test_compute_offset_azimuth_southwest(self):
        """Test azimuth for southwest direction."""
        indexer = DatasetIndexer()

        offset, azimuth = indexer._compute_offset_azimuth(
            sx=0.0, sy=0.0,
            gx=-1000.0, gy=-1000.0,
        )

        assert azimuth == pytest.approx(225.0, abs=0.1)  # Southwest

    def test_index_from_geometry(self):
        """Test indexing from geometry arrays."""
        indexer = DatasetIndexer()

        n_traces = 50
        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = np.arange(n_traces) * 25.0  # 0, 25, 50, ...
        receiver_y = np.zeros(n_traces)

        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=200,
            sample_rate_ms=4.0,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        assert index.n_traces == n_traces
        assert index.entries[0].offset == 0.0
        assert index.entries[10].offset == pytest.approx(250.0, abs=0.1)

        # Check statistics were computed
        assert 'offset_min' in index.statistics

    def test_index_from_geometry_with_inlines(self):
        """Test indexing with inline/xline."""
        indexer = DatasetIndexer()

        n_traces = 25
        inlines = np.repeat(np.arange(5), 5)
        xlines = np.tile(np.arange(5), 5)

        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=100,
            sample_rate_ms=2.0,
            source_x=np.zeros(n_traces),
            source_y=np.zeros(n_traces),
            receiver_x=np.ones(n_traces) * 500,
            receiver_y=np.zeros(n_traces),
            inlines=inlines,
            xlines=xlines,
        )

        assert index.entries[0].inline == 0
        assert index.entries[0].xline == 0
        assert index.entries[6].inline == 1
        assert index.entries[6].xline == 1


class TestBinnedDataset:
    """Tests for BinnedDataset."""

    @pytest.fixture
    def sample_binned_dataset(self):
        """Create sample binned dataset."""
        # Create index with variety of offsets
        indexer = DatasetIndexer()

        n_traces = 100
        offsets = np.random.uniform(0, 2000, n_traces)
        angles = np.linspace(0, 2 * np.pi, n_traces)

        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = offsets * np.sin(angles)
        receiver_y = offsets * np.cos(angles)

        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=100,
            sample_rate_ms=4.0,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        # Create binning table
        binning = create_uniform_offset_binning(
            offset_min=0.0,
            offset_max=2000.0,
            n_bins=4,
        )

        return BinnedDataset(index, binning)

    def test_bin_trace_counts(self, sample_binned_dataset):
        """Test bin trace count retrieval."""
        summary = sample_binned_dataset.get_bin_summary()

        assert len(summary) == 4
        assert sum(summary.values()) > 0

    def test_get_bin_trace_numbers(self, sample_binned_dataset):
        """Test getting trace numbers for a bin."""
        bin_name = list(sample_binned_dataset._bin_assignments.keys())[0]
        traces = sample_binned_dataset.get_bin_trace_numbers(bin_name)

        assert len(traces) > 0
        assert all(0 <= t < 100 for t in traces)

    def test_iterate_bins(self, sample_binned_dataset):
        """Test bin iteration."""
        bins_found = []
        for bin_name, traces in sample_binned_dataset.iterate_bins():
            bins_found.append(bin_name)
            assert len(traces) > 0

        assert len(bins_found) > 0

    def test_coverage_report(self, sample_binned_dataset):
        """Test coverage report generation."""
        report = sample_binned_dataset.get_coverage_report()

        assert report['n_traces'] == 100
        assert report['n_bins'] == 4
        assert 'bin_counts' in report
        assert 'coverage_percent' in report


class TestValidation:
    """Tests for index validation."""

    def test_validate_good_index(self):
        """Test validation of good index."""
        entries = [
            TraceIndexEntry(
                trace_number=i,
                file_position=0,
                offset=500.0 + i * 100,
                azimuth=45.0,
                source_x=0.0,
                source_y=0.0,
                receiver_x=500.0,
                receiver_y=0.0,
            )
            for i in range(10)
        ]

        index = DatasetIndex(
            filepath="/test.sgy",
            n_traces=10,
            n_samples=100,
            sample_rate_ms=4.0,
            entries=entries,
        )

        warnings = validate_index(index)
        assert len(warnings) == 0

    def test_validate_missing_coordinates(self):
        """Test validation catches missing coordinates."""
        entries = [
            TraceIndexEntry(
                trace_number=i,
                file_position=0,
                offset=500.0,
                azimuth=45.0,
                # No source/receiver coordinates
            )
            for i in range(10)
        ]

        index = DatasetIndex(
            filepath="/test.sgy",
            n_traces=10,
            n_samples=100,
            sample_rate_ms=4.0,
            entries=entries,
        )

        warnings = validate_index(index)
        assert any('source' in w.lower() for w in warnings)

    def test_validate_zero_offset(self):
        """Test validation catches many zero offsets."""
        entries = [
            TraceIndexEntry(
                trace_number=i,
                file_position=0,
                offset=0.0,  # All zero
                azimuth=45.0,
            )
            for i in range(10)
        ]

        index = DatasetIndex(
            filepath="/test.sgy",
            n_traces=10,
            n_samples=100,
            sample_rate_ms=4.0,
            entries=entries,
        )

        warnings = validate_index(index)
        assert any('zero offset' in w.lower() for w in warnings)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
