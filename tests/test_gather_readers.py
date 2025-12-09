"""
Unit tests for gather readers.

Tests:
- Sort order detection
- Common offset gather iteration
- Common shot gather iteration
- OVT gather iteration
- Streaming gather reader
"""

import numpy as np
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.sort_detector import (
    SortOrder,
    SortAnalysis,
    detect_sort_order,
    get_gather_boundaries,
)
from utils.dataset_indexer import (
    TraceIndexEntry,
    DatasetIndex,
    DatasetIndexer,
    BinnedDataset,
)
from models.binning import (
    create_uniform_offset_binning,
    create_ovt_binning,
)
from seisio.gather_readers import (
    Gather,
    GatherIterator,
    DataReader,
    NumpyDataReader,
    CommonOffsetGatherIterator,
    CommonShotGatherIterator,
    OVTGatherIterator,
    StreamingGatherReader,
    create_gather_iterator,
)


class TestSortDetector:
    """Tests for sort order detection."""

    def test_detect_common_shot_sort(self):
        """Test detection of common shot sorted data."""
        # Create data sorted by shot (each shot has 10 traces)
        n_shots = 5
        traces_per_shot = 10
        shot_ids = np.repeat(np.arange(n_shots), traces_per_shot)

        result = detect_sort_order(shot_ids)

        assert result.detected_order == SortOrder.COMMON_SHOT
        assert result.confidence > 0.8
        assert result.unique_primary_keys == n_shots

    def test_detect_common_offset_sort(self):
        """Test detection of common offset sorted data."""
        # Create data sorted by offset
        n_offsets = 4
        traces_per_offset = 20
        shot_ids = np.tile(np.arange(traces_per_offset), n_offsets)
        offsets = np.repeat(np.arange(n_offsets) * 500.0, traces_per_offset)

        result = detect_sort_order(shot_ids, offsets=offsets)

        assert result.detected_order == SortOrder.COMMON_OFFSET
        assert result.confidence > 0.5

    def test_detect_unsorted_data(self):
        """Test detection of unsorted/random data."""
        # Randomly shuffled shot IDs
        n_traces = 100
        shot_ids = np.random.randint(0, 10, size=n_traces)

        result = detect_sort_order(shot_ids)

        # Should still detect something but with lower confidence
        assert result.confidence < 0.8

    def test_single_gather(self):
        """Test with data that's all one gather."""
        shot_ids = np.ones(50, dtype=np.int32)

        result = detect_sort_order(shot_ids)

        assert result.detected_order == SortOrder.COMMON_SHOT
        assert result.unique_primary_keys == 1

    def test_too_few_traces(self):
        """Test with too few traces."""
        shot_ids = np.array([1])

        result = detect_sort_order(shot_ids)

        assert result.detected_order == SortOrder.UNKNOWN


class TestGetGatherBoundaries:
    """Tests for gather boundary detection."""

    def test_basic_boundaries(self):
        """Test basic gather boundary detection."""
        keys = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
        boundaries = get_gather_boundaries(keys)

        assert len(boundaries) == 3
        assert boundaries[0] == (0, 3)
        assert boundaries[1] == (3, 5)
        assert boundaries[2] == (5, 9)

    def test_single_gather(self):
        """Test single gather."""
        keys = np.array([5, 5, 5, 5])
        boundaries = get_gather_boundaries(keys)

        assert len(boundaries) == 1
        assert boundaries[0] == (0, 4)

    def test_empty_array(self):
        """Test empty array."""
        keys = np.array([])
        boundaries = get_gather_boundaries(keys)

        assert len(boundaries) == 0


class TestNumpyDataReader:
    """Tests for NumpyDataReader."""

    def test_read_traces(self):
        """Test reading traces from numpy array."""
        data = np.random.randn(100, 500).astype(np.float32)
        reader = NumpyDataReader(data)

        # Read specific traces
        trace_nums = np.array([0, 10, 50, 99])
        result = reader.read_traces(trace_nums)

        assert result.shape == (4, 500)
        np.testing.assert_array_equal(result[0], data[0])
        np.testing.assert_array_equal(result[2], data[50])

    def test_get_n_samples(self):
        """Test getting sample count."""
        data = np.random.randn(50, 250).astype(np.float32)
        reader = NumpyDataReader(data)

        assert reader.get_n_samples() == 250


class TestGather:
    """Tests for Gather dataclass."""

    def test_gather_properties(self):
        """Test gather properties."""
        data = np.random.randn(20, 100).astype(np.float32)
        gather = Gather(
            gather_id="test_gather",
            trace_numbers=np.arange(20),
            data=data,
            offsets=np.arange(20) * 50.0,
            azimuths=np.ones(20) * 45.0,
        )

        assert gather.n_traces == 20
        assert gather.n_samples == 100
        assert gather.gather_id == "test_gather"


class TestCommonOffsetGatherIterator:
    """Tests for CommonOffsetGatherIterator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_traces = 100
        n_samples = 200

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

        # Create binning
        binning = create_uniform_offset_binning(
            offset_min=0.0,
            offset_max=2000.0,
            n_bins=4,
        )

        # Create binned dataset
        binned = BinnedDataset(index, binning)

        # Create data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, binned, reader, data

    def test_iteration(self, sample_data):
        """Test iterating over offset gathers."""
        index, binned, reader, _ = sample_data

        iterator = CommonOffsetGatherIterator(index, reader, binned)

        gathers = list(iterator)
        assert len(gathers) == 4  # 4 offset bins

        # Check each gather has data
        for gather in gathers:
            assert gather.n_traces > 0
            assert gather.n_samples == 200

    def test_get_gather(self, sample_data):
        """Test getting specific gather."""
        index, binned, reader, _ = sample_data

        iterator = CommonOffsetGatherIterator(index, reader, binned)

        gather_ids = iterator.get_gather_ids()
        gather = iterator.get_gather(gather_ids[0])

        assert gather.gather_id == gather_ids[0]
        assert gather.n_traces > 0

    def test_gather_metadata(self, sample_data):
        """Test gather metadata."""
        index, binned, reader, _ = sample_data

        iterator = CommonOffsetGatherIterator(index, reader, binned)
        gather = next(iter(iterator))

        assert 'bin_offset_min' in gather.metadata
        assert 'bin_offset_max' in gather.metadata
        assert 'bin_offset_center' in gather.metadata


class TestCommonShotGatherIterator:
    """Tests for CommonShotGatherIterator."""

    @pytest.fixture
    def shot_data(self):
        """Create shot-sorted data for testing."""
        n_shots = 5
        traces_per_shot = 20
        n_traces = n_shots * traces_per_shot
        n_samples = 100

        # Create geometry for each shot
        source_x = np.repeat(np.arange(n_shots) * 100.0, traces_per_shot)
        source_y = np.zeros(n_traces)
        receiver_x = source_x + np.tile(np.arange(traces_per_shot) * 25.0, n_shots)
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

        # Create shot IDs
        shot_ids = np.repeat(np.arange(n_shots), traces_per_shot)

        # Create data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, reader, shot_ids, n_shots

    def test_iteration(self, shot_data):
        """Test iterating over shot gathers."""
        index, reader, shot_ids, n_shots = shot_data

        iterator = CommonShotGatherIterator(index, reader, shot_ids)

        gathers = list(iterator)
        assert len(gathers) == n_shots

        for gather in gathers:
            assert gather.n_traces == 20
            assert gather.n_samples == 100

    def test_get_gather_ids(self, shot_data):
        """Test getting gather IDs."""
        index, reader, shot_ids, n_shots = shot_data

        iterator = CommonShotGatherIterator(index, reader, shot_ids)
        gather_ids = iterator.get_gather_ids()

        assert len(gather_ids) == n_shots
        assert all(id.startswith("shot_") for id in gather_ids)


class TestOVTGatherIterator:
    """Tests for OVTGatherIterator."""

    @pytest.fixture
    def ovt_data(self):
        """Create data with OVT distribution."""
        n_traces = 200
        n_samples = 100

        # Create geometry with varied offsets and azimuths
        offsets = np.random.uniform(0, 3000, n_traces)
        angles = np.random.uniform(0, 2 * np.pi, n_traces)

        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = offsets * np.sin(angles)
        receiver_y = offsets * np.cos(angles)

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

        # Create OVT binning (3 offset x 4 azimuth = 12 bins)
        offset_ranges = [(0, 1000), (1000, 2000), (2000, 3000)]
        binning = create_ovt_binning(offset_ranges, n_azimuth_sectors=4)

        # Create binned dataset
        binned = BinnedDataset(index, binning)

        # Create data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, binned, reader

    def test_iteration(self, ovt_data):
        """Test iterating over OVT gathers."""
        index, binned, reader = ovt_data

        iterator = OVTGatherIterator(index, reader, binned)

        gathers = list(iterator)
        # Should have up to 12 bins, but may have fewer if some bins empty
        assert len(gathers) > 0
        assert len(gathers) <= 12

    def test_gather_metadata(self, ovt_data):
        """Test OVT gather metadata."""
        index, binned, reader = ovt_data

        iterator = OVTGatherIterator(index, reader, binned)
        gather = next(iter(iterator))

        assert 'bin_offset_center' in gather.metadata
        assert 'bin_azimuth_center' in gather.metadata


class TestStreamingGatherReader:
    """Tests for StreamingGatherReader."""

    @pytest.fixture
    def streaming_data(self):
        """Create data for streaming reader tests."""
        n_traces = 100
        n_samples = 150

        # Create geometry
        offsets = np.random.uniform(0, 2000, n_traces)
        angles = np.random.uniform(0, 2 * np.pi, n_traces)

        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = offsets * np.sin(angles)
        receiver_y = offsets * np.cos(angles)

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

        # Create binning
        binning = create_uniform_offset_binning(
            offset_min=0.0,
            offset_max=2000.0,
            n_bins=5,
        )
        binned = BinnedDataset(index, binning)

        # Create shot IDs
        shot_ids = np.repeat(np.arange(10), 10)

        # Create data
        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, binned, reader, shot_ids

    def test_iter_offset_gathers(self, streaming_data):
        """Test offset gather iteration."""
        index, binned, reader, _ = streaming_data

        streaming = StreamingGatherReader(index, reader, binned)
        gathers = list(streaming.iter_offset_gathers())

        assert len(gathers) == 5

    def test_iter_shot_gathers(self, streaming_data):
        """Test shot gather iteration."""
        index, binned, reader, shot_ids = streaming_data

        streaming = StreamingGatherReader(index, reader, binned, shot_ids)
        gathers = list(streaming.iter_shot_gathers())

        assert len(gathers) == 10

    def test_get_gather_count(self, streaming_data):
        """Test gather count retrieval."""
        index, binned, reader, shot_ids = streaming_data

        streaming = StreamingGatherReader(index, reader, binned, shot_ids)

        assert streaming.get_gather_count("offset") == 5
        assert streaming.get_gather_count("shot") == 10

    def test_missing_binned_dataset(self, streaming_data):
        """Test error when binned dataset missing."""
        index, _, reader, _ = streaming_data

        streaming = StreamingGatherReader(index, reader)

        with pytest.raises(ValueError, match="binned_dataset required"):
            list(streaming.iter_offset_gathers())


class TestCreateGatherIterator:
    """Tests for factory function."""

    @pytest.fixture
    def factory_data(self):
        """Create data for factory tests."""
        n_traces = 50
        n_samples = 100

        indexer = DatasetIndexer()
        source_x = np.zeros(n_traces)
        source_y = np.zeros(n_traces)
        receiver_x = np.arange(n_traces) * 50.0
        receiver_y = np.zeros(n_traces)

        index = indexer.index_from_geometry(
            n_traces=n_traces,
            n_samples=n_samples,
            sample_rate_ms=4.0,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        binning = create_uniform_offset_binning(0.0, 2500.0, 5)
        binned = BinnedDataset(index, binning)

        data = np.random.randn(n_traces, n_samples).astype(np.float32)
        reader = NumpyDataReader(data)

        return index, binned, reader

    def test_create_offset_iterator(self, factory_data):
        """Test creating offset iterator."""
        index, binned, reader = factory_data

        iterator = create_gather_iterator(
            index, reader, "offset", binned_dataset=binned
        )

        assert isinstance(iterator, CommonOffsetGatherIterator)

    def test_create_shot_iterator(self, factory_data):
        """Test creating shot iterator."""
        index, _, reader = factory_data

        iterator = create_gather_iterator(index, reader, "shot")

        assert isinstance(iterator, CommonShotGatherIterator)

    def test_create_ovt_iterator(self, factory_data):
        """Test creating OVT iterator."""
        index, binned, reader = factory_data

        iterator = create_gather_iterator(
            index, reader, "ovt", binned_dataset=binned
        )

        assert isinstance(iterator, OVTGatherIterator)

    def test_invalid_gather_type(self, factory_data):
        """Test error for invalid gather type."""
        index, binned, reader = factory_data

        with pytest.raises(ValueError, match="Unknown gather type"):
            create_gather_iterator(index, reader, "invalid")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
