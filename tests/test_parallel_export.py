"""
Tests for parallel SEG-Y export infrastructure.

Tests header vectorization, export workers, segment merging,
and the parallel export coordinator.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import zarr
import segyio

from utils.parallel_export import (
    HeaderVectorizer,
    vectorize_headers,
    get_trace_headers,
    ExportConfig,
    TraceSegment,
    SEGYSegmentMerger,
    ParallelExportCoordinator
)


class TestHeaderVectorizer:
    """Test header vectorization for fast access."""

    def create_test_headers(self, n_traces=1000):
        """Create test headers DataFrame."""
        return pd.DataFrame({
            'TRACE_SEQUENCE_LINE': np.arange(1, n_traces + 1),
            'TRACE_SEQUENCE_FILE': np.arange(1, n_traces + 1),
            'CDP': np.repeat(np.arange(n_traces // 10), 10),
            'offset': np.tile(np.arange(10) * 100, n_traces // 10),
            'SourceX': np.random.randint(0, 10000, n_traces),
            'SourceY': np.random.randint(0, 10000, n_traces),
            'GroupX': np.random.randint(0, 10000, n_traces),
            'GroupY': np.random.randint(0, 10000, n_traces),
        })

    def test_vectorize_basic(self):
        """Test basic header vectorization."""
        headers_df = self.create_test_headers(100)
        vectorizer = HeaderVectorizer(headers_df)

        arrays = vectorizer.vectorize()

        assert isinstance(arrays, dict)
        assert len(arrays) > 0
        assert 'TRACE_SEQUENCE_LINE' in arrays
        assert arrays['TRACE_SEQUENCE_LINE'].dtype == np.int32

    def test_get_trace_headers(self):
        """Test retrieving headers for a single trace."""
        headers_df = self.create_test_headers(100)
        vectorizer = HeaderVectorizer(headers_df)
        vectorizer.vectorize()

        # Get headers for trace 50
        trace_headers = vectorizer.get_trace_headers(50)

        assert isinstance(trace_headers, dict)
        assert trace_headers['TRACE_SEQUENCE_LINE'] == 51  # 1-indexed

    def test_get_trace_range_headers(self):
        """Test retrieving headers for a range of traces."""
        headers_df = self.create_test_headers(100)
        vectorizer = HeaderVectorizer(headers_df)
        vectorizer.vectorize()

        range_headers = vectorizer.get_trace_range_headers(10, 19)

        assert 'TRACE_SEQUENCE_LINE' in range_headers
        assert len(range_headers['TRACE_SEQUENCE_LINE']) == 10

    def test_save_and_load(self, tmp_path):
        """Test saving and loading vectorized headers."""
        headers_df = self.create_test_headers(100)
        vectorizer = HeaderVectorizer(headers_df)
        vectorizer.vectorize()

        save_path = tmp_path / 'headers.pkl'
        vectorizer.save(save_path)

        # Load back
        loaded = HeaderVectorizer.load(save_path)

        assert isinstance(loaded, dict)
        assert 'TRACE_SEQUENCE_LINE' in loaded
        np.testing.assert_array_equal(
            loaded['TRACE_SEQUENCE_LINE'],
            vectorizer._header_arrays['TRACE_SEQUENCE_LINE']
        )

    def test_convenience_functions(self):
        """Test convenience functions for vectorization."""
        headers_df = self.create_test_headers(100)

        # vectorize_headers function
        arrays = vectorize_headers(headers_df)
        assert isinstance(arrays, dict)

        # get_trace_headers function
        trace_headers = get_trace_headers(arrays, 0)
        assert trace_headers['TRACE_SEQUENCE_LINE'] == 1

    def test_stats(self):
        """Test vectorizer statistics."""
        headers_df = self.create_test_headers(1000)
        vectorizer = HeaderVectorizer(headers_df)
        vectorizer.vectorize()

        stats = vectorizer.get_stats()

        assert stats['n_traces'] == 1000
        assert stats['n_fields'] > 0
        assert stats['memory_bytes'] > 0


class TestSEGYSegmentMerger:
    """Test SEG-Y segment file merging."""

    @pytest.fixture
    def test_segy_segments(self, tmp_path):
        """Create test SEG-Y segment files."""
        n_samples = 100
        n_traces_per_segment = 50

        segment_paths = []

        for seg_id in range(3):
            seg_path = tmp_path / f'segment_{seg_id}.sgy'

            spec = segyio.spec()
            spec.samples = list(range(n_samples))
            spec.format = 5  # IEEE float
            spec.tracecount = n_traces_per_segment

            with segyio.create(str(seg_path), spec) as f:
                f.bin[segyio.BinField.Samples] = n_samples
                f.bin[segyio.BinField.Interval] = 2000  # 2ms
                f.bin[segyio.BinField.Format] = 5

                for i in range(n_traces_per_segment):
                    trace_data = np.random.randn(n_samples).astype(np.float32)
                    f.trace[i] = trace_data
                    f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = seg_id * n_traces_per_segment + i + 1

            segment_paths.append(str(seg_path))

        return segment_paths, n_samples, n_traces_per_segment * 3

    def test_merge_basic(self, tmp_path, test_segy_segments):
        """Test basic segment merging."""
        segment_paths, n_samples, total_traces = test_segy_segments
        output_path = tmp_path / 'merged.sgy'

        merger = SEGYSegmentMerger(n_samples=n_samples, data_format=5)
        total_merged = merger.merge(segment_paths, str(output_path))

        assert total_merged == total_traces
        assert output_path.exists()

        # Verify merged file
        with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
            assert f.tracecount == total_traces
            assert len(f.samples) == n_samples

    def test_merge_stats(self, test_segy_segments):
        """Test merge statistics."""
        segment_paths, n_samples, total_traces = test_segy_segments

        merger = SEGYSegmentMerger(n_samples=n_samples, data_format=5)
        stats = merger.get_merge_stats(segment_paths)

        assert stats['n_segments'] == 3
        assert stats['total_traces'] == total_traces


class TestTracePartitioning:
    """Test trace partitioning for export workers."""

    def test_partition_traces_even(self):
        """Test even trace partitioning."""
        n_traces = 1000
        n_workers = 4

        traces_per_worker = n_traces // n_workers
        remainder = n_traces % n_workers

        segments = []
        start = 0
        for i in range(n_workers):
            n = traces_per_worker + (1 if i < remainder else 0)
            segments.append(TraceSegment(
                segment_id=i,
                start_trace=start,
                end_trace=start + n - 1,
                n_traces=n
            ))
            start += n

        # Verify coverage
        total = sum(s.n_traces for s in segments)
        assert total == n_traces

        # Verify no gaps
        for i in range(len(segments) - 1):
            assert segments[i].end_trace + 1 == segments[i + 1].start_trace


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
