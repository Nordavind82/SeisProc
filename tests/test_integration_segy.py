"""
Integration tests for SEG-Y import/export functionality.

Tests the full workflow: SEG-Y file → import → process → export → verify.
"""
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSEGYImport:
    """Test SEG-Y file import functionality."""

    def test_read_file_info(self, sample_segy_path):
        """Test reading SEG-Y file information."""
        from utils.segy_import.segy_reader import SEGYReader
        from utils.segy_import.header_mapping import HeaderMapping

        mapping = HeaderMapping()
        reader = SEGYReader(str(sample_segy_path), mapping)

        info = reader.read_file_info()

        assert info['n_traces'] == 50
        assert info['n_samples'] == 1000
        assert 'sample_interval' in info
        assert 'text_header' in info

    def test_read_traces(self, sample_segy_path):
        """Test reading trace data from SEG-Y file."""
        from utils.segy_import.segy_reader import SEGYReader
        from utils.segy_import.header_mapping import HeaderMapping

        mapping = HeaderMapping()
        mapping.add_standard_headers()  # Required for batch header reading
        reader = SEGYReader(str(sample_segy_path), mapping)

        traces, headers = reader.read_all_traces()

        assert traces.shape[0] == 1000  # n_samples
        assert traces.shape[1] == 50    # n_traces
        assert len(headers) == 50

    def test_file_handle_context_manager(self, sample_segy_path):
        """Test SEGYFileHandle context manager for batch operations."""
        from utils.segy_import.segy_reader import SEGYReader
        from utils.segy_import.header_mapping import HeaderMapping

        mapping = HeaderMapping()
        mapping.add_standard_headers()  # Required for batch header reading
        reader = SEGYReader(str(sample_segy_path), mapping)

        with reader.open() as handle:
            info = handle.read_file_info()
            traces = handle.read_traces_range(0, 10)
            headers = handle.read_headers_range(0, 10)

        assert info['n_traces'] == 50
        assert traces.shape == (1000, 10)
        assert len(headers) == 10


class TestSEGYExport:
    """Test SEG-Y export functionality."""

    def test_export_preserves_data(self, sample_segy_path, temp_dir, seismic_data):
        """Test that export preserves trace data."""
        from utils.segy_import.segy_export import SEGYExporter
        import segyio

        output_path = temp_dir / "output.sgy"

        exporter = SEGYExporter(str(output_path))
        exporter.export(str(sample_segy_path), seismic_data)

        # Read back and verify
        with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
            assert f.tracecount == seismic_data.n_traces
            assert len(f.samples) == seismic_data.n_samples

            # Check a few traces
            for i in [0, 25, 49]:
                exported_trace = f.trace[i]
                original_trace = seismic_data.traces[:, i]
                assert np.allclose(exported_trace, original_trace, rtol=1e-5)

    def test_export_dimension_mismatch_raises(self, sample_segy_path, temp_dir):
        """Test that dimension mismatch raises ValueError."""
        from utils.segy_import.segy_export import SEGYExporter
        from models.seismic_data import SeismicData

        # Create data with wrong dimensions
        wrong_data = SeismicData(
            traces=np.zeros((500, 50), dtype=np.float32),  # Wrong sample count
            sample_rate=2.0
        )

        output_path = temp_dir / "output.sgy"
        exporter = SEGYExporter(str(output_path))

        with pytest.raises(ValueError, match="Sample count mismatch"):
            exporter.export(str(sample_segy_path), wrong_data)


class TestSEGYRoundTrip:
    """Test full SEG-Y round-trip: read → process → write → read."""

    def test_full_roundtrip(self, sample_segy_path, temp_dir):
        """Test complete processing round-trip."""
        from utils.segy_import.segy_reader import SEGYReader
        from utils.segy_import.segy_export import SEGYExporter
        from utils.segy_import.header_mapping import HeaderMapping
        from processors.bandpass_filter import BandpassFilter
        from models.seismic_data import SeismicData
        import segyio

        # Step 1: Read original SEG-Y
        mapping = HeaderMapping()
        mapping.add_standard_headers()  # Required for batch header reading
        reader = SEGYReader(str(sample_segy_path), mapping)
        original_traces, headers = reader.read_all_traces()
        info = reader.read_file_info()

        original_data = SeismicData(
            traces=original_traces,
            sample_rate=info['sample_interval']
        )

        # Step 2: Process
        processor = BandpassFilter(low_freq=10.0, high_freq=60.0)
        processed_data = processor.process(original_data)

        # Step 3: Export
        output_path = temp_dir / "processed.sgy"
        exporter = SEGYExporter(str(output_path))
        exporter.export(str(sample_segy_path), processed_data)

        # Step 4: Read back and verify
        with segyio.open(str(output_path), 'r', ignore_geometry=True) as f:
            assert f.tracecount == original_data.n_traces
            assert len(f.samples) == original_data.n_samples

            # Verify data was actually changed (filtered)
            for i in range(min(5, original_data.n_traces)):
                exported_trace = f.trace[i]
                original_trace = original_data.traces[:, i]
                processed_trace = processed_data.traces[:, i]

                # Exported should match processed, not original
                assert np.allclose(exported_trace, processed_trace, rtol=1e-5)
                # And should be different from original (filter applied)
                assert not np.allclose(exported_trace, original_trace)


class TestLazyLoading:
    """Test lazy loading functionality."""

    def test_lazy_window_access(self, zarr_data_dir):
        """Test accessing data windows from lazy loader."""
        try:
            from models.lazy_seismic_data import LazySeismicData
        except ImportError:
            pytest.skip("LazySeismicData not available")

        lazy_data = LazySeismicData.from_storage_dir(str(zarr_data_dir))

        # Access a window
        window = lazy_data.get_window(
            time_start=0,
            time_end=500,  # ms (250 samples at 2ms)
            trace_start=10,
            trace_end=30
        )

        # Window should be smaller than full data
        assert window.shape[1] == 20  # traces
        # Sample count depends on time range

    def test_lazy_iteration(self, zarr_data_dir):
        """Test iterating over lazy data."""
        try:
            from models.lazy_seismic_data import LazySeismicData
        except ImportError:
            pytest.skip("LazySeismicData not available")

        lazy_data = LazySeismicData.from_storage_dir(str(zarr_data_dir))

        # Iterate over chunks
        chunks = list(lazy_data.iterate_chunks(chunk_size=25))

        # Should have 2 chunks for 50 traces
        assert len(chunks) == 2
        assert chunks[0].shape[1] == 25
        assert chunks[1].shape[1] == 25
