"""
Integration tests for sorting memory optimization.

Tests the memory-optimized sorting pipeline to ensure:
1. Sort mappings stream correctly to disk
2. Headers are reordered properly with chunking
3. Memory usage stays within bounds
4. Results are correct (traces in expected order)

Usage:
    python -m pytest tests/test_sorting_memory.py -v
    python tests/test_sorting_memory.py  # Direct run
"""

import os
import sys
import gc
import tempfile
import shutil
import numpy as np
import pandas as pd
import zarr
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parallel_processing.worker import (
    StreamingSortWriter,
    read_streaming_sort_file,
    compute_gather_sort_indices
)
from utils.parallel_processing.config import SortOptions
from utils.trace_sorter import TraceSorter
from utils.memory_profiler_diagnostic import (
    MemoryProfiler,
    memory_checkpoint,
    check_memory_budget,
    estimate_sorting_memory
)


class TestStreamingSortWriter:
    """Tests for StreamingSortWriter binary format."""

    def test_write_single_mapping(self):
        """Test writing a single sort mapping."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            filepath = f.name

        try:
            # Write
            with StreamingSortWriter(filepath) as writer:
                indices = np.array([2, 0, 1, 3], dtype=np.int64)
                writer.write_mapping(0, 100, 103, indices)

            # Read back
            mappings = read_streaming_sort_file(filepath)

            assert len(mappings) == 1
            gather_idx, g_start, g_end, sort_indices = mappings[0]
            assert gather_idx == 0
            assert g_start == 100
            assert g_end == 103
            np.testing.assert_array_equal(sort_indices, indices)

        finally:
            os.unlink(filepath)

    def test_write_multiple_mappings(self):
        """Test writing multiple sort mappings."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            filepath = f.name

        try:
            n_gathers = 100

            # Write many mappings
            with StreamingSortWriter(filepath) as writer:
                for i in range(n_gathers):
                    n_traces = np.random.randint(10, 100)
                    indices = np.random.permutation(n_traces).astype(np.int64)
                    g_start = i * 100
                    g_end = g_start + n_traces - 1
                    writer.write_mapping(i, g_start, g_end, indices)

            # Read back
            mappings = read_streaming_sort_file(filepath)

            assert len(mappings) == n_gathers

            # Verify first and last
            assert mappings[0][0] == 0  # gather_idx
            assert mappings[-1][0] == n_gathers - 1

        finally:
            os.unlink(filepath)

    def test_backward_compatibility_pickle(self):
        """Test reading legacy pickle format."""
        import pickle

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name

        try:
            # Write legacy pickle format
            legacy_data = [
                (0, 0, 9, np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)),
                (1, 10, 19, np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64))
            ]
            with open(filepath, 'wb') as f:
                pickle.dump(legacy_data, f)

            # Read using new function (should auto-detect format)
            mappings = read_streaming_sort_file(filepath)

            assert len(mappings) == 2
            np.testing.assert_array_equal(mappings[0][3], legacy_data[0][3])

        finally:
            os.unlink(filepath)


class TestVectorizedSorting:
    """Tests for vectorized sorting operations."""

    def setup_method(self):
        """Create test data."""
        self.n_traces = 1000
        self.n_gathers = 10
        self.traces_per_gather = self.n_traces // self.n_gathers

        # Create headers with offset column for sorting
        np.random.seed(42)
        offsets = []
        for g in range(self.n_gathers):
            # Random offsets within each gather
            gather_offsets = np.random.permutation(self.traces_per_gather) * 100
            offsets.extend(gather_offsets)

        self.headers_df = pd.DataFrame({
            'trace_index': np.arange(self.n_traces),
            'offset': offsets,
            'gather_id': np.repeat(np.arange(self.n_gathers), self.traces_per_gather)
        })

        self.ensemble_df = pd.DataFrame({
            'start_trace': np.arange(self.n_gathers) * self.traces_per_gather,
            'end_trace': (np.arange(self.n_gathers) + 1) * self.traces_per_gather - 1,
            'n_traces': [self.traces_per_gather] * self.n_gathers
        })

    def test_vectorized_global_mapping(self):
        """Test that vectorized mapping produces correct results."""
        sorter = TraceSorter(sort_key='offset', ascending=True)

        # Compute mapping
        global_mapping = sorter.compute_global_sort_mapping(
            self.headers_df, self.ensemble_df
        )

        assert len(global_mapping) == self.n_traces

        # Verify each gather is sorted correctly
        for g in range(self.n_gathers):
            start = g * self.traces_per_gather
            end = start + self.traces_per_gather

            # Get original indices for this gather's sorted positions
            gather_original_indices = global_mapping[start:end]

            # Get offsets in sorted order
            sorted_offsets = self.headers_df.iloc[gather_original_indices]['offset'].values

            # Verify ascending order
            assert np.all(np.diff(sorted_offsets) >= 0), \
                f"Gather {g} not sorted correctly"

    def test_chunked_global_mapping(self):
        """Test chunked version produces same result as regular."""
        sorter = TraceSorter(sort_key='offset', ascending=True)

        regular_mapping = sorter.compute_global_sort_mapping(
            self.headers_df, self.ensemble_df
        )

        chunked_mapping = sorter.compute_global_sort_mapping_chunked(
            self.headers_df, self.ensemble_df, chunk_size=3
        )

        np.testing.assert_array_equal(regular_mapping, chunked_mapping)

    def test_chunked_header_creation(self):
        """Test chunked header creation produces correct results."""
        sorter = TraceSorter(sort_key='offset', ascending=True)
        global_mapping = sorter.compute_global_sort_mapping(
            self.headers_df, self.ensemble_df
        )

        # Small chunk size to test chunking
        sorted_headers = sorter.create_sorted_headers(
            self.headers_df, global_mapping, chunk_size=100
        )

        assert len(sorted_headers) == self.n_traces
        assert 'trace_index' in sorted_headers.columns
        assert 'original_trace_index' in sorted_headers.columns

        # Verify trace_index is sequential
        np.testing.assert_array_equal(
            sorted_headers['trace_index'].values,
            np.arange(self.n_traces)
        )


class TestMemoryProfiler:
    """Tests for memory profiler diagnostic tool."""

    def test_memory_checkpoint(self):
        """Test memory checkpoint recording."""
        profiler = MemoryProfiler("test_profiler")

        cp1 = profiler.checkpoint("start")
        assert cp1.name == "start"
        assert cp1.rss_bytes > 0

        # Allocate some memory
        data = np.zeros((1000, 1000), dtype=np.float64)

        cp2 = profiler.checkpoint("after_allocation")
        assert cp2.rss_bytes >= cp1.rss_bytes

        del data
        gc.collect()

        profiler.cleanup()

    def test_memory_budget_estimation(self):
        """Test memory budget estimation."""
        estimates = estimate_sorting_memory(
            n_traces=1_000_000,
            n_header_columns=50
        )

        assert 'mapping_array_mb' in estimates
        assert 'headers_df_mb' in estimates
        assert 'estimated_peak_mb' in estimates

        # 1M traces * 8 bytes = ~8MB for mapping
        assert estimates['mapping_array_mb'] == pytest.approx(8.0, rel=0.1)

    def test_memory_budget_check(self):
        """Test memory budget check function."""
        # Small dataset should always be safe
        is_safe, available, required, msg = check_memory_budget(
            n_traces=1000,
            sorting_enabled=True
        )

        assert is_safe == True
        assert available > 0
        assert required > 0
        assert len(msg) > 0


class TestIntegrationSorting:
    """Integration tests for the full sorting pipeline."""

    def setup_method(self):
        """Create test dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / 'input'
        self.output_dir = Path(self.temp_dir) / 'output'
        self.input_dir.mkdir()

        # Create test data
        self.n_samples = 100
        self.n_traces = 500
        self.n_gathers = 5
        self.traces_per_gather = self.n_traces // self.n_gathers

        # Create Zarr array
        zarr_path = self.input_dir / 'traces.zarr'
        traces = zarr.open(
            str(zarr_path), mode='w',
            shape=(self.n_samples, self.n_traces),
            chunks=(self.n_samples, 100),
            dtype=np.float32
        )

        # Fill with test data (each trace has unique pattern)
        np.random.seed(42)
        for i in range(self.n_traces):
            traces[:, i] = np.sin(np.linspace(0, 2*np.pi * (i+1), self.n_samples))

        # Create headers with random offsets per gather
        offsets = []
        for g in range(self.n_gathers):
            gather_offsets = np.random.permutation(self.traces_per_gather) * 100 + 50
            offsets.extend(gather_offsets)

        headers_df = pd.DataFrame({
            'trace_index': np.arange(self.n_traces),
            'offset': offsets,
            'CDP': np.repeat(np.arange(self.n_gathers), self.traces_per_gather)
        })
        headers_df.to_parquet(self.input_dir / 'headers.parquet')

        # Create ensemble index
        ensemble_df = pd.DataFrame({
            'CDP': np.arange(self.n_gathers),
            'start_trace': np.arange(self.n_gathers) * self.traces_per_gather,
            'end_trace': (np.arange(self.n_gathers) + 1) * self.traces_per_gather - 1,
            'n_traces': [self.traces_per_gather] * self.n_gathers
        })
        ensemble_df.to_parquet(self.input_dir / 'ensemble_index.parquet')

        # Create metadata
        import json
        metadata = {
            'n_samples': self.n_samples,
            'n_traces': self.n_traces,
            'sample_rate': 0.004
        }
        with open(self.input_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    def teardown_method(self):
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)

    def test_sort_mapping_workflow(self):
        """Test the complete sort mapping workflow."""
        headers_df = pd.read_parquet(self.input_dir / 'headers.parquet')
        ensemble_df = pd.read_parquet(self.input_dir / 'ensemble_index.parquet')

        sort_options = SortOptions(
            enabled=True,
            sort_key='offset',
            ascending=True
        )

        # Test worker-side sort computation
        sort_file = Path(self.temp_dir) / 'sort_mapping.bin'

        with StreamingSortWriter(str(sort_file)) as writer:
            for g_idx in range(self.n_gathers):
                g_start = g_idx * self.traces_per_gather
                g_end = g_start + self.traces_per_gather - 1

                sort_indices = compute_gather_sort_indices(
                    headers_df, g_start, g_end, sort_options
                )

                writer.write_mapping(g_idx, g_start, g_end, sort_indices)

        # Read back and verify
        mappings = read_streaming_sort_file(str(sort_file))
        assert len(mappings) == self.n_gathers

        # Build global mapping
        global_mapping = np.arange(self.n_traces, dtype=np.int64)
        for gather_idx, g_start, g_end, local_sort_indices in mappings:
            n_gather_traces = len(local_sort_indices)
            old_global_positions = g_start + local_sort_indices
            global_mapping[g_start:g_start + n_gather_traces] = old_global_positions

        # Verify sorting is correct
        sorted_offsets = headers_df.iloc[global_mapping]['offset'].values

        for g in range(self.n_gathers):
            start = g * self.traces_per_gather
            end = start + self.traces_per_gather
            gather_offsets = sorted_offsets[start:end]
            assert np.all(np.diff(gather_offsets) >= 0), f"Gather {g} not sorted"


def run_memory_stress_test(n_traces: int = 100_000):
    """
    Run a memory stress test with larger dataset.

    This can be run manually to verify memory behavior:
        python tests/test_sorting_memory.py --stress
    """
    print(f"\n{'='*60}")
    print(f"MEMORY STRESS TEST: {n_traces:,} traces")
    print(f"{'='*60}\n")

    profiler = MemoryProfiler("stress_test")

    # Check initial memory budget
    is_safe, available, required, msg = check_memory_budget(n_traces, sorting_enabled=True)
    print(f"Memory budget: {msg}")

    if not is_safe:
        print("WARNING: Insufficient memory, reducing test size")
        n_traces = 10_000

    profiler.checkpoint("start")

    # Create test data
    n_gathers = n_traces // 100
    traces_per_gather = 100

    with memory_checkpoint(profiler, "create_headers"):
        headers_df = pd.DataFrame({
            'trace_index': np.arange(n_traces),
            'offset': np.random.randint(0, 10000, n_traces),
            'CDP': np.repeat(np.arange(n_gathers), traces_per_gather)
        })

    with memory_checkpoint(profiler, "create_ensemble"):
        ensemble_df = pd.DataFrame({
            'start_trace': np.arange(n_gathers) * traces_per_gather,
            'end_trace': (np.arange(n_gathers) + 1) * traces_per_gather - 1,
            'n_traces': [traces_per_gather] * n_gathers
        })

    with memory_checkpoint(profiler, "compute_mapping"):
        sorter = TraceSorter(sort_key='offset', ascending=True)
        global_mapping = sorter.compute_global_sort_mapping(headers_df, ensemble_df)

    with memory_checkpoint(profiler, "create_sorted_headers"):
        sorted_headers = sorter.create_sorted_headers(headers_df, global_mapping)

    profiler.checkpoint("cleanup_start")
    del headers_df
    del global_mapping
    del sorted_headers
    gc.collect()
    profiler.checkpoint("cleanup_end")

    profiler.print_report()
    profiler.cleanup()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stress', action='store_true', help='Run memory stress test')
    parser.add_argument('--traces', type=int, default=100_000, help='Number of traces for stress test')
    args = parser.parse_args()

    if args.stress:
        run_memory_stress_test(args.traces)
    else:
        pytest.main([__file__, '-v'])
