"""
Unit tests for TraceSpatialIndex.

Tests:
- Index construction
- Region queries
- Point queries
- Query correctness (no false negatives)
- Grid-based alternative
- Performance benchmarks
"""

import numpy as np
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.trace_index import (
    TraceSpatialIndex,
    GridBasedTraceIndex,
    TraceBounds,
    create_trace_index,
)


class TestIndexConstruction:
    """Tests for index construction."""

    def test_basic_construction(self):
        """Test basic index construction."""
        index = TraceSpatialIndex()

        src_x = np.array([0, 100, 200, 300])
        src_y = np.array([0, 0, 0, 0])
        rcv_x = np.array([50, 150, 250, 350])
        rcv_y = np.array([0, 0, 0, 0])

        index.build(src_x, src_y, rcv_x, rcv_y)

        assert index.built
        assert index.n_traces == 4

    def test_bounds_computed(self):
        """Test that bounds are correctly computed."""
        index = TraceSpatialIndex()

        src_x = np.array([0, 1000, 2000])
        src_y = np.array([0, 500, 1000])
        rcv_x = np.array([100, 1100, 2100])
        rcv_y = np.array([100, 600, 1100])

        index.build(src_x, src_y, rcv_x, rcv_y)

        bounds = index.get_trace_bounds()

        # Midpoints: (50, 50), (1050, 550), (2050, 1050)
        assert bounds.x_min == 50
        assert bounds.x_max == 2050
        assert bounds.y_min == 50
        assert bounds.y_max == 1050

    def test_method_chaining(self):
        """Test that build returns self for chaining."""
        index = TraceSpatialIndex()

        src_x = np.array([0, 100])
        src_y = np.array([0, 0])
        rcv_x = np.array([50, 150])
        rcv_y = np.array([0, 0])

        result = index.build(src_x, src_y, rcv_x, rcv_y)

        assert result is index


class TestRegionQueries:
    """Tests for region-based queries."""

    @pytest.fixture
    def large_index(self):
        """Create index with spread traces."""
        np.random.seed(42)
        n_traces = 1000

        # Spread traces over 10km x 10km area
        src_x = np.random.uniform(0, 10000, n_traces)
        src_y = np.random.uniform(0, 10000, n_traces)
        # Receivers offset by 0-500m
        offset = np.random.uniform(100, 500, n_traces)
        angle = np.random.uniform(0, 2 * np.pi, n_traces)
        rcv_x = src_x + offset * np.cos(angle)
        rcv_y = src_y + offset * np.sin(angle)

        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)
        return index

    def test_region_finds_nearby_traces(self, large_index):
        """Region query should find nearby traces."""
        # Query a 1km x 1km region in the center
        traces = large_index.query_traces_for_region(
            4000, 5000, 4000, 5000, buffer=500
        )

        # Should find some traces
        assert len(traces) > 0

    def test_region_with_larger_buffer(self, large_index):
        """Larger buffer should find more traces."""
        # Small buffer
        traces_small = large_index.query_traces_for_region(
            4500, 5500, 4500, 5500, buffer=100
        )

        # Large buffer
        traces_large = large_index.query_traces_for_region(
            4500, 5500, 4500, 5500, buffer=2000
        )

        assert len(traces_large) >= len(traces_small)

    def test_region_outside_data_empty(self, large_index):
        """Query outside data extent should return empty."""
        traces = large_index.query_traces_for_region(
            -5000, -4000, -5000, -4000, buffer=100
        )

        assert len(traces) == 0

    def test_region_returns_unique_indices(self, large_index):
        """Returned indices should be unique."""
        traces = large_index.query_traces_for_region(
            0, 10000, 0, 10000, buffer=1000
        )

        assert len(traces) == len(np.unique(traces))


class TestPointQueries:
    """Tests for point-based queries."""

    @pytest.fixture
    def simple_index(self):
        """Create simple index for point queries."""
        # Create a grid of traces
        x_grid = np.arange(0, 5000, 100)
        y_grid = np.arange(0, 5000, 100)
        xx, yy = np.meshgrid(x_grid, y_grid)
        src_x = xx.flatten()
        src_y = yy.flatten()

        # Small offset
        rcv_x = src_x + 50
        rcv_y = src_y

        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)
        return index

    def test_point_finds_nearby_traces(self, simple_index):
        """Point query should find nearby traces."""
        traces = simple_index.query_traces_for_point(2500, 2500, max_dist=200)

        assert len(traces) > 0

    def test_point_respects_max_dist(self, simple_index):
        """Point query should respect max_dist."""
        # Small max_dist
        traces_small = simple_index.query_traces_for_point(2500, 2500, max_dist=75)

        # Large max_dist
        traces_large = simple_index.query_traces_for_point(2500, 2500, max_dist=500)

        assert len(traces_large) >= len(traces_small)

    def test_point_at_corner(self, simple_index):
        """Point query at corner should find fewer traces."""
        # Corner point
        traces_corner = simple_index.query_traces_for_point(0, 0, max_dist=200)

        # Center point
        traces_center = simple_index.query_traces_for_point(2500, 2500, max_dist=200)

        # Center should have more (or equal) traces
        assert len(traces_center) >= len(traces_corner)


class TestQueryCorrectness:
    """Tests to verify query correctness (no false negatives)."""

    def test_no_false_negatives_point_query(self):
        """Point query should not miss any valid traces."""
        np.random.seed(123)
        n_traces = 500

        src_x = np.random.uniform(0, 5000, n_traces)
        src_y = np.random.uniform(0, 5000, n_traces)
        rcv_x = src_x + np.random.uniform(-200, 200, n_traces)
        rcv_y = src_y + np.random.uniform(-200, 200, n_traces)

        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)

        # Query point
        px, py = 2500, 2500
        max_dist = 500

        # Index query
        index_traces = set(index.query_traces_for_point(px, py, max_dist))

        # Brute force
        brute_traces = set()
        for i in range(n_traces):
            d_src = np.sqrt((src_x[i] - px)**2 + (src_y[i] - py)**2)
            d_rcv = np.sqrt((rcv_x[i] - px)**2 + (rcv_y[i] - py)**2)
            if d_src <= max_dist and d_rcv <= max_dist:
                brute_traces.add(i)

        # Index should find all traces that brute force finds
        assert brute_traces.issubset(index_traces), \
            f"Missing traces: {brute_traces - index_traces}"

    def test_no_false_negatives_region_query(self):
        """Region query should not miss any valid traces."""
        np.random.seed(456)
        n_traces = 500

        src_x = np.random.uniform(0, 5000, n_traces)
        src_y = np.random.uniform(0, 5000, n_traces)
        rcv_x = src_x + np.random.uniform(-200, 200, n_traces)
        rcv_y = src_y + np.random.uniform(-200, 200, n_traces)

        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)

        # Query region
        x_min, x_max = 2000, 3000
        y_min, y_max = 2000, 3000
        buffer = 300

        # Index query (without refinement for fair comparison)
        index_traces = set(index.query_traces_for_region(
            x_min, x_max, y_min, y_max, buffer,
            refine_with_aperture=False
        ))

        # Brute force: check midpoints
        mid_x = (src_x + rcv_x) / 2
        mid_y = (src_y + rcv_y) / 2

        brute_traces = set()
        for i in range(n_traces):
            if (x_min - buffer <= mid_x[i] <= x_max + buffer and
                y_min - buffer <= mid_y[i] <= y_max + buffer):
                brute_traces.add(i)

        # Index should find all
        assert brute_traces == index_traces


class TestGridEstimation:
    """Tests for grid estimation functionality."""

    @pytest.fixture
    def spread_index(self):
        """Create index with spread traces."""
        np.random.seed(789)
        n_traces = 2000

        src_x = np.random.uniform(0, 10000, n_traces)
        src_y = np.random.uniform(0, 10000, n_traces)
        rcv_x = src_x + np.random.uniform(-300, 300, n_traces)
        rcv_y = src_y + np.random.uniform(-300, 300, n_traces)

        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)
        return index

    def test_grid_estimation(self, spread_index):
        """Test grid estimation provides useful stats."""
        stats = spread_index.estimate_traces_for_grid(
            grid_bounds=(0, 10000, 0, 10000),
            chunk_size=(2000, 2000),
            aperture=1000,
        )

        assert 'n_chunks' in stats
        assert 'counts' in stats
        assert stats['n_chunks'] == 25  # 5x5 grid

    def test_grid_estimation_covers_traces(self, spread_index):
        """Grid estimation should cover all traces."""
        stats = spread_index.estimate_traces_for_grid(
            grid_bounds=(0, 10000, 0, 10000),
            chunk_size=(10000, 10000),  # Single chunk
            aperture=500,
        )

        # With large aperture and single chunk, should find many traces
        assert stats['max_traces'] > 0


class TestGridBasedIndex:
    """Tests for grid-based alternative index."""

    def test_grid_index_construction(self):
        """Test grid index construction."""
        index = GridBasedTraceIndex(cell_size=500)

        src_x = np.array([0, 1000, 2000, 3000])
        src_y = np.array([0, 500, 1000, 1500])
        rcv_x = src_x + 100
        rcv_y = src_y

        index.build(src_x, src_y, rcv_x, rcv_y)

        assert index._built

    def test_grid_index_query(self):
        """Test grid index queries."""
        np.random.seed(111)
        n_traces = 500

        src_x = np.random.uniform(0, 5000, n_traces)
        src_y = np.random.uniform(0, 5000, n_traces)
        rcv_x = src_x + 100
        rcv_y = src_y

        index = GridBasedTraceIndex(cell_size=500)
        index.build(src_x, src_y, rcv_x, rcv_y)

        traces = index.query_traces_for_region(
            2000, 3000, 2000, 3000, buffer=500
        )

        assert len(traces) > 0

    def test_grid_vs_kdtree_consistency(self):
        """Grid and KD-tree should give similar results."""
        np.random.seed(222)
        n_traces = 500

        src_x = np.random.uniform(0, 5000, n_traces)
        src_y = np.random.uniform(0, 5000, n_traces)
        rcv_x = src_x + 100
        rcv_y = src_y

        # KD-tree index
        kd_index = TraceSpatialIndex()
        kd_index.build(src_x, src_y, rcv_x, rcv_y)

        # Grid index
        grid_index = GridBasedTraceIndex(cell_size=200)
        grid_index.build(src_x, src_y, rcv_x, rcv_y)

        # Query same region
        kd_traces = set(kd_index.query_traces_for_region(
            2000, 3000, 2000, 3000, buffer=500, refine_with_aperture=False
        ))
        grid_traces = set(grid_index.query_traces_for_region(
            2000, 3000, 2000, 3000, buffer=500
        ))

        # Should find same traces (grid may have some extra at boundaries)
        assert kd_traces.issubset(grid_traces) or grid_traces.issubset(kd_traces) or kd_traces == grid_traces


class TestFactoryFunction:
    """Tests for factory function."""

    def test_factory_creates_kdtree(self):
        """Factory should create KD-tree index by default."""
        src_x = np.array([0, 100, 200])
        src_y = np.array([0, 0, 0])
        rcv_x = np.array([50, 150, 250])
        rcv_y = np.array([0, 0, 0])

        index = create_trace_index(src_x, src_y, rcv_x, rcv_y)

        assert isinstance(index, TraceSpatialIndex)
        assert index.built

    def test_factory_creates_grid(self):
        """Factory should create grid index when requested."""
        src_x = np.array([0, 100, 200])
        src_y = np.array([0, 0, 0])
        rcv_x = np.array([50, 150, 250])
        rcv_y = np.array([0, 0, 0])

        index = create_trace_index(
            src_x, src_y, rcv_x, rcv_y,
            use_grid=True, cell_size=100
        )

        assert isinstance(index, GridBasedTraceIndex)
        assert index._built


class TestErrorHandling:
    """Tests for error handling."""

    def test_query_before_build_fails(self):
        """Query before build should fail."""
        index = TraceSpatialIndex()

        with pytest.raises(RuntimeError, match="not built"):
            index.query_traces_for_region(0, 100, 0, 100, 50)

    def test_point_query_before_build_fails(self):
        """Point query before build should fail."""
        index = TraceSpatialIndex()

        with pytest.raises(RuntimeError, match="not built"):
            index.query_traces_for_point(50, 50, 100)

    def test_bounds_before_build_fails(self):
        """Get bounds before build should fail."""
        index = TraceSpatialIndex()

        with pytest.raises(RuntimeError, match="not built"):
            index.get_trace_bounds()


class TestPerformance:
    """Performance tests."""

    def test_large_index_construction(self):
        """Test index construction with many traces."""
        np.random.seed(333)
        n_traces = 100000

        src_x = np.random.uniform(0, 50000, n_traces)
        src_y = np.random.uniform(0, 50000, n_traces)
        rcv_x = src_x + np.random.uniform(-500, 500, n_traces)
        rcv_y = src_y + np.random.uniform(-500, 500, n_traces)

        import time
        start = time.time()
        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)
        build_time = time.time() - start

        print(f"\nBuilt index for {n_traces} traces in {build_time:.3f}s")

        assert index.built
        assert build_time < 5.0  # Should be fast

    def test_query_performance(self):
        """Test query performance."""
        np.random.seed(444)
        n_traces = 100000

        src_x = np.random.uniform(0, 50000, n_traces)
        src_y = np.random.uniform(0, 50000, n_traces)
        rcv_x = src_x + np.random.uniform(-500, 500, n_traces)
        rcv_y = src_y + np.random.uniform(-500, 500, n_traces)

        index = TraceSpatialIndex()
        index.build(src_x, src_y, rcv_x, rcv_y)

        import time

        # Many queries
        n_queries = 100
        start = time.time()
        for _ in range(n_queries):
            x = np.random.uniform(10000, 40000)
            y = np.random.uniform(10000, 40000)
            index.query_traces_for_region(
                x, x + 1000, y, y + 1000, buffer=3000
            )
        query_time = time.time() - start

        print(f"\n{n_queries} region queries in {query_time:.3f}s "
              f"({query_time/n_queries*1000:.2f}ms each)")

        assert query_time < 5.0  # Should be fast


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
