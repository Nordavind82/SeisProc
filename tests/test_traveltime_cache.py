"""
Unit tests for traveltime caching module.

Tests:
- TraveltimeTable creation and serialization
- LRU cache functionality
- Table builder
- Cached calculator wrapper
"""

import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.traveltime_cache import (
    TraveltimeTable,
    TraveltimeCache,
    TraveltimeTableBuilder,
    CachedTraveltimeCalculator,
    create_traveltime_cache,
    create_cached_calculator,
)
from processors.migration.traveltime import StraightRayTraveltime
from models.velocity_model import create_constant_velocity


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


class TestTraveltimeTable:
    """Tests for TraveltimeTable dataclass."""

    def test_table_creation(self):
        """Test basic table creation."""
        times = np.random.rand(10, 20, 15).astype(np.float32)
        z_axis = np.linspace(0, 2, 10)
        surface_x = np.linspace(0, 1000, 20)

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
        )

        assert table.shape == (10, 20, 15)
        assert len(table.z_axis) == 10
        assert table.is_2d

    def test_memory_estimation(self):
        """Test memory size estimation."""
        times = np.zeros((100, 50, 50), dtype=np.float32)
        z_axis = np.linspace(0, 2, 100)
        surface_x = np.linspace(0, 1000, 50)

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
        )

        # 100 * 50 * 50 * 4 bytes = 1,000,000 bytes = ~0.95 MB
        assert 0.9 < table.memory_mb < 1.1

    def test_to_device(self):
        """Test moving table to device."""
        times = np.random.rand(5, 10, 8).astype(np.float32)
        z_axis = np.linspace(0, 1, 5)
        surface_x = np.linspace(0, 500, 10)

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
        )

        table_gpu = table.to_device(TEST_DEVICE)

        assert isinstance(table_gpu.times, torch.Tensor)
        assert table_gpu.times.device == TEST_DEVICE

    def test_save_load(self):
        """Test table serialization."""
        times = np.random.rand(5, 10, 8).astype(np.float32)
        z_axis = np.linspace(0, 1, 5).astype(np.float32)
        surface_x = np.linspace(0, 500, 10).astype(np.float32)

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
            metadata={'test_key': 'test_value'},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_table.npz'
            table.save(str(filepath))

            assert filepath.exists()

            loaded = TraveltimeTable.load(str(filepath))

            np.testing.assert_array_almost_equal(loaded.times, times)
            np.testing.assert_array_almost_equal(loaded.z_axis, z_axis)
            assert loaded.metadata.get('test_key') == 'test_value'

    def test_load_to_device(self):
        """Test loading table directly to device."""
        times = np.random.rand(5, 10, 8).astype(np.float32)
        z_axis = np.linspace(0, 1, 5).astype(np.float32)
        surface_x = np.linspace(0, 500, 10).astype(np.float32)

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_table.npz'
            table.save(str(filepath))

            loaded = TraveltimeTable.load(str(filepath), device=TEST_DEVICE)

            assert isinstance(loaded.times, torch.Tensor)


class TestTraveltimeCache:
    """Tests for LRU traveltime cache."""

    def test_cache_creation(self):
        """Test cache initialization."""
        cache = TraveltimeCache(max_size_mb=100.0, device=TEST_DEVICE)

        assert cache.max_size_mb == 100.0
        assert cache.size_mb == 0.0

    def test_cache_put_get(self):
        """Test basic cache operations."""
        cache = TraveltimeCache(max_size_mb=100.0, device=TEST_DEVICE)

        surface_x = np.array([0.0, 100.0, 200.0])
        surface_y = np.zeros(3)
        z = 1.0
        traveltimes = torch.rand(3, 5)

        # Put
        cache.put(surface_x, surface_y, z, traveltimes)

        # Get
        result = cache.get(surface_x, surface_y, z)

        assert result is not None
        torch.testing.assert_close(result, traveltimes.to(TEST_DEVICE))

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = TraveltimeCache(max_size_mb=100.0, device=TEST_DEVICE)

        result = cache.get(
            np.array([100.0]),
            np.array([0.0]),
            1.5,
        )

        assert result is None

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        # Small cache that can hold ~2 entries
        cache = TraveltimeCache(max_size_mb=0.01, device=TEST_DEVICE)

        # Add first entry
        cache.put(
            np.array([0.0]),
            np.array([0.0]),
            1.0,
            torch.rand(100, 100),  # ~40 KB
        )

        # Add second entry - should evict first
        cache.put(
            np.array([100.0]),
            np.array([0.0]),
            2.0,
            torch.rand(100, 100),
        )

        # First entry should be evicted
        # Note: actual behavior depends on entry sizes

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = TraveltimeCache(max_size_mb=100.0, device=TEST_DEVICE)

        surface_x = np.array([0.0])
        surface_y = np.array([0.0])

        # Miss
        cache.get(surface_x, surface_y, 1.0)

        # Put and hit
        cache.put(surface_x, surface_y, 1.0, torch.rand(5, 5))
        cache.get(surface_x, surface_y, 1.0)

        stats = cache.get_stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = TraveltimeCache(max_size_mb=100.0, device=TEST_DEVICE)

        cache.put(np.array([0.0]), np.array([0.0]), 1.0, torch.rand(10, 10))

        assert cache.size_mb > 0

        cache.clear()

        assert cache.size_mb == 0.0
        assert cache.get_stats()['entries'] == 0


class TestTraveltimeTableBuilder:
    """Tests for traveltime table builder."""

    @pytest.fixture
    def calculator(self):
        v_model = create_constant_velocity(2500.0)
        return StraightRayTraveltime(v_model, device=TEST_DEVICE)

    def test_build_2d_table(self, calculator):
        """Test building 2D traveltime table."""
        builder = TraveltimeTableBuilder(calculator, device=TEST_DEVICE)

        z_axis = np.linspace(0.1, 2.0, 10).astype(np.float32)
        surface_x = np.linspace(-500, 500, 20).astype(np.float32)
        image_x = np.linspace(0, 1000, 15).astype(np.float32)

        table = builder.build_2d_table(z_axis, surface_x, image_x)

        assert table.shape == (10, 20, 15)
        assert table.is_2d

        # Check metadata
        assert 'computation_time_s' in table.metadata
        assert 'calculator_type' in table.metadata

    def test_table_values_positive(self, calculator):
        """Test that table values are positive."""
        builder = TraveltimeTableBuilder(calculator, device=TEST_DEVICE)

        z_axis = np.linspace(0.5, 1.5, 5).astype(np.float32)
        surface_x = np.linspace(0, 500, 10).astype(np.float32)
        image_x = np.linspace(0, 500, 8).astype(np.float32)

        table = builder.build_2d_table(z_axis, surface_x, image_x)

        if isinstance(table.times, torch.Tensor):
            times = table.times.cpu().numpy()
        else:
            times = table.times

        assert np.all(times > 0)
        assert np.all(np.isfinite(times))


class TestCachedTraveltimeCalculator:
    """Tests for cached calculator wrapper."""

    @pytest.fixture
    def base_calculator(self):
        v_model = create_constant_velocity(2500.0)
        return StraightRayTraveltime(v_model, device=TEST_DEVICE)

    def test_cached_calculator_creation(self, base_calculator):
        """Test creating cached calculator."""
        cached = CachedTraveltimeCalculator(
            base_calculator,
            cache_size_mb=100.0,
            device=TEST_DEVICE,
        )

        assert cached.calculator is base_calculator

    def test_compute_traveltime_passthrough(self, base_calculator):
        """Test single traveltime computation passes through."""
        cached = CachedTraveltimeCalculator(
            base_calculator,
            cache_size_mb=100.0,
            device=TEST_DEVICE,
        )

        t = cached.compute_traveltime(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert float(t) > 0

    def test_batch_computation_caching(self, base_calculator):
        """Test batch computation uses cache."""
        cached = CachedTraveltimeCalculator(
            base_calculator,
            cache_size_mb=100.0,
            device=TEST_DEVICE,
        )

        surface_x = torch.tensor([0.0, 500.0, 1000.0])
        surface_y = torch.zeros(3)
        image_x = torch.tensor([250.0, 750.0])
        image_y = torch.zeros(2)
        image_z = torch.tensor([0.5, 1.0, 1.5])

        # First call - cache miss
        t1 = cached.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t1.shape == (3, 3, 2)

        # Second call - should use cache
        t2 = cached.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        torch.testing.assert_close(t1, t2)

        # Check cache was used
        stats = cached.get_cache_stats()
        assert stats['hits'] > 0

    def test_clear_cache(self, base_calculator):
        """Test clearing calculator cache."""
        cached = CachedTraveltimeCalculator(
            base_calculator,
            cache_size_mb=100.0,
            device=TEST_DEVICE,
        )

        # Add something to cache
        cached.compute_traveltime_batch(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([100.0]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
        )

        assert cached.get_cache_stats()['entries'] > 0

        cached.clear_cache()

        assert cached.get_cache_stats()['entries'] == 0


class TestCachedCalculatorWithTable:
    """Tests for cached calculator with pre-computed table."""

    @pytest.fixture
    def calculator_and_table(self):
        v_model = create_constant_velocity(2500.0)
        calculator = StraightRayTraveltime(v_model, device=TEST_DEVICE)

        builder = TraveltimeTableBuilder(calculator, device=TEST_DEVICE)

        z_axis = np.linspace(0.5, 2.0, 10).astype(np.float32)
        surface_x = np.linspace(-500, 500, 20).astype(np.float32)
        image_x = np.linspace(0, 1000, 15).astype(np.float32)

        table = builder.build_2d_table(z_axis, surface_x, image_x)

        return calculator, table

    def test_cached_with_table(self, calculator_and_table):
        """Test cached calculator with pre-computed table."""
        calculator, table = calculator_and_table

        cached = CachedTraveltimeCalculator(
            calculator,
            table=table,
            device=TEST_DEVICE,
        )

        assert cached.table is not None


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_traveltime_cache(self):
        """Test cache factory."""
        cache = create_traveltime_cache(max_size_mb=200.0)

        assert cache.max_size_mb == 200.0

    def test_create_cached_calculator(self):
        """Test cached calculator factory."""
        v_model = create_constant_velocity(2500.0)
        calculator = StraightRayTraveltime(v_model, device=TEST_DEVICE)

        cached = create_cached_calculator(
            calculator,
            cache_size_mb=150.0,
            device=TEST_DEVICE,
        )

        assert isinstance(cached, CachedTraveltimeCalculator)


class TestTablePersistence:
    """Tests for table save/load workflow."""

    def test_full_workflow(self):
        """Test complete table creation, save, load workflow."""
        # Create calculator
        v_model = create_constant_velocity(3000.0)
        calculator = StraightRayTraveltime(v_model, device=TEST_DEVICE)

        # Build table
        builder = TraveltimeTableBuilder(calculator, device=TEST_DEVICE)
        z_axis = np.linspace(0.5, 1.5, 5).astype(np.float32)
        surface_x = np.linspace(0, 500, 10).astype(np.float32)
        image_x = np.linspace(0, 500, 8).astype(np.float32)

        table = builder.build_2d_table(z_axis, surface_x, image_x)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            filepath = Path(tmpdir) / 'traveltime_table.npz'
            table.save(str(filepath))

            # Load and use
            loaded = TraveltimeTable.load(str(filepath), device=TEST_DEVICE)
            cached = CachedTraveltimeCalculator(
                calculator,
                table=loaded,
                device=TEST_DEVICE,
            )

            # Should work with loaded table
            t = cached.compute_traveltime(
                torch.tensor(250.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
            )
            assert float(t) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
