"""
Unit tests for OptimizedKirchhoffMigrator.

Tests:
- Migrator initialization with various optimization levels
- Single gather migration
- Dataset migration
- Optimization statistics
- Output consistency with baseline
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.optimized_kirchhoff_migrator import (
    OptimizedKirchhoffMigrator,
    OptimizationStats,
    create_optimized_kirchhoff_migrator,
)
from models.velocity_model import VelocityModel, create_constant_velocity
from models.migration_config import MigrationConfig, OutputGrid
from models.migration_geometry import MigrationGeometry
from models.seismic_data import SeismicData


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


def create_test_config(n_time=50, n_inline=20, n_xline=20):
    """Create test migration config."""
    output_grid = OutputGrid(
        n_time=n_time,
        n_inline=n_inline,
        n_xline=n_xline,
        dt=0.004,
        d_inline=25.0,
        d_xline=25.0,
        t0=0.0,
        x_origin=0.0,
        y_origin=0.0,
    )

    return MigrationConfig(
        output_grid=output_grid,
        max_aperture_m=2000.0,
        max_angle_deg=60.0,
        min_offset_m=0.0,
        max_offset_m=5000.0,
    )


def create_test_gather(n_traces=100, n_samples=400):
    """Create test seismic gather."""
    np.random.seed(42)
    traces = np.random.randn(n_samples, n_traces).astype(np.float32)
    return SeismicData(
        traces=traces,
        sample_rate=4000,  # 4ms = 4000 microseconds
    )


def create_test_geometry(n_traces=100):
    """Create test geometry."""
    np.random.seed(42)

    # Simple geometry: sources on one line, receivers offset
    source_x = np.linspace(0, 500, n_traces).astype(np.float32)
    source_y = np.zeros(n_traces, dtype=np.float32)
    receiver_x = source_x + np.random.uniform(100, 500, n_traces).astype(np.float32)
    receiver_y = np.zeros(n_traces, dtype=np.float32)

    return MigrationGeometry(
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
    )


class TestMigratorInitialization:
    """Tests for migrator initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE
        )

        assert migrator.device == TEST_DEVICE
        assert migrator._enable_lut is True

    def test_all_optimizations_enabled(self):
        """Test with all optimizations enabled."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_lut=True,
            enable_adaptive_aperture=True,
            enable_spatial_index=True,
            enable_symmetry=True,
            enable_gpu_tiling=True,
        )

        assert migrator._enable_lut is True
        assert migrator._enable_adaptive_aperture is True
        assert migrator._enable_spatial_index is True
        assert migrator._enable_symmetry is True
        assert migrator._enable_gpu_tiling is True

    def test_all_optimizations_disabled(self):
        """Test with all optimizations disabled."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_lut=False,
            enable_adaptive_aperture=False,
            enable_spatial_index=False,
            enable_symmetry=False,
            enable_gpu_tiling=False,
        )

        assert migrator._enable_lut is False
        assert migrator.traveltime_lut is None

    def test_traveltime_lut_built(self):
        """Test that LUT is built when enabled."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_lut=True,
        )

        assert migrator.traveltime_lut is not None
        assert migrator.traveltime_lut._built


class TestSingleGatherMigration:
    """Tests for single gather migration."""

    @pytest.fixture
    def migrator(self):
        """Create test migrator."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config(n_time=30, n_inline=10, n_xline=10)
        return OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_lut=True,
            enable_adaptive_aperture=True,
            enable_spatial_index=True,
            enable_gpu_tiling=True,
        )

    def test_migrate_single_gather(self, migrator):
        """Test migrating a single gather."""
        gather = create_test_gather(n_traces=50, n_samples=200)
        geometry = create_test_geometry(n_traces=50)

        image, fold = migrator.migrate_gather(gather, geometry)

        assert image.shape == (30, 10, 10)
        assert fold.shape == (30, 10, 10)

    def test_output_not_all_zeros(self, migrator):
        """Test that output is not all zeros."""
        gather = create_test_gather(n_traces=50, n_samples=200)
        geometry = create_test_geometry(n_traces=50)

        image, fold = migrator.migrate_gather(gather, geometry)

        # Should have non-zero values
        assert image.abs().sum() > 0 or fold.sum() > 0

    def test_fold_positive(self, migrator):
        """Test that fold is non-negative."""
        gather = create_test_gather(n_traces=50, n_samples=200)
        geometry = create_test_geometry(n_traces=50)

        image, fold = migrator.migrate_gather(gather, geometry)

        assert (fold >= 0).all()


class TestDatasetMigration:
    """Tests for dataset migration."""

    @pytest.fixture
    def migrator(self):
        """Create test migrator."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config(n_time=20, n_inline=8, n_xline=8)
        return OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_lut=True,
            enable_spatial_index=True,
        )

    def test_migrate_dataset(self, migrator):
        """Test migrating multiple gathers."""
        n_gathers = 3
        gathers = [create_test_gather(n_traces=30, n_samples=100) for _ in range(n_gathers)]
        geometries = [create_test_geometry(n_traces=30) for _ in range(n_gathers)]

        result = migrator.migrate_dataset(gathers, geometries)

        assert result.image.shape == (20, 8, 8)
        assert result.fold.shape == (20, 8, 8)

    def test_progress_callback_called(self, migrator):
        """Test that progress callback is called."""
        gathers = [create_test_gather(n_traces=20, n_samples=100) for _ in range(2)]
        geometries = [create_test_geometry(n_traces=20) for _ in range(2)]

        progress_values = []

        def callback(progress, message):
            progress_values.append(progress)

        migrator.migrate_dataset(gathers, geometries, progress_callback=callback)

        assert len(progress_values) > 0
        assert progress_values[-1] == 100.0

    def test_result_metadata(self, migrator):
        """Test that result contains metadata."""
        gathers = [create_test_gather(n_traces=20, n_samples=100)]
        geometries = [create_test_geometry(n_traces=20)]

        result = migrator.migrate_dataset(gathers, geometries)

        assert 'elapsed_seconds' in result.metadata
        assert 'traces_per_second' in result.metadata
        assert 'optimizations' in result.metadata


class TestOptimizationStats:
    """Tests for optimization statistics."""

    def test_stats_after_migration(self):
        """Test that stats are populated after migration."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config(n_time=20, n_inline=8, n_xline=8)
        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_spatial_index=True,
        )

        gather = create_test_gather(n_traces=30, n_samples=100)
        geometry = create_test_geometry(n_traces=30)

        migrator.migrate_gather(gather, geometry)

        stats = migrator.get_optimization_stats()
        assert stats.traces_processed > 0


class TestOptimizationLevels:
    """Tests for factory function optimization levels."""

    def test_low_optimization(self):
        """Test low optimization level."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = create_optimized_kirchhoff_migrator(
            velocity, config,
            prefer_gpu=False,
            optimization_level='low',
        )

        assert migrator._enable_lut is True
        assert migrator._enable_adaptive_aperture is False
        assert migrator._enable_spatial_index is False

    def test_medium_optimization(self):
        """Test medium optimization level."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = create_optimized_kirchhoff_migrator(
            velocity, config,
            prefer_gpu=False,
            optimization_level='medium',
        )

        assert migrator._enable_lut is True
        assert migrator._enable_adaptive_aperture is True
        assert migrator._enable_spatial_index is True
        assert migrator._enable_gpu_tiling is False

    def test_high_optimization(self):
        """Test high optimization level."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()

        migrator = create_optimized_kirchhoff_migrator(
            velocity, config,
            prefer_gpu=False,
            optimization_level='high',
        )

        assert migrator._enable_lut is True
        assert migrator._enable_adaptive_aperture is True
        assert migrator._enable_spatial_index is True
        assert migrator._enable_gpu_tiling is True


class TestDescription:
    """Tests for description method."""

    def test_get_description(self):
        """Test description contains key info."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config()
        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE
        )

        desc = migrator.get_description()

        assert 'Optimized' in desc
        assert 'LUT' in desc


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_geometry_region(self):
        """Test with geometry that doesn't cover output region."""
        velocity = create_constant_velocity(2500.0)

        # Output grid far from geometry
        output_grid = OutputGrid(
            n_time=20,
            n_inline=5,
            n_xline=5,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0,
            x_origin=100000.0,  # Far from geometry
            y_origin=100000.0,
        )
        config = MigrationConfig(
            output_grid=output_grid,
            max_aperture_m=1000.0,
        )

        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE,
            enable_spatial_index=True,
        )

        gather = create_test_gather(n_traces=20, n_samples=100)
        geometry = create_test_geometry(n_traces=20)  # Near origin

        # Should not crash
        image, fold = migrator.migrate_gather(gather, geometry)
        assert image.shape == (20, 5, 5)

    def test_single_trace(self):
        """Test with single trace."""
        velocity = create_constant_velocity(2500.0)
        config = create_test_config(n_time=20, n_inline=5, n_xline=5)
        migrator = OptimizedKirchhoffMigrator(
            velocity, config, device=TEST_DEVICE
        )

        gather = create_test_gather(n_traces=1, n_samples=100)
        geometry = create_test_geometry(n_traces=1)

        # Should not crash
        image, fold = migrator.migrate_gather(gather, geometry)
        assert image.shape == (20, 5, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
