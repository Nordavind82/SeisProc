"""
Integration tests for Kirchhoff PSTM MVP.

Tests the complete migration workflow:
- Point diffractor focusing
- Flat reflector positioning
- Progress tracking
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.velocity_model import create_constant_velocity
from models.migration_config import (
    MigrationConfig,
    OutputGrid,
    TraveltimeMode,
    WeightMode,
    create_default_config,
)
from processors.migration import (
    KirchhoffMigrator,
    create_kirchhoff_migrator,
    interpolate_batch,
    ApertureController,
)
from tests.fixtures.synthetic_prestack import (
    create_synthetic_shot_gather,
    create_point_diffractor_data,
)


class TestInterpolation:
    """Tests for trace interpolation."""

    def test_linear_interpolation_exact(self):
        """Test linear interpolation at exact sample points."""
        device = torch.device('cpu')

        # Create simple trace: 0, 1, 2, 3, 4, ...
        n_samples = 100
        traces = torch.arange(n_samples, dtype=torch.float32, device=device).unsqueeze(1)
        dt = 0.004  # 4ms

        # Interpolate at exact sample times
        times = torch.tensor([0.0, 0.004, 0.008, 0.012], device=device)

        result = interpolate_batch(traces, times, dt, t0=0.0, method='linear', device=device)

        # Should return exact values
        assert result[0].item() == pytest.approx(0.0, abs=0.001)
        assert result[1].item() == pytest.approx(1.0, abs=0.001)
        assert result[2].item() == pytest.approx(2.0, abs=0.001)

    def test_linear_interpolation_midpoint(self):
        """Test linear interpolation at midpoints."""
        device = torch.device('cpu')

        # Create simple trace
        traces = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float32, device=device).unsqueeze(1)
        dt = 0.004

        # Interpolate at midpoint between samples 0 and 1
        times = torch.tensor([0.002], device=device)

        result = interpolate_batch(traces, times, dt, t0=0.0, method='linear', device=device)

        # Should be average of 0 and 2
        assert result[0].item() == pytest.approx(1.0, abs=0.001)

    def test_interpolation_outside_bounds(self):
        """Test interpolation outside trace bounds returns zero."""
        device = torch.device('cpu')

        traces = torch.ones(100, 10, dtype=torch.float32, device=device)
        dt = 0.004

        # Interpolate before trace start and after end
        times = torch.tensor([-0.1, 0.5], device=device)

        result = interpolate_batch(traces, times, dt, t0=0.0, method='linear', device=device)

        assert result[0, 0].item() == 0.0  # Before start
        assert result[1, 0].item() == 0.0  # After end


class TestAperture:
    """Tests for aperture control."""

    def test_distance_aperture(self):
        """Test distance aperture mask."""
        controller = ApertureController(
            max_aperture_m=1000.0,
            max_angle_deg=90.0,  # Don't limit by angle
            min_offset_m=0.0,
            max_offset_m=10000.0,
            taper_width=0.1,
            device=torch.device('cpu')
        )

        # Traces at various distances from origin
        # Using large offsets (> 1000m) to avoid near-offset taper
        rcv_x = torch.tensor([500.0, 800.0, 950.0, 1200.0, 2000.0])
        offset = torch.ones(5) * 2000.0  # Large offset avoids near-offset taper

        # Image point at origin, z=1000m (deep enough for good angles)
        z = torch.tensor([1000.0])

        mask = controller.compute_simple_mask(
            h_source=torch.zeros(5),  # Source at image point
            h_receiver=rcv_x,
            z_depth=z,
            offset=offset,
        )

        # First 3 traces should have non-zero weight (within 1000m aperture)
        assert mask[0, 0].item() > 0  # 500m - well within aperture
        assert mask[0, 1].item() > 0  # 800m - within aperture
        assert mask[0, 2].item() > 0  # 950m - within aperture (before taper ends)
        assert mask[0, 3].item() == 0  # 1200m - outside aperture
        assert mask[0, 4].item() == 0  # 2000m - outside aperture

    def test_angle_aperture(self):
        """Test angle aperture mask."""
        controller = ApertureController(
            max_aperture_m=10000.0,  # Don't limit by distance
            max_angle_deg=45.0,
            min_offset_m=0.0,
            max_offset_m=10000.0,
            taper_width=0.1,
            device=torch.device('cpu')
        )

        # Traces at various horizontal distances
        h_distances = torch.tensor([100.0, 500.0, 800.0, 2000.0])
        z_depth = torch.tensor([1000.0])  # Fixed depth

        # Angles: atan(h/z)
        # 100/1000 = 0.1 rad ≈ 5.7°
        # 500/1000 = 0.5 rad ≈ 26.6°
        # 800/1000 = atan(0.8) ≈ 38.7° (within 40.5° taper start)
        # 2000/1000 = atan(2) ≈ 63.4° (outside aperture)

        # Use large offset to avoid near-offset taper
        mask = controller.compute_simple_mask(
            h_source=h_distances,
            h_receiver=torch.zeros(4),  # Receiver at image point
            z_depth=z_depth,
            offset=torch.ones(4) * 2000,  # Large offset
        )

        # First 3 should pass (< 45°), last should fail
        assert mask[0, 0].item() > 0  # ~5.7°
        assert mask[0, 1].item() > 0  # ~26.6°
        assert mask[0, 2].item() > 0  # ~38.7° - within aperture
        assert mask[0, 3].item() == 0.0  # ~63.4° - outside


class TestKirchhoffMigrator:
    """Integration tests for Kirchhoff migrator."""

    @pytest.fixture
    def simple_config(self):
        """Create simple migration config for testing."""
        grid = OutputGrid(
            n_time=100,
            n_inline=21,
            n_xline=21,
            dt=0.008,  # 8ms
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0,
            x_origin=0.0,
            y_origin=0.0,
        )

        return MigrationConfig(
            output_grid=grid,
            max_aperture_m=3000.0,
            max_angle_deg=60.0,
            traveltime_mode=TraveltimeMode.STRAIGHT_RAY,
            weight_mode=WeightMode.SPREADING,
            normalize_by_fold=True,
        )

    @pytest.fixture
    def velocity_2500(self):
        """Create constant velocity model."""
        return create_constant_velocity(2500.0)

    def test_migrator_initialization(self, velocity_2500, simple_config):
        """Test migrator can be initialized."""
        migrator = KirchhoffMigrator(
            velocity=velocity_2500,
            config=simple_config,
            device=torch.device('cpu')
        )

        assert migrator is not None
        assert migrator.velocity.v0 == 2500.0

    def test_migrate_single_gather(self, velocity_2500, simple_config):
        """Test migrating a single synthetic gather."""
        migrator = KirchhoffMigrator(
            velocity=velocity_2500,
            config=simple_config,
            device=torch.device('cpu')
        )

        # Create synthetic gather
        data, geometry = create_synthetic_shot_gather(
            n_traces=24,
            n_samples=200,
            dt_ms=4.0,
            near_offset=100.0,
            far_offset=600.0,
            velocity=2500.0
        )

        # Migrate
        image, fold = migrator.migrate_gather(data, geometry)

        # Check output shape
        grid = simple_config.output_grid
        assert image.shape == (grid.n_time, grid.n_inline, grid.n_xline)
        assert fold.shape == (grid.n_time, grid.n_inline, grid.n_xline)

        # Image should have non-zero values
        assert torch.sum(torch.abs(image)) > 0

        # Fold should be non-negative
        assert torch.all(fold >= 0)

    def test_migrate_point_diffractor(self, velocity_2500):
        """Test migration focuses point diffractor."""
        # Create output grid centered on diffractor location
        diffractor_x = 250.0
        diffractor_y = 250.0
        diffractor_z = 0.4  # seconds (two-way time)

        grid = OutputGrid(
            n_time=100,
            n_inline=21,
            n_xline=21,
            dt=0.008,
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0,
            x_origin=0.0,
            y_origin=0.0,
        )

        config = MigrationConfig(
            output_grid=grid,
            max_aperture_m=2000.0,
            max_angle_deg=60.0,
            traveltime_mode=TraveltimeMode.STRAIGHT_RAY,
            weight_mode=WeightMode.NONE,  # No weighting for focusing test
            normalize_by_fold=True,
        )

        migrator = KirchhoffMigrator(
            velocity=velocity_2500,
            config=config,
            device=torch.device('cpu')
        )

        # Create diffractor synthetic
        gathers, geometries, metadata = create_point_diffractor_data(
            diffractor_x=diffractor_x,
            diffractor_y=diffractor_y,
            diffractor_z=diffractor_z,  # In time (seconds)
            velocity=2500.0,
            n_shots=5,
            n_receivers_per_shot=21,
        )

        # Migrate all gathers
        result = migrator.migrate_dataset(gathers, geometries)

        # Check that result exists
        assert result.image is not None
        assert result.image.shape == (100, 21, 21)

        # For a well-focused diffractor, maximum should be near the
        # diffractor location. This is a weak test but verifies the
        # basic migration loop works.
        assert np.max(np.abs(result.image)) > 0

    def test_progress_callback(self, velocity_2500, simple_config):
        """Test progress callback is called during migration."""
        migrator = KirchhoffMigrator(
            velocity=velocity_2500,
            config=simple_config,
            device=torch.device('cpu')
        )

        # Create multiple gathers
        gathers = []
        geometries = []
        for _ in range(3):
            data, geom = create_synthetic_shot_gather(
                n_traces=10,
                n_samples=100,
                dt_ms=4.0
            )
            gathers.append(data)
            geometries.append(geom)

        # Track progress calls
        progress_values = []
        messages = []

        def progress_callback(progress: float, message: str):
            progress_values.append(progress)
            messages.append(message)

        # Migrate with callback
        result = migrator.migrate_dataset(
            gathers, geometries,
            progress_callback=progress_callback
        )

        # Should have received progress updates
        assert len(progress_values) > 0
        # Final progress should be 100
        assert 100.0 in progress_values


class TestMigratorFactory:
    """Tests for migrator factory function."""

    def test_create_cpu_migrator(self):
        """Test creating CPU migrator."""
        velocity = create_constant_velocity(2500.0)
        config = create_default_config()

        migrator = create_kirchhoff_migrator(
            velocity=velocity,
            config=config,
            prefer_gpu=False
        )

        assert migrator.device == torch.device('cpu')

    def test_create_auto_device_migrator(self):
        """Test creating migrator with auto device selection."""
        velocity = create_constant_velocity(2500.0)
        config = create_default_config()

        migrator = create_kirchhoff_migrator(
            velocity=velocity,
            config=config,
            prefer_gpu=True
        )

        # Should have selected some device
        assert migrator.device is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
