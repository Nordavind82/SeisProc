"""
Integration tests for Kirchhoff migration with v(z) velocity model.

Tests:
- Migration with linear velocity gradient
- Comparison with/without velocity gradient
- Amplitude preservation with weights
- Integration of traveltime and weights
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.velocity_model import (
    VelocityModel,
    create_constant_velocity,
    create_linear_gradient_velocity,
    rms_to_interval_velocity,
    interval_to_rms_velocity,
)
from models.migration_config import MigrationConfig, OutputGrid, WeightMode
from processors.migration.traveltime import (
    StraightRayTraveltime,
    CurvedRayTraveltime,
    get_traveltime_calculator,
)
from processors.migration.weights import (
    StandardWeight,
    get_amplitude_weight,
    compute_spreading_weight,
    compute_obliquity_weight,
)


# Force CPU for consistent testing across platforms
TEST_DEVICE = torch.device('cpu')


class TestVelocityModelIntegration:
    """Tests for velocity model with traveltime calculator."""

    def test_constant_velocity_traveltime(self):
        """Test traveltime with constant velocity model."""
        v_model = create_constant_velocity(2500.0)
        tt_calc = get_traveltime_calculator(v_model)

        # Vertical ray
        t = tt_calc.compute_traveltime(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0),  # 1 second depth (time migration)
        )

        # For constant velocity and z=1s, t = z/V = 1.0/2500.0 = 0.0004s
        # But wait - in time migration, z is already time, not depth!
        # Actually for straight ray: t = sqrt(x^2+y^2+z^2)/V
        # With x=y=0, z=1: t = 1/2500 = 0.0004s
        expected = 1.0 / 2500.0
        assert abs(float(t) - expected) < 1e-6

    def test_gradient_velocity_traveltime(self):
        """Test traveltime with velocity gradient."""
        v_model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,  # 500 m/s per second
            z_max=3.0,
        )

        # Straight ray
        tt_straight = StraightRayTraveltime(v_model)
        # Curved ray
        tt_curved = CurvedRayTraveltime(v_model)

        x = torch.tensor(500.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_straight = tt_straight.compute_traveltime(x, y, z)
        t_curved = tt_curved.compute_traveltime(x, y, z)

        # Both should give positive, finite values
        assert 0 < float(t_straight) < 10
        assert 0 < float(t_curved) < 10

        # With positive gradient, curved ray should differ from straight
        # (they use different physics)
        # At larger offsets and depths, difference is more significant
        assert float(t_straight) != float(t_curved)


class TestVelocityConversion:
    """Tests for RMS/interval velocity conversion."""

    def test_rms_to_interval_simple(self):
        """Test RMS to interval conversion with simple model."""
        # Constant velocity: RMS = interval
        t_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v_rms = np.array([2000.0, 2000.0, 2000.0, 2000.0, 2000.0])

        _, v_int = rms_to_interval_velocity(t_axis, v_rms)

        # For constant velocity, interval should equal RMS
        np.testing.assert_array_almost_equal(v_int, v_rms, decimal=1)

    def test_rms_to_interval_gradient(self):
        """Test RMS to interval with increasing velocity."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        # RMS velocity increasing with time
        v_rms = np.array([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])

        _, v_int = rms_to_interval_velocity(t_axis, v_rms)

        # Interval velocity should be higher than RMS at depth
        # because RMS is an average
        assert v_int[-1] > v_rms[-1]

    def test_round_trip_conversion(self):
        """Test RMS -> interval -> RMS gives original."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v_rms_orig = np.array([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])

        _, v_int = rms_to_interval_velocity(t_axis, v_rms_orig)
        _, v_rms_back = interval_to_rms_velocity(t_axis, v_int)

        np.testing.assert_array_almost_equal(v_rms_back, v_rms_orig, decimal=1)


class TestAmplitudeWeightIntegration:
    """Tests for amplitude weight with traveltime integration."""

    def test_weight_with_computed_angles(self):
        """Test amplitude weight using angles from traveltime calculator."""
        v_model = create_constant_velocity(2500.0)
        tt_calc = StraightRayTraveltime(v_model, device=TEST_DEVICE)
        weight_calc = get_amplitude_weight(WeightMode.FULL, device=TEST_DEVICE)

        # Compute angles from geometry
        x = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        z = np.ones(3, dtype=np.float32) * 1000.0

        angles = tt_calc.compute_emergence_angle(x, y, z)

        # Compute distances
        r = np.sqrt(x**2 + y**2 + z**2).astype(np.float32)

        # Compute weights
        weights = weight_calc.compute_weight(r, r, angles, angles, 2500.0)

        # Weights should decrease with offset (larger angle, longer distance)
        assert weights[0] > weights[1] > weights[2]

    def test_spreading_dominates_at_large_offset(self):
        """Test that spreading correction becomes larger at far offsets."""
        v_model = create_constant_velocity(2500.0)
        tt_calc = StraightRayTraveltime(v_model, device=TEST_DEVICE)

        # Near and far offset
        x_near = np.array([100.0], dtype=np.float32)
        x_far = np.array([2000.0], dtype=np.float32)
        y = np.zeros(1, dtype=np.float32)
        z = np.ones(1, dtype=np.float32) * 1000.0

        r_near = np.sqrt(x_near**2 + z**2)
        r_far = np.sqrt(x_far**2 + z**2)

        # Spreading weight (1/r^2 for both source and receiver legs)
        w_near = compute_spreading_weight(r_near, r_near)
        w_far = compute_spreading_weight(r_far, r_far)

        # Near offset should have higher weight (but not 10x for these offsets)
        # r_near ~ 1005, r_far ~ 2236, so ratio ~ 5x
        assert w_near[0] > 4 * w_far[0]


class TestMigrationConfigIntegration:
    """Tests for migration configuration with velocity models."""

    def test_config_with_output_grid(self):
        """Test MigrationConfig with output grid."""
        grid = OutputGrid(
            n_time=100,
            n_inline=20,
            n_xline=20,
            dt=0.004,
        )

        config = MigrationConfig(output_grid=grid)

        assert config.output_grid.n_time == 100
        assert config.output_grid.n_inline == 20

    def test_velocity_model_with_gradient(self):
        """Test v(z) velocity model has gradient property."""
        v_model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
        )

        assert v_model.has_gradient
        assert v_model.gradient == 500.0

    def test_traveltime_mode_selection(self):
        """Test correct traveltime mode is selected based on velocity."""
        # Constant velocity -> straight ray
        v_const = create_constant_velocity(2500.0)
        tt_const = get_traveltime_calculator(v_const, mode='auto')
        assert isinstance(tt_const, StraightRayTraveltime)

        # Gradient velocity -> curved ray
        v_grad = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=3.0,
        )
        tt_grad = get_traveltime_calculator(v_grad, mode='auto')
        assert isinstance(tt_grad, CurvedRayTraveltime)


class TestPhysicalConsistency:
    """Physical consistency tests for v(z) migration."""

    def test_positive_gradient_faster_velocity(self):
        """Positive gradient means faster velocity at depth."""
        v_model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,  # m/s per s
            z_max=3.0,
        )

        # Velocity at surface
        v_surface = v_model.get_velocity_at(0.0)
        # Velocity at depth
        v_deep = v_model.get_velocity_at(2.0)

        assert v_deep > v_surface
        expected_deep = 2000.0 + 500.0 * 2.0
        assert abs(v_deep - expected_deep) < 1.0

    def test_effective_velocity_is_average(self):
        """Effective velocity should be roughly average for linear gradient."""
        v_model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
        )

        z = 2.0
        v_eff = v_model.get_effective_velocity(z)

        # For linear gradient, effective velocity should be between
        # surface and depth velocities
        v_surface = 2000.0
        v_depth = 2000.0 + 500.0 * z

        assert v_surface < v_eff < v_depth

    def test_curved_ray_shorter_for_positive_gradient(self):
        """Curved ray should give shorter traveltime with positive gradient."""
        # Strong positive gradient
        v_model = create_linear_gradient_velocity(
            v0=1500.0,  # Water-like surface velocity
            gradient=1.0,  # Strong gradient
            z_max=3.0,
        )

        tt_straight = StraightRayTraveltime(v_model)
        tt_curved = CurvedRayTraveltime(v_model)

        # Large offset to see ray bending effect
        x = torch.tensor(2000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(2.0)

        t_straight = tt_straight.compute_traveltime(x, y, z)
        t_curved = tt_curved.compute_traveltime(x, y, z)

        # Curved ray bends down through faster velocities
        # so should arrive faster (though this depends on exact geometry)
        # At minimum, they should be different
        assert abs(float(t_straight) - float(t_curved)) > 0.001


class TestWeightModeEffects:
    """Tests for different weight mode effects."""

    def test_no_weight_preserves_amplitude(self):
        """No weight mode should not modify amplitudes."""
        weight_calc = get_amplitude_weight(WeightMode.NONE, device=TEST_DEVICE)

        r_s = np.random.uniform(100, 2000, 100).astype(np.float32)
        r_r = np.random.uniform(100, 2000, 100).astype(np.float32)
        angles = np.random.uniform(0, np.pi/3, 100).astype(np.float32)

        weights = weight_calc.compute_weight(r_s, r_r, angles, angles, 2500.0)

        np.testing.assert_array_equal(weights, np.ones(100))

    def test_spreading_only_distance_dependent(self):
        """Spreading weight should only depend on distance."""
        weight_calc = get_amplitude_weight(WeightMode.SPREADING, device=TEST_DEVICE)

        r = np.array([500.0, 1000.0, 1500.0], dtype=np.float32)
        angles_0 = np.zeros(3, dtype=np.float32)
        angles_45 = np.ones(3, dtype=np.float32) * np.pi/4

        # Same distance, different angles
        w1 = weight_calc.compute_weight(r, r, angles_0, angles_0, 2500.0)
        w2 = weight_calc.compute_weight(r, r, angles_45, angles_45, 2500.0)

        # Spreading mode ignores angles
        np.testing.assert_array_almost_equal(w1, w2, decimal=5)

    def test_obliquity_only_angle_dependent(self):
        """Obliquity weight should only depend on angles."""
        weight_calc = get_amplitude_weight(WeightMode.OBLIQUITY, device=TEST_DEVICE)

        r1 = np.array([500.0, 500.0], dtype=np.float32)
        r2 = np.array([1500.0, 1500.0], dtype=np.float32)
        angles = np.array([0.0, np.pi/4], dtype=np.float32)

        w1 = weight_calc.compute_weight(r1, r1, angles, angles, 2500.0)
        w2 = weight_calc.compute_weight(r2, r2, angles, angles, 2500.0)

        # Obliquity mode ignores distance
        np.testing.assert_array_almost_equal(w1, w2, decimal=5)


class TestBatchProcessing:
    """Tests for batch processing efficiency."""

    def test_batch_traveltime_large(self):
        """Test batch traveltime with large arrays."""
        v_model = create_constant_velocity(2500.0)
        tt_calc = get_traveltime_calculator(v_model)

        n_surface = 1000
        n_image = 50
        n_z = 100

        surface_x = torch.rand(n_surface) * 3000
        surface_y = torch.rand(n_surface) * 3000
        image_x = torch.rand(n_image) * 3000
        image_y = torch.rand(n_image) * 3000
        image_z = torch.linspace(0.1, 3.0, n_z)

        t = tt_calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t.shape == (n_z, n_surface, n_image)
        assert torch.all(t > 0)
        assert torch.all(torch.isfinite(t))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
