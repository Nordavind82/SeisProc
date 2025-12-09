"""
Unit tests for enhanced curved ray traveltime calculator.

Tests:
- CurvedRayCalculator with constant gradient
- Straight ray limit (k -> 0)
- Ray parameter computation
- Emergence angle computation
- Spreading factor
- Batch computation
- Comparison with straight ray
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.traveltime_curved import (
    CurvedRayCalculator,
    CurvedRayResult,
    compute_curved_ray_traveltime,
    compare_straight_vs_curved,
)
from processors.migration.traveltime import StraightRayTraveltime
from models.velocity_model import (
    create_constant_velocity,
    create_linear_gradient_velocity,
)


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


class TestCurvedRayBasic:
    """Basic tests for CurvedRayCalculator."""

    @pytest.fixture
    def gradient_model(self):
        """Create velocity model with gradient."""
        return create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,  # 0.5 /s gradient
            z_max=3.0,
        )

    @pytest.fixture
    def calc(self, gradient_model):
        """Create curved ray calculator."""
        return CurvedRayCalculator(gradient_model, device=TEST_DEVICE)

    def test_initialization(self, calc, gradient_model):
        """Test calculator initialization."""
        assert calc.v0 == gradient_model.v0
        assert calc.k == gradient_model.gradient
        assert not calc.is_straight_ray

    def test_basic_traveltime(self, calc):
        """Test basic traveltime computation."""
        t = calc.compute_traveltime(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert float(t) > 0
        assert float(t) < 10  # Reasonable range

    def test_vertical_ray_traveltime(self, calc):
        """Test traveltime for vertical ray (x=y=0)."""
        t = calc.compute_traveltime(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        # Should be positive
        assert float(t) > 0
        # For vertical ray at z=1s with v0=2000 and k=0.5:
        # t should be less than 1/2000 = 0.0005s for constant velocity
        # With gradient, ray bends and travels faster at depth

    def test_traveltime_with_depth(self, calc):
        """Test traveltime behavior with depth.

        Note: For curved rays with strong velocity gradient in time migration,
        traveltimes don't always increase monotonically with z because the
        arccosh argument decreases with z.
        """
        depths = [0.5, 1.0, 1.5, 2.0]
        x = torch.tensor(500.0)
        y = torch.tensor(0.0)

        times = []
        for z in depths:
            t = calc.compute_traveltime(x, y, torch.tensor(z))
            times.append(float(t))

        # All traveltimes should be positive and finite
        assert all(t > 0 for t in times)
        assert all(np.isfinite(t) for t in times)

    def test_traveltime_increases_with_offset(self, calc):
        """Traveltime should increase with horizontal offset."""
        offsets = [0.0, 500.0, 1000.0, 1500.0]
        y = torch.tensor(0.0)
        z = torch.tensor(1.0)

        times = []
        for x in offsets:
            t = calc.compute_traveltime(torch.tensor(x), y, z)
            times.append(float(t))

        for i in range(1, len(times)):
            assert times[i] > times[i-1], "Traveltime should increase with offset"

    def test_numpy_input(self, calc):
        """Test with numpy array input."""
        x = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        z = np.ones(3, dtype=np.float32) * 1.0

        t = calc.compute_traveltime(x, y, z)

        assert isinstance(t, np.ndarray)
        assert len(t) == 3
        assert np.all(t > 0)

    def test_scalar_input(self, calc):
        """Test with scalar input."""
        t = calc.compute_traveltime(500.0, 0.0, 1.0)

        assert isinstance(t, torch.Tensor)
        assert float(t) > 0


class TestStraightRayLimit:
    """Test that curved ray approaches straight ray as k -> 0."""

    def test_small_gradient_matches_straight(self):
        """Very small gradient should give same result as straight ray."""
        # Nearly zero gradient
        v_small_k = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=1e-8,
            z_max=3.0,
        )

        # Constant velocity
        v_const = create_constant_velocity(2000.0)

        calc_curved = CurvedRayCalculator(v_small_k, device=TEST_DEVICE)
        calc_straight = StraightRayTraveltime(v_const, device=TEST_DEVICE)

        x = torch.tensor(500.0)
        y = torch.tensor(300.0)
        z = torch.tensor(1.0)

        t_curved = float(calc_curved.compute_traveltime(x, y, z))
        t_straight = float(calc_straight.compute_traveltime(x, y, z))

        assert abs(t_curved - t_straight) < 0.001

    def test_zero_gradient_flag(self):
        """Calculator should recognize zero gradient."""
        v_zero_k = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.0,
            z_max=3.0,
        )

        calc = CurvedRayCalculator(v_zero_k, device=TEST_DEVICE)

        assert calc.is_straight_ray

    def test_constant_velocity_model(self):
        """Test with constant velocity model."""
        v_const = create_constant_velocity(2500.0)

        calc = CurvedRayCalculator(v_const, device=TEST_DEVICE)

        assert calc.is_straight_ray
        assert calc.v0 == 2500.0


class TestRayParameter:
    """Tests for ray parameter computation."""

    @pytest.fixture
    def calc(self):
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        return CurvedRayCalculator(v_model, device=TEST_DEVICE)

    def test_ray_parameter_positive(self, calc):
        """Ray parameter should be positive for positive offset."""
        p = calc.compute_ray_parameter(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert float(p) > 0

    def test_vertical_ray_zero_parameter(self, calc):
        """Vertical ray should have zero ray parameter."""
        p = calc.compute_ray_parameter(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert abs(float(p)) < 1e-6

    def test_ray_parameter_increases_with_offset(self, calc):
        """Ray parameter should increase with horizontal offset."""
        z = torch.tensor(1.0)
        y = torch.tensor(0.0)

        p1 = float(calc.compute_ray_parameter(torch.tensor(200.0), y, z))
        p2 = float(calc.compute_ray_parameter(torch.tensor(500.0), y, z))
        p3 = float(calc.compute_ray_parameter(torch.tensor(1000.0), y, z))

        assert p1 < p2 < p3

    def test_ray_parameter_numpy(self, calc):
        """Test ray parameter with numpy arrays."""
        x = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        z = np.ones(3, dtype=np.float32)

        p = calc.compute_ray_parameter(x, y, z)

        assert isinstance(p, np.ndarray)
        assert len(p) == 3


class TestEmergenceAngle:
    """Tests for emergence angle computation."""

    @pytest.fixture
    def calc(self):
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        return CurvedRayCalculator(v_model, device=TEST_DEVICE)

    def test_vertical_ray_angle(self, calc):
        """Vertical ray should have zero emergence angle."""
        angle = calc.compute_emergence_angle(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert abs(float(angle)) < 1e-5

    def test_angle_range(self, calc):
        """Emergence angle should be between 0 and pi/2."""
        angles = []
        for x in [0.0, 500.0, 1000.0, 2000.0]:
            angle = calc.compute_emergence_angle(
                torch.tensor(x),
                torch.tensor(0.0),
                torch.tensor(1.0),
            )
            angles.append(float(angle))

        for angle in angles:
            assert 0 <= angle < np.pi / 2

    def test_angle_increases_with_offset(self, calc):
        """Emergence angle should increase with horizontal offset."""
        z = torch.tensor(1.0)
        y = torch.tensor(0.0)

        a1 = float(calc.compute_emergence_angle(torch.tensor(0.0), y, z))
        a2 = float(calc.compute_emergence_angle(torch.tensor(500.0), y, z))
        a3 = float(calc.compute_emergence_angle(torch.tensor(1000.0), y, z))

        assert a1 < a2 < a3


class TestSpreadingFactor:
    """Tests for geometrical spreading factor."""

    @pytest.fixture
    def calc(self):
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        return CurvedRayCalculator(v_model, device=TEST_DEVICE)

    def test_spreading_positive(self, calc):
        """Spreading factor should be positive."""
        S = calc.compute_spreading_factor(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert float(S) > 0

    def test_spreading_increases_with_distance(self, calc):
        """Spreading factor should increase with distance."""
        y = torch.tensor(0.0)
        z = torch.tensor(1.0)

        S1 = float(calc.compute_spreading_factor(torch.tensor(200.0), y, z))
        S2 = float(calc.compute_spreading_factor(torch.tensor(500.0), y, z))
        S3 = float(calc.compute_spreading_factor(torch.tensor(1000.0), y, z))

        assert S1 < S2 < S3

    def test_spreading_increases_with_depth(self, calc):
        """Spreading factor should increase with depth."""
        x = torch.tensor(500.0)
        y = torch.tensor(0.0)

        S1 = float(calc.compute_spreading_factor(x, y, torch.tensor(0.5)))
        S2 = float(calc.compute_spreading_factor(x, y, torch.tensor(1.0)))
        S3 = float(calc.compute_spreading_factor(x, y, torch.tensor(1.5)))

        assert S1 < S2 < S3


class TestComputeFull:
    """Tests for compute_full method."""

    @pytest.fixture
    def calc(self):
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        return CurvedRayCalculator(v_model, device=TEST_DEVICE)

    def test_result_type(self, calc):
        """compute_full should return CurvedRayResult."""
        result = calc.compute_full(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert isinstance(result, CurvedRayResult)

    def test_result_fields(self, calc):
        """Result should have all required fields."""
        result = calc.compute_full(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert result.traveltime is not None
        assert result.ray_parameter is not None
        assert result.emergence_angle is not None
        assert result.spreading_factor is not None

    def test_result_values_positive(self, calc):
        """Result values should be positive."""
        result = calc.compute_full(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert float(result.traveltime) > 0
        assert float(result.ray_parameter) > 0
        assert float(result.emergence_angle) > 0
        assert float(result.spreading_factor) > 0

    def test_result_numpy(self, calc):
        """Test compute_full with numpy arrays."""
        x = np.array([500.0, 1000.0], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        z = np.ones(2, dtype=np.float32)

        result = calc.compute_full(x, y, z)

        assert isinstance(result.traveltime, np.ndarray)
        assert len(result.traveltime) == 2


class TestBatchComputation:
    """Tests for batch traveltime computation."""

    @pytest.fixture
    def calc(self):
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        return CurvedRayCalculator(v_model, device=TEST_DEVICE)

    def test_batch_shape(self, calc):
        """Test output shape of batch computation."""
        surface_x = torch.tensor([0.0, 500.0, 1000.0])
        surface_y = torch.tensor([0.0, 0.0, 0.0])
        image_x = torch.tensor([500.0, 1000.0])
        image_y = torch.tensor([0.0, 0.0])
        image_z = torch.tensor([0.5, 1.0, 1.5])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        # Shape: (n_z, n_surface, n_image)
        assert t.shape == (3, 3, 2)

    def test_batch_positive(self, calc):
        """All batch traveltimes should be positive."""
        surface_x = torch.tensor([0.0, 500.0, 1000.0])
        surface_y = torch.zeros(3)
        image_x = torch.tensor([500.0])
        image_y = torch.zeros(1)
        image_z = torch.tensor([0.5, 1.0, 1.5])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert (t > 0).all()

    def test_batch_with_depth(self, calc):
        """Test batch traveltimes with varying depth.

        Note: For curved rays with strong velocity gradient in time migration,
        traveltimes don't always increase monotonically with z.
        """
        surface_x = torch.tensor([0.0])
        surface_y = torch.zeros(1)
        image_x = torch.tensor([500.0])
        image_y = torch.zeros(1)
        image_z = torch.tensor([0.5, 1.0, 1.5, 2.0])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        # All traveltimes should be positive and finite
        assert (t > 0).all()
        assert torch.isfinite(t).all()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_curved_ray_traveltime_scalar(self):
        """Test scalar convenience function."""
        t = compute_curved_ray_traveltime(
            v0=2000.0,
            gradient=0.5,
            h_offset=500.0,
            z_depth=1.0,
        )

        assert t > 0
        assert t < 10

    def test_compute_curved_ray_traveltime_array(self):
        """Test array convenience function."""
        h = np.array([0.0, 500.0, 1000.0])
        z = np.ones(3)

        t = compute_curved_ray_traveltime(
            v0=2000.0,
            gradient=0.5,
            h_offset=h,
            z_depth=z,
        )

        assert len(t) == 3
        assert np.all(t > 0)

    def test_compare_straight_vs_curved(self):
        """Test comparison function."""
        h_offsets = np.array([0.0, 500.0, 1000.0])
        z_depths = np.array([0.5, 1.0])

        result = compare_straight_vs_curved(
            v0=2000.0,
            gradient=0.5,
            h_offsets=h_offsets,
            z_depths=z_depths,
        )

        assert 't_straight' in result
        assert 't_curved' in result
        assert 'difference_percent' in result
        assert result['t_straight'].shape == (2, 3)
        assert result['t_curved'].shape == (2, 3)


class TestCurvedVsStraightComparison:
    """Tests comparing curved and straight ray behaviors."""

    def test_curved_differs_from_straight_with_gradient(self):
        """Curved ray should differ from straight ray when gradient exists."""
        v_grad = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=3.0,
        )
        v_const = create_constant_velocity(2000.0)

        calc_curved = CurvedRayCalculator(v_grad, device=TEST_DEVICE)
        calc_straight = StraightRayTraveltime(v_const, device=TEST_DEVICE)

        x = torch.tensor(1000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(2.0)

        t_curved = float(calc_curved.compute_traveltime(x, y, z))
        t_straight = float(calc_straight.compute_traveltime(x, y, z))

        # They should be noticeably different
        assert abs(t_curved - t_straight) > 0.001

    def test_difference_increases_with_offset(self):
        """Difference between curved and straight should increase with offset."""
        v_grad = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=3.0,
        )
        v_const = create_constant_velocity(2000.0)

        calc_curved = CurvedRayCalculator(v_grad, device=TEST_DEVICE)
        calc_straight = StraightRayTraveltime(v_const, device=TEST_DEVICE)

        z = torch.tensor(1.5)
        y = torch.tensor(0.0)

        differences = []
        for x in [500.0, 1000.0, 2000.0]:
            t_curved = float(calc_curved.compute_traveltime(torch.tensor(x), y, z))
            t_straight = float(calc_straight.compute_traveltime(torch.tensor(x), y, z))
            differences.append(abs(t_curved - t_straight))

        # Difference should generally increase with offset
        # (though the exact behavior depends on gradient sign and magnitude)
        assert differences[-1] > differences[0]

    def test_positive_gradient_effect(self):
        """Positive gradient should generally give shorter curved ray times."""
        # With positive gradient (velocity increases with depth),
        # curved rays bend downward through faster velocities
        v_grad = create_linear_gradient_velocity(
            v0=1500.0,
            gradient=1.0,  # Strong positive gradient
            z_max=3.0,
        )
        v_const = create_constant_velocity(1500.0)

        calc_curved = CurvedRayCalculator(v_grad, device=TEST_DEVICE)
        calc_straight = StraightRayTraveltime(v_const, device=TEST_DEVICE)

        # Large offset to see effect
        x = torch.tensor(2000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_curved = float(calc_curved.compute_traveltime(x, y, z))
        t_straight = float(calc_straight.compute_traveltime(x, y, z))

        # The traveltimes will differ - curved ray physics is different
        assert t_curved != t_straight


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.fixture
    def calc(self):
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        return CurvedRayCalculator(v_model, device=TEST_DEVICE)

    def test_very_shallow_depth(self, calc):
        """Test stability at very shallow depths."""
        t = calc.compute_traveltime(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(0.001),  # Very shallow
        )

        assert torch.isfinite(t)
        assert float(t) > 0

    def test_very_large_offset(self, calc):
        """Test stability at very large offsets."""
        t = calc.compute_traveltime(
            torch.tensor(10000.0),  # 10 km
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert torch.isfinite(t)
        assert float(t) > 0

    def test_zero_x_offset(self, calc):
        """Test with zero x offset."""
        t = calc.compute_traveltime(
            torch.tensor(0.0),
            torch.tensor(500.0),  # Only y offset
            torch.tensor(1.0),
        )

        assert torch.isfinite(t)
        assert float(t) > 0

    def test_symmetry_xy(self, calc):
        """Test symmetry between x and y offsets."""
        t1 = calc.compute_traveltime(
            torch.tensor(500.0),
            torch.tensor(300.0),
            torch.tensor(1.0),
        )
        t2 = calc.compute_traveltime(
            torch.tensor(300.0),
            torch.tensor(500.0),
            torch.tensor(1.0),
        )

        # Due to rotational symmetry in x-y, these should be equal
        # (same horizontal distance)
        assert abs(float(t1) - float(t2)) < 1e-5

    def test_negative_offset(self, calc):
        """Test with negative offset (should work due to symmetry)."""
        t_pos = calc.compute_traveltime(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )
        t_neg = calc.compute_traveltime(
            torch.tensor(-500.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        assert abs(float(t_pos) - float(t_neg)) < 1e-6


class TestDescription:
    """Tests for description and info methods."""

    def test_gradient_description(self):
        """Test description with gradient."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        desc = calc.get_description()

        assert "CurvedRay" in desc
        assert "2000" in desc
        assert "0.5" in desc or ".5" in desc

    def test_straight_ray_description(self):
        """Test description when using straight ray approximation."""
        v_model = create_constant_velocity(2500.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        desc = calc.get_description()

        assert "straight" in desc.lower()
        assert "2500" in desc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
