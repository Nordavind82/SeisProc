"""
Unit tests for traveltime calculators.

Tests:
- Straight ray traveltime with constant velocity
- Straight ray traveltime with v(z)
- Curved ray traveltime with constant gradient
- Curved ray degenerates to straight ray at k->0
- Emergence angle computation
- Batch computation
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.traveltime import (
    TraveltimeCalculator,
    StraightRayTraveltime,
    CurvedRayTraveltime,
    get_traveltime_calculator,
)
from models.velocity_model import (
    VelocityModel,
    create_constant_velocity,
    create_linear_gradient_velocity,
)


# Force CPU for consistent testing across platforms
TEST_DEVICE = torch.device('cpu')


class TestStraightRayConstant:
    """Tests for straight-ray traveltime with constant velocity."""

    @pytest.fixture
    def constant_velocity(self):
        """Create constant velocity model."""
        return create_constant_velocity(2000.0)

    def test_basic_traveltime(self, constant_velocity):
        """Test basic traveltime computation."""
        calc = StraightRayTraveltime(constant_velocity)

        # Vertical ray: t = z / V
        t = calc.compute_traveltime(0.0, 0.0, torch.tensor(1000.0))
        expected = 1000.0 / 2000.0  # 0.5 s
        assert abs(float(t) - expected) < 1e-5

    def test_diagonal_ray(self, constant_velocity):
        """Test traveltime for diagonal ray path."""
        calc = StraightRayTraveltime(constant_velocity)

        # Diagonal ray: t = sqrt(x^2 + y^2 + z^2) / V
        x, y, z = 1000.0, 0.0, 1000.0
        t = calc.compute_traveltime(
            torch.tensor(x), torch.tensor(y), torch.tensor(z)
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        expected = r / 2000.0
        assert abs(float(t) - expected) < 1e-5

    def test_symmetry(self, constant_velocity):
        """Test that traveltime is symmetric in x and y."""
        calc = StraightRayTraveltime(constant_velocity)

        t1 = calc.compute_traveltime(
            torch.tensor(500.0), torch.tensor(300.0), torch.tensor(1000.0)
        )
        t2 = calc.compute_traveltime(
            torch.tensor(-500.0), torch.tensor(300.0), torch.tensor(1000.0)
        )
        t3 = calc.compute_traveltime(
            torch.tensor(500.0), torch.tensor(-300.0), torch.tensor(1000.0)
        )

        assert abs(float(t1) - float(t2)) < 1e-6
        assert abs(float(t1) - float(t3)) < 1e-6

    def test_batch_computation(self, constant_velocity):
        """Test batch traveltime computation."""
        calc = StraightRayTraveltime(constant_velocity)

        surface_x = torch.tensor([0.0, 100.0, 200.0])
        surface_y = torch.tensor([0.0, 0.0, 0.0])
        image_x = torch.tensor([0.0, 500.0])
        image_y = torch.tensor([0.0, 0.0])
        image_z = torch.tensor([500.0, 1000.0, 1500.0])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        # Check shape: (n_z, n_surface, n_image_xy)
        assert t.shape == (3, 3, 2)

        # Check positive values
        assert (t > 0).all()

        # Check that deeper points have longer traveltimes
        for i in range(3):
            for j in range(2):
                assert t[0, i, j] < t[1, i, j] < t[2, i, j]

    def test_total_traveltime(self, constant_velocity):
        """Test source-image-receiver total traveltime."""
        calc = StraightRayTraveltime(constant_velocity)

        source_x = torch.tensor([0.0])
        source_y = torch.tensor([0.0])
        receiver_x = torch.tensor([1000.0])
        receiver_y = torch.tensor([0.0])
        image_x = torch.tensor([500.0])
        image_y = torch.tensor([0.0])
        image_z = torch.tensor([1000.0])

        t_total = calc.compute_total_traveltime(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, image_z
        )

        # Manual calculation
        r_src = np.sqrt(500**2 + 1000**2)
        r_rcv = np.sqrt(500**2 + 1000**2)
        expected = (r_src + r_rcv) / 2000.0

        assert abs(float(t_total) - expected) < 1e-5


class TestStraightRayVz:
    """Tests for straight-ray traveltime with v(z)."""

    @pytest.fixture
    def vz_model(self):
        """Create v(z) velocity model with linear gradient."""
        return create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,  # 500 m/s per second
            z_max=3.0,
            dz=0.004,
            is_time=True,
        )

    def test_velocity_profile_used(self, vz_model):
        """Test that v(z) profile affects traveltimes."""
        calc_vz = StraightRayTraveltime(vz_model)

        # Constant velocity comparison
        v_const = create_constant_velocity(2000.0)
        calc_const = StraightRayTraveltime(v_const)

        z = torch.tensor(2.0)  # 2 seconds depth

        t_vz = calc_vz.compute_traveltime(
            torch.tensor(0.0), torch.tensor(0.0), z
        )
        t_const = calc_const.compute_traveltime(
            torch.tensor(0.0), torch.tensor(0.0), z
        )

        # With positive gradient, effective velocity is higher,
        # so traveltime should be shorter
        assert float(t_vz) < float(t_const)

    def test_batch_with_vz(self, vz_model):
        """Test batch computation with v(z)."""
        calc = StraightRayTraveltime(vz_model, device=TEST_DEVICE)

        surface_x = torch.tensor([0.0, 500.0], device=TEST_DEVICE)
        surface_y = torch.tensor([0.0, 0.0], device=TEST_DEVICE)
        image_x = torch.tensor([0.0], device=TEST_DEVICE)
        image_y = torch.tensor([0.0], device=TEST_DEVICE)
        image_z = torch.tensor([0.5, 1.0, 1.5, 2.0], device=TEST_DEVICE)

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t.shape == (4, 2, 1)
        assert (t > 0).all()


class TestCurvedRay:
    """Tests for curved-ray traveltime with constant gradient."""

    @pytest.fixture
    def gradient_model(self):
        """Create velocity model with gradient."""
        return create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,  # 0.5 /s gradient
            z_max=3.0,
            is_time=True,
        )

    def test_basic_curved_traveltime(self, gradient_model):
        """Test curved ray traveltime computation."""
        calc = CurvedRayTraveltime(gradient_model)

        t = calc.compute_traveltime(
            torch.tensor(500.0), torch.tensor(0.0), torch.tensor(1.0)
        )

        # Should be positive and reasonable
        assert 0 < float(t) < 10

    def test_curved_vs_straight_with_gradient(self, gradient_model):
        """Compare curved and straight ray with same gradient model."""
        calc_curved = CurvedRayTraveltime(gradient_model)
        calc_straight = StraightRayTraveltime(gradient_model)

        z = torch.tensor(2.0)
        x = torch.tensor(1000.0)
        y = torch.tensor(0.0)

        t_curved = calc_curved.compute_traveltime(x, y, z)
        t_straight = calc_straight.compute_traveltime(x, y, z)

        # They should be different with a gradient
        # Curved ray typically gives shorter traveltime for positive gradient
        assert float(t_curved) != float(t_straight)

    def test_straight_ray_limit(self):
        """Test that curved ray degenerates to straight ray at k->0."""
        # Very small gradient (effectively zero)
        model_small_k = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=1e-8,  # Essentially zero
            z_max=3.0,
        )

        # Constant velocity
        model_const = create_constant_velocity(2000.0)

        calc_curved = CurvedRayTraveltime(model_small_k)
        calc_straight = StraightRayTraveltime(model_const)

        x, y, z = torch.tensor(500.0), torch.tensor(300.0), torch.tensor(1.0)

        t_curved = calc_curved.compute_traveltime(x, y, z)
        t_straight = calc_straight.compute_traveltime(x, y, z)

        # Should match within tolerance
        assert abs(float(t_curved) - float(t_straight)) < 0.001

    def test_batch_curved_ray(self, gradient_model):
        """Test batch curved ray computation."""
        calc = CurvedRayTraveltime(gradient_model)

        surface_x = torch.tensor([0.0, 500.0, 1000.0])
        surface_y = torch.tensor([0.0, 0.0, 0.0])
        image_x = torch.tensor([500.0])
        image_y = torch.tensor([0.0])
        image_z = torch.tensor([0.5, 1.0, 1.5])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t.shape == (3, 3, 1)
        assert (t > 0).all()


class TestEmergenceAngle:
    """Tests for emergence angle computation."""

    def test_vertical_ray_angle(self):
        """Vertical ray should have zero emergence angle."""
        model = create_constant_velocity(2000.0)
        calc = StraightRayTraveltime(model)

        angle = calc.compute_emergence_angle(
            torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1000.0)
        )

        assert abs(float(angle)) < 1e-6

    def test_45_degree_angle(self):
        """Test 45 degree emergence angle."""
        model = create_constant_velocity(2000.0)
        calc = StraightRayTraveltime(model)

        # x = z gives 45 degrees
        angle = calc.compute_emergence_angle(
            torch.tensor(1000.0), torch.tensor(0.0), torch.tensor(1000.0)
        )

        expected = np.pi / 4  # 45 degrees
        assert abs(float(angle) - expected) < 0.01

    def test_angle_numpy_input(self):
        """Test emergence angle with numpy arrays."""
        model = create_constant_velocity(2000.0)
        calc = StraightRayTraveltime(model)

        x = np.array([0.0, 500.0, 1000.0])
        y = np.zeros(3)
        z = np.ones(3) * 1000.0

        angles = calc.compute_emergence_angle(x, y, z)

        assert len(angles) == 3
        assert angles[0] < angles[1] < angles[2]  # Increasing angle with offset


class TestTraveltimeFactory:
    """Tests for traveltime calculator factory."""

    def test_factory_auto_constant(self):
        """Factory should select straight ray for constant velocity."""
        model = create_constant_velocity(2000.0)
        calc = get_traveltime_calculator(model, mode='auto')

        assert isinstance(calc, StraightRayTraveltime)

    def test_factory_auto_gradient(self):
        """Factory should select curved ray for velocity with gradient."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=3.0,
        )
        calc = get_traveltime_calculator(model, mode='auto')

        assert isinstance(calc, CurvedRayTraveltime)

    def test_factory_explicit_mode(self):
        """Test explicit mode selection."""
        model = create_constant_velocity(2000.0)

        calc_straight = get_traveltime_calculator(model, mode='straight')
        assert isinstance(calc_straight, StraightRayTraveltime)

        calc_curved = get_traveltime_calculator(model, mode='curved')
        assert isinstance(calc_curved, CurvedRayTraveltime)

    def test_factory_invalid_mode(self):
        """Test invalid mode raises error."""
        model = create_constant_velocity(2000.0)

        with pytest.raises(ValueError, match="Unknown traveltime mode"):
            get_traveltime_calculator(model, mode='invalid')

    def test_factory_with_traveltime_mode_enum(self):
        """Test factory accepts TraveltimeMode enum."""
        from models.migration_config import TraveltimeMode

        model = create_constant_velocity(2000.0)

        # Test with STRAIGHT_RAY enum
        calc_straight = get_traveltime_calculator(model, mode=TraveltimeMode.STRAIGHT_RAY)
        assert isinstance(calc_straight, StraightRayTraveltime)

        # Test with CURVED_RAY enum
        calc_curved = get_traveltime_calculator(model, mode=TraveltimeMode.CURVED_RAY)
        assert isinstance(calc_curved, CurvedRayTraveltime)


class TestPhysicalConsistency:
    """Physical consistency tests for traveltimes."""

    @pytest.fixture
    def model(self):
        return create_constant_velocity(2000.0)

    def test_traveltime_increases_with_distance(self, model):
        """Traveltime should increase with distance."""
        calc = StraightRayTraveltime(model)

        z = torch.tensor(1000.0)
        t0 = calc.compute_traveltime(torch.tensor(0.0), torch.tensor(0.0), z)
        t1 = calc.compute_traveltime(torch.tensor(500.0), torch.tensor(0.0), z)
        t2 = calc.compute_traveltime(torch.tensor(1000.0), torch.tensor(0.0), z)

        assert float(t0) < float(t1) < float(t2)

    def test_traveltime_increases_with_depth(self, model):
        """Traveltime should increase with depth."""
        calc = StraightRayTraveltime(model)

        x = torch.tensor(500.0)
        y = torch.tensor(0.0)

        t1 = calc.compute_traveltime(x, y, torch.tensor(500.0))
        t2 = calc.compute_traveltime(x, y, torch.tensor(1000.0))
        t3 = calc.compute_traveltime(x, y, torch.tensor(1500.0))

        assert float(t1) < float(t2) < float(t3)

    def test_higher_velocity_shorter_time(self):
        """Higher velocity should give shorter traveltime."""
        model_slow = create_constant_velocity(1500.0)
        model_fast = create_constant_velocity(3000.0)

        calc_slow = StraightRayTraveltime(model_slow)
        calc_fast = StraightRayTraveltime(model_fast)

        x, y, z = torch.tensor(500.0), torch.tensor(0.0), torch.tensor(1000.0)

        t_slow = calc_slow.compute_traveltime(x, y, z)
        t_fast = calc_fast.compute_traveltime(x, y, z)

        assert float(t_slow) > float(t_fast)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
