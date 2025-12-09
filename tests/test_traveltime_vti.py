"""
Unit tests for VTI traveltime calculator.

Tests:
- Isotropic limit (epsilon=delta=0)
- VTI moveout curves
- Anelliptic approximation accuracy
- Comparison with isotropic traveltimes
- Batch computation
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.traveltime_vti import (
    VTITraveltimeCalculator,
    VTITraveltimeResult,
    get_vti_traveltime_calculator,
    compute_vti_traveltime,
    compare_isotropic_vs_vti,
)
from models.velocity_model import create_constant_velocity
from models.anisotropy_model import (
    AnisotropyModel,
    create_isotropic,
    create_constant_anisotropy,
    create_shale_anisotropy,
)


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


class TestIsotropicLimit:
    """Tests for isotropic limit (epsilon=delta=0)."""

    @pytest.fixture
    def v_model(self):
        return create_constant_velocity(2500.0)

    @pytest.fixture
    def iso_aniso(self):
        return create_isotropic()

    def test_isotropic_detection(self, v_model, iso_aniso):
        """Test that isotropic model is detected."""
        calc = VTITraveltimeCalculator(v_model, iso_aniso, device=TEST_DEVICE)

        assert calc.is_isotropic

    def test_isotropic_matches_straight_ray(self, v_model, iso_aniso):
        """VTI with zero anisotropy should match straight ray."""
        calc = VTITraveltimeCalculator(v_model, iso_aniso, device=TEST_DEVICE)

        x, y, z = 500.0, 0.0, 1000.0
        t_vti = calc.compute_traveltime(
            torch.tensor(x),
            torch.tensor(y),
            torch.tensor(z),
        )

        # Straight ray: t = r / v
        r = np.sqrt(x**2 + y**2 + z**2)
        t_expected = r / 2500.0

        assert abs(float(t_vti) - t_expected) < 1e-6

    def test_no_anisotropy_model(self, v_model):
        """Test with no anisotropy model (None)."""
        calc = VTITraveltimeCalculator(v_model, None, device=TEST_DEVICE)

        assert calc.is_isotropic

        t = calc.compute_traveltime(
            torch.tensor(500.0),
            torch.tensor(0.0),
            torch.tensor(1000.0),
        )

        assert float(t) > 0


class TestVTIMoveout:
    """Tests for VTI moveout curves."""

    @pytest.fixture
    def v_model(self):
        return create_constant_velocity(2500.0)

    @pytest.fixture
    def shale_aniso(self):
        return create_shale_anisotropy(intensity='moderate')

    def test_vti_traveltime_positive(self, v_model, shale_aniso):
        """VTI traveltimes should be positive."""
        calc = VTITraveltimeCalculator(v_model, shale_aniso, device=TEST_DEVICE)

        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert float(t) > 0

    def test_vti_moveout_differs_from_isotropic(self, v_model, shale_aniso):
        """VTI moveout should differ from isotropic."""
        calc_vti = VTITraveltimeCalculator(v_model, shale_aniso, device=TEST_DEVICE)
        calc_iso = VTITraveltimeCalculator(v_model, None, device=TEST_DEVICE)

        x = torch.tensor(2000.0)  # Far offset
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_vti = float(calc_vti.compute_traveltime(x, y, z))
        t_iso = float(calc_iso.compute_traveltime(x, y, z))

        # They should differ by some amount
        assert abs(t_vti - t_iso) > 0.001

    def test_vti_zero_offset(self, v_model, shale_aniso):
        """At zero offset, VTI should equal isotropic."""
        calc = VTITraveltimeCalculator(v_model, shale_aniso, device=TEST_DEVICE)

        # Zero offset = vertical ray
        t = calc.compute_traveltime(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(1.0),
        )

        # Vertical ray: t = z / v0
        t_expected = 1.0 / 2500.0
        assert abs(float(t) - t_expected) < 1e-6

    def test_moveout_curve_shape(self, v_model, shale_aniso):
        """Test that moveout curve has expected shape."""
        calc = VTITraveltimeCalculator(v_model, shale_aniso, device=TEST_DEVICE)

        offsets = [0.0, 500.0, 1000.0, 1500.0, 2000.0]
        z = torch.tensor(1.5)
        y = torch.tensor(0.0)

        times = []
        for x in offsets:
            t = calc.compute_traveltime(torch.tensor(x), y, z)
            times.append(float(t))

        # Traveltime should increase with offset
        for i in range(1, len(times)):
            assert times[i] > times[i-1]


class TestAnellipticApproximation:
    """Tests for anelliptic approximation."""

    @pytest.fixture
    def v_model(self):
        return create_constant_velocity(2500.0)

    def test_anelliptic_method(self, v_model):
        """Test anelliptic method specifically."""
        aniso = create_constant_anisotropy(0.2, 0.1)
        calc = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='anelliptic'
        )

        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert float(t) > 0
        assert np.isfinite(float(t))

    def test_weak_method(self, v_model):
        """Test weak anisotropy method."""
        aniso = create_constant_anisotropy(0.1, 0.05)  # Weak anisotropy
        calc = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='weak'
        )

        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert float(t) > 0

    def test_exact_method(self, v_model):
        """Test exact method."""
        aniso = create_constant_anisotropy(0.2, 0.1)
        calc = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='exact'
        )

        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert float(t) > 0

    def test_methods_agree_near_vertical(self, v_model):
        """Different methods should agree near vertical."""
        aniso = create_constant_anisotropy(0.2, 0.1)

        calc_anell = VTITraveltimeCalculator(v_model, aniso, TEST_DEVICE, 'anelliptic')
        calc_weak = VTITraveltimeCalculator(v_model, aniso, TEST_DEVICE, 'weak')
        calc_exact = VTITraveltimeCalculator(v_model, aniso, TEST_DEVICE, 'exact')

        # Small offset (near vertical)
        x = torch.tensor(100.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_anell = float(calc_anell.compute_traveltime(x, y, z))
        t_weak = float(calc_weak.compute_traveltime(x, y, z))
        t_exact = float(calc_exact.compute_traveltime(x, y, z))

        # Should be within 10% for near-vertical (methods have different approximations)
        assert abs(t_anell - t_weak) / t_anell < 0.1
        assert abs(t_anell - t_exact) / t_anell < 0.1


class TestNumpyInput:
    """Tests for numpy array input."""

    @pytest.fixture
    def calc(self):
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')
        return VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)

    def test_numpy_arrays(self, calc):
        """Test with numpy array input."""
        x = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        z = np.ones(3, dtype=np.float32) * 1.5

        t = calc.compute_traveltime(x, y, z)

        assert isinstance(t, (np.ndarray, float))
        if isinstance(t, np.ndarray):
            assert len(t) == 3
            assert np.all(t > 0)


class TestBatchComputation:
    """Tests for batch computation."""

    @pytest.fixture
    def calc(self):
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')
        return VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)

    def test_batch_shape(self, calc):
        """Test batch output shape."""
        surface_x = torch.tensor([0.0, 500.0, 1000.0])
        surface_y = torch.zeros(3)
        image_x = torch.tensor([500.0, 1000.0])
        image_y = torch.zeros(2)
        image_z = torch.tensor([0.5, 1.0, 1.5])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        # Shape: (n_z, n_surface, n_image)
        assert t.shape == (3, 3, 2)

    def test_batch_positive(self, calc):
        """All batch traveltimes should be positive."""
        surface_x = torch.tensor([0.0, 500.0])
        surface_y = torch.zeros(2)
        image_x = torch.tensor([500.0])
        image_y = torch.zeros(1)
        image_z = torch.tensor([0.5, 1.0, 1.5])

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert (t > 0).all()
        assert torch.isfinite(t).all()


class TestComputeFull:
    """Tests for compute_full method."""

    @pytest.fixture
    def calc(self):
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')
        return VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)

    def test_result_type(self, calc):
        """compute_full should return VTITraveltimeResult."""
        result = calc.compute_full(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert isinstance(result, VTITraveltimeResult)

    def test_result_fields(self, calc):
        """Result should have all required fields."""
        result = calc.compute_full(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert result.traveltime is not None
        assert result.phase_angle is not None
        assert result.anisotropy_correction is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_vti_traveltime_scalar(self):
        """Test scalar convenience function."""
        t = compute_vti_traveltime(
            v0=2500.0,
            epsilon=0.2,
            delta=0.1,
            h_offset=1000.0,
            z_depth=1.5,
        )

        assert t > 0
        assert np.isfinite(t)

    def test_compute_vti_traveltime_array(self):
        """Test array convenience function."""
        h = np.array([0.0, 500.0, 1000.0])
        z = np.ones(3) * 1.5

        t = compute_vti_traveltime(
            v0=2500.0,
            epsilon=0.2,
            delta=0.1,
            h_offset=h,
            z_depth=z,
        )

        assert len(t) == 3
        assert np.all(t > 0)

    def test_compare_isotropic_vs_vti(self):
        """Test comparison function."""
        h_offsets = np.array([0.0, 500.0, 1000.0, 1500.0])
        z_depths = np.array([0.5, 1.0, 1.5])

        result = compare_isotropic_vs_vti(
            v0=2500.0,
            epsilon=0.2,
            delta=0.1,
            h_offsets=h_offsets,
            z_depths=z_depths,
        )

        assert 't_isotropic' in result
        assert 't_vti' in result
        assert 'difference_percent' in result
        assert result['t_isotropic'].shape == (3, 4)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_factory_default(self):
        """Test factory with default method."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')

        calc = get_vti_traveltime_calculator(v_model, aniso)

        assert calc.method == 'anelliptic'

    def test_factory_with_method(self):
        """Test factory with specified method."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')

        calc = get_vti_traveltime_calculator(v_model, aniso, method='weak')

        assert calc.method == 'weak'


class TestPhysicalConsistency:
    """Physical consistency tests."""

    @pytest.fixture
    def v_model(self):
        return create_constant_velocity(2500.0)

    def test_eta_effect_far_offset(self, v_model):
        """Larger eta should have more effect at far offsets."""
        aniso_small_eta = create_constant_anisotropy(0.15, 0.1)  # eta ~ 0.04
        aniso_large_eta = create_constant_anisotropy(0.3, 0.1)   # eta ~ 0.17

        calc_small = VTITraveltimeCalculator(v_model, aniso_small_eta, TEST_DEVICE)
        calc_large = VTITraveltimeCalculator(v_model, aniso_large_eta, TEST_DEVICE)

        # Far offset
        x = torch.tensor(3000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_small = float(calc_small.compute_traveltime(x, y, z))
        t_large = float(calc_large.compute_traveltime(x, y, z))

        # Larger eta should give different traveltime
        assert t_small != t_large

    def test_delta_affects_nmo(self, v_model):
        """Delta parameter affects NMO velocity."""
        aniso1 = create_constant_anisotropy(0.2, 0.05)  # Small delta
        aniso2 = create_constant_anisotropy(0.2, 0.15)  # Large delta

        calc1 = VTITraveltimeCalculator(v_model, aniso1, TEST_DEVICE)
        calc2 = VTITraveltimeCalculator(v_model, aniso2, TEST_DEVICE)

        # Near offset (NMO region)
        x = torch.tensor(500.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t1 = float(calc1.compute_traveltime(x, y, z))
        t2 = float(calc2.compute_traveltime(x, y, z))

        # Larger delta = higher NMO velocity = shorter traveltime
        assert t2 < t1


class TestDescription:
    """Tests for description method."""

    def test_isotropic_description(self):
        """Test description for isotropic."""
        v_model = create_constant_velocity(2500.0)
        calc = VTITraveltimeCalculator(v_model, None, TEST_DEVICE)

        desc = calc.get_description()

        assert 'isotropic' in desc
        assert '2500' in desc

    def test_vti_description(self):
        """Test description for VTI."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')
        calc = VTITraveltimeCalculator(v_model, aniso, TEST_DEVICE)

        desc = calc.get_description()

        assert 'VTI' in desc
        assert 'eps' in desc or 'epsilon' in desc.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
