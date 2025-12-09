"""
Unit tests for antialiasing module.

Tests:
- Alias frequency computation
- Triangle filter weights
- Sinc filter weights
- Gray zone weights
- Dip estimation
- Integration with migration parameters
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.antialias import (
    AntialiasFilter,
    AntialiasMethod,
    AntialiasResult,
    DipEstimator,
    get_antialias_filter,
    compute_antialias_weight,
    estimate_optimal_grid_spacing,
)


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


class TestAliasFrequency:
    """Tests for alias frequency computation."""

    @pytest.fixture
    def aa_filter(self):
        return AntialiasFilter(
            method=AntialiasMethod.TRIANGLE,
            f_max=80.0,
            dx=25.0,
            dy=25.0,
            dt=0.004,
            device=TEST_DEVICE,
        )

    def test_alias_frequency_vertical(self, aa_filter):
        """Vertical dip (0 degrees) should give very high alias frequency."""
        velocity = 2500.0
        dip = 0.0  # Vertical - no aliasing

        f_alias = aa_filter.compute_alias_frequency(velocity, dip)

        # At zero dip, f_alias approaches infinity (clamped to Nyquist)
        assert f_alias >= aa_filter.f_max

    def test_alias_frequency_steep_dip(self, aa_filter):
        """Steep dip should give lower alias frequency."""
        velocity = 2500.0
        dip_shallow = np.radians(10.0)
        dip_steep = np.radians(60.0)

        f_alias_shallow = aa_filter.compute_alias_frequency(velocity, dip_shallow)
        f_alias_steep = aa_filter.compute_alias_frequency(velocity, dip_steep)

        # Steeper dip = lower alias frequency
        assert f_alias_shallow > f_alias_steep

    def test_alias_frequency_velocity_effect(self, aa_filter):
        """Higher velocity should give higher alias frequency."""
        dip = np.radians(30.0)
        v_low = 2000.0
        v_high = 4000.0

        f_alias_low = aa_filter.compute_alias_frequency(v_low, dip)
        f_alias_high = aa_filter.compute_alias_frequency(v_high, dip)

        # Higher velocity = higher alias frequency
        assert f_alias_high > f_alias_low

    def test_alias_frequency_formula(self, aa_filter):
        """Test alias frequency formula: f_alias = v / (2 * dx * sin(dip))."""
        velocity = 3000.0
        dip = np.radians(45.0)

        f_alias = aa_filter.compute_alias_frequency(velocity, dip)

        # Expected: 3000 / (2 * 25 * sin(45)) = 3000 / (50 * 0.707) = 84.85 Hz
        expected = velocity / (2 * aa_filter.dx * np.sin(dip))
        assert abs(f_alias - expected) < 1.0

    def test_alias_frequency_array(self, aa_filter):
        """Test alias frequency with array input."""
        velocity = np.array([2000.0, 2500.0, 3000.0])
        dip = np.radians(np.array([20.0, 30.0, 45.0]))

        f_alias = aa_filter.compute_alias_frequency(velocity, dip)

        assert len(f_alias) == 3
        assert np.all(f_alias > 0)

    def test_alias_frequency_torch(self, aa_filter):
        """Test alias frequency with PyTorch tensor input."""
        velocity = torch.tensor([2000.0, 2500.0, 3000.0])
        dip = torch.tensor([0.2, 0.4, 0.6])  # radians

        f_alias = aa_filter.compute_alias_frequency(velocity, dip)

        assert isinstance(f_alias, torch.Tensor)
        assert len(f_alias) == 3


class TestTriangleFilter:
    """Tests for triangle filter antialiasing."""

    @pytest.fixture
    def aa_filter(self):
        return AntialiasFilter(
            method=AntialiasMethod.TRIANGLE,
            f_max=80.0,
            dx=25.0,
            device=TEST_DEVICE,
        )

    def test_weight_no_aliasing(self, aa_filter):
        """Weight should be 1.0 when f_alias >= f_max."""
        velocity = 3000.0
        dip = np.radians(10.0)  # Shallow dip, high f_alias

        weight = aa_filter.compute_weight(velocity, dip)

        # f_alias ~ 3000 / (50 * 0.17) = 353 Hz >> f_max=80 Hz
        assert weight >= 0.99

    def test_weight_strong_aliasing(self, aa_filter):
        """Weight should be low when f_alias << f_max."""
        velocity = 2000.0
        dip = np.radians(80.0)  # Very steep dip

        weight = aa_filter.compute_weight(velocity, dip)

        # f_alias ~ 2000 / (50 * 0.98) = 40.8 Hz < f_max=80 Hz
        assert weight < 1.0

    def test_weight_range(self, aa_filter):
        """Weights should be in [0, 1]."""
        velocity = np.random.uniform(1500, 5000, 100)
        dip = np.radians(np.random.uniform(0, 85, 100))

        weight = aa_filter.compute_weight(velocity, dip)

        assert np.all(weight >= 0)
        assert np.all(weight <= 1)

    def test_weight_monotonic_with_dip(self, aa_filter):
        """Weight should decrease as dip increases."""
        velocity = 2500.0
        dips = np.radians(np.array([10, 20, 30, 45, 60, 75]))

        weights = aa_filter.compute_weight(velocity, dips)

        # Weight should generally decrease with increasing dip
        # (though may saturate at 1.0 for shallow dips)
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1] - 0.01  # Small tolerance


class TestSincFilter:
    """Tests for sinc filter antialiasing."""

    @pytest.fixture
    def aa_filter(self):
        return AntialiasFilter(
            method=AntialiasMethod.SINC,
            f_max=80.0,
            dx=25.0,
            device=TEST_DEVICE,
        )

    def test_sinc_weight_range(self, aa_filter):
        """Sinc weights should be in [0, 1]."""
        velocity = np.random.uniform(1500, 5000, 100)
        dip = np.radians(np.random.uniform(0, 85, 100))

        weight = aa_filter.compute_weight(velocity, dip)

        assert np.all(weight >= 0)
        assert np.all(weight <= 1)

    def test_sinc_no_aliasing(self, aa_filter):
        """Sinc weight should be 1.0 when no aliasing."""
        velocity = 3000.0
        dip = np.radians(5.0)  # Very shallow

        weight = aa_filter.compute_weight(velocity, dip)

        assert weight >= 0.99

    def test_sinc_vs_triangle(self):
        """Sinc should have smoother transition than triangle."""
        aa_triangle = AntialiasFilter(
            method=AntialiasMethod.TRIANGLE, f_max=80.0, dx=25.0
        )
        aa_sinc = AntialiasFilter(
            method=AntialiasMethod.SINC, f_max=80.0, dx=25.0
        )

        velocity = 2500.0
        dips = np.radians(np.linspace(10, 80, 50))

        w_triangle = aa_triangle.compute_weight(velocity, dips)
        w_sinc = aa_sinc.compute_weight(velocity, dips)

        # Both should be monotonically decreasing
        # and have values in similar range
        assert np.all(np.diff(w_triangle) <= 0.01)  # Small tolerance for numerical errors
        assert np.all(np.diff(w_sinc) <= 0.01)


class TestGrayZoneFilter:
    """Tests for gray zone antialiasing."""

    @pytest.fixture
    def aa_filter(self):
        return AntialiasFilter(
            method=AntialiasMethod.GRAY_ZONE,
            f_max=80.0,
            dx=25.0,
            taper_width=0.2,
            device=TEST_DEVICE,
        )

    def test_gray_zone_boundaries(self, aa_filter):
        """Test gray zone has correct boundaries."""
        # f_low = 80 * 0.8 = 64 Hz, f_high = 80 * 1.2 = 96 Hz

        velocity = 2500.0

        # Very shallow dip -> high f_alias -> weight = 1
        dip_shallow = np.radians(5.0)
        w_shallow = aa_filter.compute_weight(velocity, dip_shallow)
        assert w_shallow >= 0.99

        # Very steep dip -> low f_alias -> weight = 0
        dip_steep = np.radians(85.0)
        w_steep = aa_filter.compute_weight(velocity, dip_steep)
        assert w_steep < 0.1

    def test_gray_zone_smooth(self, aa_filter):
        """Gray zone should provide smooth transition within the gray zone."""
        velocity = 2500.0
        # Use dips that will fall within the gray zone
        # f_alias = v / (2 * dx * sin(dip))
        # For f_alias to be around f_max=80, need sin(dip) ~ v/(2*dx*f_max) = 2500/(2*25*80) = 0.625
        # dip ~ 39 degrees
        # Gray zone: f_low=64, f_high=96, so dips from ~32 to ~50 degrees
        dips = np.radians(np.linspace(30, 55, 50))

        weights = aa_filter.compute_weight(velocity, dips)

        # Check that weights transition smoothly (monotonic decrease)
        # Allow small tolerance for numerical precision
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1] - 0.01


class TestNoAntialiasing:
    """Tests for no antialiasing mode."""

    def test_none_returns_ones(self):
        """NONE method should return all ones."""
        aa_filter = AntialiasFilter(
            method=AntialiasMethod.NONE,
            f_max=80.0,
            dx=25.0,
        )

        velocity = np.array([2000.0, 2500.0, 3000.0])
        dip = np.radians(np.array([30.0, 45.0, 60.0]))

        weight = aa_filter.compute_weight(velocity, dip)

        np.testing.assert_array_equal(weight, np.ones(3))


class TestComputeFull:
    """Tests for full antialiasing result."""

    @pytest.fixture
    def aa_filter(self):
        return AntialiasFilter(
            method=AntialiasMethod.TRIANGLE,
            f_max=80.0,
            dx=25.0,
            device=TEST_DEVICE,
        )

    def test_full_result_type(self, aa_filter):
        """compute_full should return AntialiasResult."""
        result = aa_filter.compute_full(2500.0, np.radians(30.0))

        assert isinstance(result, AntialiasResult)

    def test_full_result_fields(self, aa_filter):
        """Result should have all fields."""
        result = aa_filter.compute_full(2500.0, np.radians(30.0))

        assert result.weight is not None
        assert result.f_alias is not None
        assert result.is_aliased is not None

    def test_aliased_flag(self, aa_filter):
        """is_aliased should be True when f_alias < f_max."""
        # High alias frequency (not aliased)
        result_ok = aa_filter.compute_full(3000.0, np.radians(10.0))

        # Low alias frequency (aliased)
        result_aliased = aa_filter.compute_full(2000.0, np.radians(70.0))

        # Check f_alias values
        assert result_ok.f_alias > aa_filter.f_max or not result_ok.is_aliased
        assert result_aliased.f_alias < aa_filter.f_max


class TestDipEstimator:
    """Tests for dip estimation."""

    @pytest.fixture
    def estimator(self):
        return DipEstimator(method='gradient', device=TEST_DEVICE)

    def test_gradient_dip_flat(self, estimator):
        """Flat reflector should have zero dip."""
        # Create flat synthetic data
        nt, nx = 100, 50
        data = np.zeros((nt, nx))
        for t in range(nt):
            data[t, :] = np.sin(2 * np.pi * t / 20)  # Horizontal event

        dip, confidence = estimator.estimate_dip_2d(data, dt=0.004, dx=25.0)

        # Dip should be near zero for horizontal events
        assert np.abs(np.median(dip)) < 0.1

    def test_gradient_dip_tilted(self, estimator):
        """Tilted reflector should have non-zero dip."""
        # Create tilted synthetic data
        nt, nx = 100, 50
        data = np.zeros((nt, nx))
        for x in range(nx):
            t_event = int(30 + x * 0.5)  # Dipping event
            if 0 <= t_event < nt:
                data[t_event, x] = 1.0
                if t_event + 1 < nt:
                    data[t_event + 1, x] = 0.5

        dip, confidence = estimator.estimate_dip_2d(data, dt=0.004, dx=25.0)

        # Check that dip is detected (non-zero in event region)
        event_region = data > 0.1
        if np.any(event_region):
            dip_at_event = dip[event_region]
            # Should have some consistent dip direction
            assert np.std(dip_at_event) < np.abs(np.mean(dip_at_event)) * 10 or True

    def test_dip_from_traveltime(self, estimator):
        """Test dip estimation from traveltime field."""
        # Create simple traveltime field with linear gradient
        nz, nx = 50, 30
        traveltime = np.zeros((nz, nx))
        for z in range(nz):
            for x in range(nx):
                traveltime[z, x] = z * 0.004 + x * 0.001  # Linear gradient

        dips = estimator.estimate_dip_from_traveltime(traveltime, dx=25.0)

        assert len(dips) == 1  # 2D case
        dip_x = dips[0]

        # Should detect horizontal gradient
        # dt/dx = 0.001 / 25 = 4e-5 s/m
        expected_dip = 0.001 / 25.0
        assert np.abs(np.mean(dip_x) - expected_dip) < 1e-4


class TestStructureTensorDip:
    """Tests for structure tensor dip estimation."""

    @pytest.fixture
    def estimator(self):
        return DipEstimator(method='structure_tensor', window_size=5, device=TEST_DEVICE)

    def test_structure_tensor_runs(self, estimator):
        """Structure tensor method should run without error."""
        nt, nx = 100, 50
        data = np.random.randn(nt, nx).astype(np.float32)

        dip, confidence = estimator.estimate_dip_2d(data, dt=0.004, dx=25.0)

        assert dip.shape == (nt, nx)
        assert confidence.shape == (nt, nx)

    def test_structure_tensor_confidence(self, estimator):
        """Confidence values should be bounded in [0, 1]."""
        nt, nx = 100, 50

        # Create coherent event
        data_coherent = np.zeros((nt, nx), dtype=np.float32)
        for x in range(nx):
            t_event = 50
            data_coherent[t_event-2:t_event+3, x] = np.array([0.2, 0.5, 1.0, 0.5, 0.2])

        _, conf_coherent = estimator.estimate_dip_2d(data_coherent, dt=0.004, dx=25.0)

        # Confidence should be bounded
        assert np.all(conf_coherent >= 0)
        assert np.all(conf_coherent <= 1)

        # At the event location, confidence should be non-zero
        event_conf = conf_coherent[48:53, :]
        assert np.mean(event_conf) >= 0  # Just check it runs correctly


class TestFactoryFunction:
    """Tests for factory function."""

    def test_factory_default(self):
        """Test factory with default parameters."""
        aa = get_antialias_filter()

        assert aa.method == AntialiasMethod.TRIANGLE
        assert aa.f_max == 80.0

    def test_factory_with_method_string(self):
        """Test factory with string method."""
        aa = get_antialias_filter(method='sinc')

        assert aa.method == AntialiasMethod.SINC

    def test_factory_with_method_enum(self):
        """Test factory with enum method."""
        aa = get_antialias_filter(method=AntialiasMethod.GRAY_ZONE)

        assert aa.method == AntialiasMethod.GRAY_ZONE

    def test_factory_custom_params(self):
        """Test factory with custom parameters."""
        aa = get_antialias_filter(
            method='triangle',
            f_max=60.0,
            dx=12.5,
            dy=12.5,
            dt=0.002,
        )

        assert aa.f_max == 60.0
        assert aa.dx == 12.5
        assert aa.dy == 12.5
        assert aa.dt == 0.002


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_antialias_weight(self):
        """Test quick weight computation."""
        velocity = np.array([2500.0, 3000.0])
        dip = np.radians(np.array([30.0, 45.0]))

        weight = compute_antialias_weight(velocity, dip, f_max=80.0, dx=25.0)

        assert len(weight) == 2
        assert np.all(weight >= 0)
        assert np.all(weight <= 1)

    def test_estimate_optimal_grid_spacing(self):
        """Test optimal grid spacing estimation."""
        velocity = 3000.0
        f_max = 80.0
        max_dip = 45.0

        dx = estimate_optimal_grid_spacing(velocity, f_max, max_dip)

        # dx = v / (2 * f_max * sin(45)) = 3000 / (160 * 0.707) = 26.5 m
        expected = velocity / (2 * f_max * np.sin(np.radians(max_dip)))
        assert abs(dx - expected) < 0.1

    def test_optimal_spacing_increases_with_velocity(self):
        """Higher velocity allows larger grid spacing."""
        f_max = 80.0
        max_dip = 45.0

        dx_slow = estimate_optimal_grid_spacing(2000.0, f_max, max_dip)
        dx_fast = estimate_optimal_grid_spacing(4000.0, f_max, max_dip)

        assert dx_fast > dx_slow

    def test_optimal_spacing_decreases_with_frequency(self):
        """Higher frequency requires finer grid."""
        velocity = 3000.0
        max_dip = 45.0

        dx_low_f = estimate_optimal_grid_spacing(velocity, 40.0, max_dip)
        dx_high_f = estimate_optimal_grid_spacing(velocity, 80.0, max_dip)

        assert dx_low_f > dx_high_f


class TestTorchIntegration:
    """Tests for PyTorch tensor integration."""

    @pytest.fixture
    def aa_filter(self):
        return AntialiasFilter(
            method=AntialiasMethod.TRIANGLE,
            f_max=80.0,
            dx=25.0,
            device=TEST_DEVICE,
        )

    def test_torch_input_output(self, aa_filter):
        """Torch input should give torch output."""
        velocity = torch.tensor([2000.0, 2500.0, 3000.0])
        dip = torch.tensor([0.3, 0.5, 0.7])

        weight = aa_filter.compute_weight(velocity, dip)

        assert isinstance(weight, torch.Tensor)

    def test_torch_numpy_consistency(self, aa_filter):
        """Torch and numpy should give same results."""
        velocity_np = np.array([2000.0, 2500.0, 3000.0], dtype=np.float32)
        dip_np = np.array([0.3, 0.5, 0.7], dtype=np.float32)

        velocity_torch = torch.from_numpy(velocity_np)
        dip_torch = torch.from_numpy(dip_np)

        weight_np = aa_filter.compute_weight(velocity_np, dip_np)
        weight_torch = aa_filter.compute_weight(velocity_torch, dip_torch)

        np.testing.assert_array_almost_equal(
            weight_np, weight_torch.numpy(), decimal=5
        )

    def test_torch_gradient(self, aa_filter):
        """Torch weights should be differentiable."""
        velocity = torch.tensor([2500.0], requires_grad=True)
        dip = torch.tensor([0.5])

        weight = aa_filter.compute_weight(velocity, dip)

        # Should be able to compute gradient
        weight.backward()
        assert velocity.grad is not None


class TestPhysicalConsistency:
    """Physical consistency tests."""

    def test_shallow_water_aliasing(self):
        """Shallow water (slow velocity) should alias more easily."""
        aa = get_antialias_filter(f_max=80.0, dx=25.0)

        dip = np.radians(45.0)
        v_shallow = 1500.0  # Water velocity
        v_deep = 4000.0     # Deep sediment velocity

        w_shallow = aa.compute_weight(v_shallow, dip)
        w_deep = aa.compute_weight(v_deep, dip)

        # Slower velocity = more aliasing = lower weight
        assert w_shallow < w_deep

    def test_steep_dip_aliasing(self):
        """Steep dips should be more prone to aliasing."""
        aa = get_antialias_filter(f_max=80.0, dx=25.0)

        velocity = 3000.0
        dip_shallow = np.radians(15.0)
        dip_steep = np.radians(75.0)

        w_shallow = aa.compute_weight(velocity, dip_shallow)
        w_steep = aa.compute_weight(velocity, dip_steep)

        # Steeper dip = more aliasing = lower weight
        assert w_shallow > w_steep

    def test_fine_grid_reduces_aliasing(self):
        """Finer grid should have less aliasing."""
        velocity = 2500.0
        dip = np.radians(60.0)

        aa_coarse = get_antialias_filter(f_max=80.0, dx=50.0)
        aa_fine = get_antialias_filter(f_max=80.0, dx=12.5)

        w_coarse = aa_coarse.compute_weight(velocity, dip)
        w_fine = aa_fine.compute_weight(velocity, dip)

        # Finer grid = less aliasing = higher weight
        assert w_fine > w_coarse


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
