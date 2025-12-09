"""
Unit tests for amplitude weights.

Tests:
- Geometrical spreading weight
- Obliquity factor
- Combined weights
- Aperture taper
- True-amplitude weights
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.weights import (
    AmplitudeWeight,
    StandardWeight,
    TrueAmplitudeWeight,
    get_amplitude_weight,
    compute_spreading_weight,
    compute_obliquity_weight,
    compute_aperture_mask,
)
from models.migration_config import WeightMode


# Force CPU for consistent testing across platforms
TEST_DEVICE = torch.device('cpu')


class TestStandardWeight:
    """Tests for StandardWeight class."""

    def test_no_weight_mode(self):
        """Test NONE mode returns ones."""
        weight = StandardWeight(WeightMode.NONE)

        r_s = np.array([100.0, 500.0, 1000.0])
        r_r = np.array([200.0, 400.0, 800.0])
        angles_s = np.zeros(3)
        angles_r = np.zeros(3)

        w = weight.compute_weight(r_s, r_r, angles_s, angles_r, 2000.0)

        np.testing.assert_array_almost_equal(w, np.ones(3))

    def test_spreading_weight(self):
        """Test geometrical spreading weight."""
        weight = StandardWeight(WeightMode.SPREADING, device=TEST_DEVICE)

        r_s = np.array([100.0, 200.0], dtype=np.float32)
        r_r = np.array([100.0, 200.0], dtype=np.float32)
        angles_s = np.zeros(2, dtype=np.float32)
        angles_r = np.zeros(2, dtype=np.float32)

        w = weight.compute_weight(r_s, r_r, angles_s, angles_r, 2000.0)

        # w = 1 / (r_s * r_r)
        expected = 1.0 / (r_s * r_r)
        np.testing.assert_array_almost_equal(w, expected)

    def test_spreading_decreases_with_distance(self):
        """Spreading weight should decrease with distance."""
        weight = StandardWeight(WeightMode.SPREADING, device=TEST_DEVICE)

        r_s = np.array([100.0, 500.0, 1000.0], dtype=np.float32)
        r_r = np.array([100.0, 500.0, 1000.0], dtype=np.float32)
        angles = np.zeros(3, dtype=np.float32)

        w = weight.compute_weight(r_s, r_r, angles, angles, 2000.0)

        assert w[0] > w[1] > w[2]

    def test_obliquity_weight(self):
        """Test obliquity factor."""
        weight = StandardWeight(WeightMode.OBLIQUITY, device=TEST_DEVICE)

        r_s = np.ones(3, dtype=np.float32) * 100.0
        r_r = np.ones(3, dtype=np.float32) * 100.0
        angles_s = np.array([0.0, np.pi/6, np.pi/4], dtype=np.float32)
        angles_r = np.array([0.0, np.pi/6, np.pi/4], dtype=np.float32)

        w = weight.compute_weight(r_s, r_r, angles_s, angles_r, 2000.0)

        # w = cos(theta_s) * cos(theta_r)
        expected = np.cos(angles_s) * np.cos(angles_r)
        np.testing.assert_array_almost_equal(w, expected, decimal=5)

    def test_obliquity_decreases_with_angle(self):
        """Obliquity weight should decrease with angle from vertical."""
        weight = StandardWeight(WeightMode.OBLIQUITY, device=TEST_DEVICE)

        r = np.ones(4, dtype=np.float32) * 100.0
        angles = np.array([0.0, np.pi/6, np.pi/4, np.pi/3], dtype=np.float32)

        w = weight.compute_weight(r, r, angles, angles, 2000.0)

        assert w[0] > w[1] > w[2] > w[3]

    def test_full_weight(self):
        """Test full weight mode combines all factors."""
        weight = StandardWeight(WeightMode.FULL, device=TEST_DEVICE)

        r_s = np.array([100.0, 200.0], dtype=np.float32)
        r_r = np.array([100.0, 200.0], dtype=np.float32)
        angles_s = np.array([0.0, np.pi/6], dtype=np.float32)
        angles_r = np.array([0.0, np.pi/6], dtype=np.float32)
        v = 2000.0

        w = weight.compute_weight(r_s, r_r, angles_s, angles_r, v)

        # w = cos(theta_s) * cos(theta_r) / (r_s * r_r * V)
        expected = (np.cos(angles_s) * np.cos(angles_r)) / (r_s * r_r * v)
        np.testing.assert_array_almost_equal(w, expected, decimal=5)

    def test_torch_input(self):
        """Test with PyTorch tensor input."""
        weight = StandardWeight(WeightMode.SPREADING)

        r_s = torch.tensor([100.0, 200.0])
        r_r = torch.tensor([100.0, 200.0])
        angles = torch.zeros(2)

        w = weight.compute_weight(r_s, r_r, angles, angles, 2000.0)

        assert isinstance(w, torch.Tensor)
        assert w.shape == (2,)

    def test_minimum_distance_protection(self):
        """Test that very small distances don't cause overflow."""
        weight = StandardWeight(WeightMode.SPREADING, device=TEST_DEVICE)

        r_s = np.array([0.1, 0.01, 0.001], dtype=np.float32)
        r_r = np.array([0.1, 0.01, 0.001], dtype=np.float32)
        angles = np.zeros(3, dtype=np.float32)

        w = weight.compute_weight(r_s, r_r, angles, angles, 2000.0)

        # Should not have inf or nan
        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()
        assert np.all(np.isfinite(w))


class TestTaper:
    """Tests for aperture taper computation."""

    @pytest.fixture
    def weight(self):
        return StandardWeight(WeightMode.NONE)

    def test_taper_inside_aperture(self, weight):
        """Points inside aperture should have weight 1."""
        distance = np.array([0.0, 500.0, 800.0])
        max_distance = 1000.0
        taper_width = 0.1  # 10% taper zone

        taper = weight.compute_taper(distance, max_distance, taper_width)

        # All inside (1 - 0.1) * 1000 = 900m
        assert taper[0] == 1.0
        assert taper[1] == 1.0
        assert taper[2] == 1.0

    def test_taper_in_transition_zone(self, weight):
        """Points in taper zone should have 0 < weight < 1."""
        distance = np.array([950.0])  # In the 900-1000m taper zone
        max_distance = 1000.0
        taper_width = 0.1

        taper = weight.compute_taper(distance, max_distance, taper_width)

        assert 0.0 < taper[0] < 1.0

    def test_taper_outside_aperture(self, weight):
        """Points outside aperture should have weight 0."""
        distance = np.array([1001.0, 1500.0, 2000.0])
        max_distance = 1000.0
        taper_width = 0.1

        taper = weight.compute_taper(distance, max_distance, taper_width)

        np.testing.assert_array_equal(taper, np.zeros(3))

    def test_taper_smooth_transition(self, weight):
        """Taper should smoothly transition from 1 to 0."""
        distance = np.linspace(800, 1200, 100)
        max_distance = 1000.0
        taper_width = 0.2

        taper = weight.compute_taper(distance, max_distance, taper_width)

        # Check monotonically decreasing in taper zone
        in_taper = (distance >= 800) & (distance <= 1000)
        taper_zone = taper[in_taper]

        for i in range(1, len(taper_zone)):
            assert taper_zone[i] <= taper_zone[i-1]

    def test_no_taper(self, weight):
        """Zero taper width should give all ones inside aperture."""
        distance = np.array([500.0, 999.0, 1001.0])
        max_distance = 1000.0
        taper_width = 0.0

        taper = weight.compute_taper(distance, max_distance, taper_width)

        # With zero taper width, function returns all ones (no tapering)
        assert taper[0] == 1.0
        assert taper[1] == 1.0
        assert taper[2] == 1.0  # No taper = no cutoff

    def test_taper_torch(self, weight):
        """Test taper with PyTorch tensors."""
        distance = torch.tensor([500.0, 950.0, 1100.0])
        max_distance = 1000.0
        taper_width = 0.1

        taper = weight.compute_taper(distance, max_distance, taper_width)

        assert isinstance(taper, torch.Tensor)
        assert taper[0] == 1.0
        assert 0 < taper[1] < 1
        assert taper[2] == 0.0


class TestTrueAmplitudeWeight:
    """Tests for TrueAmplitudeWeight class."""

    def test_includes_jacobian(self):
        """True amplitude weight should include Jacobian factor."""
        true_amp = TrueAmplitudeWeight(WeightMode.FULL, device=TEST_DEVICE)
        standard = StandardWeight(WeightMode.FULL, device=TEST_DEVICE)

        r_s = np.array([500.0], dtype=np.float32)
        r_r = np.array([500.0], dtype=np.float32)
        angles = np.array([np.pi/4], dtype=np.float32)

        w_true = true_amp.compute_weight(r_s, r_r, angles, angles, 2000.0)
        w_std = standard.compute_weight(r_s, r_r, angles, angles, 2000.0)

        # Convert to numpy if tensors
        if isinstance(w_true, torch.Tensor):
            w_true = w_true.cpu().numpy()
        if isinstance(w_std, torch.Tensor):
            w_std = w_std.cpu().numpy()

        # True amplitude includes Jacobian factor, so ratio should not be 1.0
        ratio = w_true / w_std
        assert abs(ratio[0] - 1.0) > 0.01  # Should differ by more than 1%


class TestConvenienceFunctions:
    """Tests for convenience weight functions."""

    def test_compute_spreading_weight(self):
        """Test quick spreading weight function."""
        r_s = np.array([100.0, 200.0, 500.0])
        r_r = np.array([100.0, 200.0, 500.0])

        w = compute_spreading_weight(r_s, r_r)

        expected = 1.0 / (r_s * r_r)
        np.testing.assert_array_almost_equal(w, expected)

    def test_compute_obliquity_weight(self):
        """Test quick obliquity weight function."""
        angles_s = np.array([0.0, np.pi/6, np.pi/4])
        angles_r = np.array([0.0, np.pi/6, np.pi/4])

        w = compute_obliquity_weight(angles_s, angles_r)

        expected = np.cos(angles_s) * np.cos(angles_r)
        np.testing.assert_array_almost_equal(w, expected)

    def test_compute_aperture_mask(self):
        """Test combined aperture mask function."""
        distance = np.array([500.0, 950.0, 1500.0])
        angle = np.array([0.0, np.pi/6, np.pi/3])
        max_distance = 1000.0
        max_angle = np.pi/2

        mask = compute_aperture_mask(
            distance, angle, max_distance, max_angle, taper_width=0.1
        )

        assert mask.shape == (3,)
        assert mask[0] > mask[1]  # First point fully inside
        assert mask[2] == 0.0  # Outside distance aperture


class TestWeightFactory:
    """Tests for weight factory function."""

    def test_factory_standard_modes(self):
        """Test factory returns StandardWeight for non-FULL modes."""
        for mode in [WeightMode.NONE, WeightMode.SPREADING, WeightMode.OBLIQUITY]:
            w = get_amplitude_weight(mode)
            assert isinstance(w, StandardWeight)
            assert w.mode == mode

    def test_factory_full_mode(self):
        """Test factory returns TrueAmplitudeWeight for FULL mode."""
        w = get_amplitude_weight(WeightMode.FULL)
        assert isinstance(w, TrueAmplitudeWeight)


class TestWeightSymmetry:
    """Tests for weight symmetry properties."""

    def test_symmetric_in_source_receiver(self):
        """Weight should be symmetric in source/receiver swap."""
        weight = StandardWeight(WeightMode.FULL, device=TEST_DEVICE)

        r_s = np.array([300.0], dtype=np.float32)
        r_r = np.array([500.0], dtype=np.float32)
        angle_s = np.array([np.pi/6], dtype=np.float32)
        angle_r = np.array([np.pi/4], dtype=np.float32)

        w1 = weight.compute_weight(r_s, r_r, angle_s, angle_r, 2000.0)
        w2 = weight.compute_weight(r_r, r_s, angle_r, angle_s, 2000.0)

        np.testing.assert_array_almost_equal(w1, w2, decimal=5)

    def test_weight_positive(self):
        """Weights should always be positive."""
        weight = StandardWeight(WeightMode.FULL, device=TEST_DEVICE)

        # Random test data
        n = 100
        r_s = np.random.uniform(100, 2000, n).astype(np.float32)
        r_r = np.random.uniform(100, 2000, n).astype(np.float32)
        angle_s = np.random.uniform(0, np.pi/2, n).astype(np.float32)
        angle_r = np.random.uniform(0, np.pi/2, n).astype(np.float32)

        w = weight.compute_weight(r_s, r_r, angle_s, angle_r, 2000.0)

        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()
        assert np.all(w > 0)


class TestWeightNormalization:
    """Tests for weight value ranges."""

    def test_obliquity_max_at_vertical(self):
        """Obliquity weight should be maximum for vertical rays."""
        weight = StandardWeight(WeightMode.OBLIQUITY, device=TEST_DEVICE)

        r = np.ones(5, dtype=np.float32) * 1000.0
        angles = np.array([0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2.5], dtype=np.float32)

        w = weight.compute_weight(r, r, angles, angles, 2000.0)

        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()

        # Maximum at angle=0
        assert w[0] == np.max(w)

    def test_spreading_consistent_units(self):
        """Spreading weight should have consistent units."""
        weight = StandardWeight(WeightMode.SPREADING, device=TEST_DEVICE)

        # Double the distances, weight should be 1/4
        r1 = np.array([500.0], dtype=np.float32)
        r2 = np.array([1000.0], dtype=np.float32)

        w1 = weight.compute_weight(r1, r1, np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), 2000.0)
        w2 = weight.compute_weight(r2, r2, np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), 2000.0)

        assert abs(w1[0] / w2[0] - 4.0) < 1e-4


class TestCurvedRayWeight:
    """Tests for CurvedRayWeight class."""

    def test_initialization(self):
        """Test curved ray weight initialization."""
        from processors.migration.weights import CurvedRayWeight

        weight = CurvedRayWeight(
            mode=WeightMode.FULL,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        assert weight.v0 == 2000.0
        assert weight.gradient == 0.5
        assert weight.mode == WeightMode.FULL

    def test_none_mode(self):
        """Test NONE mode returns ones."""
        from processors.migration.weights import CurvedRayWeight

        weight = CurvedRayWeight(
            mode=WeightMode.NONE,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        r = np.array([100.0, 500.0, 1000.0], dtype=np.float32)
        angles = np.zeros(3, dtype=np.float32)

        w = weight.compute_weight(r, r, angles, angles, 2500.0)

        np.testing.assert_array_almost_equal(w, np.ones(3))

    def test_spreading_mode(self):
        """Test spreading mode with gradient."""
        from processors.migration.weights import CurvedRayWeight

        weight = CurvedRayWeight(
            mode=WeightMode.SPREADING,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        r = np.array([500.0, 1000.0], dtype=np.float32)
        angles = np.zeros(2, dtype=np.float32)
        v = 2500.0  # Velocity at depth

        w = weight.compute_weight(r, r, angles, angles, v)

        # Should be positive and finite
        assert np.all(w > 0)
        assert np.all(np.isfinite(w))

        # Weight should decrease with distance
        assert w[0] > w[1]

    def test_full_mode(self):
        """Test full mode combines spreading and obliquity."""
        from processors.migration.weights import CurvedRayWeight

        weight = CurvedRayWeight(
            mode=WeightMode.FULL,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        r = np.array([500.0, 500.0], dtype=np.float32)
        angles = np.array([0.0, np.pi/4], dtype=np.float32)
        v = 2500.0

        w = weight.compute_weight(r, r, angles, angles, v)

        # Weight at angle=0 should be higher than at angle=45 degrees
        assert w[0] > w[1]

    def test_weight_with_spreading(self):
        """Test compute_weight_with_spreading method."""
        from processors.migration.weights import CurvedRayWeight

        weight = CurvedRayWeight(
            mode=WeightMode.FULL,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        S_s = np.array([1000.0, 2000.0], dtype=np.float32)
        S_r = np.array([1000.0, 2000.0], dtype=np.float32)
        angles = np.zeros(2, dtype=np.float32)
        v = 2500.0

        w = weight.compute_weight_with_spreading(S_s, S_r, angles, angles, v)

        # Should be positive
        assert np.all(w > 0)

        # Weight should be inversely proportional to spreading
        assert w[0] > w[1]

    def test_factory_curved_ray(self):
        """Test factory returns CurvedRayWeight when requested."""
        from processors.migration.weights import CurvedRayWeight

        w = get_amplitude_weight(
            mode=WeightMode.FULL,
            curved_ray=True,
            v0=2000.0,
            gradient=0.5,
        )

        assert isinstance(w, CurvedRayWeight)

    def test_factory_no_gradient_standard(self):
        """Test factory returns StandardWeight when gradient is zero."""
        from processors.migration.weights import CurvedRayWeight

        w = get_amplitude_weight(
            mode=WeightMode.FULL,
            curved_ray=True,
            v0=2000.0,
            gradient=0.0,  # Zero gradient
        )

        # Should return TrueAmplitudeWeight (FULL mode) not CurvedRayWeight
        assert not isinstance(w, CurvedRayWeight)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
