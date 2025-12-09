"""
Integration tests for Phases 4, 5, 6 of Kirchhoff PSTM implementation.

Tests cross-phase integration:
- Phase 4: 1D Velocity Model (v(z))
- Phase 5: Curved Ray Traveltimes
- Phase 6: VTI Anisotropy

Verifies that components work together correctly.
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
from models.anisotropy_model import (
    AnisotropyModel,
    create_isotropic,
    create_constant_anisotropy,
    create_shale_anisotropy,
    create_1d_anisotropy,
)
from models.migration_config import (
    MigrationConfig,
    OutputGrid,
    WeightMode,
    TraveltimeMode,
    AnisotropyMethod,
)
from processors.migration.traveltime import (
    StraightRayTraveltime,
    CurvedRayTraveltime,
    get_traveltime_calculator,
)
from processors.migration.traveltime_curved import (
    CurvedRayCalculator,
    CurvedRayResult,
)
from processors.migration.traveltime_vti import (
    VTITraveltimeCalculator,
    get_vti_traveltime_calculator,
)
from processors.migration.weights import (
    StandardWeight,
    CurvedRayWeight,
    TrueAmplitudeWeight,
    get_amplitude_weight,
)


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


# =============================================================================
# Phase 4 + 5 Integration: Velocity Model + Curved Ray
# =============================================================================

class TestVelocityModelWithCurvedRay:
    """Test integration of velocity model with curved ray calculator."""

    def test_gradient_model_enables_curved_ray(self):
        """Velocity model with gradient should enable curved ray mode."""
        v_model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=3.0,
        )

        assert v_model.has_gradient
        assert v_model.gradient == 0.5

        # Factory should select curved ray
        calc = get_traveltime_calculator(v_model, mode='auto')
        assert isinstance(calc, CurvedRayTraveltime)

    def test_constant_model_uses_straight_ray(self):
        """Constant velocity should use straight ray."""
        v_model = create_constant_velocity(2500.0)

        assert not v_model.has_gradient

        calc = get_traveltime_calculator(v_model, mode='auto')
        assert isinstance(calc, StraightRayTraveltime)

    def test_curved_ray_uses_velocity_properties(self):
        """Curved ray should correctly use velocity model properties."""
        v_model = create_linear_gradient_velocity(
            v0=1800.0,
            gradient=0.8,
            z_max=4.0,
        )

        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        assert calc.v0 == v_model.v0
        assert calc.k == v_model.gradient

    def test_traveltimes_differ_with_gradient(self):
        """Curved vs straight ray should differ when gradient exists."""
        v_grad = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.6,
            z_max=3.0,
        )
        v_const = create_constant_velocity(2000.0)

        calc_curved = CurvedRayCalculator(v_grad, device=TEST_DEVICE)
        calc_straight = StraightRayTraveltime(v_const, device=TEST_DEVICE)

        x = torch.tensor(1500.0)
        y = torch.tensor(0.0)
        z = torch.tensor(2.0)

        t_curved = float(calc_curved.compute_traveltime(x, y, z))
        t_straight = float(calc_straight.compute_traveltime(x, y, z))

        # Should be measurably different
        assert abs(t_curved - t_straight) > 0.001


class TestVelocityConversionWithTraveltime:
    """Test velocity conversions work correctly with traveltime."""

    def test_rms_to_interval_preserves_physics(self):
        """RMS->interval->RMS round trip should preserve values."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v_rms = np.array([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])

        _, v_int = rms_to_interval_velocity(t_axis, v_rms)
        _, v_rms_back = interval_to_rms_velocity(t_axis, v_int)

        np.testing.assert_array_almost_equal(v_rms_back, v_rms, decimal=1)

    def test_interval_velocity_higher_at_depth(self):
        """Interval velocity should be higher than RMS at depth."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v_rms = np.array([2000.0, 2100.0, 2250.0, 2400.0, 2550.0])

        _, v_int = rms_to_interval_velocity(t_axis, v_rms)

        # At depth, interval should exceed RMS
        assert v_int[-1] > v_rms[-1]


# =============================================================================
# Phase 5 + Weights Integration: Curved Ray + Amplitude Weights
# =============================================================================

class TestCurvedRayWithWeights:
    """Test curved ray calculator with amplitude weights."""

    def test_curved_ray_provides_emergence_angles(self):
        """Curved ray should provide emergence angles for weights."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        x = torch.tensor([0.0, 500.0, 1000.0])
        y = torch.zeros(3)
        z = torch.ones(3) * 1.5

        angles = calc.compute_emergence_angle(x, y, z)

        assert len(angles) == 3
        assert angles[0] < angles[1] < angles[2]  # Angle increases with offset

    def test_curved_ray_provides_spreading(self):
        """Curved ray should provide spreading factors for weights."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        x = torch.tensor([500.0, 1000.0])
        y = torch.zeros(2)
        z = torch.ones(2) * 1.5

        spreading = calc.compute_spreading_factor(x, y, z)

        assert len(spreading) == 2
        assert spreading[0] < spreading[1]  # Spreading increases with distance

    def test_curved_ray_weight_uses_gradient(self):
        """CurvedRayWeight should use velocity gradient."""
        weight = CurvedRayWeight(
            mode=WeightMode.FULL,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        assert weight.v0 == 2000.0
        assert weight.gradient == 0.5

    def test_curved_ray_full_result_feeds_weights(self):
        """Full curved ray result should provide all data needed for weights."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        result = calc.compute_full(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert isinstance(result, CurvedRayResult)
        assert result.traveltime is not None
        assert result.emergence_angle is not None
        assert result.spreading_factor is not None

        # Use these to compute weights
        weight = CurvedRayWeight(
            mode=WeightMode.FULL,
            v0=2000.0,
            gradient=0.5,
            device=TEST_DEVICE,
        )

        # Get velocity as float (not numpy.float32)
        v_at_depth = float(v_model.get_velocity_at(1.5))

        w = weight.compute_weight_with_spreading(
            np.array([float(result.spreading_factor)]),
            np.array([float(result.spreading_factor)]),
            np.array([float(result.emergence_angle)]),
            np.array([float(result.emergence_angle)]),
            v_at_depth,
        )

        # Convert to numpy if tensor
        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()

        assert np.all(w > 0)
        assert np.all(np.isfinite(w))


# =============================================================================
# Phase 6 + Phase 4/5 Integration: VTI with Velocity Model
# =============================================================================

class TestVTIWithVelocityModel:
    """Test VTI traveltime with velocity model."""

    def test_vti_with_constant_velocity(self):
        """VTI calculator with constant velocity model."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')

        calc = VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)

        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert float(t) > 0

    def test_vti_isotropic_matches_straight_ray(self):
        """VTI with zero anisotropy should match straight ray."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_isotropic()

        calc_vti = VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)
        calc_straight = StraightRayTraveltime(v_model, device=TEST_DEVICE)

        x = torch.tensor(500.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.0)

        t_vti = float(calc_vti.compute_traveltime(x, y, z))
        t_straight = float(calc_straight.compute_traveltime(x, y, z))

        assert abs(t_vti - t_straight) < 1e-5

    def test_vti_anisotropy_changes_traveltime(self):
        """Adding anisotropy should change traveltime."""
        v_model = create_constant_velocity(2500.0)

        calc_iso = VTITraveltimeCalculator(v_model, None, device=TEST_DEVICE)
        calc_vti = VTITraveltimeCalculator(
            v_model,
            create_shale_anisotropy('moderate'),
            device=TEST_DEVICE,
        )

        # Far offset to see anisotropy effect
        x = torch.tensor(2000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_iso = float(calc_iso.compute_traveltime(x, y, z))
        t_vti = float(calc_vti.compute_traveltime(x, y, z))

        assert abs(t_iso - t_vti) > 0.001


class TestVTIWithDepthVaryingAnisotropy:
    """Test VTI with 1D depth-varying anisotropy."""

    def test_1d_anisotropy_creation(self):
        """Test creating 1D anisotropy model."""
        z_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        epsilon = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        delta = np.array([0.05, 0.07, 0.1, 0.12, 0.15])

        aniso = create_1d_anisotropy(z_axis, epsilon, delta)

        # Query at z=1.0
        eps, delta_val, eta = aniso.get_parameters_at(1.0)

        assert abs(eps - 0.2) < 1e-6
        assert abs(delta_val - 0.1) < 1e-6

    def test_vti_with_1d_anisotropy(self):
        """Test VTI calculator with depth-varying anisotropy."""
        v_model = create_constant_velocity(2500.0)

        z_axis = np.array([0.0, 1.0, 2.0, 3.0])
        epsilon = np.array([0.1, 0.15, 0.2, 0.25])
        delta = np.array([0.05, 0.07, 0.1, 0.12])

        aniso = create_1d_anisotropy(z_axis, epsilon, delta)
        calc = VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)

        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )

        assert float(t) > 0
        assert np.isfinite(float(t))


# =============================================================================
# Configuration Integration
# =============================================================================

class TestMigrationConfigIntegration:
    """Test migration configuration with all phase components."""

    def test_config_traveltime_mode(self):
        """Test config traveltime mode selection."""
        config = MigrationConfig(
            traveltime_mode=TraveltimeMode.CURVED_RAY,
        )

        assert config.traveltime_mode == TraveltimeMode.CURVED_RAY

    def test_config_anisotropy_settings(self):
        """Test config anisotropy settings."""
        config = MigrationConfig(
            use_anisotropy=True,
            anisotropy_method=AnisotropyMethod.ANELLIPTIC,
        )

        assert config.use_anisotropy
        assert config.anisotropy_method == AnisotropyMethod.ANELLIPTIC

    def test_config_weight_mode(self):
        """Test config weight mode settings."""
        config = MigrationConfig(
            weight_mode=WeightMode.FULL,
        )

        assert config.weight_mode == WeightMode.FULL

    def test_factory_respects_traveltime_mode(self):
        """Factory should respect explicit traveltime mode."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)

        # Force straight ray even though gradient exists
        calc = get_traveltime_calculator(v_model, mode=TraveltimeMode.STRAIGHT_RAY)
        assert isinstance(calc, StraightRayTraveltime)

        # Force curved ray
        calc = get_traveltime_calculator(v_model, mode=TraveltimeMode.CURVED_RAY)
        assert isinstance(calc, CurvedRayTraveltime)


class TestWeightModeWithCalculators:
    """Test weight modes work with different traveltime calculators."""

    def test_weight_factory_standard_modes(self):
        """Test weight factory for standard modes."""
        for mode in [WeightMode.NONE, WeightMode.SPREADING, WeightMode.OBLIQUITY]:
            weight = get_amplitude_weight(mode)
            assert isinstance(weight, StandardWeight)

    def test_weight_factory_full_mode(self):
        """Test weight factory for FULL mode."""
        weight = get_amplitude_weight(WeightMode.FULL)
        assert isinstance(weight, TrueAmplitudeWeight)

    def test_weight_factory_curved_ray_mode(self):
        """Test weight factory for curved ray mode."""
        weight = get_amplitude_weight(
            WeightMode.FULL,
            curved_ray=True,
            v0=2000.0,
            gradient=0.5,
        )
        assert isinstance(weight, CurvedRayWeight)

    def test_weight_with_curved_ray_angles(self):
        """Test weight computation with curved ray emergence angles."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        # Get angles from curved ray
        x = np.array([500.0, 1000.0], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        z = np.ones(2, dtype=np.float32) * 1.5

        angles = calc.compute_emergence_angle(x, y, z)

        # Use with weight calculator
        weight = get_amplitude_weight(WeightMode.OBLIQUITY)
        r = np.sqrt(x**2 + y**2 + (z * 2500)**2).astype(np.float32)  # Approximate distance

        w = weight.compute_weight(r, r, angles, angles, 2500.0)

        # Convert to numpy if tensor
        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()

        assert np.all(w > 0)
        assert w[0] > w[1]  # Higher weight for smaller angle


# =============================================================================
# Full Pipeline Integration
# =============================================================================

class TestFullPipelineIntegration:
    """Test complete pipeline from velocity model to weights."""

    def test_straight_ray_pipeline(self):
        """Test complete pipeline for straight ray."""
        # Velocity model
        v_model = create_constant_velocity(2500.0)

        # Traveltime calculator
        calc = get_traveltime_calculator(v_model, mode='auto')
        assert isinstance(calc, StraightRayTraveltime)

        # Compute traveltime
        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )
        assert float(t) > 0

        # Weight calculator
        weight = get_amplitude_weight(WeightMode.FULL)

        # Compute weight
        r = np.sqrt(1000.0**2 + 1500.0**2).astype(np.float32)
        angle = np.arctan2(1000.0, 1500.0).astype(np.float32)

        w = weight.compute_weight(
            np.array([r]),
            np.array([r]),
            np.array([angle]),
            np.array([angle]),
            2500.0,
        )
        assert w[0] > 0

    def test_curved_ray_pipeline(self):
        """Test complete pipeline for curved ray."""
        # Velocity model with gradient
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)

        # Traveltime calculator - use CurvedRayCalculator directly
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        # Get full result
        result = calc.compute_full(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )
        assert isinstance(result, CurvedRayResult)

        # Weight calculator using curved ray weight
        weight = get_amplitude_weight(
            WeightMode.FULL,
            curved_ray=True,
            v0=v_model.v0,
            gradient=v_model.gradient,
        )

        # Get velocity as float
        v_at_depth = float(v_model.get_velocity_at(1.5))

        # Compute weight using spreading from curved ray
        w = weight.compute_weight_with_spreading(
            np.array([float(result.spreading_factor)]),
            np.array([float(result.spreading_factor)]),
            np.array([float(result.emergence_angle)]),
            np.array([float(result.emergence_angle)]),
            v_at_depth,
        )

        # Convert to numpy if tensor
        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()
        assert w[0] > 0

    def test_vti_pipeline(self):
        """Test complete pipeline with VTI anisotropy."""
        # Velocity model
        v_model = create_constant_velocity(2500.0)

        # Anisotropy model
        aniso = create_shale_anisotropy('moderate')

        # VTI traveltime calculator
        calc = get_vti_traveltime_calculator(v_model, aniso)

        # Compute traveltime
        t = calc.compute_traveltime(
            torch.tensor(1000.0),
            torch.tensor(0.0),
            torch.tensor(1.5),
        )
        assert float(t) > 0

        # Weight calculator (standard for VTI)
        weight = get_amplitude_weight(WeightMode.FULL)

        # Compute weight
        r = np.sqrt(1000.0**2 + 1500.0**2).astype(np.float32)
        angle = np.arctan2(1000.0, 1500.0).astype(np.float32)

        w = weight.compute_weight(
            np.array([r]),
            np.array([r]),
            np.array([angle]),
            np.array([angle]),
            2500.0,
        )
        assert w[0] > 0


# =============================================================================
# Anelliptic Approximation Tests (Phase 6.3)
# =============================================================================

class TestAnellipticApproximation:
    """Tests for anelliptic approximation accuracy."""

    def test_anelliptic_vs_weak_near_vertical(self):
        """Anelliptic and weak should agree near vertical."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_constant_anisotropy(0.2, 0.1)

        calc_anell = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='anelliptic'
        )
        calc_weak = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='weak'
        )

        # Small offset (near vertical)
        x = torch.tensor(100.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_anell = float(calc_anell.compute_traveltime(x, y, z))
        t_weak = float(calc_weak.compute_traveltime(x, y, z))

        # Should be within 10% for near-vertical
        assert abs(t_anell - t_weak) / t_anell < 0.1

    def test_anelliptic_vs_exact_moderate_angle(self):
        """Compare anelliptic and exact at moderate angles."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_constant_anisotropy(0.2, 0.1)

        calc_anell = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='anelliptic'
        )
        calc_exact = VTITraveltimeCalculator(
            v_model, aniso, device=TEST_DEVICE, method='exact'
        )

        # Moderate offset
        x = torch.tensor(1000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_anell = float(calc_anell.compute_traveltime(x, y, z))
        t_exact = float(calc_exact.compute_traveltime(x, y, z))

        # Should be within 15% for typical anisotropy values
        assert abs(t_anell - t_exact) / t_exact < 0.15

    def test_eta_effect_on_far_offset(self):
        """Larger eta should have more effect at far offsets."""
        v_model = create_constant_velocity(2500.0)

        # Small eta
        aniso_small = create_constant_anisotropy(0.15, 0.1)  # eta ~ 0.04
        # Large eta
        aniso_large = create_constant_anisotropy(0.3, 0.1)   # eta ~ 0.17

        calc_small = VTITraveltimeCalculator(v_model, aniso_small, TEST_DEVICE)
        calc_large = VTITraveltimeCalculator(v_model, aniso_large, TEST_DEVICE)

        # Far offset
        x = torch.tensor(3000.0)
        y = torch.tensor(0.0)
        z = torch.tensor(1.5)

        t_small = float(calc_small.compute_traveltime(x, y, z))
        t_large = float(calc_large.compute_traveltime(x, y, z))

        # Larger eta should give different traveltime
        assert t_small != t_large


# =============================================================================
# Batch Processing Integration
# =============================================================================

class TestBatchProcessingIntegration:
    """Test batch processing across all phases."""

    def test_batch_straight_ray(self):
        """Test batch processing for straight ray."""
        v_model = create_constant_velocity(2500.0)
        calc = get_traveltime_calculator(v_model)

        surface_x = torch.rand(100) * 3000
        surface_y = torch.rand(100) * 3000
        image_x = torch.rand(20) * 3000
        image_y = torch.rand(20) * 3000
        image_z = torch.linspace(0.1, 3.0, 50)

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t.shape == (50, 100, 20)
        assert torch.all(t > 0)
        assert torch.all(torch.isfinite(t))

    def test_batch_curved_ray(self):
        """Test batch processing for curved ray."""
        v_model = create_linear_gradient_velocity(2000.0, 0.5, 3.0)
        calc = CurvedRayCalculator(v_model, device=TEST_DEVICE)

        surface_x = torch.rand(50) * 2000
        surface_y = torch.zeros(50)
        image_x = torch.rand(10) * 2000
        image_y = torch.zeros(10)
        image_z = torch.linspace(0.5, 2.5, 20)

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t.shape == (20, 50, 10)
        assert torch.all(t > 0)
        assert torch.all(torch.isfinite(t))

    def test_batch_vti(self):
        """Test batch processing for VTI."""
        v_model = create_constant_velocity(2500.0)
        aniso = create_shale_anisotropy('moderate')
        calc = VTITraveltimeCalculator(v_model, aniso, device=TEST_DEVICE)

        surface_x = torch.rand(30) * 2000
        surface_y = torch.zeros(30)
        image_x = torch.rand(10) * 2000
        image_y = torch.zeros(10)
        image_z = torch.linspace(0.5, 2.0, 15)

        t = calc.compute_traveltime_batch(
            surface_x, surface_y, image_x, image_y, image_z
        )

        assert t.shape == (15, 30, 10)
        assert torch.all(t > 0)
        assert torch.all(torch.isfinite(t))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
