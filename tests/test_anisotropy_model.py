"""
Unit tests for AnisotropyModel.

Tests:
- Thomsen parameter handling
- Eta computation
- Parameter interpolation (1D, 2D, 3D)
- Physical bounds validation
- Factory functions
"""

import numpy as np
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.anisotropy_model import (
    AnisotropyModel,
    AnisotropyType,
    create_isotropic,
    create_constant_anisotropy,
    create_shale_anisotropy,
    create_1d_anisotropy,
    create_gradient_anisotropy,
    compute_vti_phase_velocity,
    compute_nmo_velocity,
    compute_horizontal_velocity,
    compute_effective_eta,
)


class TestAnisotropyModelBasic:
    """Basic tests for AnisotropyModel."""

    def test_constant_creation(self):
        """Test creating constant anisotropy model."""
        model = AnisotropyModel(epsilon=0.2, delta=0.1)

        assert model.epsilon == 0.2
        assert model.delta == 0.1
        assert model.anisotropy_type == AnisotropyType.CONSTANT

    def test_eta_auto_computed(self):
        """Test that eta is auto-computed from epsilon and delta."""
        model = AnisotropyModel(epsilon=0.2, delta=0.1)

        # eta = (epsilon - delta) / (1 + 2*delta)
        expected_eta = (0.2 - 0.1) / (1 + 2 * 0.1)
        assert abs(model.eta - expected_eta) < 1e-10

    def test_is_isotropic(self):
        """Test isotropic detection."""
        iso = AnisotropyModel(epsilon=0.0, delta=0.0)
        assert iso.is_isotropic

        aniso = AnisotropyModel(epsilon=0.2, delta=0.1)
        assert not aniso.is_isotropic

    def test_is_elliptic(self):
        """Test elliptic detection (epsilon = delta)."""
        elliptic = AnisotropyModel(epsilon=0.15, delta=0.15)
        assert elliptic.is_elliptic

        anelliptic = AnisotropyModel(epsilon=0.2, delta=0.1)
        assert not anelliptic.is_elliptic


class TestEtaComputation:
    """Tests for eta computation."""

    def test_eta_formula(self):
        """Test eta = (epsilon - delta) / (1 + 2*delta)."""
        eps = 0.25
        delta = 0.1

        eta = AnisotropyModel.compute_eta(eps, delta)

        expected = (eps - delta) / (1 + 2 * delta)
        assert abs(eta - expected) < 1e-10

    def test_eta_isotropic(self):
        """Eta should be zero for isotropic."""
        eta = AnisotropyModel.compute_eta(0.0, 0.0)
        assert eta == 0.0

    def test_eta_elliptic(self):
        """Eta should be zero when epsilon = delta."""
        eta = AnisotropyModel.compute_eta(0.2, 0.2)
        assert abs(eta) < 1e-10

    def test_eta_array(self):
        """Test eta computation with arrays."""
        eps = np.array([0.1, 0.2, 0.3])
        delta = np.array([0.05, 0.1, 0.15])

        eta = AnisotropyModel.compute_eta(eps, delta)

        assert len(eta) == 3
        # Check first element
        expected_0 = (0.1 - 0.05) / (1 + 2 * 0.05)
        assert abs(eta[0] - expected_0) < 1e-10


class Test1DAnisotropy:
    """Tests for 1D anisotropy model."""

    def test_1d_creation(self):
        """Test creating 1D anisotropy model."""
        z_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        epsilon = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        delta = np.array([0.05, 0.07, 0.1, 0.12, 0.15])

        model = create_1d_anisotropy(z_axis, epsilon, delta)

        assert model.anisotropy_type == AnisotropyType.V_OF_Z
        assert len(model.epsilon) == 5
        assert len(model.z_axis) == 5

    def test_1d_interpolation(self):
        """Test interpolation in 1D model."""
        z_axis = np.array([0.0, 1.0, 2.0])
        epsilon = np.array([0.1, 0.2, 0.3])
        delta = np.array([0.05, 0.1, 0.15])

        model = create_1d_anisotropy(z_axis, epsilon, delta)

        # Query at z=0.5 (should interpolate)
        eps, delta_val, eta = model.get_parameters_at(0.5)

        assert abs(eps - 0.15) < 1e-6  # Linear interpolation
        assert abs(delta_val - 0.075) < 1e-6


class TestGradientAnisotropy:
    """Tests for gradient anisotropy model."""

    def test_gradient_creation(self):
        """Test creating gradient anisotropy model."""
        model = create_gradient_anisotropy(
            epsilon_surface=0.1,
            epsilon_gradient=0.05,  # per second
            delta_surface=0.05,
            delta_gradient=0.02,
            z_max=3.0,
            dz=0.5,
        )

        assert model.anisotropy_type == AnisotropyType.V_OF_Z
        assert model.epsilon[0] == 0.1  # Surface value

    def test_gradient_values(self):
        """Test gradient anisotropy values at depth."""
        model = create_gradient_anisotropy(
            epsilon_surface=0.1,
            epsilon_gradient=0.1,  # per second
            delta_surface=0.05,
            delta_gradient=0.05,
            z_max=2.0,
            dz=1.0,
        )

        # At z=1.0: epsilon should be 0.1 + 0.1*1 = 0.2
        eps, delta_val, _ = model.get_parameters_at(1.0)
        assert abs(eps - 0.2) < 1e-6


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_isotropic(self):
        """Test creating isotropic model."""
        model = create_isotropic()

        assert model.epsilon == 0.0
        assert model.delta == 0.0
        assert model.is_isotropic

    def test_create_constant_anisotropy(self):
        """Test creating constant anisotropy."""
        model = create_constant_anisotropy(epsilon=0.25, delta=0.1)

        assert model.epsilon == 0.25
        assert model.delta == 0.1
        assert model.anisotropy_type == AnisotropyType.CONSTANT

    def test_create_shale_weak(self):
        """Test creating weak shale anisotropy."""
        model = create_shale_anisotropy(intensity='weak')

        assert model.epsilon == 0.1
        assert model.delta == 0.05

    def test_create_shale_moderate(self):
        """Test creating moderate shale anisotropy."""
        model = create_shale_anisotropy(intensity='moderate')

        assert model.epsilon == 0.2
        assert model.delta == 0.1

    def test_create_shale_strong(self):
        """Test creating strong shale anisotropy."""
        model = create_shale_anisotropy(intensity='strong')

        assert model.epsilon == 0.35
        assert model.delta == 0.15

    def test_invalid_shale_intensity(self):
        """Test error for invalid shale intensity."""
        with pytest.raises(ValueError):
            create_shale_anisotropy(intensity='invalid')


class TestPhysicsUtilities:
    """Tests for VTI physics utility functions."""

    def test_phase_velocity_vertical(self):
        """Phase velocity at theta=0 should equal V0."""
        v0 = 2500.0
        v_phase = compute_vti_phase_velocity(v0, 0.0, 0.2, 0.1)

        assert abs(v_phase - v0) < 1e-6

    def test_phase_velocity_horizontal(self):
        """Phase velocity at theta=90 degrees."""
        v0 = 2500.0
        epsilon = 0.2
        theta = np.pi / 2  # 90 degrees

        v_phase = compute_vti_phase_velocity(v0, theta, epsilon, 0.1)

        # At horizontal: V = V0 * sqrt(1 + 2*epsilon)
        expected = v0 * np.sqrt(1 + 2 * epsilon)
        assert abs(v_phase - expected) < 1.0

    def test_nmo_velocity(self):
        """Test NMO velocity computation."""
        v0 = 2500.0
        delta = 0.1

        v_nmo = compute_nmo_velocity(v0, delta)

        # V_nmo = V0 * sqrt(1 + 2*delta)
        expected = v0 * np.sqrt(1 + 2 * delta)
        assert abs(v_nmo - expected) < 1e-6

    def test_horizontal_velocity(self):
        """Test horizontal velocity computation."""
        v0 = 2500.0
        epsilon = 0.2

        v_h = compute_horizontal_velocity(v0, epsilon)

        # V_h = V0 * sqrt(1 + 2*epsilon)
        expected = v0 * np.sqrt(1 + 2 * epsilon)
        assert abs(v_h - expected) < 1e-6

    def test_effective_eta(self):
        """Test effective eta computation."""
        eta = compute_effective_eta(0.2, 0.1)

        expected = (0.2 - 0.1) / (1 + 2 * 0.1)
        assert abs(eta - expected) < 1e-10


class TestValidation:
    """Tests for parameter validation."""

    def test_shape_mismatch_error(self):
        """Test error when epsilon and delta shapes don't match."""
        with pytest.raises(ValueError):
            AnisotropyModel(
                epsilon=np.array([0.1, 0.2]),
                delta=np.array([0.05, 0.1, 0.15]),
            )

    def test_stability_condition(self):
        """Test error when 1 + 2*delta <= 0."""
        with pytest.raises(ValueError, match="Unstable"):
            AnisotropyModel(epsilon=0.0, delta=-0.6)  # 1 + 2*(-0.6) = -0.2


class TestSerialization:
    """Tests for serialization."""

    def test_constant_round_trip(self):
        """Test constant model serialization round trip."""
        model = create_constant_anisotropy(0.2, 0.1)
        model.metadata['source'] = 'test'

        d = model.to_dict()
        restored = AnisotropyModel.from_dict(d)

        assert restored.epsilon == model.epsilon
        assert restored.delta == model.delta
        assert abs(restored.eta - model.eta) < 1e-10
        assert restored.metadata.get('source') == 'test'

    def test_1d_round_trip(self):
        """Test 1D model serialization round trip."""
        z_axis = np.array([0.0, 1.0, 2.0])
        epsilon = np.array([0.1, 0.2, 0.3])
        delta = np.array([0.05, 0.1, 0.15])

        model = create_1d_anisotropy(z_axis, epsilon, delta)
        d = model.to_dict()
        restored = AnisotropyModel.from_dict(d)

        assert restored.anisotropy_type == AnisotropyType.V_OF_Z
        np.testing.assert_array_almost_equal(restored.epsilon, model.epsilon)
        np.testing.assert_array_almost_equal(restored.z_axis, model.z_axis)


class TestSummary:
    """Tests for summary output."""

    def test_constant_summary(self):
        """Test summary for constant model."""
        model = create_constant_anisotropy(0.2, 0.1)
        summary = model.get_summary()

        assert 'constant' in summary
        assert '0.2' in summary or '.20' in summary

    def test_1d_summary(self):
        """Test summary for 1D model."""
        model = create_gradient_anisotropy(0.1, 0.05, 0.05, 0.02, 2.0)
        summary = model.get_summary()

        assert 'v_of_z' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
