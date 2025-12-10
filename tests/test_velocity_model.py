"""
Tests for VelocityModel and RMS velocity computation.

Tests:
- VelocityModel class for constant, gradient, and file-based velocity
- RMS velocity computation for each velocity type
- Integration with time-domain migration
"""

import pytest
import numpy as np
import torch
import tempfile
import os

from processors.migration.velocity_model import (
    VelocityModel,
    create_velocity_model,
    create_velocity_model_from_config,
)


class TestVelocityModelConstant:
    """Tests for constant velocity model."""

    def test_create_constant_model(self):
        """Test creating a constant velocity model."""
        model = VelocityModel(velocity_type='constant', v0=2500.0)

        assert model.velocity_type == 'constant'
        assert model.v0 == 2500.0

    def test_interval_velocity_constant(self):
        """Test interval velocity for constant model."""
        model = VelocityModel(velocity_type='constant', v0=2500.0)

        # Scalar
        v = model.interval_velocity(1000.0)
        assert v == 2500.0

        # Array
        z = np.array([0, 500, 1000, 2000])
        v = model.interval_velocity(z)
        assert np.allclose(v, 2500.0)

        # Tensor
        z_t = torch.tensor([0, 500, 1000, 2000], dtype=torch.float32)
        v_t = model.interval_velocity(z_t)
        assert torch.allclose(v_t, torch.full_like(z_t, 2500.0))

    def test_rms_velocity_constant(self):
        """Test RMS velocity for constant model equals v0."""
        model = VelocityModel(velocity_type='constant', v0=2500.0)

        # Scalar
        v_rms = model.rms_velocity_at_time(1000.0)
        assert v_rms == 2500.0

        # Array
        t = np.array([0, 500, 1000, 2000])
        v_rms = model.rms_velocity_at_time(t)
        assert np.allclose(v_rms, 2500.0)

        # Tensor
        t_t = torch.tensor([0, 500, 1000, 2000], dtype=torch.float32)
        v_rms_t = model.rms_velocity_at_time(t_t)
        assert torch.allclose(v_rms_t, torch.full_like(t_t, 2500.0))


class TestVelocityModelGradient:
    """Tests for linear gradient velocity model."""

    def test_create_gradient_model(self):
        """Test creating a gradient velocity model."""
        model = VelocityModel(velocity_type='gradient', v0=2000.0, gradient=0.5)

        assert model.velocity_type == 'gradient'
        assert model.v0 == 2000.0
        assert model.gradient == 0.5

    def test_interval_velocity_gradient(self):
        """Test interval velocity for gradient model."""
        model = VelocityModel(velocity_type='gradient', v0=2000.0, gradient=0.5)

        # v(z) = 2000 + 0.5 * z
        v_0 = model.interval_velocity(0.0)
        assert v_0 == 2000.0

        v_1000 = model.interval_velocity(1000.0)
        assert v_1000 == 2500.0  # 2000 + 0.5 * 1000

        v_2000 = model.interval_velocity(2000.0)
        assert v_2000 == 3000.0  # 2000 + 0.5 * 2000

    def test_rms_velocity_gradient_increases_with_time(self):
        """Test RMS velocity increases with time for positive gradient."""
        model = VelocityModel(velocity_type='gradient', v0=2000.0, gradient=0.5)

        v_rms_500 = model.rms_velocity_at_time(500.0)
        v_rms_1000 = model.rms_velocity_at_time(1000.0)
        v_rms_2000 = model.rms_velocity_at_time(2000.0)

        # RMS velocity should increase with time for positive gradient
        assert v_rms_500 < v_rms_1000
        assert v_rms_1000 < v_rms_2000

    def test_rms_velocity_gradient_at_t0_equals_v0(self):
        """Test RMS velocity at t=0 equals v0."""
        model = VelocityModel(velocity_type='gradient', v0=2000.0, gradient=0.5)

        v_rms_0 = model.rms_velocity_at_time(0.0)
        assert v_rms_0 == 2000.0

    def test_rms_velocity_gradient_tensor(self):
        """Test RMS velocity works with torch tensors."""
        model = VelocityModel(velocity_type='gradient', v0=2000.0, gradient=0.5)

        t = torch.tensor([0.0, 500.0, 1000.0, 2000.0], dtype=torch.float32)
        v_rms = model.rms_velocity_at_time(t)

        assert isinstance(v_rms, torch.Tensor)
        assert v_rms.shape == (4,)
        # First value should be v0
        assert torch.isclose(v_rms[0], torch.tensor(2000.0))
        # Values should be increasing
        assert (v_rms[1:] > v_rms[:-1]).all()

    def test_rms_velocity_gradient_zero_gradient(self):
        """Test RMS velocity with zero gradient equals constant velocity."""
        model = VelocityModel(velocity_type='gradient', v0=2500.0, gradient=0.0)

        t = np.array([0, 500, 1000, 2000])
        v_rms = model.rms_velocity_at_time(t)

        assert np.allclose(v_rms, 2500.0)

    def test_rms_velocity_reasonable_values(self):
        """Test RMS velocity gives physically reasonable values."""
        # Typical sedimentary velocity gradient: 0.3-0.8 1/s
        model = VelocityModel(velocity_type='gradient', v0=1500.0, gradient=0.5)

        # At 2 seconds TWT, should have RMS velocity > v0 but not unreasonably high
        v_rms_2s = model.rms_velocity_at_time(2000.0)

        # Expect v_rms between v0 and v(z_max)
        assert 1500.0 < v_rms_2s < 2500.0


class TestFactoryFunctions:
    """Tests for velocity model factory functions."""

    def test_create_velocity_model_constant(self):
        """Test factory function for constant velocity."""
        model = create_velocity_model(
            velocity_type='constant',
            v0=2500.0,
        )
        assert model.velocity_type == 'constant'
        assert model.v0 == 2500.0

    def test_create_velocity_model_gradient(self):
        """Test factory function for gradient velocity."""
        model = create_velocity_model(
            velocity_type='gradient',
            v0=2000.0,
            gradient=0.5,
        )
        assert model.velocity_type == 'gradient'
        assert model.gradient == 0.5

    def test_create_from_config_constant(self):
        """Test creating model from wizard config dict."""
        config = {
            'velocity_type': 'constant',
            'velocity_v0': 3000.0,
        }
        model = create_velocity_model_from_config(config)

        assert model.velocity_type == 'constant'
        assert model.v0 == 3000.0

    def test_create_from_config_gradient(self):
        """Test creating model from wizard config with gradient."""
        config = {
            'velocity_type': 'gradient',
            'velocity_v0': 2000.0,
            'velocity_gradient': 0.6,
        }
        model = create_velocity_model_from_config(config)

        assert model.velocity_type == 'gradient'
        assert model.v0 == 2000.0
        assert model.gradient == 0.6


class TestRMSVelocityIntegration:
    """Integration tests for RMS velocity with migration."""

    def test_rms_velocity_with_time_domain_kernel(self):
        """Test RMS velocity model works with time-domain migration kernel."""
        from processors.migration.kirchhoff_kernel import migrate_kirchhoff_time_domain_rms

        # Create gradient velocity model
        velocity_model = VelocityModel(
            velocity_type='gradient',
            v0=2000.0,
            gradient=0.5,
        )

        # Create simple test data
        device = torch.device('cpu')
        n_samples = 50
        n_traces = 20
        n_il, n_xl = 5, 5

        traces = torch.randn(n_samples, n_traces, device=device)
        source_x = torch.linspace(0, 100, n_traces, device=device)
        source_y = torch.full((n_traces,), 50.0, device=device)
        receiver_x = source_x.clone()
        receiver_y = source_y.clone()

        output_x = torch.linspace(0, 100, n_il * n_xl, device=device)
        output_y = torch.linspace(0, 100, n_il * n_xl, device=device)
        time_axis_ms = torch.arange(n_samples, device=device, dtype=torch.float32) * 4.0

        # Should run without error
        image, fold = migrate_kirchhoff_time_domain_rms(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            output_x=output_x,
            output_y=output_y,
            time_axis_ms=time_axis_ms,
            velocity_model=velocity_model,
            dt_ms=4.0,
            t_min_ms=0.0,
            max_aperture_m=200.0,
            max_angle_deg=60.0,
            n_il=n_il,
            n_xl=n_xl,
            tile_size=10,
        )

        assert image.shape == (n_samples, n_il, n_xl)
        assert fold.shape == (n_samples, n_il, n_xl)

    def test_rms_vs_constant_velocity_produces_different_results(self):
        """Test that RMS velocity produces different results than constant velocity."""
        from processors.migration.kirchhoff_kernel import (
            migrate_kirchhoff_time_domain,
            migrate_kirchhoff_time_domain_rms,
        )

        # Create velocity models
        gradient_model = VelocityModel(velocity_type='gradient', v0=2000.0, gradient=0.5)

        device = torch.device('cpu')
        n_samples = 50
        n_traces = 20
        n_il, n_xl = 5, 5

        # Create data with impulse at specific time
        traces = torch.zeros(n_samples, n_traces, device=device)
        traces[20, :] = 1.0  # Impulse at sample 20

        source_x = torch.linspace(0, 100, n_traces, device=device)
        source_y = torch.full((n_traces,), 50.0, device=device)
        receiver_x = source_x.clone()
        receiver_y = source_y.clone()

        output_x = torch.linspace(0, 100, n_il * n_xl, device=device)
        output_y = torch.linspace(0, 100, n_il * n_xl, device=device)
        time_axis_ms = torch.arange(n_samples, device=device, dtype=torch.float32) * 4.0

        # Migrate with constant velocity
        image_const, _ = migrate_kirchhoff_time_domain(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            output_x=output_x,
            output_y=output_y,
            time_axis_ms=time_axis_ms,
            velocity=2000.0,
            dt_ms=4.0,
            t_min_ms=0.0,
            max_aperture_m=200.0,
            max_angle_deg=60.0,
            n_il=n_il,
            n_xl=n_xl,
            tile_size=10,
        )

        # Migrate with gradient velocity (RMS)
        image_grad, _ = migrate_kirchhoff_time_domain_rms(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            output_x=output_x,
            output_y=output_y,
            time_axis_ms=time_axis_ms,
            velocity_model=gradient_model,
            dt_ms=4.0,
            t_min_ms=0.0,
            max_aperture_m=200.0,
            max_angle_deg=60.0,
            n_il=n_il,
            n_xl=n_xl,
            tile_size=10,
        )

        # Results should be different (gradient velocity produces different time mapping)
        diff = (image_const - image_grad).abs().sum()
        assert diff > 0, "Constant and gradient velocity should produce different results"
