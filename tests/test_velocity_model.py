"""
Unit tests for velocity model module.

Tests:
- VelocityModel creation and validation
- 2D/3D velocity models
- Factory functions
- Velocity interpolation
"""

import numpy as np
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.velocity_model import (
    VelocityModel,
    VelocityType,
    create_constant_velocity,
    create_linear_gradient_velocity,
    create_from_rms_velocity,
    create_2d_velocity,
    create_3d_velocity,
    create_2d_gradient_velocity,
    create_layered_velocity,
    rms_to_interval_velocity,
    interval_to_rms_velocity,
)


class TestVelocityModelBasic:
    """Tests for basic VelocityModel functionality."""

    def test_constant_velocity_creation(self):
        """Test creating constant velocity model."""
        model = create_constant_velocity(2500.0)

        assert model.velocity_type == VelocityType.CONSTANT
        assert model.data == 2500.0
        assert model.is_constant
        assert model.v0 == 2500.0
        assert not model.has_gradient

    def test_constant_velocity_get_at(self):
        """Test getting velocity at points for constant model."""
        model = create_constant_velocity(2500.0)

        # Single point
        v = model.get_velocity_at(1.0)
        assert v == 2500.0

        # Array of points
        z = np.array([0.0, 0.5, 1.0, 1.5])
        v = model.get_velocity_at(z)
        assert v.shape == (4,)
        np.testing.assert_array_equal(v, 2500.0)

    def test_1d_velocity_creation(self):
        """Test creating 1D v(z) model."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
            dz=0.1,
        )

        assert model.velocity_type == VelocityType.V_OF_Z
        assert not model.is_constant
        assert model.v0 == 2000.0
        assert model.gradient == 500.0
        assert model.has_gradient

    def test_1d_velocity_interpolation(self):
        """Test interpolation for 1D model."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=2.0,
            dz=0.1,
        )

        # At z=0
        v0 = model.get_velocity_at(0.0)
        assert abs(v0 - 2000.0) < 1.0

        # At z=1.0: v = 2000 + 500 * 1.0 = 2500
        v1 = model.get_velocity_at(1.0)
        assert abs(v1 - 2500.0) < 1.0

        # At z=2.0: v = 2000 + 500 * 2.0 = 3000
        v2 = model.get_velocity_at(2.0)
        assert abs(v2 - 3000.0) < 1.0

    def test_velocity_validation_positive(self):
        """Test that non-positive velocities raise error."""
        with pytest.raises(ValueError, match="positive"):
            create_constant_velocity(0.0)

        with pytest.raises(ValueError, match="positive"):
            create_constant_velocity(-100.0)

    def test_velocity_model_copy(self):
        """Test copying velocity model."""
        model = create_linear_gradient_velocity(2000.0, 500.0, 3.0)
        model_copy = model.copy()

        assert model_copy.v0 == model.v0
        assert model_copy.gradient == model.gradient
        np.testing.assert_array_equal(model_copy.data, model.data)

        # Modify copy doesn't affect original
        model_copy.data[0] = 9999.0
        assert model.data[0] != 9999.0

    def test_velocity_serialization(self):
        """Test to_dict and from_dict."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=2.0,
        )
        model.metadata['test_key'] = 'test_value'

        d = model.to_dict()
        model_restored = VelocityModel.from_dict(d)

        assert model_restored.velocity_type == model.velocity_type
        assert model_restored.v0 == model.v0
        assert model_restored.gradient == model.gradient
        assert model_restored.metadata.get('test_key') == 'test_value'
        np.testing.assert_array_almost_equal(model_restored.data, model.data)


class TestCreate2DVelocity:
    """Tests for 2D velocity model creation."""

    def test_basic_2d_creation(self):
        """Test basic 2D velocity creation."""
        n_z, n_x = 10, 20
        z_axis = np.linspace(0, 2.0, n_z)
        x_axis = np.linspace(0, 1000.0, n_x)
        data = np.full((n_z, n_x), 2500.0, dtype=np.float32)

        model = create_2d_velocity(data, z_axis, x_axis)

        assert model.velocity_type == VelocityType.V_OF_XZ
        assert model.data.shape == (n_z, n_x)
        assert len(model.z_axis) == n_z
        assert len(model.x_axis) == n_x

    def test_2d_shape_validation(self):
        """Test that shape mismatches raise errors."""
        z_axis = np.linspace(0, 2.0, 10)
        x_axis = np.linspace(0, 1000.0, 20)

        # Wrong z dimension
        data_wrong = np.zeros((15, 20), dtype=np.float32) + 2500
        with pytest.raises(ValueError, match="z-dimension"):
            create_2d_velocity(data_wrong, z_axis, x_axis)

        # Wrong x dimension
        data_wrong = np.zeros((10, 25), dtype=np.float32) + 2500
        with pytest.raises(ValueError, match="x-dimension"):
            create_2d_velocity(data_wrong, z_axis, x_axis)

    def test_2d_not_2d_error(self):
        """Test error for non-2D data."""
        z_axis = np.linspace(0, 2.0, 10)
        x_axis = np.linspace(0, 1000.0, 20)
        data_1d = np.zeros(10, dtype=np.float32) + 2500

        with pytest.raises(ValueError, match="must be 2D"):
            create_2d_velocity(data_1d, z_axis, x_axis)

    def test_2d_interpolation(self):
        """Test 2D velocity interpolation."""
        n_z, n_x = 11, 21
        z_axis = np.linspace(0, 2.0, n_z)
        x_axis = np.linspace(0, 1000.0, n_x)

        # Create velocity that increases linearly with both z and x
        zz, xx = np.meshgrid(z_axis, x_axis, indexing='ij')
        data = (2000.0 + 500.0 * zz + 0.5 * xx).astype(np.float32)

        model = create_2d_velocity(data, z_axis, x_axis)

        # Test at grid points
        v = model.get_velocity_at(0.0, 0.0)
        assert abs(v[0] - 2000.0) < 1.0

        v = model.get_velocity_at(1.0, 500.0)
        # Expected: 2000 + 500*1.0 + 0.5*500 = 2750
        assert abs(v[0] - 2750.0) < 10.0

        v = model.get_velocity_at(2.0, 1000.0)
        # Expected: 2000 + 500*2.0 + 0.5*1000 = 3500
        assert abs(v[0] - 3500.0) < 10.0

    def test_2d_batch_interpolation(self):
        """Test 2D interpolation with multiple points."""
        n_z, n_x = 11, 21
        z_axis = np.linspace(0, 2.0, n_z)
        x_axis = np.linspace(0, 1000.0, n_x)
        data = np.full((n_z, n_x), 2500.0, dtype=np.float32)

        model = create_2d_velocity(data, z_axis, x_axis)

        z = np.array([0.5, 1.0, 1.5])
        x = np.array([250.0, 500.0, 750.0])

        v = model.get_velocity_at(z, x)

        assert v.shape == (3,)
        np.testing.assert_array_almost_equal(v, 2500.0, decimal=1)

    def test_2d_without_x_raises(self):
        """Test that getting velocity without x raises error for 2D model."""
        z_axis = np.linspace(0, 2.0, 10)
        x_axis = np.linspace(0, 1000.0, 20)
        data = np.full((10, 20), 2500.0, dtype=np.float32)

        model = create_2d_velocity(data, z_axis, x_axis)

        with pytest.raises(ValueError, match="x coordinate required"):
            model.get_velocity_at(1.0)


class TestCreate3DVelocity:
    """Tests for 3D velocity model creation."""

    def test_basic_3d_creation(self):
        """Test basic 3D velocity creation."""
        n_z, n_x, n_y = 10, 20, 15
        z_axis = np.linspace(0, 2.0, n_z)
        x_axis = np.linspace(0, 1000.0, n_x)
        y_axis = np.linspace(0, 500.0, n_y)
        data = np.full((n_z, n_x, n_y), 2500.0, dtype=np.float32)

        model = create_3d_velocity(data, z_axis, x_axis, y_axis)

        assert model.velocity_type == VelocityType.V_OF_XYZ
        assert model.data.shape == (n_z, n_x, n_y)
        assert len(model.z_axis) == n_z
        assert len(model.x_axis) == n_x
        assert len(model.y_axis) == n_y

    def test_3d_shape_validation(self):
        """Test that shape mismatches raise errors."""
        z_axis = np.linspace(0, 2.0, 10)
        x_axis = np.linspace(0, 1000.0, 20)
        y_axis = np.linspace(0, 500.0, 15)

        # Wrong z dimension
        data_wrong = np.zeros((12, 20, 15), dtype=np.float32) + 2500
        with pytest.raises(ValueError, match="z-dimension"):
            create_3d_velocity(data_wrong, z_axis, x_axis, y_axis)

        # Wrong x dimension
        data_wrong = np.zeros((10, 25, 15), dtype=np.float32) + 2500
        with pytest.raises(ValueError, match="x-dimension"):
            create_3d_velocity(data_wrong, z_axis, x_axis, y_axis)

        # Wrong y dimension
        data_wrong = np.zeros((10, 20, 18), dtype=np.float32) + 2500
        with pytest.raises(ValueError, match="y-dimension"):
            create_3d_velocity(data_wrong, z_axis, x_axis, y_axis)

    def test_3d_not_3d_error(self):
        """Test error for non-3D data."""
        z_axis = np.linspace(0, 2.0, 10)
        x_axis = np.linspace(0, 1000.0, 20)
        y_axis = np.linspace(0, 500.0, 15)
        data_2d = np.zeros((10, 20), dtype=np.float32) + 2500

        with pytest.raises(ValueError, match="must be 3D"):
            create_3d_velocity(data_2d, z_axis, x_axis, y_axis)

    def test_3d_interpolation(self):
        """Test 3D velocity interpolation."""
        n_z, n_x, n_y = 11, 21, 11
        z_axis = np.linspace(0, 2.0, n_z)
        x_axis = np.linspace(0, 1000.0, n_x)
        y_axis = np.linspace(0, 500.0, n_y)

        # Create velocity that varies with z, x, y
        zz, xx, yy = np.meshgrid(z_axis, x_axis, y_axis, indexing='ij')
        data = (2000.0 + 500.0 * zz + 0.3 * xx + 0.2 * yy).astype(np.float32)

        model = create_3d_velocity(data, z_axis, x_axis, y_axis)

        # Test at origin
        v = model.get_velocity_at(0.0, 0.0, 0.0)
        assert abs(v[0] - 2000.0) < 1.0

        # Test at a point
        v = model.get_velocity_at(1.0, 500.0, 250.0)
        # Expected: 2000 + 500*1 + 0.3*500 + 0.2*250 = 2700
        assert abs(v[0] - 2700.0) < 10.0

    def test_3d_without_coords_raises(self):
        """Test that getting velocity without x,y raises error for 3D model."""
        n_z, n_x, n_y = 10, 20, 15
        z_axis = np.linspace(0, 2.0, n_z)
        x_axis = np.linspace(0, 1000.0, n_x)
        y_axis = np.linspace(0, 500.0, n_y)
        data = np.full((n_z, n_x, n_y), 2500.0, dtype=np.float32)

        model = create_3d_velocity(data, z_axis, x_axis, y_axis)

        with pytest.raises(ValueError, match="x, y coordinates required"):
            model.get_velocity_at(1.0)

        with pytest.raises(ValueError, match="x, y coordinates required"):
            model.get_velocity_at(1.0, 500.0)  # Missing y


class TestCreate2DGradientVelocity:
    """Tests for 2D gradient velocity factory."""

    def test_2d_gradient_creation(self):
        """Test creating 2D gradient velocity model."""
        model = create_2d_gradient_velocity(
            v0=2000.0,
            z_gradient=500.0,
            x_gradient=0.5,
            z_max=2.0,
            x_max=1000.0,
            dz=0.1,
            dx=50.0,
        )

        assert model.velocity_type == VelocityType.V_OF_XZ
        assert 'model_type' in model.metadata
        assert model.metadata['model_type'] == '2d_linear_gradient'

    def test_2d_gradient_values(self):
        """Test that 2D gradient values are correct."""
        model = create_2d_gradient_velocity(
            v0=2000.0,
            z_gradient=500.0,
            x_gradient=0.5,
            z_max=2.0,
            x_max=1000.0,
        )

        # At origin: v = 2000
        v = model.get_velocity_at(0.0, 0.0)
        assert abs(v[0] - 2000.0) < 1.0

        # At z=1, x=0: v = 2000 + 500*1 = 2500
        v = model.get_velocity_at(1.0, 0.0)
        assert abs(v[0] - 2500.0) < 10.0

        # At z=0, x=1000: v = 2000 + 0.5*1000 = 2500
        v = model.get_velocity_at(0.0, 1000.0)
        assert abs(v[0] - 2500.0) < 10.0

        # At z=1, x=1000: v = 2000 + 500 + 500 = 3000
        v = model.get_velocity_at(1.0, 1000.0)
        assert abs(v[0] - 3000.0) < 10.0


class TestCreateLayeredVelocity:
    """Tests for layered velocity model creation."""

    def test_layered_creation(self):
        """Test creating layered velocity model."""
        layer_depths = np.array([0.0, 0.5, 1.0, 1.5])
        layer_velocities = np.array([2000.0, 2500.0, 3000.0, 3500.0])

        model = create_layered_velocity(
            layer_depths,
            layer_velocities,
            z_max=2.0,
            dz=0.1,
        )

        assert model.velocity_type == VelocityType.V_OF_Z
        assert 'model_type' in model.metadata
        assert model.metadata['model_type'] == 'layered'

    def test_layered_values(self):
        """Test that layered velocity has correct step values."""
        layer_depths = np.array([0.0, 0.5, 1.0])
        layer_velocities = np.array([2000.0, 2500.0, 3000.0])

        model = create_layered_velocity(
            layer_depths,
            layer_velocities,
            z_max=1.5,
            dz=0.1,
        )

        # In first layer (0 - 0.5)
        v = model.get_velocity_at(0.25)
        assert v == 2000.0

        # In second layer (0.5 - 1.0)
        v = model.get_velocity_at(0.75)
        assert v == 2500.0

        # In third layer (1.0+)
        v = model.get_velocity_at(1.25)
        assert v == 3000.0

    def test_layered_mismatched_lengths(self):
        """Test error for mismatched layer arrays."""
        layer_depths = np.array([0.0, 0.5, 1.0])
        layer_velocities = np.array([2000.0, 2500.0])  # One less

        with pytest.raises(ValueError, match="same length"):
            create_layered_velocity(
                layer_depths,
                layer_velocities,
                z_max=1.5,
            )


class TestRMSVelocity:
    """Tests for RMS velocity functions."""

    def test_create_from_rms(self):
        """Test creating model from RMS velocity."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5])
        v_rms = np.array([2000.0, 2200.0, 2400.0, 2600.0])

        model = create_from_rms_velocity(t_axis, v_rms)

        assert model.velocity_type == VelocityType.V_OF_Z
        assert 'velocity_type' in model.metadata
        assert model.metadata['velocity_type'] == 'rms'
        np.testing.assert_array_equal(model.data, v_rms)

    def test_rms_to_interval_basic(self):
        """Test RMS to interval conversion."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5])
        v_rms = np.array([2000.0, 2100.0, 2200.0, 2300.0])

        t_out, v_int = rms_to_interval_velocity(t_axis, v_rms)

        # First value should match
        assert v_int[0] == v_rms[0]

        # Interval velocities should be >= RMS at later times
        # (for increasing RMS)
        assert len(v_int) == len(v_rms)

    def test_interval_to_rms_basic(self):
        """Test interval to RMS conversion."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5])
        v_int = np.array([2000.0, 2200.0, 2400.0, 2600.0])

        t_out, v_rms = interval_to_rms_velocity(t_axis, v_int)

        # First value should match
        assert v_rms[0] == v_int[0]

        # RMS should be <= max interval velocity
        assert np.all(v_rms <= np.max(v_int) + 1.0)

    def test_rms_interval_roundtrip(self):
        """Test RMS -> interval -> RMS roundtrip."""
        t_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        v_rms_original = np.array([2000.0, 2150.0, 2300.0, 2450.0, 2600.0])

        # Convert to interval
        _, v_int = rms_to_interval_velocity(t_axis, v_rms_original)

        # Convert back to RMS
        _, v_rms_recovered = interval_to_rms_velocity(t_axis, v_int)

        # Should approximately recover original
        np.testing.assert_array_almost_equal(
            v_rms_recovered,
            v_rms_original,
            decimal=0,  # Allow some numerical error
        )


class TestEffectiveVelocity:
    """Tests for effective velocity computation."""

    def test_effective_velocity_constant(self):
        """Test effective velocity for constant model."""
        model = create_constant_velocity(2500.0)

        v_eff = model.get_effective_velocity(1.0)
        assert v_eff == 2500.0

    def test_effective_velocity_1d(self):
        """Test effective velocity for 1D model."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=2.0,
            dz=0.01,
        )

        # At z=0, effective velocity = v0
        v_eff = model.get_effective_velocity(0.0)
        assert abs(v_eff - 2000.0) < 1.0

        # At z=1, should be some average
        v_eff = model.get_effective_velocity(1.0)
        # For linear gradient, effective velocity < arithmetic mean
        assert 2000.0 < v_eff < 2500.0

    def test_effective_velocity_array(self):
        """Test effective velocity for array of depths."""
        model = create_constant_velocity(2500.0)

        z = np.array([0.5, 1.0, 1.5])
        v_eff = model.get_effective_velocity(z)

        assert v_eff.shape == (3,)
        np.testing.assert_array_equal(v_eff, 2500.0)


class Test2D3DSerialization:
    """Tests for 2D/3D model serialization."""

    def test_2d_to_dict_from_dict(self):
        """Test 2D model serialization roundtrip."""
        n_z, n_x = 5, 8
        z_axis = np.linspace(0, 1.0, n_z).astype(np.float32)
        x_axis = np.linspace(0, 500.0, n_x).astype(np.float32)
        data = np.random.rand(n_z, n_x).astype(np.float32) * 1000 + 2000

        model = create_2d_velocity(data, z_axis, x_axis)
        model.metadata['test'] = 'value'

        d = model.to_dict()
        model_restored = VelocityModel.from_dict(d)

        assert model_restored.velocity_type == VelocityType.V_OF_XZ
        np.testing.assert_array_almost_equal(model_restored.data, model.data)
        np.testing.assert_array_almost_equal(model_restored.z_axis, model.z_axis)
        np.testing.assert_array_almost_equal(model_restored.x_axis, model.x_axis)
        assert model_restored.metadata.get('test') == 'value'

    def test_3d_to_dict_from_dict(self):
        """Test 3D model serialization roundtrip."""
        n_z, n_x, n_y = 4, 5, 6
        z_axis = np.linspace(0, 1.0, n_z).astype(np.float32)
        x_axis = np.linspace(0, 500.0, n_x).astype(np.float32)
        y_axis = np.linspace(0, 300.0, n_y).astype(np.float32)
        data = np.random.rand(n_z, n_x, n_y).astype(np.float32) * 1000 + 2000

        model = create_3d_velocity(data, z_axis, x_axis, y_axis)

        d = model.to_dict()
        model_restored = VelocityModel.from_dict(d)

        assert model_restored.velocity_type == VelocityType.V_OF_XYZ
        np.testing.assert_array_almost_equal(model_restored.data, model.data)
        np.testing.assert_array_almost_equal(model_restored.z_axis, model.z_axis)
        np.testing.assert_array_almost_equal(model_restored.x_axis, model.x_axis)
        np.testing.assert_array_almost_equal(model_restored.y_axis, model.y_axis)


class TestRepr:
    """Tests for string representation."""

    def test_constant_repr(self):
        """Test constant velocity repr."""
        model = create_constant_velocity(2500.0)
        s = repr(model)
        assert "constant" in s
        assert "2500" in s

    def test_1d_repr(self):
        """Test 1D velocity repr."""
        model = create_linear_gradient_velocity(2000.0, 500.0, 2.0)
        s = repr(model)
        assert "v_of_z" in s
        assert "gradient" in s

    def test_2d_repr(self):
        """Test 2D velocity repr."""
        model = create_2d_gradient_velocity(2000.0, 500.0, 0.5, 2.0, 1000.0)
        s = repr(model)
        assert "v_of_xz" in s
        assert "shape" in s

    def test_3d_repr(self):
        """Test 3D velocity repr."""
        data = np.zeros((5, 6, 7), dtype=np.float32) + 2500
        model = create_3d_velocity(
            data,
            np.linspace(0, 1, 5),
            np.linspace(0, 100, 6),
            np.linspace(0, 100, 7),
        )
        s = repr(model)
        assert "v_of_xyz" in s
        assert "shape" in s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
