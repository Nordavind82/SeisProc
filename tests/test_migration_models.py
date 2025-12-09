"""
Unit tests for migration data models.

Tests:
- VelocityModel creation and interpolation
- MigrationConfig validation
- MigrationGeometry coordinate calculations
- HeaderSchema and HeaderMapping
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
    rms_to_interval_velocity,
    interval_to_rms_velocity,
)
from models.migration_config import (
    MigrationConfig,
    OutputGrid,
    TraveltimeMode,
    InterpolationMode,
    WeightMode,
    create_default_config,
)
from models.migration_geometry import (
    MigrationGeometry,
    create_synthetic_geometry,
    create_land_3d_geometry,
)
from models.header_schema import (
    HeaderSchema,
    HeaderDefinition,
    HeaderRequirement,
    get_pstm_header_schema,
)
from models.header_mapping import (
    HeaderMapping,
    HeaderMappingEntry,
    create_default_mapping,
    create_segy_mapping,
)


# =============================================================================
# VelocityModel Tests
# =============================================================================

class TestVelocityModel:
    """Tests for VelocityModel dataclass."""

    def test_constant_velocity_creation(self):
        """Test creating constant velocity model."""
        v = create_constant_velocity(3000.0)

        assert v.velocity_type == VelocityType.CONSTANT
        assert v.data == 3000.0
        assert v.is_constant is True
        assert v.v0 == 3000.0
        assert v.gradient is None
        assert v.z_axis is None

    def test_linear_gradient_velocity(self):
        """Test creating linear gradient velocity model."""
        v = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=4000.0,
            dz=10.0
        )

        assert v.velocity_type == VelocityType.V_OF_Z
        assert v.is_constant is False
        assert v.v0 == 2000.0
        assert v.gradient == 0.5
        assert v.has_gradient is True
        assert len(v.z_axis) == 401  # 0 to 4000 with step 10

        # Check velocity at known depths
        # v(z) = v0 + gradient * z
        # v(0) = 2000
        # v(1000) = 2000 + 0.5 * 1000 = 2500
        assert v.data[0] == pytest.approx(2000.0)
        assert v.data[100] == pytest.approx(2500.0)  # z=1000m

    def test_v1d_get_velocity_at_depth(self):
        """Test velocity interpolation at arbitrary depths."""
        v = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=4000.0,
            dz=10.0
        )

        # Test single depth
        vel = v.get_velocity_at(z=1500.0)
        expected = 2000.0 + 0.5 * 1500.0
        assert vel == pytest.approx(expected, rel=0.01)

    def test_velocity_model_to_dict(self):
        """Test serialization to dictionary."""
        v = create_constant_velocity(3000.0)
        d = v.to_dict()

        assert d['velocity_type'] == 'constant'  # VelocityType.CONSTANT.value
        assert d['data'] == 3000.0
        assert d['is_time'] is True

    def test_rms_to_interval_conversion(self):
        """Test RMS to interval velocity conversion."""
        # Simple case: constant velocity
        # RMS = interval for constant V
        t = np.array([0.5, 1.0, 1.5, 2.0])
        v_rms = np.array([2000.0, 2000.0, 2000.0, 2000.0])

        t_out, v_int = rms_to_interval_velocity(t, v_rms)

        # For constant RMS, interval should be approximately constant
        np.testing.assert_allclose(v_int[1:], 2000.0, rtol=0.1)

    def test_interval_to_rms_conversion(self):
        """Test interval to RMS velocity conversion."""
        t = np.array([0.5, 1.0, 1.5, 2.0])
        v_int = np.array([2000.0, 2200.0, 2400.0, 2600.0])

        t_out, v_rms = interval_to_rms_velocity(t, v_int)

        # RMS should be between min and max interval
        assert np.all(v_rms >= 2000.0)
        assert np.all(v_rms <= 2600.0)
        # RMS should be monotonically increasing or constant
        assert np.all(np.diff(v_rms) >= 0)


# =============================================================================
# MigrationConfig Tests
# =============================================================================

class TestMigrationConfig:
    """Tests for MigrationConfig dataclass."""

    def test_output_grid_creation(self):
        """Test OutputGrid creation."""
        grid = OutputGrid(
            n_time=1001,
            n_inline=401,
            n_xline=201,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0
        )

        assert grid.n_time == 1001
        assert grid.n_inline == 401
        assert grid.n_xline == 201

    def test_output_grid_properties(self):
        """Test OutputGrid computed properties."""
        grid = OutputGrid(
            n_time=5,
            n_inline=5,
            n_xline=3,
            dt=0.25,
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0
        )

        z = grid.time_axis
        np.testing.assert_allclose(z, [0, 0.25, 0.5, 0.75, 1.0])

        assert grid.t_max == pytest.approx(1.0)
        assert grid.x_extent == pytest.approx(100.0)  # (5-1) * 25
        assert grid.y_extent == pytest.approx(50.0)   # (3-1) * 25

    def test_default_config_creation(self):
        """Test default config factory function."""
        config = create_default_config(
            n_time=1000,
            n_inline=100,
            n_xline=100,
            dt_ms=4.0,
            d_inline_m=25.0,
            d_xline_m=25.0
        )

        assert config.output_grid.n_time == 1000
        assert config.output_grid.n_inline == 100
        assert config.output_grid.dt == pytest.approx(0.004)
        assert config.max_aperture_m == 5000.0  # default
        assert config.traveltime_mode == TraveltimeMode.STRAIGHT_RAY
        assert config.weight_mode == WeightMode.SPREADING

    def test_config_aperture_validation(self):
        """Test config aperture validation."""
        config = create_default_config()
        # Valid config should not raise
        assert config.max_aperture_m > 0
        assert 0 < config.max_angle_deg <= 90

    def test_config_serialization(self):
        """Test config to/from dict."""
        config = create_default_config(
            n_time=500,
            n_inline=50,
            n_xline=50
        )

        d = config.to_dict()
        config2 = MigrationConfig.from_dict(d)

        assert config2.max_aperture_m == config.max_aperture_m
        assert config2.output_grid.n_time == config.output_grid.n_time


# =============================================================================
# MigrationGeometry Tests
# =============================================================================

class TestMigrationGeometry:
    """Tests for MigrationGeometry class."""

    def test_geometry_creation(self):
        """Test basic geometry creation."""
        n_traces = 100
        sx = np.zeros(n_traces)
        sy = np.zeros(n_traces)
        rx = np.linspace(100, 1000, n_traces)
        ry = np.zeros(n_traces)

        geom = MigrationGeometry(
            source_x=sx,
            source_y=sy,
            receiver_x=rx,
            receiver_y=ry
        )

        assert geom.n_traces == n_traces
        assert len(geom.offset) == n_traces

    def test_offset_calculation(self):
        """Test offset calculation."""
        sx = np.array([0, 0, 0])
        sy = np.array([0, 0, 0])
        rx = np.array([100, 200, 300])
        ry = np.array([0, 0, 0])

        geom = MigrationGeometry(sx, sy, rx, ry)

        np.testing.assert_array_almost_equal(
            geom.offset,
            [100, 200, 300]
        )

    def test_azimuth_calculation(self):
        """Test azimuth calculation."""
        sx = np.array([0, 0, 0, 0])
        sy = np.array([0, 0, 0, 0])
        rx = np.array([100, 0, -100, 0])  # E, N, W, S
        ry = np.array([0, 100, 0, -100])

        geom = MigrationGeometry(sx, sy, rx, ry)
        az = geom.azimuth

        # Azimuth from north, clockwise
        np.testing.assert_array_almost_equal(
            az,
            [90, 0, 270, 180]
        )

    def test_cdp_calculation(self):
        """Test CDP midpoint calculation."""
        sx = np.array([0, 0, 0])
        sy = np.array([0, 0, 0])
        rx = np.array([100, 200, 300])
        ry = np.array([100, 200, 300])

        geom = MigrationGeometry(sx, sy, rx, ry)

        np.testing.assert_array_almost_equal(
            geom.cdp_x,
            [50, 100, 150]
        )
        np.testing.assert_array_almost_equal(
            geom.cdp_y,
            [50, 100, 150]
        )

    def test_filter_by_offset(self):
        """Test filtering by offset range."""
        sx = np.zeros(10)
        sy = np.zeros(10)
        rx = np.arange(100, 1100, 100).astype(np.float32)  # 100 to 1000
        ry = np.zeros(10)

        geom = MigrationGeometry(sx, sy, rx, ry)

        # Filter to offsets 300-600
        filtered = geom.filter_by_offset(300, 600)

        assert filtered.n_traces == 4  # 300, 400, 500, 600

    def test_filter_by_azimuth(self):
        """Test filtering by azimuth range."""
        # Create traces at various azimuths
        n = 36
        angles = np.linspace(0, 350, n)  # Every 10 degrees
        sx = np.zeros(n)
        sy = np.zeros(n)
        rx = 100 * np.sin(np.radians(angles))
        ry = 100 * np.cos(np.radians(angles))

        geom = MigrationGeometry(sx, sy, rx, ry)

        # Filter 45-135 degrees
        filtered = geom.filter_by_azimuth(45, 135)

        # Should have traces at 50, 60, 70, 80, 90, 100, 110, 120, 130
        assert filtered.n_traces >= 8

    def test_synthetic_geometry(self):
        """Test synthetic geometry factory."""
        geom = create_synthetic_geometry(
            n_shots=5,
            n_receivers_per_shot=24,
            receiver_spacing=25.0,
            near_offset=100.0
        )

        assert geom.n_traces == 5 * 24  # 120 traces
        # First shot, first receiver offset should be near_offset
        assert geom.offset[0] == pytest.approx(100.0)

    def test_land_3d_geometry(self):
        """Test 3D land geometry factory."""
        geom = create_land_3d_geometry(
            n_source_lines=2,
            n_sources_per_line=3,
            n_receiver_lines=2,
            n_receivers_per_line=5,
        )

        # Total traces = sources * receivers = 6 * 10 = 60
        assert geom.n_traces == 60


# =============================================================================
# HeaderSchema Tests
# =============================================================================

class TestHeaderSchema:
    """Tests for HeaderSchema and HeaderDefinition."""

    def test_pstm_schema_creation(self):
        """Test PSTM header schema factory."""
        schema = get_pstm_header_schema()

        assert len(schema.headers) > 0
        assert 'SOURCE_X' in schema.headers
        assert 'RECEIVER_X' in schema.headers
        assert 'OFFSET' in schema.headers

    def test_required_headers(self):
        """Test getting required headers."""
        schema = get_pstm_header_schema()
        required = schema.get_required_headers()

        # Source and receiver positions are required
        assert 'SOURCE_X' in required
        assert 'SOURCE_Y' in required
        assert 'RECEIVER_X' in required
        assert 'RECEIVER_Y' in required

    def test_computable_headers(self):
        """Test identifying computable headers."""
        schema = get_pstm_header_schema()
        computable = schema.get_computable_headers()

        # OFFSET and AZIMUTH can be computed
        assert 'OFFSET' in computable
        assert 'AZIMUTH' in computable

    def test_header_requirement_levels(self):
        """Test header requirement levels."""
        schema = get_pstm_header_schema()

        # SOURCE_X should be required
        assert schema.headers['SOURCE_X'].requirement == HeaderRequirement.REQUIRED

        # OFFSET is preferred (can be computed if missing)
        offset_def = schema.headers['OFFSET']
        assert offset_def.requirement == HeaderRequirement.PREFERRED
        assert offset_def.can_compute is True

    def test_auto_map_headers(self):
        """Test auto-mapping of input headers."""
        schema = get_pstm_header_schema()

        available = ['SourceX', 'SourceY', 'GroupX', 'GroupY', 'FFID', 'TraceNumber']
        mapping = schema.auto_map_headers(available)

        assert 'SOURCE_X' in mapping
        assert mapping['SOURCE_X'] == 'SourceX'
        assert 'RECEIVER_X' in mapping
        assert mapping['RECEIVER_X'] == 'GroupX'


# =============================================================================
# HeaderMapping Tests
# =============================================================================

class TestHeaderMapping:
    """Tests for HeaderMapping class."""

    def test_default_mapping(self):
        """Test default header mapping."""
        available = ['SourceX', 'SourceY', 'GroupX', 'GroupY']
        mapping = create_default_mapping(available_headers=available)

        assert mapping.get_input_name('SOURCE_X') == 'SourceX'
        assert mapping.get_input_name('RECEIVER_X') == 'GroupX'

    def test_segy_mapping(self):
        """Test SEG-Y standard mapping."""
        mapping = create_segy_mapping()

        # Check SEG-Y standard header mapping
        assert mapping.get_input_name('SOURCE_X') == 'SourceX'
        assert mapping.get_input_name('RECEIVER_X') == 'GroupX'

        # Computed headers should be marked
        assert mapping.is_computed('OFFSET')
        assert mapping.is_computed('AZIMUTH')

    def test_custom_mapping(self):
        """Test creating custom mapping."""
        mapping = HeaderMapping()
        mapping.add_mapping('SOURCE_X', 'SX')
        mapping.add_mapping('SOURCE_Y', 'SY')
        mapping.add_mapping('RECEIVER_X', 'GX')
        mapping.add_mapping('RECEIVER_Y', 'GY')

        assert mapping.get_input_name('SOURCE_X') == 'SX'
        assert mapping.get_input_name('SOURCE_Y') == 'SY'

    def test_mapping_validation(self):
        """Test mapping validation against schema."""
        mapping = create_segy_mapping()

        result = mapping.validate()

        # SEG-Y mapping should be valid (all required headers mapped)
        assert result['valid'] is True
        assert len(result['missing_required']) == 0

    def test_mapping_with_computed_headers(self):
        """Test mapping with computed headers."""
        mapping = HeaderMapping()
        mapping.add_mapping('SOURCE_X', 'SourceX')
        mapping.add_mapping('SOURCE_Y', 'SourceY')
        mapping.add_mapping('RECEIVER_X', 'GroupX')
        mapping.add_mapping('RECEIVER_Y', 'GroupY')
        mapping.add_computed('OFFSET')
        mapping.add_computed('AZIMUTH')

        assert mapping.is_computed('OFFSET')
        assert mapping.is_computed('AZIMUTH')

        computed = mapping.get_computed_headers()
        assert 'OFFSET' in computed
        assert 'AZIMUTH' in computed


# =============================================================================
# Integration Tests
# =============================================================================

class TestMigrationModelsIntegration:
    """Integration tests combining multiple model classes."""

    def test_config_with_geometry(self):
        """Test config and geometry work together."""
        config = create_default_config(
            n_time=1000,
            n_inline=200,
            n_xline=100,
            dt_ms=4.0,
            d_inline_m=25.0,
            d_xline_m=25.0
        )

        geom = create_synthetic_geometry(
            n_shots=10,
            n_receivers_per_shot=48,
            receiver_spacing=25.0,
            near_offset=100.0
        )

        # Geometry should fit within config aperture
        max_offset = np.max(geom.offset)
        assert max_offset <= config.max_aperture_m

    def test_velocity_model_for_traveltime(self):
        """Test velocity model provides data for traveltime calc."""
        v = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=0.5,
            z_max=4000.0,
            dz=10.0
        )

        # Check we can get velocity at arbitrary depth for traveltime
        depths = np.array([500.0, 1000.0, 2000.0, 3000.0])

        for z in depths:
            vel = v.get_velocity_at(z=z)
            expected = 2000.0 + 0.5 * z
            assert vel == pytest.approx(expected, rel=0.02)

    def test_full_workflow_setup(self):
        """Test setting up complete migration workflow configuration."""
        # 1. Create velocity model
        velocity = create_linear_gradient_velocity(
            v0=2000.0, gradient=0.3, z_max=5000.0, dz=10.0
        )

        # 2. Create migration config
        config = create_default_config(
            n_time=1000,
            n_inline=100,
            n_xline=100,
            dt_ms=4.0,
            d_inline_m=25.0,
            d_xline_m=25.0
        )
        config.traveltime_mode = TraveltimeMode.CURVED_RAY
        config.weight_mode = WeightMode.FULL

        # 3. Create geometry
        geometry = create_land_3d_geometry(
            n_source_lines=2,
            n_sources_per_line=10,
            n_receiver_lines=4,
            n_receivers_per_line=24,
        )

        # 4. Create header mapping
        mapping = create_segy_mapping()

        # Validate everything
        assert velocity.has_gradient is True
        assert geometry.n_traces == 1920  # 20 sources * 96 receivers
        result = mapping.validate()
        assert result['valid'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
