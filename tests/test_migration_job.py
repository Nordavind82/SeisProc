"""
Unit tests for migration job configuration.

Tests:
- MigrationJobConfig creation and validation
- Header mapping
- Serialization/deserialization
- Job templates
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.migration_job import (
    HeaderMapping,
    MigrationJobConfig,
    create_land_3d_template,
    create_marine_template,
    create_wide_azimuth_template,
    create_full_stack_template,
    list_templates,
    create_from_template,
)
from models.binning import create_uniform_offset_binning


class TestHeaderMapping:
    """Tests for HeaderMapping."""

    def test_basic_creation(self):
        """Test basic header mapping creation."""
        mapping = HeaderMapping(
            source_x='ShotX',
            source_y='ShotY',
        )

        assert mapping.source_x == 'ShotX'
        assert mapping.source_y == 'ShotY'
        assert mapping.offset is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        mapping = HeaderMapping(
            source_x='SX',
            receiver_x='GX',
            inline='IL',
        )

        d = mapping.to_dict()

        assert d['source_x'] == 'SX'
        assert d['receiver_x'] == 'GX'
        assert d['inline'] == 'IL'
        assert 'source_y' not in d  # None values excluded

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'source_x': 'SourceX',
            'source_y': 'SourceY',
            'offset': 'OFFSET',
        }

        mapping = HeaderMapping.from_dict(d)

        assert mapping.source_x == 'SourceX'
        assert mapping.source_y == 'SourceY'
        assert mapping.offset == 'OFFSET'


class TestMigrationJobConfig:
    """Tests for MigrationJobConfig."""

    @pytest.fixture
    def temp_input_file(self):
        """Create temporary input file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            f.write(b'dummy data')
            filepath = f.name
        yield filepath
        Path(filepath).unlink()

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_basic_creation(self, temp_input_file, temp_output_dir):
        """Test basic job config creation."""
        config = MigrationJobConfig(
            name="Test Migration",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
        )

        assert config.name == "Test Migration"
        assert config.velocity_v0 == 2500.0  # Default
        assert config.created_at is not None

    def test_validation_valid_config(self, temp_input_file, temp_output_dir):
        """Test validation of valid configuration."""
        config = MigrationJobConfig(
            name="Valid Config",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
        )

        errors = config.validate()
        assert len(errors) == 0
        assert config.is_valid()

    def test_validation_missing_input(self, temp_output_dir):
        """Test validation catches missing input file."""
        config = MigrationJobConfig(
            name="Missing Input",
            input_file="/nonexistent/path/file.sgy",
            output_directory=temp_output_dir,
            binning_preset='land_3d',
        )

        errors = config.validate()
        assert any('does not exist' in e for e in errors)
        assert not config.is_valid()

    def test_validation_missing_binning(self, temp_input_file, temp_output_dir):
        """Test validation catches missing binning."""
        config = MigrationJobConfig(
            name="No Binning",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset=None,
            binning_table=None,
        )

        errors = config.validate()
        assert any('binning' in e.lower() for e in errors)

    def test_validation_invalid_velocity(self, temp_input_file, temp_output_dir):
        """Test validation catches invalid velocity."""
        config = MigrationJobConfig(
            name="Bad Velocity",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
            velocity_v0=-100.0,  # Invalid
        )

        errors = config.validate()
        assert any('velocity' in e.lower() for e in errors)

    def test_validation_invalid_time_range(self, temp_input_file, temp_output_dir):
        """Test validation catches invalid time range."""
        config = MigrationJobConfig(
            name="Bad Time",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
            time_min_ms=5000.0,
            time_max_ms=2000.0,  # Invalid: max < min
        )

        errors = config.validate()
        assert any('time' in e.lower() for e in errors)

    def test_get_output_grid(self, temp_input_file, temp_output_dir):
        """Test output grid creation."""
        config = MigrationJobConfig(
            name="Grid Test",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
            time_min_ms=0.0,
            time_max_ms=4000.0,
            dt_ms=4.0,
            inline_min=100,
            inline_max=200,
            inline_step=1,
            xline_min=50,
            xline_max=150,
            xline_step=1,
        )

        grid = config.get_output_grid()

        assert grid.n_time == 1001  # (4000-0)/4 + 1
        assert grid.n_inline == 101  # (200-100)/1 + 1
        assert grid.n_xline == 101  # (150-50)/1 + 1
        assert grid.dt == 0.004  # 4 ms in seconds

    def test_get_migration_config(self, temp_input_file, temp_output_dir):
        """Test migration config creation."""
        config = MigrationJobConfig(
            name="Migration Config Test",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
            max_aperture_m=2500.0,
            max_angle_deg=55.0,
        )

        mig_config = config.get_migration_config()

        assert mig_config.max_aperture_m == 2500.0
        assert mig_config.max_angle_deg == 55.0

    def test_get_velocity_model(self, temp_input_file, temp_output_dir):
        """Test velocity model creation."""
        config = MigrationJobConfig(
            name="Velocity Test",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
            velocity_v0=3000.0,
            velocity_gradient=0.5,
        )

        velocity = config.get_velocity_model()

        assert velocity.v0 == 3000.0
        assert velocity.gradient == 0.5

    def test_get_binning_table_from_preset(self, temp_input_file, temp_output_dir):
        """Test binning table from preset."""
        config = MigrationJobConfig(
            name="Binning Test",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
        )

        binning = config.get_binning_table()

        assert binning is not None
        assert len(binning.bins) == 10  # Land 3D has 10 bins

    def test_get_binning_table_from_explicit(self, temp_input_file, temp_output_dir):
        """Test binning table from explicit table."""
        custom_binning = create_uniform_offset_binning(0, 5000, 5)

        config = MigrationJobConfig(
            name="Custom Binning",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_table=custom_binning,
        )

        binning = config.get_binning_table()

        assert binning is custom_binning
        assert len(binning.bins) == 5

    def test_serialization(self, temp_input_file, temp_output_dir):
        """Test to_dict and from_dict."""
        config = MigrationJobConfig(
            name="Serialization Test",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='marine',
            velocity_v0=1500.0,
            max_aperture_m=4000.0,
            header_mapping=HeaderMapping(source_x='SX', source_y='SY'),
        )

        d = config.to_dict()
        restored = MigrationJobConfig.from_dict(d)

        assert restored.name == config.name
        assert restored.velocity_v0 == config.velocity_v0
        assert restored.binning_preset == config.binning_preset
        assert restored.header_mapping.source_x == 'SX'

    def test_save_load(self, temp_input_file, temp_output_dir):
        """Test save and load to file."""
        config = MigrationJobConfig(
            name="Save/Load Test",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_preset='land_3d',
            description="Test job for save/load",
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            config.save(config_file)
            restored = MigrationJobConfig.load(config_file)

            assert restored.name == config.name
            assert restored.description == config.description
        finally:
            Path(config_file).unlink()

    def test_serialization_with_binning_table(self, temp_input_file, temp_output_dir):
        """Test serialization preserves custom binning table."""
        custom_binning = create_uniform_offset_binning(0, 3000, 3)
        custom_binning.name = "Custom 3 Bins"

        config = MigrationJobConfig(
            name="With Binning Table",
            input_file=temp_input_file,
            output_directory=temp_output_dir,
            binning_table=custom_binning,
        )

        d = config.to_dict()
        restored = MigrationJobConfig.from_dict(d)

        assert restored.binning_table is not None
        assert restored.binning_table.name == "Custom 3 Bins"
        assert len(restored.binning_table.bins) == 3


class TestJobTemplates:
    """Tests for job templates."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.NamedTemporaryFile(suffix='.sgy', delete=False) as f:
            f.write(b'dummy')
            input_file = f.name

        with tempfile.TemporaryDirectory() as output_dir:
            yield input_file, output_dir

        Path(input_file).unlink()

    def test_land_3d_template(self, temp_files):
        """Test land 3D template."""
        input_file, output_dir = temp_files

        config = create_land_3d_template(input_file, output_dir)

        assert config.binning_preset == 'land_3d'
        assert config.max_aperture_m == 3000.0
        assert 'Land 3D' in config.name

    def test_marine_template(self, temp_files):
        """Test marine template."""
        input_file, output_dir = temp_files

        config = create_marine_template(input_file, output_dir)

        assert config.binning_preset == 'marine'
        assert config.velocity_v0 == 1500.0  # Water velocity

    def test_wide_azimuth_template(self, temp_files):
        """Test wide azimuth OVT template."""
        input_file, output_dir = temp_files

        config = create_wide_azimuth_template(input_file, output_dir)

        assert config.binning_preset == 'wide_azimuth_ovt'

    def test_full_stack_template(self, temp_files):
        """Test full stack template."""
        input_file, output_dir = temp_files

        config = create_full_stack_template(input_file, output_dir)

        assert config.binning_preset == 'full_stack'
        assert config.create_stack_volume is True

    def test_list_templates(self):
        """Test listing available templates."""
        templates = list_templates()

        assert 'land_3d' in templates
        assert 'marine' in templates
        assert 'wide_azimuth_ovt' in templates
        assert 'full_stack' in templates

    def test_create_from_template(self, temp_files):
        """Test creating config from template name."""
        input_file, output_dir = temp_files

        config = create_from_template('marine', input_file, output_dir)

        assert config.binning_preset == 'marine'
        assert config.velocity_v0 == 1500.0

    def test_create_from_template_with_override(self, temp_files):
        """Test creating from template with override."""
        input_file, output_dir = temp_files

        config = create_from_template(
            'land_3d',
            input_file,
            output_dir,
            velocity_v0=3500.0,  # Override
        )

        assert config.velocity_v0 == 3500.0

    def test_create_from_invalid_template(self, temp_files):
        """Test error for invalid template name."""
        input_file, output_dir = temp_files

        with pytest.raises(ValueError, match="Unknown template"):
            create_from_template('nonexistent', input_file, output_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
