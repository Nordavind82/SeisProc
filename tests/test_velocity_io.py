"""
Unit tests for velocity I/O utilities.

Tests:
- Text format read/write
- JSON format read/write
- NumPy format read/write
- Auto-detection
- Velocity conversion
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.velocity_io import (
    read_velocity_text,
    write_velocity_text,
    read_velocity_json,
    write_velocity_json,
    read_velocity_npy,
    write_velocity_npy,
    read_velocity_auto,
    write_velocity_auto,
    convert_velocity_file,
    create_velocity_from_picks,
)
from models.velocity_model import (
    VelocityModel,
    VelocityType,
    create_constant_velocity,
    create_linear_gradient_velocity,
    rms_to_interval_velocity,
)


class TestTextFormat:
    """Tests for text file I/O."""

    def test_write_and_read_vz(self, tmp_path):
        """Test writing and reading v(z) model."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
            dz=0.1,
        )

        filepath = tmp_path / "velocity.txt"
        write_velocity_text(model, str(filepath))

        assert filepath.exists()

        # Read back
        model_read = read_velocity_text(
            str(filepath),
            time_unit='ms',
            velocity_unit='m/s',
        )

        # Check data matches
        assert model_read.velocity_type == VelocityType.V_OF_Z
        assert len(model_read.data) == len(model.data)
        np.testing.assert_array_almost_equal(model_read.data, model.data, decimal=1)

    def test_write_constant_velocity(self, tmp_path):
        """Test writing constant velocity model."""
        model = create_constant_velocity(2500.0)

        filepath = tmp_path / "constant.txt"
        write_velocity_text(model, str(filepath))

        assert filepath.exists()

        # Should have at least 2 lines
        with open(filepath) as f:
            lines = [l for l in f.readlines() if not l.startswith('#')]
        assert len(lines) >= 2

    def test_read_with_comments(self, tmp_path):
        """Test reading file with comment lines."""
        filepath = tmp_path / "with_comments.txt"

        with open(filepath, 'w') as f:
            f.write("# This is a velocity file\n")
            f.write("# Time (ms)  Velocity (m/s)\n")
            f.write("0\t2000\n")
            f.write("# Mid-file comment\n")
            f.write("1000\t2500\n")
            f.write("2000\t3000\n")

        model = read_velocity_text(str(filepath), time_unit='ms')

        assert len(model.data) == 3
        assert model.data[0] == 2000.0
        assert model.data[2] == 3000.0

    def test_read_with_skip_header(self, tmp_path):
        """Test reading file with header lines to skip."""
        filepath = tmp_path / "with_header.txt"

        with open(filepath, 'w') as f:
            f.write("Velocity Model Export\n")
            f.write("Version 1.0\n")
            f.write("0\t2000\n")
            f.write("1000\t2500\n")

        model = read_velocity_text(str(filepath), time_unit='ms', skip_header=2)

        assert len(model.data) == 2
        assert model.data[0] == 2000.0

    def test_read_custom_columns(self, tmp_path):
        """Test reading with custom column indices."""
        filepath = tmp_path / "multi_column.txt"

        with open(filepath, 'w') as f:
            f.write("0\t1\t0\t2000\n")  # time in column 2, velocity in column 3
            f.write("0\t1\t1000\t2500\n")
            f.write("0\t1\t2000\t3000\n")

        model = read_velocity_text(
            str(filepath),
            time_column=2,
            velocity_column=3,
            time_unit='ms',
        )

        assert len(model.data) == 3
        assert model.z_axis[0] == 0.0
        assert model.z_axis[2] == 2.0  # 2000ms = 2s

    def test_unit_conversion(self, tmp_path):
        """Test unit conversion during read."""
        filepath = tmp_path / "units.txt"

        with open(filepath, 'w') as f:
            f.write("0\t6561.68\n")  # 2000 m/s in ft/s
            f.write("1000\t8202.10\n")  # 2500 m/s in ft/s

        model = read_velocity_text(
            str(filepath),
            time_unit='ms',
            velocity_unit='ft/s',
        )

        # Velocities should be converted to m/s
        assert abs(model.data[0] - 2000.0) < 1.0
        assert abs(model.data[1] - 2500.0) < 1.0

    def test_file_not_found(self):
        """Test error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_velocity_text("/nonexistent/path/velocity.txt")


class TestJsonFormat:
    """Tests for JSON file I/O."""

    def test_write_and_read_json(self, tmp_path):
        """Test JSON round-trip."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
        )

        filepath = tmp_path / "velocity.json"
        write_velocity_json(model, str(filepath))

        assert filepath.exists()

        model_read = read_velocity_json(str(filepath))

        assert model_read.velocity_type == model.velocity_type
        assert model_read.v0 == model.v0
        assert model_read.gradient == model.gradient
        np.testing.assert_array_almost_equal(model_read.data, model.data)

    def test_json_constant_velocity(self, tmp_path):
        """Test JSON with constant velocity."""
        model = create_constant_velocity(2500.0)

        filepath = tmp_path / "constant.json"
        write_velocity_json(model, str(filepath))

        model_read = read_velocity_json(str(filepath))

        assert model_read.velocity_type == VelocityType.CONSTANT
        assert model_read.data == 2500.0

    def test_json_preserves_metadata(self, tmp_path):
        """Test that metadata is preserved in JSON."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
        )
        model.metadata['custom_field'] = 'test_value'

        filepath = tmp_path / "with_meta.json"
        write_velocity_json(model, str(filepath))

        model_read = read_velocity_json(str(filepath))

        assert model_read.metadata.get('custom_field') == 'test_value'


class TestNumpyFormat:
    """Tests for NumPy binary I/O."""

    def test_write_and_read_npy(self, tmp_path):
        """Test NumPy round-trip."""
        model = create_linear_gradient_velocity(
            v0=2000.0,
            gradient=500.0,
            z_max=3.0,
        )

        filepath = tmp_path / "velocity.npz"
        write_velocity_npy(model, str(filepath))

        assert filepath.exists()

        model_read = read_velocity_npy(str(filepath))

        assert model_read.velocity_type == model.velocity_type
        np.testing.assert_array_almost_equal(model_read.data, model.data)
        np.testing.assert_array_almost_equal(model_read.z_axis, model.z_axis)

    def test_npy_constant_velocity(self, tmp_path):
        """Test NumPy with constant velocity."""
        model = create_constant_velocity(2500.0)

        filepath = tmp_path / "constant.npz"
        write_velocity_npy(model, str(filepath))

        model_read = read_velocity_npy(str(filepath))

        assert model_read.velocity_type == VelocityType.CONSTANT
        assert model_read.data == 2500.0


class TestAutoDetection:
    """Tests for auto-detection of file format."""

    def test_auto_read_txt(self, tmp_path):
        """Test auto-detection of .txt file."""
        filepath = tmp_path / "velocity.txt"

        with open(filepath, 'w') as f:
            f.write("0\t2000\n")
            f.write("1000\t2500\n")

        model = read_velocity_auto(str(filepath), time_unit='ms')

        assert model.velocity_type == VelocityType.V_OF_Z

    def test_auto_read_json(self, tmp_path):
        """Test auto-detection of .json file."""
        model = create_constant_velocity(2000.0)
        filepath = tmp_path / "velocity.json"
        write_velocity_json(model, str(filepath))

        model_read = read_velocity_auto(str(filepath))

        assert model_read.data == 2000.0

    def test_auto_read_npz(self, tmp_path):
        """Test auto-detection of .npz file."""
        model = create_linear_gradient_velocity(2000.0, 500.0, 3.0)
        filepath = tmp_path / "velocity.npz"
        write_velocity_npy(model, str(filepath))

        model_read = read_velocity_auto(str(filepath))

        assert model_read.velocity_type == VelocityType.V_OF_Z

    def test_auto_write_by_extension(self, tmp_path):
        """Test auto-detection of output format by extension."""
        model = create_constant_velocity(2000.0)

        # Write as JSON
        json_path = tmp_path / "vel.json"
        write_velocity_auto(model, str(json_path))
        assert json_path.exists()

        # Verify it's valid JSON
        with open(json_path) as f:
            data = json.load(f)
        assert 'velocity_type' in data


class TestVelocityConversion:
    """Tests for velocity type conversion."""

    def test_rms_to_interval_file(self, tmp_path):
        """Test converting RMS to interval velocity file."""
        # Create RMS velocity file
        rms_path = tmp_path / "rms.txt"
        with open(rms_path, 'w') as f:
            f.write("0\t2000\n")
            f.write("500\t2200\n")
            f.write("1000\t2500\n")
            f.write("1500\t2800\n")

        int_path = tmp_path / "interval.txt"

        model = convert_velocity_file(
            str(rms_path),
            str(int_path),
            input_type='rms',
            output_type='interval',
            time_unit='ms',
        )

        assert int_path.exists()
        assert model.metadata.get('velocity_type') == 'interval'

        # Interval velocities should differ from RMS
        # First value should be same
        assert abs(model.data[0] - 2000.0) < 1.0


class TestVelocityFromPicks:
    """Tests for creating velocity from picks."""

    def test_create_from_picks(self):
        """Test creating velocity model from picks."""
        picks = [
            (0.0, 2000.0),
            (0.5, 2250.0),
            (1.0, 2500.0),
            (1.5, 2750.0),
        ]

        model = create_velocity_from_picks(picks, dt=0.1)

        assert model.velocity_type == VelocityType.V_OF_Z
        assert model.z_axis[0] == 0.0
        assert model.data[0] == 2000.0

        # Check interpolation at 0.5s
        idx_05 = np.argmin(np.abs(model.z_axis - 0.5))
        assert abs(model.data[idx_05] - 2250.0) < 10.0

    def test_picks_unsorted(self):
        """Test that unsorted picks are handled."""
        picks = [
            (1.0, 2500.0),
            (0.0, 2000.0),
            (0.5, 2250.0),
        ]

        model = create_velocity_from_picks(picks, dt=0.1)

        # Should be sorted
        assert model.z_axis[0] < model.z_axis[1]
        assert model.data[0] == 2000.0

    def test_empty_picks_error(self):
        """Test error for empty picks."""
        with pytest.raises(ValueError, match="No velocity picks"):
            create_velocity_from_picks([], dt=0.1)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_file(self, tmp_path):
        """Test error for empty file."""
        filepath = tmp_path / "empty.txt"
        filepath.touch()

        with pytest.raises(ValueError, match="No valid velocity data"):
            read_velocity_text(str(filepath))

    def test_invalid_data(self, tmp_path):
        """Test handling of invalid data lines."""
        filepath = tmp_path / "invalid.txt"

        with open(filepath, 'w') as f:
            f.write("0\t2000\n")
            f.write("invalid\tdata\n")  # Should be skipped
            f.write("1000\t2500\n")

        model = read_velocity_text(str(filepath), time_unit='ms')

        # Should have 2 valid entries
        assert len(model.data) == 2

    def test_create_output_directory(self, tmp_path):
        """Test that output directory is created if needed."""
        model = create_constant_velocity(2000.0)
        filepath = tmp_path / "subdir" / "velocity.json"

        write_velocity_json(model, str(filepath))

        assert filepath.exists()

    def test_comma_delimiter(self, tmp_path):
        """Test reading CSV format."""
        filepath = tmp_path / "velocity.csv"

        with open(filepath, 'w') as f:
            f.write("0,2000\n")
            f.write("1000,2500\n")

        model = read_velocity_text(str(filepath), time_unit='ms', delimiter=',')

        assert len(model.data) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
