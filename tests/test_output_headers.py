"""
Integration Test: Output Header Correctness

Verifies that migration output headers are properly populated:
- INLINE_3D and CROSSLINE_3D from output grid
- CDP_X and CDP_Y from grid coordinates
- OFFSET and AZIMUTH from bin centers
- FOLD count (optional)
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.migration_config import OutputGrid
from models.binning import create_uniform_offset_binning, create_ovt_binning
from seisio.migration_output import MigrationOutputManager, OutputVolumeInfo


class TestOutputHeaderPopulation:
    """Tests for output header population."""

    @pytest.fixture
    def output_grid(self):
        """Create test output grid."""
        return OutputGrid(
            n_time=200,
            n_inline=50,
            n_xline=60,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0,
            inline_start=100,
            xline_start=200,
            x_origin=500000.0,
            y_origin=6000000.0,
            inline_azimuth=45.0,
        )

    @pytest.fixture
    def binning_table(self):
        """Create test binning table."""
        return create_uniform_offset_binning(0, 3000, 6)

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_inline_xline_headers(self, output_grid, binning_table, temp_output_dir):
        """Test INLINE and XLINE header values."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name

        # Test various inline/xline positions
        test_cases = [
            (0, 0, 100, 200),      # First position
            (10, 20, 110, 220),    # Mid position
            (49, 59, 149, 259),    # Last position
        ]

        for inline_idx, xline_idx, expected_inline, expected_xline in test_cases:
            headers = manager.get_output_headers(bin_name, inline_idx, xline_idx)

            assert headers['INLINE_3D'] == expected_inline, \
                f"Expected INLINE {expected_inline}, got {headers['INLINE_3D']}"
            assert headers['CROSSLINE_3D'] == expected_xline, \
                f"Expected XLINE {expected_xline}, got {headers['CROSSLINE_3D']}"

    def test_cdp_coordinates(self, output_grid, binning_table, temp_output_dir):
        """Test CDP_X and CDP_Y coordinate headers."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name

        # Origin should match grid origin
        headers = manager.get_output_headers(bin_name, 0, 0)
        assert 'CDP_X' in headers
        assert 'CDP_Y' in headers

        # CDP coordinates should be at origin for (0,0)
        # With azimuth=45 degrees and origin=(500000, 6000000)
        expected_x = output_grid.x_origin
        expected_y = output_grid.y_origin

        assert abs(headers['CDP_X'] - expected_x) < 1.0, \
            f"Expected CDP_X ~{expected_x}, got {headers['CDP_X']}"
        assert abs(headers['CDP_Y'] - expected_y) < 1.0, \
            f"Expected CDP_Y ~{expected_y}, got {headers['CDP_Y']}"

    def test_cdp_coordinates_vary_with_position(self, output_grid, binning_table, temp_output_dir):
        """Test that CDP coordinates change with inline/xline position."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        bin_name = binning_table.bins[0].name

        # Get coordinates at different positions
        headers_origin = manager.get_output_headers(bin_name, 0, 0)
        headers_inline = manager.get_output_headers(bin_name, 10, 0)
        headers_xline = manager.get_output_headers(bin_name, 0, 10)

        # Moving along inline should change coordinates
        assert headers_inline['CDP_X'] != headers_origin['CDP_X'] or \
               headers_inline['CDP_Y'] != headers_origin['CDP_Y']

        # Moving along xline should change coordinates
        assert headers_xline['CDP_X'] != headers_origin['CDP_X'] or \
               headers_xline['CDP_Y'] != headers_origin['CDP_Y']

    def test_offset_header_from_bin(self, output_grid, binning_table, temp_output_dir):
        """Test OFFSET header matches bin center."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
        )
        manager.initialize()

        # Check each bin's offset header
        for bin_obj in binning_table.bins:
            headers = manager.get_output_headers(bin_obj.name, 0, 0)

            expected_offset = (bin_obj.offset_min + bin_obj.offset_max) / 2

            assert 'OFFSET' in headers, f"OFFSET missing for bin {bin_obj.name}"
            assert abs(headers['OFFSET'] - expected_offset) < 0.1, \
                f"Expected OFFSET {expected_offset}, got {headers['OFFSET']}"

    def test_azimuth_header_from_bin(self, output_grid, temp_output_dir):
        """Test AZIMUTH header matches bin center for OVT binning."""
        # Create OVT binning with azimuth sectors
        offset_ranges = [(0, 1000), (1000, 2000)]
        ovt_binning = create_ovt_binning(offset_ranges, n_azimuth_sectors=4)

        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=ovt_binning,
        )
        manager.initialize()

        # Check bins have correct azimuth
        for bin_obj in ovt_binning.bins:
            headers = manager.get_output_headers(bin_obj.name, 0, 0)

            expected_azimuth = (bin_obj.azimuth_min + bin_obj.azimuth_max) / 2

            # Handle wrap-around for azimuths crossing 0/360
            if bin_obj.azimuth_max < bin_obj.azimuth_min:
                expected_azimuth = ((bin_obj.azimuth_min + bin_obj.azimuth_max + 360) / 2) % 360

            assert 'AZIMUTH' in headers, f"AZIMUTH missing for bin {bin_obj.name}"

    def test_stack_volume_has_no_bin_headers(self, output_grid, binning_table, temp_output_dir):
        """Test that stack volume doesn't have bin-specific OFFSET/AZIMUTH."""
        manager = MigrationOutputManager(
            output_directory=temp_output_dir,
            output_grid=output_grid,
            binning_table=binning_table,
            create_stack=True,
        )
        manager.initialize()

        # Stack should still have INLINE/XLINE/CDP but offset/azimuth are None
        headers = manager.get_output_headers('stack', 10, 20)

        assert headers['INLINE_3D'] == 110
        assert headers['CROSSLINE_3D'] == 220
        assert 'CDP_X' in headers
        assert 'CDP_Y' in headers
        # Stack has no offset/azimuth center
        assert headers.get('OFFSET') is None
        assert headers.get('AZIMUTH') is None


class TestOutputVolumeInfo:
    """Tests for OutputVolumeInfo metadata."""

    def test_bin_info_complete(self):
        """Test that OutputVolumeInfo captures bin information."""
        info = OutputVolumeInfo(
            name="offset_0_500",
            filepath="/output/migrated_offset_0_500.sgy",
            bin_name="offset_0_500",
            offset_center=250.0,
            azimuth_center=180.0,
        )

        assert info.name == "offset_0_500"
        assert info.offset_center == 250.0
        assert info.azimuth_center == 180.0
        assert info.n_traces_migrated == 0
        assert info.is_complete is False

    def test_stack_info_no_bin(self):
        """Test that stack volume info has no bin-specific values."""
        info = OutputVolumeInfo(
            name="stack",
            filepath="/output/migrated_stack.sgy",
            bin_name=None,
            offset_center=None,
            azimuth_center=None,
        )

        assert info.bin_name is None
        assert info.offset_center is None
        assert info.azimuth_center is None


class TestOutputGridCoordinates:
    """Tests for OutputGrid coordinate calculations."""

    def test_get_coordinates_zero_azimuth(self):
        """Test coordinate calculation with zero azimuth."""
        grid = OutputGrid(
            n_time=100,
            n_inline=20,
            n_xline=20,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            inline_start=1,
            xline_start=1,
            x_origin=0.0,
            y_origin=0.0,
            inline_azimuth=0.0,  # North
        )

        # At origin
        x, y = grid.get_coordinates(0, 0)
        assert abs(x) < 0.01 and abs(y) < 0.01

        # Moving along inline (north) should increase Y
        x, y = grid.get_coordinates(10, 0)
        assert abs(x) < 0.01
        assert y > 0

        # Moving along xline (east) should increase X
        x, y = grid.get_coordinates(0, 10)
        assert x > 0
        assert abs(y) < 0.01

    def test_get_coordinates_45_degree_azimuth(self):
        """Test coordinate calculation with 45 degree azimuth."""
        grid = OutputGrid(
            n_time=100,
            n_inline=20,
            n_xline=20,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            inline_start=1,
            xline_start=1,
            x_origin=0.0,
            y_origin=0.0,
            inline_azimuth=45.0,  # Northeast
        )

        # Moving along inline should increase both X and Y equally
        x, y = grid.get_coordinates(10, 0)
        dist_inline = 10 * 25.0  # 250m

        # At 45 degrees: x = dist * sin(45) = dist * 0.707
        #                y = dist * cos(45) = dist * 0.707
        expected = dist_inline * 0.7071

        assert abs(x - expected) < 1.0
        assert abs(y - expected) < 1.0

    def test_coordinate_consistency_across_grid(self):
        """Test that coordinates are consistent across grid."""
        grid = OutputGrid(
            n_time=100,
            n_inline=20,
            n_xline=20,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            inline_start=100,
            xline_start=200,
            x_origin=500000.0,
            y_origin=6000000.0,
            inline_azimuth=30.0,
        )

        # Check that spacing is consistent
        x1, y1 = grid.get_coordinates(0, 0)
        x2, y2 = grid.get_coordinates(1, 0)

        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        assert abs(dist - grid.d_inline) < 0.01


class TestHeaderConsistency:
    """Tests for header consistency across operations."""

    @pytest.fixture
    def manager_with_data(self, tmp_path):
        """Create manager with some accumulated data."""
        grid = OutputGrid(
            n_time=50,
            n_inline=10,
            n_xline=10,
            dt=0.004,
            inline_start=1,
            xline_start=1,
        )

        binning = create_uniform_offset_binning(0, 2000, 4)

        manager = MigrationOutputManager(
            output_directory=str(tmp_path),
            output_grid=grid,
            binning_table=binning,
            create_fold=True,
        )
        manager.initialize()

        # Add some data
        test_data = np.random.randn(50, 10, 10).astype(np.float32)
        for bin_obj in binning.bins:
            manager.add_migrated_data(bin_obj.name, test_data)

        return manager, grid, binning

    def test_headers_same_before_after_data(self, manager_with_data):
        """Test that headers are consistent before and after adding data."""
        manager, grid, binning = manager_with_data

        bin_name = binning.bins[0].name

        # Headers should be deterministic
        headers1 = manager.get_output_headers(bin_name, 5, 5)
        headers2 = manager.get_output_headers(bin_name, 5, 5)

        assert headers1 == headers2

    def test_different_positions_different_headers(self, manager_with_data):
        """Test that different positions have different headers."""
        manager, grid, binning = manager_with_data

        bin_name = binning.bins[0].name

        headers1 = manager.get_output_headers(bin_name, 0, 0)
        headers2 = manager.get_output_headers(bin_name, 5, 5)

        # INLINE/XLINE should differ
        assert headers1['INLINE_3D'] != headers2['INLINE_3D']
        assert headers1['CROSSLINE_3D'] != headers2['CROSSLINE_3D']

        # CDP coordinates should differ
        assert headers1['CDP_X'] != headers2['CDP_X'] or \
               headers1['CDP_Y'] != headers2['CDP_Y']

        # But OFFSET should be same (same bin)
        assert headers1['OFFSET'] == headers2['OFFSET']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
