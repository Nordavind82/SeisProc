"""
Unit tests for offset-azimuth binning system.

Tests:
- OffsetAzimuthBin creation and validation
- BinningTable operations
- Trace assignment to bins
- Coverage and overlap analysis
- Binning presets
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.binning import (
    OffsetAzimuthBin,
    BinningTable,
    BinningPreset,
    create_common_offset_binning,
    create_uniform_offset_binning,
    create_ovt_binning,
    create_narrow_azimuth_binning,
    create_full_stack_binning,
)
from utils.binning_presets import (
    get_preset,
    list_presets,
    get_preset_description,
    create_custom_offset_binning,
    suggest_binning,
)


class TestOffsetAzimuthBin:
    """Tests for OffsetAzimuthBin class."""

    def test_basic_creation(self):
        """Test basic bin creation."""
        bin_def = OffsetAzimuthBin(
            name="test_bin",
            offset_min=0.0,
            offset_max=1000.0,
        )

        assert bin_def.name == "test_bin"
        assert bin_def.offset_min == 0.0
        assert bin_def.offset_max == 1000.0
        assert bin_def.azimuth_min == 0.0
        assert bin_def.azimuth_max == 360.0
        assert bin_def.enabled is True

    def test_offset_center(self):
        """Test offset center calculation."""
        bin_def = OffsetAzimuthBin(
            name="test",
            offset_min=200.0,
            offset_max=400.0,
        )
        assert bin_def.offset_center == 300.0
        assert bin_def.offset_width == 200.0

    def test_azimuth_center(self):
        """Test azimuth center calculation."""
        # Normal case
        bin_def = OffsetAzimuthBin(
            name="test",
            offset_min=0.0,
            offset_max=1000.0,
            azimuth_min=45.0,
            azimuth_max=135.0,
        )
        assert bin_def.azimuth_center == 90.0
        assert bin_def.azimuth_width == 90.0

    def test_full_azimuth_detection(self):
        """Test full azimuth range detection."""
        full = OffsetAzimuthBin(
            name="full",
            offset_min=0.0,
            offset_max=1000.0,
            azimuth_min=0.0,
            azimuth_max=360.0,
        )
        assert full.is_full_azimuth is True

        partial = OffsetAzimuthBin(
            name="partial",
            offset_min=0.0,
            offset_max=1000.0,
            azimuth_min=0.0,
            azimuth_max=90.0,
        )
        assert partial.is_full_azimuth is False

    def test_contains_offset(self):
        """Test trace containment check for offset."""
        bin_def = OffsetAzimuthBin(
            name="test",
            offset_min=200.0,
            offset_max=400.0,
        )

        # Inside
        assert bin_def.contains(300.0, 0.0) is True
        assert bin_def.contains(200.0, 0.0) is True  # min is inclusive

        # Outside
        assert bin_def.contains(100.0, 0.0) is False
        assert bin_def.contains(400.0, 0.0) is False  # max is exclusive
        assert bin_def.contains(500.0, 0.0) is False

    def test_contains_azimuth(self):
        """Test trace containment check for azimuth."""
        bin_def = OffsetAzimuthBin(
            name="test",
            offset_min=0.0,
            offset_max=1000.0,
            azimuth_min=45.0,
            azimuth_max=135.0,
        )

        # Inside
        assert bin_def.contains(500.0, 90.0) is True
        assert bin_def.contains(500.0, 45.0) is True

        # Outside
        assert bin_def.contains(500.0, 180.0) is False
        assert bin_def.contains(500.0, 0.0) is False

    def test_contains_azimuth_wraparound(self):
        """Test azimuth wrap-around (e.g., 350-10 degrees)."""
        bin_def = OffsetAzimuthBin(
            name="wrap",
            offset_min=0.0,
            offset_max=1000.0,
            azimuth_min=350.0,
            azimuth_max=10.0,  # Wraps around 0
        )

        # Inside (near 0)
        assert bin_def.contains(500.0, 0.0) is True
        assert bin_def.contains(500.0, 5.0) is True
        assert bin_def.contains(500.0, 355.0) is True

        # Outside
        assert bin_def.contains(500.0, 180.0) is False
        assert bin_def.contains(500.0, 20.0) is False

    def test_contains_batch(self):
        """Test vectorized containment check."""
        bin_def = OffsetAzimuthBin(
            name="test",
            offset_min=200.0,
            offset_max=400.0,
            azimuth_min=0.0,
            azimuth_max=180.0,
        )

        offsets = np.array([100.0, 300.0, 300.0, 500.0])
        azimuths = np.array([90.0, 90.0, 270.0, 90.0])

        result = bin_def.contains_batch(offsets, azimuths)

        assert result.tolist() == [False, True, False, False]

    def test_validation_errors(self):
        """Test validation catches invalid parameters."""
        # Negative offset
        with pytest.raises(ValueError):
            OffsetAzimuthBin(name="bad", offset_min=-100.0, offset_max=100.0)

        # offset_max <= offset_min
        with pytest.raises(ValueError):
            OffsetAzimuthBin(name="bad", offset_min=500.0, offset_max=200.0)

        # Empty name
        with pytest.raises(ValueError):
            OffsetAzimuthBin(name="", offset_min=0.0, offset_max=100.0)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        original = OffsetAzimuthBin(
            name="test_bin",
            offset_min=100.0,
            offset_max=500.0,
            azimuth_min=45.0,
            azimuth_max=135.0,
            enabled=False,
            metadata={'note': 'test'},
        )

        d = original.to_dict()
        restored = OffsetAzimuthBin.from_dict(d)

        assert restored.name == original.name
        assert restored.offset_min == original.offset_min
        assert restored.offset_max == original.offset_max
        assert restored.azimuth_min == original.azimuth_min
        assert restored.azimuth_max == original.azimuth_max
        assert restored.enabled == original.enabled
        assert restored.metadata == original.metadata


class TestBinningTable:
    """Tests for BinningTable class."""

    def test_empty_table(self):
        """Test empty table creation."""
        table = BinningTable(name="Empty")
        assert table.n_bins == 0
        assert len(table) == 0

    def test_add_bins(self):
        """Test adding bins to table."""
        table = BinningTable(name="Test")

        table.add_bin(OffsetAzimuthBin(
            name="near",
            offset_min=0.0,
            offset_max=500.0,
        ))
        table.add_bin(OffsetAzimuthBin(
            name="far",
            offset_min=500.0,
            offset_max=1000.0,
        ))

        assert table.n_bins == 2
        assert table.get_bin("near") is not None
        assert table.get_bin("far") is not None

    def test_duplicate_name_error(self):
        """Test that duplicate bin names are rejected."""
        table = BinningTable(name="Test")
        table.add_bin(OffsetAzimuthBin(
            name="test",
            offset_min=0.0,
            offset_max=500.0,
        ))

        with pytest.raises(ValueError):
            table.add_bin(OffsetAzimuthBin(
                name="test",  # Duplicate
                offset_min=500.0,
                offset_max=1000.0,
            ))

    def test_remove_bin(self):
        """Test removing bins."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=100.0),
            OffsetAzimuthBin(name="b", offset_min=100.0, offset_max=200.0),
        ])

        assert table.remove_bin("a") is True
        assert table.n_bins == 1
        assert table.get_bin("a") is None

        assert table.remove_bin("nonexistent") is False

    def test_offset_range(self):
        """Test offset range property."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=100.0, offset_max=300.0),
            OffsetAzimuthBin(name="b", offset_min=500.0, offset_max=800.0),
        ])

        min_off, max_off = table.offset_range
        assert min_off == 100.0
        assert max_off == 800.0

    def test_assign_trace(self):
        """Test single trace assignment."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="near", offset_min=0.0, offset_max=500.0),
            OffsetAzimuthBin(name="mid", offset_min=500.0, offset_max=1000.0),
            OffsetAzimuthBin(name="far", offset_min=1000.0, offset_max=2000.0),
        ])

        # Single bin
        assert table.assign_trace(250.0, 0.0) == ["near"]
        assert table.assign_trace(750.0, 0.0) == ["mid"]
        assert table.assign_trace(1500.0, 0.0) == ["far"]

        # No bin
        assert table.assign_trace(3000.0, 0.0) == []

    def test_assign_traces_batch(self):
        """Test batch trace assignment."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="near", offset_min=0.0, offset_max=500.0),
            OffsetAzimuthBin(name="far", offset_min=500.0, offset_max=1000.0),
        ])

        offsets = np.array([100.0, 600.0, 300.0, 1500.0])
        azimuths = np.zeros(4)

        assignments = table.assign_traces_batch(offsets, azimuths)

        assert assignments["near"].tolist() == [True, False, True, False]
        assert assignments["far"].tolist() == [False, True, False, False]

    def test_enabled_bins(self):
        """Test enabled/disabled bin filtering."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=100.0, enabled=True),
            OffsetAzimuthBin(name="b", offset_min=100.0, offset_max=200.0, enabled=False),
            OffsetAzimuthBin(name="c", offset_min=200.0, offset_max=300.0, enabled=True),
        ])

        assert table.n_bins == 3
        assert table.n_enabled_bins == 2
        assert [b.name for b in table.enabled_bins] == ["a", "c"]

    def test_check_coverage(self):
        """Test coverage analysis."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=500.0),
            OffsetAzimuthBin(name="b", offset_min=500.0, offset_max=1000.0),
        ])

        offsets = np.array([100.0, 600.0, 1500.0, 300.0])
        azimuths = np.zeros(4)

        coverage = table.check_coverage(offsets, azimuths)

        assert coverage['n_traces'] == 4
        assert coverage['n_unassigned'] == 1  # offset=1500
        assert coverage['bin_counts']['a'] == 2
        assert coverage['bin_counts']['b'] == 1
        assert coverage['coverage_percent'] == 75.0

    def test_check_overlaps(self):
        """Test overlap detection."""
        # No overlaps
        table1 = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=500.0),
            OffsetAzimuthBin(name="b", offset_min=500.0, offset_max=1000.0),
        ])
        assert table1.check_overlaps() == []

        # With overlap
        table2 = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=600.0),
            OffsetAzimuthBin(name="b", offset_min=400.0, offset_max=1000.0),
        ])
        overlaps = table2.check_overlaps()
        assert len(overlaps) == 1
        assert overlaps[0] == ("a", "b")

    def test_check_gaps(self):
        """Test gap detection."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=300.0),
            OffsetAzimuthBin(name="b", offset_min=500.0, offset_max=800.0),
        ])

        gaps = table.check_gaps(offset_step=50.0)
        assert len(gaps) == 1
        assert gaps[0][0] >= 300.0
        assert gaps[0][1] <= 500.0

    def test_serialization(self):
        """Test table serialization."""
        original = BinningTable(
            name="Test Table",
            bins=[
                OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=500.0),
                OffsetAzimuthBin(name="b", offset_min=500.0, offset_max=1000.0),
            ],
            preset=BinningPreset.COMMON_OFFSET,
            metadata={'version': 1},
        )

        # to_dict / from_dict
        d = original.to_dict()
        restored = BinningTable.from_dict(d)

        assert restored.name == original.name
        assert restored.n_bins == original.n_bins
        assert restored.preset == original.preset
        assert restored.bins[0].name == "a"

    def test_json_serialization(self):
        """Test JSON serialization."""
        original = BinningTable(
            name="JSON Test",
            bins=[
                OffsetAzimuthBin(name="test", offset_min=0.0, offset_max=500.0),
            ],
        )

        json_str = original.to_json()
        restored = BinningTable.from_json(json_str)

        assert restored.name == original.name
        assert restored.n_bins == 1

    def test_file_save_load(self):
        """Test saving and loading from file."""
        original = BinningTable(
            name="File Test",
            bins=[
                OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=500.0),
                OffsetAzimuthBin(name="b", offset_min=500.0, offset_max=1000.0),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            original.save(filepath)
            restored = BinningTable.load(filepath)

            assert restored.name == original.name
            assert restored.n_bins == original.n_bins
        finally:
            Path(filepath).unlink()

    def test_iteration(self):
        """Test table iteration."""
        table = BinningTable(name="Test", bins=[
            OffsetAzimuthBin(name="a", offset_min=0.0, offset_max=100.0),
            OffsetAzimuthBin(name="b", offset_min=100.0, offset_max=200.0),
        ])

        names = [b.name for b in table]
        assert names == ["a", "b"]


class TestBinningFactories:
    """Tests for binning factory functions."""

    def test_common_offset_binning(self):
        """Test common offset binning creation."""
        ranges = [(0, 200), (200, 500), (500, 1000)]
        table = create_common_offset_binning(ranges)

        assert table.n_bins == 3
        assert table.preset == BinningPreset.COMMON_OFFSET
        assert all(b.is_full_azimuth for b in table)

    def test_uniform_offset_binning(self):
        """Test uniform offset binning."""
        table = create_uniform_offset_binning(
            offset_min=0.0,
            offset_max=1000.0,
            n_bins=5,
        )

        assert table.n_bins == 5
        # Check uniform widths
        widths = [b.offset_width for b in table]
        assert all(abs(w - 200.0) < 0.1 for w in widths)

    def test_ovt_binning(self):
        """Test OVT binning creation."""
        ranges = [(0, 500), (500, 1000)]
        table = create_ovt_binning(ranges, n_azimuth_sectors=4)

        assert table.n_bins == 8  # 2 offset x 4 azimuth
        assert table.preset == BinningPreset.OVT

        # Check azimuth sectors
        az_widths = [b.azimuth_width for b in table]
        assert all(abs(w - 90.0) < 0.1 for w in az_widths)

    def test_narrow_azimuth_binning(self):
        """Test narrow azimuth binning."""
        table = create_narrow_azimuth_binning(
            offset_min=0.0,
            offset_max=5000.0,
            inline_azimuth=0.0,
            azimuth_width=20.0,
        )

        assert table.n_bins == 4
        assert table.preset == BinningPreset.NARROW_AZIMUTH

        # Check bin names
        names = [b.name for b in table]
        assert "inline_pos" in names
        assert "xline_pos" in names

    def test_full_stack_binning(self):
        """Test full stack binning."""
        table = create_full_stack_binning(offset_max=5000.0)

        assert table.n_bins == 1
        assert table.bins[0].offset_max == 5000.0
        assert table.bins[0].is_full_azimuth


class TestBinningPresets:
    """Tests for binning presets."""

    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_presets()
        assert 'land_3d' in presets
        assert 'marine' in presets
        assert 'wide_azimuth_ovt' in presets

    def test_get_preset_land_3d(self):
        """Test land 3D preset."""
        table = get_preset('land_3d')
        assert table.n_bins == 10
        assert table.preset == BinningPreset.COMMON_OFFSET

    def test_get_preset_marine(self):
        """Test marine preset."""
        table = get_preset('marine')
        assert table.n_bins == 6

    def test_get_preset_ovt(self):
        """Test OVT preset."""
        table = get_preset('wide_azimuth_ovt')
        assert table.n_bins == 16  # 4 offset x 4 azimuth
        assert table.preset == BinningPreset.OVT

    def test_get_preset_unknown(self):
        """Test error for unknown preset."""
        with pytest.raises(ValueError):
            get_preset('nonexistent')

    def test_get_preset_description(self):
        """Test preset descriptions."""
        desc = get_preset_description('land_3d')
        assert 'offset' in desc.lower()

    def test_custom_offset_binning(self):
        """Test custom offset binning creation."""
        table = create_custom_offset_binning(
            offset_min=0.0,
            offset_max=2000.0,
            n_bins=4,
            logarithmic=False,
        )
        assert table.n_bins == 4

    def test_suggest_binning(self):
        """Test automatic binning suggestion."""
        offsets = np.random.uniform(0, 3000, 1000)
        azimuths = np.random.uniform(0, 360, 1000)

        table = suggest_binning(
            offsets,
            azimuths,
            target_traces_per_bin=200,
        )

        # Should create ~5 bins for 1000 traces / 200 per bin
        assert 3 <= table.n_bins <= 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
