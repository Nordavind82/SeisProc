"""
Unit tests for DepthAdaptiveAperture.

Tests:
- Aperture calculation at different depths
- Depth grouping logic
- Contributing trace selection
- Reduction ratio estimation
- Integration with existing aperture controller
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.aperture_adaptive import (
    DepthAdaptiveAperture,
    DepthGroup,
    create_depth_adaptive_aperture,
)
from models.migration_config import MigrationConfig, OutputGrid


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


class TestApertureCalculation:
    """Tests for depth-dependent aperture calculation."""

    def test_aperture_increases_with_depth(self):
        """Effective aperture should increase with depth."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )

        z_axis = np.linspace(100, 5000, 50)
        apertures = aperture.compute_depth_apertures(z_axis)

        # Aperture should be monotonically increasing
        for i in range(len(apertures) - 1):
            assert apertures[i] <= apertures[i + 1]

    def test_aperture_limited_by_max(self):
        """Aperture should not exceed max_aperture_m."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=3000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )

        z_axis = np.linspace(100, 10000, 100)
        apertures = aperture.compute_depth_apertures(z_axis)

        assert np.all(apertures <= 3000.0)

    def test_aperture_formula(self):
        """Test that aperture follows h = z * tan(θ) formula."""
        max_angle_deg = 45.0
        aperture = DepthAdaptiveAperture(
            max_aperture_m=10000.0,  # Large enough to not limit
            max_angle_deg=max_angle_deg,
            device=TEST_DEVICE,
        )

        z_axis = np.array([1000, 2000, 3000])
        apertures = aperture.compute_depth_apertures(z_axis)

        # At 45 degrees, tan(45°) = 1, so h = z
        expected = z_axis * np.tan(np.radians(max_angle_deg))

        np.testing.assert_allclose(apertures, expected, rtol=1e-5)

    def test_shallow_aperture_small(self):
        """Shallow depths should have small effective aperture."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )

        z_axis = np.array([100, 500, 1000, 2000])
        apertures = aperture.compute_depth_apertures(z_axis)

        # At z=100m, 60°, aperture = 100 * tan(60°) ≈ 173m
        assert apertures[0] < 200
        assert apertures[0] < apertures[-1]


class TestDepthGrouping:
    """Tests for depth grouping logic."""

    @pytest.fixture
    def aperture_controller(self):
        """Create aperture controller with computed apertures."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )
        z_axis = np.linspace(100, 5000, 500)
        aperture.compute_depth_apertures(z_axis)
        return aperture

    def test_grouping_creates_groups(self, aperture_controller):
        """Test that grouping creates depth groups."""
        groups = aperture_controller.group_depths_by_aperture(n_groups=5)

        assert len(groups) > 0
        assert len(groups) <= 5

    def test_groups_cover_all_depths(self, aperture_controller):
        """Groups should cover all depth indices."""
        groups = aperture_controller.group_depths_by_aperture(n_groups=5)

        # Collect all covered indices
        covered = set()
        for group in groups:
            for idx in range(group.z_start, group.z_end):
                covered.add(idx)

        # Should cover all depths
        expected = set(range(500))
        assert covered == expected

    def test_groups_non_overlapping(self, aperture_controller):
        """Groups should not overlap."""
        groups = aperture_controller.group_depths_by_aperture(n_groups=5)

        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i != j:
                    # No overlap
                    assert g1.z_end <= g2.z_start or g2.z_end <= g1.z_start

    def test_group_aperture_increases(self, aperture_controller):
        """Later groups should have larger apertures."""
        groups = aperture_controller.group_depths_by_aperture(n_groups=5)

        for i in range(len(groups) - 1):
            assert groups[i].effective_aperture <= groups[i + 1].effective_aperture

    def test_single_group_all_same_depth(self):
        """Single depth should create single group."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )

        z_axis = np.array([1000.0])  # Single depth
        aperture.compute_depth_apertures(z_axis)
        groups = aperture.group_depths_by_aperture(n_groups=5)

        assert len(groups) == 1
        assert groups[0].z_start == 0
        assert groups[0].z_end == 1


class TestContributingTraces:
    """Tests for contributing trace selection."""

    @pytest.fixture
    def aperture_with_groups(self):
        """Create aperture controller with groups."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            min_offset_m=100.0,
            max_offset_m=8000.0,
            device=TEST_DEVICE,
        )
        z_axis = np.linspace(100, 5000, 100)
        aperture.compute_depth_apertures(z_axis)
        aperture.group_depths_by_aperture(n_groups=3)
        return aperture

    def test_close_traces_contribute(self, aperture_with_groups):
        """Traces close to image point should contribute."""
        groups = aperture_with_groups.depth_groups

        # Create traces close to image center
        h_src = np.array([100, 200, 300, 500, 1000])
        h_rcv = np.array([100, 200, 300, 500, 1000])

        # Get contributing traces for first group (smallest aperture)
        indices = aperture_with_groups.get_contributing_trace_indices(
            groups[0], h_src, h_rcv
        )

        # Some traces should contribute
        assert len(indices) > 0

    def test_far_traces_excluded(self, aperture_with_groups):
        """Traces far from image point should be excluded from shallow groups."""
        groups = aperture_with_groups.depth_groups

        # Create traces all very far away
        h_src = np.array([4000, 4500, 5000, 6000, 7000])
        h_rcv = np.array([4000, 4500, 5000, 6000, 7000])

        # Get contributing traces for first group (small aperture)
        indices = aperture_with_groups.get_contributing_trace_indices(
            groups[0], h_src, h_rcv
        )

        # Far traces should not contribute to shallow depths
        # First group has small aperture < 300m typically
        assert len(indices) == 0 or groups[0].effective_aperture >= 4000

    def test_offset_filtering(self, aperture_with_groups):
        """Offset constraints should filter traces."""
        groups = aperture_with_groups.depth_groups[-1]  # Use group with large aperture

        # Traces with various offsets
        h_src = np.array([100, 100, 100, 100])
        h_rcv = np.array([100, 100, 100, 100])
        offset = np.array([50, 500, 5000, 10000])  # 50 too small, 10000 too large

        indices = aperture_with_groups.get_contributing_trace_indices(
            groups, h_src, h_rcv, offset
        )

        # Only offset=500 and offset=5000 should pass (100-8000 range)
        assert len(indices) == 2
        assert 1 in indices  # offset=500
        assert 2 in indices  # offset=5000

    def test_torch_mask_matches_numpy(self, aperture_with_groups):
        """Torch and NumPy versions should give same result."""
        groups = aperture_with_groups.depth_groups[-1]

        h_src_np = np.array([100, 500, 1000, 2000, 5000], dtype=np.float32)
        h_rcv_np = np.array([200, 400, 900, 2100, 4800], dtype=np.float32)

        h_src_t = torch.from_numpy(h_src_np)
        h_rcv_t = torch.from_numpy(h_rcv_np)

        # NumPy version
        indices_np = aperture_with_groups.get_contributing_trace_indices(
            groups, h_src_np, h_rcv_np
        )

        # Torch version
        mask_t = aperture_with_groups.get_contributing_trace_mask(
            groups, h_src_t, h_rcv_t
        )
        indices_t = torch.where(mask_t)[0].numpy()

        np.testing.assert_array_equal(indices_np, indices_t)


class TestReductionEstimation:
    """Tests for trace reduction estimation."""

    def test_reduction_ratio_computed(self):
        """Test that reduction ratio is computed."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )

        z_axis = np.linspace(100, 5000, 100)
        aperture.compute_depth_apertures(z_axis)
        aperture.group_depths_by_aperture(n_groups=3)

        # Mix of close and far traces
        h_src = np.concatenate([
            np.ones(100) * 200,   # Close
            np.ones(100) * 3000,  # Medium
            np.ones(100) * 6000,  # Far
        ])
        h_rcv = h_src.copy()

        stats = aperture.estimate_trace_reduction(h_src, h_rcv)

        assert 'reduction_ratio' in stats
        assert stats['reduction_ratio'] >= 1.0  # At least 1x (no worse than baseline)

    def test_reduction_with_varying_geometry(self):
        """Test reduction with realistic geometry spread."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=3000.0,
            max_angle_deg=45.0,
            device=TEST_DEVICE,
        )

        z_axis = np.linspace(500, 4000, 200)
        aperture.compute_depth_apertures(z_axis)
        aperture.group_depths_by_aperture(n_groups=4)

        # Realistic: traces spread from 0 to 10km
        np.random.seed(42)
        h_src = np.random.uniform(0, 10000, 1000)
        h_rcv = np.random.uniform(0, 10000, 1000)

        stats = aperture.estimate_trace_reduction(h_src, h_rcv)

        # With spread traces, should see meaningful reduction
        assert stats['reduction_ratio'] > 1.0

        # Verify group stats are present
        assert len(stats['groups']) == len(aperture.depth_groups)


class TestErrorHandling:
    """Tests for error handling."""

    def test_group_before_compute_fails(self):
        """Grouping before computing apertures should fail."""
        aperture = DepthAdaptiveAperture()

        with pytest.raises(RuntimeError, match="compute_depth_apertures"):
            aperture.group_depths_by_aperture()

    def test_estimate_before_group_fails(self):
        """Estimating before grouping should fail."""
        aperture = DepthAdaptiveAperture()
        aperture.compute_depth_apertures(np.linspace(100, 1000, 10))

        with pytest.raises(RuntimeError, match="group_depths_by_aperture"):
            aperture.estimate_trace_reduction(np.ones(10), np.ones(10))

    def test_get_depth_aperture_before_compute_fails(self):
        """Getting depth aperture before computing should fail."""
        aperture = DepthAdaptiveAperture()

        with pytest.raises(RuntimeError, match="compute_depth_apertures"):
            aperture.get_depth_aperture(0)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_factory_creates_aperture(self):
        """Factory should create valid aperture controller."""
        output_grid = OutputGrid(
            n_time=100,
            n_inline=10,
            n_xline=10,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0,
            x_origin=0.0,
            y_origin=0.0,
        )

        config = MigrationConfig(
            output_grid=output_grid,
            max_aperture_m=4000.0,
            max_angle_deg=55.0,
            min_offset_m=200.0,
            max_offset_m=6000.0,
        )

        aperture = create_depth_adaptive_aperture(config, device=TEST_DEVICE)

        assert aperture.max_aperture == 4000.0
        assert aperture.max_angle_deg == 55.0
        assert aperture.min_offset == 200.0
        assert aperture.max_offset == 6000.0


class TestPhysicalConsistency:
    """Physical consistency tests."""

    def test_vertical_angle_zero_aperture(self):
        """Zero angle should give zero aperture."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=0.0,
            device=TEST_DEVICE,
        )

        z_axis = np.array([1000, 2000, 3000])
        apertures = aperture.compute_depth_apertures(z_axis)

        # tan(0) = 0, so aperture should be 0
        np.testing.assert_allclose(apertures, 0.0, atol=1e-10)

    def test_large_angle_approaches_max(self):
        """Large angle should approach max_aperture."""
        max_ap = 3000.0
        aperture = DepthAdaptiveAperture(
            max_aperture_m=max_ap,
            max_angle_deg=89.0,  # Very large angle
            device=TEST_DEVICE,
        )

        # At shallow depth with large angle, aperture is limited by max
        z_axis = np.array([100, 200, 500])
        apertures = aperture.compute_depth_apertures(z_axis)

        # Should hit max_aperture quickly
        assert apertures[-1] == max_ap

    def test_deeper_allows_more_traces(self):
        """Deeper depth groups should allow more distant traces."""
        aperture = DepthAdaptiveAperture(
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=TEST_DEVICE,
        )

        z_axis = np.linspace(100, 5000, 100)
        aperture.compute_depth_apertures(z_axis)
        groups = aperture.group_depths_by_aperture(n_groups=3)

        # Traces at various distances
        h_src = np.linspace(100, 4000, 50)
        h_rcv = h_src.copy()

        # Count contributing traces for each group
        counts = []
        for group in groups:
            indices = aperture.get_contributing_trace_indices(group, h_src, h_rcv)
            counts.append(len(indices))

        # Later groups (deeper) should have more or equal contributing traces
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
