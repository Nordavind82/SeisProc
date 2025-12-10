"""
Tests for GeometryPreprocessor.

Tests Tasks 1.2-1.5 of the PSTM redesign plan:
- 1.2: Midpoint-to-grid mapping
- 1.3: Traveltime computation
- 1.4: Weight computation
- 1.5: PrecomputedGeometry container
"""

import pytest
import numpy as np
import torch

from processors.migration.geometry_preprocessor import (
    compute_output_indices,
    compute_traveltimes,
    compute_weights,
    GeometryPreprocessor,
    PrecomputedGeometry,
)
from tests.fixtures.synthetic_diffractor import create_diffractor_dataset


class TestComputeOutputIndices:
    """Test Task 1.2: Midpoint-to-grid mapping."""

    def test_center_point_maps_correctly(self):
        """Trace at grid center should map to center indices."""
        # Grid: 100x100, spacing 50m, origin at (2500, 2500)
        # Center at (5000, 5000) should map to (50, 50)
        midpoint_x = np.array([5000.0], dtype=np.float32)
        midpoint_y = np.array([5000.0], dtype=np.float32)

        output_il, output_xl, valid, image_x, image_y = compute_output_indices(
            midpoint_x, midpoint_y,
            origin_x=2500.0, origin_y=2500.0,
            il_spacing=50.0, xl_spacing=50.0,
            azimuth_deg=0.0,
            n_il=100, n_xl=100
        )

        assert output_il[0] == 50
        assert output_xl[0] == 50
        assert valid[0] == True

    def test_origin_maps_to_zero(self):
        """Trace at grid origin should map to (0, 0)."""
        midpoint_x = np.array([2500.0], dtype=np.float32)
        midpoint_y = np.array([2500.0], dtype=np.float32)

        output_il, output_xl, valid, _, _ = compute_output_indices(
            midpoint_x, midpoint_y,
            origin_x=2500.0, origin_y=2500.0,
            il_spacing=50.0, xl_spacing=50.0,
            azimuth_deg=0.0,
            n_il=100, n_xl=100
        )

        assert output_il[0] == 0
        assert output_xl[0] == 0
        assert valid[0] == True

    def test_outside_grid_marked_invalid(self):
        """Traces outside grid should be marked invalid."""
        midpoint_x = np.array([0.0, 10000.0], dtype=np.float32)  # Way outside
        midpoint_y = np.array([0.0, 10000.0], dtype=np.float32)

        _, _, valid, _, _ = compute_output_indices(
            midpoint_x, midpoint_y,
            origin_x=2500.0, origin_y=2500.0,
            il_spacing=50.0, xl_spacing=50.0,
            azimuth_deg=0.0,
            n_il=100, n_xl=100
        )

        assert valid[0] == False
        assert valid[1] == False

    def test_rotated_grid(self):
        """Test grid rotation with 90 degree azimuth."""
        # With azimuth=90 (east), inline direction is east
        # A point 100m east of origin should be at il=2, xl=0
        midpoint_x = np.array([2600.0], dtype=np.float32)  # 100m east
        midpoint_y = np.array([2500.0], dtype=np.float32)  # same y

        output_il, output_xl, valid, _, _ = compute_output_indices(
            midpoint_x, midpoint_y,
            origin_x=2500.0, origin_y=2500.0,
            il_spacing=50.0, xl_spacing=50.0,
            azimuth_deg=90.0,  # Inline direction is east
            n_il=100, n_xl=100
        )

        assert output_il[0] == 2  # 100m / 50m = 2 inlines
        assert output_xl[0] == 0
        assert valid[0] == True

    def test_synthetic_dataset_mapping(self):
        """Test with synthetic diffractor dataset."""
        dataset = create_diffractor_dataset()

        output_il, output_xl, valid, image_x, image_y = compute_output_indices(
            dataset.midpoint_x, dataset.midpoint_y,
            origin_x=dataset.origin_x, origin_y=dataset.origin_y,
            il_spacing=dataset.il_spacing, xl_spacing=dataset.xl_spacing,
            azimuth_deg=0.0,
            n_il=dataset.n_il, n_xl=dataset.n_xl
        )

        # All traces should be valid (synthetic grid matches output grid)
        assert valid.sum() == dataset.n_traces

        # Check diffractor location trace
        diff_trace_idx = 50 * dataset.n_xl + 50  # il=50, xl=50
        assert output_il[diff_trace_idx] == 50
        assert output_xl[diff_trace_idx] == 50


class TestComputeTraveltimes:
    """Test Task 1.3: Traveltime computation."""

    def test_zero_offset_at_image_point(self):
        """Zero-offset trace directly above image point."""
        # Source = receiver = image point at (0, 0)
        source_x = torch.tensor([0.0])
        source_y = torch.tensor([0.0])
        receiver_x = torch.tensor([0.0])
        receiver_y = torch.tensor([0.0])
        image_x = torch.tensor([0.0])
        image_y = torch.tensor([0.0])

        # Depth = 1500m, velocity = 3000 m/s
        # Expected: t = 2 * 1500 / 3000 = 1.0 s = 1000 ms
        depth_axis = torch.tensor([1500.0])
        velocity = 3000.0

        traveltimes = compute_traveltimes(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis, velocity
        )

        assert traveltimes.shape == (1, 1)
        assert abs(traveltimes[0, 0].item() - 1000.0) < 0.01

    def test_zero_offset_with_offset_from_image(self):
        """Zero-offset trace offset horizontally from image point."""
        # Midpoint at (1000, 0), image at (0, 0)
        source_x = torch.tensor([1000.0])
        source_y = torch.tensor([0.0])
        receiver_x = torch.tensor([1000.0])
        receiver_y = torch.tensor([0.0])
        image_x = torch.tensor([0.0])
        image_y = torch.tensor([0.0])

        # h = 1000m, z = 1500m, v = 3000 m/s
        # r = sqrt(1000^2 + 1500^2) = sqrt(1000000 + 2250000) = sqrt(3250000) = 1802.8m
        # t = 2 * r / v = 2 * 1802.8 / 3000 = 1.202 s = 1202 ms
        depth_axis = torch.tensor([1500.0])
        velocity = 3000.0

        traveltimes = compute_traveltimes(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis, velocity
        )

        expected = 2 * np.sqrt(1000**2 + 1500**2) / velocity * 1000
        assert abs(traveltimes[0, 0].item() - expected) < 0.1

    def test_multiple_depths(self):
        """Test traveltime computation for multiple depths."""
        source_x = torch.tensor([0.0])
        source_y = torch.tensor([0.0])
        receiver_x = torch.tensor([0.0])
        receiver_y = torch.tensor([0.0])
        image_x = torch.tensor([0.0])
        image_y = torch.tensor([0.0])

        depths = torch.tensor([500.0, 1000.0, 1500.0, 2000.0])
        velocity = 3000.0

        traveltimes = compute_traveltimes(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depths, velocity
        )

        assert traveltimes.shape == (1, 4)

        # Check each depth
        for i, z in enumerate(depths):
            expected = 2 * z.item() / velocity * 1000
            assert abs(traveltimes[0, i].item() - expected) < 0.01

    def test_diffractor_dataset_traveltime(self):
        """Test traveltimes with synthetic diffractor dataset."""
        dataset = create_diffractor_dataset()

        # Get trace at diffractor location
        diff_trace_idx = 50 * dataset.n_xl + 50

        source_x = torch.tensor([dataset.source_x[diff_trace_idx]])
        source_y = torch.tensor([dataset.source_y[diff_trace_idx]])
        receiver_x = torch.tensor([dataset.receiver_x[diff_trace_idx]])
        receiver_y = torch.tensor([dataset.receiver_y[diff_trace_idx]])
        image_x = torch.tensor([dataset.diffractor_x])
        image_y = torch.tensor([dataset.diffractor_y])

        # Depth at diffractor
        depth_axis = torch.tensor([dataset.diffractor_z])

        traveltimes = compute_traveltimes(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis, dataset.velocity
        )

        # Should match expected diffractor time
        expected = dataset.expected_diffractor_time_ms
        assert abs(traveltimes[0, 0].item() - expected) < 1.0  # Within 1ms


class TestComputeWeights:
    """Test Task 1.4: Weight computation."""

    def test_maximum_weight_at_zero_offset(self):
        """Weight should be maximum for trace directly above image point."""
        source_x = torch.tensor([0.0])
        source_y = torch.tensor([0.0])
        receiver_x = torch.tensor([0.0])
        receiver_y = torch.tensor([0.0])
        image_x = torch.tensor([0.0])
        image_y = torch.tensor([0.0])

        depth_axis = torch.tensor([1500.0])

        weights, mask = compute_weights(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis,
            max_aperture_m=5000.0, max_angle_deg=60.0
        )

        assert mask[0, 0] == True
        assert weights[0, 0] > 0

    def test_weight_decreases_with_offset(self):
        """Weight should decrease as horizontal offset increases."""
        # Multiple traces at increasing distances
        distances = [0, 500, 1000, 1500]
        source_x = torch.tensor(distances, dtype=torch.float32)
        source_y = torch.zeros(4)
        receiver_x = source_x.clone()
        receiver_y = source_y.clone()
        image_x = torch.zeros(4)
        image_y = torch.zeros(4)

        depth_axis = torch.tensor([1500.0])

        weights, _ = compute_weights(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis,
            max_aperture_m=5000.0, max_angle_deg=60.0
        )

        # Weight should decrease monotonically
        for i in range(len(distances) - 1):
            assert weights[i, 0] >= weights[i + 1, 0]

    def test_aperture_mask_excludes_distant_traces(self):
        """Traces beyond max aperture should be masked out."""
        source_x = torch.tensor([0.0, 2000.0, 6000.0])
        source_y = torch.zeros(3)
        receiver_x = source_x.clone()
        receiver_y = source_y.clone()
        image_x = torch.zeros(3)
        image_y = torch.zeros(3)

        depth_axis = torch.tensor([3000.0])  # Deeper to keep angles reasonable

        weights, mask = compute_weights(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis,
            max_aperture_m=5000.0, max_angle_deg=60.0
        )

        # First two should be in aperture, third outside
        assert mask[0, 0] == True   # 0m offset
        assert mask[1, 0] == True   # 2000m offset, angle ~34°
        assert mask[2, 0] == False  # 6000m offset > 5000m aperture

    def test_angle_mask_excludes_steep_angles(self):
        """Traces with steep angles should be masked out."""
        # At shallow depth, even small offsets create steep angles
        source_x = torch.tensor([0.0, 500.0, 1000.0])
        source_y = torch.zeros(3)
        receiver_x = source_x.clone()
        receiver_y = source_y.clone()
        image_x = torch.zeros(3)
        image_y = torch.zeros(3)

        # Very shallow depth - angles will be steep
        depth_axis = torch.tensor([100.0])

        weights, mask = compute_weights(
            source_x, source_y, receiver_x, receiver_y,
            image_x, image_y, depth_axis,
            max_aperture_m=5000.0, max_angle_deg=45.0  # Restrictive angle
        )

        # At z=100m with 45 deg max angle:
        # max h = z * tan(45) = 100m
        assert mask[0, 0] == True   # 0m offset - OK
        assert mask[1, 0] == False  # 500m offset at z=100 -> angle > 45
        assert mask[2, 0] == False  # 1000m offset at z=100 -> angle > 45


class TestGeometryPreprocessor:
    """Test Task 1.5: Full GeometryPreprocessor."""

    def test_precompute_returns_correct_shapes(self):
        """Verify all output tensors have correct shapes."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()

        geometry = preprocessor.precompute(
            source_x=dataset.source_x,
            source_y=dataset.source_y,
            receiver_x=dataset.receiver_x,
            receiver_y=dataset.receiver_y,
            origin_x=dataset.origin_x,
            origin_y=dataset.origin_y,
            il_spacing=dataset.il_spacing,
            xl_spacing=dataset.xl_spacing,
            azimuth_deg=0.0,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_samples=dataset.n_samples,
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=torch.device('cpu'),
        )

        n_traces = dataset.n_traces
        n_depths = dataset.n_samples

        # Check shapes
        assert geometry.output_il.shape == (n_traces,)
        assert geometry.output_xl.shape == (n_traces,)
        assert geometry.valid_mask.shape == (n_traces,)
        assert geometry.image_x.shape == (n_traces,)
        assert geometry.image_y.shape == (n_traces,)
        assert geometry.traveltimes_ms.shape == (n_traces, n_depths)
        assert geometry.weights.shape == (n_traces, n_depths)
        assert geometry.aperture_mask.shape == (n_traces, n_depths)

    def test_precompute_diffractor_traveltime_correct(self):
        """Verify traveltime at diffractor location is correct."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()

        geometry = preprocessor.precompute(
            source_x=dataset.source_x,
            source_y=dataset.source_y,
            receiver_x=dataset.receiver_x,
            receiver_y=dataset.receiver_y,
            origin_x=dataset.origin_x,
            origin_y=dataset.origin_y,
            il_spacing=dataset.il_spacing,
            xl_spacing=dataset.xl_spacing,
            azimuth_deg=0.0,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_samples=dataset.n_samples,
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=torch.device('cpu'),
        )

        # Find trace at diffractor location (il=50, xl=50)
        diff_trace_idx = 50 * dataset.n_xl + 50

        # Get traveltime at diffractor depth (sample 500 = 1000ms)
        diff_sample = int(dataset.expected_diffractor_time_ms / dataset.dt_ms)
        traveltime = geometry.traveltimes_ms[diff_trace_idx, diff_sample].item()

        # At this trace/depth, traveltime should equal the time (zero offset above diffractor)
        expected = dataset.expected_diffractor_time_ms
        assert abs(traveltime - expected) < 1.0  # Within 1ms

    def test_precompute_memory_estimate(self):
        """Verify memory estimate is reasonable."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()

        geometry = preprocessor.precompute(
            source_x=dataset.source_x,
            source_y=dataset.source_y,
            receiver_x=dataset.receiver_x,
            receiver_y=dataset.receiver_y,
            origin_x=dataset.origin_x,
            origin_y=dataset.origin_y,
            il_spacing=dataset.il_spacing,
            xl_spacing=dataset.xl_spacing,
            azimuth_deg=0.0,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_samples=dataset.n_samples,
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=torch.device('cpu'),
        )

        mem_bytes = geometry.memory_bytes()
        mem_mb = mem_bytes / (1024 * 1024)

        # 10,000 traces x 1501 depths
        # Expected: ~5 tensors of (10000, 1501) float32 = 5 * 10000 * 1501 * 4 = 300 MB
        # Plus index arrays ~5 * 10000 * 4 = 0.2 MB
        # Total should be 200-400 MB range
        assert 100 < mem_mb < 500

    def test_precompute_batched(self):
        """Test batched precomputation produces same results."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()

        # Non-batched
        geometry1 = preprocessor.precompute(
            source_x=dataset.source_x,
            source_y=dataset.source_y,
            receiver_x=dataset.receiver_x,
            receiver_y=dataset.receiver_y,
            origin_x=dataset.origin_x,
            origin_y=dataset.origin_y,
            il_spacing=dataset.il_spacing,
            xl_spacing=dataset.xl_spacing,
            azimuth_deg=0.0,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_samples=dataset.n_samples,
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=torch.device('cpu'),
        )

        # Batched with small batch size
        geometry2 = preprocessor.precompute_batched(
            source_x=dataset.source_x,
            source_y=dataset.source_y,
            receiver_x=dataset.receiver_x,
            receiver_y=dataset.receiver_y,
            origin_x=dataset.origin_x,
            origin_y=dataset.origin_y,
            il_spacing=dataset.il_spacing,
            xl_spacing=dataset.xl_spacing,
            azimuth_deg=0.0,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_samples=dataset.n_samples,
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            device=torch.device('cpu'),
            batch_size=2000,  # Force batching
        )

        # Results should match
        assert torch.allclose(geometry1.traveltimes_ms, geometry2.traveltimes_ms, atol=1e-5)
        assert torch.equal(geometry1.output_il, geometry2.output_il)
        assert torch.equal(geometry1.output_xl, geometry2.output_xl)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
