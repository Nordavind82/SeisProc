"""
Tests for KirchhoffKernel.

Tests Tasks 2.1-2.3 of the PSTM redesign plan:
- 2.1: Trace interpolation
- 2.2: Scatter-add kernel
- 2.3: Full migration kernel
"""

import pytest
import numpy as np
import torch

from processors.migration.kirchhoff_kernel import (
    interpolate_traces,
    scatter_add_migration,
    KirchhoffKernel,
    normalize_by_fold,
)
from processors.migration.geometry_preprocessor import (
    GeometryPreprocessor,
    PrecomputedGeometry,
)
from tests.fixtures.synthetic_diffractor import (
    create_diffractor_dataset,
    create_expected_migration_result,
)


class TestInterpolateTraces:
    """Test Task 2.1: Trace interpolation."""

    def test_integer_sample_exact(self):
        """Interpolation at integer sample should return exact value."""
        # Create simple trace: [0, 1, 2, 3, 4]
        traces = torch.arange(5, dtype=torch.float32).unsqueeze(1)  # (5, 1)

        # Traveltime at exactly sample 2 (t=4ms with dt=2ms)
        traveltimes = torch.tensor([[4.0]])  # (1, 1) - 4ms

        amplitudes = interpolate_traces(traces, traveltimes, dt_ms=2.0, t_min_ms=0.0)

        assert amplitudes.shape == (1, 1)
        assert abs(amplitudes[0, 0].item() - 2.0) < 1e-6

    def test_half_sample_average(self):
        """Interpolation at half sample should return average of neighbors."""
        traces = torch.tensor([[0.0], [2.0], [4.0], [6.0], [8.0]])  # (5, 1)

        # Traveltime at sample 1.5 (t=3ms with dt=2ms)
        traveltimes = torch.tensor([[3.0]])  # (1, 1) - 3ms = sample 1.5

        amplitudes = interpolate_traces(traces, traveltimes, dt_ms=2.0, t_min_ms=0.0)

        # Should be average of samples 1 and 2: (2 + 4) / 2 = 3
        assert abs(amplitudes[0, 0].item() - 3.0) < 1e-6

    def test_out_of_range_clamps(self):
        """Out-of-range indices should clamp to edge values."""
        # Two traces
        traces = torch.tensor([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0]
        ])  # (5, 2)

        # Traveltimes beyond range for each trace
        traveltimes = torch.tensor([
            [-10.0],   # Trace 0: before start
            [100.0]    # Trace 1: after end
        ])  # (2, 1)

        amplitudes = interpolate_traces(
            traces,
            traveltimes,
            dt_ms=2.0,
            t_min_ms=0.0
        )

        # Should clamp: negative -> sample 0, large -> sample 4
        assert abs(amplitudes[0, 0].item() - 1.0) < 1e-3  # First trace, clamped low
        assert abs(amplitudes[1, 0].item() - 5.0) < 1e-3  # Second trace, clamped high

    def test_multiple_traces_and_depths(self):
        """Test interpolation with multiple traces and depths."""
        n_samples = 100
        n_traces = 10
        n_depths = 5

        # Create traces with known pattern
        traces = torch.arange(n_samples, dtype=torch.float32).unsqueeze(1).expand(-1, n_traces)

        # Traveltimes at samples [10, 20, 30, 40, 50] for each trace
        traveltimes = torch.tensor([20.0, 40.0, 60.0, 80.0, 100.0]).unsqueeze(0).expand(n_traces, -1)

        amplitudes = interpolate_traces(traces, traveltimes, dt_ms=2.0, t_min_ms=0.0)

        assert amplitudes.shape == (n_traces, n_depths)

        # Check values at integer samples
        assert abs(amplitudes[0, 0].item() - 10.0) < 1e-6  # t=20ms / 2ms = sample 10
        assert abs(amplitudes[0, 1].item() - 20.0) < 1e-6  # t=40ms / 2ms = sample 20


class TestScatterAddMigration:
    """Test Task 2.2: Scatter-add kernel."""

    def test_single_trace_single_location(self):
        """Single trace should add to correct output location."""
        n_traces = 1
        n_depths = 3
        n_il = 5
        n_xl = 5

        amplitudes = torch.ones(n_traces, n_depths)
        weights = torch.ones(n_traces, n_depths)
        aperture_mask = torch.ones(n_traces, n_depths, dtype=torch.bool)
        output_il = torch.tensor([2])  # Put at il=2
        output_xl = torch.tensor([3])  # Put at xl=3
        valid_mask = torch.tensor([True])

        output_image = torch.zeros(n_depths, n_il, n_xl)
        output_fold = torch.zeros(n_depths, n_il, n_xl)

        scatter_add_migration(
            amplitudes, weights, aperture_mask,
            output_il, output_xl, valid_mask,
            output_image, output_fold
        )

        # Check that only (2, 3) has values
        assert output_image[:, 2, 3].sum() == n_depths  # All depths = 1
        assert output_fold[:, 2, 3].sum() == n_depths   # Fold = 1 for each depth

        # Other locations should be zero
        output_image[:, 2, 3] = 0
        assert output_image.sum() == 0

    def test_multiple_traces_same_location(self):
        """Multiple traces at same location should accumulate."""
        n_traces = 3
        n_depths = 2
        n_il = 5
        n_xl = 5

        amplitudes = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        weights = torch.ones(n_traces, n_depths)
        aperture_mask = torch.ones(n_traces, n_depths, dtype=torch.bool)
        output_il = torch.tensor([2, 2, 2])  # All at same location
        output_xl = torch.tensor([2, 2, 2])
        valid_mask = torch.ones(n_traces, dtype=torch.bool)

        output_image = torch.zeros(n_depths, n_il, n_xl)
        output_fold = torch.zeros(n_depths, n_il, n_xl)

        scatter_add_migration(
            amplitudes, weights, aperture_mask,
            output_il, output_xl, valid_mask,
            output_image, output_fold
        )

        # Should sum: depth 0 = 1+3+5=9, depth 1 = 2+4+6=12
        assert abs(output_image[0, 2, 2].item() - 9.0) < 1e-6
        assert abs(output_image[1, 2, 2].item() - 12.0) < 1e-6
        assert output_fold[0, 2, 2].item() == 3  # 3 traces contributed

    def test_masked_traces_excluded(self):
        """Traces marked invalid should not contribute."""
        n_traces = 3
        n_depths = 1
        n_il = 5
        n_xl = 5

        amplitudes = torch.ones(n_traces, n_depths)
        weights = torch.ones(n_traces, n_depths)
        aperture_mask = torch.ones(n_traces, n_depths, dtype=torch.bool)
        output_il = torch.tensor([1, 2, 3])
        output_xl = torch.tensor([1, 2, 3])
        valid_mask = torch.tensor([True, False, True])  # Middle trace invalid

        output_image = torch.zeros(n_depths, n_il, n_xl)
        output_fold = torch.zeros(n_depths, n_il, n_xl)

        scatter_add_migration(
            amplitudes, weights, aperture_mask,
            output_il, output_xl, valid_mask,
            output_image, output_fold
        )

        # Only traces 0 and 2 should contribute
        assert output_image[0, 1, 1].item() == 1.0
        assert output_image[0, 2, 2].item() == 0.0  # Invalid trace
        assert output_image[0, 3, 3].item() == 1.0

    def test_aperture_mask_applied(self):
        """Aperture mask should zero contributions."""
        n_traces = 1
        n_depths = 3
        n_il = 5
        n_xl = 5

        amplitudes = torch.ones(n_traces, n_depths)
        weights = torch.ones(n_traces, n_depths)
        aperture_mask = torch.tensor([[True, False, True]])  # Middle depth masked
        output_il = torch.tensor([2])
        output_xl = torch.tensor([2])
        valid_mask = torch.tensor([True])

        output_image = torch.zeros(n_depths, n_il, n_xl)
        output_fold = torch.zeros(n_depths, n_il, n_xl)

        scatter_add_migration(
            amplitudes, weights, aperture_mask,
            output_il, output_xl, valid_mask,
            output_image, output_fold
        )

        assert output_image[0, 2, 2].item() == 1.0  # depth 0 - included
        assert output_image[1, 2, 2].item() == 0.0  # depth 1 - masked
        assert output_image[2, 2, 2].item() == 1.0  # depth 2 - included


class TestKirchhoffKernel:
    """Test Task 2.3: Full migration kernel."""

    def test_migrate_returns_correct_shapes(self):
        """Verify output shapes are correct."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()
        kernel = KirchhoffKernel(device=torch.device('cpu'))

        # Precompute geometry
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

        # Convert traces to tensor
        traces = torch.from_numpy(dataset.traces)

        # Migrate
        image, fold = kernel.migrate(
            traces, geometry,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
        )

        # Check shapes
        assert image.shape == (dataset.n_samples, dataset.n_il, dataset.n_xl)
        assert fold.shape == (dataset.n_samples, dataset.n_il, dataset.n_xl)

    def test_migrate_diffractor_focuses(self):
        """Verify diffractor focuses at expected location."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()
        kernel = KirchhoffKernel(device=torch.device('cpu'))

        # Precompute geometry
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

        # Convert traces to tensor
        traces = torch.from_numpy(dataset.traces)

        # Migrate
        image, fold = kernel.migrate(
            traces, geometry,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
        )

        # Normalize
        image_norm = normalize_by_fold(image, fold, min_fold=1)

        # Find peak location
        image_np = image_norm.numpy()
        peak_idx = np.unravel_index(np.argmax(np.abs(image_np)), image_np.shape)
        peak_depth, peak_il, peak_xl = peak_idx

        # Expected location
        expected_sample = int(dataset.expected_diffractor_time_ms / dataset.dt_ms)
        expected_il = dataset.expected_il
        expected_xl = dataset.expected_xl

        # Peak should be near expected location (within a few samples/grid cells)
        assert abs(peak_depth - expected_sample) <= 3, f"Depth mismatch: {peak_depth} vs {expected_sample}"
        assert abs(peak_il - expected_il) <= 2, f"IL mismatch: {peak_il} vs {expected_il}"
        assert abs(peak_xl - expected_xl) <= 2, f"XL mismatch: {peak_xl} vs {expected_xl}"

    def test_migrate_batched_matches_full(self):
        """Batched migration should produce same result as full."""
        dataset = create_diffractor_dataset()
        preprocessor = GeometryPreprocessor()
        kernel = KirchhoffKernel(device=torch.device('cpu'))

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

        traces = torch.from_numpy(dataset.traces)

        # Full migration
        image1, fold1 = kernel.migrate(
            traces, geometry,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
        )

        # Batched migration
        image2, fold2 = kernel.migrate_batched(
            traces, geometry,
            dt_ms=dataset.dt_ms,
            t_min_ms=dataset.t_min_ms,
            n_il=dataset.n_il,
            n_xl=dataset.n_xl,
            depth_batch_size=100,
        )

        # Should match
        assert torch.allclose(image1, image2, atol=1e-5)
        assert torch.allclose(fold1, fold2, atol=1e-5)


class TestNormalizeByFold:
    """Test fold normalization."""

    def test_normalizes_correctly(self):
        """Test basic normalization."""
        image = torch.tensor([[[10.0, 20.0], [30.0, 0.0]]])
        fold = torch.tensor([[[2.0, 4.0], [3.0, 0.0]]])

        normalized = normalize_by_fold(image, fold, min_fold=1)

        assert abs(normalized[0, 0, 0].item() - 5.0) < 1e-6   # 10/2
        assert abs(normalized[0, 0, 1].item() - 5.0) < 1e-6   # 20/4
        assert abs(normalized[0, 1, 0].item() - 10.0) < 1e-6  # 30/3
        assert normalized[0, 1, 1].item() == 0.0              # 0 fold stays 0

    def test_min_fold_threshold(self):
        """Test minimum fold threshold."""
        image = torch.tensor([[[10.0, 20.0]]])
        fold = torch.tensor([[[1.0, 0.5]]])  # Second has fold < 1

        normalized = normalize_by_fold(image, fold, min_fold=1)

        assert abs(normalized[0, 0, 0].item() - 10.0) < 1e-6  # Normalized
        assert normalized[0, 0, 1].item() == 20.0             # Not normalized (fold < 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
