"""
Tests for MigrationEngine.

Tests Tasks 3.1-3.2 of the PSTM redesign plan:
- 3.1: MigrationEngine orchestrator
- 3.2: Trace batching for large bins
"""

import pytest
import numpy as np
import torch
import time

from processors.migration.migration_engine import MigrationEngine
from tests.fixtures.synthetic_diffractor import (
    create_diffractor_dataset,
    create_expected_migration_result,
)


class TestMigrationEngine:
    """Test Task 3.1: MigrationEngine orchestrator."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = MigrationEngine(device=torch.device('cpu'))
        assert engine.device == torch.device('cpu')
        assert engine.preprocessor is not None
        assert engine.kernel is not None

    def test_migrate_bin_returns_correct_shapes(self):
        """Verify output shapes are correct."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine(device=torch.device('cpu'))

        image, fold = engine.migrate_bin(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
        )

        expected_shape = (dataset.n_samples, dataset.n_il, dataset.n_xl)
        assert image.shape == expected_shape
        assert fold.shape == expected_shape

    def test_migrate_diffractor_focuses_correctly(self):
        """Verify diffractor focuses at expected location."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine(device=torch.device('cpu'))

        image, fold = engine.migrate_bin(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
            normalize=True,
        )

        # Find peak location
        peak_idx = np.unravel_index(np.argmax(np.abs(image)), image.shape)
        peak_sample, peak_il, peak_xl = peak_idx

        # Expected location
        expected_sample = int(dataset.expected_diffractor_time_ms / dataset.dt_ms)
        expected_il = dataset.expected_il
        expected_xl = dataset.expected_xl

        # Peak should be near expected location
        assert abs(peak_sample - expected_sample) <= 3, \
            f"Sample mismatch: {peak_sample} vs {expected_sample}"
        assert abs(peak_il - expected_il) <= 2, \
            f"IL mismatch: {peak_il} vs {expected_il}"
        assert abs(peak_xl - expected_xl) <= 2, \
            f"XL mismatch: {peak_xl} vs {expected_xl}"

    def test_migrate_produces_nonzero_fold(self):
        """Verify fold is non-zero in migrated region."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine(device=torch.device('cpu'))

        image, fold = engine.migrate_bin(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            normalize=False,  # Don't normalize to see raw fold
        )

        # Should have significant fold coverage
        nonzero_fold = (fold > 0).sum()
        total_points = fold.size
        coverage = nonzero_fold / total_points

        # With 10,000 traces covering 100x100 grid, should have good coverage
        assert coverage > 0.5, f"Low fold coverage: {coverage*100:.1f}%"

    def test_progress_callback_called(self):
        """Verify progress callback is called during migration."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine(device=torch.device('cpu'))

        progress_calls = []

        def callback(fraction, message):
            progress_calls.append((fraction, message))

        engine.migrate_bin(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            progress_callback=callback,
        )

        # Should have multiple progress calls
        assert len(progress_calls) >= 4  # At least: start, preprocessing, kernel, complete

        # Final call should be 1.0
        assert progress_calls[-1][0] == 1.0


class TestMigrationEngineBatched:
    """Test Task 3.2: Trace batching."""

    def test_batched_matches_full(self):
        """Batched migration should produce same result as full."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine(device=torch.device('cpu'))

        # Full migration
        image1, fold1 = engine.migrate_bin(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            normalize=True,
        )

        # Batched migration with small batch size
        image2, fold2 = engine.migrate_bin_batched(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            batch_size=2000,  # Force multiple batches
            normalize=True,
        )

        # Results should be close (not exact due to floating point)
        # Check peak location matches
        peak1 = np.unravel_index(np.argmax(np.abs(image1)), image1.shape)
        peak2 = np.unravel_index(np.argmax(np.abs(image2)), image2.shape)

        assert peak1 == peak2, f"Peak locations differ: {peak1} vs {peak2}"

        # Check correlation is high
        corr = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]
        assert corr > 0.99, f"Low correlation between full and batched: {corr}"

    def test_batched_with_very_small_batch(self):
        """Test batching with very small batch size."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine(device=torch.device('cpu'))

        # Use tiny batch size to test edge cases
        image, fold = engine.migrate_bin_batched(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
            batch_size=500,  # Many batches
        )

        # Diffractor should still focus
        peak_idx = np.unravel_index(np.argmax(np.abs(image)), image.shape)
        peak_sample, peak_il, peak_xl = peak_idx

        expected_sample = int(dataset.expected_diffractor_time_ms / dataset.dt_ms)
        assert abs(peak_sample - expected_sample) <= 3


class TestMigrationEngineBenchmark:
    """Test benchmark functionality."""

    def test_benchmark_runs(self):
        """Test benchmark completes and returns results."""
        engine = MigrationEngine(device=torch.device('cpu'))

        # Run with small parameters for quick test
        results = engine.benchmark(
            n_traces=1000,
            n_samples=500,
            n_il=50,
            n_xl=50,
        )

        assert 'total_time_s' in results
        assert 'traces_per_second' in results
        assert results['n_traces'] == 1000
        assert results['traces_per_second'] > 0


class TestMigrationEnginePerformance:
    """Performance-focused tests."""

    @pytest.mark.slow
    def test_performance_10k_traces(self):
        """Test migration performance with 10K traces."""
        dataset = create_diffractor_dataset()
        engine = MigrationEngine()  # Use best available device

        t0 = time.time()
        image, fold = engine.migrate_bin(
            traces=dataset.traces,
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
            velocity_mps=dataset.velocity,
        )
        elapsed = time.time() - t0

        traces_per_sec = dataset.n_traces / elapsed

        print(f"\nPerformance: {traces_per_sec:.0f} traces/s "
              f"({elapsed:.2f}s for {dataset.n_traces} traces)")

        # Should process at least 1000 traces/s on CPU
        # GPU should be much faster
        assert traces_per_sec > 500, f"Too slow: {traces_per_sec:.0f} traces/s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
