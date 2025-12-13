"""
Unit tests for Coordinate-Based Volume Builder

Tests the three multi-fold reconstruction strategies:
1. noise_subtract - fastest, for design stage
2. residual_preserve - medium, preserves per-trace differences
3. multi_pass - most accurate, individual trace filtering
"""
import numpy as np
import pandas as pd
import pytest
from typing import Tuple

from utils.coordinate_volume_builder import (
    CoordinateVolumeBuilder,
    BinningConfig,
    BinningGeometry,
    ReconstructionMethod,
    RepresentativeMethod,
    estimate_grid_from_coordinates
)
from models.seismic_volume import SeismicVolume


def create_synthetic_data(
    n_traces: int = 100,
    n_samples: int = 500,
    grid_size: float = 25.0,
    n_bins_x: int = 5,
    n_bins_y: int = 5,
    max_fold: int = 3,
    add_coherent_noise: bool = True,
    add_random_noise: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create synthetic traces with known geometry and multi-fold bins.

    Returns:
        traces: (n_samples, n_traces)
        headers_df: DataFrame with CDP_X, CDP_Y
    """
    np.random.seed(42)

    # Generate coordinates - uniform random within grid extent
    extent_x = n_bins_x * grid_size
    extent_y = n_bins_y * grid_size

    coords_x = np.random.uniform(0, extent_x * 0.9, n_traces)
    coords_y = np.random.uniform(0, extent_y * 0.9, n_traces)

    # Create synthetic traces
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)

    # Add signal: reflection events
    t = np.arange(n_samples) * 0.004  # 4ms sample rate
    for i in range(n_traces):
        # Reflection at ~0.5s
        t0 = 0.5
        wavelet = np.exp(-((t - t0) / 0.02) ** 2) * np.sin(2 * np.pi * 30 * (t - t0))
        traces[:, i] += wavelet

        # Reflection at ~0.8s
        t0 = 0.8
        wavelet = 0.7 * np.exp(-((t - t0) / 0.02) ** 2) * np.sin(2 * np.pi * 25 * (t - t0))
        traces[:, i] += wavelet

    # Add coherent noise (ground roll) - same for traces at same location
    if add_coherent_noise:
        for i in range(n_traces):
            x, y = coords_x[i], coords_y[i]
            # Ground roll depends on location
            offset = np.sqrt(x ** 2 + y ** 2)
            v_groundroll = 300.0
            t_arrival = offset / v_groundroll

            # Low frequency ground roll
            for f in range(5, 20, 5):
                if t_arrival < t.max():
                    idx = int(t_arrival / 0.004)
                    if idx < n_samples - 50:
                        roll = 0.3 * np.exp(-((t - t_arrival) / 0.1) ** 2) * np.sin(2 * np.pi * f * (t - t_arrival))
                        traces[:, i] += roll

    # Add random noise (different per trace)
    if add_random_noise:
        traces += np.random.randn(n_samples, n_traces).astype(np.float32) * 0.05

    headers_df = pd.DataFrame({
        'CDP_X': coords_x,
        'CDP_Y': coords_y,
        'trace_idx': np.arange(n_traces)
    })

    return traces, headers_df


class TestBinningConfig:
    """Test BinningConfig validation."""

    def test_valid_config(self):
        config = BinningConfig(bin_size_x=25.0, bin_size_y=25.0)
        assert config.bin_size_x == 25.0
        assert config.bin_size_y == 25.0

    def test_invalid_bin_size(self):
        with pytest.raises(ValueError):
            BinningConfig(bin_size_x=0)

        with pytest.raises(ValueError):
            BinningConfig(bin_size_y=-10)

    def test_string_to_enum_conversion(self):
        config = BinningConfig(
            representative_method='median',
            reconstruction_method='multi_pass'
        )
        assert config.representative_method == RepresentativeMethod.MEDIAN
        assert config.reconstruction_method == ReconstructionMethod.MULTI_PASS


class TestBinningGeometry:
    """Test BinningGeometry coordinate transforms."""

    def test_world_to_bin_no_rotation(self):
        geom = BinningGeometry(
            origin_x=0, origin_y=0,
            bin_size_x=25, bin_size_y=25,
            n_bins_x=10, n_bins_y=10,
            rotation_deg=0,
            n_samples=500, dt=0.004
        )

        # Point at (12.5, 12.5) should be in bin (0, 0)
        ix, iy, dist = geom.world_to_bin(12.5, 12.5)
        assert ix == 0
        assert iy == 0
        assert dist < 0.1  # Very close to center

        # Point at (30, 40) should be in bin (1, 1)
        ix, iy, dist = geom.world_to_bin(30, 40)
        assert ix == 1
        assert iy == 1

    def test_world_to_bin_with_rotation(self):
        geom = BinningGeometry(
            origin_x=0, origin_y=0,
            bin_size_x=25, bin_size_y=25,
            n_bins_x=10, n_bins_y=10,
            rotation_deg=45,
            n_samples=500, dt=0.004
        )

        # With 45 degree rotation, point (35.35, 0) ≈ (25, -25) in rotated frame
        ix, iy, dist = geom.world_to_bin(35.35, 0)
        # After rotation: rx = 35.35 * cos(45) = 25, ry = 35.35 * sin(45) = 25
        assert ix == 1  # rx/25 = 1
        assert iy == 1  # ry/25 = 1

    def test_bin_to_world_roundtrip(self):
        geom = BinningGeometry(
            origin_x=100, origin_y=200,
            bin_size_x=25, bin_size_y=25,
            n_bins_x=10, n_bins_y=10,
            rotation_deg=30,
            n_samples=500, dt=0.004
        )

        # Roundtrip: bin center -> world -> bin should return same bin
        for ix in range(5):
            for iy in range(5):
                x, y = geom.bin_to_world(ix, iy)
                ix2, iy2, _ = geom.world_to_bin(x, y)
                assert ix == ix2
                assert iy == iy2


class TestCoordinateVolumeBuilder:
    """Test CoordinateVolumeBuilder core functionality."""

    def test_build_volume_basic(self):
        """Test basic volume building."""
        traces, headers_df = create_synthetic_data(
            n_traces=50, n_samples=200, max_fold=2
        )

        config = BinningConfig(
            bin_size_x=25.0,
            bin_size_y=25.0,
            representative_method=RepresentativeMethod.MEDIAN
        )

        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)

        assert volume is not None
        assert volume.data.ndim == 3
        assert volume.dt == 0.004
        assert volume.dx == 25.0
        assert volume.dy == 25.0

        # Check geometry
        geom = builder.get_geometry()
        assert geom is not None
        assert geom.total_traces == 50
        assert geom.max_fold >= 1

    def test_build_preserves_trace_count(self):
        """Test that all traces are accounted for."""
        traces, headers_df = create_synthetic_data(n_traces=75, max_fold=3)

        config = BinningConfig(bin_size_x=25.0, bin_size_y=25.0)
        builder = CoordinateVolumeBuilder(config)
        builder.build(traces, headers_df, sample_rate_ms=4.0)

        geom = builder.get_geometry()
        assert geom.traces_binned == 75

    def test_fold_volume(self):
        """Test fold volume computation."""
        traces, headers_df = create_synthetic_data(n_traces=100, max_fold=4)

        config = BinningConfig(bin_size_x=25.0, bin_size_y=25.0)
        builder = CoordinateVolumeBuilder(config)
        builder.build(traces, headers_df, sample_rate_ms=4.0)

        fold_vol = builder.get_fold_volume()
        assert fold_vol is not None
        assert fold_vol.sum() == 100  # Total fold = total traces


class TestReconstructionMethods:
    """Test the three reconstruction methods."""

    @pytest.fixture
    def setup_data(self):
        """Create test data and simple mock filter."""
        traces, headers_df = create_synthetic_data(
            n_traces=60, n_samples=200, max_fold=3,
            add_coherent_noise=True, add_random_noise=False
        )
        return traces, headers_df

    def _simple_filter(self, volume: SeismicVolume) -> SeismicVolume:
        """Simple lowpass-like filter for testing."""
        # Just reduce high frequencies (simple smoothing)
        from scipy.ndimage import uniform_filter1d
        filtered_data = uniform_filter1d(volume.data, size=5, axis=0)
        return SeismicVolume(
            data=filtered_data.astype(np.float32),
            dt=volume.dt, dx=volume.dx, dy=volume.dy
        )

    def test_noise_subtract_reconstruction(self, setup_data):
        """Test noise_subtract method preserves trace count."""
        traces, headers_df = setup_data

        config = BinningConfig(
            bin_size_x=25.0,
            bin_size_y=25.0,
            reconstruction_method=ReconstructionMethod.NOISE_SUBTRACT
        )

        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)

        filtered_vol = self._simple_filter(volume)
        filtered_traces = builder.reconstruct_traces(filtered_vol)

        # Same shape as input
        assert filtered_traces.shape == traces.shape

        # Not identical to input (filtering did something)
        assert not np.allclose(filtered_traces, traces)

    def test_residual_preserve_reconstruction(self, setup_data):
        """Test residual_preserve method preserves per-trace differences."""
        traces, headers_df = setup_data

        config = BinningConfig(
            bin_size_x=25.0,
            bin_size_y=25.0,
            reconstruction_method=ReconstructionMethod.RESIDUAL_PRESERVE
        )

        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)

        # Check residuals were computed
        assert builder.residuals is not None
        assert builder.residuals.shape == traces.shape

        filtered_vol = self._simple_filter(volume)
        filtered_traces = builder.reconstruct_traces(filtered_vol)

        assert filtered_traces.shape == traces.shape
        assert not np.allclose(filtered_traces, traces)

    def test_multi_pass_reconstruction(self, setup_data):
        """Test multi_pass method filters each trace individually."""
        traces, headers_df = setup_data

        config = BinningConfig(
            bin_size_x=25.0,
            bin_size_y=25.0,
            reconstruction_method=ReconstructionMethod.MULTI_PASS
        )

        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)

        filtered_vol = self._simple_filter(volume)

        # Multi-pass requires filter function
        filtered_traces = builder.reconstruct_traces(
            filtered_vol,
            filter_func=self._simple_filter
        )

        assert filtered_traces.shape == traces.shape

        # All traces should be filtered (non-zero if input was non-zero)
        input_energy = np.sum(traces ** 2, axis=0)
        output_energy = np.sum(filtered_traces ** 2, axis=0)

        # Traces with input should have output
        has_input = input_energy > 1e-10
        has_output = output_energy > 1e-10
        assert np.all(has_output[has_input])

    def test_multi_pass_requires_filter_func(self, setup_data):
        """Test that multi_pass raises error without filter_func."""
        traces, headers_df = setup_data

        config = BinningConfig(
            bin_size_x=25.0,
            bin_size_y=25.0,
            reconstruction_method=ReconstructionMethod.MULTI_PASS
        )

        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)
        filtered_vol = self._simple_filter(volume)

        with pytest.raises(ValueError, match="filter_func required"):
            builder.reconstruct_traces(filtered_vol)


class TestEstimateGrid:
    """Test grid estimation from coordinates."""

    def test_estimate_grid_basic(self):
        """Test grid estimation returns sensible values."""
        _, headers_df = create_synthetic_data(n_traces=100)

        result = estimate_grid_from_coordinates(headers_df)

        assert 'error' not in result
        assert result['n_traces'] == 100
        assert result['suggested_bin_size'] > 0
        assert result['median_spacing'] > 0

    def test_estimate_grid_missing_columns(self):
        """Test grid estimation with missing columns."""
        df = pd.DataFrame({'other_col': [1, 2, 3]})
        result = estimate_grid_from_coordinates(df)

        assert 'error' in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_trace_per_bin(self):
        """Test with exactly one trace per bin (no multi-fold)."""
        n_traces = 25
        coords_x = []
        coords_y = []

        for ix in range(5):
            for iy in range(5):
                coords_x.append(ix * 25 + 12.5)
                coords_y.append(iy * 25 + 12.5)

        traces = np.random.randn(200, n_traces).astype(np.float32)
        headers_df = pd.DataFrame({'CDP_X': coords_x, 'CDP_Y': coords_y})

        config = BinningConfig(bin_size_x=25.0, bin_size_y=25.0)
        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)

        geom = builder.get_geometry()
        assert geom.max_fold == 1
        assert geom.mean_fold == 1.0

    def test_all_traces_in_one_bin(self):
        """Test with all traces in a single bin."""
        n_traces = 20
        # All traces at approximately the same location
        coords_x = np.random.uniform(10, 15, n_traces)
        coords_y = np.random.uniform(10, 15, n_traces)

        traces = np.random.randn(200, n_traces).astype(np.float32)
        headers_df = pd.DataFrame({'CDP_X': coords_x, 'CDP_Y': coords_y})

        config = BinningConfig(bin_size_x=25.0, bin_size_y=25.0)
        builder = CoordinateVolumeBuilder(config)
        volume = builder.build(traces, headers_df, sample_rate_ms=4.0)

        geom = builder.get_geometry()
        assert geom.bins_populated == 1
        assert geom.max_fold == n_traces

    def test_empty_headers(self):
        """Test with empty headers raises error."""
        traces = np.random.randn(200, 10).astype(np.float32)
        headers_df = pd.DataFrame()

        config = BinningConfig(bin_size_x=25.0, bin_size_y=25.0)
        builder = CoordinateVolumeBuilder(config)

        with pytest.raises(ValueError):
            builder.build(traces, headers_df, sample_rate_ms=4.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
