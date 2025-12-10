"""
Tests for 3D Spatial Denoising processor.
"""
import pytest
import numpy as np
import pandas as pd
from models.seismic_data import SeismicData
from processors.denoise_3d import (
    Denoise3D,
    compute_3d_mad,
    build_volume_from_headers,
    extract_traces_from_volume,
    get_available_headers
)


@pytest.fixture
def sample_3d_data():
    """Create sample seismic data organized as shot gather with headers."""
    np.random.seed(42)

    # 4 shots, 10 receivers each = 40 traces
    n_shots = 4
    n_receivers = 10
    n_samples = 128
    n_traces = n_shots * n_receivers

    # Create traces with coherent events + noise
    traces = np.zeros((n_samples, n_traces))

    for shot in range(n_shots):
        for rec in range(n_receivers):
            trace_idx = shot * n_receivers + rec

            # Add horizontal reflector (coherent across receivers and shots)
            for t in [30, 60, 90]:
                offset = rec * 2  # Slight moveout
                t_event = t + offset
                if t_event + 8 < n_samples:
                    traces[t_event:t_event+8, trace_idx] = np.sin(
                        np.linspace(0, 2*np.pi, 8)
                    ) * 0.5

    # Add random noise
    noise = np.random.randn(n_samples, n_traces) * 0.15
    traces = traces + noise

    # Create headers DataFrame
    headers = {
        'field_record': np.repeat(np.arange(1, n_shots + 1), n_receivers),
        'trace_number': np.tile(np.arange(1, n_receivers + 1), n_shots),
        'offset': np.tile(np.arange(0, n_receivers * 25, 25), n_shots),
    }
    headers_df = pd.DataFrame(headers)

    return SeismicData(
        traces=traces.astype(np.float32),
        sample_rate=2.0,  # 2ms
        headers=headers_df
    )


@pytest.fixture
def sample_volume():
    """Create sample 3D volume for testing."""
    np.random.seed(42)
    n_samples, n_inlines, n_xlines = 64, 5, 8

    volume = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)

    # Add signal
    for il in range(n_inlines):
        for xl in range(n_xlines):
            # Horizontal reflector
            for t in [20, 40]:
                volume[t:t+5, il, xl] = np.sin(np.linspace(0, np.pi, 5)) * 0.5

    # Add noise
    volume += np.random.randn(n_samples, n_inlines, n_xlines).astype(np.float32) * 0.1

    return volume


class TestCompute3DMAD:
    """Test 3D MAD computation function."""

    def test_compute_3d_mad_shape(self, sample_volume):
        """Test that output shapes match input."""
        median_3d, mad_3d = compute_3d_mad(sample_volume, aperture_inline=3, aperture_xline=3)

        assert median_3d.shape == sample_volume.shape
        assert mad_3d.shape == sample_volume.shape

    def test_compute_3d_mad_non_negative(self, sample_volume):
        """Test that MAD is non-negative."""
        _, mad_3d = compute_3d_mad(sample_volume, aperture_inline=3, aperture_xline=3)

        assert (mad_3d >= 0).all()

    def test_compute_3d_mad_larger_aperture(self, sample_volume):
        """Test with larger aperture."""
        median_3d, mad_3d = compute_3d_mad(sample_volume, aperture_inline=5, aperture_xline=5)

        assert median_3d.shape == sample_volume.shape
        # Larger aperture should give smoother estimates
        assert np.std(mad_3d) < np.std(sample_volume) * 2


class TestBuildVolumeFromHeaders:
    """Test volume building from headers."""

    def test_build_volume_shape(self, sample_3d_data):
        """Test that volume has correct shape."""
        volume, geometry = build_volume_from_headers(
            sample_3d_data.traces,
            sample_3d_data.headers,
            inline_key='field_record',
            xline_key='trace_number'
        )

        assert volume.shape[0] == sample_3d_data.n_samples
        assert geometry['n_inlines'] == 4  # 4 shots
        assert geometry['n_xlines'] == 10  # 10 receivers

    def test_build_volume_coverage(self, sample_3d_data):
        """Test volume coverage is 100% for complete gather."""
        _, geometry = build_volume_from_headers(
            sample_3d_data.traces,
            sample_3d_data.headers,
            inline_key='field_record',
            xline_key='trace_number'
        )

        assert geometry['coverage'] == 100.0

    def test_build_volume_invalid_header(self, sample_3d_data):
        """Test error on invalid header key."""
        with pytest.raises(ValueError, match="not found"):
            build_volume_from_headers(
                sample_3d_data.traces,
                sample_3d_data.headers,
                inline_key='invalid_key',
                xline_key='trace_number'
            )

    def test_roundtrip_conservation(self, sample_3d_data):
        """Test that build → extract preserves data."""
        volume, geometry = build_volume_from_headers(
            sample_3d_data.traces,
            sample_3d_data.headers,
            inline_key='field_record',
            xline_key='trace_number'
        )

        extracted = extract_traces_from_volume(
            volume,
            sample_3d_data.headers,
            geometry
        )

        np.testing.assert_allclose(
            extracted, sample_3d_data.traces,
            rtol=1e-5, atol=1e-7
        )


class TestGetAvailableHeaders:
    """Test header availability detection."""

    def test_get_available_headers(self, sample_3d_data):
        """Test getting available headers."""
        available = get_available_headers(sample_3d_data.headers)

        assert 'field_record' in available
        assert 'trace_number' in available
        assert 'offset' in available

    def test_get_available_headers_empty(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        available = get_available_headers(empty_df)

        assert available == []


class TestDenoise3DValidation:
    """Test Denoise3D parameter validation."""

    def test_default_params(self):
        """Test default parameters."""
        processor = Denoise3D()
        assert processor.inline_key == 'field_record'
        assert processor.xline_key == 'trace_number'
        assert processor.aperture_inline == 3
        assert processor.aperture_xline == 3

    def test_invalid_aperture_inline(self):
        """Test invalid inline aperture."""
        with pytest.raises(ValueError, match="odd"):
            Denoise3D(aperture_inline=4)

    def test_invalid_aperture_xline(self):
        """Test invalid crossline aperture."""
        with pytest.raises(ValueError, match="odd"):
            Denoise3D(aperture_xline=6)

    def test_invalid_threshold_mode(self):
        """Test invalid threshold mode."""
        with pytest.raises(ValueError, match="threshold_mode"):
            Denoise3D(threshold_mode='invalid')

    def test_valid_parameters(self):
        """Test valid custom parameters."""
        processor = Denoise3D(
            inline_key='CDP',
            xline_key='Channel',
            aperture_inline=5,
            aperture_xline=7,
            wavelet='sym4',
            threshold_k=2.5,
            threshold_mode='hard'
        )

        assert processor.inline_key == 'CDP'
        assert processor.aperture_inline == 5
        assert processor.aperture_xline == 7
        assert processor.wavelet == 'sym4'
        assert processor.threshold_k == 2.5
        assert processor.threshold_mode == 'hard'


class TestDenoise3DProcessing:
    """Test Denoise3D processing."""

    def test_process_basic(self, sample_3d_data):
        """Test basic processing."""
        processor = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number',
            aperture_inline=3,
            aperture_xline=3,
            threshold_k=3.0
        )

        result = processor.process(sample_3d_data)

        assert result.traces.shape == sample_3d_data.traces.shape
        assert result.sample_rate == sample_3d_data.sample_rate
        assert result.headers is not None

    def test_process_reduces_noise(self, sample_3d_data):
        """Test that processing reduces noise."""
        processor = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number',
            aperture_inline=3,
            aperture_xline=3,
            threshold_k=2.0  # Aggressive threshold
        )

        result = processor.process(sample_3d_data)

        # Noise RMS should decrease
        input_rms = np.std(sample_3d_data.traces)
        output_rms = np.std(result.traces)

        # Allow some tolerance - denoising should reduce overall variability
        assert output_rms <= input_rms

    def test_process_preserves_headers(self, sample_3d_data):
        """Test that headers are preserved."""
        processor = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number'
        )

        result = processor.process(sample_3d_data)

        # Headers should be preserved
        pd.testing.assert_frame_equal(
            pd.DataFrame(result.headers),
            pd.DataFrame(sample_3d_data.headers)
        )

    def test_process_no_headers_error(self):
        """Test error when headers are missing."""
        data = SeismicData(
            traces=np.random.randn(100, 20).astype(np.float32),
            sample_rate=2.0,
            headers=None
        )

        processor = Denoise3D()

        with pytest.raises(ValueError, match="headers"):
            processor.process(data)

    def test_process_soft_vs_hard(self, sample_3d_data):
        """Test soft vs hard thresholding produces different results."""
        soft_proc = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number',
            threshold_mode='soft',
            threshold_k=3.0
        )
        hard_proc = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number',
            threshold_mode='hard',
            threshold_k=3.0
        )

        soft_result = soft_proc.process(sample_3d_data)
        hard_result = hard_proc.process(sample_3d_data)

        # Results should differ
        assert not np.allclose(soft_result.traces, hard_result.traces, rtol=0.01)


class TestDenoise3DDescription:
    """Test description generation."""

    def test_description_content(self):
        """Test description contains key info."""
        processor = Denoise3D(
            inline_key='CDP',
            xline_key='Channel',
            aperture_inline=5,
            aperture_xline=7,
            wavelet='db6',
            threshold_k=2.5,
            threshold_mode='hard'
        )

        desc = processor.get_description()

        assert '3D-Denoise' in desc
        assert 'CDP' in desc
        assert 'Channel' in desc
        assert '5×7' in desc
        assert 'db6' in desc
        assert '2.5' in desc
        assert 'hard' in desc


class TestDenoise3DEdgeCases:
    """Test edge cases."""

    def test_small_volume(self):
        """Test with small volume (aperture larger than dimensions)."""
        np.random.seed(42)

        # 2 shots, 3 receivers = 6 traces
        n_traces = 6
        n_samples = 64

        traces = np.random.randn(n_samples, n_traces).astype(np.float32) * 0.2
        headers = pd.DataFrame({
            'field_record': [1, 1, 1, 2, 2, 2],
            'trace_number': [1, 2, 3, 1, 2, 3]
        })

        data = SeismicData(traces=traces, sample_rate=2.0, headers=headers)

        processor = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number',
            aperture_inline=5,  # Larger than n_inlines=2
            aperture_xline=5    # Larger than n_xlines=3
        )

        # Should clamp aperture and succeed
        result = processor.process(data)

        assert result.traces.shape == traces.shape

    def test_sparse_volume(self):
        """Test with sparse volume (not all positions filled)."""
        np.random.seed(42)

        # Only some positions filled
        n_samples = 64
        n_traces = 5

        traces = np.random.randn(n_samples, n_traces).astype(np.float32) * 0.2
        headers = pd.DataFrame({
            'field_record': [1, 1, 3, 3, 3],  # Gaps in shot 2
            'trace_number': [1, 3, 1, 2, 3]    # Gaps in receivers
        })

        data = SeismicData(traces=traces, sample_rate=2.0, headers=headers)

        processor = Denoise3D(
            inline_key='field_record',
            xline_key='trace_number',
            aperture_inline=3,
            aperture_xline=3
        )

        result = processor.process(data)

        # Should handle sparse data
        assert result.traces.shape == traces.shape
        # Non-zero output
        assert np.std(result.traces) > 0
