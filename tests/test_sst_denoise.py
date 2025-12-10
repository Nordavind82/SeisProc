"""
Tests for Synchrosqueezing Transform (SST) Denoising processor.
"""
import numpy as np
import pytest

from models.seismic_data import SeismicData

# Try to import SSTDenoise
try:
    from processors.sst_denoise import SSTDenoise, SSQUEEZEPY_AVAILABLE
except ImportError:
    SSQUEEZEPY_AVAILABLE = False

# Skip all tests if ssqueezepy not available
pytestmark = pytest.mark.skipif(not SSQUEEZEPY_AVAILABLE, reason="ssqueezepy not installed")


def create_test_data(n_samples=256, n_traces=3, noise_level=0.3, seed=42):
    """Create synthetic data with chirp signal and noise."""
    np.random.seed(seed)
    t = np.linspace(0, 1, n_samples)

    # Create chirp signal (frequency sweep)
    clean = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        # Chirp from 10 Hz to 50 Hz
        phase = 2 * np.pi * (10 * t + 20 * t**2)
        clean[:, i] = np.cos(phase)

    noise = np.random.randn(n_samples, n_traces) * noise_level
    return clean, clean + noise, noise


class TestSSTDenoiseValidation:
    """Tests for parameter validation."""

    def test_valid_parameters(self):
        """Test processor accepts valid parameters."""
        processor = SSTDenoise(
            base_transform='cwt',
            wavelet='morlet',
            nv=32,
            threshold_k=3.0
        )
        assert processor.base_transform == 'cwt'

    def test_invalid_transform(self):
        """Test rejection of invalid transform type."""
        with pytest.raises(ValueError, match="base_transform must be"):
            SSTDenoise(base_transform='invalid')

    def test_invalid_wavelet(self):
        """Test rejection of invalid wavelet."""
        with pytest.raises(ValueError, match="wavelet must be"):
            SSTDenoise(wavelet='invalid_wavelet')

    def test_invalid_nv(self):
        """Test rejection of small nv."""
        with pytest.raises(ValueError, match="nv must be at least"):
            SSTDenoise(nv=2)

    def test_invalid_threshold_k(self):
        """Test rejection of invalid threshold_k."""
        with pytest.raises(ValueError, match="threshold_k must be positive"):
            SSTDenoise(threshold_k=0)


class TestSSTDenoiseProcessing:
    """Tests for SST denoising functionality."""

    def test_cwt_sst_runs(self):
        """Test CWT-based SST processes without error."""
        clean, noisy, _ = create_test_data(n_samples=128, n_traces=2)

        processor = SSTDenoise(
            base_transform='cwt',
            wavelet='morlet',
            nv=16,
            threshold_k=3.0,
            squeezing=True
        )
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape
        assert not np.allclose(result.traces, noisy)

    def test_stft_sst_runs(self):
        """Test STFT-based SST processes without error."""
        clean, noisy, _ = create_test_data(n_samples=128, n_traces=2)

        processor = SSTDenoise(
            base_transform='stft',
            threshold_k=3.0,
            squeezing=True
        )
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_no_squeezing(self):
        """Test processing without synchrosqueezing."""
        clean, noisy, _ = create_test_data(n_samples=128, n_traces=2)

        processor = SSTDenoise(
            base_transform='cwt',
            wavelet='morlet',
            squeezing=False,
            threshold_k=3.0
        )
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_output_shape(self):
        """Test output has same shape as input."""
        clean, noisy, _ = create_test_data(n_samples=128, n_traces=3)

        processor = SSTDenoise(base_transform='cwt', nv=16)
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_soft_vs_hard_threshold(self):
        """Test soft and hard thresholding produce different results."""
        clean, noisy, _ = create_test_data(n_samples=128, n_traces=2)

        processor_soft = SSTDenoise(
            base_transform='cwt',
            nv=16,
            threshold_k=3.0,
            threshold_mode='soft'
        )
        processor_hard = SSTDenoise(
            base_transform='cwt',
            nv=16,
            threshold_k=3.0,
            threshold_mode='hard'
        )

        result_soft = processor_soft.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))
        result_hard = processor_hard.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))

        # Results should be different
        assert not np.allclose(result_soft.traces, result_hard.traces)


class TestSSTDenoiseEdgeCases:
    """Tests for edge cases."""

    def test_single_trace(self):
        """Test with single trace."""
        clean, noisy, _ = create_test_data(n_samples=128, n_traces=1)

        processor = SSTDenoise(base_transform='cwt', nv=16)
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_short_trace(self):
        """Test with short trace."""
        np.random.seed(42)
        data = SeismicData(
            traces=np.random.randn(64, 2),
            sample_rate=2.0
        )

        processor = SSTDenoise(base_transform='cwt', nv=8)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape


class TestSSTDescription:
    """Tests for processor description."""

    def test_description_sst(self):
        """Test SST description with squeezing."""
        processor = SSTDenoise(
            base_transform='cwt',
            wavelet='morlet',
            nv=32,
            threshold_k=3.0,
            squeezing=True
        )
        desc = processor.get_description()

        assert 'SST' in desc
        assert 'morlet' in desc

    def test_description_no_squeezing(self):
        """Test description without squeezing."""
        processor = SSTDenoise(
            base_transform='cwt',
            squeezing=False
        )
        desc = processor.get_description()

        assert 'CWT' in desc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
