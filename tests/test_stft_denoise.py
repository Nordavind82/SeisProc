"""
Tests for STFT-Denoise processor.
"""
import pytest
import numpy as np
from models.seismic_data import SeismicData
from processors.stft_denoise import STFTDenoise


class TestSTFTDenoiseValidation:
    """Test parameter validation."""

    def test_valid_parameters(self):
        """Test processor accepts valid parameters."""
        processor = STFTDenoise(
            aperture=5,
            fmin=10.0,
            fmax=80.0,
            nperseg=32,
            threshold_k=2.5,
            threshold_mode='adaptive'
        )
        assert processor.aperture == 5
        assert processor.fmin == 10.0
        assert processor.fmax == 80.0
        assert processor.nperseg == 32
        assert processor.threshold_k == 2.5
        assert processor.threshold_mode == 'adaptive'

    def test_invalid_aperture_too_small(self):
        """Test rejection of aperture < 3."""
        with pytest.raises(ValueError, match="Aperture must be at least 3"):
            STFTDenoise(aperture=1)

    def test_invalid_aperture_even(self):
        """Test rejection of even aperture."""
        with pytest.raises(ValueError, match="Aperture must be odd"):
            STFTDenoise(aperture=4)

    def test_invalid_frequency_range(self):
        """Test rejection of fmin >= fmax."""
        with pytest.raises(ValueError, match="fmin must be less than fmax"):
            STFTDenoise(fmin=100.0, fmax=50.0)

    def test_invalid_nperseg(self):
        """Test rejection of nperseg < 8."""
        with pytest.raises(ValueError, match="nperseg must be at least 8"):
            STFTDenoise(nperseg=4)

    def test_invalid_threshold_k(self):
        """Test rejection of threshold_k <= 0."""
        with pytest.raises(ValueError, match="threshold_k must be positive"):
            STFTDenoise(threshold_k=0)

    def test_invalid_threshold_mode(self):
        """Test rejection of invalid threshold_mode."""
        with pytest.raises(ValueError, match="threshold_mode must be"):
            STFTDenoise(threshold_mode='invalid')


class TestSTFTDenoiseProcessing:
    """Test STFT denoising processing."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic seismic data with noise."""
        np.random.seed(42)
        n_samples = 500
        n_traces = 20
        sample_rate = 1000.0  # 1 kHz

        # Create clean signal: multiple sinusoids
        t = np.linspace(0, n_samples / sample_rate, n_samples)
        clean_signal = (
            np.sin(2 * np.pi * 30 * t) +
            0.5 * np.sin(2 * np.pi * 60 * t)
        )

        # Create coherent signal across traces
        traces = np.tile(clean_signal.reshape(-1, 1), (1, n_traces))

        # Add random noise
        noise = np.random.randn(n_samples, n_traces) * 0.3
        noisy_traces = traces + noise

        return SeismicData(
            traces=noisy_traces.astype(np.float32),
            sample_rate=sample_rate,
            metadata={'test': True}
        )

    def test_stft_denoise_runs(self, synthetic_data):
        """Test that STFT denoise runs without error."""
        processor = STFTDenoise(
            aperture=5,
            fmin=5.0,
            fmax=100.0,
            threshold_k=3.0
        )
        result = processor.process(synthetic_data)

        assert result is not None
        assert result.traces.shape == synthetic_data.traces.shape

    def test_output_shape(self, synthetic_data):
        """Test output shape matches input."""
        processor = STFTDenoise(aperture=5)
        result = processor.process(synthetic_data)

        assert result.traces.shape == synthetic_data.traces.shape

    def test_noise_reduction(self, synthetic_data):
        """Test that noise is reduced."""
        processor = STFTDenoise(
            aperture=7,
            fmin=5.0,
            fmax=100.0,
            threshold_k=2.5
        )
        result = processor.process(synthetic_data)

        # Compute variance (noise estimate)
        input_var = np.var(synthetic_data.traces)
        output_var = np.var(result.traces)

        # Output should have reduced variance (denoised)
        assert output_var <= input_var

    def test_threshold_modes(self, synthetic_data):
        """Test all threshold modes work."""
        modes = ['soft', 'hard', 'scaled', 'adaptive']

        for mode in modes:
            processor = STFTDenoise(
                aperture=5,
                threshold_mode=mode
            )
            result = processor.process(synthetic_data)
            assert result is not None
            assert result.traces.shape == synthetic_data.traces.shape

    def test_nperseg_parameter(self, synthetic_data):
        """Test different nperseg values."""
        for nperseg in [16, 32, 64]:
            processor = STFTDenoise(
                aperture=5,
                nperseg=nperseg
            )
            result = processor.process(synthetic_data)
            assert result is not None


class TestSTFTDenoiseEdgeCases:
    """Test edge cases."""

    def test_few_traces(self):
        """Test with fewer traces than aperture."""
        np.random.seed(42)
        n_samples = 200
        n_traces = 3  # Less than default aperture

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(
            traces=traces,
            sample_rate=1000.0,
            metadata={}
        )

        processor = STFTDenoise(aperture=7)  # Larger than n_traces
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_short_trace(self):
        """Test with short traces."""
        np.random.seed(42)
        n_samples = 50  # Short trace
        n_traces = 10

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(
            traces=traces,
            sample_rate=1000.0,
            metadata={}
        )

        processor = STFTDenoise(aperture=5, nperseg=16)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_low_amp_protection(self):
        """Test low amplitude protection feature."""
        np.random.seed(42)
        n_samples = 200
        n_traces = 10

        # Create signal with low amplitude section
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        traces[50:100, :] *= 0.1  # Low amplitude section

        data = SeismicData(
            traces=traces,
            sample_rate=1000.0,
            metadata={}
        )

        processor = STFTDenoise(
            aperture=5,
            low_amp_protection=True,
            low_amp_factor=0.3
        )
        result = processor.process(data)

        assert result is not None


class TestSTFTDenoiseDescription:
    """Test description generation."""

    def test_description(self):
        """Test get_description returns proper string."""
        processor = STFTDenoise(
            aperture=7,
            fmin=10.0,
            fmax=80.0,
            nperseg=64,
            threshold_k=3.0,
            threshold_mode='adaptive'
        )
        desc = processor.get_description()

        assert 'STFT-Denoise' in desc
        assert 'aperture=7' in desc
        assert 'freq=10-80Hz' in desc
        assert 'nperseg=64' in desc
        assert 'k=3.0' in desc
        assert 'mode=adaptive' in desc

    def test_description_with_low_amp_protection(self):
        """Test description includes low_amp_protect when enabled."""
        processor = STFTDenoise(
            aperture=5,
            low_amp_protection=True
        )
        desc = processor.get_description()

        assert 'low_amp_protect' in desc
