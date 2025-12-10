"""
Tests for Gabor Transform Denoising processor.
"""
import numpy as np
import pytest
from scipy import signal

from models.seismic_data import SeismicData
from processors.gabor_denoise import GaborDenoise, create_gaussian_window


class TestGaussianWindow:
    """Tests for Gaussian window creation."""

    def test_window_length(self):
        """Test window has correct length."""
        for nperseg in [32, 64, 128, 256]:
            window = create_gaussian_window(nperseg)
            assert len(window) == nperseg

    def test_window_symmetry(self):
        """Test window is symmetric."""
        window = create_gaussian_window(64)
        assert np.allclose(window, window[::-1])

    def test_window_peak_at_center(self):
        """Test window peaks at center."""
        window = create_gaussian_window(64)
        center = len(window) // 2
        assert window[center] == window.max()

    def test_custom_sigma(self):
        """Test custom sigma affects window width."""
        window_narrow = create_gaussian_window(64, sigma=5)
        window_wide = create_gaussian_window(64, sigma=20)
        # Wider sigma means more energy in tails
        assert window_wide[0] > window_narrow[0]

    def test_default_sigma(self):
        """Test default sigma is nperseg/6."""
        nperseg = 60
        window_default = create_gaussian_window(nperseg)
        window_explicit = create_gaussian_window(nperseg, sigma=nperseg/6)
        assert np.allclose(window_default, window_explicit)


class TestGaborDenoiseValidation:
    """Tests for parameter validation."""

    def test_valid_parameters(self):
        """Test processor accepts valid parameters."""
        processor = GaborDenoise(
            aperture=7,
            fmin=5.0,
            fmax=100.0,
            threshold_k=3.0,
            threshold_mode='soft',
            window_size=64,
            overlap_percent=75.0
        )
        assert processor.aperture == 7

    def test_invalid_aperture_even(self):
        """Test rejection of even aperture."""
        with pytest.raises(ValueError, match="Aperture must be odd"):
            GaborDenoise(aperture=6)

    def test_invalid_aperture_small(self):
        """Test rejection of small aperture."""
        with pytest.raises(ValueError, match="Aperture must be at least 3"):
            GaborDenoise(aperture=1)

    def test_invalid_frequency_range(self):
        """Test rejection of invalid frequency range."""
        with pytest.raises(ValueError, match="fmin must be less than fmax"):
            GaborDenoise(fmin=100.0, fmax=50.0)

    def test_invalid_negative_fmin(self):
        """Test rejection of negative fmin."""
        with pytest.raises(ValueError, match="fmin must be non-negative"):
            GaborDenoise(fmin=-10.0)

    def test_invalid_threshold_k(self):
        """Test rejection of non-positive threshold_k."""
        with pytest.raises(ValueError, match="threshold_k must be positive"):
            GaborDenoise(threshold_k=0)

    def test_invalid_threshold_mode(self):
        """Test rejection of invalid threshold mode."""
        with pytest.raises(ValueError, match="threshold_mode must be"):
            GaborDenoise(threshold_mode='invalid')

    def test_invalid_window_size(self):
        """Test rejection of small window size."""
        with pytest.raises(ValueError, match="window_size must be at least 8"):
            GaborDenoise(window_size=4)

    def test_invalid_overlap(self):
        """Test rejection of invalid overlap percentage."""
        with pytest.raises(ValueError, match="overlap_percent must be between"):
            GaborDenoise(overlap_percent=100.0)


class TestGaborDenoiseProcessing:
    """Tests for denoising functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample seismic data with noise."""
        np.random.seed(42)
        n_samples = 500
        n_traces = 20
        sample_rate = 2.0  # 2ms

        # Create clean signal (sweep)
        t = np.arange(n_samples) * sample_rate / 1000
        clean = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            freq = 20 + i * 2  # 20-58 Hz
            clean[:, i] = np.sin(2 * np.pi * freq * t)

        # Add Gaussian noise
        noise = np.random.randn(n_samples, n_traces) * 0.5
        noisy = clean + noise

        return SeismicData(
            traces=noisy,
            sample_rate=sample_rate,
            metadata={'clean': clean, 'noise': noise}
        )

    def test_output_shape(self, sample_data):
        """Test output has same shape as input."""
        processor = GaborDenoise(aperture=5, window_size=32)
        result = processor.process(sample_data)
        assert result.traces.shape == sample_data.traces.shape

    def test_output_sample_rate(self, sample_data):
        """Test sample rate is preserved."""
        processor = GaborDenoise(aperture=5, window_size=32)
        result = processor.process(sample_data)
        assert result.sample_rate == sample_data.sample_rate

    def test_noise_reduction(self, sample_data):
        """Test that noise is actually reduced."""
        processor = GaborDenoise(
            aperture=7,
            threshold_k=2.5,
            window_size=64,
            overlap_percent=75.0
        )
        result = processor.process(sample_data)

        clean = sample_data.metadata['clean']

        # Calculate error (closer to clean = better)
        input_error = np.mean((sample_data.traces - clean) ** 2)
        output_error = np.mean((result.traces - clean) ** 2)

        # Denoised should have less error
        assert output_error < input_error

    def test_snr_improvement(self, sample_data):
        """Test SNR improves after denoising."""
        processor = GaborDenoise(
            aperture=7,
            threshold_k=2.5,
            window_size=64
        )
        result = processor.process(sample_data)

        clean = sample_data.metadata['clean']

        # Input SNR
        input_noise = sample_data.traces - clean
        input_signal_power = np.mean(clean ** 2)
        input_noise_power = np.mean(input_noise ** 2)
        input_snr = 10 * np.log10(input_signal_power / input_noise_power)

        # Output SNR
        output_noise = result.traces - clean
        output_noise_power = np.mean(output_noise ** 2)
        output_snr = 10 * np.log10(input_signal_power / output_noise_power)

        # SNR should improve
        assert output_snr > input_snr

    def test_soft_vs_hard_threshold(self, sample_data):
        """Test soft and hard threshold produce different results."""
        processor_soft = GaborDenoise(
            aperture=5,
            threshold_k=3.0,
            threshold_mode='soft',
            window_size=32
        )
        processor_hard = GaborDenoise(
            aperture=5,
            threshold_k=3.0,
            threshold_mode='hard',
            window_size=32
        )

        result_soft = processor_soft.process(sample_data)
        result_hard = processor_hard.process(sample_data)

        # Results should be different
        assert not np.allclose(result_soft.traces, result_hard.traces)

    def test_frequency_band_filtering(self, sample_data):
        """Test frequency filtering works correctly."""
        # Narrow frequency band
        processor = GaborDenoise(
            aperture=5,
            fmin=15.0,
            fmax=25.0,
            threshold_k=3.0,
            window_size=64
        )
        result = processor.process(sample_data)

        # Should still produce valid output
        assert result.traces.shape == sample_data.traces.shape
        assert not np.any(np.isnan(result.traces))

    def test_energy_retention(self, sample_data):
        """Test reasonable energy is retained."""
        processor = GaborDenoise(
            aperture=7,
            threshold_k=3.0,
            window_size=64
        )
        result = processor.process(sample_data)

        input_energy = np.sum(sample_data.traces ** 2)
        output_energy = np.sum(result.traces ** 2)

        # Should retain reasonable energy (not over-aggressive)
        energy_ratio = output_energy / input_energy
        assert 0.3 < energy_ratio < 1.5


class TestGaborDenoiseEdgeCases:
    """Tests for edge cases."""

    def test_small_gather(self):
        """Test with fewer traces than aperture."""
        np.random.seed(42)
        data = SeismicData(
            traces=np.random.randn(100, 3),
            sample_rate=2.0
        )
        processor = GaborDenoise(aperture=7, window_size=32)
        result = processor.process(data)
        assert result.traces.shape == data.traces.shape

    def test_single_trace(self):
        """Test with single trace."""
        np.random.seed(42)
        data = SeismicData(
            traces=np.random.randn(100, 1),
            sample_rate=2.0
        )
        processor = GaborDenoise(aperture=3, window_size=16)
        result = processor.process(data)
        assert result.traces.shape == data.traces.shape

    def test_short_trace(self):
        """Test with short trace."""
        np.random.seed(42)
        data = SeismicData(
            traces=np.random.randn(50, 10),
            sample_rate=2.0
        )
        processor = GaborDenoise(aperture=5, window_size=16)
        result = processor.process(data)
        assert result.traces.shape == data.traces.shape

    def test_no_noise_passthrough(self):
        """Test clean signal passes through with minimal distortion."""
        np.random.seed(42)
        n_samples = 200
        n_traces = 10
        t = np.arange(n_samples) * 0.002

        # Create clean signal
        clean = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            clean[:, i] = np.sin(2 * np.pi * 30 * t)

        data = SeismicData(traces=clean, sample_rate=2.0)
        processor = GaborDenoise(
            aperture=5,
            threshold_k=3.0,
            window_size=32
        )
        result = processor.process(data)

        # Clean signal should be mostly preserved
        correlation = np.corrcoef(clean.flatten(), result.traces.flatten())[0, 1]
        assert correlation > 0.9


class TestGaborDenoiseDescription:
    """Tests for processor description."""

    def test_description_format(self):
        """Test description string format."""
        processor = GaborDenoise(
            aperture=7,
            fmin=10.0,
            fmax=80.0,
            threshold_k=2.5,
            threshold_mode='soft',
            window_size=64
        )
        desc = processor.get_description()

        assert 'Gabor' in desc
        assert 'aperture=7' in desc
        assert '10-80Hz' in desc
        assert 'k=2.5' in desc
        assert 'soft' in desc
        assert 'win=64' in desc

    def test_description_auto_sigma(self):
        """Test description shows auto sigma."""
        processor = GaborDenoise(sigma=None)
        desc = processor.get_description()
        assert 'sigma=auto' in desc

    def test_description_explicit_sigma(self):
        """Test description shows explicit sigma."""
        processor = GaborDenoise(sigma=10.5)
        desc = processor.get_description()
        assert 'sigma=10.5' in desc


class TestGaborVsSTFT:
    """Tests comparing Gabor to standard STFT."""

    @pytest.fixture
    def chirp_data(self):
        """Create chirp signal data."""
        np.random.seed(42)
        n_samples = 500
        n_traces = 10
        sample_rate = 2.0
        t = np.arange(n_samples) * sample_rate / 1000

        # Create chirp (frequency sweep)
        clean = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            chirp = signal.chirp(t, f0=10, f1=100, t1=t[-1], method='linear')
            clean[:, i] = chirp

        noise = np.random.randn(n_samples, n_traces) * 0.3
        return SeismicData(
            traces=clean + noise,
            sample_rate=sample_rate,
            metadata={'clean': clean}
        )

    def test_gabor_on_chirp(self, chirp_data):
        """Test Gabor performs well on chirp signals."""
        processor = GaborDenoise(
            aperture=5,
            threshold_k=2.5,
            window_size=64,
            overlap_percent=75.0
        )
        result = processor.process(chirp_data)

        clean = chirp_data.metadata['clean']

        # Calculate correlation with clean signal
        correlation = np.corrcoef(
            clean.flatten(),
            result.traces.flatten()
        )[0, 1]

        # Should have good correlation
        assert correlation > 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
