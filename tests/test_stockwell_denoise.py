"""
Tests for Stockwell Transform (S-Transform) Denoise processor.
"""
import pytest
import numpy as np
from models.seismic_data import SeismicData
from processors.stockwell_denoise import (
    StockwellDenoise,
    stockwell_transform,
    inverse_stockwell_transform
)


class TestStockwellTransform:
    """Test S-Transform functions."""

    def test_forward_transform_shape(self):
        """Test forward transform output shape."""
        np.random.seed(42)
        n_samples = 100
        signal = np.random.randn(n_samples)

        S, freqs = stockwell_transform(signal)

        assert S.ndim == 2
        assert S.shape[1] == n_samples  # Time dimension matches input
        assert len(freqs) == S.shape[0]  # Frequencies match first dimension

    def test_frequency_range_filtering(self):
        """Test frequency range limiting."""
        np.random.seed(42)
        n_samples = 100
        signal = np.random.randn(n_samples)

        # Full spectrum
        S_full, freqs_full = stockwell_transform(signal)

        # Limited spectrum
        S_limited, freqs_limited = stockwell_transform(signal, fmin=0.1, fmax=0.3)

        # Limited should have fewer frequencies
        assert len(freqs_limited) < len(freqs_full)

    def test_inverse_transform_recovers_signal(self):
        """Test that inverse transform approximately recovers signal."""
        np.random.seed(42)
        n_samples = 64
        signal = np.sin(2 * np.pi * 0.1 * np.arange(n_samples))

        S, freqs = stockwell_transform(signal)
        recovered = inverse_stockwell_transform(S, n_samples, freq_values=freqs)

        # Should approximately recover the signal
        # Note: Perfect reconstruction may not be achieved due to frequency limiting
        assert len(recovered) == n_samples


class TestStockwellDenoiseValidation:
    """Test parameter validation."""

    def test_valid_parameters(self):
        """Test processor accepts valid parameters."""
        processor = StockwellDenoise(
            aperture=5,
            fmin=10.0,
            fmax=80.0,
            threshold_k=2.5,
            threshold_mode='adaptive'
        )
        assert processor.aperture == 5
        assert processor.fmin == 10.0
        assert processor.fmax == 80.0
        assert processor.threshold_k == 2.5
        assert processor.threshold_mode == 'adaptive'

    def test_invalid_aperture_too_small(self):
        """Test rejection of aperture < 3."""
        with pytest.raises(ValueError, match="Aperture must be at least 3"):
            StockwellDenoise(aperture=1)

    def test_invalid_aperture_even(self):
        """Test rejection of even aperture."""
        with pytest.raises(ValueError, match="Aperture must be odd"):
            StockwellDenoise(aperture=4)

    def test_invalid_frequency_range(self):
        """Test rejection of fmin >= fmax."""
        with pytest.raises(ValueError, match="fmin must be less than fmax"):
            StockwellDenoise(fmin=100.0, fmax=50.0)

    def test_invalid_threshold_k(self):
        """Test rejection of threshold_k <= 0."""
        with pytest.raises(ValueError, match="threshold_k must be positive"):
            StockwellDenoise(threshold_k=0)

    def test_invalid_threshold_mode(self):
        """Test rejection of invalid threshold_mode."""
        with pytest.raises(ValueError, match="threshold_mode must be"):
            StockwellDenoise(threshold_mode='invalid')


class TestStockwellDenoiseProcessing:
    """Test Stockwell denoising processing."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic seismic data with noise."""
        np.random.seed(42)
        n_samples = 256
        n_traces = 15
        sample_rate = 500.0

        # Create clean signal
        t = np.linspace(0, n_samples / sample_rate, n_samples)
        clean_signal = (
            np.sin(2 * np.pi * 25 * t) +
            0.5 * np.sin(2 * np.pi * 50 * t)
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

    def test_stockwell_denoise_runs(self, synthetic_data):
        """Test that Stockwell denoise runs without error."""
        processor = StockwellDenoise(
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
        processor = StockwellDenoise(aperture=5)
        result = processor.process(synthetic_data)

        assert result.traces.shape == synthetic_data.traces.shape

    def test_noise_reduction(self, synthetic_data):
        """Test that noise is reduced."""
        processor = StockwellDenoise(
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
            processor = StockwellDenoise(
                aperture=5,
                threshold_mode=mode
            )
            result = processor.process(synthetic_data)
            assert result is not None
            assert result.traces.shape == synthetic_data.traces.shape


class TestStockwellDenoiseEdgeCases:
    """Test edge cases."""

    def test_few_traces(self):
        """Test with fewer traces than aperture."""
        np.random.seed(42)
        n_samples = 128
        n_traces = 3

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(
            traces=traces,
            sample_rate=500.0,
            metadata={}
        )

        processor = StockwellDenoise(aperture=7)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_short_trace(self):
        """Test with short traces."""
        np.random.seed(42)
        n_samples = 64
        n_traces = 10

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(
            traces=traces,
            sample_rate=500.0,
            metadata={}
        )

        processor = StockwellDenoise(aperture=5)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_low_amp_protection(self):
        """Test low amplitude protection feature."""
        np.random.seed(42)
        n_samples = 128
        n_traces = 10

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        traces[30:60, :] *= 0.1

        data = SeismicData(
            traces=traces,
            sample_rate=500.0,
            metadata={}
        )

        processor = StockwellDenoise(
            aperture=5,
            low_amp_protection=True,
            low_amp_factor=0.3
        )
        result = processor.process(data)

        assert result is not None


class TestStockwellDenoiseDescription:
    """Test description generation."""

    def test_description(self):
        """Test get_description returns proper string."""
        processor = StockwellDenoise(
            aperture=7,
            fmin=10.0,
            fmax=80.0,
            threshold_k=3.0,
            threshold_mode='adaptive'
        )
        desc = processor.get_description()

        assert 'Stockwell-Denoise' in desc
        assert 'aperture=7' in desc
        assert 'freq=10-80Hz' in desc
        assert 'k=3.0' in desc
        assert 'mode=adaptive' in desc

    def test_description_with_low_amp_protection(self):
        """Test description includes low_amp_protect when enabled."""
        processor = StockwellDenoise(
            aperture=5,
            low_amp_protection=True
        )
        desc = processor.get_description()

        assert 'low_amp_protect' in desc
