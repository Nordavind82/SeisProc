"""
Tests for EMD/EEMD Denoising processor.
"""
import numpy as np
import pytest

from models.seismic_data import SeismicData

# Try to import EMDDenoise
try:
    from processors.emd_denoise import EMDDenoise, PYEMD_AVAILABLE, get_imfs
except ImportError:
    PYEMD_AVAILABLE = False

# Skip all tests if PyEMD not available
pytestmark = pytest.mark.skipif(not PYEMD_AVAILABLE, reason="PyEMD not installed")


def create_test_data(n_samples=500, n_traces=5, noise_level=0.3, seed=42):
    """Create synthetic seismic data with mixed frequency components."""
    np.random.seed(seed)
    sample_rate = 2.0  # 2ms
    fs = 1000.0 / sample_rate  # Hz
    t = np.arange(n_samples) / fs

    # Create clean signal with multiple frequency components
    clean = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        # Low frequency component (should be in later IMFs)
        low_freq = np.sin(2 * np.pi * 10 * t)
        # Medium frequency component
        mid_freq = 0.7 * np.sin(2 * np.pi * 30 * t)
        # High frequency component (should be in early IMFs)
        high_freq = 0.5 * np.sin(2 * np.pi * 80 * t)

        clean[:, i] = low_freq + mid_freq + high_freq

    # Add high-frequency noise (should appear in first IMF)
    noise = np.random.randn(n_samples, n_traces) * noise_level

    return clean, clean + noise, noise


class TestEMDDenoiseValidation:
    """Tests for parameter validation."""

    def test_valid_parameters(self):
        """Test processor accepts valid parameters."""
        processor = EMDDenoise(
            method='eemd',
            remove_imfs='first',
            ensemble_size=50
        )
        assert processor.method == 'eemd'

    def test_valid_eemd_fast_parameters(self):
        """Test processor accepts EEMD-Fast parameters."""
        processor = EMDDenoise(
            method='eemd_fast',
            remove_imfs='first',
            ensemble_size=30,
            parallel_ensemble=True
        )
        assert processor.method == 'eemd_fast'
        assert processor.parallel_ensemble is True

    def test_invalid_method(self):
        """Test rejection of invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            EMDDenoise(method='invalid')

    def test_invalid_ensemble_size(self):
        """Test rejection of invalid ensemble size."""
        with pytest.raises(ValueError, match="ensemble_size must be"):
            EMDDenoise(ensemble_size=0)

    def test_invalid_noise_amplitude(self):
        """Test rejection of invalid noise amplitude."""
        with pytest.raises(ValueError, match="noise_amplitude must be"):
            EMDDenoise(noise_amplitude=-0.1)

    def test_invalid_remove_imfs_pattern(self):
        """Test rejection of invalid remove_imfs pattern."""
        with pytest.raises(ValueError, match="Invalid remove_imfs"):
            EMDDenoise(remove_imfs='invalid_pattern')


class TestEMDDenoiseProcessing:
    """Tests for EMD denoising functionality."""

    def test_emd_mode_runs(self):
        """Test EMD mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=256, n_traces=3)

        processor = EMDDenoise(method='emd', remove_imfs='first')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape
        assert not np.allclose(result.traces, noisy)

    def test_eemd_mode_runs(self):
        """Test EEMD mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=256, n_traces=3)

        processor = EMDDenoise(
            method='eemd',
            remove_imfs='first',
            ensemble_size=20  # Small for testing speed
        )
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_output_shape(self):
        """Test output has same shape as input."""
        clean, noisy, _ = create_test_data(n_samples=200, n_traces=5)

        processor = EMDDenoise(method='emd', remove_imfs='first')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_noise_reduction(self):
        """Test that high-frequency noise content is reduced."""
        clean, noisy, noise = create_test_data(n_samples=256, n_traces=3)

        processor = EMDDenoise(
            method='eemd',
            remove_imfs='first',
            ensemble_size=30
        )
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        # High-frequency content should be reduced
        # Compare std of first difference (proxy for high-freq content)
        input_hf = np.std(np.diff(noisy, axis=0))
        output_hf = np.std(np.diff(result.traces, axis=0))

        # EMD should reduce high-frequency noise
        assert output_hf < input_hf, f"HF content should decrease: {input_hf:.4f} -> {output_hf:.4f}"

    def test_remove_first_imf(self):
        """Test removing first IMF (high-frequency noise)."""
        clean, noisy, _ = create_test_data(n_samples=256)

        processor = EMDDenoise(method='emd', remove_imfs='first')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        # Output should have less high-frequency energy
        # Compare std of first difference (proxy for high-freq content)
        input_hf = np.std(np.diff(noisy, axis=0))
        output_hf = np.std(np.diff(result.traces, axis=0))

        assert output_hf < input_hf

    def test_remove_last_imf(self):
        """Test removing last IMF (trend/DC)."""
        # Create signal with trend
        np.random.seed(42)
        n_samples = 256
        n_traces = 3
        t = np.linspace(0, 1, n_samples)

        signal = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            # Add oscillation + linear trend
            signal[:, i] = np.sin(2 * np.pi * 10 * t) + 2 * t

        processor = EMDDenoise(method='emd', remove_imfs='last')
        data = SeismicData(traces=signal, sample_rate=2.0)
        result = processor.process(data)

        # Result should have less mean (trend removed)
        input_mean = np.abs(np.mean(signal))
        output_mean = np.abs(np.mean(result.traces))

        assert output_mean < input_mean

    def test_remove_multiple_imfs(self):
        """Test removing multiple IMFs."""
        clean, noisy, _ = create_test_data(n_samples=256)

        processor = EMDDenoise(method='emd', remove_imfs='first_2')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_remove_by_list(self):
        """Test removing specific IMF indices."""
        clean, noisy, _ = create_test_data(n_samples=256)

        processor = EMDDenoise(method='emd', remove_imfs=[0, 1])
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape


class TestEMDDenoiseEdgeCases:
    """Tests for edge cases."""

    def test_single_trace(self):
        """Test with single trace."""
        clean, noisy, _ = create_test_data(n_samples=200, n_traces=1)

        processor = EMDDenoise(method='emd', remove_imfs='first')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_short_trace(self):
        """Test with short trace."""
        np.random.seed(42)
        data = SeismicData(
            traces=np.random.randn(64, 3),
            sample_rate=2.0
        )

        processor = EMDDenoise(method='emd', remove_imfs='first')
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape


class TestEMDDescription:
    """Tests for processor description."""

    def test_description_emd(self):
        """Test EMD description."""
        processor = EMDDenoise(method='emd', remove_imfs='first')
        desc = processor.get_description()

        assert 'EMD' in desc
        assert 'first' in desc

    def test_description_eemd(self):
        """Test EEMD description."""
        processor = EMDDenoise(method='eemd', remove_imfs='last')
        desc = processor.get_description()

        assert 'EEMD' in desc
        assert 'last' in desc


class TestIMFExtraction:
    """Tests for IMF extraction utility."""

    def test_get_imfs(self):
        """Test IMF extraction utility function."""
        np.random.seed(42)
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
        signal += 0.5 * np.sin(2 * np.pi * 40 * np.linspace(0, 1, 256))

        imfs = get_imfs(signal, method='emd')

        # Should get multiple IMFs
        assert len(imfs) >= 2

        # Each IMF should have same length as signal
        for imf in imfs:
            assert len(imf) == len(signal)

        # Sum of IMFs should approximately equal original signal
        reconstructed = np.sum(imfs, axis=0)
        assert np.allclose(reconstructed, signal, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
