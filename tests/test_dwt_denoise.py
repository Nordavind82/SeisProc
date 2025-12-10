"""
Tests for DWT (Discrete Wavelet Transform) denoising processor.

Validates correctness, performance, and comparison with STFT.
"""
import numpy as np
import pytest
import time
import sys
sys.path.insert(0, '.')

from processors.dwt_denoise import DWTDenoise, PYWT_AVAILABLE
from processors.tf_denoise import TFDenoise
from models.seismic_data import SeismicData


# Skip all tests if PyWavelets not available
pytestmark = pytest.mark.skipif(not PYWT_AVAILABLE, reason="PyWavelets not installed")


def create_test_data(n_samples=1000, n_traces=20, noise_level=0.1, seed=42):
    """Create synthetic seismic data with known signal and noise."""
    sample_rate_hz = 500.0
    t = np.arange(n_samples) / sample_rate_hz

    def ricker_wavelet(f, t, t0):
        tau = t - t0
        return (1 - 2*(np.pi*f*tau)**2) * np.exp(-(np.pi*f*tau)**2)

    # Clean signal
    clean_traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        offset = i * 0.001
        clean_traces[:, i] = (
            1.0 * ricker_wavelet(35, t, 0.3 + offset) +
            0.7 * ricker_wavelet(45, t, 0.6 + offset*0.8) +
            0.5 * ricker_wavelet(30, t, 1.0 + offset*0.5)
        )

    # Add noise
    np.random.seed(seed)
    noise = np.random.randn(n_samples, n_traces) * noise_level
    noisy_traces = clean_traces + noise

    return clean_traces, noisy_traces, noise


class TestDWTDenoiseBasic:
    """Basic functionality tests."""

    def test_dwt_mode_runs(self):
        """Test that DWT mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=500, n_traces=10)

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='dwt'
        )

        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape
        assert not np.allclose(result.traces, noisy)  # Something changed

    def test_swt_mode_runs(self):
        """Test that SWT mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=512, n_traces=10)  # Power of 2 for SWT

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='swt'
        )

        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_dwt_spatial_mode_runs(self):
        """Test that DWT spatial mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=500, n_traces=15)

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='dwt_spatial',
            aperture=5
        )

        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape


class TestDWTDenoiseQuality:
    """Quality and correctness tests."""

    def test_snr_improvement(self):
        """Test that denoising improves SNR."""
        clean, noisy, noise = create_test_data(noise_level=0.08)

        processor = DWTDenoise(wavelet='db4', threshold_k=2.5, transform_type='dwt')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        # Calculate SNR
        input_snr = 10*np.log10(np.mean(clean**2)/np.mean(noise**2))
        error = clean - result.traces
        output_snr = 10*np.log10(np.mean(clean**2)/np.mean(error**2))

        assert output_snr > input_snr, f"SNR should improve: {input_snr:.1f} -> {output_snr:.1f} dB"

    def test_signal_preservation(self):
        """Test that signal is preserved (high correlation with clean)."""
        clean, noisy, _ = create_test_data(noise_level=0.05)

        processor = DWTDenoise(wavelet='db4', threshold_k=3.0, transform_type='dwt')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        # Check correlation
        for i in range(clean.shape[1]):
            corr = np.corrcoef(clean[:, i], result.traces[:, i])[0, 1]
            assert corr > 0.8, f"Trace {i} correlation too low: {corr:.3f}"

    def test_different_wavelets(self):
        """Test various wavelet families."""
        clean, noisy, _ = create_test_data(noise_level=0.1)

        wavelets = ['db4', 'db8', 'sym4', 'sym8', 'coif4']
        results = {}

        for wavelet in wavelets:
            processor = DWTDenoise(wavelet=wavelet, threshold_k=3.0)
            data = SeismicData(traces=noisy.copy(), sample_rate=2.0)
            result = processor.process(data)

            error = clean - result.traces
            snr = 10*np.log10(np.mean(clean**2)/np.mean(error**2))
            results[wavelet] = snr

        # All should provide reasonable SNR improvement
        for wavelet, snr in results.items():
            assert snr > 0, f"Wavelet {wavelet} should improve SNR"


class TestDWTDenoisePerformance:
    """Performance benchmarks."""

    def test_dwt_faster_than_stft(self):
        """Test that DWT is faster than STFT."""
        clean, noisy, _ = create_test_data(n_samples=1000, n_traces=50)

        # DWT timing
        dwt_processor = DWTDenoise(wavelet='db4', transform_type='dwt')
        data = SeismicData(traces=noisy.copy(), sample_rate=2.0)

        start = time.time()
        dwt_processor.process(data)
        dwt_time = time.time() - start

        # STFT timing
        stft_processor = TFDenoise(
            aperture=7, fmin=5.0, fmax=200.0,
            threshold_k=3.0, transform_type='stft'
        )

        start = time.time()
        stft_processor.process(data)
        stft_time = time.time() - start

        speedup = stft_time / dwt_time
        assert speedup > 1.5, f"DWT should be faster: {speedup:.1f}x"

    def test_throughput_reasonable(self):
        """Test processing throughput is reasonable."""
        clean, noisy, _ = create_test_data(n_samples=1000, n_traces=100)

        processor = DWTDenoise(wavelet='db4', transform_type='dwt')
        data = SeismicData(traces=noisy, sample_rate=2.0)

        start = time.time()
        processor.process(data)
        elapsed = time.time() - start

        throughput = 100 / elapsed  # traces per second
        assert throughput > 500, f"Throughput should be > 500 traces/s: {throughput:.0f}"


class TestDWTvsSTFTComparison:
    """Direct comparison between DWT and STFT."""

    def test_quality_comparison(self):
        """Compare quality metrics between DWT and STFT."""
        clean, noisy, noise = create_test_data(noise_level=0.08)

        input_snr = 10*np.log10(np.mean(clean**2)/np.mean(noise**2))

        # DWT
        dwt_proc = DWTDenoise(wavelet='db4', threshold_k=2.5, transform_type='dwt')
        dwt_result = dwt_proc.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))
        dwt_error = clean - dwt_result.traces
        dwt_snr = 10*np.log10(np.mean(clean**2)/np.mean(dwt_error**2))

        # SWT
        swt_proc = DWTDenoise(wavelet='db4', threshold_k=2.5, transform_type='swt')
        swt_result = swt_proc.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))
        swt_error = clean - swt_result.traces
        swt_snr = 10*np.log10(np.mean(clean**2)/np.mean(swt_error**2))

        # STFT
        stft_proc = TFDenoise(
            aperture=7, fmin=5.0, fmax=200.0,
            threshold_k=2.5, transform_type='stft'
        )
        stft_result = stft_proc.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))
        stft_error = clean - stft_result.traces
        stft_snr = 10*np.log10(np.mean(clean**2)/np.mean(stft_error**2))

        print(f"\nInput SNR: {input_snr:.1f} dB")
        print(f"DWT SNR:   {dwt_snr:.1f} dB (+{dwt_snr-input_snr:.1f})")
        print(f"SWT SNR:   {swt_snr:.1f} dB (+{swt_snr-input_snr:.1f})")
        print(f"STFT SNR:  {stft_snr:.1f} dB (+{stft_snr-input_snr:.1f})")

        # All should improve SNR
        assert dwt_snr > input_snr
        assert swt_snr > input_snr


class TestDWTDenoiseEdgeCases:
    """Edge case handling tests."""

    def test_short_signal(self):
        """Test with very short signal."""
        clean, noisy, _ = create_test_data(n_samples=64, n_traces=5)

        processor = DWTDenoise(wavelet='db4', transform_type='dwt')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_single_trace(self):
        """Test with single trace."""
        clean, noisy, _ = create_test_data(n_samples=500, n_traces=1)

        processor = DWTDenoise(wavelet='db4', transform_type='dwt')
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_pure_noise_doesnt_crash(self):
        """Test with pure noise input."""
        np.random.seed(42)
        noise_only = np.random.randn(500, 10)

        processor = DWTDenoise(wavelet='db4', transform_type='dwt')
        data = SeismicData(traces=noise_only, sample_rate=2.0)
        result = processor.process(data)

        # Should attenuate noise
        assert np.sqrt(np.mean(result.traces**2)) < np.sqrt(np.mean(noise_only**2))


class TestWPTDenoise:
    """Tests for Wavelet Packet Transform denoising."""

    def test_wpt_mode_runs(self):
        """Test that WPT mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=512, n_traces=10)

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='wpt',
            level=4
        )

        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape
        assert not np.allclose(result.traces, noisy)

    def test_wpt_spatial_mode_runs(self):
        """Test that WPT spatial mode processes without error."""
        clean, noisy, _ = create_test_data(n_samples=512, n_traces=15)

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='wpt_spatial',
            aperture=5,
            level=4
        )

        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_wpt_best_basis(self):
        """Test WPT with best-basis selection."""
        clean, noisy, _ = create_test_data(n_samples=512, n_traces=10)

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='wpt',
            level=4,
            best_basis=True
        )

        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        assert result.traces.shape == noisy.shape

    def test_wpt_snr_improvement(self):
        """Test that WPT improves SNR."""
        clean, noisy, noise = create_test_data(n_samples=512, noise_level=0.08)

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=2.5,
            transform_type='wpt',
            level=4
        )
        data = SeismicData(traces=noisy, sample_rate=2.0)
        result = processor.process(data)

        input_snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
        error = clean - result.traces
        output_snr = 10 * np.log10(np.mean(clean**2) / np.mean(error**2))

        assert output_snr > input_snr, f"WPT should improve SNR: {input_snr:.1f} -> {output_snr:.1f} dB"

    def test_wpt_vs_dwt_comparison(self):
        """Compare WPT and DWT results."""
        clean, noisy, _ = create_test_data(n_samples=512, n_traces=10, noise_level=0.1)

        # DWT
        dwt_proc = DWTDenoise(wavelet='db4', threshold_k=3.0, transform_type='dwt')
        dwt_result = dwt_proc.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))

        # WPT
        wpt_proc = DWTDenoise(wavelet='db4', threshold_k=3.0, transform_type='wpt', level=4)
        wpt_result = wpt_proc.process(SeismicData(traces=noisy.copy(), sample_rate=2.0))

        # Both should produce valid output
        assert dwt_result.traces.shape == noisy.shape
        assert wpt_result.traces.shape == noisy.shape

        # Results should differ (WPT uses full tree)
        assert not np.allclose(dwt_result.traces, wpt_result.traces)

    def test_wpt_frequency_selectivity(self):
        """Test WPT with narrowband signal produces valid output."""
        np.random.seed(42)
        n_samples = 512
        n_traces = 5
        sample_rate = 2.0  # ms
        fs = 1000.0 / sample_rate  # Hz

        t = np.arange(n_samples) / fs

        # Create narrowband signal (40 Hz)
        clean = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            clean[:, i] = np.sin(2 * np.pi * 40 * t)

        # Add broadband noise
        noise = np.random.randn(n_samples, n_traces) * 0.5  # Higher noise
        noisy = clean + noise

        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,  # More aggressive threshold
            transform_type='wpt',
            level=5
        )
        data = SeismicData(traces=noisy, sample_rate=sample_rate)
        result = processor.process(data)

        # WPT should produce valid output with high correlation to clean
        output_corr = np.corrcoef(clean.flatten(), result.traces.flatten())[0, 1]
        assert output_corr > 0.7, f"Output should correlate well with clean signal: {output_corr:.3f}"

        # Energy should be reduced (noise attenuated)
        input_energy = np.mean(noisy ** 2)
        output_energy = np.mean(result.traces ** 2)
        assert output_energy < input_energy, "WPT should reduce energy (attenuate noise)"


class TestWPTDescription:
    """Tests for WPT processor descriptions."""

    def test_wpt_description(self):
        """Test WPT description format."""
        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='wpt'
        )
        desc = processor.get_description()

        assert 'WPT' in desc
        assert 'db4' in desc

    def test_wpt_best_basis_description(self):
        """Test best-basis appears in description."""
        processor = DWTDenoise(
            wavelet='db4',
            threshold_k=3.0,
            transform_type='wpt',
            best_basis=True
        )
        desc = processor.get_description()

        assert 'best-basis' in desc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
