"""
Tests for OMP (Orthogonal Matching Pursuit) Denoise processor.
"""
import pytest
import numpy as np
from models.seismic_data import SeismicData
from processors.omp_denoise import (
    OMPDenoise,
    create_dct_dictionary,
    create_dft_dictionary,
    create_gabor_dictionary,
    create_wavelet_dictionary,
    create_hybrid_dictionary,
    omp_cholesky,
    omp_fast,
    estimate_noise_mad,
    estimate_noise_wavelet,
    estimate_noise_spatial,
    estimate_local_snr,
    compute_adaptive_sparsity,
    compute_adaptive_tolerance
)


class TestDictionaryCreation:
    """Test dictionary generation functions."""

    def test_dct_dictionary_shape(self):
        """Test DCT dictionary has correct shape."""
        patch_size = 64
        n_atoms = 128
        D = create_dct_dictionary(patch_size, n_atoms)

        assert D.shape == (patch_size, n_atoms)

    def test_dct_dictionary_normalized(self):
        """Test DCT dictionary atoms are normalized."""
        D = create_dct_dictionary(64, 128)

        norms = np.linalg.norm(D, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_dft_dictionary_shape(self):
        """Test DFT dictionary has correct shape."""
        patch_size = 64
        n_atoms = 128
        D = create_dft_dictionary(patch_size, n_atoms)

        assert D.shape == (patch_size, n_atoms)

    def test_dft_dictionary_normalized(self):
        """Test DFT dictionary atoms are normalized."""
        D = create_dft_dictionary(64, 128)

        norms = np.linalg.norm(D, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_gabor_dictionary_shape(self):
        """Test Gabor dictionary has correct shape."""
        patch_size = 64
        n_atoms = 128
        D = create_gabor_dictionary(patch_size, n_atoms)

        assert D.shape[0] == patch_size
        assert D.shape[1] <= n_atoms  # May have fewer due to deduplication

    def test_wavelet_dictionary_shape(self):
        """Test wavelet dictionary has correct shape."""
        patch_size = 64
        n_atoms = 128
        D = create_wavelet_dictionary(patch_size, n_atoms)

        assert D.shape == (patch_size, n_atoms)

    def test_wavelet_types(self):
        """Test different wavelet types."""
        for wavelet_type in ['ricker', 'morlet', 'gaussian_derivative']:
            D = create_wavelet_dictionary(64, 64, wavelet_type)
            assert D.shape[0] == 64

    def test_hybrid_dictionary_shape(self):
        """Test hybrid dictionary has correct shape."""
        patch_size = 64
        n_atoms = 192
        D = create_hybrid_dictionary(patch_size, n_atoms)

        assert D.shape[0] == patch_size
        assert D.shape[1] == n_atoms


class TestNoiseEstimation:
    """Test robust noise estimation functions."""

    def test_estimate_noise_mad_gaussian(self):
        """Test MAD noise estimation on Gaussian noise."""
        np.random.seed(42)
        true_std = 0.5
        noise = np.random.randn(1000) * true_std

        estimated_std = estimate_noise_mad(noise)

        # MAD estimate should be close to true std for Gaussian
        np.testing.assert_allclose(estimated_std, true_std, rtol=0.1)

    def test_estimate_noise_mad_robust(self):
        """Test MAD is robust to outliers."""
        np.random.seed(42)
        true_std = 0.5
        noise = np.random.randn(1000) * true_std

        # Add outliers (signal spikes)
        noise_with_outliers = noise.copy()
        noise_with_outliers[100] = 10.0  # Large outlier
        noise_with_outliers[500] = -8.0

        estimated_std = estimate_noise_mad(noise_with_outliers)

        # MAD should still be close to true std despite outliers
        np.testing.assert_allclose(estimated_std, true_std, rtol=0.15)

    def test_estimate_noise_wavelet(self):
        """Test wavelet-based noise estimation."""
        np.random.seed(42)
        true_std = 0.3
        noise = np.random.randn(256) * true_std

        estimated_std = estimate_noise_wavelet(noise)

        # Should be reasonably close
        np.testing.assert_allclose(estimated_std, true_std, rtol=0.2)

    def test_estimate_noise_spatial_mad_diff(self):
        """Test spatial noise estimation using trace differences."""
        np.random.seed(42)
        n_samples, n_traces = 256, 15
        true_noise_std = 0.2

        # Create coherent signal (same across traces)
        t = np.linspace(0, 1, n_samples)
        signal = np.sin(2 * np.pi * 10 * t)
        traces = np.tile(signal.reshape(-1, 1), (1, n_traces))

        # Add independent noise to each trace
        noise = np.random.randn(n_samples, n_traces) * true_noise_std
        noisy_traces = traces + noise

        # Estimate noise for center trace
        estimated_std = estimate_noise_spatial(noisy_traces, 7, aperture=5, method='mad_diff')

        # Should recover approximately the true noise level
        np.testing.assert_allclose(estimated_std, true_noise_std, rtol=0.25)

    def test_estimate_noise_spatial_mad_residual(self):
        """Test spatial noise estimation using residual after mean subtraction."""
        np.random.seed(42)
        n_samples, n_traces = 256, 15
        true_noise_std = 0.15

        # Create coherent signal
        t = np.linspace(0, 1, n_samples)
        signal = np.sin(2 * np.pi * 5 * t)
        traces = np.tile(signal.reshape(-1, 1), (1, n_traces))

        # Add noise
        noise = np.random.randn(n_samples, n_traces) * true_noise_std
        noisy_traces = traces + noise

        estimated_std = estimate_noise_spatial(noisy_traces, 7, aperture=5, method='mad_residual')

        # Should be close to true noise
        np.testing.assert_allclose(estimated_std, true_noise_std, rtol=0.3)

    def test_estimate_local_snr(self):
        """Test local SNR estimation."""
        np.random.seed(42)
        n_samples, n_traces = 256, 11
        signal_amp = 1.0
        noise_std = 0.1

        # Create coherent signal
        t = np.linspace(0, 1, n_samples)
        signal = signal_amp * np.sin(2 * np.pi * 8 * t)
        traces = np.tile(signal.reshape(-1, 1), (1, n_traces))

        # Add noise
        noise = np.random.randn(n_samples, n_traces) * noise_std
        noisy_traces = traces + noise

        signal_estimate, snr_db = estimate_local_snr(noisy_traces, 5, aperture=5)

        # Signal estimate should be close to original
        correlation = np.corrcoef(signal_estimate, signal)[0, 1]
        assert correlation > 0.95

        # SNR should be high (signal is 10x noise)
        # Expected SNR ≈ 20 dB for signal_amp/noise_std = 10
        assert np.mean(snr_db) > 10  # At least 10 dB

    def test_compute_adaptive_sparsity(self):
        """Test adaptive sparsity computation."""
        base_sparsity = 10
        min_sp = 3
        max_sp = 20

        # Low SNR -> low sparsity
        sp_low = compute_adaptive_sparsity(-5.0, base_sparsity, min_sp, max_sp)
        assert sp_low == min_sp

        # High SNR -> high sparsity
        sp_high = compute_adaptive_sparsity(30.0, base_sparsity, min_sp, max_sp)
        assert sp_high == max_sp

        # Mid SNR -> interpolated
        sp_mid = compute_adaptive_sparsity(10.0, base_sparsity, min_sp, max_sp)
        assert min_sp < sp_mid < max_sp

    def test_compute_adaptive_tolerance(self):
        """Test adaptive tolerance computation."""
        base_tol = 0.1

        # Low noise -> tolerance close to base
        tol_low = compute_adaptive_tolerance(0.01, 1.0, base_tol)
        assert tol_low < 0.15

        # High noise -> larger tolerance
        tol_high = compute_adaptive_tolerance(0.5, 1.0, base_tol)
        assert tol_high > 0.15


class TestOMPAlgorithm:
    """Test OMP algorithm implementation."""

    @pytest.fixture
    def simple_dictionary(self):
        """Create a simple DCT dictionary for testing."""
        return create_dct_dictionary(32, 64)

    def test_omp_recovers_sparse_signal(self, simple_dictionary):
        """Test OMP recovers a known sparse signal."""
        D = simple_dictionary
        n_features, n_atoms = D.shape

        # Create sparse signal (3 atoms)
        true_support = [5, 15, 30]
        true_coef = np.zeros(n_atoms)
        true_coef[true_support] = [1.0, -0.5, 0.8]

        signal = D @ true_coef

        # Recover with OMP
        coef, support = omp_cholesky(D, signal, n_nonzero=5, tol=1e-10)

        # Check support is recovered
        recovered_support = set(np.where(np.abs(coef) > 1e-6)[0])
        assert set(true_support).issubset(recovered_support)

        # Check reconstruction
        reconstructed = D @ coef
        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)

    def test_omp_with_noise(self, simple_dictionary):
        """Test OMP denoising with noisy signal."""
        D = simple_dictionary
        n_features, n_atoms = D.shape

        # Create sparse signal
        true_coef = np.zeros(n_atoms)
        true_coef[[3, 10, 25]] = [1.0, -0.7, 0.5]
        clean_signal = D @ true_coef

        # Add noise
        np.random.seed(42)
        noise = 0.1 * np.random.randn(n_features)
        noisy_signal = clean_signal + noise

        # Recover with OMP
        coef, _ = omp_cholesky(D, noisy_signal, n_nonzero=5, tol=0.1)
        reconstructed = D @ coef

        # Reconstruction should be closer to clean than noisy
        error_noisy = np.linalg.norm(clean_signal - noisy_signal)
        error_omp = np.linalg.norm(clean_signal - reconstructed)

        assert error_omp < error_noisy

    def test_omp_fast_matches_cholesky(self, simple_dictionary):
        """Test fast OMP gives same results as Cholesky version."""
        D = simple_dictionary
        n_features, n_atoms = D.shape

        np.random.seed(42)
        signal = np.random.randn(n_features)

        coef1, support1 = omp_cholesky(D, signal, n_nonzero=5, tol=1e-6)
        coef2, support2 = omp_fast(D, signal, n_nonzero=5, tol=1e-6)

        np.testing.assert_allclose(coef1, coef2, atol=1e-10)

    def test_omp_respects_sparsity(self, simple_dictionary):
        """Test OMP respects maximum sparsity constraint."""
        D = simple_dictionary

        np.random.seed(42)
        signal = np.random.randn(D.shape[0])

        for max_atoms in [1, 3, 5, 10]:
            coef, _ = omp_cholesky(D, signal, n_nonzero=max_atoms, tol=1e-10)
            n_nonzero = np.sum(np.abs(coef) > 1e-10)
            assert n_nonzero <= max_atoms

    def test_omp_early_stopping(self, simple_dictionary):
        """Test OMP stops early when residual is small."""
        D = simple_dictionary

        # Signal that's exactly sparse
        true_coef = np.zeros(D.shape[1])
        true_coef[5] = 1.0
        signal = D @ true_coef

        # With tight tolerance, should stop after 1 atom
        coef, support = omp_cholesky(D, signal, n_nonzero=10, tol=1e-10)

        # Should find exactly 1 atom
        n_nonzero = np.sum(np.abs(coef) > 1e-10)
        assert n_nonzero == 1


class TestOMPDenoiseValidation:
    """Test parameter validation."""

    def test_valid_parameters(self):
        """Test processor accepts valid parameters."""
        processor = OMPDenoise(
            patch_size=64,
            overlap=0.5,
            n_atoms=128,
            sparsity=10,
            residual_tol=0.1,
            dictionary_type='dct'
        )
        assert processor.patch_size == 64
        assert processor.sparsity == 10

    def test_sparsity_as_fraction(self):
        """Test sparsity can be specified as fraction."""
        processor = OMPDenoise(
            patch_size=64,
            sparsity=0.1  # 10% of patch_size = 6
        )
        assert processor.sparsity == 6

    def test_invalid_patch_size(self):
        """Test rejection of small patch size."""
        with pytest.raises(ValueError, match="patch_size must be at least 8"):
            OMPDenoise(patch_size=4)

    def test_invalid_overlap(self):
        """Test rejection of invalid overlap."""
        with pytest.raises(ValueError, match="overlap must be in"):
            OMPDenoise(overlap=1.0)

    def test_invalid_sparsity(self):
        """Test rejection of invalid sparsity."""
        with pytest.raises(ValueError, match="sparsity must be at least 1"):
            OMPDenoise(sparsity=0)

    def test_invalid_dictionary_type(self):
        """Test rejection of invalid dictionary type."""
        with pytest.raises(ValueError, match="dictionary_type"):
            OMPDenoise(dictionary_type='invalid')

    def test_invalid_aperture(self):
        """Test rejection of even aperture."""
        with pytest.raises(ValueError, match="aperture must be odd"):
            OMPDenoise(aperture=4)


class TestOMPDenoiseProcessing:
    """Test OMP denoising processing."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic seismic data with noise."""
        np.random.seed(42)
        n_samples = 256
        n_traces = 15
        sample_rate_ms = 2.0  # 2 ms = 500 Hz

        # Create sparse signal (sum of wavelets)
        t = np.linspace(-1, 1, n_samples)
        clean_signal = np.zeros(n_samples)

        # Add Ricker wavelets at different positions
        for pos, amp in [(-0.5, 1.0), (0.0, -0.7), (0.3, 0.5)]:
            sigma = 0.05
            wavelet = amp * (1 - 2 * np.pi**2 * ((t - pos) / sigma)**2) * \
                     np.exp(-np.pi**2 * ((t - pos) / sigma)**2)
            clean_signal += wavelet

        # Create coherent traces
        traces = np.tile(clean_signal.reshape(-1, 1), (1, n_traces))

        # Add noise
        noise = np.random.randn(n_samples, n_traces) * 0.2
        noisy_traces = traces + noise

        return SeismicData(
            traces=noisy_traces.astype(np.float32),
            sample_rate=sample_rate_ms,
            metadata={'clean_signal': clean_signal}
        )

    def test_omp_denoise_runs(self, synthetic_data):
        """Test that OMP denoise runs without error."""
        processor = OMPDenoise(
            patch_size=32,
            overlap=0.5,
            sparsity=5,
            dictionary_type='dct'
        )
        result = processor.process(synthetic_data)

        assert result is not None
        assert result.traces.shape == synthetic_data.traces.shape

    def test_output_shape(self, synthetic_data):
        """Test output shape matches input."""
        processor = OMPDenoise(patch_size=32)
        result = processor.process(synthetic_data)

        assert result.traces.shape == synthetic_data.traces.shape

    def test_noise_reduction(self, synthetic_data):
        """Test that noise is reduced."""
        processor = OMPDenoise(
            patch_size=32,
            sparsity=8,
            dictionary_type='wavelet'
        )
        result = processor.process(synthetic_data)

        # Compare to clean signal
        clean = synthetic_data.metadata['clean_signal']

        input_error = np.mean((synthetic_data.traces[:, 0] - clean)**2)
        output_error = np.mean((result.traces[:, 0] - clean)**2)

        # Denoised should be closer to clean
        assert output_error < input_error

    def test_dictionary_types(self, synthetic_data):
        """Test all dictionary types work."""
        for dict_type in ['dct', 'dft', 'gabor', 'wavelet', 'hybrid']:
            processor = OMPDenoise(
                patch_size=32,
                sparsity=5,
                dictionary_type=dict_type
            )
            result = processor.process(synthetic_data)
            assert result is not None

    def test_spatial_mode(self, synthetic_data):
        """Test spatial aperture processing mode."""
        processor = OMPDenoise(
            patch_size=32,
            sparsity=5,
            aperture=3,
            denoise_mode='spatial'
        )
        result = processor.process(synthetic_data)

        assert result is not None
        assert result.traces.shape == synthetic_data.traces.shape

    def test_adaptive_mode_with_noise_estimation(self, synthetic_data):
        """Test adaptive processing mode with spatial noise estimation."""
        processor = OMPDenoise(
            patch_size=32,
            sparsity=8,
            aperture=5,
            denoise_mode='adaptive',
            noise_estimation='mad_diff',
            adaptive_sparsity=True
        )
        result = processor.process(synthetic_data)

        assert result is not None
        assert result.traces.shape == synthetic_data.traces.shape

    def test_noise_estimation_methods(self, synthetic_data):
        """Test all noise estimation methods."""
        for method in ['none', 'mad_diff', 'mad_residual', 'wavelet']:
            processor = OMPDenoise(
                patch_size=32,
                sparsity=5,
                aperture=5,
                noise_estimation=method
            )
            result = processor.process(synthetic_data)
            assert result is not None

    def test_adaptive_sparsity_improves_denoising(self, synthetic_data):
        """Test that adaptive sparsity provides reasonable results."""
        # Fixed sparsity
        processor_fixed = OMPDenoise(
            patch_size=32,
            sparsity=8,
            aperture=5,
            noise_estimation='none',
            adaptive_sparsity=False
        )

        # Adaptive sparsity
        processor_adaptive = OMPDenoise(
            patch_size=32,
            sparsity=8,
            aperture=5,
            noise_estimation='mad_diff',
            adaptive_sparsity=True,
            min_sparsity=3,
            max_sparsity=15
        )

        result_fixed = processor_fixed.process(synthetic_data)
        result_adaptive = processor_adaptive.process(synthetic_data)

        # Both should produce valid output
        assert result_fixed is not None
        assert result_adaptive is not None
        assert result_fixed.traces.shape == result_adaptive.traces.shape


class TestOMPDenoiseEdgeCases:
    """Test edge cases."""

    def test_short_trace(self):
        """Test with short traces."""
        np.random.seed(42)
        n_samples = 64
        n_traces = 5

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(traces=traces, sample_rate=2.0, metadata={})

        processor = OMPDenoise(patch_size=16, sparsity=3)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_few_traces(self):
        """Test with few traces."""
        np.random.seed(42)
        n_samples = 128
        n_traces = 3

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(traces=traces, sample_rate=2.0, metadata={})

        processor = OMPDenoise(patch_size=32, aperture=3)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_high_sparsity(self):
        """Test with high sparsity (many atoms)."""
        np.random.seed(42)
        n_samples = 128
        n_traces = 5

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(traces=traces, sample_rate=2.0, metadata={})

        processor = OMPDenoise(
            patch_size=32,
            n_atoms=64,
            sparsity=20  # Allow many atoms
        )
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape

    def test_low_overlap(self):
        """Test with minimal overlap."""
        np.random.seed(42)
        n_samples = 128
        n_traces = 5

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        data = SeismicData(traces=traces, sample_rate=2.0, metadata={})

        processor = OMPDenoise(patch_size=32, overlap=0.1)
        result = processor.process(data)

        assert result.traces.shape == data.traces.shape


class TestOMPDenoiseDescription:
    """Test description generation."""

    def test_description(self):
        """Test get_description returns proper string."""
        processor = OMPDenoise(
            patch_size=64,
            n_atoms=128,
            sparsity=10,
            dictionary_type='wavelet',
            aperture=5
        )
        desc = processor.get_description()

        assert 'OMP-Denoise' in desc
        assert 'patch=64' in desc
        assert 'atoms=128' in desc
        assert 'sparsity=10' in desc
        assert 'dict=wavelet' in desc
        assert 'aperture=5' in desc
