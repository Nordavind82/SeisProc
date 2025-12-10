"""
Gabor Transform Denoising processor.

Implements time-frequency domain denoising using Gabor Transform (STFT with
Gaussian windows) for optimal time-frequency localization following the
uncertainty principle.

The Gabor Transform provides better time-frequency resolution trade-off
than rectangular STFT windows, with smooth spectral estimates and reduced
spectral leakage.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Optional, Literal
import logging

from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_JOBS = max(1, multiprocessing.cpu_count() - 1)
except ImportError:
    JOBLIB_AVAILABLE = False
    N_JOBS = 1


def create_gaussian_window(nperseg: int, sigma: Optional[float] = None) -> np.ndarray:
    """
    Create a Gaussian window for Gabor Transform.

    Args:
        nperseg: Window length in samples
        sigma: Standard deviation of the Gaussian. If None, uses nperseg/6
               for optimal time-frequency localization.

    Returns:
        Gaussian window array
    """
    if sigma is None:
        # Default sigma for good time-frequency trade-off
        # sigma = nperseg/6 gives ~99.7% of energy within window
        sigma = nperseg / 6.0

    return signal.windows.gaussian(nperseg, std=sigma)


class GaborDenoise(BaseProcessor):
    """
    Gabor Transform Denoising using MAD-based thresholding.

    Uses STFT with Gaussian windows (Gabor Transform) for optimal
    time-frequency localization, combined with spatial aperture
    processing and robust MAD thresholding for noise removal.

    Advantages over standard STFT:
    - Better time-frequency resolution trade-off
    - Smooth spectral estimates
    - Reduced spectral leakage
    - Optimal for chirp-like signals

    Best suited for:
    - AVO analysis
    - Spectral decomposition
    - Thin bed detection
    """

    def __init__(self,
                 aperture: int = 7,
                 fmin: float = 5.0,
                 fmax: float = 100.0,
                 threshold_k: float = 3.0,
                 threshold_mode: Literal['soft', 'hard'] = 'soft',
                 window_size: int = 64,
                 sigma: Optional[float] = None,
                 overlap_percent: float = 75.0,
                 time_smoothing: int = 1,
                 low_amp_protection: bool = True,
                 low_amp_factor: float = 0.3):
        """
        Initialize Gabor Denoise processor.

        Args:
            aperture: Spatial aperture size (number of traces, must be odd)
            fmin: Minimum frequency (Hz) for processing
            fmax: Maximum frequency (Hz) for processing
            threshold_k: MAD threshold multiplier (higher = more aggressive)
            threshold_mode: Thresholding type ('soft' or 'hard')
            window_size: Gabor window size in samples (power of 2 recommended)
            sigma: Gaussian window standard deviation. If None, auto-calculated
                   for optimal time-frequency localization (window_size/6)
            overlap_percent: Window overlap percentage (50-90% typical)
            time_smoothing: Time bins for MAD smoothing (1=no smoothing)
            low_amp_protection: Prevent inflation of low-amplitude samples
            low_amp_factor: Threshold for low-amplitude protection (0-1)
        """
        self.aperture = aperture
        self.fmin = fmin
        self.fmax = fmax
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode
        self.window_size = window_size
        self.sigma = sigma
        self.overlap_percent = overlap_percent
        self.time_smoothing = time_smoothing
        self.low_amp_protection = low_amp_protection
        self.low_amp_factor = low_amp_factor

        # Call parent init which will call _validate_params
        super().__init__(
            aperture=aperture,
            fmin=fmin,
            fmax=fmax,
            threshold_k=threshold_k,
            threshold_mode=threshold_mode,
            window_size=window_size,
            sigma=sigma,
            overlap_percent=overlap_percent,
            time_smoothing=time_smoothing,
            low_amp_protection=low_amp_protection,
            low_amp_factor=low_amp_factor
        )

    def _validate_params(self):
        """Validate processor parameters."""
        if self.aperture < 3:
            raise ValueError("Aperture must be at least 3")
        if self.aperture % 2 == 0:
            raise ValueError("Aperture must be odd")
        if self.fmin >= self.fmax:
            raise ValueError("fmin must be less than fmax")
        if self.fmin < 0:
            raise ValueError("fmin must be non-negative")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be positive")
        if self.threshold_mode not in ['soft', 'hard']:
            raise ValueError("threshold_mode must be 'soft' or 'hard'")
        if self.window_size < 8:
            raise ValueError("window_size must be at least 8")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not 0 < self.overlap_percent < 100:
            raise ValueError("overlap_percent must be between 0 and 100")
        if self.time_smoothing < 1:
            raise ValueError("time_smoothing must be at least 1")
        if self.low_amp_factor <= 0 or self.low_amp_factor >= 1:
            raise ValueError("low_amp_factor must be between 0 and 1")

    def get_description(self) -> str:
        """Get processor description."""
        sigma_str = f"sigma={self.sigma:.1f}" if self.sigma else "sigma=auto"
        return (f"Gabor-Denoise: aperture={self.aperture}, "
                f"freq={self.fmin:.0f}-{self.fmax:.0f}Hz, "
                f"k={self.threshold_k:.1f}, {self.threshold_mode}, "
                f"win={self.window_size}, {sigma_str}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply Gabor Transform denoising to seismic data.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time_total = time.time()

        traces = data.traces.copy()
        n_samples, n_traces = traces.shape

        # Log processing info
        parallel_info = f"Parallel({N_JOBS} cores)" if JOBLIB_AVAILABLE else "Sequential"
        logger.info(
            f"Gabor-Denoise: {n_traces} traces × {n_samples} samples | "
            f"Aperture: {self.aperture} | Window: {self.window_size} | "
            f"Freq: {self.fmin:.0f}-{self.fmax:.0f}Hz | "
            f"k={self.threshold_k} | {parallel_info}"
        )

        # Validate aperture for data size
        if n_traces < self.aperture:
            logger.warning(
                f"Not enough traces ({n_traces}) for aperture ({self.aperture}), "
                f"using {n_traces if n_traces % 2 == 1 else n_traces - 1}"
            )
            effective_aperture = n_traces if n_traces % 2 == 1 else n_traces - 1
        else:
            effective_aperture = self.aperture

        # Adjust window size if needed
        nperseg = min(self.window_size, n_samples // 4)
        noverlap = int(nperseg * self.overlap_percent / 100.0)

        # Create Gaussian window for Gabor Transform
        gabor_window = create_gaussian_window(nperseg, self.sigma)

        # Calculate frequency normalization
        nyquist_freq = data.nyquist_freq
        sample_rate = 2.0 * nyquist_freq

        # Process traces
        denoised_traces = np.zeros_like(traces)
        half_aperture = effective_aperture // 2

        # Check if parallel processing is beneficial
        use_parallel = JOBLIB_AVAILABLE and n_traces > 50

        if use_parallel:
            def process_single_trace(trace_idx):
                start_idx = max(0, trace_idx - half_aperture)
                end_idx = min(n_traces, trace_idx + half_aperture + 1)
                ensemble = traces[:, start_idx:end_idx]
                center_in_ensemble = trace_idx - start_idx

                return self._process_gabor(
                    ensemble, center_in_ensemble,
                    nperseg, noverlap, gabor_window, sample_rate
                )

            results = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(process_single_trace)(i) for i in range(n_traces)
            )

            for i, result in enumerate(results):
                denoised_traces[:, i] = result
        else:
            # Sequential processing
            for trace_idx in range(n_traces):
                start_idx = max(0, trace_idx - half_aperture)
                end_idx = min(n_traces, trace_idx + half_aperture + 1)
                ensemble = traces[:, start_idx:end_idx]
                center_in_ensemble = trace_idx - start_idx

                denoised_traces[:, trace_idx] = self._process_gabor(
                    ensemble, center_in_ensemble,
                    nperseg, noverlap, gabor_window, sample_rate
                )

        # Compute timing and energy metrics
        elapsed_total = time.time() - start_time_total
        throughput = n_traces / elapsed_total if elapsed_total > 0 else 0

        # Energy verification
        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"Gabor-Denoise complete: {elapsed_total:.2f}s | "
            f"{throughput:.1f} traces/s | "
            f"Energy: {energy_ratio:.1%} retained"
        )

        if energy_ratio < 0.10:
            logger.warning("Output <10% of input energy - threshold may be too aggressive")
        elif energy_ratio > 0.95:
            logger.warning("Output >95% of input energy - minimal denoising occurred")

        return SeismicData(
            traces=denoised_traces,
            sample_rate=data.sample_rate,
            metadata={
                **data.metadata,
                'processor': self.get_description()
            }
        )

    def _process_gabor(self, ensemble, center_idx, nperseg, noverlap, window, sample_rate):
        """
        Process ensemble using Gabor Transform (STFT with Gaussian window).

        Args:
            ensemble: Spatial aperture traces (n_samples, n_traces)
            center_idx: Index of center trace in ensemble
            nperseg: Window size
            noverlap: Overlap samples
            window: Gaussian window array
            sample_rate: Sample rate in Hz

        Returns:
            Denoised center trace (n_samples,)
        """
        n_samples, n_traces = ensemble.shape
        center_trace = ensemble[:, center_idx]

        # Batch Gabor Transform: compute all traces at once using custom window
        freqs, times, stft_batch = signal.stft(
            ensemble.T,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            axis=-1
        )
        # stft_batch shape: (n_traces, n_freqs, n_times)
        n_freqs, n_times = stft_batch.shape[1], stft_batch.shape[2]

        if n_freqs == 0 or n_times == 0:
            return center_trace

        # Get center trace coefficients
        stft_center = stft_batch[center_idx]  # (n_freqs, n_times)

        # === VECTORIZED THRESHOLDING ===
        # Compute spatial statistics
        all_amplitudes = np.abs(stft_batch)  # (n_traces, n_freqs, n_times)
        median_amp = np.median(all_amplitudes, axis=0)  # (n_freqs, n_times)
        mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)

        # Apply temporal smoothing
        if self.time_smoothing > 1 and n_times >= self.time_smoothing:
            mad = uniform_filter1d(mad, size=self.time_smoothing, axis=1, mode='nearest')
            median_amp = uniform_filter1d(median_amp, size=self.time_smoothing, axis=1, mode='nearest')

        # Prevent MAD=0 issues
        min_mad = np.maximum(0.01 * median_amp, 1e-10)
        mad = np.maximum(mad, min_mad)

        # Scale MAD for Gaussian consistency
        mad_scaled = mad * 1.4826

        # Compute threshold
        outlier_threshold = self.threshold_k * mad_scaled

        # Center trace processing
        magnitudes = np.abs(stft_center)
        phases = np.angle(stft_center)
        deviations = np.abs(magnitudes - median_amp)

        # Apply thresholding
        if self.threshold_mode == 'hard':
            # Hard threshold: outliers snap to median
            new_magnitudes = np.where(
                deviations > outlier_threshold,
                median_amp,
                magnitudes
            )
        else:
            # Soft threshold: shrink toward median
            signs = np.where(magnitudes >= median_amp, 1, -1)
            new_deviations = np.maximum(deviations - outlier_threshold, 0)
            new_magnitudes = np.maximum(median_amp + signs * new_deviations, 0)

        # Apply frequency filtering
        freq_hz = freqs * sample_rate
        freq_mask = (freq_hz >= self.fmin) & (freq_hz <= self.fmax)
        new_magnitudes = np.where(
            freq_mask[:, np.newaxis],
            new_magnitudes,
            magnitudes  # Keep original outside frequency range
        )

        # Low-amplitude protection
        if self.low_amp_protection:
            low_amp_threshold = median_amp * self.low_amp_factor
            low_amp_mask = magnitudes < low_amp_threshold
            inflation_mask = low_amp_mask & (new_magnitudes > magnitudes)
            new_magnitudes = np.where(inflation_mask, magnitudes, new_magnitudes)

        # Reconstruct
        stft_denoised = new_magnitudes * np.exp(1j * phases)

        # Inverse Gabor Transform
        _, denoised_trace = signal.istft(
            stft_denoised,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )

        # Handle length mismatch
        if len(denoised_trace) < n_samples:
            denoised_trace = np.pad(denoised_trace, (0, n_samples - len(denoised_trace)))
        elif len(denoised_trace) > n_samples:
            denoised_trace = denoised_trace[:n_samples]

        return denoised_trace
