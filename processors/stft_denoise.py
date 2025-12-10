"""
STFT-Denoise processor using Short-Time Fourier Transform with MAD-based thresholding.

Implements time-frequency domain random noise attenuation using STFT
with spatial aperture processing and robust thresholding.

Advantages:
- Fast computation due to FFT efficiency
- Well-understood transform with perfect reconstruction
- Good for signals with relatively stationary frequency content
- Adjustable time-frequency resolution via window size

Best suited for:
- Broadband noise removal
- Signals with slowly varying frequency content
- When computational speed is important
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


class STFTDenoise(BaseProcessor):
    """
    Short-Time Fourier Transform Denoising with MAD thresholding.

    Uses spatial aperture processing with robust noise characterization
    for effective random noise attenuation while preserving signal.

    Supports multiple threshold modes:
    - 'soft': Classical soft thresholding (partial removal)
    - 'hard': Full removal for outliers, preserve non-outliers exactly
    - 'scaled': Progressive removal based on outlier severity
    - 'adaptive': Combined hard (severe) + scaled (moderate) - recommended
    """

    def __init__(self,
                 aperture: int = 7,
                 fmin: float = 5.0,
                 fmax: float = 100.0,
                 nperseg: int = 64,
                 noverlap: Optional[int] = None,
                 threshold_k: float = 3.0,
                 threshold_mode: Literal['soft', 'hard', 'scaled', 'adaptive'] = 'adaptive',
                 time_smoothing: int = 1,
                 low_amp_protection: bool = True,
                 low_amp_factor: float = 0.3):
        """
        Initialize STFT-Denoise processor.

        Args:
            aperture: Spatial aperture size (number of traces, must be odd)
            fmin: Minimum frequency (Hz) for processing
            fmax: Maximum frequency (Hz) for processing
            nperseg: Length of each STFT segment (window size)
            noverlap: Number of points to overlap between segments (default: nperseg//2)
            threshold_k: MAD threshold multiplier
            threshold_mode: Noise removal mode (recommended: 'adaptive'):
                - 'soft': Classical soft thresholding (partial removal)
                - 'hard': Full removal for outliers
                - 'scaled': Progressive removal based on severity
                - 'adaptive': Hard for severe + scaled for moderate (recommended)
            time_smoothing: Time window size for MAD smoothing (1=no smoothing,
                           >1 averages MAD over neighboring time bins)
            low_amp_protection: Prevent inflation of low-amplitude samples
            low_amp_factor: Threshold for low-amplitude protection (fraction of median)
        """
        self.aperture = aperture
        self.fmin = fmin
        self.fmax = fmax
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode
        self.time_smoothing = time_smoothing
        self.low_amp_protection = low_amp_protection
        self.low_amp_factor = low_amp_factor

        super().__init__(
            aperture=aperture,
            fmin=fmin,
            fmax=fmax,
            nperseg=nperseg,
            noverlap=self.noverlap,
            threshold_k=threshold_k,
            threshold_mode=threshold_mode,
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
        if self.nperseg < 8:
            raise ValueError("nperseg must be at least 8")
        if self.noverlap >= self.nperseg:
            raise ValueError("noverlap must be less than nperseg")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be positive")
        if self.threshold_mode not in ['soft', 'hard', 'scaled', 'adaptive']:
            raise ValueError("threshold_mode must be 'soft', 'hard', 'scaled', or 'adaptive'")
        if self.time_smoothing < 1:
            raise ValueError("time_smoothing must be at least 1")
        if self.low_amp_factor <= 0 or self.low_amp_factor >= 1:
            raise ValueError("low_amp_factor must be between 0 and 1")

    def get_description(self) -> str:
        """Get processor description."""
        mode_str = f", mode={self.threshold_mode}"
        if self.low_amp_protection:
            mode_str += ", low_amp_protect"
        return (f"STFT-Denoise: "
                f"aperture={self.aperture}, "
                f"freq={self.fmin:.0f}-{self.fmax:.0f}Hz, "
                f"nperseg={self.nperseg}, "
                f"k={self.threshold_k:.1f}{mode_str}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply STFT-domain denoising to seismic data.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time_total = time.time()

        traces = data.traces.copy()
        n_samples, n_traces = traces.shape

        # Log gather summary
        parallel_info = f"Parallel({N_JOBS} cores)" if JOBLIB_AVAILABLE else "Sequential"
        logger.info(
            f"STFT-Denoise: {n_traces} traces x {n_samples} samples | "
            f"Aperture: {self.aperture} | nperseg={self.nperseg} | "
            f"Freq: {self.fmin:.0f}-{self.fmax:.0f}Hz | "
            f"k={self.threshold_k}, mode={self.threshold_mode} | "
            f"{parallel_info}"
        )

        # Validate aperture
        if n_traces < self.aperture:
            logger.warning(
                f"Not enough traces ({n_traces}) for aperture ({self.aperture}), "
                f"using {n_traces if n_traces % 2 == 1 else n_traces - 1}"
            )
            effective_aperture = n_traces if n_traces % 2 == 1 else n_traces - 1
        else:
            effective_aperture = self.aperture

        # Get sample rate for frequency conversion
        sample_rate = 2.0 * data.nyquist_freq

        half_aperture = effective_aperture // 2

        # Check if parallel processing is beneficial
        use_parallel = JOBLIB_AVAILABLE and n_traces > 50

        if use_parallel:
            def process_single_trace(trace_idx):
                start_idx = max(0, trace_idx - half_aperture)
                end_idx = min(n_traces, trace_idx + half_aperture + 1)
                ensemble = traces[:, start_idx:end_idx]
                center_in_ensemble = trace_idx - start_idx
                return self._process_ensemble(ensemble, center_in_ensemble, sample_rate)

            results = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(process_single_trace)(i) for i in range(n_traces)
            )
            denoised_traces = np.column_stack(results)
        else:
            denoised_traces = np.zeros_like(traces)
            for trace_idx in range(n_traces):
                start_idx = max(0, trace_idx - half_aperture)
                end_idx = min(n_traces, trace_idx + half_aperture + 1)
                ensemble = traces[:, start_idx:end_idx]
                center_in_ensemble = trace_idx - start_idx
                denoised_traces[:, trace_idx] = self._process_ensemble(
                    ensemble, center_in_ensemble, sample_rate
                )

        # Compute timing and energy metrics
        elapsed_total = time.time() - start_time_total
        throughput = n_traces / elapsed_total if elapsed_total > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"STFT-Denoise complete: {elapsed_total:.2f}s | "
            f"{throughput:.1f} traces/s | "
            f"Energy: {energy_ratio:.1%} retained"
        )

        if energy_ratio < 0.10:
            logger.warning(f"Output <10% of input energy - threshold may be too aggressive")
        elif energy_ratio > 0.95:
            logger.warning(f"Output >95% of input energy - minimal denoising occurred")

        return SeismicData(
            traces=denoised_traces,
            sample_rate=data.sample_rate,
            metadata={
                **data.metadata,
                'processor': self.get_description()
            }
        )

    def _process_ensemble(self, ensemble, center_idx, sample_rate):
        """
        Process ensemble using STFT with fully vectorized MAD thresholding.

        Args:
            ensemble: Spatial aperture traces (n_samples, n_traces)
            center_idx: Index of center trace in ensemble
            sample_rate: Sample rate in Hz

        Returns:
            Denoised center trace (n_samples,)
        """
        n_samples, n_traces = ensemble.shape
        center_trace = ensemble[:, center_idx]

        # Adapt nperseg to signal length
        nperseg = min(self.nperseg, n_samples // 4)
        if nperseg < 8:
            return center_trace
        noverlap = min(self.noverlap, nperseg - 1)

        # Batch STFT: compute all traces at once
        freqs, times, stft_batch = signal.stft(
            ensemble.T,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=-1
        )
        # stft_batch shape: (n_traces, n_freqs, n_times)
        n_freqs, n_times = stft_batch.shape[1], stft_batch.shape[2]

        if n_freqs == 0 or n_times == 0:
            return center_trace

        # Get center trace STFT
        stft_center = stft_batch[center_idx]

        # Compute spatial statistics across trace dimension
        all_amplitudes = np.abs(stft_batch)
        median_amp = np.median(all_amplitudes, axis=0)
        mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)

        # Apply temporal smoothing
        if self.time_smoothing > 1 and n_times >= self.time_smoothing:
            mad = uniform_filter1d(mad, size=self.time_smoothing, axis=1, mode='nearest')
            median_amp = uniform_filter1d(median_amp, size=self.time_smoothing, axis=1, mode='nearest')

        # Prevent MAD=0
        min_mad = np.maximum(0.01 * median_amp, 1e-10)
        mad = np.maximum(mad, min_mad)

        mad_scaled = mad * 1.4826
        outlier_threshold = self.threshold_k * mad_scaled

        # Center trace coefficients
        magnitudes = np.abs(stft_center)
        phases = np.angle(stft_center)
        deviations = np.abs(magnitudes - median_amp)

        # Apply thresholding
        new_magnitudes = self._apply_threshold_mode(
            magnitudes, median_amp, deviations, outlier_threshold
        )

        # Apply frequency filtering
        if sample_rate is not None:
            freq_hz = freqs * sample_rate
            freq_mask = (freq_hz >= self.fmin) & (freq_hz <= self.fmax)
            new_magnitudes = np.where(
                freq_mask[:, np.newaxis],
                new_magnitudes,
                magnitudes
            )

        # Reconstruct
        stft_denoised = new_magnitudes * np.exp(1j * phases)

        _, denoised_trace = signal.istft(
            stft_denoised,
            nperseg=nperseg,
            noverlap=noverlap
        )

        # Handle length mismatch
        if len(denoised_trace) < n_samples:
            denoised_trace = np.pad(denoised_trace, (0, n_samples - len(denoised_trace)))
        elif len(denoised_trace) > n_samples:
            denoised_trace = denoised_trace[:n_samples]

        return denoised_trace

    def _apply_threshold_mode(self, magnitudes, median_amp, deviations, outlier_threshold):
        """
        Apply thresholding based on the configured threshold_mode.

        Args:
            magnitudes: Original magnitudes
            median_amp: Spatial median
            deviations: Deviation from median
            outlier_threshold: k * MAD threshold

        Returns:
            new_magnitudes: Thresholded magnitudes
        """
        if self.threshold_mode == 'hard':
            new_magnitudes = np.where(
                deviations > outlier_threshold,
                median_amp,
                magnitudes
            )

        elif self.threshold_mode == 'scaled':
            outlier_ratio = deviations / (outlier_threshold + 1e-10)
            excess_ratio = np.maximum(outlier_ratio - 1.0, 0.0)
            removal_factor = 1.0 - 1.0 / (1.0 + excess_ratio ** 2)

            new_deviations = deviations * (1.0 - removal_factor)
            signs = np.where(magnitudes >= median_amp, 1, -1)

            new_magnitudes = np.where(
                outlier_ratio > 1.0,
                np.maximum(median_amp + signs * new_deviations, 0),
                magnitudes
            )

        elif self.threshold_mode == 'adaptive':
            outlier_ratio = deviations / (outlier_threshold + 1e-10)
            severe_threshold = 2.0

            moderate_removal = np.clip(
                (outlier_ratio - 1.0) / (severe_threshold - 1.0), 0.0, 1.0
            )
            signs = np.where(magnitudes >= median_amp, 1, -1)
            moderate_new_deviations = deviations * (1.0 - moderate_removal)
            moderate_magnitude = np.maximum(
                median_amp + signs * moderate_new_deviations, 0
            )

            new_magnitudes = np.where(
                outlier_ratio > severe_threshold,
                median_amp,
                np.where(
                    outlier_ratio > 1.0,
                    moderate_magnitude,
                    magnitudes
                )
            )

        else:  # 'soft'
            new_deviations = np.maximum(deviations - outlier_threshold, 0)
            signs = np.where(magnitudes >= median_amp, 1, -1)
            new_magnitudes = np.maximum(median_amp + signs * new_deviations, 0)

        # Apply low-amplitude protection
        if self.low_amp_protection:
            low_amp_threshold = median_amp * self.low_amp_factor
            low_amp_mask = magnitudes < low_amp_threshold
            inflation_mask = low_amp_mask & (new_magnitudes > magnitudes)
            new_magnitudes = np.where(inflation_mask, magnitudes, new_magnitudes)

        return new_magnitudes
