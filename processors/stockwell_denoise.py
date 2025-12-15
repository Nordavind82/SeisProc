"""
Stockwell Transform (S-Transform) Denoise processor with MAD-based thresholding.

Implements time-frequency domain random noise attenuation using S-Transform
with spatial aperture processing and robust thresholding.

Key advantages:
- Frequency-dependent resolution (better than fixed-window STFT)
- Linear phase response (no phase distortion)
- Direct time-frequency localization
- Theoretically perfect reconstruction

Best suited for:
- Non-stationary signals with varying frequency content
- Signals requiring frequency-adaptive resolution
- High-precision time-frequency analysis
- When phase preservation is critical

Optimized with Numba JIT compilation and window caching.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Optional, Literal
import logging

from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# Try to import numba for JIT acceleration
try:
    from numba import jit, prange
    from numba.core.errors import NumbaError, TypingError, UnsupportedError
    NUMBA_AVAILABLE = True
    NUMBA_JIT_FAILED = False
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_JIT_FAILED = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_JOBS = max(1, multiprocessing.cpu_count() - 1)
except ImportError:
    JOBLIB_AVAILABLE = False
    N_JOBS = 1


# LRU cache for S-Transform Gaussian windows
_STRANSFORM_WINDOW_CACHE = {}
_STRANSFORM_CACHE_MAX_SIZE = 5


def _get_cached_windows(n: int, fmin: float, fmax: float, positive_freqs: np.ndarray):
    """
    Get cached Gaussian windows or compute and cache them.

    Args:
        n: Number of samples
        fmin: Minimum frequency (normalized, 0-0.5)
        fmax: Maximum frequency (normalized, 0-0.5)
        positive_freqs: Array of positive frequencies

    Returns:
        Tuple of (windows, freq_indices, output_freqs)
    """
    global _STRANSFORM_WINDOW_CACHE, NUMBA_JIT_FAILED

    cache_key = (n, round(fmin * 1000), round(fmax * 1000))

    if cache_key in _STRANSFORM_WINDOW_CACHE:
        logger.debug(f"S-Transform window cache HIT: n={n}, fmin={fmin:.3f}, fmax={fmax:.3f}")
        return _STRANSFORM_WINDOW_CACHE[cache_key]

    logger.debug(f"S-Transform window cache MISS: n={n}, fmin={fmin:.3f}, fmax={fmax:.3f}")

    # Calculate frequency indices
    if fmin is not None or fmax is not None:
        fmin_norm = fmin if fmin is not None else 0
        fmax_norm = fmax if fmax is not None else 0.5
        freq_mask = (positive_freqs >= fmin_norm) & (positive_freqs <= fmax_norm)
        freq_indices = np.where(freq_mask)[0]
    else:
        freq_indices = np.arange(1, n//2 + 1)

    output_freqs = positive_freqs[freq_indices]

    # Compute windows
    use_numba = NUMBA_AVAILABLE and not NUMBA_JIT_FAILED and len(freq_indices) > 10

    if use_numba:
        try:
            windows = _compute_gaussian_windows_numba(freq_indices, output_freqs, n)
        except Exception as e:
            if NUMBA_AVAILABLE:
                try:
                    from numba.core.errors import NumbaError
                    if isinstance(e, (NumbaError, TypeError, RuntimeError)):
                        NUMBA_JIT_FAILED = True
                        logger.warning(f"Numba JIT compilation failed: {e}")
                except ImportError:
                    pass
            use_numba = False

    if not use_numba:
        # Fallback to pure NumPy (float32 for memory efficiency)
        n_freqs = len(freq_indices)
        windows = np.zeros((n_freqs, n), dtype=np.float32)
        freq_range = np.arange(n, dtype=np.float32)

        for i, k in enumerate(freq_indices):
            f = output_freqs[i]
            if f == 0:
                windows[i, :] = 1.0 / n
            else:
                sigma_f = np.abs(f) / (2 * np.sqrt(2 * np.log(2)))
                freq_diff = np.where(freq_range <= n//2, freq_range - k, freq_range - k - n)
                windows[i, :] = np.exp(-2 * np.pi**2 * sigma_f**2 * freq_diff**2)

    # Cache the result
    if len(_STRANSFORM_WINDOW_CACHE) >= _STRANSFORM_CACHE_MAX_SIZE:
        oldest_key = next(iter(_STRANSFORM_WINDOW_CACHE))
        del _STRANSFORM_WINDOW_CACHE[oldest_key]

    _STRANSFORM_WINDOW_CACHE[cache_key] = (windows, freq_indices, output_freqs)
    return windows, freq_indices, output_freqs


@jit(nopython=True, parallel=False, cache=True)
def _compute_gaussian_windows_numba(freq_indices, positive_freqs, n):
    """
    Numba-optimized Gaussian window computation.
    Uses float32 for 50% memory savings.
    """
    n_freqs = len(freq_indices)
    windows = np.zeros((n_freqs, n), dtype=np.float32)
    freq_range = np.arange(n, dtype=np.float32)

    for i in range(n_freqs):
        k = freq_indices[i]
        f = positive_freqs[i]

        if f == 0:
            windows[i, :] = 1.0 / n
            continue

        sigma_f = np.abs(f) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        freq_diff = np.where(freq_range <= n//2, freq_range - k, freq_range - k - n)
        windows[i, :] = np.exp(-2.0 * np.pi**2 * sigma_f**2 * freq_diff**2)

    return windows


def stockwell_transform(data, fmin=None, fmax=None):
    """
    Compute S-Transform (Stockwell Transform) of a 1D signal.

    Uses cached Gaussian windows for improved performance.

    Args:
        data: 1D array, input signal
        fmin: Minimum frequency (normalized, 0-0.5)
        fmax: Maximum frequency (normalized, 0-0.5)

    Returns:
        S: 2D complex array (frequency x time), S-Transform
        freqs: 1D array, frequency values
    """
    n = len(data)
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(n)
    positive_freqs = freqs[:n//2 + 1]

    fmin_norm = fmin if fmin is not None else 0.0
    fmax_norm = fmax if fmax is not None else 0.5

    windows, freq_indices, output_freqs = _get_cached_windows(
        n, fmin_norm, fmax_norm, positive_freqs
    )

    n_freqs = len(freq_indices)
    S = np.zeros((n_freqs, n), dtype=np.complex64)

    freq_range = np.arange(n)
    for i, k in enumerate(freq_indices):
        if output_freqs[i] == 0:
            S[i, :] = np.mean(data)
        else:
            windowed_fft = fft_data * windows[i, :]
            S[i, :] = np.fft.ifft(windowed_fft * np.exp(2j * np.pi * k * freq_range / n))

    return S, output_freqs


def inverse_stockwell_transform(S, n_samples, freq_values=None):
    """
    Compute inverse S-Transform.

    Args:
        S: 2D complex array (frequency x time), S-Transform
        n_samples: Length of output signal
        freq_values: Array of frequency values (optional)

    Returns:
        data: 1D array, reconstructed signal
    """
    if S.shape[0] == 0:
        return np.zeros(n_samples)

    n_freqs, n_times = S.shape

    # Sum across frequencies
    time_series = np.sum(S, axis=0)
    reconstructed = time_series.real

    # Adaptive normalization based on frequency coverage
    max_possible_freqs = n_samples // 2
    freq_coverage = n_freqs / max_possible_freqs

    if freq_coverage > 0.95:
        normalization = np.sqrt(n_samples / 100.0)
    else:
        normalization = 1.0

    reconstructed = reconstructed / normalization
    return reconstructed[:n_samples]


class StockwellDenoise(BaseProcessor):
    """
    Stockwell Transform (S-Transform) Denoising with MAD thresholding.

    Uses frequency-dependent Gaussian windows for optimal time-frequency
    resolution. Spatial aperture processing with robust noise characterization.

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
                 threshold_k: float = 3.0,
                 threshold_mode: Literal['soft', 'hard', 'scaled', 'adaptive'] = 'adaptive',
                 time_smoothing: int = 1,
                 low_amp_protection: bool = True,
                 low_amp_factor: float = 0.3):
        """
        Initialize Stockwell-Denoise processor.

        Args:
            aperture: Spatial aperture size (number of traces, must be odd)
            fmin: Minimum frequency (Hz) for processing
            fmax: Maximum frequency (Hz) for processing
            threshold_k: MAD threshold multiplier
            threshold_mode: Noise removal mode (recommended: 'adaptive'):
                - 'soft': Classical soft thresholding (partial removal)
                - 'hard': Full removal for outliers
                - 'scaled': Progressive removal based on severity
                - 'adaptive': Hard for severe + scaled for moderate (recommended)
            time_smoothing: Time window size for MAD smoothing (1=no smoothing)
            low_amp_protection: Prevent inflation of low-amplitude samples
            low_amp_factor: Threshold for low-amplitude protection (fraction of median)
        """
        self.aperture = aperture
        self.fmin = fmin
        self.fmax = fmax
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode
        self.time_smoothing = time_smoothing
        self.low_amp_protection = low_amp_protection
        self.low_amp_factor = low_amp_factor

        super().__init__(
            aperture=aperture,
            fmin=fmin,
            fmax=fmax,
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
        return (f"Stockwell-Denoise: "
                f"aperture={self.aperture}, "
                f"freq={self.fmin:.0f}-{self.fmax:.0f}Hz, "
                f"k={self.threshold_k:.1f}{mode_str}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply S-Transform domain denoising to seismic data.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time_total = time.time()

        # Convert to float32 for memory efficiency (50% savings)
        traces = data.traces
        if traces.dtype != np.float32:
            traces = traces.astype(np.float32)
        else:
            traces = traces.copy()
        n_samples, n_traces = traces.shape

        # Log gather summary
        parallel_info = f"Parallel({N_JOBS} cores)" if JOBLIB_AVAILABLE else "Sequential"
        numba_status = "Numba" if NUMBA_AVAILABLE and not NUMBA_JIT_FAILED else "NumPy"
        logger.info(
            f"Stockwell-Denoise: {n_traces} traces x {n_samples} samples | "
            f"Aperture: {self.aperture} | "
            f"Freq: {self.fmin:.0f}-{self.fmax:.0f}Hz | "
            f"k={self.threshold_k}, mode={self.threshold_mode} | "
            f"{parallel_info} | {numba_status}"
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

        # Convert frequencies to normalized
        nyquist_freq = data.nyquist_freq
        sample_rate = 2.0 * nyquist_freq
        fmin_norm = max(0.0, min(self.fmin / sample_rate, 0.5))
        fmax_norm = max(0.0, min(self.fmax / sample_rate, 0.5))

        half_aperture = effective_aperture // 2

        # Check if parallel processing is beneficial
        use_parallel = JOBLIB_AVAILABLE and n_traces > 50

        if use_parallel:
            def process_single_trace(trace_idx):
                start_idx = max(0, trace_idx - half_aperture)
                end_idx = min(n_traces, trace_idx + half_aperture + 1)
                ensemble = traces[:, start_idx:end_idx]
                center_in_ensemble = trace_idx - start_idx
                return self._process_ensemble(ensemble, center_in_ensemble, fmin_norm, fmax_norm)

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
                    ensemble, center_in_ensemble, fmin_norm, fmax_norm
                )

        # Compute timing and energy metrics
        elapsed_total = time.time() - start_time_total
        throughput = n_traces / elapsed_total if elapsed_total > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"Stockwell-Denoise complete: {elapsed_total:.2f}s | "
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

    def _process_ensemble(self, ensemble, center_idx, fmin_norm, fmax_norm):
        """
        Process ensemble using S-Transform.

        Args:
            ensemble: Spatial aperture traces (n_samples, n_traces)
            center_idx: Index of center trace in ensemble
            fmin_norm: Minimum normalized frequency
            fmax_norm: Maximum normalized frequency

        Returns:
            Denoised center trace (n_samples,)
        """
        n_samples, n_traces = ensemble.shape
        center_trace = ensemble[:, center_idx]

        # Compute S-transform for all traces in ensemble
        st_ensemble = []
        freq_values = None
        for i in range(n_traces):
            st, freqs = stockwell_transform(ensemble[:, i], fmin=fmin_norm, fmax=fmax_norm)
            st_ensemble.append(st)
            if i == 0:
                freq_values = freqs

        if len(st_ensemble) == 0:
            return center_trace

        st_ensemble = np.array(st_ensemble)
        st_center = st_ensemble[center_idx]
        n_freqs, n_times = st_center.shape

        # Apply MAD thresholding
        st_denoised = np.zeros_like(st_center)

        for f in range(n_freqs):
            spatial_amplitudes = np.abs(st_ensemble[:, f, :])
            median_amp = np.median(spatial_amplitudes, axis=0)
            mad = np.median(np.abs(spatial_amplitudes - median_amp), axis=0)
            mad_scaled = mad * 1.4826
            outlier_threshold = self.threshold_k * mad_scaled

            coefs = st_center[f, :]
            magnitudes = np.abs(coefs)
            phases = np.angle(coefs)
            deviations = np.abs(magnitudes - median_amp)

            new_magnitudes = self._apply_threshold_mode(
                magnitudes, median_amp, deviations, outlier_threshold
            )

            st_denoised[f, :] = new_magnitudes * np.exp(1j * phases)

        # Inverse transform
        denoised_trace = inverse_stockwell_transform(st_denoised, n_samples, freq_values=freq_values)

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
