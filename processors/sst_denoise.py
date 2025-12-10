"""
Synchrosqueezing Transform (SST) Denoising processor.

Implements time-frequency reassignment for high-resolution analysis
with denoising in the synchrosqueezed domain.

Key advantages:
- Near-perfect instantaneous frequency estimation
- Sharper time-frequency localization than STFT
- Invertible (unlike some reassignment methods)
- Excellent for mode separation

Best suited for:
- Instantaneous frequency/phase analysis
- Thin layer detection
- High-resolution spectral decomposition
- Time-varying frequency analysis

Enhanced with spatial aperture processing for robust noise estimation
across neighboring traces (coherent signal vs incoherent noise separation).
"""
import numpy as np
from typing import Optional, Literal
import logging
from scipy.ndimage import uniform_filter1d

from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# Try to import ssqueezepy
try:
    from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft, issq_stft
    from ssqueezepy.wavelets import Wavelet
    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    SSQUEEZEPY_AVAILABLE = False
    logger.warning("ssqueezepy not available. Install with: pip install ssqueezepy")

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_JOBS = max(1, multiprocessing.cpu_count() - 1)
except ImportError:
    JOBLIB_AVAILABLE = False
    N_JOBS = 1


class SSTDenoise(BaseProcessor):
    """
    Synchrosqueezing Transform Denoising.

    Uses synchrosqueezing to achieve high time-frequency resolution,
    then applies thresholding in the synchrosqueezed domain for denoising.

    Supports both CWT-based (continuous wavelet) and STFT-based
    synchrosqueezing transforms.

    Enhanced with spatial aperture processing for robust noise estimation
    across neighboring traces (like STFT/Stockwell denoisers).
    """

    def __init__(self,
                 base_transform: Literal['cwt', 'stft'] = 'cwt',
                 wavelet: str = 'morlet',
                 nv: int = 32,
                 threshold_k: float = 3.0,
                 threshold_mode: Literal['soft', 'hard', 'scaled', 'adaptive'] = 'soft',
                 squeezing: bool = True,
                 aperture: int = 1,
                 time_smoothing: int = 1,
                 low_amp_protection: bool = True,
                 low_amp_factor: float = 0.3):
        """
        Initialize SST-Denoise processor.

        Args:
            base_transform: Base transform type:
                - 'cwt': CWT-based SST (better time-frequency resolution)
                - 'stft': STFT-based SST (faster)
            wavelet: Wavelet for CWT mode ('morlet', 'bump', 'cmhat')
            nv: Number of voices per octave (frequency resolution)
            threshold_k: MAD threshold multiplier for denoising
            threshold_mode: Thresholding type:
                - 'soft': Classical soft thresholding
                - 'hard': Full removal for outliers
                - 'scaled': Progressive removal based on severity
                - 'adaptive': Hard for severe + scaled for moderate (recommended)
            squeezing: Whether to apply synchrosqueezing (False = standard CWT/STFT)
            aperture: Spatial aperture size (number of traces, must be odd, 1=single trace)
            time_smoothing: Time window for MAD smoothing (1=no smoothing)
            low_amp_protection: Prevent inflation of low-amplitude samples
            low_amp_factor: Threshold for low-amplitude protection (fraction of median)
        """
        if not SSQUEEZEPY_AVAILABLE:
            raise ImportError("ssqueezepy required. Install with: pip install ssqueezepy")

        self.base_transform = base_transform
        self.wavelet = wavelet
        self.nv = nv
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode
        self.squeezing = squeezing
        self.aperture = aperture
        self.time_smoothing = time_smoothing
        self.low_amp_protection = low_amp_protection
        self.low_amp_factor = low_amp_factor

        super().__init__(
            base_transform=base_transform,
            wavelet=wavelet,
            nv=nv,
            threshold_k=threshold_k,
            threshold_mode=threshold_mode,
            squeezing=squeezing,
            aperture=aperture,
            time_smoothing=time_smoothing,
            low_amp_protection=low_amp_protection,
            low_amp_factor=low_amp_factor
        )

    def _validate_params(self):
        """Validate processor parameters."""
        if self.base_transform not in ['cwt', 'stft']:
            raise ValueError("base_transform must be 'cwt' or 'stft'")
        if self.wavelet not in ['morlet', 'bump', 'cmhat', 'gmw']:
            raise ValueError("wavelet must be 'morlet', 'bump', 'cmhat', or 'gmw'")
        if self.nv < 4:
            raise ValueError("nv must be at least 4")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be positive")
        if self.threshold_mode not in ['soft', 'hard', 'scaled', 'adaptive']:
            raise ValueError("threshold_mode must be 'soft', 'hard', 'scaled', or 'adaptive'")
        if self.aperture > 1:
            if self.aperture < 3:
                raise ValueError("Aperture must be 1 (single trace) or at least 3")
            if self.aperture % 2 == 0:
                raise ValueError("Aperture must be odd")
        if self.time_smoothing < 1:
            raise ValueError("time_smoothing must be at least 1")
        if self.low_amp_factor <= 0 or self.low_amp_factor >= 1:
            raise ValueError("low_amp_factor must be between 0 and 1")

    def get_description(self) -> str:
        """Get processor description."""
        squeeze_str = "SST" if self.squeezing else self.base_transform.upper()
        aperture_str = f"aperture={self.aperture}, " if self.aperture > 1 else ""
        mode_str = f", mode={self.threshold_mode}"
        if self.low_amp_protection:
            mode_str += ", low_amp_protect"
        return (f"{squeeze_str}-Denoise ({self.base_transform.upper()}): "
                f"{aperture_str}wavelet={self.wavelet}, nv={self.nv}, "
                f"k={self.threshold_k:.1f}{mode_str}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply SST-domain denoising to seismic data.

        When aperture > 1, uses spatial statistics across neighboring traces
        for robust noise estimation (coherent signal vs incoherent noise).

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time = time.time()

        traces = data.traces.copy()
        n_samples, n_traces = traces.shape

        # Determine effective aperture
        if n_traces < self.aperture:
            effective_aperture = n_traces if n_traces % 2 == 1 else n_traces - 1
            if effective_aperture < 1:
                effective_aperture = 1
            if effective_aperture != self.aperture:
                logger.warning(
                    f"Not enough traces ({n_traces}) for aperture ({self.aperture}), "
                    f"using {effective_aperture}"
                )
        else:
            effective_aperture = self.aperture

        # Log processing info
        use_spatial = effective_aperture > 1
        parallel_info = f"Parallel({N_JOBS} cores)" if JOBLIB_AVAILABLE and n_traces > 10 else "Sequential"
        spatial_info = f"aperture={effective_aperture}" if use_spatial else "single-trace"
        logger.info(
            f"SST-Denoise ({self.base_transform.upper()}): {n_traces} traces × {n_samples} samples | "
            f"{spatial_info} | nv={self.nv}, k={self.threshold_k}, mode={self.threshold_mode} | {parallel_info}"
        )

        # Process based on mode
        use_parallel = JOBLIB_AVAILABLE and n_traces > 10

        if use_spatial:
            # Spatial aperture processing
            half_aperture = effective_aperture // 2
            if use_parallel:
                def process_single_trace(trace_idx):
                    start_idx = max(0, trace_idx - half_aperture)
                    end_idx = min(n_traces, trace_idx + half_aperture + 1)
                    ensemble = traces[:, start_idx:end_idx]
                    center_in_ensemble = trace_idx - start_idx
                    return self._process_ensemble(ensemble, center_in_ensemble)

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
                        ensemble, center_in_ensemble
                    )
        else:
            # Single-trace processing (original behavior)
            if use_parallel:
                results = Parallel(n_jobs=N_JOBS, prefer="threads")(
                    delayed(self._process_trace)(traces[:, i])
                    for i in range(n_traces)
                )
                denoised_traces = np.column_stack(results)
            else:
                denoised_traces = np.zeros_like(traces)
                for i in range(n_traces):
                    denoised_traces[:, i] = self._process_trace(traces[:, i])

        # Compute metrics
        elapsed = time.time() - start_time
        throughput = n_traces / elapsed if elapsed > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"SST-Denoise complete: {elapsed:.2f}s | "
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
            headers=data.headers,
            metadata={
                **data.metadata,
                'processor': self.get_description()
            }
        )

    def _process_trace(self, trace: np.ndarray) -> np.ndarray:
        """
        Process a single trace using SST.

        Args:
            trace: 1D signal array

        Returns:
            Denoised trace
        """
        try:
            n_samples = len(trace)

            # Apply forward transform
            if self.base_transform == 'cwt':
                if self.squeezing:
                    # Synchrosqueezed CWT
                    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(
                        trace,
                        wavelet=self.wavelet,
                        nv=self.nv
                    )
                    tf_matrix = Tx
                else:
                    # Standard CWT
                    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(
                        trace,
                        wavelet=self.wavelet,
                        nv=self.nv
                    )
                    tf_matrix = Wx
            else:  # stft
                if self.squeezing:
                    Tx, Wx, ssq_freqs, Sfs, *_ = ssq_stft(trace)
                    tf_matrix = Tx
                else:
                    Tx, Wx, ssq_freqs, Sfs, *_ = ssq_stft(trace)
                    tf_matrix = Wx

            # Apply thresholding
            tf_denoised = self._apply_threshold(tf_matrix)

            # Inverse transform
            if self.base_transform == 'cwt':
                if self.squeezing:
                    denoised = issq_cwt(tf_denoised, self.wavelet)
                else:
                    # For non-squeezed CWT, use the standard inverse
                    denoised = issq_cwt(tf_denoised, self.wavelet)
            else:
                if self.squeezing:
                    denoised = issq_stft(tf_denoised)
                else:
                    denoised = issq_stft(tf_denoised)

            # Handle complex output
            if np.iscomplexobj(denoised):
                denoised = np.real(denoised)

            # Handle length mismatch
            if len(denoised) < n_samples:
                denoised = np.pad(denoised, (0, n_samples - len(denoised)))
            elif len(denoised) > n_samples:
                denoised = denoised[:n_samples]

            return denoised

        except Exception as e:
            logger.warning(f"SST failed for trace: {e}. Returning original.")
            return trace

    def _process_ensemble(self, ensemble: np.ndarray, center_idx: int) -> np.ndarray:
        """
        Process ensemble of traces using spatial SST statistics.

        Computes SST for all traces in the ensemble and uses spatial statistics
        across traces for robust noise estimation.

        Args:
            ensemble: Spatial aperture traces (n_samples, n_traces)
            center_idx: Index of center trace in ensemble

        Returns:
            Denoised center trace (n_samples,)
        """
        try:
            n_samples, n_ensemble_traces = ensemble.shape
            center_trace = ensemble[:, center_idx]

            # Compute SST for all traces in ensemble
            tf_matrices = []
            for i in range(n_ensemble_traces):
                trace = ensemble[:, i]
                if self.base_transform == 'cwt':
                    if self.squeezing:
                        Tx, Wx, *_ = ssq_cwt(trace, wavelet=self.wavelet, nv=self.nv)
                        tf_matrices.append(Tx)
                    else:
                        Tx, Wx, *_ = ssq_cwt(trace, wavelet=self.wavelet, nv=self.nv)
                        tf_matrices.append(Wx)
                else:  # stft
                    if self.squeezing:
                        Tx, Wx, *_ = ssq_stft(trace)
                        tf_matrices.append(Tx)
                    else:
                        Tx, Wx, *_ = ssq_stft(trace)
                        tf_matrices.append(Wx)

            # Stack TF matrices: shape (n_traces, n_freqs, n_times)
            tf_stack = np.stack(tf_matrices, axis=0)
            center_tf = tf_stack[center_idx]

            # Compute spatial statistics across traces
            all_amplitudes = np.abs(tf_stack)
            median_amp = np.median(all_amplitudes, axis=0)  # shape (n_freqs, n_times)
            mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)

            # Apply temporal smoothing
            if self.time_smoothing > 1 and mad.shape[-1] >= self.time_smoothing:
                mad = uniform_filter1d(mad, size=self.time_smoothing, axis=-1, mode='nearest')
                median_amp = uniform_filter1d(median_amp, size=self.time_smoothing, axis=-1, mode='nearest')

            # Prevent MAD=0
            min_mad = np.maximum(0.01 * median_amp, 1e-10)
            mad = np.maximum(mad, min_mad)

            mad_scaled = mad * 1.4826
            outlier_threshold = self.threshold_k * mad_scaled

            # Get center trace coefficients
            magnitudes = np.abs(center_tf)
            phases = np.angle(center_tf)
            deviations = np.abs(magnitudes - median_amp)

            # Apply spatial thresholding
            new_magnitudes = self._apply_threshold_mode(
                magnitudes, median_amp, deviations, outlier_threshold
            )

            # Reconstruct
            tf_denoised = new_magnitudes * np.exp(1j * phases)

            # Inverse transform
            if self.base_transform == 'cwt':
                denoised = issq_cwt(tf_denoised, self.wavelet)
            else:
                denoised = issq_stft(tf_denoised)

            # Handle complex output
            if np.iscomplexobj(denoised):
                denoised = np.real(denoised)

            # Handle length mismatch
            if len(denoised) < n_samples:
                denoised = np.pad(denoised, (0, n_samples - len(denoised)))
            elif len(denoised) > n_samples:
                denoised = denoised[:n_samples]

            return denoised

        except Exception as e:
            logger.warning(f"SST ensemble processing failed: {e}. Returning original trace.")
            return ensemble[:, center_idx]

    def _apply_threshold_mode(self, magnitudes, median_amp, deviations, outlier_threshold):
        """
        Apply thresholding based on the configured threshold_mode.

        This is the spatial version that uses median_amp and deviations computed
        across the spatial aperture.

        Args:
            magnitudes: Original magnitudes (n_freqs, n_times)
            median_amp: Spatial median (n_freqs, n_times)
            deviations: Deviation from median (n_freqs, n_times)
            outlier_threshold: k * MAD threshold (n_freqs, n_times)

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

    def _apply_threshold(self, tf_matrix: np.ndarray) -> np.ndarray:
        """
        Apply MAD-based thresholding to time-frequency matrix (single-trace mode).

        Args:
            tf_matrix: Complex time-frequency representation

        Returns:
            Thresholded matrix
        """
        magnitudes = np.abs(tf_matrix)

        # Compute MAD-based threshold
        median_mag = np.median(magnitudes)
        mad = np.median(np.abs(magnitudes - median_mag))
        mad_scaled = mad * 1.4826  # Scale for Gaussian consistency

        # Prevent zero threshold
        threshold = self.threshold_k * max(mad_scaled, 1e-10)
        deviations = np.abs(magnitudes - median_mag)

        # For single-trace mode, use the same threshold everywhere
        outlier_threshold = np.full_like(magnitudes, threshold)
        median_amp = np.full_like(magnitudes, median_mag)

        new_magnitudes = self._apply_threshold_mode(
            magnitudes, median_amp, deviations, outlier_threshold
        )

        phases = np.angle(tf_matrix)
        return new_magnitudes * np.exp(1j * phases)
