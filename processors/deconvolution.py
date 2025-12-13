"""
Deconvolution Processor - Spiking and Predictive Deconvolution

Implements the Wiener-Levinson algorithm as described in Yilmaz (2001).

Spiking Deconvolution:
- Compresses the seismic wavelet to a spike
- Whitens the amplitude spectrum
- Prediction distance = 1 sample

Predictive Deconvolution:
- Attenuates multiples with a known period
- Prediction distance = multiple period in samples

The design window follows hyperbolic moveout (NMO) with offset:
    t_window(x) = sqrt(t0^2 + (x/v)^2)

References:
    Yilmaz, O. (2001). Seismic Data Analysis. SEG.
    Chapter 2: Deconvolution
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from scipy.linalg import toeplitz, solve_toeplitz
import logging

from processors.base_processor import BaseProcessor
from models.seismic_data import SeismicData

logger = logging.getLogger(__name__)


@dataclass
class DeconConfig:
    """
    Configuration for deconvolution.

    The design window is defined with start time at zero offset plus linear velocity:
        T_top = time_top_ms + (offset / velocity_top) * 1000
        T_bottom = time_bottom_ms + (offset / velocity_bottom) * 1000

    Attributes:
        mode: 'spiking' or 'predictive'
        time_top_ms: Start time for top window at zero offset (ms)
        time_bottom_ms: Start time for bottom window at zero offset (ms)
        velocity_top: Velocity for top window moveout (m/s)
        velocity_bottom: Velocity for bottom window moveout (m/s)
        filter_length_ms: Length of the deconvolution operator (ms)
        white_noise_percent: Pre-whitening percentage (0.1-10, typical 1%)
        prediction_distance_ms: Prediction distance (ms).
                               For spiking, this equals sample interval.
                               For predictive, this equals multiple period.
    """
    mode: str = 'spiking'
    time_top_ms: float = 100.0        # Start time at zero offset for top window
    time_bottom_ms: float = 500.0     # Start time at zero offset for bottom window
    velocity_top: float = 3500.0      # Top window moveout velocity
    velocity_bottom: float = 1500.0   # Bottom window moveout velocity
    filter_length_ms: float = 160.0
    white_noise_percent: float = 1.0
    prediction_distance_ms: float = 0.0  # 0 = auto (sample interval for spiking)

    def __post_init__(self):
        if self.mode not in ['spiking', 'predictive']:
            raise ValueError(f"mode must be 'spiking' or 'predictive', got {self.mode}")
        if self.time_top_ms < 0:
            raise ValueError(f"time_top_ms must be >= 0, got {self.time_top_ms}")
        if self.time_bottom_ms <= self.time_top_ms:
            raise ValueError(
                f"time_bottom_ms ({self.time_bottom_ms}) must be > time_top_ms ({self.time_top_ms})"
            )
        if self.velocity_top <= 0:
            raise ValueError(f"velocity_top must be > 0, got {self.velocity_top}")
        if self.velocity_bottom <= 0:
            raise ValueError(f"velocity_bottom must be > 0, got {self.velocity_bottom}")
        if self.filter_length_ms <= 0:
            raise ValueError(f"filter_length_ms must be > 0, got {self.filter_length_ms}")
        if self.white_noise_percent < 0 or self.white_noise_percent > 100:
            raise ValueError(f"white_noise_percent must be in [0, 100], got {self.white_noise_percent}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            'mode': self.mode,
            'time_top_ms': self.time_top_ms,
            'time_bottom_ms': self.time_bottom_ms,
            'velocity_top': self.velocity_top,
            'velocity_bottom': self.velocity_bottom,
            'filter_length_ms': self.filter_length_ms,
            'white_noise_percent': self.white_noise_percent,
            'prediction_distance_ms': self.prediction_distance_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DeconConfig':
        """Deserialize config from dictionary."""
        return cls(
            mode=d.get('mode', 'spiking'),
            time_top_ms=d.get('time_top_ms', 100.0),
            time_bottom_ms=d.get('time_bottom_ms', 500.0),
            velocity_top=d.get('velocity_top', 3500.0),
            velocity_bottom=d.get('velocity_bottom', 1500.0),
            filter_length_ms=d.get('filter_length_ms', 160.0),
            white_noise_percent=d.get('white_noise_percent', 1.0),
            prediction_distance_ms=d.get('prediction_distance_ms', 0.0),
        )


class DeconvolutionProcessor(BaseProcessor):
    """
    Wiener-Levinson Deconvolution Processor.

    Implements spiking and predictive deconvolution with offset-dependent
    design windows that follow hyperbolic NMO moveout.

    The algorithm:
    1. Extract design window for each trace (NMO-corrected time gate)
    2. Compute autocorrelation of the windowed trace
    3. Solve Wiener-Hopf equation using Levinson recursion
    4. Convolve entire trace with the inverse filter

    Parameters:
        config: DeconConfig with deconvolution parameters

    Example:
        >>> config = DeconConfig(mode='spiking', filter_length_ms=160)
        >>> processor = DeconvolutionProcessor(config=config)
        >>> result = processor.process(data)
    """

    def __init__(self, config: Optional[DeconConfig] = None, **params):
        # Handle config from params if provided via BaseProcessor interface
        if config is None and 'config' in params:
            config_dict = params.pop('config')
            if isinstance(config_dict, dict):
                config = DeconConfig.from_dict(config_dict)
            else:
                config = config_dict

        self.config = config or DeconConfig()

        # Store for serialization
        params['config'] = self.config.to_dict()
        super().__init__(**params)

    def _validate_params(self):
        """Validate processor parameters."""
        # Config is validated in DeconConfig.__post_init__
        pass

    def get_description(self) -> str:
        """Get human-readable description."""
        mode_str = self.config.mode.capitalize()
        if self.config.mode == 'predictive':
            pred_str = f", pred={self.config.prediction_distance_ms:.0f}ms"
        else:
            pred_str = ""
        return (
            f"{mode_str} Decon: T0={self.config.time_top_ms:.0f}-{self.config.time_bottom_ms:.0f}ms, "
            f"V={self.config.velocity_top:.0f}/{self.config.velocity_bottom:.0f} m/s, "
            f"filter={self.config.filter_length_ms:.0f}ms{pred_str}"
        )

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply deconvolution to seismic data.

        Args:
            data: Input SeismicData

        Returns:
            Deconvolved SeismicData
        """
        traces = data.traces.copy()
        n_samples, n_traces = traces.shape
        sample_interval_ms = data.sample_rate  # Already in ms

        # Get offsets if available
        offsets = None
        if hasattr(data, 'headers') and data.headers is not None:
            if 'offset' in data.headers:
                offsets = np.abs(data.headers['offset'])
            elif 'OFFSET' in data.headers:
                offsets = np.abs(data.headers['OFFSET'])

        if offsets is None:
            # No offsets - use zero offset for all traces
            offsets = np.zeros(n_traces)
            logger.info("No offset information found, using zero offset for all traces")

        # Compute filter length in samples
        filter_length_samples = int(np.round(self.config.filter_length_ms / sample_interval_ms))
        filter_length_samples = max(filter_length_samples, 2)

        # Compute prediction distance in samples
        if self.config.prediction_distance_ms <= 0:
            # Auto: 1 sample for spiking
            if self.config.mode == 'spiking':
                prediction_distance_samples = 1
            else:
                # For predictive, default to filter_length/2
                prediction_distance_samples = filter_length_samples // 2
        else:
            prediction_distance_samples = int(np.round(
                self.config.prediction_distance_ms / sample_interval_ms
            ))
            prediction_distance_samples = max(prediction_distance_samples, 1)

        # White noise level (as fraction)
        white_noise = self.config.white_noise_percent / 100.0

        # Process each trace
        result = np.zeros_like(traces)

        for i in range(n_traces):
            self._report_progress(i, n_traces, f"Deconvolving trace {i+1}/{n_traces}")

            trace = traces[:, i]
            offset = offsets[i]

            # Compute NMO-corrected window bounds
            window_start, window_end = self._compute_window_samples(
                offset,
                sample_interval_ms,
                n_samples
            )

            # Extract design window
            window = trace[window_start:window_end].copy()

            if len(window) < filter_length_samples + prediction_distance_samples:
                # Window too short, skip deconvolution
                result[:, i] = trace
                logger.warning(
                    f"Trace {i}: window too short ({len(window)} samples), skipping deconvolution"
                )
                continue

            # Apply taper to window
            window = self._apply_taper(window)

            # Compute autocorrelation
            autocorr = self._compute_autocorrelation(
                window,
                filter_length_samples + prediction_distance_samples
            )

            # Solve Wiener-Hopf equation using Levinson recursion
            try:
                inverse_filter = self._levinson_durbin(
                    autocorr,
                    filter_length_samples,
                    prediction_distance_samples,
                    white_noise
                )
            except Exception as e:
                logger.warning(f"Trace {i}: Levinson failed ({e}), skipping deconvolution")
                result[:, i] = trace
                continue

            # Apply deconvolution filter to entire trace
            result[:, i] = self._apply_filter(trace, inverse_filter)

        # Create output SeismicData
        metadata = data.metadata.copy()
        if 'processing_history' not in metadata:
            metadata['processing_history'] = []
        metadata['processing_history'].append(self.get_description())

        return SeismicData(
            traces=result.astype(np.float32),
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata=metadata
        )

    def _compute_window_samples(
        self,
        offset: float,
        sample_interval_ms: float,
        n_samples: int
    ) -> Tuple[int, int]:
        """
        Compute window start and end samples.

        Window times are defined as start time at zero offset plus velocity moveout:
            T_top = time_top_ms + (offset / velocity_top) * 1000
            T_bottom = time_bottom_ms + (offset / velocity_bottom) * 1000

        Args:
            offset: Source-receiver offset in meters
            sample_interval_ms: Sample interval in milliseconds
            n_samples: Total number of samples

        Returns:
            Tuple of (start_sample, end_sample)
        """
        offset = abs(offset)

        # Window time = start time at zero offset + moveout
        # T = T0 + (offset / velocity) * 1000
        t_top_ms = self.config.time_top_ms + (offset / self.config.velocity_top) * 1000.0
        t_bottom_ms = self.config.time_bottom_ms + (offset / self.config.velocity_bottom) * 1000.0

        # Convert to samples
        start_sample = int(np.round(t_top_ms / sample_interval_ms))
        end_sample = int(np.round(t_bottom_ms / sample_interval_ms))

        # Clamp to valid range
        start_sample = max(0, min(start_sample, n_samples - 1))
        end_sample = max(start_sample + 1, min(end_sample, n_samples))

        return start_sample, end_sample

    def _apply_taper(self, window: np.ndarray) -> np.ndarray:
        """
        Apply cosine taper to window edges.

        Args:
            window: Input window

        Returns:
            Tapered window
        """
        if self.config.taper_percent <= 0:
            return window

        n = len(window)
        taper_samples = int(np.round(n * self.config.taper_percent / 100.0))

        if taper_samples < 1 or taper_samples * 2 >= n:
            return window

        # Create cosine taper
        taper = np.ones(n, dtype=np.float32)

        # Taper at start
        taper[:taper_samples] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples) / taper_samples))

        # Taper at end
        taper[-taper_samples:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples, 0, -1) / taper_samples))

        return window * taper

    def _compute_autocorrelation(
        self,
        window: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """
        Compute autocorrelation of windowed trace.

        r[k] = sum(x[n] * x[n+k]) / N

        Args:
            window: Input window
            max_lag: Maximum lag (filter_length + prediction_distance)

        Returns:
            Autocorrelation array r[0:max_lag]
        """
        n = len(window)
        max_lag = min(max_lag, n - 1)

        autocorr = np.zeros(max_lag, dtype=np.float64)

        # Normalize by window energy at lag 0
        for k in range(max_lag):
            autocorr[k] = np.sum(window[:n-k] * window[k:]) / n

        return autocorr

    def _levinson_durbin(
        self,
        autocorr: np.ndarray,
        filter_length: int,
        prediction_distance: int,
        white_noise: float
    ) -> np.ndarray:
        """
        Solve Wiener-Hopf equation using Levinson-Durbin recursion.

        For spiking decon (prediction_distance = 1):
            R * f = [1, 0, 0, ..., 0]^T

        For predictive decon:
            R * f = r[alpha:alpha+filter_length]

        where R is the Toeplitz autocorrelation matrix and alpha is
        the prediction distance.

        Args:
            autocorr: Autocorrelation array
            filter_length: Length of deconvolution operator
            prediction_distance: Prediction distance in samples
            white_noise: Pre-whitening level (fraction)

        Returns:
            Inverse (deconvolution) filter coefficients
        """
        # Add white noise to zero-lag for stability (pre-whitening)
        autocorr = autocorr.copy()
        autocorr[0] = autocorr[0] * (1.0 + white_noise)

        # Ensure we have enough autocorrelation values
        if len(autocorr) < filter_length + prediction_distance:
            # Pad with zeros
            autocorr = np.pad(
                autocorr,
                (0, filter_length + prediction_distance - len(autocorr)),
                mode='constant'
            )

        # Build Toeplitz matrix from autocorrelation
        # R = toeplitz(r[0:filter_length])
        r = autocorr[:filter_length]

        # Build right-hand side
        if prediction_distance <= 1:
            # Spiking decon: d = [1, 0, 0, ..., 0]
            d = np.zeros(filter_length, dtype=np.float64)
            d[0] = 1.0
        else:
            # Predictive decon: d = r[alpha:alpha+filter_length]
            d = autocorr[prediction_distance:prediction_distance + filter_length].copy()

        # Solve Toeplitz system using scipy's efficient solver
        # This uses the Levinson-Durbin algorithm internally
        try:
            # solve_toeplitz expects (c, r) where c is first column, r is first row
            # For symmetric Toeplitz (autocorrelation), c = r
            inverse_filter = solve_toeplitz(r, d)
        except np.linalg.LinAlgError:
            # Fallback to direct solve if Toeplitz solver fails
            R = toeplitz(r)
            inverse_filter = np.linalg.solve(R, d)

        return inverse_filter.astype(np.float32)

    def _apply_filter(
        self,
        trace: np.ndarray,
        inverse_filter: np.ndarray
    ) -> np.ndarray:
        """
        Apply deconvolution filter to trace.

        Args:
            trace: Input trace
            inverse_filter: Deconvolution operator

        Returns:
            Deconvolved trace
        """
        # Convolve trace with inverse filter
        # Use 'same' mode to preserve trace length
        deconvolved = np.convolve(trace, inverse_filter, mode='same')

        return deconvolved.astype(np.float32)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for multiprocess transfer."""
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'params': {
                'config': self.config.to_dict(),
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DeconvolutionProcessor':
        """Deserialize from dictionary."""
        params = d.get('params', d)
        config = DeconConfig.from_dict(params.get('config', {}))
        return cls(config=config)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_spiking_decon(
    traces: np.ndarray,
    sample_interval_ms: float,
    filter_length_ms: float = 160.0,
    time_top_ms: float = 100.0,
    time_bottom_ms: float = 500.0,
    velocity_top: float = 3500.0,
    velocity_bottom: float = 1500.0,
    white_noise_percent: float = 1.0,
    offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience function for spiking deconvolution.

    The design window is defined as start time plus velocity moveout:
        T_top = time_top_ms + (offset / velocity_top) * 1000
        T_bottom = time_bottom_ms + (offset / velocity_bottom) * 1000

    Args:
        traces: 2D array (n_samples, n_traces)
        sample_interval_ms: Sample interval in milliseconds
        filter_length_ms: Deconvolution operator length
        time_top_ms: Top window start time at zero offset (ms)
        time_bottom_ms: Bottom window start time at zero offset (ms)
        velocity_top: Top window moveout velocity (m/s)
        velocity_bottom: Bottom window moveout velocity (m/s)
        white_noise_percent: Pre-whitening percentage
        offsets: Offset array for window computation

    Returns:
        Deconvolved traces
    """
    config = DeconConfig(
        mode='spiking',
        time_top_ms=time_top_ms,
        time_bottom_ms=time_bottom_ms,
        velocity_top=velocity_top,
        velocity_bottom=velocity_bottom,
        filter_length_ms=filter_length_ms,
        white_noise_percent=white_noise_percent,
    )

    # Create SeismicData wrapper
    headers = {'offset': offsets} if offsets is not None else None
    data = SeismicData(
        traces=traces,
        sample_rate=sample_interval_ms,
        headers=headers,
    )

    processor = DeconvolutionProcessor(config=config)
    result = processor.process(data)

    return result.traces


def apply_predictive_decon(
    traces: np.ndarray,
    sample_interval_ms: float,
    prediction_distance_ms: float,
    filter_length_ms: float = 160.0,
    time_top_ms: float = 100.0,
    time_bottom_ms: float = 500.0,
    velocity_top: float = 3500.0,
    velocity_bottom: float = 1500.0,
    white_noise_percent: float = 1.0,
    offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience function for predictive deconvolution.

    The prediction distance should be set to the multiple period
    (typically the water bottom two-way time for water bottom multiples).

    The design window is defined as start time plus velocity moveout:
        T_top = time_top_ms + (offset / velocity_top) * 1000
        T_bottom = time_bottom_ms + (offset / velocity_bottom) * 1000

    Args:
        traces: 2D array (n_samples, n_traces)
        sample_interval_ms: Sample interval in milliseconds
        prediction_distance_ms: Prediction distance (multiple period)
        filter_length_ms: Deconvolution operator length
        time_top_ms: Top window start time at zero offset (ms)
        time_bottom_ms: Bottom window start time at zero offset (ms)
        velocity_top: Top window moveout velocity (m/s)
        velocity_bottom: Bottom window moveout velocity (m/s)
        white_noise_percent: Pre-whitening percentage
        offsets: Offset array for window computation

    Returns:
        Deconvolved traces
    """
    config = DeconConfig(
        mode='predictive',
        time_top_ms=time_top_ms,
        time_bottom_ms=time_bottom_ms,
        velocity_top=velocity_top,
        velocity_bottom=velocity_bottom,
        filter_length_ms=filter_length_ms,
        white_noise_percent=white_noise_percent,
        prediction_distance_ms=prediction_distance_ms,
    )

    headers = {'offset': offsets} if offsets is not None else None
    data = SeismicData(
        traces=traces,
        sample_rate=sample_interval_ms,
        headers=headers,
    )

    processor = DeconvolutionProcessor(config=config)
    result = processor.process(data)

    return result.traces