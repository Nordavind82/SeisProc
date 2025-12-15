"""
NMO (Normal Moveout) Processor for CDP Stacking

Implements hyperbolic NMO correction:
    t_nmo = sqrt(t0^2 + (offset/v_rms)^2)

Features:
- RMS velocity interpolation at each time sample
- Stretch mute to suppress artifacts at shallow times/large offsets
- Forward and inverse NMO
- Vectorized implementation for performance
- Supports both 1D and 2D velocity models
- Numba JIT-compiled sinc interpolation for 10-20x speedup

Usage:
    from processors.nmo_processor import NMOProcessor, NMOConfig
    from models.velocity_model import VelocityModel

    config = NMOConfig(stretch_mute_factor=1.5)
    processor = NMOProcessor(config, velocity_model)
    nmo_traces = processor.apply_nmo(traces, offsets, sample_interval_ms)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
import logging

# Try to import numba for JIT acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from models.velocity_model import VelocityModel, VelocityType
from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


# =============================================================================
# Numba JIT-compiled functions for 10-20x speedup
# =============================================================================

@jit(nopython=True, parallel=False, cache=True)
def _sinc_interp_numba(trace: np.ndarray,
                       idx: np.ndarray,
                       n_sinc: int = 8) -> np.ndarray:
    """
    Numba JIT-compiled sinc interpolation.

    Note: parallel=False to avoid thread contention with multi-process workers.
    This provides 10-20x speedup over pure Python implementation.

    Args:
        trace: Input trace samples (float32)
        idx: Target sample indices for interpolation (float32)
        n_sinc: Number of sinc lobes (default 8)

    Returns:
        Interpolated values at target indices
    """
    n = len(trace)
    result = np.zeros(len(idx), dtype=np.float32)

    for i in range(len(idx)):
        x = idx[i]
        if x < 0 or x >= n - 1:
            continue

        i0 = int(np.floor(x))
        i_start = max(0, i0 - n_sinc + 1)
        i_end = min(n, i0 + n_sinc + 1)

        acc = np.float32(0.0)
        for j in range(i_start, i_end):
            arg = x - j
            if np.abs(arg) < 1e-10:
                acc += trace[j]
            else:
                # Sinc function
                sinc_val = np.sin(np.pi * arg) / (np.pi * arg)
                # Hann window
                if np.abs(arg) <= n_sinc:
                    window = 0.5 * (1.0 + np.cos(np.pi * arg / n_sinc))
                else:
                    window = 0.0
                acc += trace[j] * sinc_val * window

        result[i] = acc

    return result


def _sinc_interp_python(trace: np.ndarray,
                        t_from: np.ndarray,
                        t_to: np.ndarray,
                        n_sinc: int = 8) -> np.ndarray:
    """
    Pure Python fallback for sinc interpolation.

    Used when Numba is not available.
    """
    n = len(trace)
    dt = t_from[1] - t_from[0] if len(t_from) > 1 else 1.0

    # Convert times to sample indices
    idx = t_to / dt

    result = np.zeros_like(t_to)

    for i, x in enumerate(idx):
        if x < 0 or x >= n - 1:
            continue

        # Get surrounding samples
        i0 = int(np.floor(x))
        i_start = max(0, i0 - n_sinc + 1)
        i_end = min(n, i0 + n_sinc + 1)

        # Sinc interpolation
        for j in range(i_start, i_end):
            arg = x - j
            if abs(arg) < 1e-10:
                result[i] += trace[j]
            else:
                # Windowed sinc (Hann window)
                sinc_val = np.sin(np.pi * arg) / (np.pi * arg)
                window = 0.5 * (1 + np.cos(np.pi * arg / n_sinc)) if abs(arg) <= n_sinc else 0
                result[i] += trace[j] * sinc_val * window

    return result


@dataclass
class NMOConfig:
    """
    Configuration for NMO correction.

    Attributes:
        stretch_mute_factor: Maximum allowed stretch factor (default 1.5).
                            Samples with stretch > this are muted.
                            Set to None or 0 to disable stretch mute.
        interpolation: Trace interpolation method ('linear' or 'sinc').
                      'linear' is faster, 'sinc' preserves frequency content.
        velocity_type: Expected velocity type ('rms' or 'interval').
                      If 'interval', will convert to RMS internally.
        taper_samples: Number of samples for stretch mute taper (smooth transition).
    """
    stretch_mute_factor: float = 1.5
    interpolation: str = 'linear'
    velocity_type: str = 'rms'
    taper_samples: int = 10

    def __post_init__(self):
        if self.interpolation not in ['linear', 'sinc']:
            raise ValueError(f"interpolation must be 'linear' or 'sinc', got {self.interpolation}")
        if self.velocity_type not in ['rms', 'interval']:
            raise ValueError(f"velocity_type must be 'rms' or 'interval', got {self.velocity_type}")
        if self.stretch_mute_factor is not None and self.stretch_mute_factor < 1.0:
            raise ValueError(f"stretch_mute_factor must be >= 1.0, got {self.stretch_mute_factor}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            'stretch_mute_factor': self.stretch_mute_factor,
            'interpolation': self.interpolation,
            'velocity_type': self.velocity_type,
            'taper_samples': self.taper_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NMOConfig':
        """Deserialize config from dictionary."""
        return cls(
            stretch_mute_factor=d.get('stretch_mute_factor', 1.5),
            interpolation=d.get('interpolation', 'linear'),
            velocity_type=d.get('velocity_type', 'rms'),
            taper_samples=d.get('taper_samples', 10),
        )


class NMOProcessor(BaseProcessor):
    """
    Normal Moveout (NMO) correction processor.

    Applies hyperbolic NMO correction using RMS velocities:
        t_nmo(t0, x) = sqrt(t0^2 + (x/v_rms(t0))^2)

    This flattens hyperbolic reflections in CDP gathers, preparing
    them for stacking.

    Args:
        config: NMOConfig with correction parameters
        velocity_model: VelocityModel with RMS velocities

    Example:
        >>> config = NMOConfig(stretch_mute_factor=1.5)
        >>> nmo = NMOProcessor(config, velocity_model)
        >>> corrected = nmo.apply_nmo(gather, offsets, dt_ms=4.0)
    """

    def __init__(
        self,
        config: Optional[NMOConfig] = None,
        velocity_model: Optional[VelocityModel] = None,
        **params
    ):
        # Handle config from params if provided via BaseProcessor interface
        if config is None and 'config' in params:
            config = params.pop('config')
        if velocity_model is None and 'velocity_model' in params:
            velocity_model = params.pop('velocity_model')

        self.config = config or NMOConfig()
        self.velocity_model = velocity_model

        # Store for serialization
        params['config'] = self.config.to_dict() if self.config else {}
        params['velocity_model_dict'] = velocity_model.to_dict() if velocity_model else None

        super().__init__(**params)

    def _validate_params(self):
        """Validate processor parameters."""
        if self.velocity_model is None:
            logger.warning("NMOProcessor created without velocity model")

    def set_velocity_model(self, velocity_model: VelocityModel) -> 'NMOProcessor':
        """
        Set or update the velocity model.

        Args:
            velocity_model: VelocityModel with RMS velocities

        Returns:
            self for method chaining
        """
        self.velocity_model = velocity_model
        self.params['velocity_model_dict'] = velocity_model.to_dict()
        return self

    def get_description(self) -> str:
        """Get human-readable description."""
        desc = f"NMO Correction (stretch_mute={self.config.stretch_mute_factor}, "
        desc += f"interp={self.config.interpolation})"
        if self.velocity_model:
            desc += f"\n  Velocity: {self.velocity_model}"
        return desc

    def process(self, data: SeismicData) -> SeismicData:
        """
        Process SeismicData by applying NMO correction.

        Requires offset information in the data.

        Args:
            data: Input SeismicData with traces and offset info

        Returns:
            NMO-corrected SeismicData
        """
        if self.velocity_model is None:
            raise ValueError("Velocity model not set")

        # Get offsets from data
        if hasattr(data, 'offsets') and data.offsets is not None:
            offsets = data.offsets
        else:
            raise ValueError("SeismicData must have offset information for NMO")

        # Apply NMO
        sample_interval_ms = data.sample_interval * 1000  # Convert to ms
        corrected_traces = self.apply_nmo(
            data.traces,
            offsets,
            sample_interval_ms
        )

        # Create new SeismicData
        return SeismicData(
            traces=corrected_traces,
            sample_interval=data.sample_interval,
            start_time=data.start_time,
        )

    def apply_nmo(
        self,
        traces: np.ndarray,
        offsets: np.ndarray,
        sample_interval_ms: float,
        cdp: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply NMO correction to a gather.

        Args:
            traces: 2D array (n_samples, n_traces)
            offsets: 1D array of offsets in meters (n_traces,)
            sample_interval_ms: Sample interval in milliseconds
            cdp: Optional CDP number for 2D velocity models

        Returns:
            NMO-corrected traces (n_samples, n_traces)
        """
        if self.velocity_model is None:
            raise ValueError("Velocity model not set")

        n_samples, n_traces = traces.shape
        dt_s = sample_interval_ms / 1000.0  # Convert to seconds

        # Build time axis (two-way time in seconds)
        t0 = np.arange(n_samples) * dt_s

        # Get RMS velocities at each time sample
        if cdp is not None and self.velocity_model.velocity_type == VelocityType.V_OF_XZ:
            v_rms = self.velocity_model.get_velocity_at(t0, x=float(cdp))
        else:
            v_rms = self.velocity_model.get_velocity_at(t0)

        # Ensure v_rms is 1D array
        v_rms = np.atleast_1d(v_rms).astype(np.float32)
        if len(v_rms) == 1:
            v_rms = np.full(n_samples, v_rms[0], dtype=np.float32)

        # Apply NMO trace by trace
        corrected = np.zeros_like(traces)
        stretch_mute = np.ones_like(traces)

        for i in range(n_traces):
            offset = abs(offsets[i])
            trace = traces[:, i]

            # Compute NMO time shift
            # t_nmo = sqrt(t0^2 + (x/v)^2)
            # We need to find what t0 maps to what t_nmo (forward)
            # For correction, we resample: corrected[t0] = original[t_nmo(t0)]

            with np.errstate(invalid='ignore', divide='ignore'):
                t_nmo = np.sqrt(t0**2 + (offset / v_rms)**2)

            # Compute stretch factor: dt_nmo/dt0
            # stretch = d(t_nmo)/d(t0) = t0 / t_nmo
            with np.errstate(invalid='ignore', divide='ignore'):
                stretch = np.where(t_nmo > 0, t_nmo / t0, 1.0)
                stretch[0] = stretch[1] if len(stretch) > 1 else 1.0  # Handle t=0

            # Interpolate to get corrected trace
            if self.config.interpolation == 'linear':
                corrected[:, i] = np.interp(t_nmo, t0, trace, left=0, right=0)
            else:
                corrected[:, i] = self._sinc_interp(trace, t0, t_nmo)

            # Apply stretch mute
            if self.config.stretch_mute_factor and self.config.stretch_mute_factor > 0:
                mute_mask = stretch > self.config.stretch_mute_factor
                stretch_mute[:, i] = self._apply_stretch_mute(
                    mute_mask, self.config.taper_samples
                )

        # Apply stretch mute
        corrected = corrected * stretch_mute

        return corrected.astype(np.float32)

    def apply_inverse_nmo(
        self,
        traces: np.ndarray,
        offsets: np.ndarray,
        sample_interval_ms: float,
        cdp: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply inverse NMO correction (restore original moveout).

        Args:
            traces: 2D array of NMO-corrected traces (n_samples, n_traces)
            offsets: 1D array of offsets in meters (n_traces,)
            sample_interval_ms: Sample interval in milliseconds
            cdp: Optional CDP number for 2D velocity models

        Returns:
            Traces with original moveout restored (n_samples, n_traces)
        """
        if self.velocity_model is None:
            raise ValueError("Velocity model not set")

        n_samples, n_traces = traces.shape
        dt_s = sample_interval_ms / 1000.0

        t0 = np.arange(n_samples) * dt_s

        if cdp is not None and self.velocity_model.velocity_type == VelocityType.V_OF_XZ:
            v_rms = self.velocity_model.get_velocity_at(t0, x=float(cdp))
        else:
            v_rms = self.velocity_model.get_velocity_at(t0)

        v_rms = np.atleast_1d(v_rms).astype(np.float32)
        if len(v_rms) == 1:
            v_rms = np.full(n_samples, v_rms[0], dtype=np.float32)

        restored = np.zeros_like(traces)

        for i in range(n_traces):
            offset = abs(offsets[i])
            trace = traces[:, i]

            # For inverse: we have corrected[t0] and want original[t_nmo]
            # original[t_nmo] = corrected[t0] where t_nmo = sqrt(t0^2 + (x/v)^2)
            with np.errstate(invalid='ignore', divide='ignore'):
                t_nmo = np.sqrt(t0**2 + (offset / v_rms)**2)

            # Inverse: sample from t0 (corrected) and place at t_nmo (original)
            if self.config.interpolation == 'linear':
                restored[:, i] = np.interp(t0, t_nmo, trace, left=0, right=0)
            else:
                restored[:, i] = self._sinc_interp(trace, t_nmo, t0)

        return restored.astype(np.float32)

    def compute_stretch_mute_mask(
        self,
        offsets: np.ndarray,
        sample_interval_ms: float,
        n_samples: int,
        cdp: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute stretch mute mask without applying NMO.

        Useful for visualizing which samples will be muted.

        Args:
            offsets: 1D array of offsets in meters
            sample_interval_ms: Sample interval in milliseconds
            n_samples: Number of samples
            cdp: Optional CDP number for 2D velocity models

        Returns:
            2D boolean mask (n_samples, n_traces) where True = muted
        """
        if self.velocity_model is None:
            raise ValueError("Velocity model not set")

        n_traces = len(offsets)
        dt_s = sample_interval_ms / 1000.0
        t0 = np.arange(n_samples) * dt_s

        if cdp is not None and self.velocity_model.velocity_type == VelocityType.V_OF_XZ:
            v_rms = self.velocity_model.get_velocity_at(t0, x=float(cdp))
        else:
            v_rms = self.velocity_model.get_velocity_at(t0)

        v_rms = np.atleast_1d(v_rms).astype(np.float32)
        if len(v_rms) == 1:
            v_rms = np.full(n_samples, v_rms[0], dtype=np.float32)

        mute_mask = np.zeros((n_samples, n_traces), dtype=bool)

        for i in range(n_traces):
            offset = abs(offsets[i])

            with np.errstate(invalid='ignore', divide='ignore'):
                t_nmo = np.sqrt(t0**2 + (offset / v_rms)**2)
                stretch = np.where(t_nmo > 0, t_nmo / t0, 1.0)
                stretch[0] = stretch[1] if len(stretch) > 1 else 1.0

            if self.config.stretch_mute_factor:
                mute_mask[:, i] = stretch > self.config.stretch_mute_factor

        return mute_mask

    def _apply_stretch_mute(
        self,
        mute_mask: np.ndarray,
        taper_samples: int,
    ) -> np.ndarray:
        """
        Create smooth stretch mute weights from boolean mask.

        Args:
            mute_mask: Boolean array where True = muted
            taper_samples: Number of samples for smooth transition

        Returns:
            Weight array (1.0 = keep, 0.0 = mute)
        """
        weights = np.ones_like(mute_mask, dtype=np.float32)
        weights[mute_mask] = 0.0

        if taper_samples > 0:
            # Find first muted sample and apply taper before it
            first_mute = np.argmax(mute_mask)
            if first_mute > 0 and mute_mask[first_mute]:
                taper_start = max(0, first_mute - taper_samples)
                taper = np.linspace(1.0, 0.0, first_mute - taper_start + 1)
                weights[taper_start:first_mute + 1] = taper

        return weights

    def _sinc_interp(
        self,
        trace: np.ndarray,
        t_from: np.ndarray,
        t_to: np.ndarray,
        n_sinc: int = 8,
    ) -> np.ndarray:
        """
        Sinc interpolation for better frequency preservation.

        Uses Numba JIT compilation for 10-20x speedup when available.
        Falls back to pure Python implementation otherwise.

        Args:
            trace: Input trace samples
            t_from: Original time axis
            t_to: Target time values
            n_sinc: Number of sinc lobes (default 8)

        Returns:
            Interpolated trace values at t_to
        """
        dt = t_from[1] - t_from[0] if len(t_from) > 1 else 1.0

        # Convert times to sample indices
        idx = (t_to / dt).astype(np.float32)

        # Ensure trace is float32 for Numba
        trace_f32 = trace.astype(np.float32) if trace.dtype != np.float32 else trace

        if NUMBA_AVAILABLE:
            try:
                return _sinc_interp_numba(trace_f32, idx, n_sinc)
            except Exception as e:
                logger.warning(f"Numba sinc_interp failed, using Python fallback: {e}")
                return _sinc_interp_python(trace, t_from, t_to, n_sinc)
        else:
            return _sinc_interp_python(trace, t_from, t_to, n_sinc)

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
                'velocity_model_dict': self.velocity_model.to_dict() if self.velocity_model else None,
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NMOProcessor':
        """Deserialize from dictionary."""
        params = d.get('params', d)
        config = NMOConfig.from_dict(params.get('config', {}))

        velocity_model = None
        if params.get('velocity_model_dict'):
            velocity_model = VelocityModel.from_dict(params['velocity_model_dict'])

        return cls(config=config, velocity_model=velocity_model)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_nmo_gather(
    traces: np.ndarray,
    offsets: np.ndarray,
    velocity_model: VelocityModel,
    sample_interval_ms: float,
    stretch_mute: float = 1.5,
    cdp: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to apply NMO to a single gather.

    Args:
        traces: 2D array (n_samples, n_traces)
        offsets: 1D array of offsets (n_traces,)
        velocity_model: VelocityModel with RMS velocities
        sample_interval_ms: Sample interval in milliseconds
        stretch_mute: Stretch mute factor (default 1.5)
        cdp: Optional CDP number for 2D velocity models

    Returns:
        NMO-corrected traces
    """
    config = NMOConfig(stretch_mute_factor=stretch_mute)
    processor = NMOProcessor(config, velocity_model)
    return processor.apply_nmo(traces, offsets, sample_interval_ms, cdp)


def compute_nmo_time(
    t0: np.ndarray,
    offset: float,
    velocity: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Compute NMO time for given t0, offset, and velocity.

    t_nmo = sqrt(t0^2 + (offset/v)^2)

    Args:
        t0: Zero-offset times (seconds)
        offset: Source-receiver offset (meters)
        velocity: RMS velocity (m/s), scalar or array matching t0

    Returns:
        NMO times (seconds)
    """
    velocity = np.atleast_1d(velocity).astype(np.float32)
    if len(velocity) == 1:
        velocity = np.full_like(t0, velocity[0])

    with np.errstate(invalid='ignore', divide='ignore'):
        t_nmo = np.sqrt(t0**2 + (offset / velocity)**2)

    return t_nmo


def compute_stretch_factor(
    t0: np.ndarray,
    offset: float,
    velocity: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Compute NMO stretch factor.

    stretch = t_nmo / t0

    Args:
        t0: Zero-offset times (seconds)
        offset: Source-receiver offset (meters)
        velocity: RMS velocity (m/s)

    Returns:
        Stretch factor array
    """
    t_nmo = compute_nmo_time(t0, offset, velocity)

    with np.errstate(invalid='ignore', divide='ignore'):
        stretch = np.where(t0 > 0, t_nmo / t0, 1.0)

    stretch[0] = stretch[1] if len(stretch) > 1 else 1.0

    return stretch
