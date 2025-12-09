"""
Trace Interpolation for Kirchhoff Migration

Implements efficient trace interpolation methods:
- Linear interpolation (fast, baseline)
- Sinc interpolation (8-point, accurate)

All implementations support GPU acceleration via PyTorch.
"""

import numpy as np
import torch
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class TraceInterpolator:
    """
    Interpolates seismic traces at arbitrary times.

    Supports both linear and sinc interpolation methods.
    """

    def __init__(
        self,
        method: str = 'linear',
        sinc_half_width: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trace interpolator.

        Args:
            method: 'linear' or 'sinc'
            sinc_half_width: Half-width of sinc kernel (points on each side)
            device: Torch device for GPU computation
        """
        self.method = method
        self.sinc_half_width = sinc_half_width
        self.device = device or self._detect_device()

        # Pre-compute sinc kernel coefficients if needed
        if method == 'sinc':
            self._init_sinc_kernel()

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _init_sinc_kernel(self):
        """Initialize sinc interpolation kernel coefficients."""
        # Precompute sinc values for fractional sample positions
        # Use Lanczos window for better convergence
        self.n_interp_points = 2 * self.sinc_half_width
        logger.debug(f"Initialized sinc kernel with {self.n_interp_points} points")

    def interpolate(
        self,
        traces: torch.Tensor,
        times: torch.Tensor,
        dt: float,
        t0: float = 0.0,
    ) -> torch.Tensor:
        """
        Interpolate traces at specified times.

        Args:
            traces: Trace data (n_samples, n_traces) or (n_samples, n_traces, ...)
            times: Interpolation times (n_interp,) or matching traces shape
            dt: Sample interval (seconds)
            t0: Time of first sample (seconds)

        Returns:
            Interpolated amplitudes matching times shape
        """
        if self.method == 'linear':
            return self._interpolate_linear(traces, times, dt, t0)
        elif self.method == 'sinc':
            return self._interpolate_sinc(traces, times, dt, t0)
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")

    def _interpolate_linear(
        self,
        traces: torch.Tensor,
        times: torch.Tensor,
        dt: float,
        t0: float,
    ) -> torch.Tensor:
        """
        Linear interpolation of traces.

        Args:
            traces: (n_samples, n_traces) or (n_samples, ...)
            times: Interpolation times (same shape as output desired)
            dt: Sample interval
            t0: Time of first sample

        Returns:
            Interpolated amplitudes
        """
        n_samples = traces.shape[0]

        # Convert times to sample indices
        sample_float = (times - t0) / dt
        sample_low = torch.floor(sample_float).long()
        sample_high = sample_low + 1
        frac = sample_float - sample_low.float()

        # Handle boundary conditions
        valid_mask = (sample_low >= 0) & (sample_high < n_samples)

        # Clamp indices for safe indexing
        sample_low_clamped = torch.clamp(sample_low, 0, n_samples - 1)
        sample_high_clamped = torch.clamp(sample_high, 0, n_samples - 1)

        # Get values at integer sample positions
        if traces.dim() == 1:
            # Single trace
            val_low = traces[sample_low_clamped]
            val_high = traces[sample_high_clamped]
        elif traces.dim() == 2:
            # Multiple traces (n_samples, n_traces)
            # Need to handle broadcasting for times
            if times.dim() == 1:
                # times is (n_times,), broadcast to (n_times, n_traces)
                val_low = traces[sample_low_clamped, :]
                val_high = traces[sample_high_clamped, :]
            else:
                # times matches trace indexing
                n_traces = traces.shape[1]
                trace_idx = torch.arange(n_traces, device=self.device)
                val_low = traces[sample_low_clamped, trace_idx]
                val_high = traces[sample_high_clamped, trace_idx]
        else:
            raise ValueError(f"Unsupported trace dimensions: {traces.dim()}")

        # Linear interpolation
        result = val_low * (1 - frac) + val_high * frac

        # Zero outside valid range
        result = torch.where(valid_mask, result, torch.zeros_like(result))

        return result

    def _interpolate_sinc(
        self,
        traces: torch.Tensor,
        times: torch.Tensor,
        dt: float,
        t0: float,
    ) -> torch.Tensor:
        """
        Sinc interpolation with Lanczos window.

        Provides more accurate interpolation than linear,
        important for migration accuracy.

        Args:
            traces: (n_samples, n_traces)
            times: Interpolation times
            dt: Sample interval
            t0: Time of first sample

        Returns:
            Interpolated amplitudes
        """
        n_samples = traces.shape[0]
        a = self.sinc_half_width  # Lanczos parameter

        # Convert times to sample indices
        sample_float = (times - t0) / dt
        sample_center = torch.round(sample_float).long()
        frac = sample_float - sample_center.float()

        # Initialize output
        result = torch.zeros_like(times, dtype=traces.dtype)

        # Compute Lanczos-windowed sinc interpolation
        for k in range(-a + 1, a + 1):
            idx = sample_center + k

            # Check bounds
            valid = (idx >= 0) & (idx < n_samples)
            idx_clamped = torch.clamp(idx, 0, n_samples - 1)

            # Sinc argument
            x = frac - k

            # Compute sinc(x) * lanczos(x)
            # sinc(x) = sin(pi*x) / (pi*x) for x != 0, 1 for x = 0
            # lanczos(x) = sinc(x/a) for |x| < a, 0 otherwise

            pi_x = np.pi * x

            # Handle x = 0 case
            near_zero = torch.abs(x) < 1e-6

            sinc_val = torch.where(
                near_zero,
                torch.ones_like(x),
                torch.sin(pi_x) / (pi_x + 1e-10)
            )

            lanczos_val = torch.where(
                near_zero,
                torch.ones_like(x),
                torch.sin(pi_x / a) / (pi_x / a + 1e-10)
            )

            weight = sinc_val * lanczos_val

            # Get trace values
            if traces.dim() == 1:
                val = traces[idx_clamped]
            else:
                # Handle multi-trace case
                val = traces[idx_clamped, torch.arange(traces.shape[1], device=self.device)]

            # Accumulate weighted contribution
            result = result + torch.where(valid, weight * val, torch.zeros_like(result))

        return result


def interpolate_batch(
    traces: torch.Tensor,
    times: torch.Tensor,
    dt: float,
    t0: float = 0.0,
    method: str = 'linear',
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Batch interpolation of traces at arbitrary times.

    This is the main entry point for migration interpolation.

    Args:
        traces: Trace data (n_samples, n_traces)
        times: Interpolation times (n_image_z, n_traces) or (n_times,)
        dt: Sample interval (seconds)
        t0: Time of first sample (seconds)
        method: 'linear' or 'sinc'
        device: Torch device

    Returns:
        Interpolated amplitudes matching times shape
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # Convert to tensors if needed
    if isinstance(traces, np.ndarray):
        traces = torch.from_numpy(traces).to(device)
    else:
        traces = traces.to(device)

    if isinstance(times, np.ndarray):
        times = torch.from_numpy(times).to(device)
    else:
        times = times.to(device)

    # Use optimized linear interpolation for most cases
    if method == 'linear':
        return _linear_interp_batch(traces, times, dt, t0)
    else:
        interpolator = TraceInterpolator(method=method, device=device)
        return interpolator.interpolate(traces, times, dt, t0)


def _linear_interp_batch(
    traces: torch.Tensor,
    times: torch.Tensor,
    dt: float,
    t0: float,
) -> torch.Tensor:
    """
    Optimized batch linear interpolation.

    Handles the common case of (n_samples, n_traces) traces
    with (n_z, n_traces) times tensor.
    """
    n_samples = traces.shape[0]

    # Convert times to sample indices
    sample_float = (times - t0) / dt
    sample_low = torch.floor(sample_float).long()
    frac = (sample_float - sample_low.float()).unsqueeze(-1) if times.dim() < traces.dim() else (sample_float - sample_low.float())

    # Handle different broadcasting cases
    if times.dim() == 1:
        # times is (n_times,), result is (n_times, n_traces)
        n_times = times.shape[0]
        n_traces = traces.shape[1]

        sample_low = sample_low.view(-1, 1).expand(n_times, n_traces)
        sample_high = sample_low + 1
        frac = frac.view(-1, 1).expand(n_times, n_traces)

        valid = (sample_low >= 0) & (sample_high < n_samples)
        sample_low_c = torch.clamp(sample_low, 0, n_samples - 1)
        sample_high_c = torch.clamp(sample_high, 0, n_samples - 1)

        # Gather values
        val_low = torch.gather(
            traces.unsqueeze(0).expand(n_times, -1, -1),
            1,
            sample_low_c.unsqueeze(2)
        ).squeeze(2)
        val_high = torch.gather(
            traces.unsqueeze(0).expand(n_times, -1, -1),
            1,
            sample_high_c.unsqueeze(2)
        ).squeeze(2)

    elif times.dim() == 2 and times.shape[1] == traces.shape[1]:
        # times is (n_z, n_traces), matching traces (n_samples, n_traces)
        n_z, n_traces = times.shape

        sample_high = sample_low + 1
        valid = (sample_low >= 0) & (sample_high < n_samples)
        sample_low_c = torch.clamp(sample_low, 0, n_samples - 1)
        sample_high_c = torch.clamp(sample_high, 0, n_samples - 1)

        # Fancy indexing for each trace column
        trace_idx = torch.arange(n_traces, device=traces.device).unsqueeze(0).expand(n_z, -1)

        val_low = traces[sample_low_c, trace_idx]
        val_high = traces[sample_high_c, trace_idx]

    else:
        raise ValueError(
            f"Incompatible shapes: traces {traces.shape}, times {times.shape}"
        )

    # Linear interpolation
    frac_expanded = frac if frac.shape == val_low.shape else frac.expand_as(val_low)
    result = val_low * (1 - frac_expanded) + val_high * frac_expanded

    # Zero invalid samples
    result = torch.where(valid, result, torch.zeros_like(result))

    return result


def interpolate_at_traveltimes(
    traces: torch.Tensor,
    traveltimes: torch.Tensor,
    dt: float,
    t0: float = 0.0,
    method: str = 'linear',
) -> torch.Tensor:
    """
    Interpolate traces at computed traveltimes.

    This is the key function for Kirchhoff migration - extracts
    amplitudes from input traces at the computed traveltime surfaces.

    Args:
        traces: Input traces (n_samples, n_traces)
        traveltimes: Total traveltimes (n_z, n_image_x, n_image_y, n_traces)
                     or (n_z, n_traces) for single image point per trace
        dt: Sample interval (seconds)
        t0: First sample time (seconds)
        method: Interpolation method

    Returns:
        Interpolated amplitudes matching traveltimes shape
    """
    # Flatten spatial dimensions for batch processing
    original_shape = traveltimes.shape

    if traveltimes.dim() > 2:
        n_z = original_shape[0]
        n_traces = original_shape[-1]
        n_spatial = np.prod(original_shape[1:-1])

        traveltimes_flat = traveltimes.view(n_z, n_spatial, n_traces)

        # Process each spatial point
        results = []
        for i in range(n_spatial):
            tt_slice = traveltimes_flat[:, i, :]  # (n_z, n_traces)
            amp = interpolate_batch(traces, tt_slice, dt, t0, method)
            results.append(amp)

        result = torch.stack(results, dim=1)
        return result.view(original_shape)
    else:
        return interpolate_batch(traces, traveltimes, dt, t0, method)
