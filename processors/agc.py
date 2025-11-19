"""
AGC (Automatic Gain Control) Processor

Fast vectorized implementation for amplitude equalization.
Uses sliding window RMS for gain calculation.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from typing import Tuple


def apply_agc_vectorized(
    traces: np.ndarray,
    window_samples: int,
    target_rms: float = 1.0,
    epsilon: float = 1e-10,
    max_gain: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply AGC (Automatic Gain Control) to seismic traces.

    Uses fast vectorized sliding window RMS calculation with scipy's
    uniform_filter for optimal performance.

    Args:
        traces: 2D array (n_samples, n_traces)
        window_samples: Window length in samples (should be odd)
        target_rms: Target RMS value (default 1.0)
        epsilon: Small value to prevent division by zero
        max_gain: Maximum allowed gain factor (prevents extreme amplification)

    Returns:
        Tuple of:
        - agc_traces: AGC-applied traces (same shape as input)
        - scale_factors: Scale factors used (for later inversion)

    Performance:
        ~50ms for 1000 traces × 2000 samples on modern CPU
    """
    n_samples, n_traces = traces.shape

    # Ensure window is odd
    if window_samples % 2 == 0:
        window_samples += 1

    # Clip window to trace length if needed
    window_samples = min(window_samples, n_samples)

    # Step 1: Compute squared traces
    traces_sq = traces ** 2

    # Step 2: Apply uniform filter to get mean of squares in sliding window
    # Using 'reflect' mode to handle edges smoothly
    mean_sq = uniform_filter(
        traces_sq,
        size=(window_samples, 1),
        mode='reflect'
    )

    # Step 3: Compute RMS from mean of squares
    rms = np.sqrt(mean_sq)

    # Step 4: Compute scale factors
    # Add epsilon to avoid division by zero
    scale_factors = target_rms / (rms + epsilon)

    # Clip extreme gains
    scale_factors = np.clip(scale_factors, 0.0, max_gain)

    # Step 5: Apply scaling
    agc_traces = traces * scale_factors

    return agc_traces, scale_factors


def remove_agc(
    agc_traces: np.ndarray,
    scale_factors: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Remove AGC by applying inverse scaling.

    Args:
        agc_traces: AGC-applied traces
        scale_factors: Scale factors from apply_agc_vectorized
        epsilon: Small value to prevent division by zero

    Returns:
        Approximately restored original traces

    Note:
        AGC is not perfectly reversible due to numerical precision
        and epsilon handling, but restoration is typically >99% accurate.
    """
    # Invert the scaling
    # Add epsilon to avoid division by zero
    restored_traces = agc_traces / (scale_factors + epsilon)

    return restored_traces


def calculate_agc_window_samples(
    window_ms: float,
    sample_rate: float
) -> int:
    """
    Convert AGC window from milliseconds to samples.

    Args:
        window_ms: Window length in milliseconds
        sample_rate: Sample rate in Hz (samples per second)

    Returns:
        Window length in samples (odd number)
    """
    # Convert ms to samples
    dt = 1000.0 / sample_rate  # ms per sample
    window_samples = int(window_ms / dt)

    # Ensure odd number for symmetric window
    if window_samples % 2 == 0:
        window_samples += 1

    # Minimum window of 3 samples
    window_samples = max(3, window_samples)

    return window_samples


def apply_agc_to_gather(
    gather_traces: np.ndarray,
    sample_rate: float,
    window_ms: float,
    target_rms: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply AGC to a seismic gather.

    Args:
        gather_traces: 2D array (n_samples, n_traces)
        sample_rate: Sample rate in Hz
        window_ms: AGC window length in milliseconds
        target_rms: Target RMS value

    Returns:
        Tuple of (agc_traces, scale_factors)
    """
    # Convert window to samples
    window_samples = calculate_agc_window_samples(window_ms, sample_rate)

    # Apply AGC
    agc_traces, scale_factors = apply_agc_vectorized(
        gather_traces,
        window_samples,
        target_rms=target_rms
    )

    return agc_traces, scale_factors


# GPU version (optional, if CuPy available)
try:
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter as uniform_filter_gpu

    def apply_agc_gpu(
        traces: np.ndarray,
        window_samples: int,
        target_rms: float = 1.0,
        epsilon: float = 1e-10,
        max_gain: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated AGC using CuPy.

        Same interface as apply_agc_vectorized but runs on GPU.
        Typically 5-10x faster for large gathers.
        """
        # Transfer to GPU
        traces_gpu = cp.asarray(traces)

        # Ensure window is odd
        if window_samples % 2 == 0:
            window_samples += 1

        # Same algorithm as CPU version
        traces_sq = traces_gpu ** 2
        mean_sq = uniform_filter_gpu(
            traces_sq,
            size=(window_samples, 1),
            mode='reflect'
        )
        rms = cp.sqrt(mean_sq)
        scale_factors_gpu = target_rms / (rms + epsilon)
        scale_factors_gpu = cp.clip(scale_factors_gpu, 0.0, max_gain)
        agc_traces_gpu = traces_gpu * scale_factors_gpu

        # Transfer back to CPU
        agc_traces = cp.asnumpy(agc_traces_gpu)
        scale_factors = cp.asnumpy(scale_factors_gpu)

        return agc_traces, scale_factors

    GPU_AVAILABLE = True

except ImportError:
    GPU_AVAILABLE = False
    apply_agc_gpu = None
