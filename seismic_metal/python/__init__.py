"""
Seismic Metal Kernels

GPU-accelerated seismic processing kernels for Apple Silicon.
Provides DWT, SWT, STFT, Gabor, and FKK processing with Metal acceleration.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import the compiled extension
_metal_available = False
_metal_module = None

try:
    from . import seismic_metal as _metal_module
    _metal_available = True
    logger.info(f"Metal kernels loaded: {_metal_module.get_device_info()}")
except ImportError as e:
    logger.warning(f"Metal kernels not available: {e}")
    _metal_module = None


def is_available() -> bool:
    """Check if Metal GPU acceleration is available."""
    if not _metal_available or _metal_module is None:
        return False
    return _metal_module.is_available()


def get_device_info() -> Dict[str, Any]:
    """Get Metal device information."""
    if not _metal_available or _metal_module is None:
        return {"available": False, "device_name": "Not available"}
    return _metal_module.get_device_info()


def initialize(shader_path: str = "") -> bool:
    """
    Initialize Metal device and load shaders.

    Parameters
    ----------
    shader_path : str, optional
        Path to metallib file. If empty, uses default location.

    Returns
    -------
    bool
        True if initialization successful
    """
    if not _metal_available or _metal_module is None:
        return False
    return _metal_module.initialize(shader_path)


def cleanup():
    """Release Metal resources."""
    if _metal_available and _metal_module is not None:
        _metal_module.cleanup()


def dwt_denoise(
    traces: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    threshold_mode: str = "soft",
    threshold_k: float = 3.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply DWT denoising using Metal GPU acceleration.

    Parameters
    ----------
    traces : ndarray
        Input traces with shape [n_samples, n_traces]
    wavelet : str
        Wavelet name ('db4', 'sym4')
    level : int
        Decomposition level
    threshold_mode : str
        'soft' or 'hard' thresholding
    threshold_k : float
        Threshold multiplier for MAD

    Returns
    -------
    tuple
        (denoised_traces, metrics_dict)

    Raises
    ------
    RuntimeError
        If Metal is not available
    """
    if not is_available():
        raise RuntimeError("Metal GPU acceleration not available")

    # Ensure contiguous float32 array
    traces = np.ascontiguousarray(traces, dtype=np.float32)

    return _metal_module.dwt_denoise(
        traces, wavelet, level, threshold_mode, threshold_k
    )


def swt_denoise(
    traces: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    threshold_mode: str = "soft",
    threshold_k: float = 3.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply SWT (Stationary Wavelet Transform) denoising using Metal GPU acceleration.

    Parameters
    ----------
    traces : ndarray
        Input traces with shape [n_samples, n_traces]
    wavelet : str
        Wavelet name ('db4', 'sym4')
    level : int
        Decomposition level
    threshold_mode : str
        'soft' or 'hard' thresholding
    threshold_k : float
        Threshold multiplier for MAD

    Returns
    -------
    tuple
        (denoised_traces, metrics_dict)
    """
    if not is_available():
        raise RuntimeError("Metal GPU acceleration not available")

    traces = np.ascontiguousarray(traces, dtype=np.float32)

    return _metal_module.swt_denoise(
        traces, wavelet, level, threshold_mode, threshold_k
    )


def stft_denoise(
    traces: np.ndarray,
    nperseg: int = 64,
    noverlap: int = 48,
    aperture: int = 21,
    threshold_k: float = 3.0,
    fmin: float = 0.0,
    fmax: float = 0.0,
    sample_rate: float = 500.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply STFT denoising using Metal GPU acceleration.

    Parameters
    ----------
    traces : ndarray
        Input traces with shape [n_samples, n_traces]
    nperseg : int
        FFT window size
    noverlap : int
        Overlap between windows
    aperture : int
        Spatial aperture for median computation
    threshold_k : float
        Threshold multiplier for MAD
    fmin, fmax : float
        Frequency range (0 = no limit)
    sample_rate : float
        Sample rate in Hz

    Returns
    -------
    tuple
        (denoised_traces, metrics_dict)
    """
    if not is_available():
        raise RuntimeError("Metal GPU acceleration not available")

    traces = np.ascontiguousarray(traces, dtype=np.float32)

    return _metal_module.stft_denoise(
        traces, nperseg, noverlap, aperture,
        threshold_k, fmin, fmax, sample_rate
    )


def gabor_denoise(
    traces: np.ndarray,
    window_size: int = 64,
    sigma: float = 0.0,
    overlap_pct: float = 75.0,
    aperture: int = 21,
    threshold_k: float = 3.0,
    fmin: float = 0.0,
    fmax: float = 0.0,
    sample_rate: float = 500.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Gabor denoising using Metal GPU acceleration.

    Parameters
    ----------
    traces : ndarray
        Input traces with shape [n_samples, n_traces]
    window_size : int
        Gaussian window size
    sigma : float
        Gaussian sigma (0 = auto)
    overlap_pct : float
        Overlap percentage
    aperture : int
        Spatial aperture for median computation
    threshold_k : float
        Threshold multiplier for MAD
    fmin, fmax : float
        Frequency range (0 = no limit)
    sample_rate : float
        Sample rate in Hz

    Returns
    -------
    tuple
        (denoised_traces, metrics_dict)
    """
    if not is_available():
        raise RuntimeError("Metal GPU acceleration not available")

    traces = np.ascontiguousarray(traces, dtype=np.float32)

    return _metal_module.gabor_denoise(
        traces, window_size, sigma, overlap_pct,
        aperture, threshold_k, fmin, fmax, sample_rate
    )


def fkk_filter(
    volume: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    v_min: float,
    v_max: float,
    mode: str = "reject",
    preserve_dc: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply 3D FK filtering using Metal GPU acceleration.

    Parameters
    ----------
    volume : ndarray
        Input volume with shape [nt, nx, ny]
    dt : float
        Time sample interval (seconds)
    dx : float
        Inline spacing (meters)
    dy : float
        Crossline spacing (meters)
    v_min : float
        Minimum velocity for filter (m/s)
    v_max : float
        Maximum velocity for filter (m/s)
    mode : str
        'reject' to remove velocities in range, 'pass' to keep only
    preserve_dc : bool
        Preserve DC component

    Returns
    -------
    tuple
        (filtered_volume, metrics_dict)
    """
    if not is_available():
        raise RuntimeError("Metal GPU acceleration not available")

    volume = np.ascontiguousarray(volume, dtype=np.float32)

    return _metal_module.fkk_filter(
        volume, dt, dx, dy, v_min, v_max, mode, preserve_dc
    )


__all__ = [
    'is_available',
    'get_device_info',
    'initialize',
    'cleanup',
    'dwt_denoise',
    'swt_denoise',
    'stft_denoise',
    'gabor_denoise',
    'fkk_filter',
]

__version__ = "1.0.0"
