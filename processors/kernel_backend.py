"""
Kernel Backend Factory

Provides abstraction layer for selecting between Python and Metal C++ kernel backends.
Supports automatic detection and user-configurable backend selection.

Backend options:
- AUTO: Automatically select best available (Metal > Python)
- PYTHON: Force Python/NumPy implementation
- METAL_CPP: Force Metal C++ kernels (requires compiled module)
"""

import logging
import os
import sys
from enum import Enum
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Add seismic_metal python path if needed
_seismic_metal_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'seismic_metal', 'python'
)
if os.path.exists(_seismic_metal_path) and _seismic_metal_path not in sys.path:
    sys.path.insert(0, _seismic_metal_path)


class KernelBackend(Enum):
    """Available kernel backend types."""
    AUTO = "auto"
    PYTHON = "python"
    METAL_CPP = "metal_cpp"


# Global backend preference
_global_backend: KernelBackend = KernelBackend.AUTO

# Cached availability status
_metal_available: Optional[bool] = None
_metal_module = None


def _check_metal_availability() -> bool:
    """Check if Metal C++ kernels are available."""
    global _metal_available, _metal_module

    if _metal_available is not None:
        return _metal_available

    # Try multiple import paths for the Metal module
    import_paths = [
        "seismic_metal.python.seismic_metal",  # Package structure
        "seismic_metal",  # Direct import (if in path)
    ]

    for import_path in import_paths:
        try:
            parts = import_path.rsplit(".", 1)
            if len(parts) == 2:
                module = __import__(parts[0], fromlist=[parts[1]])
                _metal_module = getattr(module, parts[1])
            else:
                _metal_module = __import__(import_path)

            _metal_available = _metal_module.is_available()
            if _metal_available:
                device_info = _metal_module.get_device_info()
                logger.info(f"Metal C++ kernels available: {device_info.get('device_name', 'Unknown')}")
            else:
                logger.info("Metal C++ module loaded but device not available")
            return _metal_available
        except (ImportError, AttributeError) as e:
            logger.debug(f"Metal C++ import from {import_path} failed: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error checking Metal availability from {import_path}: {e}")
            continue

    _metal_available = False
    _metal_module = None
    return _metal_available


def is_metal_available() -> bool:
    """Check if Metal C++ kernels are available."""
    return _check_metal_availability()


def get_metal_module():
    """Get the Metal module if available."""
    _check_metal_availability()
    return _metal_module


def set_global_backend(backend: KernelBackend):
    """Set the global default backend preference."""
    global _global_backend
    _global_backend = backend
    logger.info(f"Global kernel backend set to: {backend.value}")


def get_global_backend() -> KernelBackend:
    """Get the current global backend preference."""
    return _global_backend


def get_effective_backend(requested: Optional[KernelBackend] = None) -> KernelBackend:
    """
    Get the effective backend to use based on request and availability.

    Args:
        requested: Explicitly requested backend, or None to use global default

    Returns:
        The actual backend that will be used
    """
    backend = requested if requested is not None else _global_backend

    if backend == KernelBackend.AUTO:
        if is_metal_available():
            return KernelBackend.METAL_CPP
        return KernelBackend.PYTHON

    if backend == KernelBackend.METAL_CPP:
        if not is_metal_available():
            logger.warning("Metal C++ requested but not available, falling back to Python")
            return KernelBackend.PYTHON

    return backend


def get_backend_info() -> Dict[str, Any]:
    """Get information about available backends."""
    metal_available = is_metal_available()
    metal_device = None

    if metal_available and _metal_module is not None:
        try:
            device_info = _metal_module.get_device_info()
            metal_device = device_info.get('device_name', 'Unknown')
        except Exception:
            pass

    return {
        'global_backend': _global_backend.value,
        'effective_backend': get_effective_backend().value,
        'python_available': True,
        'metal_cpp_available': metal_available,
        'metal_device': metal_device,
    }


class KernelDispatcher:
    """
    Dispatches kernel calls to appropriate backend.

    Provides a unified interface for calling processing kernels
    with automatic backend selection and fallback.
    """

    def __init__(self, backend: Optional[KernelBackend] = None):
        """
        Initialize dispatcher.

        Args:
            backend: Backend to use, or None for global default
        """
        self.requested_backend = backend
        self._metal_module = None

    @property
    def effective_backend(self) -> KernelBackend:
        """Get the effective backend being used."""
        return get_effective_backend(self.requested_backend)

    def _get_metal(self):
        """Get Metal module, caching for reuse."""
        if self._metal_module is None:
            self._metal_module = get_metal_module()
        return self._metal_module

    def dwt_denoise(
        self,
        traces: np.ndarray,
        wavelet: str = "db4",
        level: int = 4,
        threshold_mode: str = "soft",
        threshold_k: float = 3.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply DWT denoising using selected backend.

        Args:
            traces: Input traces [n_samples, n_traces]
            wavelet: Wavelet name
            level: Decomposition level
            threshold_mode: 'soft' or 'hard'
            threshold_k: Threshold multiplier

        Returns:
            Tuple of (denoised_traces, metrics_dict)
        """
        backend = self.effective_backend

        if backend == KernelBackend.METAL_CPP:
            metal = self._get_metal()
            if metal is not None:
                try:
                    return metal.dwt_denoise(
                        traces, wavelet, level, threshold_mode, threshold_k
                    )
                except Exception as e:
                    logger.warning(f"Metal DWT failed, falling back to Python: {e}")

        # Python fallback
        return self._dwt_denoise_python(
            traces, wavelet, level, threshold_mode, threshold_k
        )

    def _dwt_denoise_python(
        self,
        traces: np.ndarray,
        wavelet: str,
        level: int,
        threshold_mode: str,
        threshold_k: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Python implementation of DWT denoising."""
        import time
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets required. Install with: pip install PyWavelets")

        start = time.time()
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)

        for i in range(n_traces):
            coeffs = pywt.wavedec(traces[:, i], wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = threshold_k * sigma

            denoised_coeffs = [coeffs[0]]
            for c in coeffs[1:]:
                denoised_coeffs.append(
                    pywt.threshold(c, threshold, mode=threshold_mode)
                )
            denoised[:, i] = pywt.waverec(denoised_coeffs, wavelet)[:n_samples]

        elapsed = time.time() - start
        metrics = {
            'kernel_time_ms': elapsed * 1000,
            'total_time_ms': elapsed * 1000,
            'traces_processed': n_traces,
            'samples_processed': n_samples * n_traces,
            'backend': 'python'
        }
        return denoised, metrics

    def swt_denoise(
        self,
        traces: np.ndarray,
        wavelet: str = "db4",
        level: int = 4,
        threshold_mode: str = "soft",
        threshold_k: float = 3.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply SWT denoising using selected backend.

        Args:
            traces: Input traces [n_samples, n_traces]
            wavelet: Wavelet name
            level: Decomposition level
            threshold_mode: 'soft' or 'hard'
            threshold_k: Threshold multiplier

        Returns:
            Tuple of (denoised_traces, metrics_dict)
        """
        backend = self.effective_backend

        if backend == KernelBackend.METAL_CPP:
            metal = self._get_metal()
            if metal is not None:
                try:
                    return metal.swt_denoise(
                        traces, wavelet, level, threshold_mode, threshold_k
                    )
                except Exception as e:
                    logger.warning(f"Metal SWT failed, falling back to Python: {e}")

        # Python fallback
        return self._swt_denoise_python(
            traces, wavelet, level, threshold_mode, threshold_k
        )

    def _swt_denoise_python(
        self,
        traces: np.ndarray,
        wavelet: str,
        level: int,
        threshold_mode: str,
        threshold_k: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Python implementation of SWT denoising."""
        import time
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets required. Install with: pip install PyWavelets")

        start = time.time()
        n_samples, n_traces = traces.shape

        # Pad to power of 2 for SWT
        target_len = 2**int(np.ceil(np.log2(n_samples)))
        pad_len = target_len - n_samples

        if pad_len > 0:
            traces_padded = np.pad(traces, ((0, pad_len), (0, 0)), mode='reflect')
        else:
            traces_padded = traces

        denoised_padded = np.zeros_like(traces_padded)
        max_level = pywt.swt_max_level(traces_padded.shape[0])
        level = min(level, max_level)

        for i in range(n_traces):
            coeffs = pywt.swt(traces_padded[:, i], wavelet, level=level)
            sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
            threshold = threshold_k * sigma

            denoised_coeffs = []
            for cA, cD in coeffs:
                cD_thresh = pywt.threshold(cD, threshold, mode=threshold_mode)
                denoised_coeffs.append((cA, cD_thresh))

            denoised_padded[:, i] = pywt.iswt(denoised_coeffs, wavelet)

        denoised = denoised_padded[:n_samples, :]

        elapsed = time.time() - start
        metrics = {
            'kernel_time_ms': elapsed * 1000,
            'total_time_ms': elapsed * 1000,
            'traces_processed': n_traces,
            'samples_processed': n_samples * n_traces,
            'backend': 'python'
        }
        return denoised, metrics

    def stft_denoise(
        self,
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
        Apply STFT denoising using selected backend.

        Args:
            traces: Input traces [n_samples, n_traces]
            nperseg: FFT window size
            noverlap: Overlap between windows
            aperture: Spatial aperture
            threshold_k: Threshold multiplier
            fmin, fmax: Frequency range
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (denoised_traces, metrics_dict)
        """
        backend = self.effective_backend

        if backend == KernelBackend.METAL_CPP:
            metal = self._get_metal()
            if metal is not None:
                try:
                    return metal.stft_denoise(
                        traces, nperseg, noverlap, aperture,
                        threshold_k, fmin, fmax, sample_rate
                    )
                except Exception as e:
                    logger.warning(f"Metal STFT failed, falling back to Python: {e}")

        # Python fallback - use existing STFT processor
        return self._stft_denoise_python(
            traces, nperseg, noverlap, aperture,
            threshold_k, fmin, fmax, sample_rate
        )

    def _stft_denoise_python(
        self,
        traces: np.ndarray,
        nperseg: int,
        noverlap: int,
        aperture: int,
        threshold_k: float,
        fmin: float,
        fmax: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Python STFT denoising - delegates to existing processor."""
        import time
        start = time.time()

        # Import existing STFT processor
        from processors.stft_denoise import STFTDenoise
        from models.seismic_data import SeismicData

        processor = STFTDenoise(
            nperseg=nperseg,
            noverlap=noverlap,
            aperture=aperture,
            threshold_k=threshold_k,
            fmin=fmin if fmin > 0 else None,
            fmax=fmax if fmax > 0 else None
        )

        data = SeismicData(
            traces=traces,
            sample_rate=1000.0 / sample_rate  # Convert Hz to ms
        )

        result = processor.process(data)

        elapsed = time.time() - start
        n_samples, n_traces = traces.shape
        metrics = {
            'kernel_time_ms': elapsed * 1000,
            'total_time_ms': elapsed * 1000,
            'traces_processed': n_traces,
            'samples_processed': n_samples * n_traces,
            'backend': 'python'
        }
        return result.traces, metrics

    def gabor_denoise(
        self,
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
        Apply Gabor denoising using selected backend.

        Args:
            traces: Input traces [n_samples, n_traces]
            window_size: Gaussian window size
            sigma: Gaussian sigma (0 = auto)
            overlap_pct: Overlap percentage
            aperture: Spatial aperture
            threshold_k: Threshold multiplier
            fmin, fmax: Frequency range
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (denoised_traces, metrics_dict)
        """
        backend = self.effective_backend

        if backend == KernelBackend.METAL_CPP:
            metal = self._get_metal()
            if metal is not None:
                try:
                    return metal.gabor_denoise(
                        traces, window_size, sigma, overlap_pct,
                        aperture, threshold_k, fmin, fmax, sample_rate
                    )
                except Exception as e:
                    logger.warning(f"Metal Gabor failed, falling back to Python: {e}")

        # Python fallback
        return self._gabor_denoise_python(
            traces, window_size, sigma, overlap_pct,
            aperture, threshold_k, fmin, fmax, sample_rate
        )

    def _gabor_denoise_python(
        self,
        traces: np.ndarray,
        window_size: int,
        sigma: float,
        overlap_pct: float,
        aperture: int,
        threshold_k: float,
        fmin: float,
        fmax: float,
        sample_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Python Gabor denoising - delegates to existing processor."""
        import time
        start = time.time()

        from processors.gabor_denoise import GaborDenoise
        from models.seismic_data import SeismicData

        processor = GaborDenoise(
            window_size=window_size,
            sigma=sigma if sigma > 0 else None,
            overlap_pct=overlap_pct,
            aperture=aperture,
            threshold_k=threshold_k,
            fmin=fmin if fmin > 0 else None,
            fmax=fmax if fmax > 0 else None
        )

        data = SeismicData(
            traces=traces,
            sample_rate=1000.0 / sample_rate
        )

        result = processor.process(data)

        elapsed = time.time() - start
        n_samples, n_traces = traces.shape
        metrics = {
            'kernel_time_ms': elapsed * 1000,
            'total_time_ms': elapsed * 1000,
            'traces_processed': n_traces,
            'samples_processed': n_samples * n_traces,
            'backend': 'python'
        }
        return result.traces, metrics

    def fkk_filter(
        self,
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
        Apply 3D FK filtering using selected backend.

        Args:
            volume: Input volume [nt, nx, ny]
            dt: Time sample interval (seconds)
            dx: Inline spacing (meters)
            dy: Crossline spacing (meters)
            v_min: Minimum velocity (m/s)
            v_max: Maximum velocity (m/s)
            mode: 'reject' or 'pass'
            preserve_dc: Preserve DC component

        Returns:
            Tuple of (filtered_volume, metrics_dict)
        """
        backend = self.effective_backend

        if backend == KernelBackend.METAL_CPP:
            metal = self._get_metal()
            if metal is not None:
                try:
                    return metal.fkk_filter(
                        volume, dt, dx, dy, v_min, v_max, mode, preserve_dc
                    )
                except Exception as e:
                    logger.warning(f"Metal FKK failed, falling back to Python: {e}")

        # Python fallback
        return self._fkk_filter_python(
            volume, dt, dx, dy, v_min, v_max, mode, preserve_dc
        )

    def _fkk_filter_python(
        self,
        volume: np.ndarray,
        dt: float,
        dx: float,
        dy: float,
        v_min: float,
        v_max: float,
        mode: str,
        preserve_dc: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Python FKK filtering - uses existing FKKFilterCPU."""
        import time
        start = time.time()

        from processors.fkk_filter_gpu import FKKFilterCPU
        from models.seismic_volume import SeismicVolume
        from models.fkk_config import FKKConfig

        filter_cpu = FKKFilterCPU()
        vol = SeismicVolume(data=volume, dt=dt, dx=dx, dy=dy)
        config = FKKConfig(v_min=v_min, v_max=v_max, mode=mode)

        result = filter_cpu.apply_filter(vol, config)

        elapsed = time.time() - start
        nt, nx, ny = volume.shape
        metrics = {
            'kernel_time_ms': elapsed * 1000,
            'total_time_ms': elapsed * 1000,
            'traces_processed': nx * ny,
            'samples_processed': nt * nx * ny,
            'backend': 'python'
        }
        return result.data, metrics


# Module-level dispatcher instance for convenience
_default_dispatcher: Optional[KernelDispatcher] = None


def get_dispatcher(backend: Optional[KernelBackend] = None) -> KernelDispatcher:
    """Get kernel dispatcher instance."""
    global _default_dispatcher

    if backend is not None:
        return KernelDispatcher(backend)

    if _default_dispatcher is None:
        _default_dispatcher = KernelDispatcher()

    return _default_dispatcher


# Convenience functions
def dwt_denoise(traces, **kwargs):
    """Convenience function for DWT denoising."""
    return get_dispatcher().dwt_denoise(traces, **kwargs)


def swt_denoise(traces, **kwargs):
    """Convenience function for SWT denoising."""
    return get_dispatcher().swt_denoise(traces, **kwargs)


def stft_denoise(traces, **kwargs):
    """Convenience function for STFT denoising."""
    return get_dispatcher().stft_denoise(traces, **kwargs)


def gabor_denoise(traces, **kwargs):
    """Convenience function for Gabor denoising."""
    return get_dispatcher().gabor_denoise(traces, **kwargs)


def fkk_filter(volume, **kwargs):
    """Convenience function for FKK filtering."""
    return get_dispatcher().fkk_filter(volume, **kwargs)
