"""
GPU-Accelerated Short-Time Fourier Transform (STFT).

Provides GPU-accelerated STFT implementation using PyTorch for both
forward and inverse transforms.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

from .utils_gpu import numpy_to_tensor, tensor_to_numpy

logger = logging.getLogger(__name__)


class STFT_GPU:
    """
    GPU-accelerated Short-Time Fourier Transform.

    Uses PyTorch's native STFT implementation optimized for CUDA/MPS.
    """

    def __init__(
        self,
        device: torch.device,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        normalized: bool = False,
        return_complex: bool = True
    ):
        """
        Initialize STFT GPU processor.

        Args:
            device: PyTorch device (cuda, mps, or cpu)
            n_fft: FFT window size
            hop_length: Number of samples between windows (default: n_fft // 4)
            window: Window function ('hann', 'hamming', 'blackman', etc.)
            center: Whether to pad signal for centered windows
            normalized: Whether to normalize the STFT
            return_complex: If True, return complex tensor, else separate real/imag
        """
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.window_type = window
        self.center = center
        self.normalized = normalized
        self.return_complex = return_complex

        # Create window on device
        self.window = self._create_window().to(device)

    def _create_window(self) -> torch.Tensor:
        """Create window function."""
        if self.window_type == 'hann':
            return torch.hann_window(self.n_fft)
        elif self.window_type == 'hamming':
            return torch.hamming_window(self.n_fft)
        elif self.window_type == 'blackman':
            return torch.blackman_window(self.n_fft)
        elif self.window_type == 'bartlett':
            return torch.bartlett_window(self.n_fft)
        else:
            # Default to Hann window
            logger.warning(f"Unknown window '{self.window_type}', using 'hann'")
            return torch.hann_window(self.n_fft)

    def forward(
        self,
        signal: np.ndarray,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        sample_rate: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute STFT on GPU.

        Args:
            signal: Input signal (1D array)
            fmin: Minimum frequency to return (Hz, requires sample_rate)
            fmax: Maximum frequency to return (Hz, requires sample_rate)
            sample_rate: Sample rate in Hz (for frequency filtering)

        Returns:
            Tuple of (STFT coefficients, frequencies)
            - STFT coefficients: complex array of shape (n_freqs, n_frames)
            - frequencies: frequency array
        """
        # Transfer to GPU
        signal_gpu = numpy_to_tensor(signal, self.device, dtype=torch.float32)

        # Compute STFT
        stft_result = torch.stft(
            signal_gpu,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            return_complex=True  # Always return complex internally
        )

        # STFT result shape: (n_freqs, n_frames)
        n_freqs = stft_result.shape[0]

        # Create frequency array
        if sample_rate is not None:
            freqs = np.fft.rfftfreq(self.n_fft, d=1.0/sample_rate)
        else:
            freqs = np.fft.rfftfreq(self.n_fft)

        # Note: We don't filter frequencies here because torch.istft requires
        # the full frequency range for reconstruction. Frequency filtering
        # should be done via thresholding in the TF domain instead.
        # The fmin/fmax parameters are kept for API compatibility but not used.

        # Transfer back to CPU
        stft_numpy = tensor_to_numpy(stft_result)

        return stft_numpy, freqs

    def inverse(
        self,
        stft_coeffs: np.ndarray,
        signal_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute inverse STFT on GPU.

        Args:
            stft_coeffs: STFT coefficients (complex array, shape: n_freqs x n_frames)
            signal_length: Length of output signal (for proper reconstruction)

        Returns:
            Reconstructed signal (1D array)
        """
        # Transfer to GPU
        stft_gpu = torch.from_numpy(stft_coeffs).to(self.device)

        # Compute inverse STFT
        signal = torch.istft(
            stft_gpu,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            length=signal_length,
            return_complex=False
        )

        # Transfer back to CPU
        return tensor_to_numpy(signal)

    def batch_forward(
        self,
        signals: np.ndarray,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        sample_rate: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute STFT for multiple signals in batch (fully vectorized).

        Args:
            signals: Input signals (2D array: n_samples x n_signals)
            fmin: Minimum frequency to return (Hz)
            fmax: Maximum frequency to return (Hz)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (STFT coefficients, frequencies)
            - STFT coefficients: shape (n_signals, n_freqs, n_frames)
            - frequencies: frequency array
        """
        n_samples, n_signals = signals.shape

        # Transfer to GPU
        signals_gpu = numpy_to_tensor(signals.T, self.device, dtype=torch.float32)
        # Shape: (n_signals, n_samples)

        # Vectorized STFT: compute all signals at once
        # torch.stft supports batched input (batch_dim, signal_length)
        stft_batch = torch.stft(
            signals_gpu,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            return_complex=True
        )
        # Shape: (n_signals, n_freqs, n_frames)

        # Create frequency array
        if sample_rate is not None:
            freqs = np.fft.rfftfreq(self.n_fft, d=1.0/sample_rate)
        else:
            freqs = np.fft.rfftfreq(self.n_fft)

        # Note: We don't filter frequencies here because torch.istft requires
        # the full frequency range for reconstruction. Frequency filtering
        # should be done via thresholding in the TF domain instead.

        # Transfer back to CPU
        stft_numpy = tensor_to_numpy(stft_batch)

        return stft_numpy, freqs

    def batch_inverse(
        self,
        stft_coeffs: np.ndarray,
        signal_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute inverse STFT for multiple coefficient sets (fully vectorized).

        Args:
            stft_coeffs: STFT coefficients (3D array: n_signals x n_freqs x n_frames)
            signal_length: Length of output signals

        Returns:
            Reconstructed signals (2D array: n_samples x n_signals)
        """
        n_signals = stft_coeffs.shape[0]

        # Transfer to GPU
        stft_gpu = torch.from_numpy(stft_coeffs).to(self.device)

        # Vectorized inverse STFT: compute all signals at once
        # torch.istft supports batched input (batch, freq, time)
        signals_batch = torch.istft(
            stft_gpu,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            length=signal_length,
            return_complex=False
        )
        # Shape: (n_signals, n_samples)

        # Transfer back to CPU and transpose
        signals_numpy = tensor_to_numpy(signals_batch).T  # (n_samples, n_signals)

        return signals_numpy


def compute_stft_gpu(
    signal: np.ndarray,
    device: torch.device,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    window: str = 'hann',
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    sample_rate: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute STFT on GPU.

    Args:
        signal: Input signal (1D array)
        device: PyTorch device
        n_fft: FFT window size
        hop_length: Hop length (default: n_fft // 4)
        window: Window type
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        sample_rate: Sample rate (Hz)

    Returns:
        Tuple of (STFT coefficients, frequencies)
    """
    stft_processor = STFT_GPU(
        device=device,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )

    return stft_processor.forward(
        signal,
        fmin=fmin,
        fmax=fmax,
        sample_rate=sample_rate
    )


def compute_istft_gpu(
    stft_coeffs: np.ndarray,
    device: torch.device,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    window: str = 'hann',
    signal_length: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function to compute inverse STFT on GPU.

    Args:
        stft_coeffs: STFT coefficients
        device: PyTorch device
        n_fft: FFT window size
        hop_length: Hop length
        window: Window type
        signal_length: Output signal length

    Returns:
        Reconstructed signal
    """
    stft_processor = STFT_GPU(
        device=device,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )

    return stft_processor.inverse(stft_coeffs, signal_length)
