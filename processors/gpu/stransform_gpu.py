"""
GPU-Accelerated S-Transform (Stockwell Transform).

Provides GPU-accelerated S-Transform implementation using PyTorch for
both forward and inverse transforms with frequency-adaptive Gaussian windows.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

from .utils_gpu import numpy_to_tensor, tensor_to_numpy

logger = logging.getLogger(__name__)


class STransformGPU:
    """
    GPU-accelerated S-Transform (Stockwell Transform).

    Implements multi-resolution time-frequency transform with
    frequency-adaptive Gaussian windows, fully accelerated on GPU.
    """

    def __init__(self, device: torch.device):
        """
        Initialize S-Transform GPU processor.

        Args:
            device: PyTorch device (cuda, mps, or cpu)
        """
        self.device = device

    def forward(
        self,
        signal: np.ndarray,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        sample_rate: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute S-Transform on GPU.

        Args:
            signal: Input signal (1D array)
            fmin: Minimum frequency to compute (Hz)
            fmax: Maximum frequency to compute (Hz)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (S-Transform coefficients, frequencies)
            - S coefficients: complex array of shape (n_freqs, n_times)
            - frequencies: frequency array (Hz)
        """
        n = len(signal)

        # Transfer signal to GPU
        signal_gpu = numpy_to_tensor(signal, self.device, dtype=torch.float32)

        # Compute FFT on GPU
        fft_signal = torch.fft.fft(signal_gpu)

        # Create frequency array
        freqs_all = torch.fft.fftfreq(n, d=1.0/sample_rate).to(self.device)

        # Select positive frequencies only
        positive_mask = freqs_all >= 0
        freqs_positive = freqs_all[positive_mask]

        # Apply frequency range filter
        if fmin is None:
            # Skip DC component - convert to Python float to avoid MPS issues
            fmin = freqs_positive[1].item() if len(freqs_positive) > 1 else 0.0
        if fmax is None:
            fmax = sample_rate / 2.0  # Nyquist frequency

        freq_mask = (freqs_positive >= fmin) & (freqs_positive <= fmax)
        selected_freqs = freqs_positive[freq_mask]
        freq_indices = torch.where(positive_mask)[0][freq_mask]

        n_freqs = len(selected_freqs)
        logger.debug(f"Computing S-Transform for {n_freqs} frequencies on GPU")

        # Compute S-Transform on GPU
        S = self._compute_stransform_gpu(
            fft_signal,
            freqs_all,
            selected_freqs,
            freq_indices,
            n
        )

        # Transfer results back to CPU
        S_numpy = tensor_to_numpy(S)
        freqs_numpy = tensor_to_numpy(selected_freqs)

        return S_numpy, freqs_numpy

    def _compute_stransform_gpu(
        self,
        fft_signal: torch.Tensor,
        freqs_all: torch.Tensor,
        selected_freqs: torch.Tensor,
        freq_indices: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Compute S-Transform using vectorized GPU operations.

        Args:
            fft_signal: FFT of input signal (on GPU)
            freqs_all: All frequency values (on GPU)
            selected_freqs: Selected frequencies to compute (on GPU)
            freq_indices: Indices of selected frequencies (on GPU)
            n: Signal length

        Returns:
            S-Transform coefficients on GPU (n_freqs x n_times)
        """
        n_freqs = len(selected_freqs)

        # Pre-allocate result
        S = torch.zeros((n_freqs, n), dtype=torch.complex64, device=self.device)

        # Compute Gaussian windows for all frequencies (vectorized)
        windows = self._compute_gaussian_windows_vectorized(
            selected_freqs,
            freq_indices,
            freqs_all,
            n
        )

        # Apply windows and compute IFFT for all frequencies
        # Broadcast FFT signal: (1, n) and windows: (n_freqs, n)
        fft_windowed = fft_signal.unsqueeze(0) * windows

        # Compute IFFT for each frequency
        S = torch.fft.ifft(fft_windowed, dim=1)

        return S

    def _compute_gaussian_windows_vectorized(
        self,
        selected_freqs: torch.Tensor,
        freq_indices: torch.Tensor,
        freqs_all: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Compute Gaussian windows for all frequencies (vectorized).

        Args:
            selected_freqs: Selected frequency values (n_freqs,)
            freq_indices: Indices of selected frequencies (n_freqs,)
            freqs_all: All frequency values (n,)
            n: Signal length

        Returns:
            Windows tensor (n_freqs, n)
        """
        n_freqs = len(selected_freqs)

        # Frequency-adaptive window width
        # sigma_f = |f| / (2 * sqrt(2 * ln(2)))
        # Compute constant: 2 * sqrt(2 * ln(2)) = 2.35482...
        # Using numpy to avoid MPS tensor creation issues
        import numpy as np
        sigma_constant = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.35482

        sigma_f = torch.abs(selected_freqs) / sigma_constant
        # Shape: (n_freqs,)

        # Create frequency difference matrix
        # freq_indices: (n_freqs, 1)
        # freqs_all: (1, n)
        freq_indices_expanded = freq_indices.unsqueeze(1)  # (n_freqs, 1)
        freq_range = torch.arange(n, device=self.device).unsqueeze(0)  # (1, n)

        # Compute frequency differences (handle wraparound)
        freq_diff = torch.where(
            freq_range <= n // 2,
            freq_range - freq_indices_expanded,
            freq_range - freq_indices_expanded - n
        ).float()
        # Shape: (n_freqs, n)

        # Compute Gaussian window
        # W(f, k) = exp(-2π² σ_f² (k - k_f)²)
        sigma_f_expanded = sigma_f.unsqueeze(1)  # (n_freqs, 1)
        exponent = -2.0 * np.pi**2 * (sigma_f_expanded**2) * (freq_diff**2)
        windows = torch.exp(exponent)

        return windows

    def batch_forward(
        self,
        signals: np.ndarray,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        sample_rate: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute S-Transform for multiple signals in batch.

        Args:
            signals: Input signals (2D array: n_samples x n_signals)
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)
            sample_rate: Sample rate (Hz)

        Returns:
            Tuple of (S-Transform coefficients, frequencies)
            - S coefficients: shape (n_signals, n_freqs, n_times)
            - frequencies: frequency array
        """
        n_samples, n_signals = signals.shape

        # Process each signal (can be further optimized with batching)
        S_list = []
        freqs = None

        for i in range(n_signals):
            S_i, freqs_i = self.forward(
                signals[:, i],
                fmin=fmin,
                fmax=fmax,
                sample_rate=sample_rate
            )
            S_list.append(S_i)
            if freqs is None:
                freqs = freqs_i

        # Stack results
        S_batch = np.stack(S_list, axis=0)

        return S_batch, freqs

    def inverse(
        self,
        S: np.ndarray,
        freqs: np.ndarray,
        signal_length: int,
        sample_rate: float = 1.0
    ) -> np.ndarray:
        """
        Compute inverse S-Transform on GPU.

        Args:
            S: S-Transform coefficients (2D complex array: n_freqs x n_times)
            freqs: Frequency array
            signal_length: Length of output signal
            sample_rate: Sample rate in Hz

        Returns:
            Reconstructed signal (1D array)
        """
        n_freqs, n_times = S.shape

        # Transfer to GPU
        S_gpu = torch.from_numpy(S).to(self.device)

        # Average across frequencies to reconstruct
        # (Simple inverse - can be improved with proper synthesis)
        signal_gpu = torch.mean(S_gpu, dim=0).real

        # Transfer back to CPU
        signal = tensor_to_numpy(signal_gpu)

        # Ensure correct length
        if len(signal) < signal_length:
            signal = np.pad(signal, (0, signal_length - len(signal)))
        elif len(signal) > signal_length:
            signal = signal[:signal_length]

        return signal


def compute_stransform_gpu(
    signal: np.ndarray,
    device: torch.device,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    sample_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute S-Transform on GPU.

    Args:
        signal: Input signal (1D array)
        device: PyTorch device
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        sample_rate: Sample rate (Hz)

    Returns:
        Tuple of (S-Transform coefficients, frequencies)
    """
    st_processor = STransformGPU(device=device)
    return st_processor.forward(
        signal,
        fmin=fmin,
        fmax=fmax,
        sample_rate=sample_rate
    )


def compute_stransform_batch_gpu(
    signals: np.ndarray,
    device: torch.device,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    sample_rate: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute S-Transform for multiple signals.

    Args:
        signals: Input signals (2D array: n_samples x n_signals)
        device: PyTorch device
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        sample_rate: Sample rate (Hz)

    Returns:
        Tuple of (S-Transform coefficients, frequencies)
    """
    st_processor = STransformGPU(device=device)
    return st_processor.batch_forward(
        signals,
        fmin=fmin,
        fmax=fmax,
        sample_rate=sample_rate
    )
