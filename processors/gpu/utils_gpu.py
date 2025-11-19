"""
GPU Utilities - Data transfer and common GPU operations.

Provides utility functions for:
- CPU ↔ GPU data transfer
- Tensor type conversions
- Memory management
- Error handling and fallback
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def numpy_to_tensor(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = False
) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor on specified device.

    Args:
        array: Input NumPy array
        device: Target device (cuda, mps, or cpu)
        dtype: Target data type (default: float32)
        pin_memory: Use pinned memory for faster GPU transfer (CUDA only)

    Returns:
        PyTorch tensor on specified device
    """
    # Convert to tensor on CPU first
    tensor = torch.from_numpy(array).to(dtype)

    # Pin memory for faster transfer (CUDA only)
    if pin_memory and device.type == 'cuda':
        tensor = tensor.pin_memory()

    # Transfer to device
    tensor = tensor.to(device)

    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Handles device transfer automatically (GPU → CPU).

    Args:
        tensor: Input PyTorch tensor

    Returns:
        NumPy array
    """
    # Move to CPU if on GPU
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    # Convert to NumPy
    return tensor.detach().numpy()


def safe_transfer_to_gpu(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Safely transfer NumPy array to GPU with error handling.

    Args:
        array: Input NumPy array
        device: Target device
        dtype: Target data type

    Returns:
        Tuple of (tensor on device or None, success flag)
    """
    try:
        tensor = numpy_to_tensor(array, device, dtype)
        return tensor, True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.warning(f"GPU out of memory: {e}")
            # Try to clear cache and retry
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return None, False
        else:
            logger.error(f"Error transferring to GPU: {e}")
            return None, False


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is contiguous in memory.

    Non-contiguous tensors can cause performance issues on GPU.

    Args:
        tensor: Input tensor

    Returns:
        Contiguous tensor
    """
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def get_optimal_dtype(device: torch.device, prefer_fp16: bool = False) -> torch.dtype:
    """
    Get optimal data type for device.

    Args:
        device: Target device
        prefer_fp16: If True, use FP16 for CUDA with tensor cores

    Returns:
        Recommended torch.dtype
    """
    if prefer_fp16 and device.type == 'cuda':
        # Check if GPU supports tensor cores (compute capability >= 7.0)
        if torch.cuda.get_device_capability(0)[0] >= 7:
            return torch.float16
    return torch.float32


def batch_transfer_to_gpu(
    arrays: list,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> list:
    """
    Transfer multiple NumPy arrays to GPU in batch.

    Args:
        arrays: List of NumPy arrays
        device: Target device
        dtype: Target data type

    Returns:
        List of tensors on device
    """
    tensors = []
    for array in arrays:
        tensor = numpy_to_tensor(array, device, dtype)
        tensors.append(tensor)
    return tensors


def batch_transfer_to_cpu(tensors: list) -> list:
    """
    Transfer multiple tensors from GPU to CPU.

    Args:
        tensors: List of PyTorch tensors

    Returns:
        List of NumPy arrays
    """
    arrays = []
    for tensor in tensors:
        array = tensor_to_numpy(tensor)
        arrays.append(array)
    return arrays


def estimate_tensor_memory(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32
) -> int:
    """
    Estimate memory required for tensor in bytes.

    Args:
        shape: Tensor shape
        dtype: Data type

    Returns:
        Memory in bytes
    """
    # Size per element
    dtype_size = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.complex64: 8,
        torch.complex128: 16,
    }

    element_size = dtype_size.get(dtype, 4)
    n_elements = np.prod(shape)

    return int(n_elements * element_size)


def check_memory_available(
    device: torch.device,
    required_bytes: int,
    safety_factor: float = 0.8
) -> bool:
    """
    Check if enough memory is available on device.

    Args:
        device: Device to check
        required_bytes: Required memory in bytes
        safety_factor: Use only this fraction of available memory

    Returns:
        True if enough memory available, False otherwise
    """
    if device.type == 'cuda':
        available = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated(0)
        )
        return required_bytes < (available * safety_factor)
    elif device.type == 'mps':
        # MPS uses unified memory - conservative estimate: 8GB available
        return required_bytes < (8 * 1024**3 * safety_factor)
    else:
        # CPU - assume always available
        return True


def create_frequency_tensor(
    n: int,
    sample_rate: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create frequency array for FFT operations.

    Args:
        n: Number of samples
        sample_rate: Sample rate in Hz
        device: Target device
        dtype: Data type

    Returns:
        Frequency tensor on device
    """
    # Create on CPU first
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    # Transfer to device
    return numpy_to_tensor(freqs, device, dtype)


def create_time_tensor(
    n: int,
    sample_rate: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create time array.

    Args:
        n: Number of samples
        sample_rate: Sample rate in Hz
        device: Target device
        dtype: Data type

    Returns:
        Time tensor on device
    """
    # Create on CPU first
    times = np.arange(n) / sample_rate
    # Transfer to device
    return numpy_to_tensor(times, device, dtype)


def compute_median_gpu(
    tensor: torch.Tensor,
    dim: int = 0,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute median along dimension (GPU-accelerated).

    MPS workaround: torch.median() is not fully supported on MPS,
    so we compute on CPU and transfer back.

    Args:
        tensor: Input tensor
        dim: Dimension to compute median along
        keepdim: Keep dimension after operation

    Returns:
        Median values tensor
    """
    if tensor.device.type == 'mps':
        # MPS workaround: compute median on CPU
        original_device = tensor.device
        tensor_cpu = tensor.cpu()
        median_values, _ = torch.median(tensor_cpu, dim=dim, keepdim=keepdim)
        # Transfer back to MPS
        median_values = median_values.to(original_device)
    else:
        # CUDA/CPU: use native median
        median_values, _ = torch.median(tensor, dim=dim, keepdim=keepdim)
    return median_values


def compute_mad_gpu(
    tensor: torch.Tensor,
    dim: int = 0,
    keepdim: bool = False,
    scale_factor: float = 1.4826
) -> torch.Tensor:
    """
    Compute Median Absolute Deviation (MAD) on GPU.

    Args:
        tensor: Input tensor
        dim: Dimension to compute MAD along
        keepdim: Keep dimension after operation
        scale_factor: Scale factor for normalization (1.4826 for Gaussian)

    Returns:
        MAD values tensor
    """
    # Compute median
    median = compute_median_gpu(tensor, dim=dim, keepdim=True)

    # Compute absolute deviations
    abs_deviations = torch.abs(tensor - median)

    # Compute MAD (median of absolute deviations)
    mad = compute_median_gpu(abs_deviations, dim=dim, keepdim=keepdim)

    # Scale to be consistent with standard deviation for Gaussian distribution
    return mad * scale_factor


def soft_threshold_gpu(
    tensor: torch.Tensor,
    threshold: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Apply soft thresholding on GPU.

    Soft threshold: sign(x) * max(|x| - threshold, 0)

    Args:
        tensor: Input tensor (complex or real)
        threshold: Threshold value(s)

    Returns:
        Thresholded tensor
    """
    if torch.is_complex(tensor):
        # Complex soft thresholding
        magnitude = torch.abs(tensor)
        phase = torch.angle(tensor)

        # Shrink magnitude
        new_magnitude = torch.maximum(
            magnitude - threshold,
            torch.zeros_like(magnitude)
        )

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)
    else:
        # Real soft thresholding
        return torch.sign(tensor) * torch.maximum(
            torch.abs(tensor) - threshold,
            torch.zeros_like(tensor)
        )


def garrote_threshold_gpu(
    tensor: torch.Tensor,
    threshold: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Apply Garrote thresholding on GPU.

    Garrote: x * max(1 - threshold²/x², 0)

    Args:
        tensor: Input tensor (complex or real)
        threshold: Threshold value(s)

    Returns:
        Thresholded tensor
    """
    if torch.is_complex(tensor):
        # Complex Garrote thresholding
        magnitude = torch.abs(tensor)
        phase = torch.angle(tensor)

        # Garrote shrinkage
        threshold_sq = threshold ** 2
        magnitude_sq = magnitude ** 2

        # Avoid division by zero
        safe_magnitude_sq = torch.where(
            magnitude_sq > 1e-10,
            magnitude_sq,
            torch.ones_like(magnitude_sq)
        )

        shrinkage = torch.maximum(
            1.0 - threshold_sq / safe_magnitude_sq,
            torch.zeros_like(magnitude)
        )

        new_magnitude = magnitude * shrinkage

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)
    else:
        # Real Garrote thresholding
        threshold_sq = threshold ** 2
        tensor_sq = tensor ** 2

        # Avoid division by zero
        safe_tensor_sq = torch.where(
            tensor_sq > 1e-10,
            tensor_sq,
            torch.ones_like(tensor_sq)
        )

        shrinkage = torch.maximum(
            1.0 - threshold_sq / safe_tensor_sq,
            torch.zeros_like(tensor)
        )

        return tensor * shrinkage


def benchmark_transfer(
    shape: Tuple[int, ...],
    device: torch.device,
    n_iterations: int = 10
) -> dict:
    """
    Benchmark data transfer speed between CPU and GPU.

    Args:
        shape: Shape of test array
        device: Target device
        n_iterations: Number of iterations for averaging

    Returns:
        Dictionary with timing statistics
    """
    import time

    # Create test data
    test_data = np.random.randn(*shape).astype(np.float32)

    # Benchmark CPU → GPU
    cpu_to_gpu_times = []
    for _ in range(n_iterations):
        start = time.time()
        tensor = numpy_to_tensor(test_data, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        elapsed = time.time() - start
        cpu_to_gpu_times.append(elapsed)

    # Benchmark GPU → CPU
    tensor_gpu = numpy_to_tensor(test_data, device)
    gpu_to_cpu_times = []
    for _ in range(n_iterations):
        start = time.time()
        result = tensor_to_numpy(tensor_gpu)
        elapsed = time.time() - start
        gpu_to_cpu_times.append(elapsed)

    data_size_mb = test_data.nbytes / (1024 ** 2)

    return {
        'shape': shape,
        'size_mb': data_size_mb,
        'cpu_to_gpu_ms': np.mean(cpu_to_gpu_times) * 1000,
        'gpu_to_cpu_ms': np.mean(gpu_to_cpu_times) * 1000,
        'cpu_to_gpu_bandwidth_gbps': (
            data_size_mb / np.mean(cpu_to_gpu_times) / 1024
        ),
        'gpu_to_cpu_bandwidth_gbps': (
            data_size_mb / np.mean(gpu_to_cpu_times) / 1024
        ),
    }
