"""
GPU acceleration modules for seismic processing.

Provides GPU-accelerated implementations of time-frequency transforms
and denoising algorithms using PyTorch with MPS (MacBook) and CUDA (NVIDIA) backends.
"""

from .device_manager import DeviceManager, get_device_manager
from .stft_gpu import STFT_GPU, compute_stft_gpu, compute_istft_gpu
from .stransform_gpu import STransformGPU, compute_stransform_gpu, compute_stransform_batch_gpu
from .thresholding_gpu import ThresholdingGPU, apply_mad_threshold_gpu
from . import utils_gpu

__all__ = [
    'DeviceManager',
    'get_device_manager',
    'STFT_GPU',
    'compute_stft_gpu',
    'compute_istft_gpu',
    'STransformGPU',
    'compute_stransform_gpu',
    'compute_stransform_batch_gpu',
    'ThresholdingGPU',
    'apply_mad_threshold_gpu',
    'utils_gpu',
]
