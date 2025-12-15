"""
GPU Device Manager - Automatic device detection and management.

Handles automatic detection of available GPU devices (CUDA, MPS) and provides
utilities for memory management, device selection, and graceful CPU fallback.
"""

import torch
import warnings
from typing import Optional, Dict, Any
import logging


class DeviceManager:
    """
    Manages GPU device detection, selection, and memory management.

    Automatically detects the best available device in order of preference:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon Metal Performance Shaders)
    3. CPU (fallback)

    Provides utilities for:
    - Device information and capabilities
    - Memory management and monitoring
    - Graceful fallback handling
    - Error recovery
    """

    def __init__(self, prefer_device: str = 'auto', enable_gpu: bool = True):
        """
        Initialize device manager.

        Args:
            prefer_device: Preferred device type ('auto', 'cuda', 'mps', 'cpu')
            enable_gpu: If False, force CPU usage
        """
        self.enable_gpu = enable_gpu
        self.prefer_device = prefer_device
        self.device = self._detect_device()
        self.device_info = self._get_device_info()
        self.logger = logging.getLogger(__name__)

        # Log device selection
        self.logger.info(f"GPU Device Manager initialized: {self.get_device_name()}")

    def _detect_device(self) -> torch.device:
        """
        Auto-detect best available device.

        Returns:
            torch.device object for the selected device
        """
        # If GPU disabled, return CPU
        if not self.enable_gpu:
            return torch.device('cpu')

        # If specific device requested, try to use it
        if self.prefer_device != 'auto':
            if self.prefer_device == 'cuda' and torch.cuda.is_available():
                # Verify CUDA actually works (may fail in forked subprocess)
                try:
                    torch.cuda.current_device()
                    return torch.device('cuda')
                except RuntimeError as e:
                    warnings.warn(
                        f"CUDA requested but initialization failed "
                        f"(forked process?): {e}. Falling back to CPU."
                    )
                    return torch.device('cpu')
            elif self.prefer_device == 'mps' and torch.backends.mps.is_available():
                return torch.device('mps')
            elif self.prefer_device == 'cpu':
                return torch.device('cpu')
            else:
                warnings.warn(
                    f"Requested device '{self.prefer_device}' not available, "
                    f"falling back to auto-detection"
                )

        # Auto-detect: CUDA > MPS > CPU
        if torch.cuda.is_available():
            # Verify CUDA actually works (may fail in forked subprocess)
            try:
                torch.cuda.current_device()  # This will fail if CUDA context is broken
                return torch.device('cuda')
            except RuntimeError as e:
                warnings.warn(
                    f"CUDA reported available but initialization failed "
                    f"(forked process?): {e}. Falling back to CPU."
                )
                return torch.device('cpu')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the selected device.

        Returns:
            Dictionary with device specifications
        """
        info = {
            'type': self.device.type,
            'available': self.is_gpu_available(),
            'name': 'CPU',
            'memory_total': None,
            'memory_available': None,
            'compute_capability': None,
        }

        if self.device.type == 'cuda':
            try:
                info['name'] = torch.cuda.get_device_name(0)
                info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
                info['memory_available'] = (
                    info['memory_total'] - torch.cuda.memory_allocated(0)
                )
                info['compute_capability'] = torch.cuda.get_device_capability(0)
                info['cuda_version'] = torch.version.cuda
            except RuntimeError as e:
                # CUDA context broken (e.g., in forked subprocess)
                # Fall back to CPU-like info
                logging.getLogger(__name__).warning(
                    f"CUDA info unavailable (forked process?): {e}"
                )
                info['name'] = 'CUDA (unavailable in subprocess)'
                info['cuda_error'] = str(e)

        elif self.device.type == 'mps':
            info['name'] = 'Apple Silicon (Metal Performance Shaders)'
            # MPS uses unified memory, not easily queryable
            info['backend'] = 'MPS'
            info['macos_version'] = 'macOS 13.0+'  # MPS requires macOS 13.0+

        return info

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used."""
        return self.device.type in ('cuda', 'mps')

    def get_device_name(self) -> str:
        """Get human-readable device name."""
        return self.device_info['name']

    def get_device_type(self) -> str:
        """Get device type string ('cuda', 'mps', or 'cpu')."""
        return self.device.type

    def get_memory_info(self) -> Dict[str, Optional[int]]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory statistics (bytes)
        """
        if self.device.type == 'cuda':
            try:
                return {
                    'allocated': torch.cuda.memory_allocated(0),
                    'reserved': torch.cuda.memory_reserved(0),
                    'total': self.device_info['memory_total'],
                    'available': (
                        self.device_info['memory_total'] - torch.cuda.memory_allocated(0)
                    ),
                }
            except RuntimeError:
                # CUDA context broken - return empty info
                return {
                    'allocated': None,
                    'reserved': None,
                    'total': None,
                    'available': None,
                }
        elif self.device.type == 'mps':
            # MPS doesn't provide detailed memory stats
            return {
                'allocated': None,
                'reserved': None,
                'total': None,
                'available': None,
            }
        else:
            return {
                'allocated': 0,
                'reserved': 0,
                'total': None,
                'available': None,
            }

    def get_memory_usage_mb(self) -> Optional[float]:
        """
        Get current GPU memory usage in MB.

        Returns:
            Memory usage in MB, or None if not available
        """
        mem_info = self.get_memory_info()
        if mem_info['allocated'] is not None:
            return mem_info['allocated'] / (1024 ** 2)
        return None

    def calculate_batch_size(
        self,
        n_samples: int,
        n_traces: int,
        dtype_size: int = 4,
        safety_factor: float = 0.7
    ) -> int:
        """
        Calculate optimal batch size based on available GPU memory.

        Args:
            n_samples: Number of samples per trace
            n_traces: Total number of traces to process
            dtype_size: Size of data type in bytes (4 for float32, 8 for float64)
            safety_factor: Use only this fraction of available memory (0.7 = 70%)

        Returns:
            Recommended batch size (number of traces per batch)
        """
        mem_info = self.get_memory_info()

        if mem_info['available'] is None:
            # MPS or CPU - use conservative estimate
            if self.device.type == 'mps':
                # Assume 8GB available for MPS
                available_memory = 8 * (1024 ** 3) * safety_factor
            else:
                # CPU - process all at once
                return n_traces
        else:
            available_memory = mem_info['available'] * safety_factor

        # Estimate memory per trace (includes intermediate results)
        # Factor of 5: input + FFT + windows + thresholding + output
        memory_per_trace = n_samples * dtype_size * 5

        # Calculate batch size
        batch_size = int(available_memory / memory_per_trace)

        # Ensure at least 1, at most n_traces
        batch_size = max(1, min(batch_size, n_traces))

        self.logger.debug(
            f"Calculated batch size: {batch_size} traces "
            f"(available memory: {available_memory / (1024**3):.2f} GB)"
        )

        return batch_size

    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                self.logger.debug("CUDA cache cleared")
            except RuntimeError:
                # CUDA context broken - skip cache clearing
                pass
        elif self.device.type == 'mps':
            # MPS doesn't have explicit cache clearing
            # But we can trigger garbage collection
            import gc
            gc.collect()
            self.logger.debug("Garbage collection triggered for MPS")

    def synchronize(self):
        """Synchronize GPU operations (wait for all kernels to complete)."""
        if self.device.type == 'cuda':
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                # CUDA context broken - skip synchronization
                pass
        elif self.device.type == 'mps':
            # MPS synchronization
            torch.mps.synchronize()

    def can_fit_data(
        self,
        n_samples: int,
        n_traces: int,
        dtype_size: int = 4,
        safety_factor: float = 0.7
    ) -> bool:
        """
        Check if data can fit in GPU memory.

        Args:
            n_samples: Number of samples per trace
            n_traces: Number of traces
            dtype_size: Size of data type in bytes
            safety_factor: Use only this fraction of available memory

        Returns:
            True if data can fit, False otherwise
        """
        mem_info = self.get_memory_info()

        if mem_info['available'] is None:
            # MPS - assume it can handle moderate datasets
            # Conservative limit: 2GB
            return (n_samples * n_traces * dtype_size * 5) < (2 * 1024**3)

        available = mem_info['available'] * safety_factor
        required = n_samples * n_traces * dtype_size * 5  # Factor of 5 for intermediates

        return required < available

    def get_status_string(self) -> str:
        """
        Get formatted status string for UI display.

        Returns:
            Status string with device info and memory usage
        """
        if self.device.type == 'cuda':
            mem_info = self.get_memory_info()
            if mem_info['allocated'] is not None and mem_info['total'] is not None:
                mem_used_gb = mem_info['allocated'] / (1024**3)
                mem_total_gb = mem_info['total'] / (1024**3)
                return (
                    f"🟢 GPU Active: {self.get_device_name()} "
                    f"({mem_used_gb:.1f} / {mem_total_gb:.1f} GB)"
                )
            else:
                return f"🟡 GPU: {self.get_device_name()} (memory info unavailable)"
        elif self.device.type == 'mps':
            return f"🟢 GPU Active: {self.get_device_name()}"
        else:
            return "🟡 CPU Mode: GPU not available"

    def get_info_dict(self) -> Dict[str, Any]:
        """
        Get complete device information as dictionary.

        Returns:
            Dictionary with all device info and memory stats
        """
        info = {
            **self.device_info,
            'memory': self.get_memory_info(),
            'pytorch_version': torch.__version__,
        }
        return info

    def __repr__(self) -> str:
        """String representation of device manager."""
        return (
            f"DeviceManager(device={self.device.type}, "
            f"name='{self.get_device_name()}')"
        )


# Global device manager instance (lazy initialization)
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(
    prefer_device: str = 'auto',
    enable_gpu: bool = True,
    force_new: bool = False
) -> DeviceManager:
    """
    Get global device manager instance (singleton pattern).

    Args:
        prefer_device: Preferred device type ('auto', 'cuda', 'mps', 'cpu')
        enable_gpu: If False, force CPU usage
        force_new: If True, create new instance even if one exists

    Returns:
        DeviceManager instance
    """
    global _global_device_manager

    if _global_device_manager is None or force_new:
        _global_device_manager = DeviceManager(
            prefer_device=prefer_device,
            enable_gpu=enable_gpu
        )

    return _global_device_manager


def reset_device_manager():
    """
    Reset the global device manager singleton.

    IMPORTANT: Call this before forking worker processes!

    When using multiprocessing with 'fork' context, the global _global_device_manager
    is inherited by child processes. If CUDA was initialized in the parent (e.g., by
    the UI checking GPU availability), the inherited DeviceManager has a broken CUDA
    context. Resetting it before fork ensures child processes create their own fresh
    DeviceManager with valid CUDA context.

    This is not needed with 'spawn' context since spawn creates fresh processes.
    """
    global _global_device_manager
    _global_device_manager = None
