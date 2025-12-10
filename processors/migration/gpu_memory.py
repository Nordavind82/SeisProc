"""
GPU Memory Manager for Kirchhoff Migration.

Optimizes GPU memory usage by:
- Pinning trace data in GPU memory for reuse across output tiles
- Tiling output to fit within available memory
- Managing memory pressure gracefully
- Efficient data transfers

Expected speedup: 1.2-1.5x from reduced memory transfers
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import logging
import gc

logger = logging.getLogger(__name__)


@dataclass
class OutputTile:
    """Represents a tile of the output image."""
    il_start: int
    il_end: int
    xl_start: int
    xl_end: int
    z_start: int = 0
    z_end: int = -1  # -1 means all depths

    @property
    def il_size(self) -> int:
        return self.il_end - self.il_start

    @property
    def xl_size(self) -> int:
        return self.xl_end - self.xl_start

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_il, n_xl) shape."""
        return (self.il_size, self.xl_size)


class GPUMemoryManager:
    """
    Manages GPU memory for efficient Kirchhoff migration.

    Features:
    - Pins trace data in GPU memory for reuse
    - Computes optimal output tile sizes based on available memory
    - Handles memory allocation and deallocation
    - Supports both CUDA and MPS (Apple Silicon)

    Example:
        >>> manager = GPUMemoryManager(device=torch.device('mps'))
        >>> traces_gpu = manager.transfer_traces_to_gpu(traces)
        >>> tile_size = manager.get_optimal_tile_size(n_traces, n_samples)
        >>> for tile in manager.generate_output_tiles(n_il, n_xl, tile_size):
        ...     output_tile = manager.allocate_output_tile(tile.shape, n_depths)
        ...     # Process tile...
        ...     manager.free_tensor(output_tile)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        target_memory_fraction: float = 0.7,
        min_tile_size: int = 64,  # Increased from 10 to reduce kernel launch overhead
        max_tile_size: int = 256,  # Increased from 200 for better GPU utilization
    ):
        """
        Initialize GPU memory manager.

        Args:
            device: Torch device (None = auto-detect)
            target_memory_fraction: Fraction of GPU memory to use (0-1)
            min_tile_size: Minimum tile size in each dimension (64 for efficiency)
            max_tile_size: Maximum tile size in each dimension (256 for large GPUs)
        """
        self.device = device or self._detect_device()
        self.target_memory_fraction = target_memory_fraction
        self.min_tile_size = min_tile_size
        self.max_tile_size = max_tile_size

        # Pinned buffers
        self._trace_buffer: Optional[torch.Tensor] = None
        self._geometry_buffer: Optional[dict] = None

        # Memory tracking
        self._allocated_bytes: int = 0
        self._peak_bytes: int = 0

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def get_available_memory(self) -> int:
        """
        Get available GPU memory in bytes.

        Returns:
            Available memory in bytes
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            return int(free * self.target_memory_fraction)
        elif self.device.type == 'mps':
            # MPS doesn't have memory query API, estimate conservatively
            # Assume 8GB for Apple Silicon, use 70% of target fraction
            estimated_total = 8 * 1024**3  # 8GB
            return int(estimated_total * self.target_memory_fraction * 0.7)
        else:
            # CPU: use system memory estimate
            import psutil
            available = psutil.virtual_memory().available
            return int(available * self.target_memory_fraction * 0.5)

    def get_optimal_tile_size(
        self,
        n_traces: int,
        n_samples: int,
        n_depths: int,
        bytes_per_element: int = 4,
    ) -> int:
        """
        Compute optimal output tile size based on available memory.

        Args:
            n_traces: Number of input traces
            n_samples: Samples per trace
            n_depths: Number of depth levels in output
            bytes_per_element: Bytes per element (4 for float32)

        Returns:
            Optimal tile size (same for both dimensions)
        """
        available = self.get_available_memory()

        # Memory needed for traces (already on GPU or will be transferred)
        trace_mem = n_traces * n_samples * bytes_per_element

        # Memory available for output tiles and working memory
        working_mem = available - trace_mem
        working_mem = max(working_mem, available * 0.3)  # At least 30%

        # For each tile, we need:
        # - Output image: n_depths * tile_size^2 * 4
        # - Output fold: n_depths * tile_size^2 * 4
        # - Working tensors: ~5 * n_depths * tile_size^2 * n_trace_batch * 4
        # Assuming trace_batch ~50:
        trace_batch = 50
        mem_per_tile_point = (
            2 * n_depths * bytes_per_element +  # image + fold
            5 * n_depths * trace_batch * bytes_per_element  # working
        )

        # Solve for tile_size^2
        max_tile_points = working_mem / mem_per_tile_point
        tile_size = int(np.sqrt(max_tile_points))

        # Clamp to valid range
        tile_size = max(self.min_tile_size, min(self.max_tile_size, tile_size))

        # MPS backend performs better with smaller tiles due to overhead
        # 35x35 tiles with depth batching showed best performance
        # Cap to 40x40 max for MPS, larger for CUDA
        if self.device.type == 'mps':
            tile_size = min(tile_size, 40)
        elif n_depths > 2000:
            max_tile_for_depth = int(np.sqrt(500 * 1e6 / (n_depths * 10 * 4)))
            tile_size = min(tile_size, max(64, max_tile_for_depth))

        logger.debug(
            f"Optimal tile size: {tile_size}x{tile_size} "
            f"(available: {available/1e9:.1f}GB, traces: {trace_mem/1e6:.0f}MB)"
        )

        return tile_size

    def transfer_traces_to_gpu(
        self,
        traces: np.ndarray,
        pin_memory: bool = True,
    ) -> torch.Tensor:
        """
        Transfer trace data to GPU and optionally pin for reuse.

        Args:
            traces: Trace data (n_samples, n_traces) or (n_traces, n_samples)
            pin_memory: If True, keep reference to prevent garbage collection

        Returns:
            GPU tensor with trace data
        """
        # Ensure float32
        if traces.dtype != np.float32:
            traces = traces.astype(np.float32)

        # Transfer to GPU
        traces_gpu = torch.from_numpy(traces).to(self.device)

        if pin_memory:
            self._trace_buffer = traces_gpu
            self._allocated_bytes += traces_gpu.numel() * 4
            self._peak_bytes = max(self._peak_bytes, self._allocated_bytes)

        logger.debug(f"Transferred {traces.nbytes/1e6:.1f}MB traces to {self.device}")

        return traces_gpu

    def transfer_geometry_to_gpu(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        offset: np.ndarray,
    ) -> dict:
        """
        Transfer geometry arrays to GPU.

        Args:
            source_x, source_y: Source coordinates
            receiver_x, receiver_y: Receiver coordinates
            offset: Source-receiver offsets

        Returns:
            Dict of GPU tensors
        """
        geometry_gpu = {
            'source_x': torch.from_numpy(source_x.astype(np.float32)).to(self.device),
            'source_y': torch.from_numpy(source_y.astype(np.float32)).to(self.device),
            'receiver_x': torch.from_numpy(receiver_x.astype(np.float32)).to(self.device),
            'receiver_y': torch.from_numpy(receiver_y.astype(np.float32)).to(self.device),
            'offset': torch.from_numpy(offset.astype(np.float32)).to(self.device),
        }

        self._geometry_buffer = geometry_gpu

        mem = sum(t.numel() * 4 for t in geometry_gpu.values())
        self._allocated_bytes += mem
        self._peak_bytes = max(self._peak_bytes, self._allocated_bytes)

        logger.debug(f"Transferred {mem/1e6:.1f}MB geometry to {self.device}")

        return geometry_gpu

    def allocate_output_tile(
        self,
        shape: Tuple[int, int],
        n_depths: int,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate GPU tensors for an output tile.

        Args:
            shape: (n_il, n_xl) tile shape
            n_depths: Number of depth levels
            dtype: Data type

        Returns:
            Tuple of (image_tile, fold_tile) tensors
        """
        n_il, n_xl = shape

        image_tile = torch.zeros(
            n_depths, n_il, n_xl,
            device=self.device, dtype=dtype
        )
        fold_tile = torch.zeros(
            n_depths, n_il, n_xl,
            device=self.device, dtype=dtype
        )

        mem = 2 * n_depths * n_il * n_xl * 4
        self._allocated_bytes += mem
        self._peak_bytes = max(self._peak_bytes, self._allocated_bytes)

        return image_tile, fold_tile

    def free_tensor(self, tensor: torch.Tensor):
        """
        Free a GPU tensor.

        Args:
            tensor: Tensor to free
        """
        mem = tensor.numel() * tensor.element_size()
        self._allocated_bytes = max(0, self._allocated_bytes - mem)

        del tensor

    def free_trace_buffer(self):
        """Free the pinned trace buffer."""
        if self._trace_buffer is not None:
            mem = self._trace_buffer.numel() * 4
            self._allocated_bytes = max(0, self._allocated_bytes - mem)
            del self._trace_buffer
            self._trace_buffer = None

    def free_geometry_buffer(self):
        """Free the pinned geometry buffer."""
        if self._geometry_buffer is not None:
            mem = sum(t.numel() * 4 for t in self._geometry_buffer.values())
            self._allocated_bytes = max(0, self._allocated_bytes - mem)
            del self._geometry_buffer
            self._geometry_buffer = None

    def empty_cache(self):
        """Empty GPU cache to free unused memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            # MPS has automatic memory management
            pass

        gc.collect()

    def generate_output_tiles(
        self,
        n_inline: int,
        n_xline: int,
        tile_size: int,
    ) -> List[OutputTile]:
        """
        Generate output tiles to cover the full grid.

        Args:
            n_inline: Total inline dimension
            n_xline: Total crossline dimension
            tile_size: Tile size in each dimension

        Returns:
            List of OutputTile objects
        """
        tiles = []

        for il_start in range(0, n_inline, tile_size):
            il_end = min(il_start + tile_size, n_inline)

            for xl_start in range(0, n_xline, tile_size):
                xl_end = min(xl_start + tile_size, n_xline)

                tiles.append(OutputTile(
                    il_start=il_start,
                    il_end=il_end,
                    xl_start=xl_start,
                    xl_end=xl_end,
                ))

        logger.debug(f"Generated {len(tiles)} output tiles of size ~{tile_size}x{tile_size}")

        return tiles

    def sync_output_to_cpu(
        self,
        tensor_gpu: torch.Tensor,
    ) -> np.ndarray:
        """
        Synchronize GPU tensor to CPU numpy array.

        Args:
            tensor_gpu: GPU tensor

        Returns:
            CPU numpy array
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        return tensor_gpu.cpu().numpy()

    def get_memory_stats(self) -> dict:
        """
        Get memory usage statistics.

        Returns:
            Dict with memory stats
        """
        stats = {
            'device': str(self.device),
            'allocated_bytes': self._allocated_bytes,
            'allocated_mb': self._allocated_bytes / 1e6,
            'peak_bytes': self._peak_bytes,
            'peak_mb': self._peak_bytes / 1e6,
            'available_bytes': self.get_available_memory(),
            'available_mb': self.get_available_memory() / 1e6,
        }

        if self.device.type == 'cuda':
            stats['cuda_allocated'] = torch.cuda.memory_allocated()
            stats['cuda_reserved'] = torch.cuda.memory_reserved()

        return stats

    @property
    def trace_buffer(self) -> Optional[torch.Tensor]:
        """Get pinned trace buffer."""
        return self._trace_buffer

    @property
    def geometry_buffer(self) -> Optional[dict]:
        """Get pinned geometry buffer."""
        return self._geometry_buffer


def create_gpu_memory_manager(
    device: Optional[torch.device] = None,
    target_memory_fraction: float = 0.7,
) -> GPUMemoryManager:
    """
    Factory function to create GPU memory manager.

    Args:
        device: Torch device (None = auto-detect)
        target_memory_fraction: Fraction of GPU memory to use

    Returns:
        Configured GPUMemoryManager
    """
    return GPUMemoryManager(
        device=device,
        target_memory_fraction=target_memory_fraction,
    )
