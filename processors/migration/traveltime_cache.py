"""
Traveltime Table Caching for Kirchhoff Migration

Provides caching mechanisms to speed up traveltime computation:
1. Pre-computed traveltime tables for regular grids
2. LRU cache for repeated lookups
3. Disk caching for large tables

Key optimization strategies:
- Pre-compute traveltimes on coarse grid, interpolate for fine positions
- Cache traveltimes per output point to avoid recomputation
- Store tables in GPU memory for fast lookup during migration

Trade-offs:
- Memory vs compute: Large tables require memory but save computation
- Accuracy vs speed: Coarser tables are faster but less accurate
- Pre-compute time vs migration time: Table generation is one-time cost
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import hashlib
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class TraveltimeTable:
    """
    Pre-computed traveltime table.

    Stores traveltimes from surface points to subsurface image points.

    Attributes:
        times: Traveltime array (n_z, n_surface_x, n_surface_y, n_image_x, n_image_y)
               or simpler shapes for 2D
        z_axis: Depth/time axis
        surface_x: Surface X coordinates
        surface_y: Surface Y coordinates
        image_x: Image point X coordinates
        image_y: Image point Y coordinates
        metadata: Additional info (velocity model hash, computation time, etc.)
    """
    times: Union[np.ndarray, torch.Tensor]
    z_axis: np.ndarray
    surface_x: np.ndarray
    surface_y: Optional[np.ndarray] = None
    image_x: Optional[np.ndarray] = None
    image_y: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get table shape."""
        return self.times.shape

    @property
    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        if isinstance(self.times, torch.Tensor):
            return self.times.numel() * self.times.element_size() / (1024 * 1024)
        else:
            return self.times.nbytes / (1024 * 1024)

    @property
    def is_2d(self) -> bool:
        """Check if table is for 2D migration."""
        return self.surface_y is None

    def to_device(self, device: torch.device) -> 'TraveltimeTable':
        """Move table to specified device."""
        if isinstance(self.times, np.ndarray):
            times = torch.from_numpy(self.times).to(device)
        else:
            times = self.times.to(device)

        return TraveltimeTable(
            times=times,
            z_axis=self.z_axis,
            surface_x=self.surface_x,
            surface_y=self.surface_y,
            image_x=self.image_x,
            image_y=self.image_y,
            metadata=self.metadata.copy(),
        )

    def save(self, filepath: str) -> None:
        """Save table to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        times = self.times
        if isinstance(times, torch.Tensor):
            times = times.cpu().numpy()

        save_dict = {
            'times': times,
            'z_axis': self.z_axis,
            'surface_x': self.surface_x,
            'metadata': json.dumps(self.metadata),
        }
        if self.surface_y is not None:
            save_dict['surface_y'] = self.surface_y
        if self.image_x is not None:
            save_dict['image_x'] = self.image_x
        if self.image_y is not None:
            save_dict['image_y'] = self.image_y

        np.savez_compressed(filepath, **save_dict)
        logger.info(f"Saved traveltime table to {filepath} ({self.memory_mb:.1f} MB)")

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> 'TraveltimeTable':
        """Load table from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Traveltime table not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)

        times = data['times']
        if device is not None:
            times = torch.from_numpy(times).to(device)

        metadata = {}
        if 'metadata' in data:
            try:
                metadata = json.loads(str(data['metadata']))
            except:
                pass

        table = cls(
            times=times,
            z_axis=data['z_axis'],
            surface_x=data['surface_x'],
            surface_y=data.get('surface_y'),
            image_x=data.get('image_x'),
            image_y=data.get('image_y'),
            metadata=metadata,
        )

        logger.info(f"Loaded traveltime table from {filepath} ({table.memory_mb:.1f} MB)")
        return table


class TraveltimeCache:
    """
    LRU cache for traveltime computations.

    Caches computed traveltimes to avoid redundant calculations.
    Uses hash of input coordinates as cache key.
    """

    def __init__(
        self,
        max_size_mb: float = 500.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize cache.

        Args:
            max_size_mb: Maximum cache size in MB
            device: Torch device
        """
        self.max_size_mb = max_size_mb
        self.device = device or torch.device('cpu')

        self._cache: Dict[str, torch.Tensor] = {}
        self._access_order: list = []  # For LRU eviction
        self._current_size_mb = 0.0

        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        surface_x: Union[float, np.ndarray, torch.Tensor],
        surface_y: Union[float, np.ndarray, torch.Tensor],
        z: float,
    ) -> str:
        """Create cache key from coordinates."""
        # Convert to numpy for hashing
        if isinstance(surface_x, torch.Tensor):
            surface_x = surface_x.cpu().numpy()
        if isinstance(surface_y, torch.Tensor):
            surface_y = surface_y.cpu().numpy()

        surface_x = np.atleast_1d(surface_x)
        surface_y = np.atleast_1d(surface_y)

        # Create hash
        data = np.concatenate([surface_x.flatten(), surface_y.flatten(), [z]])
        key = hashlib.md5(data.tobytes()).hexdigest()

        return key

    def get(
        self,
        surface_x: Union[float, np.ndarray, torch.Tensor],
        surface_y: Union[float, np.ndarray, torch.Tensor],
        z: float,
    ) -> Optional[torch.Tensor]:
        """
        Get cached traveltimes.

        Args:
            surface_x: Surface X coordinates
            surface_y: Surface Y coordinates
            z: Depth/time value

        Returns:
            Cached tensor or None if not found
        """
        key = self._make_key(surface_x, surface_y, z)

        if key in self._cache:
            self._hits += 1
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        self._misses += 1
        return None

    def put(
        self,
        surface_x: Union[float, np.ndarray, torch.Tensor],
        surface_y: Union[float, np.ndarray, torch.Tensor],
        z: float,
        traveltimes: torch.Tensor,
    ) -> None:
        """
        Store traveltimes in cache.

        Args:
            surface_x: Surface X coordinates
            surface_y: Surface Y coordinates
            z: Depth/time value
            traveltimes: Traveltime tensor
        """
        key = self._make_key(surface_x, surface_y, z)

        # Calculate size
        size_mb = traveltimes.numel() * traveltimes.element_size() / (1024 * 1024)

        # Evict if necessary
        while self._current_size_mb + size_mb > self.max_size_mb and self._access_order:
            self._evict_oldest()

        # Store
        self._cache[key] = traveltimes.to(self.device)
        self._access_order.append(key)
        self._current_size_mb += size_mb

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._access_order:
            return

        oldest_key = self._access_order.pop(0)
        if oldest_key in self._cache:
            tensor = self._cache.pop(oldest_key)
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            self._current_size_mb -= size_mb

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        self._current_size_mb = 0.0
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size_mb(self) -> float:
        """Get current cache size in MB."""
        return self._current_size_mb

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'size_mb': self.size_mb,
            'max_size_mb': self.max_size_mb,
            'entries': len(self._cache),
        }


class TraveltimeTableBuilder:
    """
    Builder for pre-computed traveltime tables.

    Creates tables optimized for migration performance.
    """

    def __init__(
        self,
        calculator,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize builder.

        Args:
            calculator: Traveltime calculator instance
            device: Torch device
        """
        self.calculator = calculator
        self.device = device or torch.device('cpu')

    def build_2d_table(
        self,
        z_axis: np.ndarray,
        surface_x: np.ndarray,
        image_x: np.ndarray,
        show_progress: bool = True,
    ) -> TraveltimeTable:
        """
        Build 2D traveltime table.

        Args:
            z_axis: Depth/time axis
            surface_x: Surface X coordinates
            image_x: Image point X coordinates
            show_progress: Show progress bar

        Returns:
            TraveltimeTable
        """
        start_time = time.time()

        n_z = len(z_axis)
        n_surface = len(surface_x)
        n_image = len(image_x)

        logger.info(
            f"Building 2D traveltime table: "
            f"{n_z} depths x {n_surface} surface x {n_image} image points"
        )

        # Allocate table
        times = torch.zeros((n_z, n_surface, n_image), device=self.device)

        # Convert to tensors
        surface_x_t = torch.from_numpy(surface_x.astype(np.float32)).to(self.device)
        surface_y_t = torch.zeros_like(surface_x_t)
        image_x_t = torch.from_numpy(image_x.astype(np.float32)).to(self.device)
        image_y_t = torch.zeros_like(image_x_t)
        z_axis_t = torch.from_numpy(z_axis.astype(np.float32)).to(self.device)

        # Compute traveltimes
        times = self.calculator.compute_traveltime_batch(
            surface_x_t, surface_y_t, image_x_t, image_y_t, z_axis_t
        )

        elapsed = time.time() - start_time

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
            image_x=image_x,
            metadata={
                'computation_time_s': elapsed,
                'calculator_type': type(self.calculator).__name__,
            },
        )

        logger.info(
            f"Built traveltime table in {elapsed:.2f}s, "
            f"size: {table.memory_mb:.1f} MB"
        )

        return table

    def build_3d_table(
        self,
        z_axis: np.ndarray,
        surface_x: np.ndarray,
        surface_y: np.ndarray,
        image_x: np.ndarray,
        image_y: np.ndarray,
        batch_size: int = 1000,
        show_progress: bool = True,
    ) -> TraveltimeTable:
        """
        Build 3D traveltime table.

        Args:
            z_axis: Depth/time axis
            surface_x: Surface X coordinates
            surface_y: Surface Y coordinates
            image_x: Image point X coordinates
            image_y: Image point Y coordinates
            batch_size: Batch size for computation
            show_progress: Show progress bar

        Returns:
            TraveltimeTable
        """
        start_time = time.time()

        n_z = len(z_axis)
        n_surface_x = len(surface_x)
        n_surface_y = len(surface_y)
        n_image_x = len(image_x)
        n_image_y = len(image_y)

        logger.info(
            f"Building 3D traveltime table: "
            f"{n_z} depths x {n_surface_x}x{n_surface_y} surface x "
            f"{n_image_x}x{n_image_y} image points"
        )

        # Full 3D table would be huge - use compressed representation
        # Store traveltimes for each (surface, image) pair across all z

        # Create meshgrids for surface and image points
        surf_xx, surf_yy = np.meshgrid(surface_x, surface_y, indexing='ij')
        img_xx, img_yy = np.meshgrid(image_x, image_y, indexing='ij')

        surf_x_flat = surf_xx.flatten()
        surf_y_flat = surf_yy.flatten()
        img_x_flat = img_xx.flatten()
        img_y_flat = img_yy.flatten()

        n_surface_total = len(surf_x_flat)
        n_image_total = len(img_x_flat)

        # Allocate table
        times = torch.zeros((n_z, n_surface_total, n_image_total), device=self.device)

        # Convert to tensors
        surface_x_t = torch.from_numpy(surf_x_flat.astype(np.float32)).to(self.device)
        surface_y_t = torch.from_numpy(surf_y_flat.astype(np.float32)).to(self.device)
        image_x_t = torch.from_numpy(img_x_flat.astype(np.float32)).to(self.device)
        image_y_t = torch.from_numpy(img_y_flat.astype(np.float32)).to(self.device)
        z_axis_t = torch.from_numpy(z_axis.astype(np.float32)).to(self.device)

        # Compute in batches if needed
        times = self.calculator.compute_traveltime_batch(
            surface_x_t, surface_y_t, image_x_t, image_y_t, z_axis_t
        )

        elapsed = time.time() - start_time

        table = TraveltimeTable(
            times=times,
            z_axis=z_axis,
            surface_x=surface_x,
            surface_y=surface_y,
            image_x=image_x,
            image_y=image_y,
            metadata={
                'computation_time_s': elapsed,
                'calculator_type': type(self.calculator).__name__,
                'surface_shape': (n_surface_x, n_surface_y),
                'image_shape': (n_image_x, n_image_y),
            },
        )

        logger.info(
            f"Built traveltime table in {elapsed:.2f}s, "
            f"size: {table.memory_mb:.1f} MB"
        )

        return table


class CachedTraveltimeCalculator:
    """
    Wrapper that adds caching to any traveltime calculator.

    Can use either LRU cache or pre-computed table.
    """

    def __init__(
        self,
        calculator,
        cache_size_mb: float = 500.0,
        table: Optional[TraveltimeTable] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize cached calculator.

        Args:
            calculator: Base traveltime calculator
            cache_size_mb: LRU cache size in MB
            table: Pre-computed table (optional)
            device: Torch device
        """
        self.calculator = calculator
        self.device = device or torch.device('cpu')
        self.cache = TraveltimeCache(max_size_mb=cache_size_mb, device=self.device)
        self.table = table

        if table is not None:
            self.table = table.to_device(self.device)

    def compute_traveltime(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute traveltime with caching.

        Args:
            x_offset: Horizontal offset in X
            y_offset: Horizontal offset in Y
            z_depth: Depth/time

        Returns:
            Traveltime
        """
        # For single point lookups, use the base calculator
        # Caching is more effective for batch operations
        return self.calculator.compute_traveltime(x_offset, y_offset, z_depth)

    def compute_traveltime_batch(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute batch traveltimes with caching.

        Args:
            surface_x: Surface X coordinates
            surface_y: Surface Y coordinates
            image_x: Image X coordinates
            image_y: Image Y coordinates
            image_z: Image Z (depth/time) values

        Returns:
            Traveltime tensor (n_z, n_surface, n_image)
        """
        # If we have a pre-computed table, use it for interpolation
        if self.table is not None:
            return self._lookup_from_table(
                surface_x, surface_y, image_x, image_y, image_z
            )

        # Otherwise use LRU cache for each depth level
        n_z = len(image_z)
        n_surface = len(surface_x)
        n_image = len(image_x)

        result = torch.zeros((n_z, n_surface, n_image), device=self.device)

        for iz, z in enumerate(image_z):
            z_val = float(z)

            # Check cache
            cached = self.cache.get(surface_x, surface_y, z_val)
            if cached is not None:
                result[iz] = cached
            else:
                # Compute and cache
                t = self._compute_single_depth(
                    surface_x, surface_y, image_x, image_y, z
                )
                self.cache.put(surface_x, surface_y, z_val, t)
                result[iz] = t

        return result

    def _compute_single_depth(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute traveltimes for single depth."""
        n_surface = len(surface_x)
        n_image = len(image_x)

        # Broadcast computation
        # For each (surface, image) pair, compute traveltime
        sx = surface_x.unsqueeze(1).expand(n_surface, n_image)
        sy = surface_y.unsqueeze(1).expand(n_surface, n_image)
        ix = image_x.unsqueeze(0).expand(n_surface, n_image)
        iy = image_y.unsqueeze(0).expand(n_surface, n_image)

        # Offset from surface to image
        dx = ix - sx
        dy = iy - sy

        # Compute traveltime
        t = self.calculator.compute_traveltime(dx.flatten(), dy.flatten(), z)

        return t.reshape(n_surface, n_image)

    def _lookup_from_table(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """Look up traveltimes from pre-computed table with interpolation."""
        # Simple nearest-neighbor lookup for now
        # Could implement trilinear interpolation for better accuracy

        n_z = len(image_z)
        n_surface = len(surface_x)
        n_image = len(image_x)

        result = torch.zeros((n_z, n_surface, n_image), device=self.device)

        table_z = torch.from_numpy(self.table.z_axis).to(self.device)
        table_sx = torch.from_numpy(self.table.surface_x).to(self.device)

        for iz, z in enumerate(image_z):
            # Find nearest z index
            z_idx = torch.argmin(torch.abs(table_z - z))

            for i_surf, (sx, sy) in enumerate(zip(surface_x, surface_y)):
                # Find nearest surface index
                surf_idx = torch.argmin(torch.abs(table_sx - sx))

                # Get traveltimes for all image points
                if self.table.is_2d:
                    result[iz, i_surf, :] = self.table.times[z_idx, surf_idx, :len(image_x)]
                else:
                    result[iz, i_surf, :] = self.table.times[z_idx, surf_idx, :len(image_x)]

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def create_traveltime_cache(
    max_size_mb: float = 500.0,
    device: Optional[torch.device] = None,
) -> TraveltimeCache:
    """
    Create traveltime cache.

    Args:
        max_size_mb: Maximum cache size
        device: Torch device

    Returns:
        TraveltimeCache instance
    """
    return TraveltimeCache(max_size_mb=max_size_mb, device=device)


def create_cached_calculator(
    calculator,
    cache_size_mb: float = 500.0,
    table_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> CachedTraveltimeCalculator:
    """
    Create cached traveltime calculator.

    Args:
        calculator: Base traveltime calculator
        cache_size_mb: LRU cache size
        table_path: Path to pre-computed table (optional)
        device: Torch device

    Returns:
        CachedTraveltimeCalculator instance
    """
    table = None
    if table_path is not None:
        table = TraveltimeTable.load(table_path, device=device)

    return CachedTraveltimeCalculator(
        calculator=calculator,
        cache_size_mb=cache_size_mb,
        table=table,
        device=device,
    )
