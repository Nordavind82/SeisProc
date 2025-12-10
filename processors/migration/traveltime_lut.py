"""
Traveltime Lookup Table for Kirchhoff Migration.

Pre-computes traveltimes for discrete (offset, depth) pairs and provides
fast bilinear interpolation for arbitrary queries. Replaces expensive
sqrt(h² + z²) / v calculations with table lookups.

Expected speedup: 2-3x for traveltime computation.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class TraveltimeLUT:
    """
    Traveltime Lookup Table with bilinear interpolation.

    Pre-computes t(h, z) = sqrt(h² + z²) / v for a grid of
    horizontal offsets (h) and depths (z), then provides fast
    GPU-accelerated interpolation for arbitrary (h, z) queries.

    Features:
    - Bilinear interpolation for smooth results
    - GPU tensor storage for fast batch queries
    - Disk save/load for caching between sessions
    - Support for 1D velocity v(z)

    Example:
        >>> lut = TraveltimeLUT()
        >>> lut.build(velocity=2500.0, max_offset=5000.0, max_depth=5000.0)
        >>> t = lut.lookup(h=1000.0, z=2000.0)  # Single point
        >>> t_batch = lut.lookup_batch(h_array, z_array)  # Vectorized
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize TraveltimeLUT.

        Args:
            device: Torch device for computations. If None, auto-detects GPU.
        """
        self.device = device or self._detect_device()

        # Table storage
        self._table: Optional[torch.Tensor] = None
        self._h_axis: Optional[torch.Tensor] = None
        self._z_axis: Optional[torch.Tensor] = None

        # Table parameters
        self._max_offset: float = 0.0
        self._max_depth: float = 0.0
        self._n_offsets: int = 0
        self._n_depths: int = 0
        self._dh: float = 0.0
        self._dz: float = 0.0

        # Velocity model info
        self._velocity_type: str = 'constant'
        self._v0: float = 0.0
        self._gradient: float = 0.0

        # Metadata
        self._built: bool = False
        self._build_time: float = 0.0

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def build(
        self,
        velocity: Union[float, np.ndarray],
        max_offset: float = 5000.0,
        max_depth: float = 5000.0,
        n_offsets: int = 500,
        n_depths: int = 1000,
        gradient: float = 0.0,
    ) -> 'TraveltimeLUT':
        """
        Build the traveltime lookup table.

        Args:
            velocity: Constant velocity (m/s) or 1D velocity array v(z)
            max_offset: Maximum horizontal offset in meters
            max_depth: Maximum depth in meters
            n_offsets: Number of offset samples
            n_depths: Number of depth samples
            gradient: Velocity gradient (m/s per m) for linear v(z)

        Returns:
            self for method chaining
        """
        import time
        start_time = time.time()

        logger.info(f"Building traveltime LUT: {n_offsets}x{n_depths} table")

        # Store parameters
        self._max_offset = max_offset
        self._max_depth = max_depth
        self._n_offsets = n_offsets
        self._n_depths = n_depths
        self._dh = max_offset / (n_offsets - 1) if n_offsets > 1 else max_offset
        self._dz = max_depth / (n_depths - 1) if n_depths > 1 else max_depth

        # Create axes
        h_axis = np.linspace(0, max_offset, n_offsets, dtype=np.float32)
        z_axis = np.linspace(0, max_depth, n_depths, dtype=np.float32)

        # Handle velocity model
        if isinstance(velocity, (int, float)):
            self._velocity_type = 'constant' if gradient == 0 else 'gradient'
            self._v0 = float(velocity)
            self._gradient = gradient

            # Build table for constant or linear gradient velocity
            table = self._build_table_analytic(h_axis, z_axis, velocity, gradient)
        else:
            self._velocity_type = '1d_array'
            self._v0 = float(velocity[0]) if len(velocity) > 0 else 2500.0

            # Build table for arbitrary v(z)
            table = self._build_table_vz(h_axis, z_axis, velocity)

        # Convert to torch tensors on device
        self._table = torch.from_numpy(table).to(self.device)
        self._h_axis = torch.from_numpy(h_axis).to(self.device)
        self._z_axis = torch.from_numpy(z_axis).to(self.device)

        self._built = True
        self._build_time = time.time() - start_time

        logger.info(
            f"LUT built in {self._build_time:.2f}s: "
            f"{self.memory_mb:.1f} MB, velocity={self._v0:.0f} m/s"
        )

        return self

    def _build_table_analytic(
        self,
        h_axis: np.ndarray,
        z_axis: np.ndarray,
        v0: float,
        gradient: float = 0.0,
    ) -> np.ndarray:
        """Build table using analytic formulas for constant/gradient velocity."""
        # Create meshgrid: h varies along axis 0, z along axis 1
        h_grid, z_grid = np.meshgrid(h_axis, z_axis, indexing='ij')

        # Avoid division by zero at z=0
        z_safe = np.maximum(z_grid, 1e-6)

        # Handle None gradient
        gradient = gradient if gradient is not None else 0.0

        if gradient == 0 or abs(gradient) < 1e-10:
            # Constant velocity: t = sqrt(h² + z²) / v
            r = np.sqrt(h_grid**2 + z_safe**2)
            table = r / v0
        else:
            # Linear gradient v(z) = v0 + k*z
            # Use curved ray formula: t = (1/k) * arccosh(1 + k*r²/(2*v0*z))
            k = gradient
            r_sq = h_grid**2 + z_safe**2

            # Argument for arccosh
            arg = 1.0 + k * r_sq / (2.0 * v0 * z_safe)
            arg = np.maximum(arg, 1.0)  # Ensure valid for arccosh

            table = np.arccosh(arg) / k

        return table.astype(np.float32)

    def _build_table_vz(
        self,
        h_axis: np.ndarray,
        z_axis: np.ndarray,
        v_z: np.ndarray,
    ) -> np.ndarray:
        """Build table for arbitrary 1D velocity v(z) using numerical integration."""
        n_h = len(h_axis)
        n_z = len(z_axis)

        # Interpolate v(z) to table depth axis
        v_depths = np.linspace(0, self._max_depth, len(v_z))
        v_interp = np.interp(z_axis, v_depths, v_z)

        # For each (h, z), compute traveltime via effective velocity
        # Approximation: t ≈ sqrt(h² + z²) / v_eff(z)
        # where v_eff is RMS velocity to depth z

        table = np.zeros((n_h, n_z), dtype=np.float32)

        # Compute RMS velocity for each depth
        v_rms = np.zeros(n_z, dtype=np.float32)
        for i, z in enumerate(z_axis):
            if z < 1e-6:
                v_rms[i] = v_interp[0]
            else:
                # RMS velocity: sqrt(integral(v²dz) / z)
                z_samples = z_axis[:i+1]
                v_samples = v_interp[:i+1]
                v_sq_integral = np.trapz(v_samples**2, z_samples)
                v_rms[i] = np.sqrt(v_sq_integral / z)

        # Build table using RMS velocity
        h_grid, z_grid = np.meshgrid(h_axis, z_axis, indexing='ij')
        z_safe = np.maximum(z_grid, 1e-6)

        # Broadcast v_rms to grid shape
        v_grid = np.broadcast_to(v_rms, (n_h, n_z))

        r = np.sqrt(h_grid**2 + z_safe**2)
        table = r / v_grid

        return table.astype(np.float32)

    def lookup(self, h: float, z: float) -> float:
        """
        Look up traveltime for single (h, z) point.

        Args:
            h: Horizontal offset in meters
            z: Depth in meters

        Returns:
            Traveltime in seconds
        """
        if not self._built:
            raise RuntimeError("LUT not built. Call build() first.")

        # Convert to tensor and use batch lookup
        h_t = torch.tensor([h], device=self.device, dtype=torch.float32)
        z_t = torch.tensor([z], device=self.device, dtype=torch.float32)
        result = self.lookup_batch(h_t, z_t)
        return result.item()

    def lookup_batch(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch lookup with bilinear interpolation.

        Args:
            h: Horizontal offsets tensor (any shape)
            z: Depths tensor (broadcastable with h)

        Returns:
            Traveltimes tensor (same shape as broadcast(h, z))
        """
        if not self._built:
            raise RuntimeError("LUT not built. Call build() first.")

        # Ensure tensors are on correct device
        if h.device != self.device:
            h = h.to(self.device)
        if z.device != self.device:
            z = z.to(self.device)

        # Clamp to valid range
        h_clamped = torch.clamp(h, 0, self._max_offset - 1e-6)
        z_clamped = torch.clamp(z, 0, self._max_depth - 1e-6)

        # Convert to fractional indices
        h_idx = h_clamped / self._dh
        z_idx = z_clamped / self._dz

        # Get integer indices and fractions
        h_idx_floor = h_idx.long()
        z_idx_floor = z_idx.long()

        h_frac = h_idx - h_idx_floor.float()
        z_frac = z_idx - z_idx_floor.float()

        # Clamp indices to valid range
        h_idx_floor = torch.clamp(h_idx_floor, 0, self._n_offsets - 2)
        z_idx_floor = torch.clamp(z_idx_floor, 0, self._n_depths - 2)

        h_idx_ceil = h_idx_floor + 1
        z_idx_ceil = z_idx_floor + 1

        # Gather four corner values for bilinear interpolation
        # Table shape: (n_offsets, n_depths)
        t00 = self._table[h_idx_floor, z_idx_floor]
        t01 = self._table[h_idx_floor, z_idx_ceil]
        t10 = self._table[h_idx_ceil, z_idx_floor]
        t11 = self._table[h_idx_ceil, z_idx_ceil]

        # Bilinear interpolation
        t0 = t00 * (1 - z_frac) + t01 * z_frac
        t1 = t10 * (1 - z_frac) + t11 * z_frac
        result = t0 * (1 - h_frac) + t1 * h_frac

        return result

    def lookup_batch_2way(
        self,
        h_src: torch.Tensor,
        h_rcv: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute two-way traveltime: t_src + t_rcv.

        Args:
            h_src: Horizontal distances from source to image points
            h_rcv: Horizontal distances from receiver to image points
            z: Depths

        Returns:
            Two-way traveltimes
        """
        t_src = self.lookup_batch(h_src, z)
        t_rcv = self.lookup_batch(h_rcv, z)
        return t_src + t_rcv

    @property
    def memory_mb(self) -> float:
        """Memory usage in megabytes."""
        if self._table is None:
            return 0.0
        return self._table.numel() * 4 / (1024 * 1024)  # float32 = 4 bytes

    @property
    def shape(self) -> Tuple[int, int]:
        """Table shape (n_offsets, n_depths)."""
        return (self._n_offsets, self._n_depths)

    def to_device(self, device: torch.device) -> 'TraveltimeLUT':
        """Move table to specified device."""
        if self._table is not None:
            self._table = self._table.to(device)
            self._h_axis = self._h_axis.to(device)
            self._z_axis = self._z_axis.to(device)
        self.device = device
        return self

    def save(self, path: Union[str, Path]) -> None:
        """
        Save LUT to disk.

        Args:
            path: Output file path (will save as .npz)
        """
        if not self._built:
            raise RuntimeError("LUT not built. Call build() first.")

        path = Path(path)

        # Save table and metadata
        np.savez_compressed(
            path,
            table=self._table.cpu().numpy(),
            h_axis=self._h_axis.cpu().numpy(),
            z_axis=self._z_axis.cpu().numpy(),
        )

        # Save metadata as JSON
        metadata = {
            'max_offset': self._max_offset,
            'max_depth': self._max_depth,
            'n_offsets': self._n_offsets,
            'n_depths': self._n_depths,
            'dh': self._dh,
            'dz': self._dz,
            'velocity_type': self._velocity_type,
            'v0': self._v0,
            'gradient': self._gradient,
            'build_time': self._build_time,
        }

        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved LUT to {path} ({self.memory_mb:.1f} MB)")

    def load(self, path: Union[str, Path]) -> 'TraveltimeLUT':
        """
        Load LUT from disk.

        Args:
            path: Input file path (.npz)

        Returns:
            self for method chaining
        """
        path = Path(path)

        # Load arrays
        data = np.load(path)
        self._table = torch.from_numpy(data['table']).to(self.device)
        self._h_axis = torch.from_numpy(data['h_axis']).to(self.device)
        self._z_axis = torch.from_numpy(data['z_axis']).to(self.device)

        # Load metadata
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            self._max_offset = metadata['max_offset']
            self._max_depth = metadata['max_depth']
            self._n_offsets = metadata['n_offsets']
            self._n_depths = metadata['n_depths']
            self._dh = metadata['dh']
            self._dz = metadata['dz']
            self._velocity_type = metadata['velocity_type']
            self._v0 = metadata['v0']
            self._gradient = metadata['gradient']
            self._build_time = metadata.get('build_time', 0.0)
        else:
            # Infer from arrays
            self._n_offsets, self._n_depths = self._table.shape
            self._max_offset = float(self._h_axis[-1])
            self._max_depth = float(self._z_axis[-1])
            self._dh = self._max_offset / (self._n_offsets - 1)
            self._dz = self._max_depth / (self._n_depths - 1)

        self._built = True

        logger.info(f"Loaded LUT from {path} ({self.memory_mb:.1f} MB)")

        return self

    def get_stats(self) -> dict:
        """Get LUT statistics."""
        return {
            'built': self._built,
            'shape': self.shape,
            'memory_mb': self.memory_mb,
            'max_offset': self._max_offset,
            'max_depth': self._max_depth,
            'velocity_type': self._velocity_type,
            'v0': self._v0,
            'gradient': self._gradient,
            'device': str(self.device),
            'build_time': self._build_time,
        }


def create_traveltime_lut(
    velocity_model,
    config,
    device: Optional[torch.device] = None,
) -> TraveltimeLUT:
    """
    Factory function to create TraveltimeLUT from velocity model and config.

    Args:
        velocity_model: VelocityModel instance
        config: MigrationConfig instance
        device: Torch device

    Returns:
        Configured TraveltimeLUT
    """
    lut = TraveltimeLUT(device=device)

    # Determine max depth from output grid
    max_time = config.output_grid.t_max
    max_depth = max_time * velocity_model.v0 / 2.0  # Two-way time to one-way depth

    # Build table
    lut.build(
        velocity=velocity_model.v0,
        max_offset=config.max_aperture_m,
        max_depth=max_depth,
        n_offsets=500,
        n_depths=1000,
        gradient=getattr(velocity_model, 'gradient', 0.0),
    )

    return lut
