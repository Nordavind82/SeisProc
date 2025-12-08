"""
3D Seismic Volume Data Model

Container for 3D seismic volumes (time × inline × crossline) with slice accessors.
Follows the same patterns as SeismicData for 2D gathers.
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class SeismicVolume:
    """
    Container for 3D seismic volume data.

    Attributes:
        data: 3D array (n_samples, n_inlines, n_xlines) of seismic amplitudes
        dt: Sample interval in seconds (e.g., 0.002 for 2ms)
        dx: Inline spacing in meters
        dy: Crossline spacing in meters
        metadata: Additional metadata (survey info, processing history, etc.)
    """
    data: np.ndarray
    dt: float  # seconds
    dx: float  # meters
    dy: float  # meters
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data integrity."""
        if self.data.ndim != 3:
            raise ValueError(f"Data must be 3D array, got shape {self.data.shape}")

        if self.dt <= 0:
            raise ValueError(f"Sample interval dt must be positive, got {self.dt}")

        if self.dx <= 0:
            raise ValueError(f"Inline spacing dx must be positive, got {self.dx}")

        if self.dy <= 0:
            raise ValueError(f"Crossline spacing dy must be positive, got {self.dy}")

        # Ensure data is float32 for processing
        if self.data.dtype != np.float32:
            object.__setattr__(self, 'data', self.data.astype(np.float32))

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape (nt, nx, ny)."""
        return self.data.shape

    @property
    def nt(self) -> int:
        """Number of time samples."""
        return self.data.shape[0]

    @property
    def nx(self) -> int:
        """Number of inlines (X direction)."""
        return self.data.shape[1]

    @property
    def ny(self) -> int:
        """Number of crosslines (Y direction)."""
        return self.data.shape[2]

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return (self.nt - 1) * self.dt * 1000.0

    @property
    def f_nyquist(self) -> float:
        """Nyquist frequency in Hz."""
        return 1.0 / (2.0 * self.dt)

    @property
    def kx_nyquist(self) -> float:
        """Nyquist wavenumber in X direction (cycles/m)."""
        return 1.0 / (2.0 * self.dx)

    @property
    def ky_nyquist(self) -> float:
        """Nyquist wavenumber in Y direction (cycles/m)."""
        return 1.0 / (2.0 * self.dy)

    # =========================================================================
    # Slice Accessors
    # =========================================================================

    def time_slice(self, t_idx: int) -> np.ndarray:
        """
        Extract horizontal time slice (X-Y plane at fixed time).

        Args:
            t_idx: Time sample index

        Returns:
            2D array (nx, ny)
        """
        if not 0 <= t_idx < self.nt:
            raise IndexError(f"Time index {t_idx} out of range [0, {self.nt})")
        return self.data[t_idx, :, :]

    def inline_slice(self, y_idx: int) -> np.ndarray:
        """
        Extract inline section (T-X plane at fixed Y/crossline).

        Args:
            y_idx: Crossline index

        Returns:
            2D array (nt, nx)
        """
        if not 0 <= y_idx < self.ny:
            raise IndexError(f"Crossline index {y_idx} out of range [0, {self.ny})")
        return self.data[:, :, y_idx]

    def xline_slice(self, x_idx: int) -> np.ndarray:
        """
        Extract crossline section (T-Y plane at fixed X/inline).

        Args:
            x_idx: Inline index

        Returns:
            2D array (nt, ny)
        """
        if not 0 <= x_idx < self.nx:
            raise IndexError(f"Inline index {x_idx} out of range [0, {self.nx})")
        return self.data[:, x_idx, :]

    # =========================================================================
    # Axis Generators
    # =========================================================================

    def time_axis_ms(self) -> np.ndarray:
        """Get time axis in milliseconds."""
        return np.arange(self.nt) * self.dt * 1000.0

    def time_axis_s(self) -> np.ndarray:
        """Get time axis in seconds."""
        return np.arange(self.nt) * self.dt

    def inline_axis(self) -> np.ndarray:
        """Get inline axis (0 to nx-1)."""
        return np.arange(self.nx)

    def xline_axis(self) -> np.ndarray:
        """Get crossline axis (0 to ny-1)."""
        return np.arange(self.ny)

    def x_axis_m(self) -> np.ndarray:
        """Get X axis in meters (inline direction)."""
        return np.arange(self.nx) * self.dx

    def y_axis_m(self) -> np.ndarray:
        """Get Y axis in meters (crossline direction)."""
        return np.arange(self.ny) * self.dy

    # =========================================================================
    # Memory and Utilities
    # =========================================================================

    def memory_bytes(self) -> int:
        """Return memory size in bytes."""
        return self.data.nbytes

    def memory_mb(self) -> float:
        """Return memory size in megabytes."""
        return self.data.nbytes / (1024 * 1024)

    def memory_gb(self) -> float:
        """Return memory size in gigabytes."""
        return self.data.nbytes / (1024 * 1024 * 1024)

    def copy(self) -> 'SeismicVolume':
        """Create a deep copy of this volume."""
        return SeismicVolume(
            data=self.data.copy(),
            dt=self.dt,
            dx=self.dx,
            dy=self.dy,
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        return (
            f"SeismicVolume(shape={self.shape}, "
            f"dt={self.dt*1000:.1f}ms, dx={self.dx:.1f}m, dy={self.dy:.1f}m, "
            f"size={self.memory_mb():.1f}MB)"
        )


def create_synthetic_volume(
    nt: int = 256,
    nx: int = 64,
    ny: int = 64,
    dt: float = 0.004,
    dx: float = 25.0,
    dy: float = 25.0,
    add_events: bool = True
) -> SeismicVolume:
    """
    Create synthetic 3D volume for testing.

    Args:
        nt: Number of time samples
        nx: Number of inlines
        ny: Number of crosslines
        dt: Sample interval in seconds
        dx: Inline spacing in meters
        dy: Crossline spacing in meters
        add_events: If True, add synthetic seismic events

    Returns:
        SeismicVolume with synthetic data
    """
    data = np.random.randn(nt, nx, ny).astype(np.float32) * 0.1

    if add_events:
        # Add some planar events at different velocities
        t_axis = np.arange(nt) * dt
        x_axis = np.arange(nx) * dx
        y_axis = np.arange(ny) * dy

        # Event 1: Flat reflector at 0.5s
        t0 = 0.5
        for ix in range(nx):
            for iy in range(ny):
                t_event = t0
                it = int(t_event / dt)
                if 0 <= it < nt:
                    data[it, ix, iy] += 1.0

        # Event 2: Dipping event (ground roll - low velocity)
        v_groundroll = 500.0  # m/s
        t0 = 0.2
        for ix in range(nx):
            for iy in range(ny):
                offset = np.sqrt((x_axis[ix] - x_axis[nx//2])**2 +
                                 (y_axis[iy] - y_axis[ny//2])**2)
                t_event = t0 + offset / v_groundroll
                it = int(t_event / dt)
                if 0 <= it < nt:
                    # Add wavelet
                    for dt_off in range(-5, 6):
                        if 0 <= it + dt_off < nt:
                            w = np.exp(-0.5 * (dt_off / 2.0)**2)
                            data[it + dt_off, ix, iy] += 0.5 * w

        # Event 3: Reflection (high velocity)
        v_reflection = 3000.0  # m/s
        t0 = 0.8
        for ix in range(nx):
            for iy in range(ny):
                offset = np.sqrt((x_axis[ix] - x_axis[nx//2])**2 +
                                 (y_axis[iy] - y_axis[ny//2])**2)
                t_event = np.sqrt(t0**2 + (offset / v_reflection)**2)
                it = int(t_event / dt)
                if 0 <= it < nt:
                    data[it, ix, iy] += 0.8

    return SeismicVolume(
        data=data,
        dt=dt,
        dx=dx,
        dy=dy,
        metadata={'synthetic': True}
    )
