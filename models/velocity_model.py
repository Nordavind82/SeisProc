"""
Velocity Model for Kirchhoff Migration

Supports multiple velocity model types:
- Constant velocity (scalar)
- 1D velocity function v(z) or v(t)
- 2D velocity field v(x,z)
- 3D velocity cube v(x,y,z)

Includes utilities for velocity interpolation and gradient computation.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VelocityType(Enum):
    """Type of velocity model."""
    CONSTANT = "constant"
    V_OF_Z = "v_of_z"       # 1D: velocity as function of depth/time
    V_OF_XZ = "v_of_xz"     # 2D: velocity field
    V_OF_XYZ = "v_of_xyz"   # 3D: velocity cube


@dataclass
class VelocityModel:
    """
    Velocity model for seismic migration.

    Supports multiple representations:
    - Constant: single velocity value
    - 1D: v(z) or v(t) array with corresponding axis
    - 2D: v(x,z) grid (future)
    - 3D: v(x,y,z) cube (future)

    Attributes:
        data: Velocity values in m/s. Can be:
              - float: constant velocity
              - 1D array: v(z) function
              - 2D array: v(x,z) field
              - 3D array: v(x,y,z) cube
        z_axis: Depth/time axis for 1D+ models (meters or seconds)
        x_axis: X axis for 2D/3D models (meters)
        y_axis: Y axis for 3D models (meters)
        is_time: If True, z_axis is two-way time (seconds);
                 if False, z_axis is depth (meters)
        gradient: Vertical velocity gradient dV/dz (m/s per m or per s)
                  Used for curved ray calculations. If None, computed from data.
        v0: Reference velocity at z=0 (m/s). If None, computed from data.
        metadata: Additional metadata (source file, processing history, etc.)
    """
    data: Union[float, np.ndarray]
    z_axis: Optional[np.ndarray] = None
    x_axis: Optional[np.ndarray] = None
    y_axis: Optional[np.ndarray] = None
    is_time: bool = True  # True = time migration, False = depth migration
    gradient: Optional[float] = None
    v0: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and process velocity model."""
        # Convert scalar to float
        if isinstance(self.data, (int, float)):
            self.data = float(self.data)
            self._type = VelocityType.CONSTANT
        elif isinstance(self.data, np.ndarray):
            if self.data.ndim == 1:
                self._type = VelocityType.V_OF_Z
                if self.z_axis is None:
                    raise ValueError("z_axis required for 1D velocity model")
                if len(self.z_axis) != len(self.data):
                    raise ValueError(
                        f"z_axis length ({len(self.z_axis)}) must match "
                        f"data length ({len(self.data)})"
                    )
            elif self.data.ndim == 2:
                self._type = VelocityType.V_OF_XZ
                if self.z_axis is None or self.x_axis is None:
                    raise ValueError("z_axis and x_axis required for 2D velocity model")
            elif self.data.ndim == 3:
                self._type = VelocityType.V_OF_XYZ
                if self.z_axis is None or self.x_axis is None or self.y_axis is None:
                    raise ValueError("z_axis, x_axis, y_axis required for 3D velocity model")
            else:
                raise ValueError(f"Unsupported velocity data dimension: {self.data.ndim}")

            # Ensure float32 for GPU efficiency
            if self.data.dtype != np.float32:
                self.data = self.data.astype(np.float32)
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}")

        # Compute v0 if not provided
        if self.v0 is None:
            if self._type == VelocityType.CONSTANT:
                self.v0 = self.data
            elif self._type == VelocityType.V_OF_Z:
                # Interpolate to z=0 or use first value
                if self.z_axis[0] == 0:
                    self.v0 = float(self.data[0])
                else:
                    self.v0 = float(np.interp(0, self.z_axis, self.data))

        # Compute gradient if not provided and model is 1D
        if self.gradient is None and self._type == VelocityType.V_OF_Z:
            self._compute_gradient()

        # Validate velocity values
        self._validate_velocities()

    def _validate_velocities(self):
        """Validate that velocities are physically reasonable."""
        MIN_VELOCITY = 0.1  # Minimum allowed velocity in m/s

        if self._type == VelocityType.CONSTANT:
            if self.data < MIN_VELOCITY:
                raise ValueError(f"Velocity must be >= {MIN_VELOCITY} m/s, got {self.data}")
            if self.data < 300 or self.data > 10000:
                logger.warning(
                    f"Velocity {self.data} m/s outside typical range [300, 10000] m/s"
                )
        else:
            v_min = np.min(self.data)
            v_max = np.max(self.data)
            # Clip values below minimum instead of raising error
            if v_min < MIN_VELOCITY:
                logger.warning(
                    f"Clipping {np.sum(self.data < MIN_VELOCITY)} velocity values "
                    f"below {MIN_VELOCITY} m/s (min was {v_min})"
                )
                self.data = np.clip(self.data, MIN_VELOCITY, None)
                v_min = MIN_VELOCITY
            if v_min < 300 or v_max > 10000:
                logger.warning(
                    f"Velocity range [{v_min}, {v_max}] m/s outside typical "
                    f"range [300, 10000] m/s"
                )

    def _compute_gradient(self):
        """Compute average velocity gradient from 1D model."""
        if self._type != VelocityType.V_OF_Z:
            return

        # Linear fit to get average gradient
        if len(self.z_axis) >= 2:
            coeffs = np.polyfit(self.z_axis, self.data, 1)
            self.gradient = float(coeffs[0])  # slope = dV/dz
            logger.debug(f"Computed velocity gradient: {self.gradient:.4f} m/s per unit")

    @property
    def velocity_type(self) -> VelocityType:
        """Get velocity model type."""
        return self._type

    @property
    def is_constant(self) -> bool:
        """Check if velocity is constant."""
        return self._type == VelocityType.CONSTANT

    @property
    def has_gradient(self) -> bool:
        """Check if velocity model has a gradient (non-constant)."""
        return self.gradient is not None and abs(self.gradient) > 1e-6

    @property
    def z_unit(self) -> str:
        """Get z-axis unit string."""
        return "s" if self.is_time else "m"

    @property
    def gradient_unit(self) -> str:
        """Get gradient unit string."""
        return "m/s/s" if self.is_time else "1/s"

    def get_velocity_at(
        self,
        z: Union[float, np.ndarray],
        x: Optional[Union[float, np.ndarray]] = None,
        y: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Get velocity at specified location(s).

        Args:
            z: Depth/time value(s)
            x: X coordinate(s) for 2D/3D models
            y: Y coordinate(s) for 3D models

        Returns:
            Velocity value(s) at specified location(s)
        """
        if self._type == VelocityType.CONSTANT:
            if isinstance(z, np.ndarray):
                return np.full_like(z, self.data, dtype=np.float32)
            return self.data

        elif self._type == VelocityType.V_OF_Z:
            return np.interp(z, self.z_axis, self.data).astype(np.float32)

        elif self._type == VelocityType.V_OF_XZ:
            if x is None:
                raise ValueError("x coordinate required for 2D velocity model")
            # Bilinear interpolation
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (self.z_axis, self.x_axis),
                self.data,
                bounds_error=False,
                fill_value=None
            )
            z_arr = np.atleast_1d(z)
            x_arr = np.atleast_1d(x)
            # Broadcast x to match z length if x is scalar
            if x_arr.size == 1 and z_arr.size > 1:
                x_arr = np.full_like(z_arr, x_arr[0])
            points = np.column_stack([z_arr, x_arr])
            return interp(points).astype(np.float32)

        elif self._type == VelocityType.V_OF_XYZ:
            if x is None or y is None:
                raise ValueError("x, y coordinates required for 3D velocity model")
            # Trilinear interpolation
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (self.z_axis, self.x_axis, self.y_axis),
                self.data,
                bounds_error=False,
                fill_value=None
            )
            z_arr = np.atleast_1d(z)
            x_arr = np.atleast_1d(x)
            y_arr = np.atleast_1d(y)
            # Broadcast scalars to match z length
            if x_arr.size == 1 and z_arr.size > 1:
                x_arr = np.full_like(z_arr, x_arr[0])
            if y_arr.size == 1 and z_arr.size > 1:
                y_arr = np.full_like(z_arr, y_arr[0])
            points = np.column_stack([z_arr, x_arr, y_arr])
            return interp(points).astype(np.float32)

    def get_effective_velocity(
        self,
        z: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Get effective (average) velocity from surface to depth z.

        For straight-ray time migration, effective velocity is:
        V_eff = z / integral(dz'/V(z'))

        Args:
            z: Depth/time value(s)

        Returns:
            Effective velocity to depth z
        """
        if self._type == VelocityType.CONSTANT:
            if isinstance(z, np.ndarray):
                return np.full_like(z, self.data, dtype=np.float32)
            return self.data

        elif self._type == VelocityType.V_OF_Z:
            z = np.atleast_1d(z)
            v_eff = np.zeros_like(z, dtype=np.float32)

            for i, zi in enumerate(z):
                if zi <= 0:
                    v_eff[i] = self.v0
                else:
                    # Numerical integration of slowness
                    z_interp = np.linspace(0, zi, 100)
                    v_interp = np.interp(z_interp, self.z_axis, self.data)
                    slowness_integral = np.trapz(1.0 / v_interp, z_interp)
                    v_eff[i] = zi / slowness_integral if slowness_integral > 0 else self.v0

            return v_eff[0] if len(v_eff) == 1 else v_eff

        else:
            raise NotImplementedError(
                f"Effective velocity not implemented for {self._type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize velocity model to dictionary."""
        result = {
            'velocity_type': self._type.value,
            'is_time': self.is_time,
            'gradient': self.gradient,
            'v0': self.v0,
            'metadata': self.metadata.copy(),
        }

        if self._type == VelocityType.CONSTANT:
            result['data'] = self.data
        else:
            result['data'] = self.data.tolist()
            result['z_axis'] = self.z_axis.tolist()
            if self.x_axis is not None:
                result['x_axis'] = self.x_axis.tolist()
            if self.y_axis is not None:
                result['y_axis'] = self.y_axis.tolist()

        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VelocityModel':
        """Deserialize velocity model from dictionary."""
        velocity_type = VelocityType(d['velocity_type'])

        if velocity_type == VelocityType.CONSTANT:
            data = d['data']
        else:
            data = np.array(d['data'], dtype=np.float32)

        z_axis = np.array(d['z_axis']) if 'z_axis' in d and d['z_axis'] else None
        x_axis = np.array(d['x_axis']) if 'x_axis' in d and d['x_axis'] else None
        y_axis = np.array(d['y_axis']) if 'y_axis' in d and d['y_axis'] else None

        return cls(
            data=data,
            z_axis=z_axis,
            x_axis=x_axis,
            y_axis=y_axis,
            is_time=d.get('is_time', True),
            gradient=d.get('gradient'),
            v0=d.get('v0'),
            metadata=d.get('metadata', {}),
        )

    def copy(self) -> 'VelocityModel':
        """Create a deep copy of this velocity model."""
        if self._type == VelocityType.CONSTANT:
            data_copy = self.data
        else:
            data_copy = self.data.copy()

        return VelocityModel(
            data=data_copy,
            z_axis=self.z_axis.copy() if self.z_axis is not None else None,
            x_axis=self.x_axis.copy() if self.x_axis is not None else None,
            y_axis=self.y_axis.copy() if self.y_axis is not None else None,
            is_time=self.is_time,
            gradient=self.gradient,
            v0=self.v0,
            metadata=self.metadata.copy(),
        )

    def __repr__(self) -> str:
        if self._type == VelocityType.CONSTANT:
            return f"VelocityModel(constant={self.data} m/s)"
        elif self._type == VelocityType.V_OF_Z:
            return (
                f"VelocityModel(v_of_z, n={len(self.data)}, "
                f"v0={self.v0:.0f} m/s, gradient={self.gradient:.2f} {self.gradient_unit})"
            )
        else:
            return f"VelocityModel({self._type.value}, shape={self.data.shape})"


# =============================================================================
# Factory Functions
# =============================================================================

def create_constant_velocity(velocity: float, is_time: bool = True) -> VelocityModel:
    """
    Create constant velocity model.

    Args:
        velocity: Velocity in m/s
        is_time: If True, model is for time migration

    Returns:
        VelocityModel with constant velocity
    """
    return VelocityModel(data=velocity, is_time=is_time)


def create_linear_gradient_velocity(
    v0: float,
    gradient: float,
    z_max: float,
    dz: float = 0.004,
    is_time: bool = True,
) -> VelocityModel:
    """
    Create 1D velocity model with linear gradient.

    v(z) = v0 + gradient * z

    Args:
        v0: Velocity at z=0 (m/s)
        gradient: Velocity gradient (m/s per unit z)
        z_max: Maximum z value (seconds for time, meters for depth)
        dz: Z sampling interval
        is_time: If True, z is two-way time in seconds

    Returns:
        VelocityModel with linear gradient
    """
    z_axis = np.arange(0, z_max + dz, dz, dtype=np.float32)
    data = (v0 + gradient * z_axis).astype(np.float32)

    return VelocityModel(
        data=data,
        z_axis=z_axis,
        is_time=is_time,
        gradient=gradient,
        v0=v0,
        metadata={'model_type': 'linear_gradient'},
    )


def create_from_rms_velocity(
    t_axis: np.ndarray,
    v_rms: np.ndarray,
    is_time: bool = True,
) -> VelocityModel:
    """
    Create velocity model from RMS velocity function.

    Args:
        t_axis: Two-way time axis (seconds)
        v_rms: RMS velocity values (m/s)
        is_time: If True, z-axis is time (default for RMS input)

    Returns:
        VelocityModel with RMS velocities
    """
    return VelocityModel(
        data=np.array(v_rms, dtype=np.float32),
        z_axis=np.array(t_axis, dtype=np.float32),
        is_time=is_time,
        metadata={'velocity_type': 'rms'},
    )


def rms_to_interval_velocity(
    t_axis: np.ndarray,
    v_rms: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert RMS velocity to interval velocity using Dix equation.

    V_int(n) = sqrt((V_rms(n)^2 * t(n) - V_rms(n-1)^2 * t(n-1)) / (t(n) - t(n-1)))

    Args:
        t_axis: Two-way time axis (seconds)
        v_rms: RMS velocity values (m/s)

    Returns:
        Tuple of (t_axis, v_interval)
    """
    n = len(t_axis)
    v_int = np.zeros(n, dtype=np.float32)

    # First interval uses RMS directly
    v_int[0] = v_rms[0]

    for i in range(1, n):
        dt = t_axis[i] - t_axis[i-1]
        if dt > 0:
            numerator = v_rms[i]**2 * t_axis[i] - v_rms[i-1]**2 * t_axis[i-1]
            if numerator > 0:
                v_int[i] = np.sqrt(numerator / dt)
            else:
                # Fallback if Dix gives negative (can happen with noisy picks)
                v_int[i] = v_rms[i]
                logger.warning(f"Dix equation gave negative value at t={t_axis[i]:.3f}s")
        else:
            v_int[i] = v_rms[i]

    return t_axis, v_int


def interval_to_rms_velocity(
    t_axis: np.ndarray,
    v_int: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert interval velocity to RMS velocity.

    V_rms(n) = sqrt(sum(V_int(i)^2 * dt(i)) / t(n))

    Args:
        t_axis: Two-way time axis (seconds)
        v_int: Interval velocity values (m/s)

    Returns:
        Tuple of (t_axis, v_rms)
    """
    n = len(t_axis)
    v_rms = np.zeros(n, dtype=np.float32)

    # First point
    v_rms[0] = v_int[0]

    cumulative_sum = v_int[0]**2 * t_axis[0] if t_axis[0] > 0 else 0

    for i in range(1, n):
        dt = t_axis[i] - t_axis[i-1]
        cumulative_sum += v_int[i]**2 * dt
        if t_axis[i] > 0:
            v_rms[i] = np.sqrt(cumulative_sum / t_axis[i])
        else:
            v_rms[i] = v_int[i]

    return t_axis, v_rms


def create_2d_velocity(
    data: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    is_time: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> VelocityModel:
    """
    Create 2D velocity model v(x,z).

    Args:
        data: 2D velocity array (n_z, n_x) in m/s
        z_axis: Z (depth/time) axis
        x_axis: X (horizontal) axis in meters
        is_time: If True, z is two-way time in seconds
        metadata: Additional metadata

    Returns:
        VelocityModel with 2D velocity field
    """
    data = np.asarray(data, dtype=np.float32)
    z_axis = np.asarray(z_axis, dtype=np.float32)
    x_axis = np.asarray(x_axis, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    if data.shape[0] != len(z_axis):
        raise ValueError(f"Data z-dimension ({data.shape[0]}) must match z_axis length ({len(z_axis)})")
    if data.shape[1] != len(x_axis):
        raise ValueError(f"Data x-dimension ({data.shape[1]}) must match x_axis length ({len(x_axis)})")

    return VelocityModel(
        data=data,
        z_axis=z_axis,
        x_axis=x_axis,
        is_time=is_time,
        metadata=metadata or {'model_type': '2d_field'},
    )


def create_3d_velocity(
    data: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    is_time: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> VelocityModel:
    """
    Create 3D velocity model v(x,y,z).

    Args:
        data: 3D velocity array (n_z, n_x, n_y) in m/s
        z_axis: Z (depth/time) axis
        x_axis: X (horizontal) axis in meters
        y_axis: Y (horizontal) axis in meters
        is_time: If True, z is two-way time in seconds
        metadata: Additional metadata

    Returns:
        VelocityModel with 3D velocity cube
    """
    data = np.asarray(data, dtype=np.float32)
    z_axis = np.asarray(z_axis, dtype=np.float32)
    x_axis = np.asarray(x_axis, dtype=np.float32)
    y_axis = np.asarray(y_axis, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Data must be 3D, got shape {data.shape}")
    if data.shape[0] != len(z_axis):
        raise ValueError(f"Data z-dimension ({data.shape[0]}) must match z_axis length ({len(z_axis)})")
    if data.shape[1] != len(x_axis):
        raise ValueError(f"Data x-dimension ({data.shape[1]}) must match x_axis length ({len(x_axis)})")
    if data.shape[2] != len(y_axis):
        raise ValueError(f"Data y-dimension ({data.shape[2]}) must match y_axis length ({len(y_axis)})")

    return VelocityModel(
        data=data,
        z_axis=z_axis,
        x_axis=x_axis,
        y_axis=y_axis,
        is_time=is_time,
        metadata=metadata or {'model_type': '3d_cube'},
    )


def create_2d_gradient_velocity(
    v0: float,
    z_gradient: float,
    x_gradient: float,
    z_max: float,
    x_max: float,
    dz: float = 0.004,
    dx: float = 25.0,
    is_time: bool = True,
) -> VelocityModel:
    """
    Create 2D velocity model with linear gradients in z and x.

    v(x,z) = v0 + z_gradient * z + x_gradient * x

    Args:
        v0: Velocity at origin (m/s)
        z_gradient: Vertical gradient (m/s per unit z)
        x_gradient: Horizontal gradient (m/s per meter)
        z_max: Maximum z value
        x_max: Maximum x value (meters)
        dz: Z sampling
        dx: X sampling (meters)
        is_time: If True, z is time in seconds

    Returns:
        VelocityModel with 2D linear gradient
    """
    z_axis = np.arange(0, z_max + dz, dz, dtype=np.float32)
    x_axis = np.arange(0, x_max + dx, dx, dtype=np.float32)

    # Create 2D grid
    zz, xx = np.meshgrid(z_axis, x_axis, indexing='ij')
    data = (v0 + z_gradient * zz + x_gradient * xx).astype(np.float32)

    return VelocityModel(
        data=data,
        z_axis=z_axis,
        x_axis=x_axis,
        is_time=is_time,
        metadata={
            'model_type': '2d_linear_gradient',
            'v0': v0,
            'z_gradient': z_gradient,
            'x_gradient': x_gradient,
        },
    )


def create_layered_velocity(
    layer_depths: np.ndarray,
    layer_velocities: np.ndarray,
    z_max: float,
    dz: float = 0.004,
    is_time: bool = True,
) -> VelocityModel:
    """
    Create 1D layered velocity model.

    Args:
        layer_depths: Top depth of each layer (first should be 0)
        layer_velocities: Velocity of each layer (m/s)
        z_max: Maximum z value
        dz: Z sampling interval
        is_time: If True, z is time in seconds

    Returns:
        VelocityModel with step-wise constant layers
    """
    layer_depths = np.asarray(layer_depths)
    layer_velocities = np.asarray(layer_velocities)

    if len(layer_depths) != len(layer_velocities):
        raise ValueError("layer_depths and layer_velocities must have same length")

    z_axis = np.arange(0, z_max + dz, dz, dtype=np.float32)
    data = np.zeros_like(z_axis)

    # Assign velocities layer by layer
    for i in range(len(layer_depths)):
        if i < len(layer_depths) - 1:
            mask = (z_axis >= layer_depths[i]) & (z_axis < layer_depths[i + 1])
        else:
            mask = z_axis >= layer_depths[i]
        data[mask] = layer_velocities[i]

    return VelocityModel(
        data=data.astype(np.float32),
        z_axis=z_axis,
        is_time=is_time,
        metadata={
            'model_type': 'layered',
            'n_layers': len(layer_velocities),
        },
    )
