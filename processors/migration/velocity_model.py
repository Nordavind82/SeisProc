"""
Velocity Model for PSTM.

Handles different velocity types and computes RMS velocity for time-domain migration.

Supported velocity types:
- Constant: v(z) = v0
- Linear gradient: v(z) = v0 + k*z
- From file: v(z) from 1D or 3D velocity model

For time-domain migration, we need RMS (Root Mean Square) velocity which
accounts for the effective velocity through the overburden.

RMS velocity definition:
    v_rms(t)² = (1/t) * integral[0 to t] v(τ)² dτ

For constant velocity: v_rms = v0
For linear gradient v(z) = v0 + k*z: v_rms(t) has analytical solution
"""

import numpy as np
import torch
from typing import Union, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VelocityModel:
    """
    Velocity model for migration.

    Attributes:
        velocity_type: 'constant', 'gradient', or 'file'
        v0: Surface velocity (m/s)
        gradient: Velocity gradient (1/s), only for 'gradient' type
        velocity_array: 1D or 3D velocity array, only for 'file' type
        z_axis: Depth axis for velocity_array
    """
    velocity_type: str
    v0: float
    gradient: float = 0.0
    velocity_array: Optional[np.ndarray] = None
    z_axis: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate and precompute."""
        if self.velocity_type not in ('constant', 'gradient', 'file'):
            raise ValueError(f"Unknown velocity type: {self.velocity_type}")

        if self.velocity_type == 'file' and self.velocity_array is None:
            raise ValueError("velocity_array required for 'file' type")

        # Precompute for gradient model
        if self.velocity_type == 'gradient' and self.gradient != 0:
            # For v(z) = v0 + k*z, compute useful constants
            self._k = self.gradient
            self._v0 = self.v0

    def interval_velocity(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Get interval velocity at depth z.

        Args:
            z: Depth in meters (can be scalar, numpy array, or torch tensor)

        Returns:
            Interval velocity at depth z
        """
        if self.velocity_type == 'constant':
            if isinstance(z, torch.Tensor):
                return torch.full_like(z, self.v0)
            elif isinstance(z, np.ndarray):
                return np.full_like(z, self.v0)
            else:
                return self.v0

        elif self.velocity_type == 'gradient':
            return self.v0 + self.gradient * z

        else:  # file
            # Interpolate from velocity array
            if isinstance(z, torch.Tensor):
                z_np = z.cpu().numpy()
                v_np = np.interp(z_np, self.z_axis, self.velocity_array)
                return torch.from_numpy(v_np).to(z.device)
            else:
                return np.interp(z, self.z_axis, self.velocity_array)

    def rms_velocity_at_time(self, t_ms: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute RMS velocity at two-way time t.

        The RMS velocity is defined as:
            v_rms(t)² = (1/t) * integral[0 to t] v(τ)² dτ

        For constant velocity: v_rms = v0
        For linear gradient: analytical formula exists
        For arbitrary v(z): numerical integration

        Args:
            t_ms: Two-way time in milliseconds

        Returns:
            RMS velocity at time t (m/s)
        """
        if self.velocity_type == 'constant':
            # v_rms = v0 for constant velocity
            if isinstance(t_ms, torch.Tensor):
                return torch.full_like(t_ms, self.v0)
            elif isinstance(t_ms, np.ndarray):
                return np.full_like(t_ms, self.v0, dtype=np.float32)
            else:
                return self.v0

        elif self.velocity_type == 'gradient':
            return self._rms_velocity_gradient(t_ms)

        else:  # file
            return self._rms_velocity_numerical(t_ms)

    def _rms_velocity_gradient(self, t_ms: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        RMS velocity for linear gradient v(z) = v0 + k*z.

        For linear gradient, the RMS velocity has an analytical solution.

        Given:
            v(z) = v0 + k*z
            z(t) = (v0/k) * (exp(k*t*v0/2) - 1)  [approximate for small k*z]

        The RMS velocity approximation (valid for k*z << v0):
            v_rms(t) ≈ v0 * sqrt(1 + (k*v0*t_s)/3 + (k*v0*t_s)²/5)

        where t_s = t_ms/1000 (time in seconds) and t is one-way time.
        For two-way time, use t_owt = t_ms/2000.

        More accurate formula using depth:
            z = v0 * t_owt + 0.5 * k * (v0 * t_owt)²  (approximate depth at time t)
            v_rms² = v0² + v0*k*z + (k*z)²/3
        """
        v0 = self.v0
        k = self.gradient

        # Handle edge case of zero gradient
        if abs(k) < 1e-10:
            if isinstance(t_ms, torch.Tensor):
                return torch.full_like(t_ms, v0)
            elif isinstance(t_ms, np.ndarray):
                return np.full_like(t_ms, v0, dtype=np.float32)
            else:
                return v0

        # Convert to one-way time in seconds
        t_owt = t_ms / 2000.0  # two-way time ms -> one-way time s

        if isinstance(t_ms, torch.Tensor):
            # Approximate depth at this time (using constant v0 as first approximation)
            # z ≈ v0 * t_owt for small gradients
            # More accurate: solve t = integral(dz/v(z)) iteratively

            # First approximation: z ≈ v0 * t_owt
            z_approx = v0 * t_owt

            # Better approximation accounting for gradient:
            # z = v0*t + 0.5*k*v0²*t² (from integrating dt = dz/v)
            z_better = v0 * t_owt + 0.5 * k * v0 * t_owt * t_owt
            z_better = torch.clamp(z_better, min=0)  # Ensure non-negative

            # RMS velocity formula for linear gradient:
            # v_rms² = (1/z) * integral[0 to z] (v0 + k*z')² dz'
            #        = (1/z) * [v0²*z + v0*k*z² + (k²*z³)/3]
            #        = v0² + v0*k*z + (k²*z²)/3

            # Handle t=0 case
            v_rms_sq = v0 * v0 + v0 * k * z_better + (k * k * z_better * z_better) / 3.0
            v_rms = torch.sqrt(torch.clamp(v_rms_sq, min=v0*v0*0.5))  # Ensure positive

            # At t=0, v_rms should equal v0
            v_rms = torch.where(t_ms < 1.0, torch.full_like(v_rms, v0), v_rms)

            return v_rms

        elif isinstance(t_ms, np.ndarray):
            t_owt = np.asarray(t_owt, dtype=np.float64)

            z_better = v0 * t_owt + 0.5 * k * v0 * t_owt * t_owt
            z_better = np.maximum(z_better, 0)

            v_rms_sq = v0 * v0 + v0 * k * z_better + (k * k * z_better * z_better) / 3.0
            v_rms = np.sqrt(np.maximum(v_rms_sq, v0*v0*0.5))

            v_rms = np.where(t_ms < 1.0, v0, v_rms)

            return v_rms.astype(np.float32)

        else:
            # Scalar
            if t_ms < 1.0:
                return v0

            z_better = v0 * t_owt + 0.5 * k * v0 * t_owt * t_owt
            z_better = max(z_better, 0)

            v_rms_sq = v0 * v0 + v0 * k * z_better + (k * k * z_better * z_better) / 3.0
            v_rms = np.sqrt(max(v_rms_sq, v0*v0*0.5))

            return float(v_rms)

    def _rms_velocity_numerical(self, t_ms: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        RMS velocity by numerical integration for arbitrary v(z).

        Uses trapezoidal integration over depth intervals.
        """
        if self.velocity_array is None or self.z_axis is None:
            raise ValueError("velocity_array and z_axis required for numerical RMS")

        # Build lookup table of v_rms vs time
        # This is computed once and interpolated
        if not hasattr(self, '_vrms_table'):
            self._build_vrms_table()

        # Interpolate from table
        if isinstance(t_ms, torch.Tensor):
            t_np = t_ms.cpu().numpy()
            vrms_np = np.interp(t_np, self._t_table_ms, self._vrms_table)
            return torch.from_numpy(vrms_np.astype(np.float32)).to(t_ms.device)
        else:
            return np.interp(t_ms, self._t_table_ms, self._vrms_table).astype(np.float32)

    def _build_vrms_table(self):
        """Build lookup table of v_rms vs two-way time."""
        z = self.z_axis
        v = self.velocity_array

        # Compute one-way time to each depth
        # t(z) = integral[0 to z] dz'/v(z')
        dz = np.diff(z)
        v_avg = (v[:-1] + v[1:]) / 2  # Average velocity in each interval
        dt_owt = dz / v_avg  # One-way time for each interval
        t_owt = np.concatenate([[0], np.cumsum(dt_owt)])  # Cumulative one-way time

        # Compute v_rms at each depth
        # v_rms(z)² = (1/t) * integral[0 to t] v² dt
        #           = (1/t) * sum(v_i² * dt_i)
        v_sq_dt = np.concatenate([[0], np.cumsum(v_avg**2 * dt_owt)])

        # v_rms² = v_sq_dt / t_owt
        with np.errstate(divide='ignore', invalid='ignore'):
            vrms_sq = v_sq_dt / np.maximum(t_owt, 1e-10)
            vrms_sq[0] = v[0]**2  # At t=0, v_rms = v(0)

        self._vrms_table = np.sqrt(vrms_sq).astype(np.float32)
        self._t_table_ms = t_owt * 2000  # Convert one-way to two-way time in ms

        logger.info(f"Built v_rms table: {len(self._t_table_ms)} points, "
                   f"t_max={self._t_table_ms[-1]:.0f}ms, "
                   f"v_rms range: {self._vrms_table.min():.0f}-{self._vrms_table.max():.0f} m/s")

    def get_summary(self) -> str:
        """Get human-readable summary of velocity model."""
        if self.velocity_type == 'constant':
            return f"Constant velocity: {self.v0:.0f} m/s"
        elif self.velocity_type == 'gradient':
            return f"Linear gradient: v(z) = {self.v0:.0f} + {self.gradient:.4f}*z m/s"
        else:
            return f"From file: {len(self.velocity_array)} points, v0={self.velocity_array[0]:.0f} m/s"


def create_velocity_model(
    velocity_type: str,
    v0: float,
    gradient: float = 0.0,
    velocity_file: str = None,
) -> VelocityModel:
    """
    Factory function to create velocity model from config.

    Args:
        velocity_type: 'constant', 'gradient', or 'file'
        v0: Surface velocity (m/s)
        gradient: Velocity gradient (1/s) for gradient type
        velocity_file: Path to velocity file for file type

    Returns:
        VelocityModel instance
    """
    if velocity_type == 'constant':
        return VelocityModel(
            velocity_type='constant',
            v0=v0,
        )

    elif velocity_type == 'gradient':
        return VelocityModel(
            velocity_type='gradient',
            v0=v0,
            gradient=gradient,
        )

    elif velocity_type == 'file':
        if velocity_file is None:
            raise ValueError("velocity_file required for 'file' type")

        # Load velocity file (assuming simple z,v format for now)
        # TODO: Support more velocity file formats
        data = np.loadtxt(velocity_file)
        z_axis = data[:, 0]
        v_array = data[:, 1]

        return VelocityModel(
            velocity_type='file',
            v0=v_array[0],
            velocity_array=v_array.astype(np.float32),
            z_axis=z_axis.astype(np.float32),
        )

    else:
        raise ValueError(f"Unknown velocity type: {velocity_type}")


def create_velocity_model_from_config(config: dict) -> VelocityModel:
    """
    Create velocity model from wizard config dictionary.

    Args:
        config: Configuration dictionary with velocity_type, velocity_v0, etc.

    Returns:
        VelocityModel instance
    """
    velocity_type = config.get('velocity_type', 'constant')
    v0 = config.get('velocity_v0', 2500.0)
    gradient = config.get('velocity_gradient', 0.0)
    velocity_file = config.get('velocity_file', None)

    return create_velocity_model(
        velocity_type=velocity_type,
        v0=v0,
        gradient=gradient,
        velocity_file=velocity_file,
    )
