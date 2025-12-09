"""
Traveltime Calculator Interface and Implementations

Abstract interface for traveltime computation in migration.
Supports multiple ray-tracing methods:
- Straight ray (isotropic)
- Curved ray (constant gradient)
- VTI anisotropic corrections
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import torch
import logging

from models.velocity_model import VelocityModel

logger = logging.getLogger(__name__)


class TraveltimeCalculator(ABC):
    """
    Abstract base class for traveltime calculation.

    Defines the interface for computing traveltimes from source/receiver
    to subsurface image points.
    """

    def __init__(
        self,
        velocity_model: VelocityModel,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize traveltime calculator.

        Args:
            velocity_model: Velocity model for traveltime computation
            device: Torch device for GPU computation (None = auto-detect)
        """
        self.velocity_model = velocity_model
        self.device = device or self._detect_device()

        logger.info(
            f"Initialized {self.__class__.__name__} on {self.device}"
        )

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    @abstractmethod
    def compute_traveltime(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute one-way traveltime from surface to subsurface point.

        Args:
            x_offset: Horizontal X offset from surface point (meters)
            y_offset: Horizontal Y offset from surface point (meters)
            z_depth: Vertical depth/time to image point

        Returns:
            Traveltime(s) in seconds
        """
        pass

    @abstractmethod
    def compute_traveltime_batch(
        self,
        surface_x: torch.Tensor,    # (n_surface,)
        surface_y: torch.Tensor,    # (n_surface,)
        image_x: torch.Tensor,      # (n_image,)
        image_y: torch.Tensor,      # (n_image,)
        image_z: torch.Tensor,      # (n_z,)
    ) -> torch.Tensor:
        """
        Compute traveltimes for batch of surface-to-image point pairs.

        Args:
            surface_x: Surface point X coordinates
            surface_y: Surface point Y coordinates
            image_x: Image point X coordinates
            image_y: Image point Y coordinates
            image_z: Image point depths/times

        Returns:
            Traveltime tensor of shape (n_z, n_surface, n_image) or similar
        """
        pass

    def compute_total_traveltime(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        receiver_x: torch.Tensor,
        receiver_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total two-way traveltime (source -> image -> receiver).

        Args:
            source_x, source_y: Source coordinates
            receiver_x, receiver_y: Receiver coordinates
            image_x, image_y, image_z: Image point coordinates

        Returns:
            Total traveltime tensor
        """
        # Source to image
        t_src = self.compute_traveltime(
            image_x - source_x,
            image_y - source_y,
            image_z,
        )

        # Image to receiver
        t_rcv = self.compute_traveltime(
            image_x - receiver_x,
            image_y - receiver_y,
            image_z,
        )

        return t_src + t_rcv

    @abstractmethod
    def compute_emergence_angle(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute ray emergence angle at surface (from vertical).

        Used for obliquity factor in amplitude weights.

        Args:
            x_offset: Horizontal X offset
            y_offset: Horizontal Y offset
            z_depth: Depth

        Returns:
            Emergence angle in radians
        """
        pass

    def get_description(self) -> str:
        """Get human-readable description."""
        return f"{self.__class__.__name__}"


class StraightRayTraveltime(TraveltimeCalculator):
    """
    Straight-ray traveltime calculator (isotropic).

    t = sqrt(x^2 + y^2 + z^2) / V

    Supports:
    - Constant velocity
    - 1D velocity v(z) using effective velocity approximation
    """

    def __init__(
        self,
        velocity_model: VelocityModel,
        device: Optional[torch.device] = None,
    ):
        super().__init__(velocity_model, device)

        # Precompute velocity on GPU if 1D model
        if velocity_model.is_constant:
            self._v_constant = velocity_model.data
            self._v_tensor = None
        else:
            self._v_constant = None
            self._v_tensor = torch.from_numpy(
                velocity_model.data
            ).to(self.device)
            self._z_tensor = torch.from_numpy(
                velocity_model.z_axis
            ).to(self.device)

    def compute_traveltime(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute straight-ray traveltime.

        t = sqrt(x^2 + y^2 + z^2) / V_eff(z)
        """
        # Convert to tensors if needed
        if isinstance(x_offset, np.ndarray):
            x_offset = torch.from_numpy(x_offset).to(self.device)
        if isinstance(y_offset, np.ndarray):
            y_offset = torch.from_numpy(y_offset).to(self.device)
        if isinstance(z_depth, np.ndarray):
            z_depth = torch.from_numpy(z_depth).to(self.device)

        # Compute distance
        r = torch.sqrt(x_offset**2 + y_offset**2 + z_depth**2)

        # Get velocity
        if self._v_constant is not None:
            v = self._v_constant
        else:
            # Use effective velocity at z
            v = self._get_effective_velocity(z_depth)

        # Avoid division by zero
        v_safe = torch.clamp(v, min=100.0) if isinstance(v, torch.Tensor) else max(v, 100.0)

        return r / v_safe

    def _get_effective_velocity(self, z: torch.Tensor) -> torch.Tensor:
        """Get effective velocity at depth z."""
        # Linear interpolation of velocity
        v = torch.zeros_like(z)

        # Find indices for interpolation
        z_min = self._z_tensor[0]
        z_max = self._z_tensor[-1]
        dz = self._z_tensor[1] - self._z_tensor[0]

        idx_float = (z - z_min) / dz
        idx_low = torch.clamp(idx_float.long(), 0, len(self._v_tensor) - 2)
        idx_high = idx_low + 1

        frac = idx_float - idx_low.float()

        v = self._v_tensor[idx_low] * (1 - frac) + self._v_tensor[idx_high] * frac

        return v

    def compute_traveltime_batch(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute traveltimes for batch.

        Returns:
            Tensor of shape (n_z, n_surface, n_image_xy)
        """
        n_surface = len(surface_x)
        n_z = len(image_z)

        # Compute horizontal offsets: (n_surface, n_image_xy)
        dx = image_x.unsqueeze(0) - surface_x.unsqueeze(1)  # (n_surface, n_image_xy)
        dy = image_y.unsqueeze(0) - surface_y.unsqueeze(1)

        # Compute distances for each z: (n_z, n_surface, n_image_xy)
        h_squared = dx**2 + dy**2  # (n_surface, n_image_xy)
        z_squared = image_z**2     # (n_z,)

        r = torch.sqrt(
            h_squared.unsqueeze(0) + z_squared.unsqueeze(1).unsqueeze(2)
        )

        # Get velocity
        if self._v_constant is not None:
            t = r / self._v_constant
        else:
            # Velocity varies with z
            v = self._get_effective_velocity(image_z)  # (n_z,)
            t = r / v.unsqueeze(1).unsqueeze(2)

        return t

    def compute_emergence_angle(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute emergence angle for straight ray.

        theta = arctan(horizontal_distance / z)
        """
        if isinstance(x_offset, np.ndarray):
            h = np.sqrt(x_offset**2 + y_offset**2)
            return np.arctan2(h, np.abs(z_depth))
        elif isinstance(x_offset, torch.Tensor):
            h = torch.sqrt(x_offset**2 + y_offset**2)
            return torch.atan2(h, torch.abs(z_depth))
        else:
            import math
            h = math.sqrt(x_offset**2 + y_offset**2)
            return math.atan2(h, abs(z_depth))

    def get_description(self) -> str:
        if self._v_constant is not None:
            return f"StraightRay(V={self._v_constant:.0f} m/s)"
        else:
            return f"StraightRay(V(z), n={len(self._v_tensor)})"


class CurvedRayTraveltime(TraveltimeCalculator):
    """
    Curved-ray traveltime calculator for constant velocity gradient.

    For v(z) = v0 + k*z:
    t = (1/k) * arccosh(1 + k*r^2 / (2*v0*z))

    Degenerates to straight ray as k -> 0.
    """

    def __init__(
        self,
        velocity_model: VelocityModel,
        device: Optional[torch.device] = None,
    ):
        super().__init__(velocity_model, device)

        self.v0 = velocity_model.v0
        self.gradient = velocity_model.gradient or 0.0

        logger.info(
            f"CurvedRay: v0={self.v0:.0f} m/s, "
            f"gradient={self.gradient:.4f} {velocity_model.gradient_unit}"
        )

    def compute_traveltime(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute curved-ray traveltime.

        Uses analytic formula for constant gradient.
        """
        # Convert to tensors
        if isinstance(x_offset, np.ndarray):
            x_offset = torch.from_numpy(x_offset).to(self.device)
        if isinstance(y_offset, np.ndarray):
            y_offset = torch.from_numpy(y_offset).to(self.device)
        if isinstance(z_depth, np.ndarray):
            z_depth = torch.from_numpy(z_depth).to(self.device)

        # Horizontal distance squared
        h_sq = x_offset**2 + y_offset**2

        # Total distance squared
        r_sq = h_sq + z_depth**2

        k = self.gradient
        v0 = self.v0

        # Avoid z=0 singularity
        z_safe = torch.clamp(torch.abs(z_depth), min=1e-6)

        if abs(k) < 1e-6:
            # Straight ray limit
            r = torch.sqrt(r_sq)
            return r / v0
        else:
            # Curved ray formula
            arg = 1.0 + k * r_sq / (2.0 * v0 * z_safe)
            # Clamp to valid arccosh domain
            arg = torch.clamp(arg, min=1.0)
            t = torch.acosh(arg) / k
            return t

    def compute_traveltime_batch(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute batch curved-ray traveltimes."""
        # Compute horizontal offsets
        dx = image_x.unsqueeze(0) - surface_x.unsqueeze(1)
        dy = image_y.unsqueeze(0) - surface_y.unsqueeze(1)

        h_sq = dx**2 + dy**2

        k = self.gradient
        v0 = self.v0

        n_z = len(image_z)
        n_surface = len(surface_x)
        n_xy = len(image_x)

        # Expand z for broadcasting
        z = image_z.view(n_z, 1, 1).expand(n_z, n_surface, n_xy)
        z_safe = torch.clamp(torch.abs(z), min=1e-6)

        h_sq_exp = h_sq.unsqueeze(0).expand(n_z, n_surface, n_xy)
        r_sq = h_sq_exp + z**2

        if abs(k) < 1e-6:
            r = torch.sqrt(r_sq)
            return r / v0
        else:
            arg = 1.0 + k * r_sq / (2.0 * v0 * z_safe)
            arg = torch.clamp(arg, min=1.0)
            return torch.acosh(arg) / k

    def compute_emergence_angle(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute emergence angle for curved ray.

        For constant gradient, ray parameter p is constant along ray.
        At surface: sin(theta) = p * v0
        """
        if abs(self.gradient) < 1e-6:
            # Straight ray
            if isinstance(x_offset, torch.Tensor):
                h = torch.sqrt(x_offset**2 + y_offset**2)
                return torch.atan2(h, torch.abs(z_depth))
            else:
                h = np.sqrt(x_offset**2 + y_offset**2)
                return np.arctan2(h, np.abs(z_depth))

        # For curved ray, need to solve for ray parameter
        # Simplified: use approximate angle
        if isinstance(x_offset, torch.Tensor):
            h = torch.sqrt(x_offset**2 + y_offset**2)
            r = torch.sqrt(h**2 + z_depth**2)
            # Approximate emergence angle (would need full ray tracing for exact)
            return torch.asin(torch.clamp(h / (r + 1e-6), -1, 1))
        else:
            h = np.sqrt(x_offset**2 + y_offset**2)
            r = np.sqrt(h**2 + z_depth**2)
            return np.arcsin(np.clip(h / (r + 1e-6), -1, 1))

    def get_description(self) -> str:
        return (
            f"CurvedRay(v0={self.v0:.0f} m/s, "
            f"k={self.gradient:.4f})"
        )


# =============================================================================
# Factory Function
# =============================================================================

def get_traveltime_calculator(
    velocity_model: VelocityModel,
    mode: Union[str, 'TraveltimeMode'] = 'auto',
    device: Optional[torch.device] = None,
) -> TraveltimeCalculator:
    """
    Factory function to get appropriate traveltime calculator.

    Args:
        velocity_model: Velocity model
        mode: 'straight', 'curved', 'auto', or TraveltimeMode enum
        device: Torch device

    Returns:
        Appropriate TraveltimeCalculator instance
    """
    # Handle TraveltimeMode enum
    from models.migration_config import TraveltimeMode
    if isinstance(mode, TraveltimeMode):
        if mode == TraveltimeMode.STRAIGHT_RAY:
            mode = 'straight'
        elif mode == TraveltimeMode.CURVED_RAY:
            mode = 'curved'

    if mode == 'auto':
        # Use curved ray if gradient is significant
        if velocity_model.has_gradient:
            mode = 'curved'
        else:
            mode = 'straight'

    if mode == 'straight':
        return StraightRayTraveltime(velocity_model, device)
    elif mode == 'curved':
        return CurvedRayTraveltime(velocity_model, device)
    else:
        raise ValueError(f"Unknown traveltime mode: {mode}")
