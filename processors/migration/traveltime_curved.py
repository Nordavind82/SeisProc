"""
Curved Ray Traveltime Calculator

Implements analytic curved ray traveltimes for constant velocity gradient media.
For v(z) = v0 + k*z, rays follow circular arcs.

Key formulas:
- Traveltime: t = (1/k) * arccosh(1 + k*r²/(2*v0*z))
- Ray parameter: p = sin(θ)/v (constant along ray)
- Emergence angle: θ = arcsin(p * v0)
- Arc radius: R = v0/(k * cos(θ0))

References:
- Slotnick (1959): "Curvature of sound rays in layered media"
- Cerveny (2001): "Seismic Ray Theory" Chapter 3
"""

from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import logging

from models.velocity_model import VelocityModel

logger = logging.getLogger(__name__)


@dataclass
class CurvedRayResult:
    """
    Results from curved ray computation.

    Attributes:
        traveltime: One-way traveltime in seconds
        ray_parameter: Ray parameter p = sin(θ)/v (s/m)
        emergence_angle: Angle from vertical at surface (radians)
        turning_depth: Depth where ray turns (if any) (same units as z)
        arc_length: Total arc length of ray path (meters)
        spreading_factor: Geometrical spreading factor for amplitude correction
    """
    traveltime: Union[float, np.ndarray, torch.Tensor]
    ray_parameter: Union[float, np.ndarray, torch.Tensor]
    emergence_angle: Union[float, np.ndarray, torch.Tensor]
    turning_depth: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    arc_length: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    spreading_factor: Optional[Union[float, np.ndarray, torch.Tensor]] = None


class CurvedRayCalculator:
    """
    Enhanced curved ray traveltime calculator for constant velocity gradient.

    For velocity model v(z) = v0 + k*z where:
    - v0: velocity at z=0 (surface velocity)
    - k: velocity gradient (dv/dz)

    Rays follow circular arcs with:
    - Arc center at depth z_c = -v0/k (above surface for k > 0)
    - Arc radius R = v0 / (k * cos(θ0)) where θ0 is takeoff angle

    Supports:
    - Analytic traveltime computation
    - Ray parameter and emergence angle calculation
    - Turning ray detection
    - Geometrical spreading for amplitude correction
    - GPU batch computation
    """

    # Threshold for switching to straight ray approximation
    GRADIENT_THRESHOLD = 1e-6

    # Minimum depth to avoid singularity at z=0
    MIN_DEPTH = 1e-6

    def __init__(
        self,
        velocity_model: VelocityModel,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize curved ray calculator.

        Args:
            velocity_model: Velocity model (should have gradient property)
            device: Torch device for GPU computation
        """
        self.velocity_model = velocity_model
        self.device = device or self._detect_device()

        # Extract v0 and gradient
        self.v0 = velocity_model.v0
        self.k = velocity_model.gradient or 0.0

        # Precompute useful quantities
        self._is_straight_ray = abs(self.k) < self.GRADIENT_THRESHOLD

        if not self._is_straight_ray:
            # Center of circular ray paths (negative = above surface for k > 0)
            self._z_center = -self.v0 / self.k
        else:
            self._z_center = None

        logger.info(
            f"CurvedRayCalculator: v0={self.v0:.0f} m/s, "
            f"k={self.k:.6f}, straight_ray={self._is_straight_ray}"
        )

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def compute_traveltime(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute curved ray one-way traveltime.

        For v(z) = v0 + k*z:
        t = (1/k) * arccosh(1 + k*r²/(2*v0*z))

        Args:
            x_offset: Horizontal X offset from surface point (meters)
            y_offset: Horizontal Y offset from surface point (meters)
            z_depth: Vertical depth/time to image point

        Returns:
            One-way traveltime in seconds
        """
        use_numpy = isinstance(x_offset, np.ndarray)

        # Convert to tensors
        if use_numpy:
            x_offset = torch.from_numpy(x_offset.astype(np.float32)).to(self.device)
            y_offset = torch.from_numpy(y_offset.astype(np.float32)).to(self.device)
            z_depth = torch.from_numpy(z_depth.astype(np.float32)).to(self.device)
        elif not isinstance(x_offset, torch.Tensor):
            x_offset = torch.tensor(x_offset, dtype=torch.float32, device=self.device)
            y_offset = torch.tensor(y_offset, dtype=torch.float32, device=self.device)
            z_depth = torch.tensor(z_depth, dtype=torch.float32, device=self.device)

        t = self._compute_traveltime_tensor(x_offset, y_offset, z_depth)

        if use_numpy:
            return t.cpu().numpy()
        return t

    def _compute_traveltime_tensor(
        self,
        x_offset: torch.Tensor,
        y_offset: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Core traveltime computation on tensors."""
        # Horizontal and total distance squared
        h_sq = x_offset**2 + y_offset**2
        r_sq = h_sq + z_depth**2

        # Avoid z=0 singularity
        z_safe = torch.clamp(torch.abs(z_depth), min=self.MIN_DEPTH)

        if self._is_straight_ray:
            # Straight ray limit: t = r / v0
            r = torch.sqrt(r_sq)
            return r / self.v0
        else:
            # Curved ray formula: t = (1/k) * arccosh(1 + k*r²/(2*v0*z))
            arg = 1.0 + self.k * r_sq / (2.0 * self.v0 * z_safe)

            # Clamp to valid arccosh domain [1, inf)
            # Values < 1 can occur due to numerical issues at shallow depths
            arg = torch.clamp(arg, min=1.0)

            t = torch.acosh(arg) / self.k
            return t

    def compute_ray_parameter(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute ray parameter p = sin(θ)/v.

        For curved rays, p is constant along the entire ray path.
        This is Snell's law in a gradient medium.

        Args:
            x_offset, y_offset: Horizontal offsets (meters)
            z_depth: Depth (meters or seconds for time migration)

        Returns:
            Ray parameter in s/m
        """
        use_numpy = isinstance(x_offset, np.ndarray)

        if use_numpy:
            x_offset = torch.from_numpy(x_offset.astype(np.float32)).to(self.device)
            y_offset = torch.from_numpy(y_offset.astype(np.float32)).to(self.device)
            z_depth = torch.from_numpy(z_depth.astype(np.float32)).to(self.device)
        elif not isinstance(x_offset, torch.Tensor):
            x_offset = torch.tensor(x_offset, dtype=torch.float32, device=self.device)
            y_offset = torch.tensor(y_offset, dtype=torch.float32, device=self.device)
            z_depth = torch.tensor(z_depth, dtype=torch.float32, device=self.device)

        p = self._compute_ray_parameter_tensor(x_offset, y_offset, z_depth)

        if use_numpy:
            return p.cpu().numpy()
        return p

    def _compute_ray_parameter_tensor(
        self,
        x_offset: torch.Tensor,
        y_offset: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Core ray parameter computation."""
        h = torch.sqrt(x_offset**2 + y_offset**2)
        z_safe = torch.clamp(torch.abs(z_depth), min=self.MIN_DEPTH)

        if self._is_straight_ray:
            # For straight ray: p = sin(θ)/v = h/(r*v0)
            r = torch.sqrt(h**2 + z_safe**2)
            p = h / (r * self.v0)
        else:
            # For curved ray in constant gradient:
            # p = h / (v0 * sqrt(h² + (z + v0/k)²))
            # This comes from the geometry of circular ray paths
            z_shifted = z_safe + self.v0 / self.k
            denom = self.v0 * torch.sqrt(h**2 + z_shifted**2)
            p = h / torch.clamp(denom, min=1e-10)

        return p

    def compute_emergence_angle(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute ray emergence angle at surface.

        The emergence angle θ is measured from vertical (z-axis).
        For curved rays: sin(θ) = p * v0 where p is ray parameter.

        Args:
            x_offset, y_offset: Horizontal offsets
            z_depth: Depth

        Returns:
            Emergence angle in radians (0 = vertical, π/2 = horizontal)
        """
        use_numpy = isinstance(x_offset, np.ndarray)

        if use_numpy:
            x_offset = torch.from_numpy(x_offset.astype(np.float32)).to(self.device)
            y_offset = torch.from_numpy(y_offset.astype(np.float32)).to(self.device)
            z_depth = torch.from_numpy(z_depth.astype(np.float32)).to(self.device)
        elif not isinstance(x_offset, torch.Tensor):
            x_offset = torch.tensor(x_offset, dtype=torch.float32, device=self.device)
            y_offset = torch.tensor(y_offset, dtype=torch.float32, device=self.device)
            z_depth = torch.tensor(z_depth, dtype=torch.float32, device=self.device)

        angle = self._compute_emergence_angle_tensor(x_offset, y_offset, z_depth)

        if use_numpy:
            return angle.cpu().numpy()
        return angle

    def _compute_emergence_angle_tensor(
        self,
        x_offset: torch.Tensor,
        y_offset: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Core emergence angle computation."""
        h = torch.sqrt(x_offset**2 + y_offset**2)
        z_safe = torch.clamp(torch.abs(z_depth), min=self.MIN_DEPTH)

        if self._is_straight_ray:
            # For straight ray: θ = arctan(h/z)
            return torch.atan2(h, z_safe)
        else:
            # For curved ray: sin(θ) = p * v0
            p = self._compute_ray_parameter_tensor(x_offset, y_offset, z_depth)
            sin_theta = torch.clamp(p * self.v0, -1.0, 1.0)
            return torch.asin(sin_theta)

    def compute_turning_depth(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute turning depth for a ray with given horizontal offset.

        The turning point is where the ray becomes horizontal (θ = 90°).
        This occurs at depth z_turn where v(z_turn) = 1/p.

        For v(z) = v0 + k*z: z_turn = (1/p - v0) / k

        Args:
            x_offset, y_offset: Horizontal offsets at surface

        Returns:
            Turning depth (or inf if ray doesn't turn)
        """
        if self._is_straight_ray:
            # Straight rays never turn
            if isinstance(x_offset, np.ndarray):
                return np.full_like(x_offset, np.inf)
            elif isinstance(x_offset, torch.Tensor):
                return torch.full_like(x_offset, float('inf'))
            else:
                return float('inf')

        # For curved rays with positive gradient, rays can turn
        # if they are steep enough (large horizontal offset)
        use_numpy = isinstance(x_offset, np.ndarray)

        if use_numpy:
            h = np.sqrt(x_offset**2 + y_offset**2)
            # Maximum ray parameter for a turning ray
            # At turning point: v(z_turn) = 1/p, so p_max = 1/v0 (horizontal at surface)
            p_max = 1.0 / self.v0

            # For a ray reaching depth z with offset h, approximate p
            # This is complex - for now return inf (no turning)
            return np.full_like(x_offset, np.inf)
        else:
            return float('inf')

    def compute_spreading_factor(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute geometrical spreading factor for curved rays.

        For curved rays, the spreading factor differs from straight rays
        due to ray path curvature. The factor accounts for:
        1. Distance spreading (1/r effect)
        2. Ray tube divergence due to velocity gradient

        The spreading factor S is defined such that amplitude A ~ A0 / S

        For constant gradient:
        S = r * sqrt(v(z) / v0) * cos(θ_emerge) / cos(θ_image)

        Args:
            x_offset, y_offset: Horizontal offsets
            z_depth: Depth

        Returns:
            Spreading factor (dimensionless)
        """
        use_numpy = isinstance(x_offset, np.ndarray)

        if use_numpy:
            x_offset = torch.from_numpy(x_offset.astype(np.float32)).to(self.device)
            y_offset = torch.from_numpy(y_offset.astype(np.float32)).to(self.device)
            z_depth = torch.from_numpy(z_depth.astype(np.float32)).to(self.device)
        elif not isinstance(x_offset, torch.Tensor):
            x_offset = torch.tensor(x_offset, dtype=torch.float32, device=self.device)
            y_offset = torch.tensor(y_offset, dtype=torch.float32, device=self.device)
            z_depth = torch.tensor(z_depth, dtype=torch.float32, device=self.device)

        S = self._compute_spreading_tensor(x_offset, y_offset, z_depth)

        if use_numpy:
            return S.cpu().numpy()
        return S

    def _compute_spreading_tensor(
        self,
        x_offset: torch.Tensor,
        y_offset: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Core spreading factor computation."""
        h = torch.sqrt(x_offset**2 + y_offset**2)
        z_safe = torch.clamp(torch.abs(z_depth), min=self.MIN_DEPTH)
        r = torch.sqrt(h**2 + z_safe**2)

        if self._is_straight_ray:
            # For straight ray: S = r (simple distance spreading)
            return r
        else:
            # Velocity at depth
            v_z = self.v0 + self.k * z_safe

            # Emergence angle
            theta = self._compute_emergence_angle_tensor(x_offset, y_offset, z_depth)
            cos_theta = torch.cos(theta)

            # Velocity ratio factor
            v_ratio = torch.sqrt(v_z / self.v0)

            # Spreading factor for curved ray
            # This accounts for ray tube divergence in gradient medium
            S = r * v_ratio / torch.clamp(cos_theta, min=0.1)

            return S

    def compute_full(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> CurvedRayResult:
        """
        Compute all curved ray quantities at once.

        More efficient than calling individual methods when multiple
        quantities are needed.

        Args:
            x_offset, y_offset: Horizontal offsets
            z_depth: Depth

        Returns:
            CurvedRayResult with all computed quantities
        """
        use_numpy = isinstance(x_offset, np.ndarray)

        if use_numpy:
            x_offset = torch.from_numpy(x_offset.astype(np.float32)).to(self.device)
            y_offset = torch.from_numpy(y_offset.astype(np.float32)).to(self.device)
            z_depth = torch.from_numpy(z_depth.astype(np.float32)).to(self.device)
        elif not isinstance(x_offset, torch.Tensor):
            x_offset = torch.tensor(x_offset, dtype=torch.float32, device=self.device)
            y_offset = torch.tensor(y_offset, dtype=torch.float32, device=self.device)
            z_depth = torch.tensor(z_depth, dtype=torch.float32, device=self.device)

        # Compute all quantities
        t = self._compute_traveltime_tensor(x_offset, y_offset, z_depth)
        p = self._compute_ray_parameter_tensor(x_offset, y_offset, z_depth)
        theta = self._compute_emergence_angle_tensor(x_offset, y_offset, z_depth)
        S = self._compute_spreading_tensor(x_offset, y_offset, z_depth)

        # Convert back if needed
        if use_numpy:
            return CurvedRayResult(
                traveltime=t.cpu().numpy(),
                ray_parameter=p.cpu().numpy(),
                emergence_angle=theta.cpu().numpy(),
                spreading_factor=S.cpu().numpy(),
            )
        else:
            return CurvedRayResult(
                traveltime=t,
                ray_parameter=p,
                emergence_angle=theta,
                spreading_factor=S,
            )

    def compute_traveltime_batch(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute traveltimes for batch of surface-to-image point pairs.

        Optimized for GPU computation with proper broadcasting.

        Args:
            surface_x: Surface point X coordinates (n_surface,)
            surface_y: Surface point Y coordinates (n_surface,)
            image_x: Image point X coordinates (n_image,)
            image_y: Image point Y coordinates (n_image,)
            image_z: Image point depths (n_z,)

        Returns:
            Traveltime tensor of shape (n_z, n_surface, n_image)
        """
        n_z = len(image_z)
        n_surface = len(surface_x)
        n_xy = len(image_x)

        # Compute horizontal offsets: (n_surface, n_image)
        dx = image_x.unsqueeze(0) - surface_x.unsqueeze(1)
        dy = image_y.unsqueeze(0) - surface_y.unsqueeze(1)
        h_sq = dx**2 + dy**2

        # Expand z for broadcasting: (n_z, n_surface, n_image)
        z = image_z.view(n_z, 1, 1).expand(n_z, n_surface, n_xy)
        z_safe = torch.clamp(torch.abs(z), min=self.MIN_DEPTH)

        # Expand h_sq: (n_z, n_surface, n_image)
        h_sq_exp = h_sq.unsqueeze(0).expand(n_z, n_surface, n_xy)
        r_sq = h_sq_exp + z**2

        if self._is_straight_ray:
            r = torch.sqrt(r_sq)
            return r / self.v0
        else:
            arg = 1.0 + self.k * r_sq / (2.0 * self.v0 * z_safe)
            arg = torch.clamp(arg, min=1.0)
            return torch.acosh(arg) / self.k

    @property
    def is_straight_ray(self) -> bool:
        """Check if calculator is using straight ray approximation."""
        return self._is_straight_ray

    @property
    def gradient(self) -> float:
        """Get velocity gradient."""
        return self.k

    @property
    def surface_velocity(self) -> float:
        """Get surface velocity v0."""
        return self.v0

    def get_description(self) -> str:
        """Get human-readable description."""
        if self._is_straight_ray:
            return f"CurvedRay(straight approximation, v0={self.v0:.0f} m/s)"
        else:
            return f"CurvedRay(v0={self.v0:.0f} m/s, k={self.k:.6f})"


def compute_curved_ray_traveltime(
    v0: float,
    gradient: float,
    h_offset: Union[float, np.ndarray],
    z_depth: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convenience function for curved ray traveltime.

    Args:
        v0: Surface velocity (m/s)
        gradient: Velocity gradient dv/dz
        h_offset: Horizontal offset (meters)
        z_depth: Depth (meters or seconds)

    Returns:
        One-way traveltime (seconds)
    """
    if abs(gradient) < 1e-6:
        # Straight ray
        r = np.sqrt(h_offset**2 + z_depth**2)
        return r / v0
    else:
        # Curved ray
        r_sq = h_offset**2 + z_depth**2
        z_safe = np.maximum(np.abs(z_depth), 1e-6)
        arg = 1.0 + gradient * r_sq / (2.0 * v0 * z_safe)
        arg = np.maximum(arg, 1.0)
        return np.arccosh(arg) / gradient


def compare_straight_vs_curved(
    v0: float,
    gradient: float,
    h_offsets: np.ndarray,
    z_depths: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compare straight and curved ray traveltimes.

    Useful for analyzing when curved ray correction is significant.

    Args:
        v0: Surface velocity
        gradient: Velocity gradient
        h_offsets: Array of horizontal offsets
        z_depths: Array of depths

    Returns:
        Dictionary with 't_straight', 't_curved', 'difference_percent'
    """
    from models.velocity_model import create_constant_velocity, create_linear_gradient_velocity

    # Straight ray (constant v0)
    v_const = create_constant_velocity(v0)
    from processors.migration.traveltime import StraightRayTraveltime
    tt_straight = StraightRayTraveltime(v_const, device=torch.device('cpu'))

    # Curved ray
    v_grad = create_linear_gradient_velocity(v0, gradient, z_max=float(z_depths.max()) * 2)
    tt_curved = CurvedRayCalculator(v_grad, device=torch.device('cpu'))

    # Compute for all combinations
    H, Z = np.meshgrid(h_offsets, z_depths)
    H_flat = H.flatten().astype(np.float32)
    Z_flat = Z.flatten().astype(np.float32)
    Y_flat = np.zeros_like(H_flat)

    t_straight = tt_straight.compute_traveltime(
        torch.from_numpy(H_flat),
        torch.from_numpy(Y_flat),
        torch.from_numpy(Z_flat),
    ).numpy().reshape(H.shape)

    t_curved = tt_curved.compute_traveltime(H_flat, Y_flat, Z_flat).reshape(H.shape)

    diff_percent = 100.0 * (t_curved - t_straight) / t_straight

    return {
        't_straight': t_straight,
        't_curved': t_curved,
        'difference_percent': diff_percent,
        'h_offsets': h_offsets,
        'z_depths': z_depths,
    }
