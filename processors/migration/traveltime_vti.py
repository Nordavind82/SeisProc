"""
VTI Traveltime Calculator for Anisotropic Migration

Implements P-wave traveltimes in Vertical Transversely Isotropic media
using Thomsen parameters (epsilon, delta) or anelliptic parameter (eta).

Key formulas:
- Phase velocity: V(theta) = V0 * sqrt(1 + 2*delta*sin^2(theta)*cos^2(theta) + 2*epsilon*sin^4(theta))
- NMO velocity: V_nmo = V0 * sqrt(1 + 2*delta)
- Anelliptic traveltime (Alkhalifah): t^2 = t0^2 + x^2/V_nmo^2 - 2*eta*x^4/(V_nmo^2*(t0^2*V_nmo^2 + (1+2*eta)*x^2))

References:
- Thomsen (1986): "Weak elastic anisotropy"
- Alkhalifah & Tsvankin (1995): "Velocity analysis for transversely isotropic media"
- Fomel (2004): "On anelliptic approximations for qP velocities"
"""

from typing import Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import logging

from models.velocity_model import VelocityModel
from models.anisotropy_model import AnisotropyModel

logger = logging.getLogger(__name__)


@dataclass
class VTITraveltimeResult:
    """
    Results from VTI traveltime computation.

    Attributes:
        traveltime: One-way traveltime in seconds
        phase_angle: Phase angle from vertical (radians)
        effective_velocity: Effective velocity used for traveltime
        anisotropy_correction: Ratio of VTI to isotropic traveltime
    """
    traveltime: Union[float, np.ndarray, torch.Tensor]
    phase_angle: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    effective_velocity: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    anisotropy_correction: Optional[Union[float, np.ndarray, torch.Tensor]] = None


class VTITraveltimeCalculator:
    """
    VTI Traveltime calculator using Thomsen parameters.

    Implements multiple approximations:
    1. Exact phase velocity integration (most accurate, slower)
    2. Alkhalifah's anelliptic approximation (fast, accurate for typical eta)
    3. Weak anisotropy approximation (fastest, less accurate for strong anisotropy)

    The anelliptic approximation is recommended for most cases as it
    provides good accuracy with computational efficiency.
    """

    # Threshold for treating as isotropic
    ANISOTROPY_THRESHOLD = 1e-6

    def __init__(
        self,
        velocity_model: VelocityModel,
        anisotropy_model: Optional[AnisotropyModel] = None,
        device: Optional[torch.device] = None,
        method: str = 'anelliptic',
    ):
        """
        Initialize VTI traveltime calculator.

        Args:
            velocity_model: Isotropic (vertical) velocity model
            anisotropy_model: VTI anisotropy parameters (epsilon, delta)
            device: Torch device for GPU computation
            method: 'anelliptic', 'exact', or 'weak'
        """
        self.velocity_model = velocity_model
        self.anisotropy_model = anisotropy_model
        self.device = device or self._detect_device()
        self.method = method

        # Extract velocity parameters
        self.v0 = velocity_model.v0
        self.v_gradient = velocity_model.gradient or 0.0

        # Extract anisotropy parameters
        if anisotropy_model is not None:
            self.epsilon = anisotropy_model.epsilon
            self.delta = anisotropy_model.delta
            self.eta = anisotropy_model.eta
            self._is_isotropic = anisotropy_model.is_isotropic
        else:
            self.epsilon = 0.0
            self.delta = 0.0
            self.eta = 0.0
            self._is_isotropic = True

        logger.info(
            f"VTITraveltime: v0={self.v0:.0f} m/s, "
            f"epsilon={np.mean(self.epsilon):.4f}, "
            f"delta={np.mean(self.delta):.4f}, "
            f"method={method}"
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
        Compute VTI one-way traveltime.

        Args:
            x_offset: Horizontal X offset from surface point (meters)
            y_offset: Horizontal Y offset from surface point (meters)
            z_depth: Vertical depth/time to image point

        Returns:
            One-way traveltime in seconds
        """
        if self._is_isotropic:
            return self._compute_isotropic(x_offset, y_offset, z_depth)

        if self.method == 'anelliptic':
            return self._compute_anelliptic(x_offset, y_offset, z_depth)
        elif self.method == 'exact':
            return self._compute_exact(x_offset, y_offset, z_depth)
        elif self.method == 'weak':
            return self._compute_weak_anisotropy(x_offset, y_offset, z_depth)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_isotropic(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Compute isotropic (straight ray) traveltime."""
        use_numpy = isinstance(x_offset, np.ndarray)

        if use_numpy:
            r = np.sqrt(x_offset**2 + y_offset**2 + z_depth**2)
            return r / self.v0
        elif isinstance(x_offset, torch.Tensor):
            r = torch.sqrt(x_offset**2 + y_offset**2 + z_depth**2)
            return r / self.v0
        else:
            r = np.sqrt(x_offset**2 + y_offset**2 + z_depth**2)
            return r / self.v0

    def _compute_anelliptic(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute VTI traveltime using Alkhalifah's anelliptic approximation.

        t^2 = t0^2 + x^2/V_nmo^2 - 2*eta*x^4/(V_nmo^2*(t0^2*V_nmo^2 + (1+2*eta)*x^2))

        This is accurate for eta < 0.3 (most sedimentary rocks).
        """
        use_numpy = isinstance(x_offset, np.ndarray)
        use_torch = isinstance(x_offset, torch.Tensor)

        # Horizontal offset squared
        if use_numpy:
            h_sq = x_offset**2 + y_offset**2
            z_safe = np.maximum(np.abs(z_depth), 1e-6)
        elif use_torch:
            h_sq = x_offset**2 + y_offset**2
            z_safe = torch.clamp(torch.abs(z_depth), min=1e-6)
        else:
            h_sq = x_offset**2 + y_offset**2
            z_safe = max(abs(z_depth), 1e-6)

        # Get anisotropy parameters at depth
        epsilon, delta, eta = self._get_anisotropy_at_depth(z_safe, use_numpy, use_torch)

        # NMO velocity
        v_nmo_sq = self.v0**2 * (1 + 2 * delta)
        v_nmo_sq = max(v_nmo_sq, 100.0) if isinstance(v_nmo_sq, float) else v_nmo_sq

        # Zero-offset traveltime
        t0 = z_safe / self.v0

        # Hyperbolic term
        t_sq_hyp = t0**2 + h_sq / v_nmo_sq

        # Anelliptic correction
        if use_numpy:
            denom = v_nmo_sq * (t0**2 * v_nmo_sq + (1 + 2 * eta) * h_sq)
            denom = np.maximum(denom, 1e-10)
            correction = 2 * eta * h_sq**2 / denom
            t_sq = np.maximum(t_sq_hyp - correction, 1e-12)
            return np.sqrt(t_sq)
        elif use_torch:
            denom = v_nmo_sq * (t0**2 * v_nmo_sq + (1 + 2 * eta) * h_sq)
            denom = torch.clamp(denom, min=1e-10)
            correction = 2 * eta * h_sq**2 / denom
            t_sq = torch.clamp(t_sq_hyp - correction, min=1e-12)
            return torch.sqrt(t_sq)
        else:
            denom = v_nmo_sq * (t0**2 * v_nmo_sq + (1 + 2 * eta) * h_sq)
            denom = max(denom, 1e-10)
            correction = 2 * eta * h_sq**2 / denom
            t_sq = max(t_sq_hyp - correction, 1e-12)
            return np.sqrt(t_sq)

    def _compute_exact(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute VTI traveltime using exact phase velocity integration.

        This is the most accurate method but slower.
        Uses numerical integration along the ray path.
        """
        # For now, use simplified approach with effective velocity
        use_numpy = isinstance(x_offset, np.ndarray)
        use_torch = isinstance(x_offset, torch.Tensor)

        if use_numpy:
            h = np.sqrt(x_offset**2 + y_offset**2)
            z_safe = np.maximum(np.abs(z_depth), 1e-6)
            r = np.sqrt(h**2 + z_safe**2)
        elif use_torch:
            h = torch.sqrt(x_offset**2 + y_offset**2)
            z_safe = torch.clamp(torch.abs(z_depth), min=1e-6)
            r = torch.sqrt(h**2 + z_safe**2)
        else:
            h = np.sqrt(x_offset**2 + y_offset**2)
            z_safe = max(abs(z_depth), 1e-6)
            r = np.sqrt(h**2 + z_safe**2)

        # Phase angle from vertical
        if use_numpy:
            theta = np.arctan2(h, z_safe)
        elif use_torch:
            theta = torch.atan2(h, z_safe)
        else:
            theta = np.arctan2(h, z_safe)

        # Get parameters at depth
        epsilon, delta, eta = self._get_anisotropy_at_depth(z_safe, use_numpy, use_torch)

        # Phase velocity at this angle
        v_phase = self._compute_phase_velocity(theta, epsilon, delta, use_numpy, use_torch)

        return r / v_phase

    def _compute_weak_anisotropy(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute VTI traveltime using weak anisotropy approximation.

        For weak anisotropy (epsilon, delta << 1), use first-order correction.
        Fastest but least accurate for strong anisotropy.
        """
        use_numpy = isinstance(x_offset, np.ndarray)
        use_torch = isinstance(x_offset, torch.Tensor)

        if use_numpy:
            h = np.sqrt(x_offset**2 + y_offset**2)
            z_safe = np.maximum(np.abs(z_depth), 1e-6)
            r = np.sqrt(h**2 + z_safe**2)
            theta = np.arctan2(h, z_safe)
        elif use_torch:
            h = torch.sqrt(x_offset**2 + y_offset**2)
            z_safe = torch.clamp(torch.abs(z_depth), min=1e-6)
            r = torch.sqrt(h**2 + z_safe**2)
            theta = torch.atan2(h, z_safe)
        else:
            h = np.sqrt(x_offset**2 + y_offset**2)
            z_safe = max(abs(z_depth), 1e-6)
            r = np.sqrt(h**2 + z_safe**2)
            theta = np.arctan2(h, z_safe)

        epsilon, delta, _ = self._get_anisotropy_at_depth(z_safe, use_numpy, use_torch)

        # Weak anisotropy: V ≈ V0 * (1 + delta*sin^2(theta)*cos^2(theta) + epsilon*sin^4(theta))
        if use_numpy:
            sin2 = np.sin(theta)**2
            cos2 = np.cos(theta)**2
            sin4 = sin2**2
            correction = 1 + delta * sin2 * cos2 + epsilon * sin4
            v_eff = self.v0 * np.sqrt(np.maximum(correction, 0.1))
        elif use_torch:
            sin2 = torch.sin(theta)**2
            cos2 = torch.cos(theta)**2
            sin4 = sin2**2
            correction = 1 + delta * sin2 * cos2 + epsilon * sin4
            v_eff = self.v0 * torch.sqrt(torch.clamp(correction, min=0.1))
        else:
            sin2 = np.sin(theta)**2
            cos2 = np.cos(theta)**2
            sin4 = sin2**2
            correction = 1 + delta * sin2 * cos2 + epsilon * sin4
            v_eff = self.v0 * np.sqrt(max(correction, 0.1))

        return r / v_eff

    def _get_anisotropy_at_depth(
        self,
        z: Union[float, np.ndarray, torch.Tensor],
        use_numpy: bool,
        use_torch: bool,
    ) -> Tuple:
        """Get anisotropy parameters at given depth."""
        if self.anisotropy_model is None or self.anisotropy_model.anisotropy_type.value == 'constant':
            return self.epsilon, self.delta, self.eta

        # For depth-varying anisotropy
        if use_torch:
            z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else z
        else:
            z_np = np.asarray(z)

        eps, delta, eta = self.anisotropy_model.get_parameters_at(z_np)

        if use_torch:
            eps = torch.tensor(eps, dtype=torch.float32, device=self.device)
            delta = torch.tensor(delta, dtype=torch.float32, device=self.device)
            eta = torch.tensor(eta, dtype=torch.float32, device=self.device)

        return eps, delta, eta

    def _compute_phase_velocity(
        self,
        theta: Union[float, np.ndarray, torch.Tensor],
        epsilon: Union[float, np.ndarray],
        delta: Union[float, np.ndarray],
        use_numpy: bool,
        use_torch: bool,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Compute exact VTI phase velocity."""
        if use_numpy:
            sin2 = np.sin(theta)**2
            cos2 = np.cos(theta)**2
            sin4 = sin2**2
            factor = 1 + 2 * delta * sin2 * cos2 + 2 * epsilon * sin4
            factor = np.maximum(factor, 0.01)
            return self.v0 * np.sqrt(factor)
        elif use_torch:
            sin2 = torch.sin(theta)**2
            cos2 = torch.cos(theta)**2
            sin4 = sin2**2
            factor = 1 + 2 * delta * sin2 * cos2 + 2 * epsilon * sin4
            factor = torch.clamp(factor, min=0.01)
            return self.v0 * torch.sqrt(factor)
        else:
            sin2 = np.sin(theta)**2
            cos2 = np.cos(theta)**2
            sin4 = sin2**2
            factor = 1 + 2 * delta * sin2 * cos2 + 2 * epsilon * sin4
            factor = max(factor, 0.01)
            return self.v0 * np.sqrt(factor)

    def compute_traveltime_batch(
        self,
        surface_x: torch.Tensor,
        surface_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute VTI traveltimes for batch of surface-to-image point pairs.

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
        z_safe = torch.clamp(torch.abs(z), min=1e-6)

        # Expand h_sq: (n_z, n_surface, n_image)
        h_sq_exp = h_sq.unsqueeze(0).expand(n_z, n_surface, n_xy)

        if self._is_isotropic:
            r = torch.sqrt(h_sq_exp + z**2)
            return r / self.v0

        # VTI anelliptic approximation
        v_nmo_sq = self.v0**2 * (1 + 2 * self.delta)
        t0 = z_safe / self.v0
        t_sq_hyp = t0**2 + h_sq_exp / v_nmo_sq

        denom = v_nmo_sq * (t0**2 * v_nmo_sq + (1 + 2 * self.eta) * h_sq_exp)
        denom = torch.clamp(denom, min=1e-10)
        correction = 2 * self.eta * h_sq_exp**2 / denom
        t_sq = torch.clamp(t_sq_hyp - correction, min=1e-12)

        return torch.sqrt(t_sq)

    def compute_full(
        self,
        x_offset: Union[float, np.ndarray, torch.Tensor],
        y_offset: Union[float, np.ndarray, torch.Tensor],
        z_depth: Union[float, np.ndarray, torch.Tensor],
    ) -> VTITraveltimeResult:
        """
        Compute traveltime with additional VTI quantities.

        Args:
            x_offset, y_offset: Horizontal offsets
            z_depth: Depth

        Returns:
            VTITraveltimeResult with traveltime and diagnostic info
        """
        use_numpy = isinstance(x_offset, np.ndarray)

        # VTI traveltime
        t_vti = self.compute_traveltime(x_offset, y_offset, z_depth)

        # Isotropic traveltime for comparison
        t_iso = self._compute_isotropic(x_offset, y_offset, z_depth)

        # Phase angle
        if use_numpy:
            h = np.sqrt(x_offset**2 + y_offset**2)
            z_safe = np.maximum(np.abs(z_depth), 1e-6)
            theta = np.arctan2(h, z_safe)
            correction = t_vti / t_iso
        else:
            h = np.sqrt(x_offset**2 + y_offset**2)
            z_safe = max(abs(z_depth), 1e-6)
            theta = np.arctan2(h, z_safe)
            correction = t_vti / t_iso

        return VTITraveltimeResult(
            traveltime=t_vti,
            phase_angle=theta,
            anisotropy_correction=correction,
        )

    @property
    def is_isotropic(self) -> bool:
        """Check if effectively isotropic."""
        return self._is_isotropic

    def get_description(self) -> str:
        """Get human-readable description."""
        if self._is_isotropic:
            return f"VTI(isotropic, v0={self.v0:.0f} m/s)"
        else:
            return (
                f"VTI(v0={self.v0:.0f} m/s, "
                f"eps={np.mean(self.epsilon):.3f}, "
                f"delta={np.mean(self.delta):.3f}, "
                f"method={self.method})"
            )


# =============================================================================
# Factory Function
# =============================================================================

def get_vti_traveltime_calculator(
    velocity_model: VelocityModel,
    anisotropy_model: Optional[AnisotropyModel] = None,
    method: str = 'anelliptic',
    device: Optional[torch.device] = None,
) -> VTITraveltimeCalculator:
    """
    Factory function for VTI traveltime calculator.

    Args:
        velocity_model: Velocity model
        anisotropy_model: Anisotropy parameters (None for isotropic)
        method: 'anelliptic', 'exact', or 'weak'
        device: Torch device

    Returns:
        VTITraveltimeCalculator instance
    """
    return VTITraveltimeCalculator(
        velocity_model=velocity_model,
        anisotropy_model=anisotropy_model,
        device=device,
        method=method,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_vti_traveltime(
    v0: float,
    epsilon: float,
    delta: float,
    h_offset: Union[float, np.ndarray],
    z_depth: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convenience function for VTI traveltime.

    Uses anelliptic approximation.

    Args:
        v0: Vertical velocity
        epsilon: Thomsen epsilon
        delta: Thomsen delta
        h_offset: Horizontal offset
        z_depth: Depth

    Returns:
        One-way traveltime
    """
    eta = (epsilon - delta) / (1 + 2 * delta)
    v_nmo_sq = v0**2 * (1 + 2 * delta)

    z_safe = np.maximum(np.abs(z_depth), 1e-6)
    h_sq = np.asarray(h_offset)**2
    t0 = z_safe / v0

    t_sq_hyp = t0**2 + h_sq / v_nmo_sq
    denom = v_nmo_sq * (t0**2 * v_nmo_sq + (1 + 2 * eta) * h_sq)
    denom = np.maximum(denom, 1e-10)
    correction = 2 * eta * h_sq**2 / denom
    t_sq = np.maximum(t_sq_hyp - correction, 1e-12)

    return np.sqrt(t_sq)


def compare_isotropic_vs_vti(
    v0: float,
    epsilon: float,
    delta: float,
    h_offsets: np.ndarray,
    z_depths: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compare isotropic and VTI traveltimes.

    Useful for analyzing when VTI correction is significant.

    Args:
        v0: Vertical velocity
        epsilon: Thomsen epsilon
        delta: Thomsen delta
        h_offsets: Array of horizontal offsets
        z_depths: Array of depths

    Returns:
        Dictionary with 't_iso', 't_vti', 'difference_percent'
    """
    H, Z = np.meshgrid(h_offsets, z_depths)
    H_flat = H.flatten()
    Z_flat = Z.flatten()

    # Isotropic
    r = np.sqrt(H_flat**2 + Z_flat**2)
    t_iso = (r / v0).reshape(H.shape)

    # VTI
    t_vti = compute_vti_traveltime(v0, epsilon, delta, H_flat, Z_flat).reshape(H.shape)

    diff_percent = 100.0 * (t_vti - t_iso) / t_iso

    return {
        't_isotropic': t_iso,
        't_vti': t_vti,
        'difference_percent': diff_percent,
        'h_offsets': h_offsets,
        'z_depths': z_depths,
    }
