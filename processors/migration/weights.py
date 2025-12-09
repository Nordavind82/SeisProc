"""
Amplitude Weight Calculator for Kirchhoff Migration

Computes amplitude weighting factors:
- Geometrical spreading correction
- Obliquity factor
- Aperture taper
- True-amplitude Jacobian (optional)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import torch
import logging

from models.migration_config import WeightMode

logger = logging.getLogger(__name__)


class AmplitudeWeight(ABC):
    """
    Abstract base class for amplitude weight computation.

    Amplitude weights correct for:
    1. Geometrical spreading (1/r decay)
    2. Obliquity (angle from vertical)
    3. Aperture edge effects (taper)
    """

    def __init__(
        self,
        mode: WeightMode = WeightMode.SPREADING,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize amplitude weight calculator.

        Args:
            mode: Weight computation mode
            device: Torch device for GPU computation
        """
        self.mode = mode
        self.device = device or self._detect_device()

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    @abstractmethod
    def compute_weight(
        self,
        r_source: Union[np.ndarray, torch.Tensor],
        r_receiver: Union[np.ndarray, torch.Tensor],
        angle_source: Union[np.ndarray, torch.Tensor],
        angle_receiver: Union[np.ndarray, torch.Tensor],
        velocity: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute amplitude weight.

        Args:
            r_source: Distance from source to image point
            r_receiver: Distance from receiver to image point
            angle_source: Ray angle at source (from vertical, radians)
            angle_receiver: Ray angle at receiver (from vertical, radians)
            velocity: Velocity at image point

        Returns:
            Amplitude weight factor
        """
        pass

    @abstractmethod
    def compute_taper(
        self,
        distance: Union[np.ndarray, torch.Tensor],
        max_distance: float,
        taper_width: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute aperture taper weight.

        Args:
            distance: Distance from image point to surface point
            max_distance: Maximum aperture distance
            taper_width: Taper width as fraction of max_distance

        Returns:
            Taper weight (0 to 1)
        """
        pass


class StandardWeight(AmplitudeWeight):
    """
    Standard Kirchhoff amplitude weighting.

    Supports multiple modes:
    - NONE: No weighting (w=1)
    - SPREADING: Geometrical spreading only
    - OBLIQUITY: Obliquity factor only
    - FULL: Complete true-amplitude weighting
    """

    def compute_weight(
        self,
        r_source: Union[np.ndarray, torch.Tensor],
        r_receiver: Union[np.ndarray, torch.Tensor],
        angle_source: Union[np.ndarray, torch.Tensor],
        angle_receiver: Union[np.ndarray, torch.Tensor],
        velocity: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute amplitude weight based on mode.

        Full weight:
        W = cos(theta_s) * cos(theta_r) / (r_s * r_r * V)
        """
        if self.mode == WeightMode.NONE:
            if isinstance(r_source, torch.Tensor):
                return torch.ones_like(r_source)
            else:
                return np.ones_like(r_source)

        # Convert to tensors if needed
        use_torch = isinstance(r_source, torch.Tensor)
        if not use_torch and isinstance(r_source, np.ndarray):
            r_source = torch.from_numpy(r_source).to(self.device)
            r_receiver = torch.from_numpy(r_receiver).to(self.device)
            angle_source = torch.from_numpy(angle_source).to(self.device)
            angle_receiver = torch.from_numpy(angle_receiver).to(self.device)
            if isinstance(velocity, np.ndarray):
                velocity = torch.from_numpy(velocity).to(self.device)
            use_torch = True

        if use_torch:
            weight = torch.ones_like(r_source)

            # Geometrical spreading: 1 / (r_s * r_r)
            if self.mode in [WeightMode.SPREADING, WeightMode.FULL]:
                # Avoid division by zero
                r_s_safe = torch.clamp(r_source, min=1.0)
                r_r_safe = torch.clamp(r_receiver, min=1.0)
                weight = weight / (r_s_safe * r_r_safe)

            # Obliquity factor: cos(theta_s) * cos(theta_r)
            if self.mode in [WeightMode.OBLIQUITY, WeightMode.FULL]:
                cos_s = torch.cos(angle_source)
                cos_r = torch.cos(angle_receiver)
                weight = weight * cos_s * cos_r

            # Velocity factor for full true-amplitude
            if self.mode == WeightMode.FULL:
                if isinstance(velocity, (int, float)):
                    v_safe = max(velocity, 100.0)
                else:
                    v_safe = torch.clamp(velocity, min=100.0)
                weight = weight / v_safe

            return weight

        else:
            # NumPy implementation
            weight = np.ones_like(r_source)

            if self.mode in [WeightMode.SPREADING, WeightMode.FULL]:
                r_s_safe = np.maximum(r_source, 1.0)
                r_r_safe = np.maximum(r_receiver, 1.0)
                weight = weight / (r_s_safe * r_r_safe)

            if self.mode in [WeightMode.OBLIQUITY, WeightMode.FULL]:
                cos_s = np.cos(angle_source)
                cos_r = np.cos(angle_receiver)
                weight = weight * cos_s * cos_r

            if self.mode == WeightMode.FULL:
                if isinstance(velocity, (int, float)):
                    v_safe = max(velocity, 100.0)
                else:
                    v_safe = np.maximum(velocity, 100.0)
                weight = weight / v_safe

            return weight.astype(np.float32)

    def compute_taper(
        self,
        distance: Union[np.ndarray, torch.Tensor],
        max_distance: float,
        taper_width: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute cosine taper at aperture edges.

        Taper is 1.0 inside (1-taper_width)*max_distance,
        tapers to 0.0 at max_distance.
        """
        if taper_width <= 0 or taper_width >= 1:
            if isinstance(distance, torch.Tensor):
                return torch.ones_like(distance)
            else:
                return np.ones_like(distance)

        taper_start = max_distance * (1.0 - taper_width)

        if isinstance(distance, torch.Tensor):
            taper = torch.ones_like(distance)

            # In taper zone
            in_taper = (distance > taper_start) & (distance <= max_distance)
            if in_taper.any():
                x = (distance[in_taper] - taper_start) / (max_distance - taper_start)
                taper[in_taper] = 0.5 * (1.0 + torch.cos(np.pi * x))

            # Outside aperture
            taper[distance > max_distance] = 0.0

            return taper

        else:
            taper = np.ones_like(distance)

            in_taper = (distance > taper_start) & (distance <= max_distance)
            if np.any(in_taper):
                x = (distance[in_taper] - taper_start) / (max_distance - taper_start)
                taper[in_taper] = 0.5 * (1.0 + np.cos(np.pi * x))

            taper[distance > max_distance] = 0.0

            return taper.astype(np.float32)

    def compute_angle_taper(
        self,
        angle: Union[np.ndarray, torch.Tensor],
        max_angle: float,
        taper_width: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute cosine taper at angle aperture edges.

        Args:
            angle: Ray angle from vertical (radians)
            max_angle: Maximum allowed angle (radians)
            taper_width: Taper width as fraction

        Returns:
            Angle taper weight
        """
        return self.compute_taper(angle, max_angle, taper_width)


class TrueAmplitudeWeight(AmplitudeWeight):
    """
    True-amplitude weight with full Beylkin Jacobian.

    For accurate amplitude-preserving migration (AVO analysis).
    Includes:
    - Geometrical spreading
    - Obliquity
    - Jacobian of reflection angle to offset transformation
    """

    def compute_weight(
        self,
        r_source: Union[np.ndarray, torch.Tensor],
        r_receiver: Union[np.ndarray, torch.Tensor],
        angle_source: Union[np.ndarray, torch.Tensor],
        angle_receiver: Union[np.ndarray, torch.Tensor],
        velocity: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute true-amplitude weight including Jacobian.

        W = cos(theta_s) * cos(theta_r) / (r_s * r_r * V) * J

        where J is the Beylkin Jacobian.
        """
        use_torch = isinstance(r_source, torch.Tensor)

        if not use_torch:
            r_source = torch.from_numpy(r_source).to(self.device)
            r_receiver = torch.from_numpy(r_receiver).to(self.device)
            angle_source = torch.from_numpy(angle_source).to(self.device)
            angle_receiver = torch.from_numpy(angle_receiver).to(self.device)
            if isinstance(velocity, np.ndarray):
                velocity = torch.from_numpy(velocity).to(self.device)

        # Standard weight components
        r_s_safe = torch.clamp(r_source, min=1.0)
        r_r_safe = torch.clamp(r_receiver, min=1.0)

        cos_s = torch.cos(angle_source)
        cos_r = torch.cos(angle_receiver)

        if isinstance(velocity, (int, float)):
            v_safe = max(velocity, 100.0)
        else:
            v_safe = torch.clamp(velocity, min=100.0)

        # Base weight
        weight = cos_s * cos_r / (r_s_safe * r_r_safe * v_safe)

        # Simplified Jacobian for 2.5D (assuming y-invariance)
        # Full 3D Jacobian would require more information
        # J ~ 1 / cos(theta_avg) for simple case
        theta_avg = (angle_source + angle_receiver) / 2
        jacobian = 1.0 / torch.clamp(torch.cos(theta_avg), min=0.1)

        weight = weight * jacobian

        if not use_torch:
            return weight.cpu().numpy().astype(np.float32)
        return weight

    def compute_taper(
        self,
        distance: Union[np.ndarray, torch.Tensor],
        max_distance: float,
        taper_width: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Use standard cosine taper."""
        standard = StandardWeight(WeightMode.FULL, self.device)
        return standard.compute_taper(distance, max_distance, taper_width)


class CurvedRayWeight(AmplitudeWeight):
    """
    Amplitude weight for curved ray migration.

    Accounts for geometrical spreading along curved ray paths,
    which differs from straight ray due to ray tube divergence
    in gradient media.

    For v(z) = v0 + k*z:
    - Spreading factor includes velocity ratio sqrt(v(z)/v0)
    - Emergence angle affects obliquity differently
    """

    def __init__(
        self,
        mode: WeightMode = WeightMode.FULL,
        v0: float = 2000.0,
        gradient: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize curved ray weight calculator.

        Args:
            mode: Weight computation mode
            v0: Surface velocity (m/s)
            gradient: Velocity gradient dv/dz
            device: Torch device
        """
        super().__init__(mode, device)
        self.v0 = v0
        self.gradient = gradient

    def compute_weight(
        self,
        r_source: Union[np.ndarray, torch.Tensor],
        r_receiver: Union[np.ndarray, torch.Tensor],
        angle_source: Union[np.ndarray, torch.Tensor],
        angle_receiver: Union[np.ndarray, torch.Tensor],
        velocity: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute curved ray amplitude weight.

        W = cos(theta_s) * cos(theta_r) * sqrt(v(z)/v0) / (S_s * S_r * V)

        where S_s, S_r are curved ray spreading factors.
        """
        if self.mode == WeightMode.NONE:
            if isinstance(r_source, torch.Tensor):
                return torch.ones_like(r_source)
            else:
                return np.ones_like(r_source)

        use_torch = isinstance(r_source, torch.Tensor)
        if not use_torch and isinstance(r_source, np.ndarray):
            r_source = torch.from_numpy(r_source.astype(np.float32)).to(self.device)
            r_receiver = torch.from_numpy(r_receiver.astype(np.float32)).to(self.device)
            angle_source = torch.from_numpy(angle_source.astype(np.float32)).to(self.device)
            angle_receiver = torch.from_numpy(angle_receiver.astype(np.float32)).to(self.device)
            if isinstance(velocity, np.ndarray):
                velocity = torch.from_numpy(velocity.astype(np.float32)).to(self.device)

        # Spreading factor for curved rays
        r_s_safe = torch.clamp(r_source, min=1.0)
        r_r_safe = torch.clamp(r_receiver, min=1.0)

        # Velocity ratio factor for curved rays
        if isinstance(velocity, (int, float)):
            v_at_z = velocity
        else:
            v_at_z = velocity

        # Compute spreading correction for gradient
        if abs(self.gradient) > 1e-6:
            # Velocity ratio affects spreading in gradient media
            v_ratio = torch.sqrt(torch.tensor(v_at_z / self.v0, dtype=torch.float32))
        else:
            v_ratio = torch.tensor(1.0, dtype=torch.float32)

        weight = torch.ones_like(r_source)

        # Geometrical spreading with curved ray correction
        if self.mode in [WeightMode.SPREADING, WeightMode.FULL]:
            weight = weight * v_ratio / (r_s_safe * r_r_safe)

        # Obliquity factor
        if self.mode in [WeightMode.OBLIQUITY, WeightMode.FULL]:
            cos_s = torch.cos(angle_source)
            cos_r = torch.cos(angle_receiver)
            weight = weight * cos_s * cos_r

        # Velocity normalization for full weight
        if self.mode == WeightMode.FULL:
            if isinstance(velocity, (int, float)):
                v_safe = max(velocity, 100.0)
            else:
                v_safe = torch.clamp(velocity, min=100.0)
            weight = weight / v_safe

        if not use_torch:
            return weight.cpu().numpy().astype(np.float32)
        return weight

    def compute_taper(
        self,
        distance: Union[np.ndarray, torch.Tensor],
        max_distance: float,
        taper_width: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Use standard cosine taper."""
        standard = StandardWeight(WeightMode.FULL, self.device)
        return standard.compute_taper(distance, max_distance, taper_width)

    def compute_weight_with_spreading(
        self,
        spreading_source: Union[np.ndarray, torch.Tensor],
        spreading_receiver: Union[np.ndarray, torch.Tensor],
        angle_source: Union[np.ndarray, torch.Tensor],
        angle_receiver: Union[np.ndarray, torch.Tensor],
        velocity: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute weight using pre-computed spreading factors.

        Use this when spreading factors have already been computed
        by CurvedRayCalculator for efficiency.

        Args:
            spreading_source: Spreading factor for source leg
            spreading_receiver: Spreading factor for receiver leg
            angle_source: Emergence angle at source
            angle_receiver: Emergence angle at receiver
            velocity: Velocity at image point

        Returns:
            Amplitude weight
        """
        if self.mode == WeightMode.NONE:
            if isinstance(spreading_source, torch.Tensor):
                return torch.ones_like(spreading_source)
            else:
                return np.ones_like(spreading_source)

        use_torch = isinstance(spreading_source, torch.Tensor)
        if not use_torch:
            spreading_source = torch.from_numpy(spreading_source.astype(np.float32)).to(self.device)
            spreading_receiver = torch.from_numpy(spreading_receiver.astype(np.float32)).to(self.device)
            angle_source = torch.from_numpy(angle_source.astype(np.float32)).to(self.device)
            angle_receiver = torch.from_numpy(angle_receiver.astype(np.float32)).to(self.device)
            if isinstance(velocity, np.ndarray):
                velocity = torch.from_numpy(velocity.astype(np.float32)).to(self.device)

        S_s_safe = torch.clamp(spreading_source, min=1.0)
        S_r_safe = torch.clamp(spreading_receiver, min=1.0)

        weight = torch.ones_like(spreading_source)

        # Spreading using curved ray factors
        if self.mode in [WeightMode.SPREADING, WeightMode.FULL]:
            weight = weight / (S_s_safe * S_r_safe)

        # Obliquity
        if self.mode in [WeightMode.OBLIQUITY, WeightMode.FULL]:
            cos_s = torch.cos(angle_source)
            cos_r = torch.cos(angle_receiver)
            weight = weight * cos_s * cos_r

        # Velocity normalization
        if self.mode == WeightMode.FULL:
            if isinstance(velocity, (int, float)):
                v_safe = max(velocity, 100.0)
            else:
                v_safe = torch.clamp(velocity, min=100.0)
            weight = weight / v_safe

        if not use_torch:
            return weight.cpu().numpy().astype(np.float32)
        return weight


# =============================================================================
# Factory Function
# =============================================================================

def get_amplitude_weight(
    mode: WeightMode = WeightMode.SPREADING,
    device: Optional[torch.device] = None,
    curved_ray: bool = False,
    v0: float = 2000.0,
    gradient: float = 0.0,
) -> AmplitudeWeight:
    """
    Factory function for amplitude weight calculator.

    Args:
        mode: Weight computation mode
        device: Torch device
        curved_ray: If True, use curved ray weight calculator
        v0: Surface velocity for curved ray (m/s)
        gradient: Velocity gradient for curved ray

    Returns:
        AmplitudeWeight instance
    """
    if curved_ray and abs(gradient) > 1e-6:
        return CurvedRayWeight(mode, v0, gradient, device)
    elif mode == WeightMode.FULL:
        # For production true-amplitude, use specialized class
        return TrueAmplitudeWeight(mode, device)
    else:
        return StandardWeight(mode, device)


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_spreading_weight(
    r_source: np.ndarray,
    r_receiver: np.ndarray,
) -> np.ndarray:
    """
    Quick geometrical spreading weight computation.

    Args:
        r_source: Source-to-image distances
        r_receiver: Image-to-receiver distances

    Returns:
        Spreading weight array
    """
    r_s_safe = np.maximum(r_source, 1.0)
    r_r_safe = np.maximum(r_receiver, 1.0)
    return (1.0 / (r_s_safe * r_r_safe)).astype(np.float32)


def compute_obliquity_weight(
    angle_source: np.ndarray,
    angle_receiver: np.ndarray,
) -> np.ndarray:
    """
    Quick obliquity weight computation.

    Args:
        angle_source: Source ray angles (radians)
        angle_receiver: Receiver ray angles (radians)

    Returns:
        Obliquity weight array
    """
    return (np.cos(angle_source) * np.cos(angle_receiver)).astype(np.float32)


def compute_aperture_mask(
    distance: np.ndarray,
    angle: np.ndarray,
    max_distance: float,
    max_angle: float,
    taper_width: float = 0.1,
) -> np.ndarray:
    """
    Compute combined aperture mask with distance and angle limits.

    Args:
        distance: Horizontal distances
        angle: Ray angles from vertical (radians)
        max_distance: Maximum aperture distance
        max_angle: Maximum aperture angle (radians)
        taper_width: Taper width fraction

    Returns:
        Combined aperture mask (0 to 1)
    """
    weight_calc = StandardWeight(WeightMode.NONE)

    dist_taper = weight_calc.compute_taper(distance, max_distance, taper_width)
    angle_taper = weight_calc.compute_angle_taper(angle, max_angle, taper_width)

    return (dist_taper * angle_taper).astype(np.float32)
