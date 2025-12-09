"""
Antialiasing for Kirchhoff Migration

Implements dip-dependent antialiasing to prevent operator aliasing artifacts.

Key concepts:
- Operator aliasing occurs when the migration operator samples spatial wavenumbers
  beyond the Nyquist limit of the output grid
- The alias frequency depends on the local dip and velocity: f_alias = v / (2 * dx * sin(theta))
- We apply a low-pass filter with cutoff at or below f_alias

Methods implemented:
1. Triangle filter (linear interpolation AA) - fast, approximate
2. Sinc-based filter - accurate, more expensive
3. Gray zone handling - smooth transition at alias boundary

References:
- Lumley, Claerbout, Bevc (1994) "Antialiased Kirchhoff 3-D migration"
- Zhang, Zhang, Bleistein (2003) "True amplitude wave equation migration"
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AntialiasMethod(Enum):
    """Antialiasing method selection."""
    NONE = "none"                    # No antialiasing
    TRIANGLE = "triangle"            # Triangle (linear interpolation) filter
    SINC = "sinc"                    # Sinc-based filter
    GRAY_ZONE = "gray_zone"          # Gray zone method with smooth transition


@dataclass
class AntialiasResult:
    """Result from antialiasing computation."""
    weight: Union[np.ndarray, torch.Tensor]   # Antialiasing weight (0-1)
    f_alias: Union[np.ndarray, torch.Tensor]  # Alias frequency at each point
    is_aliased: Union[np.ndarray, torch.Tensor]  # Boolean mask of aliased samples


class AntialiasFilter:
    """
    Antialiasing filter for Kirchhoff migration.

    Computes dip-dependent antialiasing weights to prevent operator aliasing.
    The weight is applied multiplicatively to the migration amplitude.

    Attributes:
        method: Antialiasing method to use
        f_max: Maximum frequency in the data (Hz)
        dx: Spatial sampling in x direction (meters)
        dy: Spatial sampling in y direction (meters)
        dt: Time sampling (seconds)
        taper_width: Width of taper zone as fraction of f_alias (0-1)
        device: Torch device for computation
    """

    def __init__(
        self,
        method: AntialiasMethod = AntialiasMethod.TRIANGLE,
        f_max: float = 80.0,
        dx: float = 25.0,
        dy: float = 25.0,
        dt: float = 0.004,
        taper_width: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize antialiasing filter.

        Args:
            method: Antialiasing method
            f_max: Maximum frequency to preserve (Hz)
            dx: Output grid spacing in x (meters)
            dy: Output grid spacing in y (meters)
            dt: Time sampling (seconds)
            taper_width: Fractional width of taper zone
            device: Torch device
        """
        self.method = method
        self.f_max = f_max
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.taper_width = taper_width
        self.device = device or torch.device('cpu')

        # Nyquist frequency
        self.f_nyquist = 1.0 / (2.0 * dt)

        # Effective spatial Nyquist (average of x and y)
        self.k_nyquist_x = 1.0 / (2.0 * dx)  # cycles/meter
        self.k_nyquist_y = 1.0 / (2.0 * dy)

        logger.debug(
            f"AntialiasFilter initialized: method={method.value}, "
            f"f_max={f_max}Hz, dx={dx}m, dy={dy}m"
        )

    def compute_alias_frequency(
        self,
        velocity: Union[float, np.ndarray, torch.Tensor],
        dip_x: Union[float, np.ndarray, torch.Tensor],
        dip_y: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute alias frequency for given velocity and dip.

        The alias frequency is: f_alias = v / (2 * dx * |sin(theta)|)
        where theta is the dip angle.

        For 3D with dips in both x and y:
        f_alias = v / (2 * sqrt((dx*px)^2 + (dy*py)^2))
        where px, py are horizontal slownesses.

        Args:
            velocity: Local velocity (m/s)
            dip_x: Dip in x direction (radians or as sin(theta))
            dip_y: Dip in y direction (radians or as sin(theta)), optional

        Returns:
            Alias frequency in Hz
        """
        use_torch = isinstance(velocity, torch.Tensor) or isinstance(dip_x, torch.Tensor)

        if use_torch:
            velocity = self._to_tensor(velocity)
            dip_x = self._to_tensor(dip_x)

            # Compute horizontal slowness component
            sin_dip_x = torch.abs(torch.sin(dip_x)) if dip_x.abs().max() <= np.pi else torch.abs(dip_x)

            if dip_y is not None:
                dip_y = self._to_tensor(dip_y)
                sin_dip_y = torch.abs(torch.sin(dip_y)) if dip_y.abs().max() <= np.pi else torch.abs(dip_y)

                # Combined spatial aliasing
                denom = 2.0 * torch.sqrt(
                    (self.dx * sin_dip_x)**2 + (self.dy * sin_dip_y)**2 + 1e-10
                )
            else:
                denom = 2.0 * self.dx * (sin_dip_x + 1e-10)

            f_alias = velocity / denom

            # Clamp to Nyquist
            f_alias = torch.clamp(f_alias, max=self.f_nyquist)

        else:
            velocity = np.asarray(velocity)
            dip_x = np.asarray(dip_x)

            sin_dip_x = np.abs(np.sin(dip_x)) if np.abs(dip_x).max() <= np.pi else np.abs(dip_x)

            if dip_y is not None:
                dip_y = np.asarray(dip_y)
                sin_dip_y = np.abs(np.sin(dip_y)) if np.abs(dip_y).max() <= np.pi else np.abs(dip_y)

                denom = 2.0 * np.sqrt(
                    (self.dx * sin_dip_x)**2 + (self.dy * sin_dip_y)**2 + 1e-10
                )
            else:
                denom = 2.0 * self.dx * (sin_dip_x + 1e-10)

            f_alias = velocity / denom
            f_alias = np.minimum(f_alias, self.f_nyquist)

        return f_alias

    def compute_weight(
        self,
        velocity: Union[float, np.ndarray, torch.Tensor],
        dip_x: Union[float, np.ndarray, torch.Tensor],
        dip_y: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute antialiasing weight.

        The weight is 1.0 where f_max < f_alias (no aliasing),
        0.0 where aliasing would occur, with a smooth taper in between.

        Args:
            velocity: Local velocity (m/s)
            dip_x: Dip in x direction (radians)
            dip_y: Dip in y direction (radians), optional

        Returns:
            Antialiasing weight (0-1)
        """
        if self.method == AntialiasMethod.NONE:
            if isinstance(velocity, torch.Tensor):
                return torch.ones_like(velocity)
            else:
                return np.ones_like(np.asarray(velocity))

        f_alias = self.compute_alias_frequency(velocity, dip_x, dip_y)

        if self.method == AntialiasMethod.TRIANGLE:
            return self._triangle_weight(f_alias)
        elif self.method == AntialiasMethod.SINC:
            return self._sinc_weight(f_alias)
        elif self.method == AntialiasMethod.GRAY_ZONE:
            return self._gray_zone_weight(f_alias)
        else:
            raise ValueError(f"Unknown antialiasing method: {self.method}")

    def _triangle_weight(
        self,
        f_alias: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Triangle filter antialiasing weight.

        Simple linear ramp from 1 to 0 as f approaches f_alias.
        Weight = f_alias / f_max when f_alias < f_max, else 1.
        """
        use_torch = isinstance(f_alias, torch.Tensor)

        if use_torch:
            weight = torch.clamp(f_alias / self.f_max, min=0.0, max=1.0)
        else:
            weight = np.clip(f_alias / self.f_max, 0.0, 1.0)

        return weight

    def _sinc_weight(
        self,
        f_alias: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Sinc-based antialiasing weight.

        More accurate than triangle but computationally more expensive.
        Uses sinc function shape for smoother transition.
        """
        use_torch = isinstance(f_alias, torch.Tensor)

        if use_torch:
            # Normalized frequency ratio
            f_ratio = f_alias / self.f_max

            # Sinc-like weight with soft transition
            # Weight = 1 for f_alias >= f_max, smooth decay below
            weight = torch.where(
                f_ratio >= 1.0,
                torch.ones_like(f_ratio),
                torch.sin(np.pi * f_ratio / 2.0) ** 2
            )
        else:
            f_ratio = f_alias / self.f_max

            weight = np.where(
                f_ratio >= 1.0,
                np.ones_like(f_ratio),
                np.sin(np.pi * f_ratio / 2.0) ** 2
            )

        return weight

    def _gray_zone_weight(
        self,
        f_alias: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Gray zone antialiasing weight.

        Implements smooth transition in a "gray zone" around f_alias = f_max.
        Uses cosine taper for smooth derivatives.

        Weight = 1 when f_alias >= f_high (no aliasing)
        Weight = 0 when f_alias <= f_low (full aliasing)
        Weight = smooth transition in between
        """
        use_torch = isinstance(f_alias, torch.Tensor)

        # Define gray zone boundaries
        f_low = self.f_max * (1.0 - self.taper_width)
        f_high = self.f_max * (1.0 + self.taper_width)

        if use_torch:
            weight = torch.zeros_like(f_alias)

            # Above gray zone: no attenuation (weight = 1)
            weight[f_alias >= f_high] = 1.0

            # In gray zone: cosine taper (smooth transition from 0 to 1)
            in_zone = (f_alias > f_low) & (f_alias < f_high)
            if in_zone.any():
                # t goes from 0 (at f_low) to 1 (at f_high)
                t = (f_alias[in_zone] - f_low) / (f_high - f_low)
                # Cosine taper: 0.5*(1 - cos(pi*t)) goes from 0 to 1
                weight[in_zone] = 0.5 * (1.0 - torch.cos(np.pi * t))

            # Below gray zone: full attenuation (weight = 0, already initialized)

        else:
            weight = np.zeros_like(f_alias, dtype=np.float64)

            # Above gray zone: no attenuation
            weight[f_alias >= f_high] = 1.0

            # In gray zone: cosine taper
            in_zone = (f_alias > f_low) & (f_alias < f_high)
            if np.any(in_zone):
                t = (f_alias[in_zone] - f_low) / (f_high - f_low)
                weight[in_zone] = 0.5 * (1.0 - np.cos(np.pi * t))

            # Below gray zone: already 0

        return weight

    def compute_full(
        self,
        velocity: Union[float, np.ndarray, torch.Tensor],
        dip_x: Union[float, np.ndarray, torch.Tensor],
        dip_y: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
    ) -> AntialiasResult:
        """
        Compute full antialiasing result including diagnostics.

        Args:
            velocity: Local velocity (m/s)
            dip_x: Dip in x direction (radians)
            dip_y: Dip in y direction (radians), optional

        Returns:
            AntialiasResult with weight, alias frequency, and aliased mask
        """
        f_alias = self.compute_alias_frequency(velocity, dip_x, dip_y)
        weight = self.compute_weight(velocity, dip_x, dip_y)

        use_torch = isinstance(f_alias, torch.Tensor)

        if use_torch:
            is_aliased = f_alias < self.f_max
        else:
            is_aliased = f_alias < self.f_max

        return AntialiasResult(
            weight=weight,
            f_alias=f_alias,
            is_aliased=is_aliased,
        )

    def _to_tensor(self, x: Union[float, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor on correct device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        else:
            return torch.tensor(x, device=self.device)

    def get_description(self) -> str:
        """Get human-readable description."""
        return (
            f"AntialiasFilter({self.method.value}): "
            f"f_max={self.f_max}Hz, dx={self.dx}m, dy={self.dy}m"
        )


class DipEstimator:
    """
    Local dip estimator for antialiasing.

    Estimates local dip from neighboring traces using various methods:
    - Finite difference gradient
    - Structure tensor
    - Local slant stack
    """

    def __init__(
        self,
        method: str = 'gradient',
        window_size: int = 3,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize dip estimator.

        Args:
            method: Estimation method ('gradient', 'structure_tensor')
            window_size: Window size for estimation
            device: Torch device
        """
        self.method = method
        self.window_size = window_size
        self.device = device or torch.device('cpu')

    def estimate_dip_2d(
        self,
        data: Union[np.ndarray, torch.Tensor],
        dt: float,
        dx: float,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Estimate local dip from 2D section (time x space).

        Args:
            data: 2D array (n_time, n_traces)
            dt: Time sampling (seconds)
            dx: Trace spacing (meters)

        Returns:
            Tuple of (dip, confidence) arrays
        """
        use_torch = isinstance(data, torch.Tensor)

        if self.method == 'gradient':
            return self._gradient_dip_2d(data, dt, dx, use_torch)
        elif self.method == 'structure_tensor':
            return self._structure_tensor_dip_2d(data, dt, dx, use_torch)
        else:
            raise ValueError(f"Unknown dip estimation method: {self.method}")

    def _gradient_dip_2d(
        self,
        data: Union[np.ndarray, torch.Tensor],
        dt: float,
        dx: float,
        use_torch: bool,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Estimate dip using finite difference gradients."""
        if use_torch:
            data = data.to(self.device)

            # Compute gradients using central differences
            # Pad for boundary handling
            padded = torch.nn.functional.pad(data.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            padded = padded.squeeze()

            # Time gradient (vertical)
            grad_t = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * dt)

            # Space gradient (horizontal)
            grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * dx)

            # Dip = -grad_x / grad_t (in seconds per meter)
            # Convert to angle: dip_angle = arctan(dip * velocity)
            eps = 1e-10
            dip = -grad_x / (grad_t + eps * torch.sign(grad_t + eps))

            # Confidence based on gradient magnitude
            grad_mag = torch.sqrt(grad_t**2 + grad_x**2)
            confidence = torch.tanh(grad_mag / grad_mag.mean())

        else:
            data = np.asarray(data)

            # Compute gradients
            grad_t = np.gradient(data, dt, axis=0)
            grad_x = np.gradient(data, dx, axis=1)

            # Dip
            eps = 1e-10
            dip = -grad_x / (grad_t + eps * np.sign(grad_t + eps))

            # Confidence
            grad_mag = np.sqrt(grad_t**2 + grad_x**2)
            confidence = np.tanh(grad_mag / (grad_mag.mean() + eps))

        return dip, confidence

    def _structure_tensor_dip_2d(
        self,
        data: Union[np.ndarray, torch.Tensor],
        dt: float,
        dx: float,
        use_torch: bool,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Estimate dip using structure tensor analysis."""
        if use_torch:
            data = data.to(self.device)

            # Compute gradients
            padded = torch.nn.functional.pad(data.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            padded = padded.squeeze()

            grad_t = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * dt)
            grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * dx)

            # Structure tensor components
            Jtt = grad_t ** 2
            Jxx = grad_x ** 2
            Jtx = grad_t * grad_x

            # Smooth the tensor components
            kernel_size = self.window_size
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size ** 2)

            Jtt_smooth = torch.nn.functional.conv2d(
                Jtt.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2
            ).squeeze()
            Jxx_smooth = torch.nn.functional.conv2d(
                Jxx.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2
            ).squeeze()
            Jtx_smooth = torch.nn.functional.conv2d(
                Jtx.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2
            ).squeeze()

            # Eigenvalue analysis for dominant direction
            # For 2x2 symmetric matrix, dominant eigenvector gives dip
            trace = Jtt_smooth + Jxx_smooth
            det = Jtt_smooth * Jxx_smooth - Jtx_smooth ** 2

            # Eigenvalues
            disc = torch.sqrt(torch.clamp(trace**2 / 4 - det, min=0))
            lambda1 = trace / 2 + disc
            lambda2 = trace / 2 - disc

            # Dip from dominant eigenvector
            eps = 1e-10
            dip = -Jtx_smooth / (Jtt_smooth - lambda2 + eps)

            # Confidence from eigenvalue ratio (coherence)
            confidence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

        else:
            data = np.asarray(data)
            from scipy.ndimage import uniform_filter

            # Compute gradients
            grad_t = np.gradient(data, dt, axis=0)
            grad_x = np.gradient(data, dx, axis=1)

            # Structure tensor components
            Jtt = grad_t ** 2
            Jxx = grad_x ** 2
            Jtx = grad_t * grad_x

            # Smooth
            size = self.window_size
            Jtt_smooth = uniform_filter(Jtt, size=size)
            Jxx_smooth = uniform_filter(Jxx, size=size)
            Jtx_smooth = uniform_filter(Jtx, size=size)

            # Eigenvalue analysis
            trace = Jtt_smooth + Jxx_smooth
            det = Jtt_smooth * Jxx_smooth - Jtx_smooth ** 2

            disc = np.sqrt(np.maximum(trace**2 / 4 - det, 0))
            lambda1 = trace / 2 + disc
            lambda2 = trace / 2 - disc

            eps = 1e-10
            dip = -Jtx_smooth / (Jtt_smooth - lambda2 + eps)

            confidence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

        return dip, confidence

    def estimate_dip_from_traveltime(
        self,
        traveltime: Union[np.ndarray, torch.Tensor],
        dx: float,
        dy: Optional[float] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
        """
        Estimate dip from traveltime field.

        Dip is computed directly from traveltime gradient:
        p_x = dt/dx, p_y = dt/dy

        Args:
            traveltime: Traveltime field (2D or 3D)
            dx: X spacing
            dy: Y spacing (for 3D)

        Returns:
            Tuple of (dip_x, dip_y) or just (dip_x,) for 2D
        """
        use_torch = isinstance(traveltime, torch.Tensor)

        if use_torch:
            traveltime = traveltime.to(self.device)

            if traveltime.ndim == 2:
                # 2D case: (n_z, n_x)
                dip_x = torch.zeros_like(traveltime)
                dip_x[:, 1:-1] = (traveltime[:, 2:] - traveltime[:, :-2]) / (2 * dx)
                dip_x[:, 0] = (traveltime[:, 1] - traveltime[:, 0]) / dx
                dip_x[:, -1] = (traveltime[:, -1] - traveltime[:, -2]) / dx
                return (dip_x,)
            else:
                # 3D case: (n_z, n_x, n_y)
                dip_x = torch.zeros_like(traveltime)
                dip_y = torch.zeros_like(traveltime)

                dip_x[:, 1:-1, :] = (traveltime[:, 2:, :] - traveltime[:, :-2, :]) / (2 * dx)
                dip_x[:, 0, :] = (traveltime[:, 1, :] - traveltime[:, 0, :]) / dx
                dip_x[:, -1, :] = (traveltime[:, -1, :] - traveltime[:, -2, :]) / dx

                if dy is not None:
                    dip_y[:, :, 1:-1] = (traveltime[:, :, 2:] - traveltime[:, :, :-2]) / (2 * dy)
                    dip_y[:, :, 0] = (traveltime[:, :, 1] - traveltime[:, :, 0]) / dy
                    dip_y[:, :, -1] = (traveltime[:, :, -1] - traveltime[:, :, -2]) / dy

                return (dip_x, dip_y)
        else:
            traveltime = np.asarray(traveltime)

            if traveltime.ndim == 2:
                dip_x = np.gradient(traveltime, dx, axis=1)
                return (dip_x,)
            else:
                dip_x = np.gradient(traveltime, dx, axis=1)
                dip_y = np.gradient(traveltime, dy, axis=2) if dy else np.zeros_like(traveltime)
                return (dip_x, dip_y)


# =============================================================================
# Factory Functions
# =============================================================================

def get_antialias_filter(
    method: Union[str, AntialiasMethod] = 'triangle',
    f_max: float = 80.0,
    dx: float = 25.0,
    dy: float = 25.0,
    dt: float = 0.004,
    device: Optional[torch.device] = None,
) -> AntialiasFilter:
    """
    Factory function to create antialiasing filter.

    Args:
        method: Antialiasing method ('none', 'triangle', 'sinc', 'gray_zone')
        f_max: Maximum frequency (Hz)
        dx: X grid spacing (meters)
        dy: Y grid spacing (meters)
        dt: Time sampling (seconds)
        device: Torch device

    Returns:
        AntialiasFilter instance
    """
    if isinstance(method, str):
        method = AntialiasMethod(method.lower())

    return AntialiasFilter(
        method=method,
        f_max=f_max,
        dx=dx,
        dy=dy,
        dt=dt,
        device=device,
    )


def compute_antialias_weight(
    velocity: Union[float, np.ndarray],
    dip: Union[float, np.ndarray],
    f_max: float = 80.0,
    dx: float = 25.0,
) -> np.ndarray:
    """
    Quick antialiasing weight computation.

    Args:
        velocity: Velocity (m/s)
        dip: Dip angle (radians)
        f_max: Maximum frequency (Hz)
        dx: Grid spacing (meters)

    Returns:
        Antialiasing weight (0-1)
    """
    velocity = np.asarray(velocity)
    dip = np.asarray(dip)

    # Alias frequency
    sin_dip = np.abs(np.sin(dip))
    f_alias = velocity / (2.0 * dx * (sin_dip + 1e-10))
    f_alias = np.minimum(f_alias, 1.0 / (2.0 * 0.004))  # Assume 4ms sampling

    # Triangle weight
    weight = np.clip(f_alias / f_max, 0.0, 1.0)

    return weight


def estimate_optimal_grid_spacing(
    velocity: float,
    f_max: float,
    max_dip: float = 60.0,
) -> float:
    """
    Estimate optimal grid spacing to avoid aliasing.

    For given velocity, frequency, and maximum dip,
    computes the grid spacing where f_alias = f_max.

    Args:
        velocity: Velocity (m/s)
        f_max: Maximum frequency (Hz)
        max_dip: Maximum expected dip (degrees)

    Returns:
        Recommended grid spacing (meters)
    """
    sin_dip = np.sin(np.radians(max_dip))
    dx = velocity / (2.0 * f_max * sin_dip)

    return dx
