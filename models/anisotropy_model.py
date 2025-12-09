"""
Anisotropy Model for VTI (Vertical Transversely Isotropic) Media

Implements Thomsen parameters (epsilon, delta) for anisotropic migration:
- epsilon: P-wave anisotropy (horizontal vs vertical)
- delta: Near-vertical anisotropy (affects NMO)
- eta: Anelliptic parameter (computed from epsilon, delta)

References:
- Thomsen (1986): "Weak elastic anisotropy"
- Alkhalifah & Tsvankin (1995): "Velocity analysis for transversely isotropic media"
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Tuple
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AnisotropyType(Enum):
    """Type of anisotropy parameter field."""
    CONSTANT = "constant"      # Single scalar value
    V_OF_Z = "v_of_z"          # 1D function of depth/time
    V_2D = "v_2d"              # 2D grid (x, z)
    V_3D = "v_3d"              # 3D cube (x, y, z)


@dataclass
class AnisotropyModel:
    """
    VTI Anisotropy model using Thomsen parameters.

    Thomsen Parameters:
    - epsilon: P-wave anisotropy, controls horizontal velocity
              V_horizontal = V_vertical * sqrt(1 + 2*epsilon)
    - delta: Near-vertical anisotropy, affects NMO velocity
              V_nmo ≈ V_vertical * sqrt(1 + 2*delta)
    - eta: Anelliptic parameter (auto-computed if not provided)
              eta = (epsilon - delta) / (1 + 2*delta)

    Physical Bounds (typical):
    - epsilon: [-0.2, 0.5] (often 0.1-0.3 for shales)
    - delta: [-0.3, 0.3] (can be negative unlike epsilon)
    - eta: Usually positive for sedimentary rocks

    Attributes:
        epsilon: P-wave horizontal anisotropy parameter
        delta: Near-vertical anisotropy parameter
        eta: Anelliptic parameter (optional, auto-computed)
        z_axis: Depth/time axis for 1D models
        x_axis: X axis for 2D/3D models
        y_axis: Y axis for 3D models
        anisotropy_type: Type of parameter field
        metadata: Additional model metadata
    """
    epsilon: Union[float, np.ndarray] = 0.0
    delta: Union[float, np.ndarray] = 0.0
    eta: Optional[Union[float, np.ndarray]] = None

    z_axis: Optional[np.ndarray] = None
    x_axis: Optional[np.ndarray] = None
    y_axis: Optional[np.ndarray] = None

    anisotropy_type: AnisotropyType = AnisotropyType.CONSTANT

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate the model."""
        self._infer_type()
        self._compute_eta_if_needed()
        self._validate()

    def _infer_type(self):
        """Infer anisotropy type from parameter shapes."""
        if isinstance(self.epsilon, np.ndarray):
            if self.epsilon.ndim == 1:
                self.anisotropy_type = AnisotropyType.V_OF_Z
            elif self.epsilon.ndim == 2:
                self.anisotropy_type = AnisotropyType.V_2D
            elif self.epsilon.ndim == 3:
                self.anisotropy_type = AnisotropyType.V_3D
        else:
            self.anisotropy_type = AnisotropyType.CONSTANT

    def _compute_eta_if_needed(self):
        """Compute eta from epsilon and delta if not provided."""
        if self.eta is None:
            self.eta = self.compute_eta(self.epsilon, self.delta)

    def _validate(self):
        """Validate parameter ranges and consistency."""
        # Check epsilon and delta shapes match
        eps_shape = np.asarray(self.epsilon).shape
        delta_shape = np.asarray(self.delta).shape

        if eps_shape != delta_shape:
            raise ValueError(
                f"epsilon shape {eps_shape} must match delta shape {delta_shape}"
            )

        # Validate physical bounds (warning only, don't raise)
        self._check_bounds()

    def _check_bounds(self):
        """Check if parameters are within typical physical bounds."""
        eps = np.asarray(self.epsilon)
        delta = np.asarray(self.delta)

        # Epsilon should typically be in [-0.2, 0.5]
        if np.any(eps < -0.3) or np.any(eps > 0.6):
            logger.warning(
                f"Epsilon values outside typical range [-0.2, 0.5]: "
                f"min={np.min(eps):.3f}, max={np.max(eps):.3f}"
            )

        # Delta should typically be in [-0.3, 0.3]
        if np.any(delta < -0.4) or np.any(delta > 0.4):
            logger.warning(
                f"Delta values outside typical range [-0.3, 0.3]: "
                f"min={np.min(delta):.3f}, max={np.max(delta):.3f}"
            )

        # Stability condition: 1 + 2*delta > 0
        if np.any(1 + 2 * delta <= 0):
            raise ValueError(
                "Unstable anisotropy: (1 + 2*delta) must be positive"
            )

    @staticmethod
    def compute_eta(
        epsilon: Union[float, np.ndarray],
        delta: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Compute anelliptic parameter eta from epsilon and delta.

        eta = (epsilon - delta) / (1 + 2*delta)

        This parameter controls the deviation from elliptic moveout.

        Args:
            epsilon: P-wave anisotropy parameter
            delta: Near-vertical anisotropy parameter

        Returns:
            Anelliptic parameter eta
        """
        denom = 1 + 2 * np.asarray(delta)
        # Protect against division by zero
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        return (np.asarray(epsilon) - np.asarray(delta)) / denom

    @property
    def is_isotropic(self) -> bool:
        """Check if model is effectively isotropic (epsilon=delta=0)."""
        eps = np.asarray(self.epsilon)
        delta = np.asarray(self.delta)
        return bool(np.allclose(eps, 0, atol=1e-6) and np.allclose(delta, 0, atol=1e-6))

    @property
    def is_elliptic(self) -> bool:
        """Check if model is elliptic (eta=0, epsilon=delta)."""
        eta = np.asarray(self.eta)
        return bool(np.allclose(eta, 0, atol=1e-6))

    def get_parameters_at(
        self,
        z: Union[float, np.ndarray],
        x: Optional[Union[float, np.ndarray]] = None,
        y: Optional[Union[float, np.ndarray]] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Get Thomsen parameters at specified location(s).

        Args:
            z: Depth/time coordinate
            x: X coordinate (for 2D/3D)
            y: Y coordinate (for 3D)

        Returns:
            Tuple of (epsilon, delta, eta) at the location
        """
        if self.anisotropy_type == AnisotropyType.CONSTANT:
            return self.epsilon, self.delta, self.eta

        elif self.anisotropy_type == AnisotropyType.V_OF_Z:
            return self._interpolate_1d(z)

        elif self.anisotropy_type == AnisotropyType.V_2D:
            return self._interpolate_2d(x, z)

        elif self.anisotropy_type == AnisotropyType.V_3D:
            return self._interpolate_3d(x, y, z)

        else:
            raise ValueError(f"Unknown anisotropy type: {self.anisotropy_type}")

    def _interpolate_1d(
        self,
        z: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate 1D anisotropy model."""
        if self.z_axis is None:
            raise ValueError("z_axis required for V_OF_Z model")

        eps = np.interp(z, self.z_axis, self.epsilon)
        delta = np.interp(z, self.z_axis, self.delta)
        eta = np.interp(z, self.z_axis, self.eta)

        return eps, delta, eta

    def _interpolate_2d(
        self,
        x: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate 2D anisotropy model using bilinear interpolation."""
        if self.x_axis is None or self.z_axis is None:
            raise ValueError("x_axis and z_axis required for V_2D model")

        from scipy.interpolate import RegularGridInterpolator

        # Create interpolators
        interp_eps = RegularGridInterpolator(
            (self.x_axis, self.z_axis), self.epsilon,
            bounds_error=False, fill_value=None
        )
        interp_delta = RegularGridInterpolator(
            (self.x_axis, self.z_axis), self.delta,
            bounds_error=False, fill_value=None
        )
        interp_eta = RegularGridInterpolator(
            (self.x_axis, self.z_axis), self.eta,
            bounds_error=False, fill_value=None
        )

        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(z)])

        eps = interp_eps(points)
        delta = interp_delta(points)
        eta = interp_eta(points)

        return eps, delta, eta

    def _interpolate_3d(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate 3D anisotropy model using trilinear interpolation."""
        if self.x_axis is None or self.y_axis is None or self.z_axis is None:
            raise ValueError("x_axis, y_axis, and z_axis required for V_3D model")

        from scipy.interpolate import RegularGridInterpolator

        interp_eps = RegularGridInterpolator(
            (self.x_axis, self.y_axis, self.z_axis), self.epsilon,
            bounds_error=False, fill_value=None
        )
        interp_delta = RegularGridInterpolator(
            (self.x_axis, self.y_axis, self.z_axis), self.delta,
            bounds_error=False, fill_value=None
        )
        interp_eta = RegularGridInterpolator(
            (self.x_axis, self.y_axis, self.z_axis), self.eta,
            bounds_error=False, fill_value=None
        )

        points = np.column_stack([
            np.atleast_1d(x),
            np.atleast_1d(y),
            np.atleast_1d(z)
        ])

        eps = interp_eps(points)
        delta = interp_delta(points)
        eta = interp_eta(points)

        return eps, delta, eta

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        d = {
            'anisotropy_type': self.anisotropy_type.value,
            'metadata': self.metadata.copy(),
        }

        if self.anisotropy_type == AnisotropyType.CONSTANT:
            d['epsilon'] = float(self.epsilon)
            d['delta'] = float(self.delta)
            d['eta'] = float(self.eta)
        else:
            d['epsilon'] = self.epsilon.tolist() if isinstance(self.epsilon, np.ndarray) else self.epsilon
            d['delta'] = self.delta.tolist() if isinstance(self.delta, np.ndarray) else self.delta
            d['eta'] = self.eta.tolist() if isinstance(self.eta, np.ndarray) else self.eta

            if self.z_axis is not None:
                d['z_axis'] = self.z_axis.tolist()
            if self.x_axis is not None:
                d['x_axis'] = self.x_axis.tolist()
            if self.y_axis is not None:
                d['y_axis'] = self.y_axis.tolist()

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AnisotropyModel':
        """Deserialize model from dictionary."""
        anisotropy_type = AnisotropyType(d.get('anisotropy_type', 'constant'))

        epsilon = d.get('epsilon', 0.0)
        delta = d.get('delta', 0.0)
        eta = d.get('eta')

        if anisotropy_type != AnisotropyType.CONSTANT:
            epsilon = np.array(epsilon)
            delta = np.array(delta)
            if eta is not None:
                eta = np.array(eta)

        z_axis = None
        x_axis = None
        y_axis = None

        if 'z_axis' in d:
            z_axis = np.array(d['z_axis'])
        if 'x_axis' in d:
            x_axis = np.array(d['x_axis'])
        if 'y_axis' in d:
            y_axis = np.array(d['y_axis'])

        return cls(
            epsilon=epsilon,
            delta=delta,
            eta=eta,
            z_axis=z_axis,
            x_axis=x_axis,
            y_axis=y_axis,
            anisotropy_type=anisotropy_type,
            metadata=d.get('metadata', {}),
        )

    def get_summary(self) -> str:
        """Get human-readable summary."""
        eps = np.asarray(self.epsilon)
        delta = np.asarray(self.delta)
        eta = np.asarray(self.eta)

        if self.anisotropy_type == AnisotropyType.CONSTANT:
            return (
                f"AnisotropyModel(type=constant, "
                f"epsilon={self.epsilon:.4f}, delta={self.delta:.4f}, "
                f"eta={self.eta:.4f})"
            )
        else:
            return (
                f"AnisotropyModel(type={self.anisotropy_type.value}, "
                f"epsilon=[{eps.min():.3f}, {eps.max():.3f}], "
                f"delta=[{delta.min():.3f}, {delta.max():.3f}], "
                f"eta=[{eta.min():.3f}, {eta.max():.3f}])"
            )


# =============================================================================
# Factory Functions
# =============================================================================

def create_isotropic() -> AnisotropyModel:
    """Create isotropic (no anisotropy) model."""
    return AnisotropyModel(epsilon=0.0, delta=0.0)


def create_constant_anisotropy(
    epsilon: float,
    delta: float,
) -> AnisotropyModel:
    """
    Create constant anisotropy model.

    Args:
        epsilon: P-wave horizontal anisotropy (typical: 0.1-0.3)
        delta: Near-vertical anisotropy (typical: 0-0.2)

    Returns:
        AnisotropyModel with constant parameters
    """
    return AnisotropyModel(epsilon=epsilon, delta=delta)


def create_shale_anisotropy(
    intensity: str = 'moderate'
) -> AnisotropyModel:
    """
    Create anisotropy model with typical shale parameters.

    Args:
        intensity: 'weak', 'moderate', or 'strong'

    Returns:
        AnisotropyModel with shale-like parameters
    """
    params = {
        'weak': (0.1, 0.05),
        'moderate': (0.2, 0.1),
        'strong': (0.35, 0.15),
    }

    if intensity not in params:
        raise ValueError(f"intensity must be one of {list(params.keys())}")

    epsilon, delta = params[intensity]
    return AnisotropyModel(
        epsilon=epsilon,
        delta=delta,
        metadata={'formation': 'shale', 'intensity': intensity}
    )


def create_1d_anisotropy(
    z_axis: np.ndarray,
    epsilon: np.ndarray,
    delta: np.ndarray,
) -> AnisotropyModel:
    """
    Create 1D anisotropy model (parameters vary with depth).

    Args:
        z_axis: Depth/time axis
        epsilon: Epsilon values at each depth
        delta: Delta values at each depth

    Returns:
        AnisotropyModel with 1D parameter variation
    """
    return AnisotropyModel(
        epsilon=epsilon,
        delta=delta,
        z_axis=z_axis,
        anisotropy_type=AnisotropyType.V_OF_Z,
    )


def create_gradient_anisotropy(
    epsilon_surface: float,
    epsilon_gradient: float,
    delta_surface: float,
    delta_gradient: float,
    z_max: float,
    dz: float = 0.01,
) -> AnisotropyModel:
    """
    Create 1D anisotropy model with linear gradient.

    Anisotropy often increases with depth due to compaction.

    Args:
        epsilon_surface: Epsilon at surface
        epsilon_gradient: d(epsilon)/dz
        delta_surface: Delta at surface
        delta_gradient: d(delta)/dz
        z_max: Maximum depth/time
        dz: Sample interval

    Returns:
        AnisotropyModel with linear gradient
    """
    z_axis = np.arange(0, z_max + dz, dz)
    epsilon = epsilon_surface + epsilon_gradient * z_axis
    delta = delta_surface + delta_gradient * z_axis

    return AnisotropyModel(
        epsilon=epsilon,
        delta=delta,
        z_axis=z_axis,
        anisotropy_type=AnisotropyType.V_OF_Z,
        metadata={
            'epsilon_surface': epsilon_surface,
            'epsilon_gradient': epsilon_gradient,
            'delta_surface': delta_surface,
            'delta_gradient': delta_gradient,
        }
    )


# =============================================================================
# VTI Physics Utilities
# =============================================================================

def compute_vti_phase_velocity(
    v0: float,
    theta: Union[float, np.ndarray],
    epsilon: Union[float, np.ndarray],
    delta: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute VTI P-wave phase velocity.

    V(theta) = V0 * sqrt(1 + 2*delta*sin^2(theta)*cos^2(theta) + 2*epsilon*sin^4(theta))

    This is the exact Thomsen phase velocity for weak anisotropy.

    Args:
        v0: Vertical P-wave velocity
        theta: Phase angle from vertical (radians)
        epsilon: Thomsen epsilon parameter
        delta: Thomsen delta parameter

    Returns:
        Phase velocity at given angle
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    sin2 = sin_theta ** 2
    sin4 = sin_theta ** 4
    cos2 = cos_theta ** 2

    # Thomsen phase velocity formula
    factor = 1 + 2 * delta * sin2 * cos2 + 2 * epsilon * sin4

    # Protect against negative values (shouldn't happen for physical parameters)
    factor = np.maximum(factor, 0.01)

    return v0 * np.sqrt(factor)


def compute_nmo_velocity(
    v0: float,
    delta: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute VTI NMO (moveout) velocity.

    V_nmo = V0 * sqrt(1 + 2*delta)

    This controls the short-offset moveout.

    Args:
        v0: Vertical P-wave velocity
        delta: Thomsen delta parameter

    Returns:
        NMO velocity
    """
    factor = 1 + 2 * np.asarray(delta)
    factor = np.maximum(factor, 0.01)
    return v0 * np.sqrt(factor)


def compute_horizontal_velocity(
    v0: float,
    epsilon: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute VTI horizontal P-wave velocity.

    V_horizontal = V0 * sqrt(1 + 2*epsilon)

    Args:
        v0: Vertical P-wave velocity
        epsilon: Thomsen epsilon parameter

    Returns:
        Horizontal velocity
    """
    factor = 1 + 2 * np.asarray(epsilon)
    factor = np.maximum(factor, 0.01)
    return v0 * np.sqrt(factor)


def compute_effective_eta(
    epsilon: Union[float, np.ndarray],
    delta: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute effective anelliptic parameter.

    eta = (epsilon - delta) / (1 + 2*delta)

    This single parameter controls the non-hyperbolic moveout.
    For time processing, eta is often the only anisotropy parameter needed.

    Args:
        epsilon: Thomsen epsilon
        delta: Thomsen delta

    Returns:
        Anelliptic parameter eta
    """
    return AnisotropyModel.compute_eta(epsilon, delta)
