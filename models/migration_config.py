"""
Migration Configuration for Kirchhoff PSTM

Contains all parameters needed to configure a migration job:
- Aperture control (spatial and angular)
- Output grid definition
- Antialiasing settings
- Processing options
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TraveltimeMode(Enum):
    """Traveltime computation mode."""
    STRAIGHT_RAY = "straight_ray"
    CURVED_RAY = "curved_ray"
    VTI = "vti"  # VTI anisotropic


class AnisotropyMethod(Enum):
    """VTI anisotropy computation method."""
    ANELLIPTIC = "anelliptic"  # Alkhalifah's approximation (recommended)
    EXACT = "exact"            # Full phase velocity integration
    WEAK = "weak"              # Weak anisotropy approximation


class InterpolationMode(Enum):
    """Trace interpolation method."""
    LINEAR = "linear"
    SINC = "sinc"


class WeightMode(Enum):
    """Amplitude weight mode."""
    NONE = "none"                      # No amplitude correction
    SPREADING = "spreading"            # Geometrical spreading only
    OBLIQUITY = "obliquity"           # Obliquity factor only
    FULL = "full"                     # Full true-amplitude weights


@dataclass
class OutputGrid:
    """
    Definition of the output migration grid.

    Attributes:
        n_time: Number of output time/depth samples
        n_inline: Number of output inlines (X direction)
        n_xline: Number of output crosslines (Y direction)
        dt: Output time/depth sample interval (seconds or meters)
        d_inline: Inline spacing (meters)
        d_xline: Crossline spacing (meters)
        t0: First output time/depth (seconds or meters)
        inline_start: First inline number
        xline_start: First crossline number
        x_origin: X coordinate of origin (meters)
        y_origin: Y coordinate of origin (meters)
        inline_azimuth: Azimuth of inline direction (degrees from north)
    """
    n_time: int = 1000
    n_inline: int = 100
    n_xline: int = 100
    dt: float = 0.004          # seconds (4 ms)
    d_inline: float = 25.0     # meters
    d_xline: float = 25.0      # meters
    t0: float = 0.0            # seconds
    inline_start: int = 1
    xline_start: int = 1
    x_origin: float = 0.0      # meters
    y_origin: float = 0.0      # meters
    inline_azimuth: float = 0.0  # degrees from north

    def __post_init__(self):
        """Validate output grid parameters."""
        if self.n_time <= 0:
            raise ValueError(f"n_time must be positive, got {self.n_time}")
        if self.n_inline <= 0:
            raise ValueError(f"n_inline must be positive, got {self.n_inline}")
        if self.n_xline <= 0:
            raise ValueError(f"n_xline must be positive, got {self.n_xline}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.d_inline <= 0:
            raise ValueError(f"d_inline must be positive, got {self.d_inline}")
        if self.d_xline <= 0:
            raise ValueError(f"d_xline must be positive, got {self.d_xline}")

    @property
    def t_max(self) -> float:
        """Maximum output time/depth."""
        return self.t0 + (self.n_time - 1) * self.dt

    @property
    def time_axis(self) -> np.ndarray:
        """Get time/depth axis array."""
        return np.arange(self.n_time) * self.dt + self.t0

    @property
    def inline_axis(self) -> np.ndarray:
        """Get inline number axis."""
        return np.arange(self.n_inline) + self.inline_start

    @property
    def xline_axis(self) -> np.ndarray:
        """Get crossline number axis."""
        return np.arange(self.n_xline) + self.xline_start

    @property
    def x_extent(self) -> float:
        """Total extent in X (inline) direction in meters."""
        return (self.n_inline - 1) * self.d_inline

    @property
    def y_extent(self) -> float:
        """Total extent in Y (crossline) direction in meters."""
        return (self.n_xline - 1) * self.d_xline

    @property
    def total_samples(self) -> int:
        """Total number of output samples."""
        return self.n_time * self.n_inline * self.n_xline

    @property
    def memory_gb(self) -> float:
        """Estimated memory for output volume in GB (float32)."""
        return self.total_samples * 4 / (1024**3)

    def get_coordinates(
        self,
        inline_idx: int,
        xline_idx: int
    ) -> Tuple[float, float]:
        """
        Get world coordinates (X, Y) for given inline/crossline indices.

        Args:
            inline_idx: Inline index (0-based)
            xline_idx: Crossline index (0-based)

        Returns:
            Tuple of (X, Y) coordinates in meters
        """
        # Convert azimuth to radians
        az_rad = np.radians(self.inline_azimuth)

        # Distance along inline and crossline
        dist_inline = inline_idx * self.d_inline
        dist_xline = xline_idx * self.d_xline

        # Rotate to world coordinates
        # Inline direction: azimuth angle from north (Y-axis)
        # Crossline direction: azimuth + 90 degrees
        x = self.x_origin + dist_inline * np.sin(az_rad) + dist_xline * np.cos(az_rad)
        y = self.y_origin + dist_inline * np.cos(az_rad) - dist_xline * np.sin(az_rad)

        return x, y

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'n_time': self.n_time,
            'n_inline': self.n_inline,
            'n_xline': self.n_xline,
            'dt': self.dt,
            'd_inline': self.d_inline,
            'd_xline': self.d_xline,
            't0': self.t0,
            'inline_start': self.inline_start,
            'xline_start': self.xline_start,
            'x_origin': self.x_origin,
            'y_origin': self.y_origin,
            'inline_azimuth': self.inline_azimuth,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OutputGrid':
        """Deserialize from dictionary."""
        return cls(**d)


@dataclass
class MigrationConfig:
    """
    Configuration for Kirchhoff Pre-Stack Time Migration.

    Groups parameters into logical categories:
    - Aperture control
    - Traveltime computation
    - Amplitude handling
    - Antialiasing
    - Processing options

    Attributes:
        output_grid: Output volume grid definition

        # Aperture Control
        max_aperture_m: Maximum lateral aperture in meters
        max_angle_deg: Maximum migration angle from vertical (degrees)
        max_offset_m: Maximum source-receiver offset to include (meters)
        min_offset_m: Minimum source-receiver offset to include (meters)
        taper_width: Cosine taper width at aperture edges (fraction 0-1)

        # Traveltime
        traveltime_mode: Straight ray or curved ray
        use_anisotropy: Enable VTI anisotropy corrections

        # Amplitude
        weight_mode: Amplitude weighting scheme
        preserve_amplitudes: If True, apply true-amplitude corrections

        # Antialiasing
        antialias_enabled: Apply dip-dependent antialiasing
        antialias_fmax: Maximum frequency for antialiasing (Hz)

        # Interpolation
        interpolation_mode: Trace interpolation method

        # Processing
        min_fold: Minimum fold threshold (mute if below)
        normalize_by_fold: Divide output by fold count
    """
    # Output grid
    output_grid: OutputGrid = field(default_factory=OutputGrid)

    # Aperture Control
    max_aperture_m: float = 5000.0
    max_angle_deg: float = 60.0
    max_offset_m: float = 10000.0
    min_offset_m: float = 0.0
    taper_width: float = 0.1  # 10% taper

    # Traveltime
    traveltime_mode: TraveltimeMode = TraveltimeMode.STRAIGHT_RAY
    use_anisotropy: bool = False
    anisotropy_method: AnisotropyMethod = AnisotropyMethod.ANELLIPTIC

    # Amplitude
    weight_mode: WeightMode = WeightMode.SPREADING
    preserve_amplitudes: bool = False

    # Antialiasing
    antialias_enabled: bool = True
    antialias_fmax: float = 80.0  # Hz

    # Interpolation
    interpolation_mode: InterpolationMode = InterpolationMode.SINC

    # Processing
    min_fold: int = 1
    normalize_by_fold: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_aperture()
        self._validate_processing()

    def _validate_aperture(self):
        """Validate aperture parameters."""
        if self.max_aperture_m <= 0:
            raise ValueError(f"max_aperture_m must be positive, got {self.max_aperture_m}")
        if not 0 < self.max_angle_deg <= 90:
            raise ValueError(f"max_angle_deg must be in (0, 90], got {self.max_angle_deg}")
        if self.max_offset_m <= 0:
            raise ValueError(f"max_offset_m must be positive, got {self.max_offset_m}")
        if self.min_offset_m < 0:
            raise ValueError(f"min_offset_m must be non-negative, got {self.min_offset_m}")
        if self.min_offset_m >= self.max_offset_m:
            raise ValueError(
                f"min_offset_m ({self.min_offset_m}) must be less than "
                f"max_offset_m ({self.max_offset_m})"
            )
        if not 0 <= self.taper_width <= 1:
            raise ValueError(f"taper_width must be in [0, 1], got {self.taper_width}")

    def _validate_processing(self):
        """Validate processing parameters."""
        if self.min_fold < 1:
            raise ValueError(f"min_fold must be >= 1, got {self.min_fold}")
        if self.antialias_fmax <= 0:
            raise ValueError(f"antialias_fmax must be positive, got {self.antialias_fmax}")

    @property
    def max_angle_rad(self) -> float:
        """Maximum migration angle in radians."""
        return np.radians(self.max_angle_deg)

    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        aniso_str = ""
        if self.use_anisotropy:
            aniso_str = f"\n  Anisotropy: VTI ({self.anisotropy_method.value})"
        return (
            f"Migration Config:\n"
            f"  Output: {self.output_grid.n_inline}x{self.output_grid.n_xline} "
            f"traces, {self.output_grid.n_time} samples\n"
            f"  Aperture: {self.max_aperture_m}m, max angle {self.max_angle_deg}°\n"
            f"  Offset range: {self.min_offset_m}-{self.max_offset_m}m\n"
            f"  Traveltime: {self.traveltime_mode.value}\n"
            f"  Weights: {self.weight_mode.value}\n"
            f"  Antialiasing: {'enabled' if self.antialias_enabled else 'disabled'}"
            f"{aniso_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            'output_grid': self.output_grid.to_dict(),
            'max_aperture_m': self.max_aperture_m,
            'max_angle_deg': self.max_angle_deg,
            'max_offset_m': self.max_offset_m,
            'min_offset_m': self.min_offset_m,
            'taper_width': self.taper_width,
            'traveltime_mode': self.traveltime_mode.value,
            'use_anisotropy': self.use_anisotropy,
            'anisotropy_method': self.anisotropy_method.value,
            'weight_mode': self.weight_mode.value,
            'preserve_amplitudes': self.preserve_amplitudes,
            'antialias_enabled': self.antialias_enabled,
            'antialias_fmax': self.antialias_fmax,
            'interpolation_mode': self.interpolation_mode.value,
            'min_fold': self.min_fold,
            'normalize_by_fold': self.normalize_by_fold,
            'metadata': self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MigrationConfig':
        """Deserialize configuration from dictionary."""
        # Handle nested OutputGrid
        output_grid = OutputGrid.from_dict(d.pop('output_grid'))

        # Convert enum values
        traveltime_mode = TraveltimeMode(d.pop('traveltime_mode', 'straight_ray'))
        weight_mode = WeightMode(d.pop('weight_mode', 'spreading'))
        interpolation_mode = InterpolationMode(d.pop('interpolation_mode', 'sinc'))
        anisotropy_method = AnisotropyMethod(d.pop('anisotropy_method', 'anelliptic'))

        return cls(
            output_grid=output_grid,
            traveltime_mode=traveltime_mode,
            weight_mode=weight_mode,
            interpolation_mode=interpolation_mode,
            anisotropy_method=anisotropy_method,
            **d
        )

    def copy(self) -> 'MigrationConfig':
        """Create a deep copy of this configuration."""
        return MigrationConfig.from_dict(self.to_dict())


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_config(
    n_time: int = 1000,
    n_inline: int = 100,
    n_xline: int = 100,
    dt_ms: float = 4.0,
    d_inline_m: float = 25.0,
    d_xline_m: float = 25.0,
) -> MigrationConfig:
    """
    Create migration config with common default parameters.

    Args:
        n_time: Number of output time samples
        n_inline: Number of output inlines
        n_xline: Number of output crosslines
        dt_ms: Time sampling in milliseconds
        d_inline_m: Inline spacing in meters
        d_xline_m: Crossline spacing in meters

    Returns:
        MigrationConfig with default settings
    """
    output_grid = OutputGrid(
        n_time=n_time,
        n_inline=n_inline,
        n_xline=n_xline,
        dt=dt_ms / 1000.0,  # Convert to seconds
        d_inline=d_inline_m,
        d_xline=d_xline_m,
    )

    return MigrationConfig(output_grid=output_grid)


def create_high_resolution_config(
    n_time: int = 2000,
    n_inline: int = 200,
    n_xline: int = 200,
    max_aperture_m: float = 8000.0,
    max_angle_deg: float = 70.0,
) -> MigrationConfig:
    """
    Create high-resolution migration config for detailed imaging.

    Args:
        n_time: Number of output time samples
        n_inline: Number of output inlines
        n_xline: Number of output crosslines
        max_aperture_m: Maximum aperture
        max_angle_deg: Maximum migration angle

    Returns:
        MigrationConfig optimized for high resolution
    """
    output_grid = OutputGrid(
        n_time=n_time,
        n_inline=n_inline,
        n_xline=n_xline,
        dt=0.002,  # 2 ms
        d_inline=12.5,  # 12.5 m
        d_xline=12.5,
    )

    return MigrationConfig(
        output_grid=output_grid,
        max_aperture_m=max_aperture_m,
        max_angle_deg=max_angle_deg,
        interpolation_mode=InterpolationMode.SINC,
        antialias_enabled=True,
        weight_mode=WeightMode.FULL,
    )
