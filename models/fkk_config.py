"""
3D FKK Filter Configuration

Configuration dataclass for 3D FKK (Frequency-Wavenumber-Wavenumber) velocity cone filter.
Follows the same patterns as FKFilterConfig for 2D FK filtering.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class FKKConfig:
    """
    Configuration for 3D FKK velocity cone filter.

    The velocity cone filter passes or rejects energy based on apparent velocity:
        v = f / sqrt(kx² + ky²)

    where f is frequency and kx, ky are wavenumbers in X and Y directions.

    Attributes:
        v_min: Minimum velocity boundary (m/s)
        v_max: Maximum velocity boundary (m/s)
        azimuth_min: Minimum azimuth angle in degrees (0-360, 0=+kx direction)
        azimuth_max: Maximum azimuth angle in degrees (0-360)
        taper_width: Taper width as fraction of velocity boundary (0.0-0.5)
        mode: Filter mode - 'reject' removes velocities in range, 'pass' keeps them
    """
    v_min: float = 200.0       # m/s
    v_max: float = 1500.0      # m/s
    azimuth_min: float = 0.0   # degrees
    azimuth_max: float = 360.0 # degrees
    taper_width: float = 0.1   # fraction of boundary
    mode: str = 'reject'       # 'reject' or 'pass'

    def __post_init__(self):
        """Validate configuration."""
        if self.v_min <= 0:
            raise ValueError(f"v_min must be positive, got {self.v_min}")

        if self.v_max <= self.v_min:
            raise ValueError(f"v_max ({self.v_max}) must be greater than v_min ({self.v_min})")

        if not 0.0 <= self.azimuth_min <= 360.0:
            raise ValueError(f"azimuth_min must be in [0, 360], got {self.azimuth_min}")

        if not 0.0 <= self.azimuth_max <= 360.0:
            raise ValueError(f"azimuth_max must be in [0, 360], got {self.azimuth_max}")

        if not 0.0 <= self.taper_width <= 0.5:
            raise ValueError(f"taper_width must be in [0, 0.5], got {self.taper_width}")

        if self.mode not in ('reject', 'pass'):
            raise ValueError(f"mode must be 'reject' or 'pass', got {self.mode}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FKKConfig':
        """Create from dictionary."""
        return cls(**data)

    def get_summary(self) -> str:
        """Get one-line summary of configuration."""
        mode_str = "Pass" if self.mode == 'pass' else "Reject"
        az_str = ""
        if self.azimuth_min != 0.0 or self.azimuth_max != 360.0:
            az_str = f", Az: {self.azimuth_min:.0f}-{self.azimuth_max:.0f}°"
        return f"{mode_str}: {self.v_min:.0f}-{self.v_max:.0f} m/s{az_str}"

    def copy(self) -> 'FKKConfig':
        """Create a copy of this configuration."""
        return FKKConfig(
            v_min=self.v_min,
            v_max=self.v_max,
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            taper_width=self.taper_width,
            mode=self.mode
        )


# Preset configurations for common use cases
FKK_PRESETS = {
    'Ground Roll Rejection': FKKConfig(
        v_min=100.0,
        v_max=800.0,
        taper_width=0.15,
        mode='reject'
    ),
    'Linear Noise Rejection': FKKConfig(
        v_min=200.0,
        v_max=1200.0,
        taper_width=0.1,
        mode='reject'
    ),
    'Reflection Pass': FKKConfig(
        v_min=1500.0,
        v_max=6000.0,
        taper_width=0.1,
        mode='pass'
    ),
    'Low Velocity Rejection': FKKConfig(
        v_min=50.0,
        v_max=500.0,
        taper_width=0.2,
        mode='reject'
    ),
}


def get_preset(name: str) -> FKKConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name (see FKK_PRESETS keys)

    Returns:
        Copy of preset configuration

    Raises:
        KeyError: If preset not found
    """
    if name not in FKK_PRESETS:
        raise KeyError(f"Preset '{name}' not found. Available: {list(FKK_PRESETS.keys())}")
    return FKK_PRESETS[name].copy()
