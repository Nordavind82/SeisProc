"""
3D FKK Filter Configuration

Configuration dataclass for 3D FKK (Frequency-Wavenumber-Wavenumber) velocity cone filter.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


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

        # AGC Parameters
        apply_agc: Whether to apply AGC before filtering and remove after
        agc_window_ms: AGC window length in milliseconds
        agc_max_gain: Maximum AGC amplification factor

        # Frequency Band Selection
        f_min: Minimum frequency for filter action (Hz), None=0
        f_max: Maximum frequency for filter action (Hz), None=Nyquist

        # Temporal Tapering - reduces top/bottom artifacts
        taper_ms_top: Taper length at top of traces (ms)
        taper_ms_bottom: Taper length at bottom of traces (ms)

        # Temporal Pad-Copy - reduces top/bottom edge artifacts from high amplitudes
        pad_time_top_ms: Pad top with copies of first sample, taper only padded zone (ms)
        pad_time_bottom_ms: Pad bottom with copies of last sample, taper only padded zone (ms)

        # Spatial Edge Handling - reduces edge artifacts from FFT
        edge_method: Method for handling spatial edges:
            'none' - No edge treatment (may have artifacts)
            'pad_copy' - Pad with copies of edge traces, taper only the padded zone
        pad_traces_x: Number of traces to pad in X direction (0=auto based on dimension)
        pad_traces_y: Number of traces to pad in Y direction (0=auto based on dimension)
        padding_factor: Extra padding multiplier (1.0=power-of-2, 2.0=double, etc.)
    """
    # Core velocity cone parameters
    v_min: float = 200.0       # m/s
    v_max: float = 1500.0      # m/s
    azimuth_min: float = 0.0   # degrees
    azimuth_max: float = 360.0 # degrees
    taper_width: float = 0.1   # fraction of boundary
    mode: str = 'reject'       # 'reject' or 'pass'

    # AGC parameters
    apply_agc: bool = False
    agc_window_ms: float = 500.0  # ms
    agc_max_gain: float = 10.0

    # Frequency band selection
    f_min: Optional[float] = None  # Hz, None = 0
    f_max: Optional[float] = None  # Hz, None = Nyquist

    # Temporal tapering
    taper_ms_top: float = 0.0    # ms at top
    taper_ms_bottom: float = 0.0  # ms at bottom

    # Temporal pad-copy (for high-amplitude top/bottom edge artifacts)
    pad_time_top_ms: float = 0.0    # ms to pad at top (0=disabled)
    pad_time_bottom_ms: float = 0.0  # ms to pad at bottom (0=disabled)

    # Spatial edge handling
    edge_method: str = 'pad_copy'  # 'none' or 'pad_copy'
    pad_traces_x: int = 0          # Traces to pad in X (0=auto: ~10% of dimension)
    pad_traces_y: int = 0          # Traces to pad in Y (0=auto: ~10% of dimension)
    padding_factor: float = 1.0    # Extra padding multiplier

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

        if self.agc_window_ms <= 0:
            raise ValueError(f"agc_window_ms must be positive, got {self.agc_window_ms}")

        if self.agc_max_gain <= 0:
            raise ValueError(f"agc_max_gain must be positive, got {self.agc_max_gain}")

        if self.f_min is not None and self.f_min < 0:
            raise ValueError(f"f_min must be non-negative, got {self.f_min}")

        if self.f_max is not None and self.f_max <= 0:
            raise ValueError(f"f_max must be positive, got {self.f_max}")

        if self.f_min is not None and self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError(f"f_max ({self.f_max}) must be greater than f_min ({self.f_min})")

        if self.taper_ms_top < 0:
            raise ValueError(f"taper_ms_top must be non-negative, got {self.taper_ms_top}")

        if self.taper_ms_bottom < 0:
            raise ValueError(f"taper_ms_bottom must be non-negative, got {self.taper_ms_bottom}")

        if self.pad_time_top_ms < 0:
            raise ValueError(f"pad_time_top_ms must be non-negative, got {self.pad_time_top_ms}")

        if self.pad_time_bottom_ms < 0:
            raise ValueError(f"pad_time_bottom_ms must be non-negative, got {self.pad_time_bottom_ms}")

        # Validate spatial edge handling
        valid_edge_methods = ('none', 'pad_copy')
        if self.edge_method not in valid_edge_methods:
            raise ValueError(f"edge_method must be one of {valid_edge_methods}, got {self.edge_method}")

        if self.pad_traces_x < 0:
            raise ValueError(f"pad_traces_x must be non-negative, got {self.pad_traces_x}")

        if self.pad_traces_y < 0:
            raise ValueError(f"pad_traces_y must be non-negative, got {self.pad_traces_y}")

        if self.padding_factor < 1.0:
            raise ValueError(f"padding_factor must be >= 1.0, got {self.padding_factor}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FKKConfig':
        """Create from dictionary."""
        # Filter out unknown keys for backward compatibility
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def get_summary(self) -> str:
        """Get one-line summary of configuration."""
        mode_str = "Pass" if self.mode == 'pass' else "Reject"
        parts = [f"{mode_str}: {self.v_min:.0f}-{self.v_max:.0f} m/s"]

        if self.azimuth_min != 0.0 or self.azimuth_max != 360.0:
            parts.append(f"Az: {self.azimuth_min:.0f}-{self.azimuth_max:.0f}")

        if self.f_min is not None or self.f_max is not None:
            f_min_str = f"{self.f_min:.0f}" if self.f_min else "0"
            f_max_str = f"{self.f_max:.0f}" if self.f_max else "Nyq"
            parts.append(f"f: {f_min_str}-{f_max_str} Hz")

        if self.apply_agc:
            parts.append(f"AGC({self.agc_window_ms:.0f}ms)")

        if self.edge_method != 'none':
            parts.append(f"Edge: {self.edge_method}")

        return ", ".join(parts)

    def copy(self) -> 'FKKConfig':
        """Create a copy of this configuration."""
        return FKKConfig(
            v_min=self.v_min,
            v_max=self.v_max,
            azimuth_min=self.azimuth_min,
            azimuth_max=self.azimuth_max,
            taper_width=self.taper_width,
            mode=self.mode,
            apply_agc=self.apply_agc,
            agc_window_ms=self.agc_window_ms,
            agc_max_gain=self.agc_max_gain,
            f_min=self.f_min,
            f_max=self.f_max,
            taper_ms_top=self.taper_ms_top,
            taper_ms_bottom=self.taper_ms_bottom,
            pad_time_top_ms=self.pad_time_top_ms,
            pad_time_bottom_ms=self.pad_time_bottom_ms,
            edge_method=self.edge_method,
            pad_traces_x=self.pad_traces_x,
            pad_traces_y=self.pad_traces_y,
            padding_factor=self.padding_factor
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
