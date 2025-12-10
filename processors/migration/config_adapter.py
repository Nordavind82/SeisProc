"""
Config Adapter - Maps wizard configuration to MigrationEngine parameters.

This module provides the bridge between the PSTM wizard configuration dictionary
and the MigrationEngine's migrate_bin_full() method.

Usage:
    adapter = ConfigAdapter(wizard_config)
    engine_params = adapter.get_engine_params(traces, geometry)
    image, fold = engine.migrate_bin_full(**engine_params)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MigrationParams:
    """
    Extracted migration parameters ready for MigrationEngine.

    These map directly to migrate_bin_full() or migrate_bin_time_domain() arguments.
    """
    # Output grid
    origin_x: float
    origin_y: float
    il_spacing: float
    xl_spacing: float
    azimuth_deg: float
    n_il: int
    n_xl: int

    # Time axis
    dt_ms: float
    t_min_ms: float
    n_samples: int

    # Velocity
    velocity_mps: float

    # Migration parameters
    max_aperture_m: float
    max_angle_deg: float

    # Advanced parameters
    tile_size: int = 100
    use_time_domain: bool = False
    use_kdtree: bool = False
    sample_batch_size: int = 200


class ConfigAdapter:
    """
    Adapts wizard configuration dictionary to MigrationEngine parameters.

    The wizard uses config keys like:
        - x_origin, y_origin (grid origin)
        - output_bin_il, output_bin_xl (bin sizes in meters)
        - grid_azimuth_deg (inline azimuth)
        - inline_min, inline_max, inline_step (inline range)
        - xline_min, xline_max, xline_step (crossline range)
        - time_min_ms, time_max_ms, dt_ms (time axis)
        - velocity_v0 (constant velocity)
        - max_aperture_m, max_angle_deg (migration parameters)

    MigrationEngine.migrate_bin_full() expects:
        - origin_x, origin_y
        - il_spacing, xl_spacing
        - azimuth_deg
        - n_il, n_xl
        - dt_ms, t_min_ms
        - velocity_mps
        - max_aperture_m, max_angle_deg
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with wizard configuration.

        Args:
            config: Dictionary from PSTM wizard (job_configured signal)
        """
        self.config = config
        self._params: Optional[MigrationParams] = None

    @property
    def params(self) -> MigrationParams:
        """Get extracted parameters, computing if needed."""
        if self._params is None:
            self._params = self._extract_params()
        return self._params

    def _extract_params(self) -> MigrationParams:
        """Extract MigrationParams from wizard config."""
        cfg = self.config

        # Output grid origin
        origin_x = cfg.get('x_origin', 0.0)
        origin_y = cfg.get('y_origin', 0.0)

        # Bin sizes (output spacing)
        # Fallback chain: output_bin_il -> input_bin_il -> 25.0
        il_spacing = cfg.get('output_bin_il', cfg.get('input_bin_il', 25.0))
        xl_spacing = cfg.get('output_bin_xl', cfg.get('input_bin_xl', 25.0))

        # Grid azimuth
        azimuth_deg = cfg.get('grid_azimuth_deg', 0.0)

        # Inline/crossline counts
        inline_min = cfg.get('inline_min', 1)
        inline_max = cfg.get('inline_max', 100)
        inline_step = cfg.get('inline_step', 1)
        n_il = max(1, (inline_max - inline_min) // inline_step + 1)

        xline_min = cfg.get('xline_min', 1)
        xline_max = cfg.get('xline_max', 100)
        xline_step = cfg.get('xline_step', 1)
        n_xl = max(1, (xline_max - xline_min) // xline_step + 1)

        # Time axis
        dt_ms = cfg.get('dt_ms', 4.0)
        if dt_ms <= 0:
            dt_ms = 4.0  # Fall back to default for invalid values
        t_min_ms = cfg.get('time_min_ms', 0.0)
        t_max_ms = cfg.get('time_max_ms', 6000.0)
        n_samples = max(1, int((t_max_ms - t_min_ms) / dt_ms) + 1)

        # Velocity (for now, constant velocity only)
        velocity_mps = cfg.get('velocity_v0', 2500.0)

        # Migration parameters
        max_aperture_m = cfg.get('max_aperture_m', 5000.0)
        max_angle_deg = cfg.get('max_angle_deg', 60.0)

        # Advanced parameters (with defaults)
        tile_size = cfg.get('tile_size', 100)
        use_time_domain = cfg.get('use_time_domain', False)
        use_kdtree = cfg.get('use_kdtree', False)
        sample_batch_size = cfg.get('sample_batch_size', 200)

        logger.info(f"ConfigAdapter extracted params: "
                   f"grid={n_il}x{n_xl}, dt={dt_ms}ms, v={velocity_mps}m/s, "
                   f"aperture={max_aperture_m}m, angle={max_angle_deg}°")
        logger.info(f"  Advanced: tile_size={tile_size}, time_domain={use_time_domain}, "
                   f"kdtree={use_kdtree}, sample_batch={sample_batch_size}")

        return MigrationParams(
            origin_x=origin_x,
            origin_y=origin_y,
            il_spacing=il_spacing,
            xl_spacing=xl_spacing,
            azimuth_deg=azimuth_deg,
            n_il=n_il,
            n_xl=n_xl,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            n_samples=n_samples,
            velocity_mps=velocity_mps,
            max_aperture_m=max_aperture_m,
            max_angle_deg=max_angle_deg,
            tile_size=tile_size,
            use_time_domain=use_time_domain,
            use_kdtree=use_kdtree,
            sample_batch_size=sample_batch_size,
        )

    def get_engine_params(
        self,
        traces: np.ndarray,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        normalize: bool = True,
        min_fold: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        enable_profiling: bool = False,
        use_time_dependent_aperture: bool = False,
    ) -> Dict[str, Any]:
        """
        Get complete parameter dict for MigrationEngine.migrate_bin_full().

        Args:
            traces: Trace data (n_samples, n_traces)
            source_x: Source X coordinates (n_traces,)
            source_y: Source Y coordinates (n_traces,)
            receiver_x: Receiver X coordinates (n_traces,)
            receiver_y: Receiver Y coordinates (n_traces,)
            normalize: Whether to normalize by fold
            min_fold: Minimum fold for normalization
            progress_callback: Optional progress callback
            enable_profiling: If True, logs detailed timing breakdown
            use_time_dependent_aperture: If True, aperture varies with depth/time

        Returns:
            Dict ready to pass as **kwargs to migrate_bin_full()
        """
        p = self.params
        return {
            # Trace data
            'traces': traces,
            # Geometry
            'source_x': source_x,
            'source_y': source_y,
            'receiver_x': receiver_x,
            'receiver_y': receiver_y,
            # Output grid
            'origin_x': p.origin_x,
            'origin_y': p.origin_y,
            'il_spacing': p.il_spacing,
            'xl_spacing': p.xl_spacing,
            'azimuth_deg': p.azimuth_deg,
            'n_il': p.n_il,
            'n_xl': p.n_xl,
            # Time axis
            'dt_ms': p.dt_ms,
            't_min_ms': p.t_min_ms,
            'n_times': p.n_samples,  # Output time samples from wizard settings
            # Velocity
            'velocity_mps': p.velocity_mps,
            # Migration parameters
            'max_aperture_m': p.max_aperture_m,
            'max_angle_deg': p.max_angle_deg,
            # Options
            'normalize': normalize,
            'min_fold': min_fold,
            'progress_callback': progress_callback,
            'enable_profiling': enable_profiling,
            'use_time_dependent_aperture': use_time_dependent_aperture,
        }

    def get_time_domain_params(
        self,
        traces: np.ndarray,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        normalize: bool = True,
        min_fold: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        enable_profiling: bool = False,
        use_time_dependent_aperture: bool = True,
    ) -> Dict[str, Any]:
        """
        Get complete parameter dict for MigrationEngine.migrate_bin_time_domain().

        This is for the fast time-domain migration that eliminates the depth loop.

        Args:
            traces: Trace data (n_samples, n_traces)
            source_x: Source X coordinates (n_traces,)
            source_y: Source Y coordinates (n_traces,)
            receiver_x: Receiver X coordinates (n_traces,)
            receiver_y: Receiver Y coordinates (n_traces,)
            normalize: Whether to normalize by fold
            min_fold: Minimum fold for normalization
            progress_callback: Optional progress callback
            enable_profiling: If True, logs detailed timing breakdown
            use_time_dependent_aperture: If True, aperture varies with time

        Returns:
            Dict ready to pass as **kwargs to migrate_bin_time_domain()
        """
        p = self.params
        return {
            # Trace data
            'traces': traces,
            # Geometry
            'source_x': source_x,
            'source_y': source_y,
            'receiver_x': receiver_x,
            'receiver_y': receiver_y,
            # Output grid
            'origin_x': p.origin_x,
            'origin_y': p.origin_y,
            'il_spacing': p.il_spacing,
            'xl_spacing': p.xl_spacing,
            'azimuth_deg': p.azimuth_deg,
            'n_il': p.n_il,
            'n_xl': p.n_xl,
            # Time axis
            'dt_ms': p.dt_ms,
            't_min_ms': p.t_min_ms,
            'n_times': p.n_samples,
            # Velocity
            'velocity_mps': p.velocity_mps,
            # Migration parameters
            'max_aperture_m': p.max_aperture_m,
            'max_angle_deg': p.max_angle_deg,
            'tile_size': p.tile_size,
            # Options
            'normalize': normalize,
            'min_fold': min_fold,
            'progress_callback': progress_callback,
            'enable_profiling': enable_profiling,
            'use_time_dependent_aperture': use_time_dependent_aperture,
        }

    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration for migration.

        Checks both raw config values and computed params.

        Returns:
            (is_valid, message) tuple
        """
        issues = []
        cfg = self.config

        # Check raw config values first (before defaults applied)
        raw_dt = cfg.get('dt_ms', 4.0)
        if raw_dt <= 0:
            issues.append(f"Invalid sample interval: {raw_dt}ms")

        raw_inline_min = cfg.get('inline_min', 1)
        raw_inline_max = cfg.get('inline_max', 100)
        if raw_inline_max < raw_inline_min:
            issues.append(f"Invalid inline range: max ({raw_inline_max}) < min ({raw_inline_min})")

        raw_xline_min = cfg.get('xline_min', 1)
        raw_xline_max = cfg.get('xline_max', 100)
        if raw_xline_max < raw_xline_min:
            issues.append(f"Invalid crossline range: max ({raw_xline_max}) < min ({raw_xline_min})")

        # Now check computed params (includes fallbacks/defaults)
        p = self.params

        # Check grid size
        if p.n_il <= 0:
            issues.append(f"Invalid inline count: {p.n_il}")
        if p.n_xl <= 0:
            issues.append(f"Invalid crossline count: {p.n_xl}")
        if p.n_il * p.n_xl > 1e8:
            issues.append(f"Grid too large: {p.n_il}x{p.n_xl} = {p.n_il*p.n_xl:.0e} points")

        # Check time axis
        if p.n_samples <= 0:
            issues.append(f"Invalid sample count: {p.n_samples}")

        # Check velocity
        if p.velocity_mps <= 0:
            issues.append(f"Invalid velocity: {p.velocity_mps} m/s")
        elif p.velocity_mps < 1000 or p.velocity_mps > 8000:
            issues.append(f"Unusual velocity: {p.velocity_mps} m/s (typical range: 1000-8000)")

        # Check aperture
        if p.max_aperture_m <= 0:
            issues.append(f"Invalid aperture: {p.max_aperture_m}m")

        # Check angle
        if p.max_angle_deg <= 0 or p.max_angle_deg > 90:
            issues.append(f"Invalid max angle: {p.max_angle_deg}°")

        if issues:
            return False, "; ".join(issues)
        return True, "Configuration valid"

    def get_summary(self) -> str:
        """Get human-readable summary of migration parameters."""
        p = self.params
        lines = [
            "Migration Parameters:",
            f"  Output grid: {p.n_il} IL x {p.n_xl} XL",
            f"  Origin: ({p.origin_x:.1f}, {p.origin_y:.1f})",
            f"  Bin sizes: {p.il_spacing:.1f}m IL, {p.xl_spacing:.1f}m XL",
            f"  Azimuth: {p.azimuth_deg:.1f}°",
            f"  Time: {p.t_min_ms:.0f} - {p.t_min_ms + (p.n_samples-1)*p.dt_ms:.0f} ms @ {p.dt_ms:.1f}ms",
            f"  Velocity: {p.velocity_mps:.0f} m/s",
            f"  Aperture: {p.max_aperture_m:.0f}m, angle: {p.max_angle_deg:.0f}°",
        ]
        return "\n".join(lines)


def create_adapter_from_wizard(config: Dict[str, Any]) -> ConfigAdapter:
    """
    Factory function to create ConfigAdapter from wizard config.

    This is the main entry point for integration.

    Args:
        config: Configuration dict from PSTM wizard

    Returns:
        ConfigAdapter instance
    """
    return ConfigAdapter(config)
