"""
Trace spacing calculation utilities with SEGY scalar support.

Handles coordinate scaling, multiple coordinate sources, and provides
detailed statistics for quality control.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TraceSpacingStats:
    """Statistics for trace spacing calculation."""
    spacing: float  # Median spacing (primary value)
    mean: float
    std: float
    min_spacing: float
    max_spacing: float
    n_spacings: int  # Number of spacing measurements
    coordinate_source: str  # Which header was used (e.g., "GroupX", "SourceX")
    scalar_applied: float  # SEGY scalar that was applied (1.0 if none)
    coordinates_raw: np.ndarray  # Raw coordinate values (before scalar)
    coordinates_scaled: np.ndarray  # Scaled coordinate values
    spacings_all: np.ndarray  # All spacing measurements


def apply_segy_scalar(coordinates: np.ndarray, scalar_value: float) -> np.ndarray:
    """
    Apply SEGY coordinate scalar following SEG-Y standard.

    From SEG-Y spec:
    - If scalar < 0: divide coordinates by abs(scalar)
    - If scalar > 0: multiply coordinates by scalar
    - If scalar == 0: no scaling (use coordinates as-is)

    Args:
        coordinates: Raw coordinate values from SEGY
        scalar_value: Scalar value from scalco or scalel header

    Returns:
        Scaled coordinates in meters

    Examples:
        >>> coords = np.array([100000, 125000, 150000])  # cm
        >>> apply_segy_scalar(coords, -100)  # scalar -100 = divide by 100
        array([1000., 1250., 1500.])  # meters

        >>> coords = np.array([1000, 1250, 1500])  # already in meters
        >>> apply_segy_scalar(coords, 1)  # scalar 1 = multiply by 1
        array([1000., 1250., 1500.])  # meters
    """
    if scalar_value == 0:
        # No scaling
        return coordinates.astype(float)
    elif scalar_value < 0:
        # Negative scalar: divide by absolute value
        return coordinates.astype(float) / abs(scalar_value)
    else:
        # Positive scalar: multiply
        return coordinates.astype(float) * scalar_value


def calculate_trace_spacing_with_stats(
    headers_df: pd.DataFrame,
    default_spacing: float = 25.0
) -> TraceSpacingStats:
    """
    Calculate trace spacing with detailed statistics and SEGY scalar support.

    Priority order for coordinate sources:
    1. receiver_x with scalar_coord (most common for receiver spacing)
    2. source_x with scalar_coord (for shot gathers with source spacing)
    3. receiver_y with scalar_coord (if X not available)
    4. source_y with scalar_coord (source Y)
    5. d3 header (explicit spacing, no scalar needed)
    6. Default (fallback)

    Note: Header names follow SEGY import convention (lowercase with underscores).
          Legacy names (GroupX, ReceiverX, SourceX) also supported for compatibility.

    Args:
        headers_df: DataFrame with trace headers
        default_spacing: Default spacing if calculation fails

    Returns:
        TraceSpacingStats object with spacing and statistics
    """
    # Try coordinate sources in priority order
    # Include both standard names (lowercase) and legacy names (capitalized) for compatibility
    coordinate_headers = [
        ('receiver_x', 'scalar_coord'),  # Standard SEGY import name
        ('ReceiverX', 'scalco'),          # Legacy name (for backward compatibility)
        ('GroupX', 'scalco'),             # Legacy name (older SEG-Y)
        ('source_x', 'scalar_coord'),     # Standard SEGY import name
        ('SourceX', 'scalco'),            # Legacy name
        ('receiver_y', 'scalar_coord'),   # Standard SEGY import name
        ('ReceiverY', 'scalco'),          # Legacy name
        ('GroupY', 'scalco'),             # Legacy name
        ('source_y', 'scalar_coord'),     # Standard SEGY import name
        ('SourceY', 'scalco'),            # Legacy name
    ]

    for coord_header, scalar_header in coordinate_headers:
        if coord_header not in headers_df.columns:
            continue

        if len(headers_df) < 2:
            continue  # Need at least 2 traces to calculate spacing

        # Get raw coordinates
        coords_raw = headers_df[coord_header].values

        # Check if all zeros or all same value
        if np.all(coords_raw == 0) or len(np.unique(coords_raw)) == 1:
            continue

        # Get scalar if available
        scalar = 1.0
        if scalar_header in headers_df.columns:
            scalar_value = headers_df[scalar_header].iloc[0]
            # Use scalar if non-zero
            if scalar_value != 0:
                scalar = scalar_value

        # Apply scalar
        coords_scaled = apply_segy_scalar(coords_raw, scalar)

        # Calculate spacings
        spacings = np.abs(np.diff(coords_scaled))

        # Filter out zero spacings (duplicate coordinates)
        spacings_nonzero = spacings[spacings > 0]

        if len(spacings_nonzero) == 0:
            continue  # All spacings are zero

        # Calculate statistics
        median_spacing = np.median(spacings_nonzero)

        # Sanity check: spacing should be reasonable (0.1m to 1000m)
        if not (0.1 <= median_spacing <= 1000):
            continue

        # Success! Return statistics
        return TraceSpacingStats(
            spacing=float(median_spacing),
            mean=float(np.mean(spacings_nonzero)),
            std=float(np.std(spacings_nonzero)),
            min_spacing=float(np.min(spacings_nonzero)),
            max_spacing=float(np.max(spacings_nonzero)),
            n_spacings=len(spacings_nonzero),
            coordinate_source=coord_header,
            scalar_applied=scalar,
            coordinates_raw=coords_raw,
            coordinates_scaled=coords_scaled,
            spacings_all=spacings_nonzero
        )

    # Try d3 header (explicit spacing, no coordinates)
    if 'd3' in headers_df.columns:
        d3 = headers_df['d3'].iloc[0]
        if 0.1 < d3 < 1000:
            # Create stats with single value
            return TraceSpacingStats(
                spacing=float(d3),
                mean=float(d3),
                std=0.0,
                min_spacing=float(d3),
                max_spacing=float(d3),
                n_spacings=1,
                coordinate_source='d3',
                scalar_applied=1.0,
                coordinates_raw=np.array([d3]),
                coordinates_scaled=np.array([d3]),
                spacings_all=np.array([d3])
            )

    # Fallback to default
    return TraceSpacingStats(
        spacing=default_spacing,
        mean=default_spacing,
        std=0.0,
        min_spacing=default_spacing,
        max_spacing=default_spacing,
        n_spacings=0,
        coordinate_source='default',
        scalar_applied=1.0,
        coordinates_raw=np.array([]),
        coordinates_scaled=np.array([]),
        spacings_all=np.array([])
    )


@dataclass
class OffsetStepAnalysis:
    """Analysis of offset step uniformity."""
    has_gaps: bool
    gap_indices: np.ndarray
    median_step: float
    step_std: float
    step_cv: float  # Coefficient of variation
    n_gaps: int
    suggested_headers: list


def analyze_offset_step_uniformity(
    headers_df: pd.DataFrame,
    gap_threshold_multiplier: float = 3.0
) -> Optional[OffsetStepAnalysis]:
    """
    Analyze uniformity of offset steps to detect sub-gather boundaries.

    Calculates source-receiver offsets and checks if offset steps are uniform.
    Large gaps in offset steps indicate sub-gather boundaries.

    Args:
        headers_df: DataFrame with trace headers
        gap_threshold_multiplier: Gap threshold as multiple of median step

    Returns:
        OffsetStepAnalysis or None if cannot be calculated
    """
    # Get offset column (try multiple names)
    offset_col = None
    for name in ['offset', 'Offset', 'source_receiver_distance']:
        if name in headers_df.columns:
            offset_col = name
            break

    if offset_col is None:
        return None

    offsets = headers_df[offset_col].values

    if len(offsets) < 2:
        return None

    # Calculate offset steps (differences between consecutive offsets)
    offset_steps = np.diff(offsets)
    offset_steps_abs = np.abs(offset_steps)

    # Filter out zeros
    nonzero_steps = offset_steps_abs[offset_steps_abs > 0]

    if len(nonzero_steps) == 0:
        return None

    # Calculate statistics
    median_step = np.median(nonzero_steps)
    step_std = np.std(nonzero_steps)
    step_cv = (step_std / median_step * 100) if median_step > 0 else 0

    # Detect gaps
    gap_threshold = median_step * gap_threshold_multiplier
    gap_mask = offset_steps_abs > gap_threshold
    gap_indices = np.where(gap_mask)[0]

    has_gaps = len(gap_indices) > 0

    # Suggest boundary headers based on gap pattern
    suggested = []
    if has_gaps:
        # Try common headers that might correlate with gaps
        candidates = ['fldr', 'FFID', 'ep', 'shot_point', 'inline', 'ffid']
        for header in candidates:
            if header in headers_df.columns:
                # Check if header values change at gap locations
                values = headers_df[header].values
                changes_at_gaps = sum(
                    values[idx] != values[idx + 1]
                    for idx in gap_indices
                    if idx + 1 < len(values)
                )
                if changes_at_gaps >= len(gap_indices) * 0.8:  # 80% correlation
                    suggested.append(header)

    return OffsetStepAnalysis(
        has_gaps=has_gaps,
        gap_indices=gap_indices,
        median_step=median_step,
        step_std=step_std,
        step_cv=step_cv,
        n_gaps=len(gap_indices),
        suggested_headers=suggested
    )


def format_spacing_stats(stats: TraceSpacingStats) -> str:
    """
    Format trace spacing statistics for display.

    Args:
        stats: TraceSpacingStats object

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append(f"Trace Spacing: {stats.spacing:.2f} m (median)")
    lines.append(f"Source: {stats.coordinate_source}")

    if stats.coordinate_source == 'default':
        lines.append("(using default - no coordinates found)")
    elif stats.coordinate_source == 'd3':
        lines.append("(from d3 header)")
    else:
        lines.append(f"SEGY Scalar: {stats.scalar_applied}")
        lines.append(f"Statistics ({stats.n_spacings} measurements):")
        lines.append(f"  Mean: {stats.mean:.2f} m")
        lines.append(f"  Std Dev: {stats.std:.2f} m")
        lines.append(f"  Range: {stats.min_spacing:.2f} - {stats.max_spacing:.2f} m")

        # Calculate coefficient of variation for quality check
        if stats.mean > 0:
            cv = (stats.std / stats.mean) * 100
            lines.append(f"  Variation: {cv:.1f}%")

            if cv < 5:
                lines.append("  Quality: Excellent (regular spacing)")
            elif cv < 15:
                lines.append("  Quality: Good")
            elif cv < 30:
                lines.append("  Quality: Fair (irregular spacing)")
            else:
                lines.append("  Quality: Poor (highly irregular)")

    return "\n".join(lines)


def calculate_subgather_trace_spacing_with_stats(
    headers_df: pd.DataFrame,
    start_trace: int,
    end_trace: int,
    default_spacing: float = 25.0
) -> TraceSpacingStats:
    """
    Calculate trace spacing for a sub-gather with statistics.

    Args:
        headers_df: Full DataFrame with trace headers
        start_trace: Start index of sub-gather
        end_trace: End index of sub-gather (inclusive)
        default_spacing: Default spacing if calculation fails

    Returns:
        TraceSpacingStats object
    """
    # Extract headers for this sub-gather
    sg_headers = headers_df.iloc[start_trace:end_trace + 1]

    # Use main calculation function
    return calculate_trace_spacing_with_stats(sg_headers, default_spacing)
