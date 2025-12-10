"""
3D Volume Builder

Constructs a 3D SeismicVolume from 2D gathers using user-specified header keys
to define the inline and crossline axes.

Distance Calculation:
- Physical distances (dx, dy) are calculated from coordinate headers
- For inline direction: uses coordinate variation along inline axis
- For crossline direction: uses coordinate variation along crossline axis
- User can override calculated values if needed
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from models.seismic_volume import SeismicVolume

logger = logging.getLogger(__name__)


def calculate_spatial_distances(
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str,
    coord_scalar: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate physical spatial distances from CDP coordinates if available.

    Looks for CDP_X/CDP_Y or cdp_x/cdp_y columns to calculate actual distances.
    If not available, returns None and user must enter distances manually.

    Args:
        headers_df: DataFrame with trace headers
        inline_key: Header name for inline axis
        xline_key: Header name for crossline axis
        coord_scalar: Coordinate scalar from SEG-Y (e.g., -100 means divide by 100)

    Returns:
        Dictionary with:
        - dx: Inline spacing in meters (None if cannot calculate)
        - dy: Crossline spacing in meters (None if cannot calculate)
        - dx_source: Source of dx calculation (None if not calculated)
        - dy_source: Source of dy calculation (None if not calculated)
    """
    result = {
        'dx': None,
        'dy': None,
        'dx_source': None,
        'dy_source': None,
    }

    # Apply coordinate scalar
    if coord_scalar < 0:
        scale = 1.0 / abs(coord_scalar)
    else:
        scale = coord_scalar if coord_scalar > 0 else 1.0

    # Look for CDP coordinates (the correct way to determine grid spacing)
    cdp_x_col = next((c for c in ['CDP_X', 'cdp_x', 'CDPX'] if c in headers_df.columns), None)
    cdp_y_col = next((c for c in ['CDP_Y', 'cdp_y', 'CDPY'] if c in headers_df.columns), None)

    if not (cdp_x_col and cdp_y_col):
        logger.info("CDP_X/CDP_Y not found - user must enter dx/dy manually")
        return result

    inline_vals = np.sort(headers_df[inline_key].unique())
    xline_vals = np.sort(headers_df[xline_key].unique())

    # Calculate dx: spacing along inline direction
    # Fix crossline to first value, vary inline
    if len(xline_vals) > 0 and len(inline_vals) >= 2:
        first_xline = xline_vals[0]
        subset = headers_df[headers_df[xline_key] == first_xline]

        if len(subset) >= 2:
            coords = []
            for il_val in np.sort(subset[inline_key].unique()):
                mask = subset[inline_key] == il_val
                x = subset.loc[mask, cdp_x_col].iloc[0] * scale
                y = subset.loc[mask, cdp_y_col].iloc[0] * scale
                coords.append((x, y))

            if len(coords) >= 2:
                distances = []
                for i in range(1, len(coords)):
                    x1, y1 = coords[i-1]
                    x2, y2 = coords[i]
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if dist > 0.01:
                        distances.append(dist)

                if distances:
                    result['dx'] = float(np.median(distances))
                    result['dx_source'] = 'CDP coordinates'
                    logger.info(f"dx={result['dx']:.2f}m from CDP coordinates")

    # Calculate dy: spacing along crossline direction
    # Fix inline to first value, vary crossline
    if len(inline_vals) > 0 and len(xline_vals) >= 2:
        first_inline = inline_vals[0]
        subset = headers_df[headers_df[inline_key] == first_inline]

        if len(subset) >= 2:
            coords = []
            for xl_val in np.sort(subset[xline_key].unique()):
                mask = subset[xline_key] == xl_val
                x = subset.loc[mask, cdp_x_col].iloc[0] * scale
                y = subset.loc[mask, cdp_y_col].iloc[0] * scale
                coords.append((x, y))

            if len(coords) >= 2:
                distances = []
                for i in range(1, len(coords)):
                    x1, y1 = coords[i-1]
                    x2, y2 = coords[i]
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if dist > 0.01:
                        distances.append(dist)

                if distances:
                    result['dy'] = float(np.median(distances))
                    result['dy_source'] = 'CDP coordinates'
                    logger.info(f"dy={result['dy']:.2f}m from CDP coordinates")

    return result


@dataclass
class VolumeGeometry:
    """Geometry information for the built 3D volume."""
    inline_key: str
    xline_key: str
    inline_values: np.ndarray
    xline_values: np.ndarray
    n_inlines: int
    n_xlines: int
    n_samples: int
    dt: float  # seconds
    dx: float  # meters (inline spacing)
    dy: float  # meters (xline spacing)
    n_traces_used: int
    n_traces_total: int
    coverage_percent: float


def get_available_volume_headers(headers_df: pd.DataFrame) -> List[str]:
    """
    Get list of headers suitable for defining 3D volume axes.

    Args:
        headers_df: DataFrame with trace headers

    Returns:
        List of header names that could be used as inline/xline keys
    """
    if headers_df is None or headers_df.empty:
        return []

    # Headers that are typically useful for volume building
    preferred_headers = [
        # Receiver-based
        'ReceiverLine', 'ReceiverStation', 'ReceiverPoint',
        'GroupX', 'GroupY',
        # Source-based
        'SourceLine', 'SourceStation', 'SourcePoint',
        'SourceX', 'SourceY',
        # CDP/CMP based
        'CDP', 'CDPLine', 'Crossline', 'Inline',
        'CMP', 'CMPLine',
        # Generic
        'FFID', 'FieldRecord', 'Channel',
        'TraceNumber', 'TracesPerEnsemble',
    ]

    available = []
    for header in preferred_headers:
        if header in headers_df.columns:
            # Check if header has at least 2 unique values
            n_unique = headers_df[header].nunique()
            if n_unique >= 2:
                available.append(header)

    # Also add any other numeric columns with multiple values
    for col in headers_df.columns:
        if col not in available:
            if pd.api.types.is_numeric_dtype(headers_df[col]):
                n_unique = headers_df[col].nunique()
                if 2 <= n_unique <= 10000:  # Reasonable range for axis
                    available.append(col)

    return available


def estimate_volume_size_fast(
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str,
    n_samples: int
) -> Dict[str, Any]:
    """
    Fast estimate of volume size without distance calculation.
    Used for auto-detect header combination scoring.
    """
    inline_vals = headers_df[inline_key].unique()
    xline_vals = headers_df[xline_key].unique()

    n_inlines = len(inline_vals)
    n_xlines = len(xline_vals)
    n_total = n_samples * n_inlines * n_xlines

    size_bytes = n_total * 4
    size_mb = size_bytes / (1024 * 1024)

    n_traces = len(headers_df)
    theoretical_traces = n_inlines * n_xlines
    coverage = (n_traces / theoretical_traces) * 100 if theoretical_traces > 0 else 0

    unique_pairs = headers_df.groupby([inline_key, xline_key]).size()
    n_unique_positions = len(unique_pairs)

    return {
        'n_inlines': n_inlines,
        'n_xlines': n_xlines,
        'size_mb': size_mb,
        'n_traces': n_traces,
        'theoretical_traces': theoretical_traces,
        'coverage_percent': coverage,
        'n_unique_positions': n_unique_positions,
    }


def estimate_volume_size(
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str,
    n_samples: int,
    coord_scalar: float = 1.0
) -> Dict[str, Any]:
    """
    Estimate the size of the 3D volume that would be created.

    Args:
        headers_df: DataFrame with trace headers
        inline_key: Header name for inline axis
        xline_key: Header name for crossline axis
        n_samples: Number of time samples
        coord_scalar: Coordinate scalar from SEG-Y headers

    Returns:
        Dictionary with size estimates including calculated spatial distances
    """
    inline_vals = headers_df[inline_key].unique()
    xline_vals = headers_df[xline_key].unique()

    n_inlines = len(inline_vals)
    n_xlines = len(xline_vals)
    n_total = n_samples * n_inlines * n_xlines

    size_bytes = n_total * 4  # float32
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)

    # Count how many traces we actually have
    n_traces = len(headers_df)
    theoretical_traces = n_inlines * n_xlines
    coverage = (n_traces / theoretical_traces) * 100 if theoretical_traces > 0 else 0

    # Count actual unique (inline, xline) pairs to check for duplicates
    unique_pairs = headers_df.groupby([inline_key, xline_key]).size()
    n_unique_positions = len(unique_pairs)
    n_duplicates = n_traces - n_unique_positions

    # Calculate spatial distances from CDP coordinates (if available)
    distances = calculate_spatial_distances(
        headers_df, inline_key, xline_key, coord_scalar
    )

    return {
        'n_samples': n_samples,
        'n_inlines': n_inlines,
        'n_xlines': n_xlines,
        'shape': (n_samples, n_inlines, n_xlines),
        'size_mb': size_mb,
        'size_gb': size_gb,
        'n_traces': n_traces,
        'theoretical_traces': theoretical_traces,
        'coverage_percent': coverage,
        'inline_range': (inline_vals.min(), inline_vals.max()),
        'xline_range': (xline_vals.min(), xline_vals.max()),
        'n_unique_positions': n_unique_positions,
        'n_duplicates': n_duplicates,
        # Distance information (None if CDP coords not available)
        'dx': distances['dx'],
        'dy': distances['dy'],
        'dx_source': distances['dx_source'],
        'dy_source': distances['dy_source'],
    }


def build_volume_from_gathers(
    traces_data: np.ndarray,
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str,
    sample_rate_ms: float,
    coordinate_units: str = 'meters',
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    coord_scalar: float = 1.0,
    progress_callback: Optional[callable] = None
) -> Tuple[SeismicVolume, VolumeGeometry]:
    """
    Build a 3D SeismicVolume from 2D gather data.

    Args:
        traces_data: 2D array of traces (n_samples, n_traces)
        headers_df: DataFrame with trace headers (one row per trace)
        inline_key: Header name for inline axis (X direction)
        xline_key: Header name for crossline axis (Y direction)
        sample_rate_ms: Sample rate in milliseconds
        coordinate_units: 'meters' or 'feet'
        dx: Inline spacing in meters (if None, calculated from coordinates)
        dy: Crossline spacing in meters (if None, calculated from coordinates)
        coord_scalar: Coordinate scalar from SEG-Y headers
        progress_callback: Optional callback(percent, message)

    Returns:
        Tuple of (SeismicVolume, VolumeGeometry)

    Raises:
        ValueError: If headers not found or data is inconsistent
    """
    n_samples, n_traces = traces_data.shape

    if inline_key not in headers_df.columns:
        raise ValueError(f"Inline key '{inline_key}' not found in headers. "
                        f"Available: {list(headers_df.columns[:10])}")

    if xline_key not in headers_df.columns:
        raise ValueError(f"Crossline key '{xline_key}' not found in headers. "
                        f"Available: {list(headers_df.columns[:10])}")

    if len(headers_df) != n_traces:
        raise ValueError(f"Header count ({len(headers_df)}) doesn't match "
                        f"trace count ({n_traces})")

    # Get unique values for each axis
    inline_vals = np.sort(headers_df[inline_key].unique())
    xline_vals = np.sort(headers_df[xline_key].unique())

    n_inlines = len(inline_vals)
    n_xlines = len(xline_vals)

    logger.info(f"Building volume: {n_samples} samples x {n_inlines} inlines x {n_xlines} xlines")

    # Create mapping from value to index
    inline_to_idx = {v: i for i, v in enumerate(inline_vals)}
    xline_to_idx = {v: i for i, v in enumerate(xline_vals)}

    # Initialize 3D volume with zeros (or NaN for missing traces)
    volume_data = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)

    # Fill volume with trace data
    traces_placed = 0
    for trace_idx in range(n_traces):
        inline_val = headers_df.iloc[trace_idx][inline_key]
        xline_val = headers_df.iloc[trace_idx][xline_key]

        i_idx = inline_to_idx.get(inline_val)
        j_idx = xline_to_idx.get(xline_val)

        if i_idx is not None and j_idx is not None:
            volume_data[:, i_idx, j_idx] = traces_data[:, trace_idx]
            traces_placed += 1

        if progress_callback and trace_idx % 1000 == 0:
            pct = (trace_idx / n_traces) * 100
            progress_callback(pct, f"Placing traces: {trace_idx}/{n_traces}")

    # Require explicit dx/dy values
    if dx is None or dy is None:
        raise ValueError(
            f"dx and dy must be provided explicitly. "
            f"Got dx={dx}, dy={dy}. "
            f"Use calculate_spatial_distances() to compute values from coordinates."
        )

    logger.info(f"Building volume with dx={dx:.2f}m, dy={dy:.2f}m")

    # Convert sample rate to seconds
    dt = sample_rate_ms / 1000.0

    # Validate spacing - must be positive
    if dx <= 0 or dy <= 0:
        raise ValueError(f"dx and dy must be positive. Got dx={dx}, dy={dy}")

    # Create SeismicVolume
    volume = SeismicVolume(
        data=volume_data,
        dt=dt,
        dx=dx,
        dy=dy,
        metadata={
            'inline_key': inline_key,
            'xline_key': xline_key,
            'inline_values': inline_vals.tolist(),
            'xline_values': xline_vals.tolist(),
            'coordinate_units': coordinate_units,
            'built_from_gathers': True,
        }
    )

    # Create geometry info
    geometry = VolumeGeometry(
        inline_key=inline_key,
        xline_key=xline_key,
        inline_values=inline_vals,
        xline_values=xline_vals,
        n_inlines=n_inlines,
        n_xlines=n_xlines,
        n_samples=n_samples,
        dt=dt,
        dx=dx,
        dy=dy,
        n_traces_used=traces_placed,
        n_traces_total=n_traces,
        coverage_percent=(traces_placed / (n_inlines * n_xlines)) * 100
    )

    logger.info(f"Volume built: {volume.shape}, {volume.memory_mb():.1f} MB, "
               f"{geometry.coverage_percent:.1f}% coverage")

    return volume, geometry


def extract_traces_from_volume(
    volume: SeismicVolume,
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str
) -> np.ndarray:
    """
    Extract traces from a 3D volume back to 2D gather format.

    Args:
        volume: 3D SeismicVolume
        headers_df: DataFrame with trace headers defining extraction points
        inline_key: Header name for inline axis
        xline_key: Header name for crossline axis

    Returns:
        2D array of traces (n_samples, n_traces)
    """
    n_traces = len(headers_df)
    n_samples = volume.nt

    # Get axis mappings from volume metadata
    inline_vals = np.array(volume.metadata.get('inline_values', []))
    xline_vals = np.array(volume.metadata.get('xline_values', []))

    if len(inline_vals) == 0 or len(xline_vals) == 0:
        # Reconstruct from volume shape
        inline_vals = np.arange(volume.nx)
        xline_vals = np.arange(volume.ny)

    inline_to_idx = {v: i for i, v in enumerate(inline_vals)}
    xline_to_idx = {v: i for i, v in enumerate(xline_vals)}

    # Extract traces
    traces_data = np.zeros((n_samples, n_traces), dtype=np.float32)

    for trace_idx in range(n_traces):
        inline_val = headers_df.iloc[trace_idx][inline_key]
        xline_val = headers_df.iloc[trace_idx][xline_key]

        i_idx = inline_to_idx.get(inline_val)
        j_idx = xline_to_idx.get(xline_val)

        if i_idx is not None and j_idx is not None:
            traces_data[:, trace_idx] = volume.data[:, i_idx, j_idx]

    return traces_data
