"""
3D Volume Builder

Constructs a 3D SeismicVolume from 2D gathers using user-specified header keys
to define the inline and crossline axes.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from models.seismic_volume import SeismicVolume

logger = logging.getLogger(__name__)


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


def estimate_volume_size(
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str,
    n_samples: int
) -> Dict[str, Any]:
    """
    Estimate the size of the 3D volume that would be created.

    Args:
        headers_df: DataFrame with trace headers
        inline_key: Header name for inline axis
        xline_key: Header name for crossline axis
        n_samples: Number of time samples

    Returns:
        Dictionary with size estimates
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
    }


def build_volume_from_gathers(
    traces_data: np.ndarray,
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str,
    sample_rate_ms: float,
    coordinate_units: str = 'meters',
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

    # Calculate spatial spacing
    if n_inlines > 1:
        dx = float(np.median(np.diff(inline_vals)))
    else:
        dx = 1.0

    if n_xlines > 1:
        dy = float(np.median(np.diff(xline_vals)))
    else:
        dy = 1.0

    # Convert sample rate to seconds
    dt = sample_rate_ms / 1000.0

    # Handle negative or zero spacing
    dx = abs(dx) if dx != 0 else 25.0
    dy = abs(dy) if dy != 0 else 25.0

    # If spacing seems too small (header values might be indices), use defaults
    if dx < 0.1:
        dx = 25.0  # Default 25m
        logger.warning(f"Inline spacing too small ({dx}), using default 25m")
    if dy < 0.1:
        dy = 25.0
        logger.warning(f"Crossline spacing too small ({dy}), using default 25m")

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
