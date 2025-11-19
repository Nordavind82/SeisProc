"""
Sub-Gather Detection Utility

Detects sub-gather boundaries within a gather based on header value changes.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from models.fk_config import SubGather


def detect_subgathers(
    headers_df: pd.DataFrame,
    boundary_header: str,
    min_traces: int = 8
) -> List[SubGather]:
    """
    Detect sub-gathers based on header value changes.

    Args:
        headers_df: DataFrame with trace headers
        boundary_header: Header column name to use for boundaries
        min_traces: Minimum traces required for a valid sub-gather

    Returns:
        List of SubGather objects

    Raises:
        ValueError: If boundary_header not found or no valid sub-gathers detected
    """
    # Check if header exists
    if boundary_header not in headers_df.columns:
        available = ', '.join(headers_df.columns[:10])
        raise ValueError(
            f"Header '{boundary_header}' not found in gather.\n"
            f"Available headers: {available}..."
        )

    # Extract boundary column values
    boundary_values = headers_df[boundary_header].values
    n_traces = len(boundary_values)

    # Detect where values change
    # diff() gives us differences between consecutive values
    # For strings or other types, convert to categorical codes
    if boundary_values.dtype == object or boundary_values.dtype.name == 'string':
        # Convert to categorical
        categories = pd.Categorical(boundary_values)
        numeric_values = categories.codes
    else:
        numeric_values = boundary_values

    # Find change points
    changes = np.diff(numeric_values) != 0
    change_indices = np.where(changes)[0] + 1  # +1 because diff reduces length by 1

    # Create boundary list: [0, change_points..., n_traces]
    boundaries = [0] + change_indices.tolist() + [n_traces]

    # Create SubGather objects
    subgathers = []
    for i in range(len(boundaries) - 1):
        start_trace = boundaries[i]
        end_trace = boundaries[i + 1] - 1  # Inclusive end
        n_sg_traces = end_trace - start_trace + 1

        # Get boundary value for this sub-gather
        boundary_value = boundary_values[start_trace]

        # Create description
        description = f"{boundary_header}={boundary_value}"

        # Create SubGather
        subgather = SubGather(
            sub_id=i,
            start_trace=start_trace,
            end_trace=end_trace,
            n_traces=n_sg_traces,
            boundary_header=boundary_header,
            boundary_value=boundary_value,
            description=description
        )

        subgathers.append(subgather)

    # Validate sub-gathers
    if len(subgathers) == 0:
        raise ValueError(
            f"No sub-gathers detected with boundary header '{boundary_header}'"
        )

    # Check if all values are the same (no boundaries)
    if len(subgathers) == 1:
        # This is okay - just means no sub-division
        pass

    # Warn about small sub-gathers
    small_subgathers = [sg for sg in subgathers if sg.n_traces < min_traces]
    if small_subgathers:
        print(
            f"Warning: {len(small_subgathers)} sub-gather(s) have < {min_traces} traces "
            f"and may not be suitable for FK filtering"
        )

    return subgathers


def extract_subgather_traces(
    full_traces: np.ndarray,
    subgather: SubGather
) -> np.ndarray:
    """
    Extract traces for a specific sub-gather.

    Args:
        full_traces: Full gather traces (n_samples, n_traces)
        subgather: SubGather object with boundary information

    Returns:
        Sub-gather traces (n_samples, n_subgather_traces)
        Note: This is a view, not a copy
    """
    return full_traces[:, subgather.start_trace:subgather.end_trace + 1]


def calculate_subgather_trace_spacing(
    headers_df: pd.DataFrame,
    subgather: SubGather,
    default_spacing: float = 25.0
) -> float:
    """
    Calculate trace spacing for a specific sub-gather.

    Args:
        headers_df: DataFrame with trace headers
        subgather: SubGather object
        default_spacing: Default spacing if cannot be calculated

    Returns:
        Trace spacing in meters
    """
    try:
        # Extract headers for this sub-gather
        sg_headers = headers_df.iloc[subgather.start_trace:subgather.end_trace + 1]

        # Try to calculate from GroupX coordinates
        if 'GroupX' in sg_headers.columns and len(sg_headers) > 1:
            group_x = sg_headers['GroupX'].values
            spacings = np.abs(np.diff(group_x))
            median_spacing = np.median(spacings[spacings > 0])
            if median_spacing > 0 and median_spacing < 1000:  # Sanity check
                return float(median_spacing)

        # Try d3 header
        if 'd3' in sg_headers.columns:
            d3 = sg_headers['d3'].iloc[0]
            if d3 > 0 and d3 < 1000:
                return float(d3)

    except Exception as e:
        print(f"Warning: Could not calculate trace spacing for sub-gather: {e}")

    return default_spacing


def get_available_boundary_headers(headers_df: pd.DataFrame) -> List[str]:
    """
    Get list of headers suitable for sub-gather boundaries.

    Returns all headers that have at least 2 unique values
    (i.e., can create at least one boundary).

    Args:
        headers_df: DataFrame with trace headers

    Returns:
        List of header names sorted by likelihood of usefulness
    """
    # Prioritized list of common boundary headers (shown first)
    priority_headers = [
        'ReceiverLine', 'SourceLine', 'FFID',
        'ReceiverLineNumber', 'SourceLineNumber',
        'CableNumber', 'GroupNumber', 'ShotPoint',
        'Offset', 'OffsetBin', 'AzimuthBin',
        'InlineNumber', 'CrosslineNumber',
        'Inline', 'Xline', 'IL', 'XL',
        'CDP', 'CMP', 'Ensemble'
    ]

    # Separate headers into priority and other
    priority_available = []
    other_available = []

    for header in headers_df.columns:
        try:
            n_unique = headers_df[header].nunique()

            # Skip headers with only 1 unique value (constant)
            if n_unique < 2:
                continue

            # Skip headers with too many unique values (likely coordinates or sequential)
            # But be more permissive than before
            if n_unique > len(headers_df) * 0.9:  # If >90% traces are unique
                continue

            # Add to appropriate list
            if header in priority_headers:
                priority_available.append(header)
            else:
                other_available.append(header)

        except Exception:
            # Skip headers that can't be analyzed
            continue

    # Return priority headers first, then others
    return priority_available + sorted(other_available)


def validate_subgather_boundaries(
    subgathers: List[SubGather],
    min_traces: int = 8
) -> tuple[bool, List[str]]:
    """
    Validate sub-gather boundaries.

    Args:
        subgathers: List of SubGather objects
        min_traces: Minimum traces required

    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []

    if not subgathers:
        return False, ["No sub-gathers detected"]

    # Check for sub-gathers that are too small
    for sg in subgathers:
        if sg.n_traces < min_traces:
            warnings.append(
                f"Sub-gather {sg.sub_id} ({sg.description}) has only "
                f"{sg.n_traces} traces (minimum {min_traces} required)"
            )

    # Check if we have meaningful sub-division
    if len(subgathers) == 1:
        warnings.append(
            f"Only 1 sub-gather detected - header '{subgathers[0].boundary_header}' "
            f"has constant value. No sub-division performed."
        )

    # Check if too many sub-gathers (might not be meaningful)
    if len(subgathers) > 20:
        warnings.append(
            f"Detected {len(subgathers)} sub-gathers - this may be too many for "
            f"meaningful FK filtering. Consider using a different boundary header."
        )

    # Valid if we have at least one sub-gather with enough traces
    valid_subgathers = [sg for sg in subgathers if sg.n_traces >= min_traces]
    is_valid = len(valid_subgathers) > 0

    return is_valid, warnings
