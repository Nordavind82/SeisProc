"""
Geometry Utilities for PSTM

Coordinate extraction and manipulation utilities for seismic geometry:
- SEG-Y header coordinate extraction
- Coordinate scaling and transformation
- Offset and azimuth calculations
- CDP computation
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Coordinate Extraction from Headers
# =============================================================================

def extract_coordinates_from_headers(
    headers: Dict[str, np.ndarray],
    sx_key: str = 'SourceX',
    sy_key: str = 'SourceY',
    gx_key: str = 'GroupX',
    gy_key: str = 'GroupY',
    scalar_key: Optional[str] = 'SourceGroupScalar',
    default_scalar: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract source and receiver coordinates from SEG-Y headers.

    Handles coordinate scalar according to SEG-Y convention:
    - If scalar > 0: multiply coordinates by scalar
    - If scalar < 0: divide coordinates by abs(scalar)
    - If scalar = 0: treat as 1.0

    Args:
        headers: Dictionary mapping header names to arrays
        sx_key: Header key for Source X
        sy_key: Header key for Source Y
        gx_key: Header key for Group (receiver) X
        gy_key: Header key for Group (receiver) Y
        scalar_key: Header key for coordinate scalar (optional)
        default_scalar: Default scalar if not in headers

    Returns:
        Tuple of (source_x, source_y, receiver_x, receiver_y, applied_scalar)
        All coordinates in meters.
    """
    # Get raw coordinates
    sx = headers[sx_key].astype(np.float64)
    sy = headers[sy_key].astype(np.float64)
    gx = headers[gx_key].astype(np.float64)
    gy = headers[gy_key].astype(np.float64)

    # Determine coordinate scalar
    if scalar_key and scalar_key in headers:
        # Use first non-zero scalar value
        scalars = headers[scalar_key]
        non_zero_scalars = scalars[scalars != 0]
        if len(non_zero_scalars) > 0:
            scalar = int(non_zero_scalars[0])
        else:
            scalar = int(default_scalar)
    else:
        scalar = int(default_scalar)

    # Apply SEG-Y scalar convention
    if scalar > 0:
        applied_scalar = float(scalar)
    elif scalar < 0:
        applied_scalar = 1.0 / abs(scalar)
    else:
        applied_scalar = 1.0

    logger.debug(f"Applying coordinate scalar: {applied_scalar}")

    # Apply scalar
    source_x = (sx * applied_scalar).astype(np.float32)
    source_y = (sy * applied_scalar).astype(np.float32)
    receiver_x = (gx * applied_scalar).astype(np.float32)
    receiver_y = (gy * applied_scalar).astype(np.float32)

    return source_x, source_y, receiver_x, receiver_y, applied_scalar


def get_coordinate_scalar(
    headers: Dict[str, np.ndarray],
    scalar_key: str = 'SourceGroupScalar',
) -> float:
    """
    Get effective coordinate scalar from headers.

    Args:
        headers: Dictionary mapping header names to arrays
        scalar_key: Header key for coordinate scalar

    Returns:
        Effective scalar multiplier
    """
    if scalar_key not in headers:
        return 1.0

    scalars = headers[scalar_key]
    non_zero = scalars[scalars != 0]

    if len(non_zero) == 0:
        return 1.0

    scalar = int(non_zero[0])

    if scalar > 0:
        return float(scalar)
    elif scalar < 0:
        return 1.0 / abs(scalar)
    else:
        return 1.0


# =============================================================================
# Offset and Azimuth Calculations
# =============================================================================

def compute_offset(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
) -> np.ndarray:
    """
    Compute source-receiver offset.

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates

    Returns:
        Offset array in same units as input coordinates
    """
    return np.sqrt(
        (receiver_x - source_x)**2 +
        (receiver_y - source_y)**2
    ).astype(np.float32)


def compute_azimuth(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    as_degrees: bool = True,
) -> np.ndarray:
    """
    Compute source-to-receiver azimuth.

    Azimuth is measured clockwise from north (positive Y axis).
    0° = North, 90° = East, 180° = South, 270° = West

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates
        as_degrees: If True, return degrees; if False, return radians

    Returns:
        Azimuth array in degrees (0-360) or radians (0-2*pi)
    """
    dx = receiver_x - source_x
    dy = receiver_y - source_y

    # atan2(dx, dy) gives angle from north (Y-axis)
    azimuth_rad = np.arctan2(dx, dy)

    # Convert to 0-2*pi range
    azimuth_rad = (azimuth_rad + 2 * np.pi) % (2 * np.pi)

    if as_degrees:
        return np.degrees(azimuth_rad).astype(np.float32)
    else:
        return azimuth_rad.astype(np.float32)


def compute_signed_offset(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    inline_azimuth: float = 0.0,
) -> np.ndarray:
    """
    Compute signed offset relative to inline direction.

    Positive offset = receiver ahead of source in inline direction
    Negative offset = receiver behind source

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates
        inline_azimuth: Azimuth of inline direction (degrees from north)

    Returns:
        Signed offset array
    """
    dx = receiver_x - source_x
    dy = receiver_y - source_y

    # Rotate to inline direction
    az_rad = np.radians(inline_azimuth)

    # Component along inline direction
    inline_offset = dx * np.sin(az_rad) + dy * np.cos(az_rad)

    # Component perpendicular to inline
    xline_offset = dx * np.cos(az_rad) - dy * np.sin(az_rad)

    # Signed offset (positive = forward, negative = backward)
    sign = np.sign(inline_offset)
    offset = np.sqrt(dx**2 + dy**2)

    return (sign * offset).astype(np.float32)


# =============================================================================
# CDP (Midpoint) Calculations
# =============================================================================

def compute_cdp(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CDP (Common Depth Point / midpoint) coordinates.

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates

    Returns:
        Tuple of (cdp_x, cdp_y) arrays
    """
    cdp_x = ((source_x + receiver_x) / 2).astype(np.float32)
    cdp_y = ((source_y + receiver_y) / 2).astype(np.float32)
    return cdp_x, cdp_y


def compute_inline_xline_from_cdp(
    cdp_x: np.ndarray,
    cdp_y: np.ndarray,
    origin_x: float,
    origin_y: float,
    inline_spacing: float,
    xline_spacing: float,
    inline_azimuth: float = 0.0,
    inline_start: int = 1,
    xline_start: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inline and crossline numbers from CDP coordinates.

    Args:
        cdp_x: CDP X coordinates
        cdp_y: CDP Y coordinates
        origin_x: X coordinate of origin (inline=1, xline=1)
        origin_y: Y coordinate of origin
        inline_spacing: Distance between inlines (meters)
        xline_spacing: Distance between crosslines (meters)
        inline_azimuth: Azimuth of inline direction (degrees from north)
        inline_start: First inline number
        xline_start: First crossline number

    Returns:
        Tuple of (inline_numbers, xline_numbers) as integer arrays
    """
    # Offset from origin
    dx = cdp_x - origin_x
    dy = cdp_y - origin_y

    # Rotate to inline/crossline coordinate system
    az_rad = np.radians(inline_azimuth)

    # Distance along inline direction
    inline_dist = dx * np.sin(az_rad) + dy * np.cos(az_rad)

    # Distance along crossline direction (perpendicular, 90° clockwise)
    xline_dist = dx * np.cos(az_rad) - dy * np.sin(az_rad)

    # Convert to line numbers
    inline_num = np.round(inline_dist / inline_spacing).astype(np.int32) + inline_start
    xline_num = np.round(xline_dist / xline_spacing).astype(np.int32) + xline_start

    return inline_num, xline_num


# =============================================================================
# Distance Calculations
# =============================================================================

def compute_distance(
    x1: Union[float, np.ndarray],
    y1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    y2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute Euclidean distance between points.

    Args:
        x1, y1: First point(s) coordinates
        x2, y2: Second point(s) coordinates

    Returns:
        Distance(s)
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_distance_3d(
    x1: Union[float, np.ndarray],
    y1: Union[float, np.ndarray],
    z1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    y2: Union[float, np.ndarray],
    z2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute 3D Euclidean distance between points.

    Args:
        x1, y1, z1: First point(s) coordinates
        x2, y2, z2: Second point(s) coordinates

    Returns:
        3D distance(s)
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def compute_angle_from_vertical(
    horizontal_distance: Union[float, np.ndarray],
    vertical_distance: Union[float, np.ndarray],
    as_degrees: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute angle from vertical (incidence/emergence angle).

    Args:
        horizontal_distance: Horizontal distance (x, or sqrt(x^2 + y^2))
        vertical_distance: Vertical distance (z or time * velocity)
        as_degrees: If True, return degrees; if False, radians

    Returns:
        Angle from vertical
    """
    # Avoid division by zero
    v_safe = np.maximum(np.abs(vertical_distance), 1e-10)
    angle_rad = np.arctan(np.abs(horizontal_distance) / v_safe)

    if as_degrees:
        return np.degrees(angle_rad)
    else:
        return angle_rad


# =============================================================================
# Coordinate Validation
# =============================================================================

def validate_coordinates(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    max_expected_offset: float = 50000.0,
) -> Dict[str, Any]:
    """
    Validate coordinate arrays for common issues.

    Args:
        source_x: Source X coordinates
        source_y: Source Y coordinates
        receiver_x: Receiver X coordinates
        receiver_y: Receiver Y coordinates
        max_expected_offset: Maximum reasonable offset (meters)

    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
    }

    n_traces = len(source_x)

    # Check for NaN/Inf values
    for name, arr in [('source_x', source_x), ('source_y', source_y),
                      ('receiver_x', receiver_x), ('receiver_y', receiver_y)]:
        if np.any(np.isnan(arr)):
            results['errors'].append(f"{name} contains NaN values")
            results['valid'] = False
        if np.any(np.isinf(arr)):
            results['errors'].append(f"{name} contains Inf values")
            results['valid'] = False

    # Check for zero coordinates (common mistake)
    all_sx_zero = np.all(source_x == 0)
    all_sy_zero = np.all(source_y == 0)
    all_gx_zero = np.all(receiver_x == 0)
    all_gy_zero = np.all(receiver_y == 0)

    if all_sx_zero and all_sy_zero:
        results['warnings'].append("All source coordinates are (0, 0)")
    if all_gx_zero and all_gy_zero:
        results['warnings'].append("All receiver coordinates are (0, 0)")

    # Check offset range
    offsets = compute_offset(source_x, source_y, receiver_x, receiver_y)
    min_offset = np.min(offsets)
    max_offset = np.max(offsets)

    if max_offset > max_expected_offset:
        results['warnings'].append(
            f"Maximum offset ({max_offset:.0f}m) exceeds expected range - "
            f"check coordinate scalar"
        )

    if max_offset < 1.0:
        results['warnings'].append(
            f"Maximum offset ({max_offset:.3f}m) is very small - "
            f"coordinates may need scaling"
        )

    # Check for duplicate traces (same source and receiver)
    combined = source_x + 1j * source_y + 1e6 * (receiver_x + 1j * receiver_y)
    n_unique = len(np.unique(combined))
    if n_unique < n_traces:
        n_duplicates = n_traces - n_unique
        results['warnings'].append(f"{n_duplicates} duplicate source-receiver pairs found")

    results['statistics'] = {
        'n_traces': n_traces,
        'offset_range': (float(min_offset), float(max_offset)),
        'n_unique_positions': n_unique,
    }

    return results


# =============================================================================
# Coordinate Transformations
# =============================================================================

def rotate_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    angle_deg: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate coordinates around an origin point.

    Args:
        x: X coordinates
        y: Y coordinates
        angle_deg: Rotation angle in degrees (positive = counterclockwise)
        origin_x: X coordinate of rotation center
        origin_y: Y coordinate of rotation center

    Returns:
        Tuple of (rotated_x, rotated_y)
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate to origin
    dx = x - origin_x
    dy = y - origin_y

    # Rotate
    x_rot = dx * cos_a - dy * sin_a + origin_x
    y_rot = dx * sin_a + dy * cos_a + origin_y

    return x_rot.astype(np.float32), y_rot.astype(np.float32)


def scale_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    scale_factor: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale coordinates relative to an origin point.

    Args:
        x: X coordinates
        y: Y coordinates
        scale_factor: Scale multiplier
        origin_x: X coordinate of scale center
        origin_y: Y coordinate of scale center

    Returns:
        Tuple of (scaled_x, scaled_y)
    """
    x_scaled = (x - origin_x) * scale_factor + origin_x
    y_scaled = (y - origin_y) * scale_factor + origin_y

    return x_scaled.astype(np.float32), y_scaled.astype(np.float32)
