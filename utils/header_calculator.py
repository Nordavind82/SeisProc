"""
Header Calculator for PSTM

Computes derived headers from source/receiver coordinates:
- OFFSET (source-receiver distance)
- AZIMUTH (source-to-receiver azimuth)
- CDP_X, CDP_Y (midpoint coordinates)
- INLINE, CROSSLINE (from CDP and survey geometry)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class HeaderCalculator:
    """
    Calculator for derived seismic trace headers.

    Computes headers that can be derived from basic source/receiver coordinates.
    Used when input data is missing these headers.
    """

    def __init__(self, coordinate_scalar: float = 1.0):
        """
        Initialize header calculator.

        Args:
            coordinate_scalar: Scalar to apply to coordinates (SEG-Y convention)
        """
        self.coordinate_scalar = coordinate_scalar

        # Registry of computation functions
        self._compute_funcs: Dict[str, Callable] = {
            'OFFSET': self._compute_offset,
            'AZIMUTH': self._compute_azimuth,
            'CDP_X': self._compute_cdp_x,
            'CDP_Y': self._compute_cdp_y,
        }

    def can_compute(self, header_name: str) -> bool:
        """Check if a header can be computed."""
        return header_name in self._compute_funcs

    def get_dependencies(self, header_name: str) -> list:
        """Get list of headers required to compute a given header."""
        dependencies = {
            'OFFSET': ['SOURCE_X', 'SOURCE_Y', 'RECEIVER_X', 'RECEIVER_Y'],
            'AZIMUTH': ['SOURCE_X', 'SOURCE_Y', 'RECEIVER_X', 'RECEIVER_Y'],
            'CDP_X': ['SOURCE_X', 'RECEIVER_X'],
            'CDP_Y': ['SOURCE_Y', 'RECEIVER_Y'],
        }
        return dependencies.get(header_name, [])

    def compute(
        self,
        header_name: str,
        headers: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute a derived header.

        Args:
            header_name: Name of header to compute
            headers: Dictionary of available headers

        Returns:
            Computed header values as numpy array
        """
        if header_name not in self._compute_funcs:
            raise ValueError(f"Cannot compute header '{header_name}'")

        # Check dependencies
        deps = self.get_dependencies(header_name)
        missing = [d for d in deps if d not in headers]
        if missing:
            raise ValueError(
                f"Cannot compute '{header_name}': missing dependencies {missing}"
            )

        return self._compute_funcs[header_name](headers)

    def compute_all_possible(
        self,
        headers: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute all possible derived headers from available data.

        Args:
            headers: Dictionary of available headers

        Returns:
            Dictionary of computed headers
        """
        computed = {}

        for name, func in self._compute_funcs.items():
            deps = self.get_dependencies(name)
            if all(d in headers for d in deps):
                try:
                    computed[name] = func(headers)
                    logger.debug(f"Computed header: {name}")
                except Exception as e:
                    logger.warning(f"Failed to compute {name}: {e}")

        return computed

    def _compute_offset(self, headers: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute source-receiver offset."""
        sx = headers['SOURCE_X'].astype(np.float64) * self.coordinate_scalar
        sy = headers['SOURCE_Y'].astype(np.float64) * self.coordinate_scalar
        gx = headers['RECEIVER_X'].astype(np.float64) * self.coordinate_scalar
        gy = headers['RECEIVER_Y'].astype(np.float64) * self.coordinate_scalar

        offset = np.sqrt((gx - sx)**2 + (gy - sy)**2)
        return offset.astype(np.float32)

    def _compute_azimuth(self, headers: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute source-to-receiver azimuth.

        Azimuth is measured clockwise from north (Y-axis positive).
        Returns values in range [0, 360) degrees.
        """
        sx = headers['SOURCE_X'].astype(np.float64) * self.coordinate_scalar
        sy = headers['SOURCE_Y'].astype(np.float64) * self.coordinate_scalar
        gx = headers['RECEIVER_X'].astype(np.float64) * self.coordinate_scalar
        gy = headers['RECEIVER_Y'].astype(np.float64) * self.coordinate_scalar

        dx = gx - sx
        dy = gy - sy

        # atan2(dx, dy) gives angle from north (Y-axis)
        azimuth_rad = np.arctan2(dx, dy)

        # Convert to degrees in [0, 360) range
        azimuth_deg = np.degrees(azimuth_rad)
        azimuth_deg = (azimuth_deg + 360) % 360

        return azimuth_deg.astype(np.float32)

    def _compute_cdp_x(self, headers: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute CDP X coordinate (midpoint)."""
        sx = headers['SOURCE_X'].astype(np.float64)
        gx = headers['RECEIVER_X'].astype(np.float64)

        cdp_x = (sx + gx) / 2 * self.coordinate_scalar
        return cdp_x.astype(np.float64)

    def _compute_cdp_y(self, headers: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute CDP Y coordinate (midpoint)."""
        sy = headers['SOURCE_Y'].astype(np.float64)
        gy = headers['RECEIVER_Y'].astype(np.float64)

        cdp_y = (sy + gy) / 2 * self.coordinate_scalar
        return cdp_y.astype(np.float64)


class InlineXlineCalculator:
    """
    Calculator for inline/crossline numbers from CDP coordinates.

    Requires survey geometry definition (origin, spacing, azimuth).
    """

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        inline_spacing: float,
        xline_spacing: float,
        inline_azimuth: float = 0.0,
        inline_start: int = 1,
        xline_start: int = 1,
    ):
        """
        Initialize inline/crossline calculator.

        Args:
            origin_x: X coordinate of survey origin
            origin_y: Y coordinate of survey origin
            inline_spacing: Distance between inlines (meters)
            xline_spacing: Distance between crosslines (meters)
            inline_azimuth: Azimuth of inline direction (degrees from north)
            inline_start: First inline number
            xline_start: First crossline number
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.inline_spacing = inline_spacing
        self.xline_spacing = xline_spacing
        self.inline_azimuth = inline_azimuth
        self.inline_start = inline_start
        self.xline_start = xline_start

        # Precompute rotation factors
        self._az_rad = np.radians(inline_azimuth)
        self._sin_az = np.sin(self._az_rad)
        self._cos_az = np.cos(self._az_rad)

    def compute_inline_xline(
        self,
        cdp_x: np.ndarray,
        cdp_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute inline and crossline numbers from CDP coordinates.

        Args:
            cdp_x: CDP X coordinates
            cdp_y: CDP Y coordinates

        Returns:
            Tuple of (inline_numbers, xline_numbers) as integer arrays
        """
        # Offset from origin
        dx = cdp_x - self.origin_x
        dy = cdp_y - self.origin_y

        # Rotate to inline/crossline coordinate system
        # Inline direction: azimuth angle from north (Y-axis)
        # Crossline direction: 90° clockwise from inline
        inline_dist = dx * self._sin_az + dy * self._cos_az
        xline_dist = dx * self._cos_az - dy * self._sin_az

        # Convert to line numbers
        inline_num = np.round(inline_dist / self.inline_spacing).astype(np.int32) + self.inline_start
        xline_num = np.round(xline_dist / self.xline_spacing).astype(np.int32) + self.xline_start

        return inline_num, xline_num

    def compute_cdp_from_inline_xline(
        self,
        inline: np.ndarray,
        xline: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CDP coordinates from inline/crossline numbers.

        Args:
            inline: Inline numbers
            xline: Crossline numbers

        Returns:
            Tuple of (cdp_x, cdp_y) coordinate arrays
        """
        # Convert line numbers to distances
        inline_dist = (inline - self.inline_start) * self.inline_spacing
        xline_dist = (xline - self.xline_start) * self.xline_spacing

        # Rotate from inline/crossline to world coordinates
        dx = inline_dist * self._sin_az + xline_dist * self._cos_az
        dy = inline_dist * self._cos_az - xline_dist * self._sin_az

        cdp_x = dx + self.origin_x
        cdp_y = dy + self.origin_y

        return cdp_x.astype(np.float64), cdp_y.astype(np.float64)


def apply_header_mapping(
    raw_headers: Dict[str, np.ndarray],
    mapping: 'HeaderMapping',
    calculator: Optional[HeaderCalculator] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply header mapping to convert raw headers to schema headers.

    Args:
        raw_headers: Headers from input file
        mapping: HeaderMapping configuration
        calculator: Optional HeaderCalculator for computed headers

    Returns:
        Dictionary of headers using schema names
    """
    from models.header_mapping import HeaderMapping

    result = {}

    # Apply direct mappings
    for schema_name, entry in mapping.entries.items():
        if entry.is_computed:
            continue

        if entry.input_name and entry.input_name in raw_headers:
            result[schema_name] = raw_headers[entry.input_name].copy()

            # Apply transform if specified
            if entry.transform:
                result[schema_name] = _apply_transform(
                    result[schema_name],
                    entry.transform
                )

    # Compute derived headers
    if calculator is None:
        calculator = HeaderCalculator(mapping.coordinate_scalar)

    computed = mapping.get_computed_headers()
    for header_name in computed:
        try:
            result[header_name] = calculator.compute(header_name, result)
        except Exception as e:
            logger.warning(f"Could not compute {header_name}: {e}")

    return result


def _apply_transform(
    values: np.ndarray,
    transform: str,
) -> np.ndarray:
    """
    Apply a transformation to header values.

    Supported transforms:
    - 'scale:factor' - multiply by factor
    - 'offset:value' - add value
    - 'negate' - multiply by -1
    - 'abs' - absolute value

    Args:
        values: Input values
        transform: Transform specification string

    Returns:
        Transformed values
    """
    parts = transform.split(':')
    op = parts[0].lower()

    if op == 'scale':
        factor = float(parts[1])
        return values * factor
    elif op == 'offset':
        offset = float(parts[1])
        return values + offset
    elif op == 'negate':
        return -values
    elif op == 'abs':
        return np.abs(values)
    else:
        logger.warning(f"Unknown transform '{op}', returning unchanged")
        return values


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_offset_azimuth(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute offset and azimuth.

    Args:
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates

    Returns:
        Tuple of (offset, azimuth) arrays
    """
    calc = HeaderCalculator()
    headers = {
        'SOURCE_X': source_x,
        'SOURCE_Y': source_y,
        'RECEIVER_X': receiver_x,
        'RECEIVER_Y': receiver_y,
    }

    offset = calc._compute_offset(headers)
    azimuth = calc._compute_azimuth(headers)

    return offset, azimuth


def compute_cdp(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute CDP coordinates.

    Args:
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates

    Returns:
        Tuple of (cdp_x, cdp_y) arrays
    """
    cdp_x = (source_x + receiver_x) / 2
    cdp_y = (source_y + receiver_y) / 2
    return cdp_x, cdp_y
