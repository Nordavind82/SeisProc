"""
Migration Geometry - Survey Geometry Container for PSTM

Stores and manages survey geometry including:
- Source positions
- Receiver positions
- Computed attributes (offsets, azimuths, CDPs)
- Survey statistics and bounds
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MigrationGeometry:
    """
    Survey geometry container for pre-stack migration.

    Stores coordinates for all traces in a gather or dataset,
    and provides methods for computing derived quantities.

    Attributes:
        source_x: Source X coordinates (n_traces,) in meters
        source_y: Source Y coordinates (n_traces,) in meters
        receiver_x: Receiver X coordinates (n_traces,) in meters
        receiver_y: Receiver Y coordinates (n_traces,) in meters
        source_z: Source Z (elevation/depth) if available (n_traces,)
        receiver_z: Receiver Z (elevation/depth) if available (n_traces,)
        trace_indices: Original trace indices in source file
        metadata: Additional geometry metadata
    """
    source_x: np.ndarray
    source_y: np.ndarray
    receiver_x: np.ndarray
    receiver_y: np.ndarray
    source_z: Optional[np.ndarray] = None
    receiver_z: Optional[np.ndarray] = None
    trace_indices: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached computed values
    _offset: Optional[np.ndarray] = field(default=None, repr=False)
    _azimuth: Optional[np.ndarray] = field(default=None, repr=False)
    _cdp_x: Optional[np.ndarray] = field(default=None, repr=False)
    _cdp_y: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate geometry arrays."""
        # Ensure all coordinate arrays are float32
        self.source_x = np.asarray(self.source_x, dtype=np.float32)
        self.source_y = np.asarray(self.source_y, dtype=np.float32)
        self.receiver_x = np.asarray(self.receiver_x, dtype=np.float32)
        self.receiver_y = np.asarray(self.receiver_y, dtype=np.float32)

        # Validate shapes
        n = len(self.source_x)
        if len(self.source_y) != n:
            raise ValueError(f"source_y length {len(self.source_y)} != source_x length {n}")
        if len(self.receiver_x) != n:
            raise ValueError(f"receiver_x length {len(self.receiver_x)} != source_x length {n}")
        if len(self.receiver_y) != n:
            raise ValueError(f"receiver_y length {len(self.receiver_y)} != source_x length {n}")

        if self.source_z is not None:
            self.source_z = np.asarray(self.source_z, dtype=np.float32)
            if len(self.source_z) != n:
                raise ValueError(f"source_z length {len(self.source_z)} != n_traces {n}")

        if self.receiver_z is not None:
            self.receiver_z = np.asarray(self.receiver_z, dtype=np.float32)
            if len(self.receiver_z) != n:
                raise ValueError(f"receiver_z length {len(self.receiver_z)} != n_traces {n}")

        if self.trace_indices is not None:
            self.trace_indices = np.asarray(self.trace_indices, dtype=np.int32)

    @property
    def n_traces(self) -> int:
        """Number of traces in geometry."""
        return len(self.source_x)

    @property
    def offset(self) -> np.ndarray:
        """
        Source-receiver offset in meters.

        Computed as: sqrt((rx - sx)^2 + (ry - sy)^2)
        """
        if self._offset is None:
            self._offset = np.sqrt(
                (self.receiver_x - self.source_x)**2 +
                (self.receiver_y - self.source_y)**2
            ).astype(np.float32)
        return self._offset

    @property
    def azimuth(self) -> np.ndarray:
        """
        Source-to-receiver azimuth in degrees (0-360).

        Measured clockwise from north (Y-axis positive).
        0° = North, 90° = East, 180° = South, 270° = West
        """
        if self._azimuth is None:
            dx = self.receiver_x - self.source_x
            dy = self.receiver_y - self.source_y
            # atan2 gives angle from X-axis, convert to from Y-axis (north)
            az_rad = np.arctan2(dx, dy)  # Note: (dx, dy) for azimuth from north
            az_deg = np.degrees(az_rad)
            # Convert to 0-360 range
            self._azimuth = (az_deg + 360) % 360
            self._azimuth = self._azimuth.astype(np.float32)
        return self._azimuth

    @property
    def cdp_x(self) -> np.ndarray:
        """CDP (midpoint) X coordinate in meters."""
        if self._cdp_x is None:
            self._cdp_x = ((self.source_x + self.receiver_x) / 2).astype(np.float32)
        return self._cdp_x

    @property
    def cdp_y(self) -> np.ndarray:
        """CDP (midpoint) Y coordinate in meters."""
        if self._cdp_y is None:
            self._cdp_y = ((self.source_y + self.receiver_y) / 2).astype(np.float32)
        return self._cdp_y

    def get_source_position(self, idx: int) -> Tuple[float, float]:
        """Get source position for trace index."""
        return float(self.source_x[idx]), float(self.source_y[idx])

    def get_receiver_position(self, idx: int) -> Tuple[float, float]:
        """Get receiver position for trace index."""
        return float(self.receiver_x[idx]), float(self.receiver_y[idx])

    def get_cdp_position(self, idx: int) -> Tuple[float, float]:
        """Get CDP position for trace index."""
        return float(self.cdp_x[idx]), float(self.cdp_y[idx])

    def get_survey_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get survey spatial bounds.

        Returns:
            Dictionary with min/max for each coordinate
        """
        all_x = np.concatenate([self.source_x, self.receiver_x])
        all_y = np.concatenate([self.source_y, self.receiver_y])

        return {
            'x': (float(np.min(all_x)), float(np.max(all_x))),
            'y': (float(np.min(all_y)), float(np.max(all_y))),
            'offset': (float(np.min(self.offset)), float(np.max(self.offset))),
            'azimuth': (float(np.min(self.azimuth)), float(np.max(self.azimuth))),
        }

    def get_offset_range(self) -> Tuple[float, float]:
        """Get min and max offset."""
        return float(np.min(self.offset)), float(np.max(self.offset))

    def get_azimuth_range(self) -> Tuple[float, float]:
        """Get min and max azimuth."""
        return float(np.min(self.azimuth)), float(np.max(self.azimuth))

    def filter_by_offset(
        self,
        min_offset: float,
        max_offset: float,
    ) -> 'MigrationGeometry':
        """
        Create new geometry with traces filtered by offset range.

        Args:
            min_offset: Minimum offset (inclusive)
            max_offset: Maximum offset (inclusive)

        Returns:
            New MigrationGeometry with filtered traces
        """
        mask = (self.offset >= min_offset) & (self.offset <= max_offset)
        return self._apply_mask(mask)

    def filter_by_azimuth(
        self,
        min_azimuth: float,
        max_azimuth: float,
    ) -> 'MigrationGeometry':
        """
        Create new geometry with traces filtered by azimuth range.

        Handles wrap-around (e.g., 350-10 degrees).

        Args:
            min_azimuth: Minimum azimuth in degrees
            max_azimuth: Maximum azimuth in degrees

        Returns:
            New MigrationGeometry with filtered traces
        """
        if min_azimuth <= max_azimuth:
            mask = (self.azimuth >= min_azimuth) & (self.azimuth <= max_azimuth)
        else:
            # Wrap-around case (e.g., 350 to 10 degrees)
            mask = (self.azimuth >= min_azimuth) | (self.azimuth <= max_azimuth)

        return self._apply_mask(mask)

    def filter_by_offset_azimuth(
        self,
        min_offset: float,
        max_offset: float,
        min_azimuth: float,
        max_azimuth: float,
    ) -> 'MigrationGeometry':
        """
        Create new geometry filtered by both offset and azimuth.

        Args:
            min_offset: Minimum offset (inclusive)
            max_offset: Maximum offset (inclusive)
            min_azimuth: Minimum azimuth in degrees
            max_azimuth: Maximum azimuth in degrees

        Returns:
            New MigrationGeometry with filtered traces
        """
        offset_mask = (self.offset >= min_offset) & (self.offset <= max_offset)

        if min_azimuth <= max_azimuth:
            azimuth_mask = (self.azimuth >= min_azimuth) & (self.azimuth <= max_azimuth)
        else:
            azimuth_mask = (self.azimuth >= min_azimuth) | (self.azimuth <= max_azimuth)

        mask = offset_mask & azimuth_mask
        return self._apply_mask(mask)

    def _apply_mask(self, mask: np.ndarray) -> 'MigrationGeometry':
        """Apply boolean mask to create filtered geometry."""
        return MigrationGeometry(
            source_x=self.source_x[mask],
            source_y=self.source_y[mask],
            receiver_x=self.receiver_x[mask],
            receiver_y=self.receiver_y[mask],
            source_z=self.source_z[mask] if self.source_z is not None else None,
            receiver_z=self.receiver_z[mask] if self.receiver_z is not None else None,
            trace_indices=self.trace_indices[mask] if self.trace_indices is not None else None,
            metadata=self.metadata.copy(),
        )

    def get_indices_for_bin(
        self,
        min_offset: float,
        max_offset: float,
        min_azimuth: float,
        max_azimuth: float,
    ) -> np.ndarray:
        """
        Get trace indices falling within offset-azimuth bin.

        Args:
            min_offset: Minimum offset
            max_offset: Maximum offset
            min_azimuth: Minimum azimuth (degrees)
            max_azimuth: Maximum azimuth (degrees)

        Returns:
            Array of trace indices within the bin
        """
        offset_mask = (self.offset >= min_offset) & (self.offset <= max_offset)

        if min_azimuth <= max_azimuth:
            azimuth_mask = (self.azimuth >= min_azimuth) & (self.azimuth <= max_azimuth)
        else:
            azimuth_mask = (self.azimuth >= min_azimuth) | (self.azimuth <= max_azimuth)

        mask = offset_mask & azimuth_mask
        return np.where(mask)[0]

    def get_unique_sources(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get unique source positions.

        Returns:
            Tuple of (unique_x, unique_y) arrays
        """
        # Combine into complex for easy uniqueness check
        combined = self.source_x + 1j * self.source_y
        _, unique_idx = np.unique(combined, return_index=True)
        return self.source_x[unique_idx], self.source_y[unique_idx]

    def get_unique_receivers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get unique receiver positions.

        Returns:
            Tuple of (unique_x, unique_y) arrays
        """
        combined = self.receiver_x + 1j * self.receiver_y
        _, unique_idx = np.unique(combined, return_index=True)
        return self.receiver_x[unique_idx], self.receiver_y[unique_idx]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get geometry statistics summary.

        Returns:
            Dictionary with geometry statistics
        """
        bounds = self.get_survey_bounds()
        unique_src_x, unique_src_y = self.get_unique_sources()
        unique_rcv_x, unique_rcv_y = self.get_unique_receivers()

        return {
            'n_traces': self.n_traces,
            'n_unique_sources': len(unique_src_x),
            'n_unique_receivers': len(unique_rcv_x),
            'x_range': bounds['x'],
            'y_range': bounds['y'],
            'offset_range': bounds['offset'],
            'offset_mean': float(np.mean(self.offset)),
            'offset_std': float(np.std(self.offset)),
            'azimuth_range': bounds['azimuth'],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize geometry to dictionary."""
        result = {
            'source_x': self.source_x.tolist(),
            'source_y': self.source_y.tolist(),
            'receiver_x': self.receiver_x.tolist(),
            'receiver_y': self.receiver_y.tolist(),
            'metadata': self.metadata.copy(),
        }

        if self.source_z is not None:
            result['source_z'] = self.source_z.tolist()
        if self.receiver_z is not None:
            result['receiver_z'] = self.receiver_z.tolist()
        if self.trace_indices is not None:
            result['trace_indices'] = self.trace_indices.tolist()

        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MigrationGeometry':
        """Deserialize geometry from dictionary."""
        return cls(
            source_x=np.array(d['source_x'], dtype=np.float32),
            source_y=np.array(d['source_y'], dtype=np.float32),
            receiver_x=np.array(d['receiver_x'], dtype=np.float32),
            receiver_y=np.array(d['receiver_y'], dtype=np.float32),
            source_z=np.array(d['source_z'], dtype=np.float32) if 'source_z' in d else None,
            receiver_z=np.array(d['receiver_z'], dtype=np.float32) if 'receiver_z' in d else None,
            trace_indices=np.array(d['trace_indices'], dtype=np.int32) if 'trace_indices' in d else None,
            metadata=d.get('metadata', {}),
        )

    @classmethod
    def from_headers(
        cls,
        headers: Dict[str, np.ndarray],
        sx_key: str = 'SourceX',
        sy_key: str = 'SourceY',
        gx_key: str = 'GroupX',
        gy_key: str = 'GroupY',
        coordinate_scalar: float = 1.0,
    ) -> 'MigrationGeometry':
        """
        Create geometry from SEG-Y style headers dictionary.

        Args:
            headers: Dictionary mapping header names to arrays
            sx_key: Key for source X coordinate
            sy_key: Key for source Y coordinate
            gx_key: Key for receiver (group) X coordinate
            gy_key: Key for receiver (group) Y coordinate
            coordinate_scalar: Multiplier for coordinates (e.g., 0.1 for decimeters)

        Returns:
            MigrationGeometry instance
        """
        return cls(
            source_x=headers[sx_key].astype(np.float32) * coordinate_scalar,
            source_y=headers[sy_key].astype(np.float32) * coordinate_scalar,
            receiver_x=headers[gx_key].astype(np.float32) * coordinate_scalar,
            receiver_y=headers[gy_key].astype(np.float32) * coordinate_scalar,
            metadata={'coordinate_scalar': coordinate_scalar},
        )

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"MigrationGeometry(n_traces={self.n_traces}, "
            f"n_sources={stats['n_unique_sources']}, "
            f"offset=[{stats['offset_range'][0]:.0f}, {stats['offset_range'][1]:.0f}]m)"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_synthetic_geometry(
    n_shots: int = 10,
    n_receivers_per_shot: int = 100,
    shot_spacing: float = 100.0,
    receiver_spacing: float = 25.0,
    near_offset: float = 100.0,
    shot_line_azimuth: float = 0.0,
) -> MigrationGeometry:
    """
    Create synthetic marine-style acquisition geometry.

    Args:
        n_shots: Number of shot points
        n_receivers_per_shot: Receivers per shot
        shot_spacing: Distance between shots (meters)
        receiver_spacing: Distance between receivers (meters)
        near_offset: Near offset distance (meters)
        shot_line_azimuth: Azimuth of shot line (degrees from north)

    Returns:
        MigrationGeometry with synthetic positions
    """
    n_total = n_shots * n_receivers_per_shot

    source_x = np.zeros(n_total, dtype=np.float32)
    source_y = np.zeros(n_total, dtype=np.float32)
    receiver_x = np.zeros(n_total, dtype=np.float32)
    receiver_y = np.zeros(n_total, dtype=np.float32)

    az_rad = np.radians(shot_line_azimuth)

    idx = 0
    for shot_idx in range(n_shots):
        # Shot position along line
        shot_dist = shot_idx * shot_spacing
        sx = shot_dist * np.sin(az_rad)
        sy = shot_dist * np.cos(az_rad)

        for rcv_idx in range(n_receivers_per_shot):
            # Receiver position (streamer behind shot)
            rcv_dist = near_offset + rcv_idx * receiver_spacing
            rx = sx - rcv_dist * np.sin(az_rad)  # Behind shot
            ry = sy - rcv_dist * np.cos(az_rad)

            source_x[idx] = sx
            source_y[idx] = sy
            receiver_x[idx] = rx
            receiver_y[idx] = ry
            idx += 1

    return MigrationGeometry(
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        metadata={
            'synthetic': True,
            'n_shots': n_shots,
            'n_receivers_per_shot': n_receivers_per_shot,
            'shot_spacing': shot_spacing,
            'receiver_spacing': receiver_spacing,
        }
    )


def create_land_3d_geometry(
    n_source_lines: int = 5,
    n_sources_per_line: int = 20,
    n_receiver_lines: int = 10,
    n_receivers_per_line: int = 50,
    source_line_spacing: float = 400.0,
    source_spacing: float = 50.0,
    receiver_line_spacing: float = 200.0,
    receiver_spacing: float = 25.0,
) -> MigrationGeometry:
    """
    Create synthetic land 3D orthogonal geometry.

    Args:
        n_source_lines: Number of source lines
        n_sources_per_line: Sources per line
        n_receiver_lines: Number of receiver lines
        n_receivers_per_line: Receivers per line
        source_line_spacing: Distance between source lines (meters)
        source_spacing: Distance between sources on line (meters)
        receiver_line_spacing: Distance between receiver lines (meters)
        receiver_spacing: Distance between receivers on line (meters)

    Returns:
        MigrationGeometry with 3D orthogonal positions
    """
    # Generate source positions (lines along X)
    all_sx = []
    all_sy = []

    for line_idx in range(n_source_lines):
        y_pos = line_idx * source_line_spacing
        for src_idx in range(n_sources_per_line):
            x_pos = src_idx * source_spacing
            all_sx.append(x_pos)
            all_sy.append(y_pos)

    # Generate receiver positions (lines along Y)
    all_rx = []
    all_ry = []

    for line_idx in range(n_receiver_lines):
        x_pos = line_idx * receiver_line_spacing
        for rcv_idx in range(n_receivers_per_line):
            y_pos = rcv_idx * receiver_spacing
            all_rx.append(x_pos)
            all_ry.append(y_pos)

    # Create all source-receiver pairs (full fold)
    n_sources = len(all_sx)
    n_receivers = len(all_rx)
    n_total = n_sources * n_receivers

    source_x = np.zeros(n_total, dtype=np.float32)
    source_y = np.zeros(n_total, dtype=np.float32)
    receiver_x = np.zeros(n_total, dtype=np.float32)
    receiver_y = np.zeros(n_total, dtype=np.float32)

    idx = 0
    for src_idx in range(n_sources):
        for rcv_idx in range(n_receivers):
            source_x[idx] = all_sx[src_idx]
            source_y[idx] = all_sy[src_idx]
            receiver_x[idx] = all_rx[rcv_idx]
            receiver_y[idx] = all_ry[rcv_idx]
            idx += 1

    return MigrationGeometry(
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        metadata={
            'synthetic': True,
            'geometry_type': 'land_3d_orthogonal',
            'n_source_lines': n_source_lines,
            'n_receiver_lines': n_receiver_lines,
        }
    )
