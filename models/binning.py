"""
Offset-Azimuth Binning for Pre-Stack Migration

Defines data structures for organizing traces into offset-azimuth bins
for common offset migration workflows.

Supports:
- Individual bin definitions (offset/azimuth ranges)
- Binning tables (collections of bins)
- Preset binning schemes (common offset, OVT, narrow azimuth)
- Bin validation and coverage analysis
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class BinningPreset(Enum):
    """Standard binning presets."""
    COMMON_OFFSET = "common_offset"
    OVT = "ovt"  # Offset Vector Tile
    NARROW_AZIMUTH = "narrow_azimuth"
    FULL_STACK = "full_stack"
    CUSTOM = "custom"


@dataclass
class OffsetAzimuthBin:
    """
    Definition of a single offset-azimuth bin.

    Traces are assigned to this bin if their offset and azimuth
    fall within the specified ranges.

    Attributes:
        name: Unique bin identifier (e.g., "near_offset", "off_200_400")
        offset_min: Minimum offset in meters (inclusive)
        offset_max: Maximum offset in meters (exclusive)
        azimuth_min: Minimum azimuth in degrees (inclusive, 0-360)
        azimuth_max: Maximum azimuth in degrees (exclusive, 0-360)
        enabled: Whether this bin is active for processing
        metadata: Optional additional parameters
    """
    name: str
    offset_min: float
    offset_max: float
    azimuth_min: float = 0.0
    azimuth_max: float = 360.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate bin parameters."""
        self._validate()

    def _validate(self):
        """Validate bin parameters."""
        if self.offset_min < 0:
            raise ValueError(f"offset_min must be non-negative, got {self.offset_min}")
        if self.offset_max <= self.offset_min:
            raise ValueError(
                f"offset_max ({self.offset_max}) must be greater than "
                f"offset_min ({self.offset_min})"
            )
        if not 0 <= self.azimuth_min < 360:
            raise ValueError(f"azimuth_min must be in [0, 360), got {self.azimuth_min}")
        if not 0 < self.azimuth_max <= 360:
            raise ValueError(f"azimuth_max must be in (0, 360], got {self.azimuth_max}")
        if not self.name:
            raise ValueError("Bin name cannot be empty")

    @property
    def offset_center(self) -> float:
        """Center offset of the bin."""
        return (self.offset_min + self.offset_max) / 2

    @property
    def offset_width(self) -> float:
        """Width of offset range."""
        return self.offset_max - self.offset_min

    @property
    def azimuth_center(self) -> float:
        """Center azimuth of the bin."""
        # Handle wrap-around (e.g., 350-10 degrees)
        if self.azimuth_max < self.azimuth_min:
            # Wraps around 0
            center = (self.azimuth_min + self.azimuth_max + 360) / 2
            if center >= 360:
                center -= 360
            return center
        return (self.azimuth_min + self.azimuth_max) / 2

    @property
    def azimuth_width(self) -> float:
        """Width of azimuth range."""
        if self.azimuth_max < self.azimuth_min:
            return (360 - self.azimuth_min) + self.azimuth_max
        return self.azimuth_max - self.azimuth_min

    @property
    def is_full_azimuth(self) -> bool:
        """Check if bin covers full azimuth range."""
        return self.azimuth_width >= 359.9

    def contains(self, offset: float, azimuth: float) -> bool:
        """
        Check if a trace with given offset/azimuth falls in this bin.

        Args:
            offset: Source-receiver offset in meters
            azimuth: Source-receiver azimuth in degrees (0-360)

        Returns:
            True if trace is within this bin
        """
        # Check offset
        if not (self.offset_min <= offset < self.offset_max):
            return False

        # Check azimuth (handle wrap-around)
        if self.azimuth_max < self.azimuth_min:
            # Wraps around 0 (e.g., 350 to 10)
            in_azimuth = azimuth >= self.azimuth_min or azimuth < self.azimuth_max
        else:
            in_azimuth = self.azimuth_min <= azimuth < self.azimuth_max

        return in_azimuth

    def contains_batch(
        self,
        offsets: np.ndarray,
        azimuths: np.ndarray
    ) -> np.ndarray:
        """
        Check which traces fall within this bin (vectorized).

        Args:
            offsets: Array of offset values
            azimuths: Array of azimuth values

        Returns:
            Boolean array indicating bin membership
        """
        offset_ok = (offsets >= self.offset_min) & (offsets < self.offset_max)

        if self.azimuth_max < self.azimuth_min:
            azimuth_ok = (azimuths >= self.azimuth_min) | (azimuths < self.azimuth_max)
        else:
            azimuth_ok = (azimuths >= self.azimuth_min) & (azimuths < self.azimuth_max)

        return offset_ok & azimuth_ok

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'offset_min': self.offset_min,
            'offset_max': self.offset_max,
            'azimuth_min': self.azimuth_min,
            'azimuth_max': self.azimuth_max,
            'enabled': self.enabled,
            'metadata': self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OffsetAzimuthBin':
        """Deserialize from dictionary."""
        return cls(
            name=d['name'],
            offset_min=d['offset_min'],
            offset_max=d['offset_max'],
            azimuth_min=d.get('azimuth_min', 0.0),
            azimuth_max=d.get('azimuth_max', 360.0),
            enabled=d.get('enabled', True),
            metadata=d.get('metadata', {}),
        )

    def __repr__(self) -> str:
        return (
            f"OffsetAzimuthBin(name='{self.name}', "
            f"offset=[{self.offset_min:.0f}, {self.offset_max:.0f}), "
            f"azimuth=[{self.azimuth_min:.0f}, {self.azimuth_max:.0f}))"
        )


@dataclass
class BinningTable:
    """
    Collection of offset-azimuth bins defining a migration job.

    The binning table specifies how input traces are partitioned
    into groups for separate migration outputs.

    Attributes:
        name: Table name/description
        bins: List of bin definitions
        preset: Binning preset used (if any)
        metadata: Additional table parameters
    """
    name: str = "Custom Binning"
    bins: List[OffsetAzimuthBin] = field(default_factory=list)
    preset: BinningPreset = BinningPreset.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate binning table."""
        self._validate_unique_names()

    def _validate_unique_names(self):
        """Ensure all bin names are unique."""
        names = [b.name for b in self.bins]
        if len(names) != len(set(names)):
            raise ValueError("Bin names must be unique")

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return len(self.bins)

    @property
    def enabled_bins(self) -> List[OffsetAzimuthBin]:
        """List of enabled bins only."""
        return [b for b in self.bins if b.enabled]

    @property
    def n_enabled_bins(self) -> int:
        """Number of enabled bins."""
        return len(self.enabled_bins)

    @property
    def offset_range(self) -> Tuple[float, float]:
        """Overall offset range covered by all bins."""
        if not self.bins:
            return (0.0, 0.0)
        return (
            min(b.offset_min for b in self.bins),
            max(b.offset_max for b in self.bins)
        )

    def add_bin(self, bin_def: OffsetAzimuthBin) -> None:
        """
        Add a bin to the table.

        Args:
            bin_def: Bin definition to add

        Raises:
            ValueError: If bin name already exists
        """
        if any(b.name == bin_def.name for b in self.bins):
            raise ValueError(f"Bin name '{bin_def.name}' already exists")
        self.bins.append(bin_def)

    def remove_bin(self, name: str) -> bool:
        """
        Remove a bin by name.

        Args:
            name: Bin name to remove

        Returns:
            True if bin was removed, False if not found
        """
        for i, b in enumerate(self.bins):
            if b.name == name:
                del self.bins[i]
                return True
        return False

    def get_bin(self, name: str) -> Optional[OffsetAzimuthBin]:
        """Get bin by name."""
        for b in self.bins:
            if b.name == name:
                return b
        return None

    def assign_trace(
        self,
        offset: float,
        azimuth: float,
        enabled_only: bool = True
    ) -> List[str]:
        """
        Assign a trace to bins based on offset/azimuth.

        Args:
            offset: Source-receiver offset
            azimuth: Source-receiver azimuth
            enabled_only: Only consider enabled bins

        Returns:
            List of bin names the trace belongs to
        """
        bins_to_check = self.enabled_bins if enabled_only else self.bins
        return [b.name for b in bins_to_check if b.contains(offset, azimuth)]

    def assign_traces_batch(
        self,
        offsets: np.ndarray,
        azimuths: np.ndarray,
        enabled_only: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Assign traces to bins (vectorized).

        Args:
            offsets: Array of offset values
            azimuths: Array of azimuth values
            enabled_only: Only consider enabled bins

        Returns:
            Dictionary mapping bin name to boolean mask of included traces
        """
        bins_to_check = self.enabled_bins if enabled_only else self.bins
        return {
            b.name: b.contains_batch(offsets, azimuths)
            for b in bins_to_check
        }

    def check_coverage(
        self,
        offsets: np.ndarray,
        azimuths: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze how well bins cover the input data.

        Args:
            offsets: Array of trace offsets
            azimuths: Array of trace azimuths

        Returns:
            Coverage statistics
        """
        n_traces = len(offsets)
        assignments = self.assign_traces_batch(offsets, azimuths)

        # Count traces per bin
        bin_counts = {name: np.sum(mask) for name, mask in assignments.items()}

        # Find unassigned traces
        any_bin = np.zeros(n_traces, dtype=bool)
        for mask in assignments.values():
            any_bin |= mask
        n_unassigned = n_traces - np.sum(any_bin)

        # Find traces in multiple bins
        n_multi = 0
        bin_count_per_trace = np.zeros(n_traces, dtype=int)
        for mask in assignments.values():
            bin_count_per_trace += mask.astype(int)
        n_multi = np.sum(bin_count_per_trace > 1)

        return {
            'n_traces': n_traces,
            'n_unassigned': int(n_unassigned),
            'n_multi_bin': int(n_multi),
            'bin_counts': bin_counts,
            'coverage_percent': 100.0 * (n_traces - n_unassigned) / n_traces if n_traces > 0 else 0.0,
        }

    def check_overlaps(self) -> List[Tuple[str, str]]:
        """
        Find pairs of bins with overlapping offset-azimuth ranges.

        Returns:
            List of (bin1_name, bin2_name) pairs that overlap
        """
        overlaps = []
        for i, b1 in enumerate(self.bins):
            for b2 in self.bins[i+1:]:
                if self._bins_overlap(b1, b2):
                    overlaps.append((b1.name, b2.name))
        return overlaps

    def _bins_overlap(self, b1: OffsetAzimuthBin, b2: OffsetAzimuthBin) -> bool:
        """Check if two bins have overlapping ranges."""
        # Check offset overlap
        offset_overlap = not (b1.offset_max <= b2.offset_min or b2.offset_max <= b1.offset_min)

        if not offset_overlap:
            return False

        # Check azimuth overlap (more complex due to wrap-around)
        # Simplified: check if any point in one range is in the other
        test_azimuths = [b1.azimuth_min, b1.azimuth_center, b2.azimuth_min, b2.azimuth_center]
        for az in test_azimuths:
            if b1.contains(b1.offset_center, az) and b2.contains(b2.offset_center, az):
                return True

        return False

    def check_gaps(self, offset_step: float = 100.0) -> List[Tuple[float, float]]:
        """
        Find gaps in offset coverage.

        Args:
            offset_step: Resolution for gap detection

        Returns:
            List of (offset_min, offset_max) gap ranges
        """
        if not self.bins:
            return []

        offset_min, offset_max = self.offset_range
        gaps = []

        current_offset = offset_min
        while current_offset < offset_max:
            # Check if any bin covers this offset
            covered = any(
                b.offset_min <= current_offset < b.offset_max
                for b in self.bins
            )
            if not covered:
                # Start of a gap
                gap_start = current_offset
                while current_offset < offset_max:
                    covered = any(
                        b.offset_min <= current_offset < b.offset_max
                        for b in self.bins
                    )
                    if covered:
                        break
                    current_offset += offset_step
                gaps.append((gap_start, current_offset))
            else:
                current_offset += offset_step

        return gaps

    def __iter__(self) -> Iterator[OffsetAzimuthBin]:
        """Iterate over bins."""
        return iter(self.bins)

    def __len__(self) -> int:
        return len(self.bins)

    def __getitem__(self, idx: int) -> OffsetAzimuthBin:
        return self.bins[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'preset': self.preset.value,
            'bins': [b.to_dict() for b in self.bins],
            'metadata': self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BinningTable':
        """Deserialize from dictionary."""
        bins = [OffsetAzimuthBin.from_dict(b) for b in d.get('bins', [])]
        return cls(
            name=d.get('name', 'Custom Binning'),
            bins=bins,
            preset=BinningPreset(d.get('preset', 'custom')),
            metadata=d.get('metadata', {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'BinningTable':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: str) -> None:
        """Save binning table to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved binning table to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BinningTable':
        """Load binning table from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())

    def __repr__(self) -> str:
        return (
            f"BinningTable(name='{self.name}', n_bins={self.n_bins}, "
            f"preset={self.preset.value})"
        )


# =============================================================================
# Factory Functions for Common Binning Presets
# =============================================================================

def create_common_offset_binning(
    offset_ranges: List[Tuple[float, float]],
    name_prefix: str = "offset",
) -> BinningTable:
    """
    Create common offset binning table.

    Each bin covers full azimuth (0-360) with specified offset range.

    Args:
        offset_ranges: List of (offset_min, offset_max) tuples
        name_prefix: Prefix for bin names

    Returns:
        BinningTable with common offset bins
    """
    bins = []
    for i, (off_min, off_max) in enumerate(offset_ranges):
        name = f"{name_prefix}_{int(off_min):04d}_{int(off_max):04d}"
        bins.append(OffsetAzimuthBin(
            name=name,
            offset_min=off_min,
            offset_max=off_max,
            azimuth_min=0.0,
            azimuth_max=360.0,
        ))

    return BinningTable(
        name="Common Offset Binning",
        bins=bins,
        preset=BinningPreset.COMMON_OFFSET,
    )


def create_uniform_offset_binning(
    offset_min: float = 0.0,
    offset_max: float = 5000.0,
    n_bins: int = 10,
) -> BinningTable:
    """
    Create uniform offset binning with equal-width bins.

    Args:
        offset_min: Minimum offset
        offset_max: Maximum offset
        n_bins: Number of bins

    Returns:
        BinningTable with uniform offset bins
    """
    bin_width = (offset_max - offset_min) / n_bins
    ranges = [
        (offset_min + i * bin_width, offset_min + (i + 1) * bin_width)
        for i in range(n_bins)
    ]
    return create_common_offset_binning(ranges)


def create_ovt_binning(
    offset_ranges: List[Tuple[float, float]],
    n_azimuth_sectors: int = 4,
) -> BinningTable:
    """
    Create Offset Vector Tile (OVT) binning.

    Combines offset ranges with azimuth sectors for wide-azimuth data.

    Args:
        offset_ranges: List of (offset_min, offset_max) tuples
        n_azimuth_sectors: Number of azimuth sectors (typically 4 or 8)

    Returns:
        BinningTable with OVT bins
    """
    azimuth_width = 360.0 / n_azimuth_sectors
    bins = []

    for off_min, off_max in offset_ranges:
        for az_idx in range(n_azimuth_sectors):
            az_min = az_idx * azimuth_width
            az_max = (az_idx + 1) * azimuth_width

            name = f"ovt_off{int(off_min):04d}_az{int(az_min):03d}"
            bins.append(OffsetAzimuthBin(
                name=name,
                offset_min=off_min,
                offset_max=off_max,
                azimuth_min=az_min,
                azimuth_max=az_max,
            ))

    return BinningTable(
        name="OVT Binning",
        bins=bins,
        preset=BinningPreset.OVT,
        metadata={'n_azimuth_sectors': n_azimuth_sectors},
    )


def create_narrow_azimuth_binning(
    offset_min: float = 0.0,
    offset_max: float = 5000.0,
    inline_azimuth: float = 0.0,
    azimuth_width: float = 20.0,
) -> BinningTable:
    """
    Create narrow azimuth binning for inline/crossline separation.

    Creates 4 bins: inline positive, inline negative,
    crossline positive, crossline negative.

    Args:
        offset_min: Minimum offset
        offset_max: Maximum offset
        inline_azimuth: Inline direction azimuth (degrees from north)
        azimuth_width: Width of each azimuth sector

    Returns:
        BinningTable with narrow azimuth bins
    """
    half_width = azimuth_width / 2

    # Normalize inline azimuth
    inline_azimuth = inline_azimuth % 360

    bins = [
        # Inline positive
        OffsetAzimuthBin(
            name="inline_pos",
            offset_min=offset_min,
            offset_max=offset_max,
            azimuth_min=(inline_azimuth - half_width) % 360,
            azimuth_max=(inline_azimuth + half_width) % 360,
        ),
        # Inline negative (opposite direction)
        OffsetAzimuthBin(
            name="inline_neg",
            offset_min=offset_min,
            offset_max=offset_max,
            azimuth_min=(inline_azimuth + 180 - half_width) % 360,
            azimuth_max=(inline_azimuth + 180 + half_width) % 360,
        ),
        # Crossline positive (90 degrees from inline)
        OffsetAzimuthBin(
            name="xline_pos",
            offset_min=offset_min,
            offset_max=offset_max,
            azimuth_min=(inline_azimuth + 90 - half_width) % 360,
            azimuth_max=(inline_azimuth + 90 + half_width) % 360,
        ),
        # Crossline negative
        OffsetAzimuthBin(
            name="xline_neg",
            offset_min=offset_min,
            offset_max=offset_max,
            azimuth_min=(inline_azimuth + 270 - half_width) % 360,
            azimuth_max=(inline_azimuth + 270 + half_width) % 360,
        ),
    ]

    return BinningTable(
        name="Narrow Azimuth Binning",
        bins=bins,
        preset=BinningPreset.NARROW_AZIMUTH,
        metadata={'inline_azimuth': inline_azimuth, 'azimuth_width': azimuth_width},
    )


def create_full_stack_binning(
    offset_max: float = 10000.0,
) -> BinningTable:
    """
    Create single-bin table for full stack migration.

    Args:
        offset_max: Maximum offset to include

    Returns:
        BinningTable with single full-stack bin
    """
    return BinningTable(
        name="Full Stack",
        bins=[
            OffsetAzimuthBin(
                name="full_stack",
                offset_min=0.0,
                offset_max=offset_max,
                azimuth_min=0.0,
                azimuth_max=360.0,
            )
        ],
        preset=BinningPreset.FULL_STACK,
    )
