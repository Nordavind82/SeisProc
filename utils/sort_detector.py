"""
Sort Order Detection for Seismic Data

Detects the sort order of input seismic data:
- Common Shot
- Common Offset
- Common Receiver
- Common CDP
- OVT (Offset Vector Tile)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


class SortOrder(Enum):
    """Data sort order types."""
    COMMON_SHOT = "common_shot"
    COMMON_RECEIVER = "common_receiver"
    COMMON_OFFSET = "common_offset"
    COMMON_CDP = "common_cdp"
    OVT = "offset_vector_tile"
    UNKNOWN = "unknown"


@dataclass
class SortAnalysis:
    """Results of sort order analysis."""
    detected_order: SortOrder
    confidence: float  # 0.0 to 1.0
    primary_key_changes: int  # How many times the primary key changes
    secondary_key_changes: int  # How many times secondary key changes within primary
    average_gather_size: float
    unique_primary_keys: int
    details: str


def detect_sort_order(
    shot_ids: np.ndarray,
    receiver_ids: Optional[np.ndarray] = None,
    offsets: Optional[np.ndarray] = None,
    inlines: Optional[np.ndarray] = None,
    xlines: Optional[np.ndarray] = None,
    cdp_x: Optional[np.ndarray] = None,
    cdp_y: Optional[np.ndarray] = None,
    azimuths: Optional[np.ndarray] = None,
) -> SortAnalysis:
    """
    Detect the sort order of seismic data.

    Analyzes trace header values to determine how data is sorted.

    Args:
        shot_ids: Shot/source point identifiers
        receiver_ids: Receiver/group identifiers (optional)
        offsets: Source-receiver offsets (optional)
        inlines: Inline numbers (optional)
        xlines: Crossline numbers (optional)
        cdp_x: CDP X coordinates (optional)
        cdp_y: CDP Y coordinates (optional)
        azimuths: Source-receiver azimuths (optional)

    Returns:
        SortAnalysis with detected order and confidence
    """
    n_traces = len(shot_ids)
    if n_traces < 2:
        return SortAnalysis(
            detected_order=SortOrder.UNKNOWN,
            confidence=0.0,
            primary_key_changes=0,
            secondary_key_changes=0,
            average_gather_size=float(n_traces),
            unique_primary_keys=1,
            details="Too few traces to determine sort order"
        )

    candidates = []

    # Analyze shot sort
    shot_analysis = _analyze_key_continuity(shot_ids, "shot")
    candidates.append((SortOrder.COMMON_SHOT, shot_analysis))

    # Analyze offset sort
    if offsets is not None:
        offset_bins = _discretize_offsets(offsets)
        offset_analysis = _analyze_key_continuity(offset_bins, "offset")
        candidates.append((SortOrder.COMMON_OFFSET, offset_analysis))

    # Analyze receiver sort
    if receiver_ids is not None:
        receiver_analysis = _analyze_key_continuity(receiver_ids, "receiver")
        candidates.append((SortOrder.COMMON_RECEIVER, receiver_analysis))

    # Analyze CDP sort
    if inlines is not None and xlines is not None:
        cdp_keys = inlines * 100000 + xlines  # Combine into single key
        cdp_analysis = _analyze_key_continuity(cdp_keys, "cdp")
        candidates.append((SortOrder.COMMON_CDP, cdp_analysis))

    # Analyze OVT sort (offset + azimuth sectors)
    if offsets is not None and azimuths is not None:
        ovt_keys = _compute_ovt_keys(offsets, azimuths)
        ovt_analysis = _analyze_key_continuity(ovt_keys, "ovt")
        candidates.append((SortOrder.OVT, ovt_analysis))

    # Select best candidate based on gather continuity
    best_order = SortOrder.UNKNOWN
    best_score = -1.0
    best_analysis = None

    for order, analysis in candidates:
        if analysis['score'] > best_score:
            best_score = analysis['score']
            best_order = order
            best_analysis = analysis

    if best_analysis is None:
        return SortAnalysis(
            detected_order=SortOrder.UNKNOWN,
            confidence=0.0,
            primary_key_changes=0,
            secondary_key_changes=0,
            average_gather_size=1.0,
            unique_primary_keys=n_traces,
            details="Could not analyze any keys"
        )

    return SortAnalysis(
        detected_order=best_order,
        confidence=min(1.0, best_score),
        primary_key_changes=best_analysis['key_changes'],
        secondary_key_changes=0,  # TODO: add secondary key analysis
        average_gather_size=best_analysis['avg_gather_size'],
        unique_primary_keys=best_analysis['unique_keys'],
        details=best_analysis['details']
    )


def _analyze_key_continuity(keys: np.ndarray, key_name: str) -> dict:
    """
    Analyze how continuously grouped a key is.

    Well-sorted data has keys that change infrequently and form
    contiguous groups.

    Returns dict with score (0-1), key_changes, unique_keys, avg_gather_size
    """
    n = len(keys)

    # Count key changes (transitions)
    changes = np.sum(keys[1:] != keys[:-1])

    # Count unique keys
    unique_keys = len(np.unique(keys))

    # Calculate average gather size
    avg_gather_size = n / max(1, unique_keys)

    # Score: perfect sort has changes == unique_keys - 1
    # Worse sort has more changes (same key appears in multiple places)
    min_changes = unique_keys - 1

    if changes == 0:
        # All same key
        score = 1.0 if unique_keys == 1 else 0.0
    elif min_changes == 0:
        score = 1.0
    else:
        # How close to optimal?
        excess_changes = changes - min_changes
        # Score decreases with excess changes
        score = max(0.0, 1.0 - (excess_changes / n))

    # Bonus for reasonable gather sizes (not too small)
    if avg_gather_size >= 10:
        score *= 1.0
    elif avg_gather_size >= 5:
        score *= 0.9
    else:
        score *= 0.7

    details = (
        f"{key_name}: {unique_keys} unique values, "
        f"{changes} transitions, avg gather size {avg_gather_size:.1f}"
    )

    return {
        'score': score,
        'key_changes': changes,
        'unique_keys': unique_keys,
        'avg_gather_size': avg_gather_size,
        'details': details,
    }


def _discretize_offsets(offsets: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Convert continuous offsets to discrete bin indices."""
    offset_min = np.min(offsets)
    offset_max = np.max(offsets)

    if offset_max == offset_min:
        return np.zeros(len(offsets), dtype=np.int32)

    bin_width = (offset_max - offset_min) / n_bins
    bins = ((offsets - offset_min) / bin_width).astype(np.int32)
    bins = np.clip(bins, 0, n_bins - 1)

    return bins


def _compute_ovt_keys(offsets: np.ndarray, azimuths: np.ndarray,
                       n_offset_bins: int = 4, n_azimuth_sectors: int = 4) -> np.ndarray:
    """Compute OVT (Offset Vector Tile) keys."""
    offset_min = np.min(offsets)
    offset_max = np.max(offsets)

    if offset_max == offset_min:
        offset_bins = np.zeros(len(offsets), dtype=np.int32)
    else:
        offset_width = (offset_max - offset_min) / n_offset_bins
        offset_bins = ((offsets - offset_min) / offset_width).astype(np.int32)
        offset_bins = np.clip(offset_bins, 0, n_offset_bins - 1)

    azimuth_width = 360.0 / n_azimuth_sectors
    azimuth_sectors = (azimuths / azimuth_width).astype(np.int32) % n_azimuth_sectors

    # Combine into single key
    ovt_keys = offset_bins * n_azimuth_sectors + azimuth_sectors

    return ovt_keys


def get_gather_boundaries(
    keys: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Get gather boundaries from sorted keys.

    Args:
        keys: Primary sort key values (e.g., shot IDs, offset bins)

    Returns:
        List of (start_idx, end_idx) tuples for each gather
    """
    if len(keys) == 0:
        return []

    # Find where key changes
    change_points = np.where(keys[1:] != keys[:-1])[0] + 1

    # Build boundaries
    boundaries = []
    start = 0

    for end in change_points:
        boundaries.append((start, end))
        start = end

    # Add final gather
    boundaries.append((start, len(keys)))

    return boundaries
