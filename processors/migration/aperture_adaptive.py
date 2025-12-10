"""
Depth-Adaptive Aperture for Kirchhoff Migration.

Optimizes migration by adjusting aperture based on depth:
- Shallow depths naturally have smaller effective aperture
- Deep depths can use larger aperture but often with fewer contributing traces
- Groups depths by aperture requirements for efficient batch processing

Expected speedup: 1.5-2x (geometry dependent)
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthGroup:
    """A group of depth indices with similar aperture requirements."""
    z_indices: np.ndarray  # Indices into z_axis
    z_start: int  # Start index
    z_end: int  # End index (exclusive)
    effective_aperture: float  # Maximum aperture for this group (meters)
    n_traces_estimate: int  # Estimated number of contributing traces


class DepthAdaptiveAperture:
    """
    Depth-adaptive aperture controller for optimized migration.

    Key insight: At depth z with maximum angle θ, the effective aperture is:
        h_max(z) = z * tan(θ)

    For shallow depths, this is much smaller than the user-specified max_aperture,
    allowing us to skip many traces that cannot contribute.

    Features:
    - Pre-computes effective aperture for each depth level
    - Groups depths by similar aperture requirements
    - Provides sparse trace masks for each depth group
    - Estimates computational savings

    Example:
        >>> aperture = DepthAdaptiveAperture(max_aperture=5000, max_angle=60)
        >>> aperture.compute_depth_apertures(z_axis)
        >>> groups = aperture.group_depths_by_aperture(n_groups=5)
        >>> for group in groups:
        ...     mask = aperture.get_trace_mask(group, h_src, h_rcv)
    """

    def __init__(
        self,
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        min_offset_m: float = 0.0,
        max_offset_m: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize depth-adaptive aperture controller.

        Args:
            max_aperture_m: Maximum horizontal aperture (user-specified limit)
            max_angle_deg: Maximum angle from vertical (degrees)
            min_offset_m: Minimum source-receiver offset
            max_offset_m: Maximum source-receiver offset
            device: Torch device for computation
        """
        self.max_aperture = max_aperture_m
        self.max_angle_rad = np.radians(max_angle_deg)
        self.max_angle_deg = max_angle_deg
        self.min_offset = min_offset_m
        self.max_offset = max_offset_m
        self.device = device or self._detect_device()

        # Computed values
        self._z_axis: Optional[np.ndarray] = None
        self._effective_apertures: Optional[np.ndarray] = None
        self._depth_groups: Optional[List[DepthGroup]] = None

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def compute_depth_apertures(self, z_axis: np.ndarray) -> np.ndarray:
        """
        Compute effective aperture for each depth level.

        The effective aperture at depth z is:
            h_eff(z) = min(max_aperture, z * tan(max_angle))

        Args:
            z_axis: Depth values in meters

        Returns:
            Effective aperture for each depth
        """
        self._z_axis = np.asarray(z_axis)

        # Angle-limited aperture: h = z * tan(θ)
        angle_aperture = np.abs(self._z_axis) * np.tan(self.max_angle_rad)

        # Effective aperture is minimum of angle and distance limits
        self._effective_apertures = np.minimum(angle_aperture, self.max_aperture)

        logger.debug(
            f"Computed apertures for {len(z_axis)} depths: "
            f"range {self._effective_apertures.min():.0f}-{self._effective_apertures.max():.0f}m"
        )

        return self._effective_apertures

    def group_depths_by_aperture(
        self,
        n_groups: int = 5,
        min_group_size: int = 10,
    ) -> List[DepthGroup]:
        """
        Group depth levels by similar aperture requirements.

        Depths with similar effective apertures are grouped together
        for efficient batch processing.

        Args:
            n_groups: Target number of groups
            min_group_size: Minimum depths per group

        Returns:
            List of DepthGroup objects
        """
        if self._effective_apertures is None:
            raise RuntimeError("Call compute_depth_apertures first")

        n_depths = len(self._z_axis)
        groups = []

        # Define aperture thresholds for grouping
        # Use logarithmic spacing for better coverage
        min_ap = max(100, self._effective_apertures[self._effective_apertures > 0].min())
        max_ap = self._effective_apertures.max()

        if max_ap <= min_ap:
            # All same aperture, single group
            thresholds = [max_ap * 1.1]
        else:
            thresholds = np.geomspace(min_ap, max_ap * 1.1, n_groups + 1)[1:]

        # Assign depths to groups
        current_start = 0
        for thresh in thresholds:
            # Find depths with aperture <= threshold
            mask = self._effective_apertures[current_start:] <= thresh

            if not mask.any():
                continue

            # Find the last depth in this group
            group_end = current_start + np.argmax(~mask) if not mask.all() else n_depths

            if group_end <= current_start:
                continue

            # Ensure minimum group size
            if group_end - current_start < min_group_size and groups:
                # Merge with previous group
                prev_group = groups[-1]
                groups[-1] = DepthGroup(
                    z_indices=np.arange(prev_group.z_start, group_end),
                    z_start=prev_group.z_start,
                    z_end=group_end,
                    effective_aperture=thresh,
                    n_traces_estimate=0,  # Will be computed later
                )
            else:
                groups.append(DepthGroup(
                    z_indices=np.arange(current_start, group_end),
                    z_start=current_start,
                    z_end=group_end,
                    effective_aperture=thresh,
                    n_traces_estimate=0,
                ))

            current_start = group_end

            if current_start >= n_depths:
                break

        # Handle remaining depths
        if current_start < n_depths:
            if groups:
                # Merge with last group
                prev = groups[-1]
                groups[-1] = DepthGroup(
                    z_indices=np.arange(prev.z_start, n_depths),
                    z_start=prev.z_start,
                    z_end=n_depths,
                    effective_aperture=self.max_aperture,
                    n_traces_estimate=0,
                )
            else:
                groups.append(DepthGroup(
                    z_indices=np.arange(current_start, n_depths),
                    z_start=current_start,
                    z_end=n_depths,
                    effective_aperture=self.max_aperture,
                    n_traces_estimate=0,
                ))

        self._depth_groups = groups

        logger.info(f"Created {len(groups)} depth groups:")
        for i, g in enumerate(groups):
            logger.info(f"  Group {i}: depths {g.z_start}-{g.z_end}, aperture {g.effective_aperture:.0f}m")

        return groups

    def get_contributing_trace_mask(
        self,
        group: DepthGroup,
        h_src: torch.Tensor,
        h_rcv: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get mask of traces that contribute to a depth group.

        Args:
            group: DepthGroup object
            h_src: Horizontal distance from source to image points (n_traces,)
            h_rcv: Horizontal distance from receiver to image points (n_traces,)
            offset: Source-receiver offset (n_traces,), optional

        Returns:
            Boolean mask of contributing traces
        """
        aperture = group.effective_aperture

        # Distance constraint
        mask = (h_src <= aperture) & (h_rcv <= aperture)

        # Offset constraint if provided
        if offset is not None:
            mask = mask & (offset >= self.min_offset) & (offset <= self.max_offset)

        return mask

    def get_contributing_trace_indices(
        self,
        group: DepthGroup,
        h_src: np.ndarray,
        h_rcv: np.ndarray,
        offset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get indices of traces that contribute to a depth group (NumPy version).

        Args:
            group: DepthGroup object
            h_src: Horizontal distance from source to image points
            h_rcv: Horizontal distance from receiver to image points
            offset: Source-receiver offset, optional

        Returns:
            Array of trace indices
        """
        aperture = group.effective_aperture

        # Distance constraint
        mask = (h_src <= aperture) & (h_rcv <= aperture)

        # Offset constraint if provided
        if offset is not None:
            mask = mask & (offset >= self.min_offset) & (offset <= self.max_offset)

        return np.where(mask)[0]

    def estimate_trace_reduction(
        self,
        h_src: np.ndarray,
        h_rcv: np.ndarray,
        offset: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Estimate trace reduction from depth-adaptive aperture.

        Args:
            h_src: Horizontal distance source to center of output area
            h_rcv: Horizontal distance receiver to center of output area
            offset: Source-receiver offset

        Returns:
            Statistics dict with reduction estimates
        """
        if self._depth_groups is None:
            raise RuntimeError("Call group_depths_by_aperture first")

        total_traces = len(h_src)
        total_pairs_baseline = total_traces * len(self._z_axis)
        total_pairs_adaptive = 0

        stats = {
            'n_depths': len(self._z_axis),
            'n_traces': total_traces,
            'n_groups': len(self._depth_groups),
            'baseline_trace_depth_pairs': total_pairs_baseline,
            'groups': [],
        }

        for group in self._depth_groups:
            n_contributing = len(self.get_contributing_trace_indices(
                group, h_src, h_rcv, offset
            ))
            n_depths_in_group = group.z_end - group.z_start
            pairs_in_group = n_contributing * n_depths_in_group
            total_pairs_adaptive += pairs_in_group

            stats['groups'].append({
                'z_start': group.z_start,
                'z_end': group.z_end,
                'n_depths': n_depths_in_group,
                'aperture': group.effective_aperture,
                'n_traces': n_contributing,
                'trace_depth_pairs': pairs_in_group,
            })

        stats['adaptive_trace_depth_pairs'] = total_pairs_adaptive
        stats['reduction_ratio'] = total_pairs_baseline / max(1, total_pairs_adaptive)
        stats['speedup_estimate'] = stats['reduction_ratio']

        return stats

    def get_depth_aperture(self, z_index: int) -> float:
        """Get effective aperture for a specific depth index."""
        if self._effective_apertures is None:
            raise RuntimeError("Call compute_depth_apertures first")
        return float(self._effective_apertures[z_index])

    @property
    def depth_groups(self) -> Optional[List[DepthGroup]]:
        """Get computed depth groups."""
        return self._depth_groups

    @property
    def effective_apertures(self) -> Optional[np.ndarray]:
        """Get effective apertures array."""
        return self._effective_apertures


def create_depth_adaptive_aperture(
    config,
    device: Optional[torch.device] = None,
) -> DepthAdaptiveAperture:
    """
    Factory function to create DepthAdaptiveAperture from config.

    Args:
        config: MigrationConfig instance
        device: Torch device

    Returns:
        Configured DepthAdaptiveAperture
    """
    return DepthAdaptiveAperture(
        max_aperture_m=config.max_aperture_m,
        max_angle_deg=config.max_angle_deg,
        min_offset_m=config.min_offset_m,
        max_offset_m=config.max_offset_m,
        device=device,
    )
