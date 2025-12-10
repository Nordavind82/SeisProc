"""
Spatial Index for Trace Selection in Kirchhoff Migration.

Provides efficient spatial queries to select only traces that can
contribute to a given output region. Uses KD-tree for fast nearest
neighbor and range queries.

Expected speedup: 3-5x (highly geometry dependent)
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Optional, Tuple, List, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TraceBounds:
    """Bounding box of trace coverage."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class TraceSpatialIndex:
    """
    Spatial index for efficient trace selection in migration.

    Builds a KD-tree over trace midpoints (CDP locations) and provides
    fast queries to find traces that can contribute to a given output region.

    Key insight: A trace can only contribute to an image point if the
    image point is within max_aperture of both the source and receiver.
    By indexing trace midpoints, we can quickly find candidate traces
    for any output region.

    Features:
    - KD-tree based spatial index for O(log n) queries
    - Bounding box queries with aperture buffer
    - Point queries with distance threshold
    - Grid-based binning for additional optimization

    Example:
        >>> index = TraceSpatialIndex()
        >>> index.build(src_x, src_y, rcv_x, rcv_y)
        >>> traces = index.query_traces_for_region(0, 1000, 0, 1000, buffer=5000)
    """

    def __init__(self):
        """Initialize trace spatial index."""
        self._kdtree: Optional[KDTree] = None
        self._midpoints: Optional[np.ndarray] = None
        self._n_traces: int = 0
        self._bounds: Optional[TraceBounds] = None

        # Original coordinates for additional filtering
        self._source_x: Optional[np.ndarray] = None
        self._source_y: Optional[np.ndarray] = None
        self._receiver_x: Optional[np.ndarray] = None
        self._receiver_y: Optional[np.ndarray] = None

        # Half-offsets for tighter filtering
        self._half_offset: Optional[np.ndarray] = None

        # Build stats
        self._built: bool = False

    def build(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
    ) -> 'TraceSpatialIndex':
        """
        Build spatial index from trace coordinates.

        Args:
            source_x, source_y: Source coordinates (n_traces,)
            receiver_x, receiver_y: Receiver coordinates (n_traces,)

        Returns:
            self for method chaining
        """
        self._source_x = np.asarray(source_x, dtype=np.float64)
        self._source_y = np.asarray(source_y, dtype=np.float64)
        self._receiver_x = np.asarray(receiver_x, dtype=np.float64)
        self._receiver_y = np.asarray(receiver_y, dtype=np.float64)

        self._n_traces = len(source_x)

        # Compute midpoints (CDP locations)
        mid_x = (self._source_x + self._receiver_x) / 2
        mid_y = (self._source_y + self._receiver_y) / 2
        self._midpoints = np.column_stack([mid_x, mid_y])

        # Compute half-offsets for tighter filtering
        self._half_offset = np.sqrt(
            (self._receiver_x - self._source_x)**2 +
            (self._receiver_y - self._source_y)**2
        ) / 2

        # Build KD-tree
        self._kdtree = KDTree(self._midpoints)

        # Compute bounds
        self._bounds = TraceBounds(
            x_min=float(mid_x.min()),
            x_max=float(mid_x.max()),
            y_min=float(mid_y.min()),
            y_max=float(mid_y.max()),
        )

        self._built = True

        logger.info(
            f"Built spatial index for {self._n_traces} traces, "
            f"bounds: ({self._bounds.x_min:.0f}, {self._bounds.y_min:.0f}) - "
            f"({self._bounds.x_max:.0f}, {self._bounds.y_max:.0f})"
        )

        return self

    def query_traces_for_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        buffer: float,
        refine_with_aperture: bool = True,
    ) -> np.ndarray:
        """
        Query traces that may contribute to a rectangular output region.

        Args:
            x_min, x_max: X bounds of output region
            y_min, y_max: Y bounds of output region
            buffer: Aperture buffer (typically max_aperture_m)
            refine_with_aperture: If True, apply tighter filtering based on
                                 actual source/receiver positions

        Returns:
            Array of trace indices that may contribute
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        # Expand region by buffer to include all possibly contributing traces
        # A trace midpoint at distance d from region can still contribute
        # if source or receiver is closer
        search_x_min = x_min - buffer
        search_x_max = x_max + buffer
        search_y_min = y_min - buffer
        search_y_max = y_max + buffer

        # Find all traces with midpoints in expanded region
        # Use ball query from each corner for KD-tree
        candidates = self._query_box(
            search_x_min, search_x_max,
            search_y_min, search_y_max,
        )

        if len(candidates) == 0:
            return np.array([], dtype=np.int64)

        if not refine_with_aperture:
            return candidates

        # Refine: check that at least source or receiver is within buffer
        # of at least one corner of the output region
        corners = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max],
        ])

        # Get source and receiver coords for candidates
        src_x = self._source_x[candidates]
        src_y = self._source_y[candidates]
        rcv_x = self._receiver_x[candidates]
        rcv_y = self._receiver_y[candidates]

        # For each candidate, check if either source or receiver
        # is within buffer of any point in the output region
        # (simplified: check corners)
        valid_mask = np.zeros(len(candidates), dtype=bool)

        for corner in corners:
            # Distance from source to corner
            d_src = np.sqrt((src_x - corner[0])**2 + (src_y - corner[1])**2)
            # Distance from receiver to corner
            d_rcv = np.sqrt((rcv_x - corner[0])**2 + (rcv_y - corner[1])**2)

            # Trace can contribute if both source AND receiver are within buffer
            valid_mask |= (d_src <= buffer) & (d_rcv <= buffer)

        return candidates[valid_mask]

    def query_traces_for_point(
        self,
        x: float,
        y: float,
        max_dist: float,
    ) -> np.ndarray:
        """
        Query traces that may contribute to a single output point.

        Args:
            x, y: Output point coordinates
            max_dist: Maximum aperture distance

        Returns:
            Array of trace indices
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        # Query KD-tree for midpoints within max_dist + half_offset_max
        # This accounts for traces where midpoint is far but source/receiver is close
        max_half_offset = self._half_offset.max() if len(self._half_offset) > 0 else 0
        search_radius = max_dist + max_half_offset

        # KD-tree query
        indices = self._kdtree.query_ball_point([x, y], search_radius)

        if len(indices) == 0:
            return np.array([], dtype=np.int64)

        indices = np.array(indices, dtype=np.int64)

        # Refine: check actual source and receiver distances
        src_x = self._source_x[indices]
        src_y = self._source_y[indices]
        rcv_x = self._receiver_x[indices]
        rcv_y = self._receiver_y[indices]

        d_src = np.sqrt((src_x - x)**2 + (src_y - y)**2)
        d_rcv = np.sqrt((rcv_x - x)**2 + (rcv_y - y)**2)

        # Both must be within max_dist
        valid = (d_src <= max_dist) & (d_rcv <= max_dist)

        return indices[valid]

    def _query_box(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> np.ndarray:
        """
        Query all trace midpoints within a bounding box.

        Uses KD-tree ball queries from corners and merges results.
        """
        # Box diagonal / 2 gives minimum radius to cover the box from center
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        half_diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) / 2

        # Query from center with radius covering the box
        candidates = self._kdtree.query_ball_point([cx, cy], half_diag * 1.1)

        if len(candidates) == 0:
            return np.array([], dtype=np.int64)

        candidates = np.array(candidates, dtype=np.int64)

        # Filter to actual box
        mid_x = self._midpoints[candidates, 0]
        mid_y = self._midpoints[candidates, 1]

        in_box = (
            (mid_x >= x_min) & (mid_x <= x_max) &
            (mid_y >= y_min) & (mid_y <= y_max)
        )

        return candidates[in_box]

    def get_trace_bounds(self) -> TraceBounds:
        """Get bounding box of all trace midpoints."""
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")
        return self._bounds

    def estimate_traces_for_grid(
        self,
        grid_bounds: Tuple[float, float, float, float],
        chunk_size: Tuple[float, float],
        aperture: float,
    ) -> dict:
        """
        Estimate trace counts for a grid of output chunks.

        Useful for planning and load balancing.

        Args:
            grid_bounds: (x_min, x_max, y_min, y_max) of output grid
            chunk_size: (dx, dy) size of each chunk
            aperture: Max aperture in meters

        Returns:
            Dict with statistics about trace distribution
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        x_min, x_max, y_min, y_max = grid_bounds
        dx, dy = chunk_size

        n_chunks_x = int(np.ceil((x_max - x_min) / dx))
        n_chunks_y = int(np.ceil((y_max - y_min) / dy))

        counts = np.zeros((n_chunks_x, n_chunks_y), dtype=np.int64)
        total_traces = 0

        for i in range(n_chunks_x):
            for j in range(n_chunks_y):
                chunk_x_min = x_min + i * dx
                chunk_x_max = min(x_min + (i + 1) * dx, x_max)
                chunk_y_min = y_min + j * dy
                chunk_y_max = min(y_min + (j + 1) * dy, y_max)

                traces = self.query_traces_for_region(
                    chunk_x_min, chunk_x_max,
                    chunk_y_min, chunk_y_max,
                    buffer=aperture,
                    refine_with_aperture=False,  # Fast estimate
                )
                counts[i, j] = len(traces)
                total_traces += len(traces)

        return {
            'n_chunks': n_chunks_x * n_chunks_y,
            'counts': counts,
            'min_traces': int(counts.min()),
            'max_traces': int(counts.max()),
            'mean_traces': float(counts.mean()),
            'total_trace_chunk_pairs': total_traces,
            'empty_chunks': int((counts == 0).sum()),
        }

    @property
    def n_traces(self) -> int:
        """Number of traces in index."""
        return self._n_traces

    @property
    def built(self) -> bool:
        """Whether index has been built."""
        return self._built


class GridBasedTraceIndex:
    """
    Grid-based trace index for very large datasets.

    Alternative to KD-tree when memory is constrained or
    when uniform spatial distribution allows for simpler binning.
    """

    def __init__(self, cell_size: float = 500.0):
        """
        Initialize grid-based index.

        Args:
            cell_size: Size of grid cells in meters
        """
        self.cell_size = cell_size
        self._grid: Optional[dict] = None
        self._bounds: Optional[TraceBounds] = None
        self._built: bool = False

    def build(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
    ) -> 'GridBasedTraceIndex':
        """Build grid-based index."""
        source_x = np.asarray(source_x)
        source_y = np.asarray(source_y)
        receiver_x = np.asarray(receiver_x)
        receiver_y = np.asarray(receiver_y)

        # Compute midpoints
        mid_x = (source_x + receiver_x) / 2
        mid_y = (source_y + receiver_y) / 2

        # Compute bounds
        self._bounds = TraceBounds(
            x_min=float(mid_x.min()),
            x_max=float(mid_x.max()),
            y_min=float(mid_y.min()),
            y_max=float(mid_y.max()),
        )

        # Assign traces to grid cells
        self._grid = {}
        for i, (x, y) in enumerate(zip(mid_x, mid_y)):
            cell_i = int((x - self._bounds.x_min) / self.cell_size)
            cell_j = int((y - self._bounds.y_min) / self.cell_size)
            key = (cell_i, cell_j)

            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(i)

        self._built = True
        return self

    def query_traces_for_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        buffer: float,
    ) -> np.ndarray:
        """Query traces for a region using grid."""
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        # Expand by buffer
        x_min -= buffer
        x_max += buffer
        y_min -= buffer
        y_max += buffer

        # Find overlapping cells
        cell_i_min = max(0, int((x_min - self._bounds.x_min) / self.cell_size))
        cell_i_max = int((x_max - self._bounds.x_min) / self.cell_size) + 1
        cell_j_min = max(0, int((y_min - self._bounds.y_min) / self.cell_size))
        cell_j_max = int((y_max - self._bounds.y_min) / self.cell_size) + 1

        indices = []
        for i in range(cell_i_min, cell_i_max + 1):
            for j in range(cell_j_min, cell_j_max + 1):
                key = (i, j)
                if key in self._grid:
                    indices.extend(self._grid[key])

        return np.array(indices, dtype=np.int64)


def create_trace_index(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    use_grid: bool = False,
    cell_size: float = 500.0,
) -> TraceSpatialIndex:
    """
    Factory function to create trace spatial index.

    Args:
        source_x, source_y: Source coordinates
        receiver_x, receiver_y: Receiver coordinates
        use_grid: If True, use grid-based index instead of KD-tree
        cell_size: Cell size for grid-based index

    Returns:
        Configured trace index
    """
    if use_grid:
        index = GridBasedTraceIndex(cell_size=cell_size)
    else:
        index = TraceSpatialIndex()

    index.build(source_x, source_y, receiver_x, receiver_y)
    return index
