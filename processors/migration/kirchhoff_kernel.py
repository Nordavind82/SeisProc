"""
Kirchhoff Migration Kernel for PSTM.

Implements the core migration algorithm using output-point centric approach:
1. For each output point and depth, compute traveltimes from all traces
2. Interpolate trace amplitudes at traveltimes
3. Sum weighted contributions from all traces

This version uses a proper Kirchhoff summation where each output point
gathers energy from multiple input traces within its aperture.

Supports:
- Output-point centric gather (correct Kirchhoff)
- Time-domain mapping (eliminates depth loop)
- KD-tree spatial index for fast aperture queries
- Time-dependent aperture
"""

import numpy as np
import torch
from typing import Tuple, Optional, Callable, Dict, List
import logging

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from processors.migration.geometry_preprocessor import PrecomputedGeometry

logger = logging.getLogger(__name__)


class TraceSpatialIndex:
    """
    KD-tree based spatial index for fast trace aperture queries.

    Maps trace midpoints to trace indices for efficient aperture-based filtering.
    Instead of checking all traces against each output point, query nearby traces.

    Usage:
        index = TraceSpatialIndex()
        index.build(source_x, source_y, receiver_x, receiver_y)
        trace_indices = index.query_aperture(output_x, output_y, aperture_radius)
    """

    def __init__(self):
        self.kdtree = None
        self.midpoints = None
        self.n_traces = 0
        self._built = False

    def build(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
    ) -> None:
        """
        Build spatial index from source/receiver coordinates.

        Args:
            source_x: Source X coordinates (n_traces,)
            source_y: Source Y coordinates (n_traces,)
            receiver_x: Receiver X coordinates (n_traces,)
            receiver_y: Receiver Y coordinates (n_traces,)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, KD-tree index disabled")
            return

        self.n_traces = len(source_x)

        # Compute midpoints
        self.midpoints = np.column_stack([
            (source_x + receiver_x) / 2.0,
            (source_y + receiver_y) / 2.0,
        ])

        # Build KD-tree
        self.kdtree = cKDTree(self.midpoints)
        self._built = True

        logger.info(f"TraceSpatialIndex: built KD-tree with {self.n_traces} traces")

    def query_aperture(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> np.ndarray:
        """
        Query traces within aperture radius of a point.

        Args:
            x: Query point X coordinate
            y: Query point Y coordinate
            radius: Aperture radius in meters

        Returns:
            Array of trace indices within aperture
        """
        if not self._built or self.kdtree is None:
            # Return all traces if index not built
            return np.arange(self.n_traces)

        indices = self.kdtree.query_ball_point([x, y], radius)
        return np.array(indices, dtype=np.int64)

    def query_aperture_batch(
        self,
        points: np.ndarray,
        radius: float,
    ) -> List[np.ndarray]:
        """
        Query traces for multiple points at once.

        Args:
            points: (n_points, 2) array of [x, y] coordinates
            radius: Aperture radius in meters

        Returns:
            List of arrays of trace indices for each point
        """
        if not self._built or self.kdtree is None:
            return [np.arange(self.n_traces) for _ in range(len(points))]

        return self.kdtree.query_ball_point(points, radius)

    def query_tile(
        self,
        tile_x: np.ndarray,
        tile_y: np.ndarray,
        max_aperture: float,
    ) -> np.ndarray:
        """
        Get traces that could contribute to any point in a tile.

        Uses tile bounding box + aperture for conservative filtering.

        Args:
            tile_x: X coordinates of tile points
            tile_y: Y coordinates of tile points
            max_aperture: Maximum aperture radius

        Returns:
            Array of trace indices that could contribute to the tile
        """
        if not self._built or self.kdtree is None:
            return np.arange(self.n_traces)

        # Tile bounding box
        x_min, x_max = float(tile_x.min()), float(tile_x.max())
        y_min, y_max = float(tile_y.min()), float(tile_y.max())

        # Tile center and diagonal radius
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        tile_radius = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) / 2

        # Query with expanded radius
        search_radius = tile_radius + max_aperture + 100  # extra margin
        indices = self.kdtree.query_ball_point([cx, cy], search_radius)

        return np.array(indices, dtype=np.int64)

    @property
    def is_built(self) -> bool:
        return self._built


def interpolate_traces(
    traces: torch.Tensor,
    traveltimes_ms: torch.Tensor,
    dt_ms: float,
    t_min_ms: float = 0.0,
) -> torch.Tensor:
    """
    Interpolate trace amplitudes at specified traveltimes.

    Uses linear interpolation between adjacent samples.

    Args:
        traces: Trace data (n_samples, n_traces)
        traveltimes_ms: Traveltimes in ms (n_traces, n_depths)
        dt_ms: Sample interval in ms
        t_min_ms: Start time in ms

    Returns:
        amplitudes: Interpolated amplitudes (n_traces, n_depths)
    """
    n_samples, n_traces = traces.shape
    n_traces_tt, n_depths = traveltimes_ms.shape

    assert n_traces == n_traces_tt, f"Trace count mismatch: {n_traces} vs {n_traces_tt}"

    # Convert traveltime to sample index (fractional)
    sample_idx = (traveltimes_ms - t_min_ms) / dt_ms  # (n_traces, n_depths)

    # Clamp to valid range
    sample_idx = torch.clamp(sample_idx, 0, n_samples - 1.001)

    # Integer indices for floor and ceil
    idx_floor = sample_idx.long()
    idx_ceil = torch.clamp(idx_floor + 1, 0, n_samples - 1)

    # Fractional part for interpolation
    frac = sample_idx - idx_floor.float()

    # Create trace index for gathering
    # trace_idx[i, j] = i for all j (each row indexes the same trace)
    trace_idx = torch.arange(n_traces, device=traces.device).view(n_traces, 1).expand(-1, n_depths)

    # Gather amplitudes - traces is (n_samples, n_traces), need to index [sample, trace]
    # Flatten for advanced indexing
    idx_f_flat = idx_floor.reshape(-1)
    idx_c_flat = idx_ceil.reshape(-1)
    trace_idx_flat = trace_idx.reshape(-1)

    amp_floor = traces[idx_f_flat, trace_idx_flat].reshape(n_traces, n_depths)
    amp_ceil = traces[idx_c_flat, trace_idx_flat].reshape(n_traces, n_depths)

    # Linear interpolation
    amplitudes = amp_floor + frac * (amp_ceil - amp_floor)

    return amplitudes


def scatter_add_migration(
    amplitudes: torch.Tensor,
    weights: torch.Tensor,
    aperture_mask: torch.Tensor,
    output_il: torch.Tensor,
    output_xl: torch.Tensor,
    valid_mask: torch.Tensor,
    output_image: torch.Tensor,
    output_fold: torch.Tensor,
) -> None:
    """
    Scatter-add weighted amplitudes to output image.

    Each trace contributes a depth column to its assigned output location.
    Multiple traces at the same location are summed (scattered).

    Args:
        amplitudes: Interpolated amplitudes (n_traces, n_depths)
        weights: Migration weights (n_traces, n_depths)
        aperture_mask: Aperture mask (n_traces, n_depths)
        output_il: Inline indices (n_traces,)
        output_xl: Crossline indices (n_traces,)
        valid_mask: Valid trace mask (n_traces,)
        output_image: Output image (n_depths, n_il, n_xl) - modified in place
        output_fold: Output fold (n_depths, n_il, n_xl) - modified in place
    """
    n_traces, n_depths = amplitudes.shape
    n_z, n_il, n_xl = output_image.shape

    assert n_depths == n_z, f"Depth mismatch: {n_depths} vs {n_z}"

    # Apply weights and mask
    weighted_amp = amplitudes * weights * aperture_mask.float()  # (n_traces, n_depths)

    # Filter to valid traces only
    valid_indices = torch.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        return

    valid_amp = weighted_amp[valid_indices]  # (n_valid, n_depths)
    valid_il = output_il[valid_indices]       # (n_valid,)
    valid_xl = output_xl[valid_indices]       # (n_valid,)
    valid_aperture = aperture_mask[valid_indices].float()  # (n_valid, n_depths)

    # Convert (il, xl) to linear index for scatter_add
    # linear_idx = il * n_xl + xl
    linear_idx = valid_il * n_xl + valid_xl  # (n_valid,)

    # Expand linear_idx to match depths
    # linear_idx_exp[d, i] = linear_idx[i] for all d
    linear_idx_exp = linear_idx.unsqueeze(0).expand(n_depths, -1)  # (n_depths, n_valid)

    # Transpose amplitudes for scatter: (n_depths, n_valid)
    valid_amp_t = valid_amp.t()  # (n_depths, n_valid)
    valid_aperture_t = valid_aperture.t()  # (n_depths, n_valid)

    # Flatten output for scatter_add
    output_flat = output_image.reshape(n_depths, -1)  # (n_depths, n_il * n_xl)
    fold_flat = output_fold.reshape(n_depths, -1)

    # Scatter add - for each depth, add contributions to output
    output_flat.scatter_add_(1, linear_idx_exp, valid_amp_t)
    fold_flat.scatter_add_(1, linear_idx_exp, valid_aperture_t)


class KirchhoffKernel:
    """
    Kirchhoff migration kernel.

    Performs migration using precomputed geometry:
    1. Interpolate traces at traveltimes
    2. Apply weights
    3. Scatter-add to output

    Example:
        kernel = KirchhoffKernel(device)
        image, fold = kernel.migrate(traces, precomputed, dt_ms, n_il, n_xl)
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize kernel.

        Args:
            device: Target device (CPU/GPU)
        """
        self.device = device or torch.device('cpu')

    def migrate(
        self,
        traces: torch.Tensor,
        precomputed: PrecomputedGeometry,
        dt_ms: float,
        t_min_ms: float,
        n_il: int,
        n_xl: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Migrate traces to output image.

        Args:
            traces: Trace data (n_samples, n_traces) on device
            precomputed: Precomputed geometry from GeometryPreprocessor
            dt_ms: Sample interval in ms
            t_min_ms: Start time in ms
            n_il: Number of inlines in output
            n_xl: Number of crosslines in output

        Returns:
            image: Migrated image (n_depths, n_il, n_xl)
            fold: Fold count (n_depths, n_il, n_xl)
        """
        n_samples, n_traces = traces.shape
        n_depths = precomputed.n_depths

        logger.debug(f"Migrating {n_traces} traces, {n_depths} depths to {n_il}x{n_xl} grid")

        # Initialize output tensors
        output_image = torch.zeros(n_depths, n_il, n_xl, device=self.device, dtype=torch.float32)
        output_fold = torch.zeros(n_depths, n_il, n_xl, device=self.device, dtype=torch.float32)

        # Step 1: Interpolate traces at traveltimes
        amplitudes = interpolate_traces(traces, precomputed.traveltimes_ms, dt_ms, t_min_ms)

        # Step 2: Scatter-add to output
        scatter_add_migration(
            amplitudes,
            precomputed.weights,
            precomputed.aperture_mask,
            precomputed.output_il,
            precomputed.output_xl,
            precomputed.valid_mask,
            output_image,
            output_fold,
        )

        return output_image, output_fold

    def migrate_batched(
        self,
        traces: torch.Tensor,
        precomputed: PrecomputedGeometry,
        dt_ms: float,
        t_min_ms: float,
        n_il: int,
        n_xl: int,
        depth_batch_size: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Migrate traces with depth batching for memory efficiency.

        Processes depths in batches to limit peak memory usage.

        Args:
            traces: Trace data (n_samples, n_traces)
            precomputed: Precomputed geometry
            dt_ms: Sample interval in ms
            t_min_ms: Start time in ms
            n_il: Number of inlines
            n_xl: Number of crosslines
            depth_batch_size: Number of depths per batch

        Returns:
            image: Migrated image (n_depths, n_il, n_xl)
            fold: Fold count (n_depths, n_il, n_xl)
        """
        n_samples, n_traces = traces.shape
        n_depths = precomputed.n_depths

        # Initialize output
        output_image = torch.zeros(n_depths, n_il, n_xl, device=self.device, dtype=torch.float32)
        output_fold = torch.zeros(n_depths, n_il, n_xl, device=self.device, dtype=torch.float32)

        # Process in depth batches
        for d_start in range(0, n_depths, depth_batch_size):
            d_end = min(d_start + depth_batch_size, n_depths)

            # Slice precomputed geometry for this depth range
            batch_traveltimes = precomputed.traveltimes_ms[:, d_start:d_end]
            batch_weights = precomputed.weights[:, d_start:d_end]
            batch_aperture = precomputed.aperture_mask[:, d_start:d_end]

            # Interpolate for this depth batch
            amplitudes = interpolate_traces(traces, batch_traveltimes, dt_ms, t_min_ms)

            # Create temporary output for this batch
            batch_image = output_image[d_start:d_end]
            batch_fold = output_fold[d_start:d_end]

            # Scatter-add
            scatter_add_migration(
                amplitudes,
                batch_weights,
                batch_aperture,
                precomputed.output_il,
                precomputed.output_xl,
                precomputed.valid_mask,
                batch_image,
                batch_fold,
            )

        return output_image, output_fold


def normalize_by_fold(
    image: torch.Tensor,
    fold: torch.Tensor,
    min_fold: int = 1,
) -> torch.Tensor:
    """
    Normalize migrated image by fold.

    Args:
        image: Migrated image (n_depths, n_il, n_xl)
        fold: Fold count (n_depths, n_il, n_xl)
        min_fold: Minimum fold for normalization (avoid divide by zero)

    Returns:
        Normalized image (same shape)
    """
    normalized = image.clone()
    mask = fold >= min_fold
    normalized[mask] = image[mask] / fold[mask]
    return normalized


def migrate_kirchhoff_full(
    traces: torch.Tensor,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    output_x: torch.Tensor,
    output_y: torch.Tensor,
    depth_axis: torch.Tensor,
    velocity: float,
    dt_ms: float,
    t_min_ms: float,
    max_aperture_m: float,
    max_angle_deg: float,
    n_il: int,
    n_xl: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    enable_profiling: bool = False,
    use_time_dependent_aperture: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full Kirchhoff migration - output-point centric.

    For each output point, gathers energy from ALL traces within aperture.
    This is the correct Kirchhoff summation but is memory intensive.

    Args:
        traces: Input traces (n_samples, n_traces)
        source_x, source_y: Source coordinates (n_traces,)
        receiver_x, receiver_y: Receiver coordinates (n_traces,)
        output_x, output_y: Output grid coordinates (n_il * n_xl,)
        depth_axis: Depth values in meters (n_depths,)
        velocity: Velocity in m/s
        dt_ms: Sample interval in ms
        t_min_ms: Start time in ms
        max_aperture_m: Maximum aperture in meters (used as cap when time-dependent)
        max_angle_deg: Maximum angle from vertical
        n_il, n_xl: Output grid dimensions
        progress_callback: Optional callback(percent, message) for progress updates
        enable_profiling: If True, collect detailed timing statistics
        use_time_dependent_aperture: If True, use aperture(z) = min(max_aperture, z*tan(angle))

    Returns:
        image: (n_depths, n_il, n_xl)
        fold: (n_depths, n_il, n_xl)
    """
    device = traces.device
    n_samples, n_traces = traces.shape
    n_depths = depth_axis.shape[0]
    n_output = n_il * n_xl

    # Initialize output
    image = torch.zeros(n_depths, n_il, n_xl, device=device)
    fold = torch.zeros(n_depths, n_il, n_xl, device=device)

    max_angle_rad = np.radians(max_angle_deg)
    tan_max_angle = np.tan(max_angle_rad)  # Pre-compute for time-dependent aperture

    # Pre-compute inverse velocity for traveltime calculation
    inv_velocity_ms = 1000.0 / velocity  # Convert to ms

    # Process each output point
    # For efficiency, we process in tiles of output points
    tile_size = 100  # Process 100 output points at a time
    n_tiles = (n_output + tile_size - 1) // tile_size

    import time
    start_time = time.time()
    last_progress_time = start_time

    # Log key parameters once
    logger.info(f"Kernel params: n_traces={n_traces}, n_depths={n_depths}, "
                f"n_output={n_output}, tile_size={tile_size}, n_tiles={n_tiles}")
    logger.info(f"Aperture: max={max_aperture_m}m, max_angle={max_angle_deg}°, "
                f"time_dependent={use_time_dependent_aperture}")

    # Profiling accumulators
    if enable_profiling:
        profile_times = {
            'horizontal_dist': 0.0,
            'sqrt_horizontal': 0.0,
            'depth_loop_total': 0.0,
            'sqrt_ray_dist': 0.0,
            'angle_computation': 0.0,
            'aperture_mask': 0.0,
            'traveltime': 0.0,
            'weight': 0.0,
            'interpolation': 0.0,
            'sum_accumulate': 0.0,
        }
        profile_counts = {
            'tiles_processed': 0,
            'depths_processed': 0,
            'total_trace_pairs': 0,
            'aperture_passed': 0,
        }
        # Per-depth aperture statistics (sampled at 10 depths)
        aperture_by_depth = []
        sample_depths = set([0, n_depths // 4, n_depths // 2, 3 * n_depths // 4, n_depths - 1])
        sample_depths = {d for d in sample_depths if d < n_depths}

    for tile_idx, out_start in enumerate(range(0, n_output, tile_size)):
        out_end = min(out_start + tile_size, n_output)
        n_tile = out_end - out_start

        # Get output coordinates for this tile
        tile_x = output_x[out_start:out_end]  # (n_tile,)
        tile_y = output_y[out_start:out_end]

        # Compute inline/crossline indices
        tile_il = torch.arange(out_start, out_end, device=device) // n_xl
        tile_xl = torch.arange(out_start, out_end, device=device) % n_xl

        # Pre-filter traces: find traces that could contribute to ANY output point in this tile
        # Use tile bounding box center for rough distance filtering
        tile_cx = (tile_x.min() + tile_x.max()) / 2
        tile_cy = (tile_y.min() + tile_y.max()) / 2
        tile_radius = torch.sqrt(((tile_x - tile_cx)**2 + (tile_y - tile_cy)**2).max())

        # Source midpoint for each trace
        mid_x = (source_x + receiver_x) / 2
        mid_y = (source_y + receiver_y) / 2

        # Distance from tile center to trace midpoint
        dist_to_center = torch.sqrt((mid_x - tile_cx)**2 + (mid_y - tile_cy)**2)

        # A trace can contribute if its midpoint is within (max_aperture + tile_radius)
        # This is a conservative filter - may include traces that ultimately don't contribute
        trace_filter = dist_to_center < (max_aperture_m + tile_radius + 100)  # +100m margin

        # Get indices of filtered traces
        filtered_indices = torch.where(trace_filter)[0]
        n_filtered = len(filtered_indices)

        # If no traces can contribute, skip this tile
        if n_filtered == 0:
            continue

        # Extract filtered trace data
        source_x_f = source_x[filtered_indices]
        source_y_f = source_y[filtered_indices]
        receiver_x_f = receiver_x[filtered_indices]
        receiver_y_f = receiver_y[filtered_indices]
        traces_f = traces[:, filtered_indices]

        # For each output point, compute traveltime to filtered traces at all depths
        # tile_x: (n_tile,), source_x_f: (n_filtered,), depth_axis: (n_depths,)

        if enable_profiling:
            t0 = time.time()

        # Horizontal distances: (n_tile, n_filtered)
        dx_src = tile_x.unsqueeze(1) - source_x_f.unsqueeze(0)  # (n_tile, n_filtered)
        dy_src = tile_y.unsqueeze(1) - source_y_f.unsqueeze(0)
        dx_rcv = tile_x.unsqueeze(1) - receiver_x_f.unsqueeze(0)
        dy_rcv = tile_y.unsqueeze(1) - receiver_y_f.unsqueeze(0)

        h_src_sq = dx_src**2 + dy_src**2  # (n_tile, n_filtered)
        h_rcv_sq = dx_rcv**2 + dy_rcv**2

        if enable_profiling:
            if device.type != 'cpu':
                torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize()
            profile_times['horizontal_dist'] += time.time() - t0
            t0 = time.time()

        h_src = torch.sqrt(h_src_sq)
        h_rcv = torch.sqrt(h_rcv_sq)

        if enable_profiling:
            if device.type != 'cpu':
                torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize()
            profile_times['sqrt_horizontal'] += time.time() - t0

        # Aperture mask based on horizontal distance (constant aperture)
        # This is used as a pre-filter; depth-dependent aperture applied inside loop
        aperture_mask_h = (h_src < max_aperture_m) & (h_rcv < max_aperture_m)  # (n_tile, n_filtered)

        if enable_profiling:
            t_depth_start = time.time()

        # Depth batching: process multiple depths at once for better GPU utilization
        # Memory limit: ~500MB working memory per batch
        bytes_per_depth = n_tile * n_filtered * 4 * 8  # 8 float32 tensors per depth
        max_depth_batch = max(1, min(50, int(500_000_000 / max(bytes_per_depth, 1))))

        # Process depths in batches
        for d_batch_start in range(0, n_depths, max_depth_batch):
            d_batch_end = min(d_batch_start + max_depth_batch, n_depths)
            d_batch_size = d_batch_end - d_batch_start

            # Get depth values for this batch
            z_batch = depth_axis[d_batch_start:d_batch_end]  # (d_batch_size,)
            z_sq_batch = z_batch ** 2  # (d_batch_size,)

            # Expand for broadcasting: h_src_sq is (n_tile, n_filtered)
            # We want r_src to be (d_batch_size, n_tile, n_filtered)
            z_sq_expanded = z_sq_batch.view(-1, 1, 1)  # (d_batch_size, 1, 1)
            h_src_sq_expanded = h_src_sq.unsqueeze(0)  # (1, n_tile, n_filtered)
            h_rcv_sq_expanded = h_rcv_sq.unsqueeze(0)

            # Ray distances for all depths in batch
            r_src_batch = torch.sqrt(h_src_sq_expanded + z_sq_expanded)  # (d_batch_size, n_tile, n_filtered)
            r_rcv_batch = torch.sqrt(h_rcv_sq_expanded + z_sq_expanded)

            # Expand h_src, h_rcv for angle computation
            h_src_expanded = h_src.unsqueeze(0)  # (1, n_tile, n_filtered)
            h_rcv_expanded = h_rcv.unsqueeze(0)
            z_expanded = z_batch.view(-1, 1, 1)  # (d_batch_size, 1, 1)

            # Angles from vertical
            angle_src_batch = torch.atan2(h_src_expanded, z_expanded + 1e-6)
            angle_rcv_batch = torch.atan2(h_rcv_expanded, z_expanded + 1e-6)

            # Time-dependent aperture mask for batch
            if use_time_dependent_aperture:
                aperture_at_depth_batch = torch.clamp(z_batch * tan_max_angle, max=max_aperture_m)
                aperture_at_depth_expanded = aperture_at_depth_batch.view(-1, 1, 1)
                aperture_mask_depth_batch = (h_src_expanded < aperture_at_depth_expanded) & \
                                            (h_rcv_expanded < aperture_at_depth_expanded)
            else:
                aperture_mask_depth_batch = aperture_mask_h.unsqueeze(0).expand(d_batch_size, -1, -1)

            # Full aperture mask
            aperture_batch = aperture_mask_depth_batch & \
                            (angle_src_batch < max_angle_rad) & \
                            (angle_rcv_batch < max_angle_rad)

            # Traveltime in ms
            traveltime_ms_batch = (r_src_batch + r_rcv_batch) * inv_velocity_ms

            # Weight: obliquity and spreading
            cos_src_batch = z_expanded / (r_src_batch + 1e-6)
            cos_rcv_batch = z_expanded / (r_rcv_batch + 1e-6)
            weight_batch = (cos_src_batch * cos_rcv_batch) / (r_src_batch * r_rcv_batch + 1e-6)

            # Pre-create trace index for gather operations
            trace_idx = torch.arange(n_filtered, device=device).unsqueeze(0).expand(n_tile, -1)
            trace_idx_flat = trace_idx.reshape(-1)

            # Interpolate traces at traveltimes - process each depth in mini-batch
            for d_local in range(d_batch_size):
                d_idx = d_batch_start + d_local

                traveltime_ms = traveltime_ms_batch[d_local]
                weight = weight_batch[d_local]
                aperture = aperture_batch[d_local]

                # Interpolation indices
                sample_idx = (traveltime_ms - t_min_ms) / dt_ms
                sample_idx = torch.clamp(sample_idx, 0, n_samples - 1.001)

                idx_floor = sample_idx.long()
                idx_ceil = (idx_floor + 1).clamp(max=n_samples - 1)
                frac = sample_idx - idx_floor.float()

                # Gather from filtered traces using pre-computed trace index
                idx_floor_flat = idx_floor.reshape(-1)
                idx_ceil_flat = idx_ceil.reshape(-1)

                amp_floor = traces_f[idx_floor_flat, trace_idx_flat].reshape(n_tile, n_filtered)
                amp_ceil = traces_f[idx_ceil_flat, trace_idx_flat].reshape(n_tile, n_filtered)
                amp = amp_floor + frac * (amp_ceil - amp_floor)

                # Apply weight and aperture, then sum over traces
                weighted_amp = amp * weight * aperture.float()
                summed = weighted_amp.sum(dim=1)
                fold_count = aperture.float().sum(dim=1)

                # Write to output
                image[d_idx, tile_il, tile_xl] = summed
                fold[d_idx, tile_il, tile_xl] = fold_count

                if enable_profiling:
                    profile_counts['depths_processed'] += 1
                    profile_counts['total_trace_pairs'] += n_tile * n_filtered
                    aperture_count = int(aperture.sum().item())
                    profile_counts['aperture_passed'] += aperture_count

                    # Sample aperture stats at key depths (first tile only)
                    if tile_idx == 0 and d_idx in sample_depths:
                        z = float(z_batch[d_local])
                        depth_m = z
                        time_ms = depth_m / velocity * 2000.0
                        aperture_pct = aperture_count / (n_tile * n_filtered) * 100 if n_filtered > 0 else 0
                        aperture_by_depth.append({
                            'depth_idx': d_idx,
                            'depth_m': depth_m,
                            'time_ms': time_ms,
                            'aperture_pct': aperture_pct,
                            'traces_in_aperture': aperture_count,
                            'total_traces': n_tile * n_filtered,
                            'filtered_of_total': f"{n_filtered}/{n_traces}",
                        })

        if enable_profiling:
            if device.type != 'cpu':
                torch.mps.synchronize() if device.type == 'mps' else torch.cuda.synchronize()
            profile_times['depth_loop_total'] += time.time() - t_depth_start
            profile_counts['tiles_processed'] += 1

        # Progress reporting (every 2 seconds or every 10% of tiles)
        if progress_callback is not None:
            current_time = time.time()
            if current_time - last_progress_time >= 2.0 or (tile_idx + 1) % max(1, n_tiles // 10) == 0:
                last_progress_time = current_time
                pct = (tile_idx + 1) / n_tiles
                elapsed = current_time - start_time
                if pct > 0:
                    eta = elapsed / pct * (1 - pct)
                    rate = (tile_idx + 1) * tile_size * n_depths / elapsed
                    progress_callback(
                        pct,
                        f"Tile {tile_idx+1}/{n_tiles} ({pct*100:.1f}%), "
                        f"{rate:.0f} output-samples/s, ETA: {eta:.0f}s"
                    )

    # Log profiling results
    if enable_profiling:
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("KIRCHHOFF MIGRATION PROFILING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("")
        logger.info("TIME BREAKDOWN:")
        logger.info("-" * 40)

        # Sort by time spent
        sorted_times = sorted(profile_times.items(), key=lambda x: x[1], reverse=True)
        for name, t in sorted_times:
            pct = (t / total_time * 100) if total_time > 0 else 0
            logger.info(f"  {name:20s}: {t:8.2f}s ({pct:5.1f}%)")

        logger.info("")
        logger.info("OPERATION COUNTS:")
        logger.info("-" * 40)
        logger.info(f"  Tiles processed:     {profile_counts['tiles_processed']:,}")
        logger.info(f"  Depths processed:    {profile_counts['depths_processed']:,}")
        logger.info(f"  Total trace pairs:   {profile_counts['total_trace_pairs']:,}")
        logger.info(f"  Aperture passed:     {profile_counts['aperture_passed']:,}")

        if profile_counts['total_trace_pairs'] > 0:
            aperture_ratio = profile_counts['aperture_passed'] / profile_counts['total_trace_pairs'] * 100
            logger.info(f"  Aperture pass rate:  {aperture_ratio:.1f}%")

        logger.info("")
        logger.info("DERIVED METRICS:")
        logger.info("-" * 40)
        if total_time > 0:
            ops_per_sec = profile_counts['total_trace_pairs'] / total_time
            logger.info(f"  Trace-pairs/sec:     {ops_per_sec:,.0f}")
            depths_per_sec = profile_counts['depths_processed'] / total_time
            logger.info(f"  Depth-samples/sec:   {depths_per_sec:,.0f}")

        # Per-depth aperture statistics
        if aperture_by_depth:
            logger.info("")
            logger.info("APERTURE BY DEPTH (sampled from first tile):")
            logger.info("-" * 60)
            logger.info(f"  {'Time(ms)':>10} {'Depth(m)':>10} {'Aperture%':>10} {'In/Total':>20}")
            logger.info("-" * 60)
            for stats in sorted(aperture_by_depth, key=lambda x: x['depth_idx']):
                in_ap = stats['traces_in_aperture']
                total = stats['total_traces']
                logger.info(
                    f"  {stats['time_ms']:10.0f} {stats['depth_m']:10.0f} "
                    f"{stats['aperture_pct']:9.1f}% {in_ap:>8,}/{total:<10,}"
                )
            logger.info("")
            logger.info("OPTIMIZATION INSIGHT:")
            logger.info("-" * 40)
            # Calculate potential savings with time-dependent aperture
            if len(aperture_by_depth) >= 2:
                # Find first non-zero shallow and the deepest
                shallow = None
                for s in aperture_by_depth:
                    if s['aperture_pct'] > 0:
                        shallow = s
                        break
                deep = aperture_by_depth[-1]
                if shallow and deep['aperture_pct'] > shallow['aperture_pct']:
                    ratio = deep['aperture_pct'] / shallow['aperture_pct']
                    logger.info(f"  Aperture at shallow ({shallow['time_ms']:.0f}ms): {shallow['aperture_pct']:.1f}%")
                    logger.info(f"  Aperture at deep ({deep['time_ms']:.0f}ms): {deep['aperture_pct']:.1f}%")
                    if ratio > 1:
                        savings = (1 - 1/ratio) * 100
                        logger.info(f"  Time-dependent aperture could reduce work by ~{savings:.0f}%")
                        logger.info(f"  at shallow times if using aperture(t) = v*t/2*tan(angle)")
                elif deep['aperture_pct'] < 100:
                    logger.info(f"  Max aperture rate: {deep['aperture_pct']:.1f}% at {deep['time_ms']:.0f}ms")
                    logger.info(f"  Consider increasing max_aperture_m or max_angle_deg")

        logger.info("=" * 60)

    return image, fold


def migrate_kirchhoff_time_domain(
    traces: torch.Tensor,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    output_x: torch.Tensor,
    output_y: torch.Tensor,
    time_axis_ms: torch.Tensor,
    velocity: float,
    dt_ms: float,
    t_min_ms: float,
    max_aperture_m: float,
    max_angle_deg: float,
    n_il: int,
    n_xl: int,
    tile_size: int = 100,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    enable_profiling: bool = False,
    use_time_dependent_aperture: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Direct time-domain Kirchhoff migration - eliminates depth loop.

    Instead of iterating over depths, this maps input time samples directly
    to output time samples using the equation:
        t_out = sqrt(t_in² + 4*h²/v²)

    where h is the horizontal distance from trace midpoint to output point.

    This is ~50-100x faster than depth-domain migration for time migration
    because it eliminates the expensive depth loop entirely.

    Args:
        traces: Input traces (n_samples, n_traces)
        source_x, source_y: Source coordinates (n_traces,)
        receiver_x, receiver_y: Receiver coordinates (n_traces,)
        output_x, output_y: Output grid coordinates (n_il * n_xl,)
        time_axis_ms: Output time values in ms (n_times,)
        velocity: Constant velocity in m/s
        dt_ms: Sample interval in ms
        t_min_ms: Start time in ms
        max_aperture_m: Maximum aperture in meters
        max_angle_deg: Maximum angle from vertical
        n_il, n_xl: Output grid dimensions
        tile_size: Number of output points per tile (default 100)
        progress_callback: Optional callback(percent, message)
        enable_profiling: If True, collect detailed timing statistics
        use_time_dependent_aperture: If True, use aperture(t) = v*t/2*tan(angle)

    Returns:
        image: (n_times, n_il, n_xl)
        fold: (n_times, n_il, n_xl)
    """
    import time

    device = traces.device
    n_samples, n_traces = traces.shape
    n_times = time_axis_ms.shape[0]
    n_output = n_il * n_xl

    # Initialize output
    image = torch.zeros(n_times, n_il, n_xl, device=device)
    fold = torch.zeros(n_times, n_il, n_xl, device=device)

    # Pre-compute constants
    max_angle_rad = np.radians(max_angle_deg)
    tan_max_angle = np.tan(max_angle_rad)
    v_squared = velocity * velocity
    four_over_v_sq = 4.0 / v_squared

    # For time-dependent aperture: aperture(t) = v * t / 2 * tan(angle)
    # This gives the horizontal distance at which rays reach max angle at time t
    aperture_factor = velocity / 2.0 * tan_max_angle  # aperture = t_ms * aperture_factor / 1000

    # Compute trace midpoints and half-offsets
    mid_x = (source_x + receiver_x) / 2.0
    mid_y = (source_y + receiver_y) / 2.0
    half_offset_x = (receiver_x - source_x) / 2.0
    half_offset_y = (receiver_y - source_y) / 2.0
    half_offset_sq = half_offset_x**2 + half_offset_y**2  # (n_traces,)

    # Process in tiles
    n_tiles = (n_output + tile_size - 1) // tile_size

    start_time = time.time()
    last_progress_time = start_time

    logger.info(f"Time-domain kernel: n_traces={n_traces}, n_times={n_times}, "
                f"n_output={n_output}, tile_size={tile_size}, n_tiles={n_tiles}")
    logger.info(f"Aperture: max={max_aperture_m}m, max_angle={max_angle_deg}°, "
                f"time_dependent={use_time_dependent_aperture}")

    # Profiling
    if enable_profiling:
        profile_times = {
            'trace_filtering': 0.0,
            'time_mapping': 0.0,
            'aperture_mask': 0.0,
            'interpolation': 0.0,
            'accumulation': 0.0,
        }
        profile_counts = {
            'tiles_processed': 0,
            'total_trace_point_pairs': 0,
            'aperture_passed': 0,
        }

    for tile_idx, out_start in enumerate(range(0, n_output, tile_size)):
        out_end = min(out_start + tile_size, n_output)
        n_tile = out_end - out_start

        # Get output coordinates for this tile
        tile_x = output_x[out_start:out_end]  # (n_tile,)
        tile_y = output_y[out_start:out_end]

        # Compute inline/crossline indices
        tile_il = torch.arange(out_start, out_end, device=device) // n_xl
        tile_xl = torch.arange(out_start, out_end, device=device) % n_xl

        if enable_profiling:
            t0 = time.time()

        # Pre-filter traces by distance to tile center
        tile_cx = (tile_x.min() + tile_x.max()) / 2
        tile_cy = (tile_y.min() + tile_y.max()) / 2
        tile_radius = torch.sqrt(((tile_x - tile_cx)**2 + (tile_y - tile_cy)**2).max())

        dist_to_center = torch.sqrt((mid_x - tile_cx)**2 + (mid_y - tile_cy)**2)
        trace_filter = dist_to_center < (max_aperture_m + tile_radius + 100)

        filtered_indices = torch.where(trace_filter)[0]
        n_filtered = len(filtered_indices)

        if n_filtered == 0:
            continue

        # Extract filtered data
        mid_x_f = mid_x[filtered_indices]
        mid_y_f = mid_y[filtered_indices]
        half_offset_sq_f = half_offset_sq[filtered_indices]
        traces_f = traces[:, filtered_indices]  # (n_samples, n_filtered)

        if enable_profiling:
            if device.type == 'mps':
                torch.mps.synchronize()
            profile_times['trace_filtering'] += time.time() - t0
            t0 = time.time()

        # Horizontal distance from each output point to each trace midpoint
        # (n_tile, n_filtered)
        dx = tile_x.unsqueeze(1) - mid_x_f.unsqueeze(0)
        dy = tile_y.unsqueeze(1) - mid_y_f.unsqueeze(0)
        h_sq = dx**2 + dy**2  # horizontal distance squared
        h = torch.sqrt(h_sq)

        # For each input time sample, compute output time
        # t_out² = t_in² + 4*h²/v² (NMO equation for zero-offset equivalent)
        # But we also need to account for the half-offset
        # Full equation: t_out = sqrt((t_in)² + 4*(h² + half_offset²)/v²)
        # where h is horizontal distance to midpoint

        # Actually, for Kirchhoff PSTM with constant velocity:
        # t_out = (r_src + r_rcv) / v
        # where r_src = sqrt(h_src² + z²), r_rcv = sqrt(h_rcv² + z²)
        # and z = v * t_out / 2 (for zero-offset time)

        # Simplified approach: use the diffraction equation
        # For each input sample at t_in, it contributes to output at t_out where:
        # t_out = sqrt(t_in² + 4*h_eff²/v²)
        # h_eff² = h² + half_offset² (effective horizontal distance)

        h_eff_sq = h_sq + half_offset_sq_f.unsqueeze(0)  # (n_tile, n_filtered)

        # Time shift term: 4*h_eff²/v² in ms²
        time_shift_sq = h_eff_sq * four_over_v_sq * 1e6  # convert to ms² (h in m, v in m/s)

        if enable_profiling:
            if device.type == 'mps':
                torch.mps.synchronize()
            profile_times['time_mapping'] += time.time() - t0

        # Process input samples in batches to manage memory
        # For each input sample, compute where it maps in output time
        sample_batch_size = min(200, n_samples)

        for sample_start in range(0, n_samples, sample_batch_size):
            sample_end = min(sample_start + sample_batch_size, n_samples)
            n_sample_batch = sample_end - sample_start

            if enable_profiling:
                t0 = time.time()

            # Input times for this batch (ms)
            t_in_ms = t_min_ms + torch.arange(sample_start, sample_end, device=device) * dt_ms
            t_in_sq = t_in_ms ** 2  # (n_sample_batch,)

            # Output times: t_out² = t_in² + time_shift_sq
            # Shape: (n_sample_batch, n_tile, n_filtered)
            t_out_sq = t_in_sq.view(-1, 1, 1) + time_shift_sq.unsqueeze(0)
            t_out_ms = torch.sqrt(t_out_sq)

            # Convert output time to output sample index
            out_sample_idx = (t_out_ms - t_min_ms) / dt_ms

            # Valid output range
            valid_time = (out_sample_idx >= 0) & (out_sample_idx < n_times - 0.001)

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['time_mapping'] += time.time() - t0
                t0 = time.time()

            # Aperture mask
            if use_time_dependent_aperture:
                # aperture(t) = v * t / 2 * tan(angle) in meters
                # t_out_ms is in ms, so: aperture = t_out_ms * v / 2000 * tan(angle)
                aperture_at_t = t_out_ms * aperture_factor / 1000.0  # (n_sample_batch, n_tile, n_filtered)
                aperture_at_t = torch.clamp(aperture_at_t, max=max_aperture_m)
                aperture_mask = h.unsqueeze(0) < aperture_at_t
            else:
                aperture_mask = h.unsqueeze(0) < max_aperture_m

            # Angle mask: check if angle from vertical is within limits
            # At output time t_out, depth z = v * t_out / 2
            z_at_t = t_out_ms * velocity / 2000.0  # depth in meters
            angle_from_vertical = torch.atan2(h.unsqueeze(0), z_at_t + 1e-6)
            angle_mask = angle_from_vertical < max_angle_rad

            # Combined mask
            full_mask = valid_time & aperture_mask & angle_mask

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['aperture_mask'] += time.time() - t0
                t0 = time.time()

            # Get input amplitudes for this batch
            # traces_f: (n_samples, n_filtered)
            amp_in = traces_f[sample_start:sample_end, :]  # (n_sample_batch, n_filtered)

            # Expand for broadcasting: (n_sample_batch, 1, n_filtered)
            amp_in_expanded = amp_in.unsqueeze(1).expand(-1, n_tile, -1)

            # Weight: obliquity factor (cos/r²)
            # For time-domain, use simplified weight based on output time
            # weight = cos(angle) / r² ≈ z / r³ = z / (z² + h²)^1.5
            # For constant velocity: z = v*t/2, r = v*t/2 / cos(angle)
            r_sq = z_at_t**2 + h_sq.unsqueeze(0)
            r = torch.sqrt(r_sq)
            weight = z_at_t / (r * r_sq + 1e-10)  # (n_sample_batch, n_tile, n_filtered)

            # Weighted amplitude
            weighted_amp = amp_in_expanded * weight * full_mask.float()

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['interpolation'] += time.time() - t0
                t0 = time.time()

            # Accumulate to output using scatter_add
            # Need to map (sample_batch, tile, filtered) -> output indices
            out_idx_floor = out_sample_idx.long().clamp(0, n_times - 1)

            # For each valid contribution, add to output
            # Use loop over sample batch (can't easily vectorize scatter across multiple dims)
            for s_idx in range(n_sample_batch):
                out_t_idx = out_idx_floor[s_idx]  # (n_tile, n_filtered)
                w_amp = weighted_amp[s_idx]  # (n_tile, n_filtered)
                mask = full_mask[s_idx]  # (n_tile, n_filtered)

                # Sum over filtered traces for each output point
                for t_idx in range(n_tile):
                    # Get unique output time indices and aggregate
                    t_indices = out_t_idx[t_idx]  # (n_filtered,)
                    amplitudes = w_amp[t_idx]  # (n_filtered,)
                    masks = mask[t_idx]  # (n_filtered,)

                    # Use scatter_add for efficiency
                    il_idx = tile_il[t_idx]
                    xl_idx = tile_xl[t_idx]

                    # Scatter add weighted amplitudes
                    image[:, il_idx, xl_idx].scatter_add_(
                        0, t_indices, amplitudes
                    )
                    fold[:, il_idx, xl_idx].scatter_add_(
                        0, t_indices, masks.float()
                    )

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['accumulation'] += time.time() - t0
                profile_counts['total_trace_point_pairs'] += n_sample_batch * n_tile * n_filtered
                profile_counts['aperture_passed'] += int(full_mask.sum().item())

        if enable_profiling:
            profile_counts['tiles_processed'] += 1

        # Progress reporting
        if progress_callback is not None:
            current_time = time.time()
            if current_time - last_progress_time >= 2.0 or (tile_idx + 1) % max(1, n_tiles // 10) == 0:
                last_progress_time = current_time
                pct = (tile_idx + 1) / n_tiles
                elapsed = current_time - start_time
                if pct > 0:
                    eta = elapsed / pct * (1 - pct)
                    rate = (tile_idx + 1) * tile_size * n_times / elapsed
                    progress_callback(
                        pct,
                        f"Tile {tile_idx+1}/{n_tiles} ({pct*100:.1f}%), "
                        f"{rate:.0f} output-samples/s, ETA: {eta:.0f}s"
                    )

    # Log profiling results
    if enable_profiling:
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("TIME-DOMAIN MIGRATION PROFILING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("")
        logger.info("TIME BREAKDOWN:")
        logger.info("-" * 40)

        sorted_times = sorted(profile_times.items(), key=lambda x: x[1], reverse=True)
        for name, t in sorted_times:
            pct = (t / total_time * 100) if total_time > 0 else 0
            logger.info(f"  {name:20s}: {t:8.2f}s ({pct:5.1f}%)")

        logger.info("")
        logger.info("OPERATION COUNTS:")
        logger.info("-" * 40)
        logger.info(f"  Tiles processed:        {profile_counts['tiles_processed']:,}")
        logger.info(f"  Total trace-point pairs: {profile_counts['total_trace_point_pairs']:,}")
        logger.info(f"  Aperture passed:        {profile_counts['aperture_passed']:,}")

        if profile_counts['total_trace_point_pairs'] > 0:
            aperture_ratio = profile_counts['aperture_passed'] / profile_counts['total_trace_point_pairs'] * 100
            logger.info(f"  Aperture pass rate:     {aperture_ratio:.1f}%")

        if total_time > 0:
            logger.info("")
            logger.info("DERIVED METRICS:")
            logger.info("-" * 40)
            ops_per_sec = profile_counts['total_trace_point_pairs'] / total_time
            logger.info(f"  Trace-point pairs/sec: {ops_per_sec:,.0f}")

        logger.info("=" * 60)

    return image, fold


# Compiled kernel for time-domain batch processing
# torch.compile provides JIT compilation for significant speedup on MPS/CUDA
def _time_domain_batch_kernel(
    t_in_sq: torch.Tensor,          # (n_sample_batch,)
    v_rms_sq: torch.Tensor,         # (n_sample_batch, 1, 1)
    h_eff_sq_expanded: torch.Tensor,  # (1, n_tile, n_filtered)
    h_expanded: torch.Tensor,       # (1, n_tile, n_filtered)
    h_sq_expanded: torch.Tensor,    # (1, n_tile, n_filtered)
    v_rms_out: torch.Tensor,        # (n_sample_batch, n_tile, n_filtered)
    amp_in_expanded: torch.Tensor,  # (n_sample_batch, n_tile, n_filtered)
    t_min_ms: float,
    dt_ms: float,
    n_times: int,
    max_angle_rad: float,
    tan_max_angle: float,
    max_aperture_m: float,
    use_time_dependent_aperture: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT-compiled kernel for time-domain migration batch processing.

    Returns:
        out_idx_floor: Output time indices
        weighted_amp: Weighted amplitudes
        full_mask: Valid contribution mask
        n_sample_batch: Batch size (for fold counting)
    """
    # Time shift: 4 * h_eff² / v_rms² (in ms²)
    time_shift_sq = 4.0 * h_eff_sq_expanded * 1e6 / v_rms_sq

    # Output time: t_out² = t_in² + time_shift²
    t_out_sq = t_in_sq.view(-1, 1, 1) + time_shift_sq
    t_out_ms = torch.sqrt(t_out_sq)

    # Output sample index
    out_sample_idx = (t_out_ms - t_min_ms) / dt_ms

    # Valid output range
    valid_time = (out_sample_idx >= 0) & (out_sample_idx < n_times - 0.001)

    # Aperture mask
    if use_time_dependent_aperture:
        aperture_at_t = t_out_ms * v_rms_out / 2000.0 * tan_max_angle
        aperture_at_t = torch.clamp(aperture_at_t, max=max_aperture_m)
        aperture_mask = h_expanded < aperture_at_t
    else:
        aperture_mask = h_expanded < max_aperture_m

    # Depth at output time (for angle calculation)
    z_at_t = v_rms_out * t_out_ms / 2000.0

    # Angle mask
    angle_from_vertical = torch.atan2(h_expanded, z_at_t + 1e-6)
    angle_mask = angle_from_vertical < max_angle_rad

    # Combined mask
    full_mask = valid_time & aperture_mask & angle_mask

    # Weight: cos(angle) / r² (obliquity and spreading correction)
    r_sq = z_at_t**2 + h_sq_expanded
    r = torch.sqrt(r_sq)
    weight = z_at_t / (r * r_sq + 1e-10)

    # Weighted amplitude
    weighted_amp = amp_in_expanded * weight * full_mask.float()

    # Output index (clamped for scatter_add)
    out_idx_floor = out_sample_idx.long().clamp(0, n_times - 1)

    return out_idx_floor, weighted_amp, full_mask


# Try to compile the kernel if torch.compile is available (PyTorch 2.0+)
# NOTE: torch.compile with MPS backend has float64 issues, so we disable it for MPS
# and only use compilation for CUDA devices
def _get_compiled_kernel():
    """Get compiled or uncompiled kernel based on device availability."""
    # Check if CUDA is available (torch.compile works well with CUDA)
    if torch.cuda.is_available():
        try:
            compiled = torch.compile(
                _time_domain_batch_kernel,
                mode="reduce-overhead",
                fullgraph=False,
            )
            logger.info("torch.compile available (CUDA) - using JIT-compiled kernel")
            return compiled, True
        except Exception as e:
            logger.info(f"torch.compile failed ({e}) - using standard kernel")
            return _time_domain_batch_kernel, False
    else:
        # MPS or CPU - use uncompiled version (MPS has float64 issues with torch.compile)
        logger.info("Using standard kernel (torch.compile disabled for MPS/CPU)")
        return _time_domain_batch_kernel, False

_compiled_time_domain_batch_kernel, TORCH_COMPILE_AVAILABLE = _get_compiled_kernel()


def migrate_kirchhoff_time_domain_rms(
    traces: torch.Tensor,
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    output_x: torch.Tensor,
    output_y: torch.Tensor,
    time_axis_ms: torch.Tensor,
    velocity_model: 'VelocityModel',
    dt_ms: float,
    t_min_ms: float,
    max_aperture_m: float,
    max_angle_deg: float,
    n_il: int,
    n_xl: int,
    origin_x: float,
    origin_y: float,
    il_spacing: float,
    xl_spacing: float,
    azimuth_deg: float,
    tile_size: int = 100,
    sample_batch_size: int = 200,
    max_traces_per_tile: int = 5000,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    enable_profiling: bool = False,
    use_time_dependent_aperture: bool = True,
    use_half_precision: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Time-domain Kirchhoff migration with RMS velocity support.

    This version supports non-constant velocity by using RMS velocity
    computed from the velocity model. Works with:
    - Constant velocity: v_rms = v0
    - Linear gradient: v_rms computed analytically
    - From file: v_rms computed numerically

    The time mapping equation becomes:
        t_out = sqrt(t_in² + 4*h_eff²/v_rms(t_out)²)

    Since v_rms depends on t_out, we iterate to find the solution,
    or use v_rms at the input time as an approximation.

    Args:
        traces: Input traces (n_samples, n_traces)
        source_x, source_y: Source coordinates (n_traces,)
        receiver_x, receiver_y: Receiver coordinates (n_traces,)
        output_x, output_y: Output grid coordinates (n_il * n_xl,)
        time_axis_ms: Output time values in ms (n_times,)
        velocity_model: VelocityModel instance with RMS velocity support
        dt_ms: Sample interval in ms
        t_min_ms: Start time in ms
        max_aperture_m: Maximum aperture in meters
        max_angle_deg: Maximum angle from vertical
        n_il, n_xl: Output grid dimensions
        tile_size: Number of output points per tile (default 100, try 200-500 for more memory)
        sample_batch_size: Input samples per batch (default 200, try 100-400)
        max_traces_per_tile: Maximum traces per tile (default 5000, increase for more memory)
        progress_callback: Optional callback(percent, message)
        enable_profiling: If True, collect detailed timing statistics
        use_time_dependent_aperture: If True, use aperture(t) = v*t/2*tan(angle)
        use_half_precision: If True, use float16 for computation (2x faster on MPS)

    Returns:
        image: (n_times, n_il, n_xl)
        fold: (n_times, n_il, n_xl)
    """
    import time as time_module
    from processors.migration.velocity_model import VelocityModel

    device = traces.device
    n_samples, n_traces = traces.shape
    n_times = time_axis_ms.shape[0]
    n_output = n_il * n_xl

    # NOTE: Float16 causes overflow for seismic data!
    # Values like t_ms²=4e6, v²=4e6, h²=1e6 exceed float16 max of 65504
    # Always use float32 for numerical stability
    compute_dtype = torch.float32
    # Float16 disabled due to overflow - the code is kept for reference
    # if use_half_precision and device.type == 'mps':
    #     compute_dtype = torch.float16
    #     logger.info("  Using float16 precision for MPS acceleration")

    # Initialize output (always float32 for accumulation precision)
    image = torch.zeros(n_times, n_il, n_xl, device=device, dtype=torch.float32)
    fold = torch.zeros(n_times, n_il, n_xl, device=device, dtype=torch.float32)

    # Pre-compute constants
    max_angle_rad = np.radians(max_angle_deg)
    tan_max_angle = np.tan(max_angle_rad)
    v0 = velocity_model.v0

    # For time-dependent aperture with RMS velocity
    # aperture(t) = v_rms(t) * t / 2 * tan(angle)
    # We'll compute this per output time

    # Compute trace midpoints and half-offsets
    mid_x = (source_x + receiver_x) / 2.0
    mid_y = (source_y + receiver_y) / 2.0
    half_offset_x = (receiver_x - source_x) / 2.0
    half_offset_y = (receiver_y - source_y) / 2.0
    half_offset_sq = half_offset_x**2 + half_offset_y**2  # (n_traces,)

    # CRITICAL: Compute trace midpoint grid indices for scatter
    # Each trace scatters to its OWN midpoint location, not the tile's output points
    cos_az = np.cos(np.radians(azimuth_deg))
    sin_az = np.sin(np.radians(azimuth_deg))

    # Transform midpoint to grid coordinates
    dx_from_origin = mid_x - origin_x
    dy_from_origin = mid_y - origin_y

    # Rotate by azimuth to get inline/crossline offsets
    il_offset = dx_from_origin * cos_az + dy_from_origin * sin_az
    xl_offset = -dx_from_origin * sin_az + dy_from_origin * cos_az

    # Convert to grid indices (can be fractional, will clamp to valid range)
    trace_il_float = il_offset / il_spacing
    trace_xl_float = xl_offset / xl_spacing

    # Clamp to valid grid range and convert to integer indices
    trace_il_all = trace_il_float.long().clamp(0, n_il - 1)  # (n_traces,)
    trace_xl_all = trace_xl_float.long().clamp(0, n_xl - 1)  # (n_traces,)

    # Debug: show coordinate transformation details
    logger.info(f"  Grid params: origin=({origin_x:.1f}, {origin_y:.1f}), spacing=({il_spacing:.1f}, {xl_spacing:.1f}), azimuth={azimuth_deg:.1f}°")
    logger.info(f"  Midpoint range: x=[{float(mid_x.min()):.1f}, {float(mid_x.max()):.1f}], y=[{float(mid_y.min()):.1f}, {float(mid_y.max()):.1f}]")
    logger.info(f"  IL float idx: [{float(trace_il_float.min()):.1f}, {float(trace_il_float.max()):.1f}] (grid 0-{n_il-1})")
    logger.info(f"  XL float idx: [{float(trace_xl_float.min()):.1f}, {float(trace_xl_float.max()):.1f}] (grid 0-{n_xl-1})")

    # Check for traces outside grid - this is a WARNING condition
    traces_in_il = ((trace_il_float >= 0) & (trace_il_float < n_il)).sum().item()
    traces_in_xl = ((trace_xl_float >= 0) & (trace_xl_float < n_xl)).sum().item()
    traces_in_grid = ((trace_il_float >= 0) & (trace_il_float < n_il) &
                      (trace_xl_float >= 0) & (trace_xl_float < n_xl)).sum().item()

    if traces_in_grid < n_traces:
        pct_outside = 100.0 * (1 - traces_in_grid / n_traces)
        logger.warning(f"  WARNING: {pct_outside:.1f}% of traces ({n_traces - traces_in_grid}/{n_traces}) fall OUTSIDE the output grid!")
        logger.warning(f"    Traces in IL range: {traces_in_il}, in XL range: {traces_in_xl}, in both: {traces_in_grid}")
        logger.warning(f"    Check that output grid origin/extent matches the input trace geometry")
        logger.warning(f"    Grid covers IL=[0, {n_il-1}], XL=[0, {n_xl-1}] but traces map to IL=[{float(trace_il_float.min()):.0f}, {float(trace_il_float.max()):.0f}], XL=[{float(trace_xl_float.min()):.0f}, {float(trace_xl_float.max()):.0f}]")
    else:
        logger.info(f"  All {n_traces} traces map within output grid")

    logger.info(f"  Trace midpoint grid (clamped): IL [{int(trace_il_all.min())}, {int(trace_il_all.max())}], "
                f"XL [{int(trace_xl_all.min())}, {int(trace_xl_all.max())}]")

    # Memory-aware tile size adjustment for RMS kernel
    # This kernel creates many large tensors of shape (n_sample_batch, n_tile, n_filtered)
    # With many traces, reduce tile size to avoid OOM
    # Target: n_tile * max_traces_per_tile * sample_batch_size * 10 tensors * 4 bytes < target_memory
    # Use the user-provided parameters, don't hardcode limits
    target_memory_gb = 12.0  # 12GB target for 48GB systems, adjust down if OOM
    max_tensor_elements = int(target_memory_gb * 1024**3 // (4 * 10))  # bytes / (4 bytes * 10 tensors)
    # Max tile = target_elements / (traces * sample_batch)
    max_tile_for_memory = max(50, max_tensor_elements // max(1, max_traces_per_tile * sample_batch_size))
    effective_tile_size = min(tile_size, max_tile_for_memory)

    # Compute INPUT time axis from trace parameters
    # Input traces may have different dt and time range than output grid
    # We assume input traces start at t=0 with the same dt as specified (this is standard)
    input_dt_ms = dt_ms  # TODO: could be passed separately if input dt differs from output dt
    input_t_min_ms = 0.0  # Input traces typically start at t=0
    input_time_axis_ms = input_t_min_ms + torch.arange(n_samples, device=device, dtype=torch.float32) * input_dt_ms

    # PRE-COMPUTE v_rms for all INPUT times ONCE (critical performance optimization!)
    # We iterate over input samples and map them to output times
    # The approximation v_rms(t_out) ≈ v_rms(t_in) is good for mild gradients.
    v_rms_all = velocity_model.rms_velocity_at_time(input_time_axis_ms)  # (n_samples,)
    if not isinstance(v_rms_all, torch.Tensor):
        v_rms_all = torch.tensor(v_rms_all, device=device, dtype=torch.float32)
    else:
        v_rms_all = v_rms_all.to(device=device, dtype=torch.float32)
    logger.info(f"  Pre-computed v_rms: {float(v_rms_all[0]):.1f} - {float(v_rms_all[-1]):.1f} m/s "
                f"over {float(input_time_axis_ms[0]):.0f} - {float(input_time_axis_ms[-1]):.0f} ms (input times)")

    # Process in tiles
    n_tiles = (n_output + effective_tile_size - 1) // effective_tile_size

    start_time = time_module.time()
    last_progress_time = start_time

    logger.info(f"Time-domain kernel (RMS): n_traces={n_traces}, n_times={n_times}, "
                f"n_output={n_output}, tile_size={effective_tile_size} (requested {tile_size}), n_tiles={n_tiles}")
    logger.info(f"Velocity model: {velocity_model.get_summary()}")
    logger.info(f"Aperture: max={max_aperture_m}m, max_angle={max_angle_deg}°, "
                f"time_dependent={use_time_dependent_aperture}")
    logger.info(f"  Params: sample_batch={sample_batch_size}, max_traces/tile={max_traces_per_tile}")
    logger.info("Starting time-domain migration...")

    # Immediate first progress callback
    if progress_callback is not None:
        progress_callback(0.0, f"Starting migration: 0/{n_tiles} tiles")

    # Profiling
    if enable_profiling:
        profile_times = {
            'trace_filtering': 0.0,
            'rms_velocity': 0.0,
            'time_mapping': 0.0,
            'aperture_mask': 0.0,
            'interpolation': 0.0,
            'accumulation': 0.0,
        }
        profile_counts = {
            'tiles_processed': 0,
            'total_trace_point_pairs': 0,
            'aperture_passed': 0,
        }

    # Disable gradient tracking for inference - reduces overhead significantly
    with torch.no_grad():
        for tile_idx, out_start in enumerate(range(0, n_output, effective_tile_size)):
            out_end = min(out_start + effective_tile_size, n_output)
            n_tile = out_end - out_start

            # Get output coordinates for this tile
            tile_x = output_x[out_start:out_end]
            tile_y = output_y[out_start:out_end]

            # Compute inline/crossline indices
            tile_il = torch.arange(out_start, out_end, device=device) // n_xl
            tile_xl = torch.arange(out_start, out_end, device=device) % n_xl

            if enable_profiling:
                t0 = time_module.time()

            # Pre-filter traces by distance to tile center
            tile_cx = (tile_x.min() + tile_x.max()) / 2
            tile_cy = (tile_y.min() + tile_y.max()) / 2
            tile_radius = torch.sqrt(((tile_x - tile_cx)**2 + (tile_y - tile_cy)**2).max())

            dist_to_center = torch.sqrt((mid_x - tile_cx)**2 + (mid_y - tile_cy)**2)
            trace_filter = dist_to_center < (max_aperture_m + tile_radius + 100)

            filtered_indices = torch.where(trace_filter)[0]
            n_filtered = len(filtered_indices)

            if n_filtered == 0:
                continue

            # Memory limit: cap filtered traces to avoid OOM
            # max_traces_per_tile is passed as parameter (default 5000 for 48GB systems)
            # With n_tile=500, n_sample_batch=50, n_filtered=5000:
            # tensor size = 500*50*5000*4*10 tensors = 5GB (manageable for 48GB)
            if n_filtered > max_traces_per_tile:
                # Keep only the closest traces to tile center
                distances_filtered = dist_to_center[filtered_indices]
                sorted_idx = torch.argsort(distances_filtered)[:max_traces_per_tile]
                filtered_indices = filtered_indices[sorted_idx]
                n_filtered = max_traces_per_tile
                if tile_idx == 0:
                    logger.info(f"  Memory limit: capped traces to {n_filtered} closest per tile")

            # Extract filtered data
            mid_x_f = mid_x[filtered_indices]
            mid_y_f = mid_y[filtered_indices]
            half_offset_sq_f = half_offset_sq[filtered_indices]
            traces_f = traces[:, filtered_indices]

            # CRITICAL: Get the grid indices for filtered traces
            # These determine WHERE each trace scatters to (its midpoint location)
            trace_il_f = trace_il_all[filtered_indices]  # (n_filtered,)
            trace_xl_f = trace_xl_all[filtered_indices]  # (n_filtered,)

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['trace_filtering'] += time_module.time() - t0
                t0 = time_module.time()

            # Horizontal distance from each output point to each trace midpoint
            dx = tile_x.unsqueeze(1) - mid_x_f.unsqueeze(0)
            dy = tile_y.unsqueeze(1) - mid_y_f.unsqueeze(0)
            h_sq = dx**2 + dy**2
            h = torch.sqrt(h_sq)

            # Effective horizontal distance including half-offset
            h_eff_sq = h_sq + half_offset_sq_f.unsqueeze(0)  # (n_tile, n_filtered)

            # Use the sample_batch_size passed as parameter (user-configurable via wizard)
            # Memory usage: n_sample_batch * n_tile * n_filtered * bytes_per_element * ~10 tensors
            # For n_sample_batch=200, n_tile=300, n_filtered=5000: 200*300*5000*2*10 = 6GB (float16)
            # Adjust sample_batch_size down if it would exceed reasonable memory
            target_memory_gb = 12.0  # Target max working memory per batch (48GB system)
            bytes_per_element = 2 if compute_dtype == torch.float16 else 4  # float16 = 2 bytes
            tensors_per_batch = 10
            max_elements = int(target_memory_gb * 1024**3 / bytes_per_element / tensors_per_batch)

            # Compute max batch size that fits in memory
            max_batch_for_memory = max(10, max_elements // max(1, n_tile * n_filtered))
            # Iterate over INPUT samples (traces) not OUTPUT samples
            # We map input times -> output times in time-domain migration
            effective_sample_batch = min(sample_batch_size, max_batch_for_memory, n_samples)

            n_sample_batches = (n_samples + effective_sample_batch - 1) // effective_sample_batch

            # Log first tile details
            if tile_idx == 0:
                logger.info(f"  Tile 0: {n_filtered} traces within aperture, batch_size={effective_sample_batch} "
                            f"(requested {sample_batch_size}), n_sample_batches={n_sample_batches}")

            # Pre-compute expanded tensors once per tile for reuse in kernel
            # Convert to compute_dtype for potential float16 acceleration
            h_expanded = h.unsqueeze(0).to(compute_dtype)  # (1, n_tile, n_filtered)
            h_sq_expanded = h_sq.unsqueeze(0).to(compute_dtype)  # (1, n_tile, n_filtered)
            h_eff_sq_expanded = h_eff_sq.unsqueeze(0).to(compute_dtype)  # (1, n_tile, n_filtered)

            for sample_batch_idx, sample_start in enumerate(range(0, n_samples, effective_sample_batch)):
                sample_end = min(sample_start + effective_sample_batch, n_samples)
                n_sample_batch = sample_end - sample_start

                if enable_profiling:
                    t0 = time_module.time()

                # Input times for this batch (ms) - use INPUT time axis, not OUTPUT
                # sample_start/sample_end are indices into input traces
                t_in_ms = input_t_min_ms + torch.arange(sample_start, sample_end, device=device, dtype=compute_dtype) * input_dt_ms
                t_in_sq = t_in_ms ** 2  # (n_sample_batch,)

                # Use PRE-COMPUTED v_rms from the lookup table (critical optimization!)
                v_rms_in = v_rms_all[sample_start:sample_end].to(compute_dtype)  # (n_sample_batch,)
                v_rms_sq = (v_rms_in ** 2).view(-1, 1, 1)  # (n_sample_batch, 1, 1)
                v_rms_out = v_rms_in.view(-1, 1, 1).expand(n_sample_batch, n_tile, n_filtered)

                # Get input amplitudes and expand for batch processing
                amp_in = traces_f[sample_start:sample_end, :].to(compute_dtype)
                amp_in_expanded = amp_in.unsqueeze(1).expand(-1, n_tile, -1)

                if enable_profiling:
                    if device.type == 'mps':
                        torch.mps.synchronize()
                    profile_times['rms_velocity'] += time_module.time() - t0
                    t0 = time_module.time()

                # Use kernel for the heavy computation (time mapping, aperture/angle masking, weighting)
                out_idx_floor, weighted_amp, full_mask = _compiled_time_domain_batch_kernel(
                    t_in_sq,
                    v_rms_sq,
                    h_eff_sq_expanded,
                    h_expanded,
                    h_sq_expanded,
                    v_rms_out,
                    amp_in_expanded,
                    t_min_ms,
                    dt_ms,
                    n_times,
                    max_angle_rad,
                    tan_max_angle,
                    max_aperture_m,
                    use_time_dependent_aperture,
                )

                if enable_profiling:
                    if device.type == 'mps':
                        torch.mps.synchronize()
                    profile_times['time_mapping'] += time_module.time() - t0
                    t0 = time_module.time()

                # CRITICAL FIX: Use TRACE midpoint grid indices, NOT tile output indices!
                # Each trace scatters to its own midpoint location on the grid
                # trace_il_f, trace_xl_f are (n_filtered,) - expand to match (n_sample_batch, n_tile, n_filtered)
                trace_il_exp = trace_il_f.view(1, 1, n_filtered).expand(n_sample_batch, n_tile, n_filtered)
                trace_xl_exp = trace_xl_f.view(1, 1, n_filtered).expand(n_sample_batch, n_tile, n_filtered)

                # 3D linear index: t * (n_il * n_xl) + il * n_xl + xl
                # Note: out_idx_floor is (n_sample_batch, n_tile, n_filtered) - output TIME index
                # trace_il_exp, trace_xl_exp are (n_sample_batch, n_tile, n_filtered) - trace LOCATION index
                linear_idx = out_idx_floor * (n_il * n_xl) + trace_il_exp * n_xl + trace_xl_exp

                # Flatten everything - convert back to float32 for accumulation precision
                linear_idx_flat = linear_idx.reshape(-1)
                amp_flat = weighted_amp.reshape(-1).to(torch.float32)
                mask_flat = full_mask.reshape(-1).float()

                # Flatten output for scatter_add
                image_flat = image.reshape(-1)  # (n_times * n_il * n_xl,)
                fold_flat = fold.reshape(-1)

                # ONE scatter_add call for entire batch!
                image_flat.scatter_add_(0, linear_idx_flat, amp_flat)
                fold_flat.scatter_add_(0, linear_idx_flat, mask_flat)

                if enable_profiling:
                    if device.type == 'mps':
                        torch.mps.synchronize()
                    profile_times['accumulation'] += time_module.time() - t0
                    profile_counts['total_trace_point_pairs'] += n_sample_batch * n_tile * n_filtered
                    profile_counts['aperture_passed'] += int(full_mask.sum().item())

                # Progress reporting within sample batches (every 2 seconds)
                if progress_callback is not None:
                    current_time = time_module.time()
                    if current_time - last_progress_time >= 2.0:
                        last_progress_time = current_time
                        # Calculate overall progress: tiles + fraction of current tile
                        batch_frac = (sample_batch_idx + 1) / n_sample_batches
                        pct = (tile_idx + batch_frac) / n_tiles
                        elapsed = current_time - start_time
                        if pct > 0:
                            eta = elapsed / pct * (1 - pct)
                            progress_callback(
                                pct,
                                f"Tile {tile_idx+1}/{n_tiles} batch {sample_batch_idx+1}/{n_sample_batches} "
                                f"({pct*100:.1f}%), ETA: {eta:.0f}s"
                            )

            if enable_profiling:
                profile_counts['tiles_processed'] += 1

            # Progress reporting at tile completion
            if progress_callback is not None:
                current_time = time_module.time()
                if current_time - last_progress_time >= 2.0 or (tile_idx + 1) % max(1, n_tiles // 10) == 0:
                    last_progress_time = current_time
                    pct = (tile_idx + 1) / n_tiles
                    elapsed = current_time - start_time
                    if pct > 0:
                        eta = elapsed / pct * (1 - pct)
                        rate = (tile_idx + 1) * effective_tile_size * n_times / elapsed
                        progress_callback(
                            pct,
                            f"Tile {tile_idx+1}/{n_tiles} ({pct*100:.1f}%), "
                            f"{rate:.0f} output-samples/s, ETA: {eta:.0f}s"
                        )

            # Periodic GPU memory cleanup
            if tile_idx % 50 == 0 and device.type == 'mps':
                torch.mps.empty_cache()

    # Log profiling results
    if enable_profiling:
        total_time = time_module.time() - start_time
        logger.info("=" * 60)
        logger.info("TIME-DOMAIN MIGRATION (RMS) PROFILING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Velocity model: {velocity_model.get_summary()}")
        logger.info("")
        logger.info("TIME BREAKDOWN:")
        logger.info("-" * 40)

        sorted_times = sorted(profile_times.items(), key=lambda x: x[1], reverse=True)
        for name, t in sorted_times:
            pct_t = (t / total_time * 100) if total_time > 0 else 0
            logger.info(f"  {name:20s}: {t:8.2f}s ({pct_t:5.1f}%)")

        logger.info("")
        logger.info("OPERATION COUNTS:")
        logger.info("-" * 40)
        logger.info(f"  Tiles processed:        {profile_counts['tiles_processed']:,}")
        logger.info(f"  Total trace-point pairs: {profile_counts['total_trace_point_pairs']:,}")
        logger.info(f"  Aperture passed:        {profile_counts['aperture_passed']:,}")

        if profile_counts['total_trace_point_pairs'] > 0:
            aperture_ratio = profile_counts['aperture_passed'] / profile_counts['total_trace_point_pairs'] * 100
            logger.info(f"  Aperture pass rate:     {aperture_ratio:.1f}%")

        if total_time > 0:
            logger.info("")
            logger.info("DERIVED METRICS:")
            logger.info("-" * 40)
            ops_per_sec = profile_counts['total_trace_point_pairs'] / total_time
            logger.info(f"  Trace-point pairs/sec: {ops_per_sec:,.0f}")

        logger.info("=" * 60)

    return image, fold
