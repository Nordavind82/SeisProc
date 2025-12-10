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
    tile_size: int = 100,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    enable_profiling: bool = False,
    use_time_dependent_aperture: bool = True,
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
        tile_size: Number of output points per tile
        progress_callback: Optional callback(percent, message)
        enable_profiling: If True, collect detailed timing statistics
        use_time_dependent_aperture: If True, use aperture(t) = v*t/2*tan(angle)

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

    # Initialize output
    image = torch.zeros(n_times, n_il, n_xl, device=device)
    fold = torch.zeros(n_times, n_il, n_xl, device=device)

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

    # Process in tiles
    n_tiles = (n_output + tile_size - 1) // tile_size

    start_time = time_module.time()
    last_progress_time = start_time

    logger.info(f"Time-domain kernel (RMS): n_traces={n_traces}, n_times={n_times}, "
                f"n_output={n_output}, tile_size={tile_size}, n_tiles={n_tiles}")
    logger.info(f"Velocity model: {velocity_model.get_summary()}")
    logger.info(f"Aperture: max={max_aperture_m}m, max_angle={max_angle_deg}°, "
                f"time_dependent={use_time_dependent_aperture}")

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

    for tile_idx, out_start in enumerate(range(0, n_output, tile_size)):
        out_end = min(out_start + tile_size, n_output)
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

        # Extract filtered data
        mid_x_f = mid_x[filtered_indices]
        mid_y_f = mid_y[filtered_indices]
        half_offset_sq_f = half_offset_sq[filtered_indices]
        traces_f = traces[:, filtered_indices]

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

        # Process input samples in batches
        sample_batch_size = min(200, n_samples)

        for sample_start in range(0, n_samples, sample_batch_size):
            sample_end = min(sample_start + sample_batch_size, n_samples)
            n_sample_batch = sample_end - sample_start

            if enable_profiling:
                t0 = time_module.time()

            # Input times for this batch (ms)
            t_in_ms = t_min_ms + torch.arange(sample_start, sample_end, device=device, dtype=torch.float32) * dt_ms
            t_in_sq = t_in_ms ** 2  # (n_sample_batch,)

            # Get RMS velocity at input times (as initial estimate)
            # Shape: (n_sample_batch,)
            v_rms_in = velocity_model.rms_velocity_at_time(t_in_ms)
            if not isinstance(v_rms_in, torch.Tensor):
                v_rms_in = torch.tensor(v_rms_in, device=device, dtype=torch.float32)

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['rms_velocity'] += time_module.time() - t0
                t0 = time_module.time()

            # Time shift: 4 * h_eff² / v_rms²
            # Shape: (n_sample_batch, n_tile, n_filtered)
            v_rms_sq = (v_rms_in ** 2).view(-1, 1, 1)  # (n_sample_batch, 1, 1)
            h_eff_sq_expanded = h_eff_sq.unsqueeze(0)  # (1, n_tile, n_filtered)

            # Convert h_eff_sq from m² to ms² using v_rms
            # time_shift² = 4 * h_eff² / v² in seconds² = 4 * h_eff² * 1e6 / v² in ms²
            time_shift_sq = 4.0 * h_eff_sq_expanded * 1e6 / v_rms_sq  # ms²

            # Output time: t_out² = t_in² + time_shift²
            t_out_sq = t_in_sq.view(-1, 1, 1) + time_shift_sq
            t_out_ms = torch.sqrt(t_out_sq)

            # Iterative refinement: update v_rms using output time
            # One iteration usually sufficient for mild gradients
            v_rms_out = None
            if velocity_model.velocity_type != 'constant':
                # Get v_rms at output time (more accurate)
                t_out_flat = t_out_ms.reshape(-1)
                v_rms_out_flat = velocity_model.rms_velocity_at_time(t_out_flat)
                if not isinstance(v_rms_out_flat, torch.Tensor):
                    v_rms_out_flat = torch.tensor(v_rms_out_flat, device=device, dtype=torch.float32)
                v_rms_out = v_rms_out_flat.reshape(n_sample_batch, n_tile, n_filtered)

                # Recompute with updated v_rms
                v_rms_sq_updated = v_rms_out ** 2
                time_shift_sq_updated = 4.0 * h_eff_sq_expanded * 1e6 / v_rms_sq_updated
                t_out_sq = t_in_sq.view(-1, 1, 1) + time_shift_sq_updated
                t_out_ms = torch.sqrt(t_out_sq)

            # Convert output time to output sample index
            out_sample_idx = (t_out_ms - t_min_ms) / dt_ms

            # Valid output range
            valid_time = (out_sample_idx >= 0) & (out_sample_idx < n_times - 0.001)

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['time_mapping'] += time_module.time() - t0
                t0 = time_module.time()

            # Aperture mask
            if use_time_dependent_aperture:
                # aperture(t) = v_rms(t) * t / 2 * tan(angle)
                # Use v_rms_out if available, else approximate with v0
                if v_rms_out is not None:
                    v_for_aperture = v_rms_out
                else:
                    v_for_aperture = v0
                aperture_at_t = t_out_ms * v_for_aperture / 2000.0 * tan_max_angle
                aperture_at_t = torch.clamp(aperture_at_t, max=max_aperture_m)
                aperture_mask = h.unsqueeze(0) < aperture_at_t
            else:
                aperture_mask = h.unsqueeze(0) < max_aperture_m

            # Angle mask
            # At output time t_out, depth z ≈ v_rms * t_out / 2
            if v_rms_out is not None:
                z_at_t = v_rms_out * t_out_ms / 2000.0
            else:
                z_at_t = v0 * t_out_ms / 2000.0
            angle_from_vertical = torch.atan2(h.unsqueeze(0), z_at_t + 1e-6)
            angle_mask = angle_from_vertical < max_angle_rad

            # Combined mask
            full_mask = valid_time & aperture_mask & angle_mask

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['aperture_mask'] += time_module.time() - t0
                t0 = time_module.time()

            # Get input amplitudes
            amp_in = traces_f[sample_start:sample_end, :]
            amp_in_expanded = amp_in.unsqueeze(1).expand(-1, n_tile, -1)

            # Weight: cos(angle) / r²
            r_sq = z_at_t**2 + h_sq.unsqueeze(0)
            r = torch.sqrt(r_sq)
            weight = z_at_t / (r * r_sq + 1e-10)

            weighted_amp = amp_in_expanded * weight * full_mask.float()

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['interpolation'] += time_module.time() - t0
                t0 = time_module.time()

            # Accumulate to output using scatter_add
            out_idx_floor = out_sample_idx.long().clamp(0, n_times - 1)

            for s_idx in range(n_sample_batch):
                out_t_idx = out_idx_floor[s_idx]
                w_amp = weighted_amp[s_idx]
                mask = full_mask[s_idx]

                for t_idx in range(n_tile):
                    t_indices = out_t_idx[t_idx]
                    amplitudes = w_amp[t_idx]
                    masks = mask[t_idx]

                    il_idx = tile_il[t_idx]
                    xl_idx = tile_xl[t_idx]

                    image[:, il_idx, xl_idx].scatter_add_(0, t_indices, amplitudes)
                    fold[:, il_idx, xl_idx].scatter_add_(0, t_indices, masks.float())

            if enable_profiling:
                if device.type == 'mps':
                    torch.mps.synchronize()
                profile_times['accumulation'] += time_module.time() - t0
                profile_counts['total_trace_point_pairs'] += n_sample_batch * n_tile * n_filtered
                profile_counts['aperture_passed'] += int(full_mask.sum().item())

        if enable_profiling:
            profile_counts['tiles_processed'] += 1

        # Progress reporting
        if progress_callback is not None:
            current_time = time_module.time()
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
