"""
Geometry Preprocessor for PSTM Migration Engine.

Precomputes all geometry-dependent values for efficient GPU migration:
- Output grid indices for each trace
- Traveltimes for all trace-depth combinations
- Weights and aperture masks

This replaces the per-tile computation in the old implementation with
a single vectorized precomputation step.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedGeometry:
    """
    Container for precomputed geometry tensors.

    All tensors are on the same device (CPU or GPU).
    """
    # Output grid mapping (n_traces,)
    output_il: torch.Tensor      # int32 - inline index for each trace
    output_xl: torch.Tensor      # int32 - crossline index for each trace
    valid_mask: torch.Tensor     # bool - True if trace maps to valid grid point

    # Image point coordinates (n_traces,) - for traveltime computation
    image_x: torch.Tensor        # float32 - x coord of output point
    image_y: torch.Tensor        # float32 - y coord of output point

    # Traveltimes (n_traces, n_depths)
    traveltimes_ms: torch.Tensor  # float32 - two-way traveltime in ms

    # Weights and masks (n_traces, n_depths)
    weights: torch.Tensor         # float32 - obliquity/spreading correction
    aperture_mask: torch.Tensor   # bool - within aperture

    @property
    def n_traces(self) -> int:
        return self.output_il.shape[0]

    @property
    def n_depths(self) -> int:
        return self.traveltimes_ms.shape[1]

    @property
    def device(self) -> torch.device:
        return self.output_il.device

    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        total = 0
        for tensor in [self.output_il, self.output_xl, self.valid_mask,
                       self.image_x, self.image_y, self.traveltimes_ms,
                       self.weights, self.aperture_mask]:
            total += tensor.numel() * tensor.element_size()
        return total


def compute_output_indices(
    midpoint_x: np.ndarray,
    midpoint_y: np.ndarray,
    origin_x: float,
    origin_y: float,
    il_spacing: float,
    xl_spacing: float,
    azimuth_deg: float,
    n_il: int,
    n_xl: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute output grid indices for each trace midpoint.

    Transforms midpoint coordinates to inline/crossline indices
    accounting for grid rotation (azimuth).

    Args:
        midpoint_x: X coordinates of midpoints (n_traces,)
        midpoint_y: Y coordinates of midpoints (n_traces,)
        origin_x: X coordinate of grid origin (il=0, xl=0)
        origin_y: Y coordinate of grid origin
        il_spacing: Inline spacing in meters
        xl_spacing: Crossline spacing in meters
        azimuth_deg: Grid azimuth (inline direction from north, clockwise)
        n_il: Number of inlines in output grid
        n_xl: Number of crosslines in output grid

    Returns:
        output_il: (n_traces,) int32 - inline index
        output_xl: (n_traces,) int32 - crossline index
        valid_mask: (n_traces,) bool - True if within grid
        image_x: (n_traces,) float32 - x coord of nearest grid point
        image_y: (n_traces,) float32 - y coord of nearest grid point
    """
    n_traces = len(midpoint_x)

    # Translate to origin
    dx = midpoint_x - origin_x
    dy = midpoint_y - origin_y

    # Rotate to grid coordinates
    # Azimuth is inline direction from north (clockwise positive)
    # Need to rotate (dx, dy) to (inline_coord, xline_coord)
    azimuth_rad = np.radians(azimuth_deg)
    cos_az = np.cos(azimuth_rad)
    sin_az = np.sin(azimuth_rad)

    # Rotation: inline is along azimuth direction
    # For azimuth=0 (north): inline = dy, xline = dx
    # For azimuth=90 (east): inline = dx, xline = -dy
    inline_coord = dx * sin_az + dy * cos_az
    xline_coord = dx * cos_az - dy * sin_az

    # Convert to indices (round to nearest)
    il_float = inline_coord / il_spacing
    xl_float = xline_coord / xl_spacing

    output_il = np.round(il_float).astype(np.int32)
    output_xl = np.round(xl_float).astype(np.int32)

    # Check validity
    valid_mask = (
        (output_il >= 0) & (output_il < n_il) &
        (output_xl >= 0) & (output_xl < n_xl)
    )

    # Clamp indices for invalid traces (will be masked anyway)
    output_il = np.clip(output_il, 0, n_il - 1)
    output_xl = np.clip(output_xl, 0, n_xl - 1)

    # Compute image point coordinates (center of assigned cell)
    # Reverse transform from grid indices to world coordinates
    image_inline_coord = output_il.astype(np.float32) * il_spacing
    image_xline_coord = output_xl.astype(np.float32) * xl_spacing

    # Rotate back to world coordinates
    image_x = origin_x + image_inline_coord * sin_az + image_xline_coord * cos_az
    image_y = origin_y + image_inline_coord * cos_az - image_xline_coord * sin_az

    return output_il, output_xl, valid_mask, image_x.astype(np.float32), image_y.astype(np.float32)


def compute_traveltimes(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    image_x: torch.Tensor,
    image_y: torch.Tensor,
    depth_axis_m: torch.Tensor,
    velocity_mps: float,
) -> torch.Tensor:
    """
    Compute two-way traveltimes for all trace-depth combinations.

    Uses straight-ray assumption with constant velocity.

    Formula (pre-stack):
        t = (r_src + r_rcv) / v
        where r_src = sqrt((sx-ix)^2 + (sy-iy)^2 + z^2)
              r_rcv = sqrt((rx-ix)^2 + (ry-iy)^2 + z^2)

    For zero-offset (sx=rx, sy=ry):
        t = 2 * sqrt(h^2 + z^2) / v
        where h = sqrt((mx-ix)^2 + (my-iy)^2)

    Args:
        source_x: Source X coordinates (n_traces,)
        source_y: Source Y coordinates (n_traces,)
        receiver_x: Receiver X coordinates (n_traces,)
        receiver_y: Receiver Y coordinates (n_traces,)
        image_x: Image point X coordinates (n_traces,)
        image_y: Image point Y coordinates (n_traces,)
        depth_axis_m: Depth values in meters (n_depths,)
        velocity_mps: Constant velocity in m/s

    Returns:
        traveltimes_ms: (n_traces, n_depths) float32 - two-way traveltime in ms
    """
    n_traces = source_x.shape[0]
    n_depths = depth_axis_m.shape[0]

    # Horizontal distances from image point to source and receiver
    # Shape: (n_traces,)
    h_src_sq = (source_x - image_x)**2 + (source_y - image_y)**2
    h_rcv_sq = (receiver_x - image_x)**2 + (receiver_y - image_y)**2

    # Expand for broadcasting with depths
    # h_sq: (n_traces, 1), z_sq: (1, n_depths)
    h_src_sq = h_src_sq.unsqueeze(1)  # (n_traces, 1)
    h_rcv_sq = h_rcv_sq.unsqueeze(1)  # (n_traces, 1)
    z_sq = (depth_axis_m ** 2).unsqueeze(0)  # (1, n_depths)

    # Ray distances
    r_src = torch.sqrt(h_src_sq + z_sq)  # (n_traces, n_depths)
    r_rcv = torch.sqrt(h_rcv_sq + z_sq)  # (n_traces, n_depths)

    # Two-way traveltime in seconds
    traveltime_s = (r_src + r_rcv) / velocity_mps

    # Convert to milliseconds
    traveltimes_ms = traveltime_s * 1000.0

    return traveltimes_ms


def compute_weights(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    image_x: torch.Tensor,
    image_y: torch.Tensor,
    depth_axis_m: torch.Tensor,
    max_aperture_m: float,
    max_angle_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute migration weights and aperture mask.

    Weight is based on geometric spreading (1/r) and obliquity.
    Aperture mask excludes traces outside maximum aperture or angle.

    Args:
        source_x: Source X coordinates (n_traces,)
        source_y: Source Y coordinates (n_traces,)
        receiver_x: Receiver X coordinates (n_traces,)
        receiver_y: Receiver Y coordinates (n_traces,)
        image_x: Image point X coordinates (n_traces,)
        image_y: Image point Y coordinates (n_traces,)
        depth_axis_m: Depth values in meters (n_depths,)
        max_aperture_m: Maximum aperture in meters
        max_angle_deg: Maximum angle from vertical in degrees

    Returns:
        weights: (n_traces, n_depths) float32 - spreading/obliquity correction
        aperture_mask: (n_traces, n_depths) bool - within aperture
    """
    n_traces = source_x.shape[0]
    n_depths = depth_axis_m.shape[0]

    # Horizontal distances
    h_src_sq = (source_x - image_x)**2 + (source_y - image_y)**2
    h_rcv_sq = (receiver_x - image_x)**2 + (receiver_y - image_y)**2
    h_src = torch.sqrt(h_src_sq)
    h_rcv = torch.sqrt(h_rcv_sq)

    # Expand for broadcasting
    h_src = h_src.unsqueeze(1)  # (n_traces, 1)
    h_rcv = h_rcv.unsqueeze(1)
    h_src_sq = h_src_sq.unsqueeze(1)
    h_rcv_sq = h_rcv_sq.unsqueeze(1)
    z = depth_axis_m.unsqueeze(0)  # (1, n_depths)
    z_sq = z ** 2

    # Ray distances
    r_src_sq = h_src_sq + z_sq
    r_rcv_sq = h_rcv_sq + z_sq
    r_src = torch.sqrt(r_src_sq)
    r_rcv = torch.sqrt(r_rcv_sq)

    # Angles from vertical
    # angle = atan2(h, z) - angle is 0 when directly below, 90 at horizontal
    max_angle_rad = np.radians(max_angle_deg)
    angle_src = torch.atan2(h_src, z.abs() + 1e-6)
    angle_rcv = torch.atan2(h_rcv, z.abs() + 1e-6)

    # Aperture mask: within max aperture and max angle
    aperture_mask = (
        (h_src.squeeze(1) < max_aperture_m).unsqueeze(1).expand(-1, n_depths) &
        (h_rcv.squeeze(1) < max_aperture_m).unsqueeze(1).expand(-1, n_depths) &
        (angle_src < max_angle_rad) &
        (angle_rcv < max_angle_rad)
    )

    # Weight: geometric spreading correction
    # w = 1 / (r_src * r_rcv) with obliquity factor
    # Obliquity: cos(theta) = z / r
    cos_src = z / (r_src + 1e-6)
    cos_rcv = z / (r_rcv + 1e-6)

    # Combined weight with spreading and obliquity
    weights = (cos_src * cos_rcv) / (r_src * r_rcv + 1e-6)

    # Zero weight outside aperture
    weights = weights * aperture_mask.float()

    return weights, aperture_mask


class GeometryPreprocessor:
    """
    Precomputes geometry for efficient GPU migration.

    Usage:
        preprocessor = GeometryPreprocessor()
        geometry = preprocessor.precompute(
            source_x, source_y, receiver_x, receiver_y,
            output_grid_params, velocity, device
        )
    """

    def __init__(self):
        pass

    def precompute(
        self,
        # Trace geometry (numpy arrays)
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        # Output grid parameters
        origin_x: float,
        origin_y: float,
        il_spacing: float,
        xl_spacing: float,
        azimuth_deg: float,
        n_il: int,
        n_xl: int,
        # Time/depth axis
        dt_ms: float,
        t_min_ms: float,
        n_samples: int,
        # Velocity
        velocity_mps: float,
        # Migration parameters
        max_aperture_m: float,
        max_angle_deg: float,
        # Device
        device: torch.device = None,
    ) -> PrecomputedGeometry:
        """
        Precompute all geometry-dependent values.

        Args:
            source_x: Source X coordinates (n_traces,)
            source_y: Source Y coordinates (n_traces,)
            receiver_x: Receiver X coordinates (n_traces,)
            receiver_y: Receiver Y coordinates (n_traces,)
            origin_x: Grid origin X
            origin_y: Grid origin Y
            il_spacing: Inline spacing (m)
            xl_spacing: Crossline spacing (m)
            azimuth_deg: Grid azimuth
            n_il: Number of inlines
            n_xl: Number of crosslines
            dt_ms: Sample interval (ms)
            t_min_ms: Start time (ms)
            n_samples: Number of time samples
            velocity_mps: Velocity (m/s)
            max_aperture_m: Maximum aperture (m)
            max_angle_deg: Maximum angle (degrees)
            device: Target device (CPU/GPU)

        Returns:
            PrecomputedGeometry with all tensors on device
        """
        if device is None:
            device = torch.device('cpu')

        n_traces = len(source_x)
        logger.info(f"Precomputing geometry for {n_traces} traces, {n_samples} depths")

        # Step 1: Compute midpoints and output indices (on CPU)
        midpoint_x = (source_x + receiver_x) / 2.0
        midpoint_y = (source_y + receiver_y) / 2.0

        output_il, output_xl, valid_mask, image_x, image_y = compute_output_indices(
            midpoint_x, midpoint_y,
            origin_x, origin_y,
            il_spacing, xl_spacing,
            azimuth_deg,
            n_il, n_xl
        )

        n_valid = valid_mask.sum()
        logger.info(f"  {n_valid}/{n_traces} traces map to valid grid points")

        # Step 2: Create depth axis from time axis
        # For constant velocity: z = v * t / 2 (two-way time)
        time_axis_ms = np.arange(n_samples) * dt_ms + t_min_ms
        depth_axis_m = velocity_mps * (time_axis_ms / 1000.0) / 2.0  # meters

        # Step 3: Transfer to GPU
        source_x_t = torch.from_numpy(source_x.astype(np.float32)).to(device)
        source_y_t = torch.from_numpy(source_y.astype(np.float32)).to(device)
        receiver_x_t = torch.from_numpy(receiver_x.astype(np.float32)).to(device)
        receiver_y_t = torch.from_numpy(receiver_y.astype(np.float32)).to(device)
        image_x_t = torch.from_numpy(image_x).to(device)
        image_y_t = torch.from_numpy(image_y).to(device)
        depth_axis_t = torch.from_numpy(depth_axis_m.astype(np.float32)).to(device)

        # Step 4: Compute traveltimes
        traveltimes_ms = compute_traveltimes(
            source_x_t, source_y_t,
            receiver_x_t, receiver_y_t,
            image_x_t, image_y_t,
            depth_axis_t,
            velocity_mps
        )

        # Step 5: Compute weights and aperture mask
        weights, aperture_mask = compute_weights(
            source_x_t, source_y_t,
            receiver_x_t, receiver_y_t,
            image_x_t, image_y_t,
            depth_axis_t,
            max_aperture_m,
            max_angle_deg
        )

        # Step 6: Transfer indices to GPU
        output_il_t = torch.from_numpy(output_il).to(device)
        output_xl_t = torch.from_numpy(output_xl).to(device)
        valid_mask_t = torch.from_numpy(valid_mask).to(device)

        geometry = PrecomputedGeometry(
            output_il=output_il_t,
            output_xl=output_xl_t,
            valid_mask=valid_mask_t,
            image_x=image_x_t,
            image_y=image_y_t,
            traveltimes_ms=traveltimes_ms,
            weights=weights,
            aperture_mask=aperture_mask,
        )

        mem_mb = geometry.memory_bytes() / (1024 * 1024)
        logger.info(f"  Precomputed geometry: {mem_mb:.1f} MB")

        return geometry

    def precompute_batched(
        self,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        origin_x: float,
        origin_y: float,
        il_spacing: float,
        xl_spacing: float,
        azimuth_deg: float,
        n_il: int,
        n_xl: int,
        dt_ms: float,
        t_min_ms: float,
        n_samples: int,
        velocity_mps: float,
        max_aperture_m: float,
        max_angle_deg: float,
        device: torch.device = None,
        batch_size: int = 10000,
    ) -> PrecomputedGeometry:
        """
        Precompute geometry in batches for large trace counts.

        Uses batched processing to limit memory usage.
        Returns single PrecomputedGeometry with all results concatenated.
        """
        n_traces = len(source_x)

        if n_traces <= batch_size:
            return self.precompute(
                source_x, source_y, receiver_x, receiver_y,
                origin_x, origin_y, il_spacing, xl_spacing, azimuth_deg,
                n_il, n_xl, dt_ms, t_min_ms, n_samples,
                velocity_mps, max_aperture_m, max_angle_deg, device
            )

        logger.info(f"Precomputing geometry in {(n_traces + batch_size - 1) // batch_size} batches")

        # Process in batches and concatenate
        all_results = []

        for start in range(0, n_traces, batch_size):
            end = min(start + batch_size, n_traces)

            batch_geom = self.precompute(
                source_x[start:end], source_y[start:end],
                receiver_x[start:end], receiver_y[start:end],
                origin_x, origin_y, il_spacing, xl_spacing, azimuth_deg,
                n_il, n_xl, dt_ms, t_min_ms, n_samples,
                velocity_mps, max_aperture_m, max_angle_deg, device
            )
            all_results.append(batch_geom)

        # Concatenate all batches
        return PrecomputedGeometry(
            output_il=torch.cat([g.output_il for g in all_results]),
            output_xl=torch.cat([g.output_xl for g in all_results]),
            valid_mask=torch.cat([g.valid_mask for g in all_results]),
            image_x=torch.cat([g.image_x for g in all_results]),
            image_y=torch.cat([g.image_y for g in all_results]),
            traveltimes_ms=torch.cat([g.traveltimes_ms for g in all_results]),
            weights=torch.cat([g.weights for g in all_results]),
            aperture_mask=torch.cat([g.aperture_mask for g in all_results]),
        )
