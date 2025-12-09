"""
Aperture Control for Kirchhoff Migration

Implements aperture masks for controlling which traces contribute
to each output image point:
- Distance aperture (maximum horizontal distance)
- Angle aperture (maximum angle from vertical)
- Offset mute (source-receiver offset limits)
- Taper functions for smooth aperture edges
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ApertureController:
    """
    Controls migration aperture for each image point.

    Computes which input traces contribute to each output point
    based on distance, angle, and offset criteria.
    """

    def __init__(
        self,
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        min_offset_m: float = 0.0,
        max_offset_m: float = 10000.0,
        taper_width: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize aperture controller.

        Args:
            max_aperture_m: Maximum horizontal distance from image point to
                           surface point (source or receiver)
            max_angle_deg: Maximum angle from vertical (degrees)
            min_offset_m: Minimum source-receiver offset to include
            max_offset_m: Maximum source-receiver offset to include
            taper_width: Fractional width of cosine taper (0-1)
            device: Torch device for GPU computation
        """
        self.max_aperture = max_aperture_m
        self.max_angle_rad = np.radians(max_angle_deg)
        self.min_offset = min_offset_m
        self.max_offset = max_offset_m
        self.taper_width = taper_width
        self.device = device or self._detect_device()

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def compute_aperture_mask(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        receiver_x: torch.Tensor,
        receiver_y: torch.Tensor,
        image_x: torch.Tensor,
        image_y: torch.Tensor,
        image_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined aperture mask for all image points.

        Args:
            source_x, source_y: Source coordinates (n_traces,)
            receiver_x, receiver_y: Receiver coordinates (n_traces,)
            image_x, image_y: Image point horizontal coordinates (n_x,) or (n_x, n_y)
            image_z: Image point vertical coordinates (n_z,)

        Returns:
            Aperture mask (n_z, n_traces, n_x) or (n_z, n_traces, n_x, n_y)
            Values 0-1 including taper
        """
        n_traces = len(source_x)
        n_z = len(image_z)

        # Flatten image coordinates if 2D grid
        if image_x.dim() == 2:
            n_x, n_y = image_x.shape
            image_x_flat = image_x.flatten()
            image_y_flat = image_y.flatten()
            n_xy = n_x * n_y
        else:
            n_xy = len(image_x)
            image_x_flat = image_x
            image_y_flat = image_y

        # Compute offset mask (independent of z)
        offset = torch.sqrt(
            (receiver_x - source_x)**2 + (receiver_y - source_y)**2
        )
        offset_mask = self._compute_offset_mask(offset)

        # Compute horizontal distances from traces to image points
        # (n_traces, n_xy)
        dx_src = image_x_flat.unsqueeze(0) - source_x.unsqueeze(1)
        dy_src = image_y_flat.unsqueeze(0) - source_y.unsqueeze(1)
        dx_rcv = image_x_flat.unsqueeze(0) - receiver_x.unsqueeze(1)
        dy_rcv = image_y_flat.unsqueeze(0) - receiver_y.unsqueeze(1)

        h_src = torch.sqrt(dx_src**2 + dy_src**2)  # (n_traces, n_xy)
        h_rcv = torch.sqrt(dx_rcv**2 + dy_rcv**2)  # (n_traces, n_xy)

        # Compute aperture mask for each depth
        # (n_z, n_traces, n_xy)
        mask = torch.zeros(n_z, n_traces, n_xy, device=self.device, dtype=torch.float32)

        for iz, z in enumerate(image_z):
            z_val = z.item() if z.dim() == 0 else z

            # Distance aperture
            dist_mask_src = self._compute_distance_taper(h_src, self.max_aperture)
            dist_mask_rcv = self._compute_distance_taper(h_rcv, self.max_aperture)
            dist_mask = dist_mask_src * dist_mask_rcv

            # Angle aperture
            angle_src = torch.atan2(h_src, torch.abs(torch.tensor(z_val, device=self.device)))
            angle_rcv = torch.atan2(h_rcv, torch.abs(torch.tensor(z_val, device=self.device)))

            angle_mask_src = self._compute_angle_taper(angle_src, self.max_angle_rad)
            angle_mask_rcv = self._compute_angle_taper(angle_rcv, self.max_angle_rad)
            angle_mask = angle_mask_src * angle_mask_rcv

            # Combined mask
            mask[iz, :, :] = dist_mask * angle_mask * offset_mask.unsqueeze(1)

        # Reshape if input was 2D grid
        if image_x.dim() == 2:
            mask = mask.view(n_z, n_traces, n_x, n_y)

        return mask

    def compute_simple_mask(
        self,
        h_source: torch.Tensor,
        h_receiver: torch.Tensor,
        z_depth: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute aperture mask from pre-computed distances.

        Simpler interface when distances are already available.

        Args:
            h_source: Horizontal distance source-to-image (n_traces,) or (n_z, n_traces)
            h_receiver: Horizontal distance receiver-to-image
            z_depth: Depth/time of image points (n_z,)
            offset: Source-receiver offset (n_traces,)

        Returns:
            Aperture mask with values 0-1
        """
        # Offset mask
        offset_mask = self._compute_offset_mask(offset)

        # Handle different dimensionality
        if h_source.dim() == 1:
            # Single depth, expand
            h_source = h_source.unsqueeze(0)
            h_receiver = h_receiver.unsqueeze(0)

        n_z = h_source.shape[0]
        n_traces = h_source.shape[1]

        # Distance masks
        dist_mask_src = self._compute_distance_taper(h_source, self.max_aperture)
        dist_mask_rcv = self._compute_distance_taper(h_receiver, self.max_aperture)
        dist_mask = dist_mask_src * dist_mask_rcv

        # Angle masks
        z_expanded = z_depth.view(-1, 1).expand(n_z, n_traces)
        angle_src = torch.atan2(h_source, torch.abs(z_expanded))
        angle_rcv = torch.atan2(h_receiver, torch.abs(z_expanded))

        angle_mask_src = self._compute_angle_taper(angle_src, self.max_angle_rad)
        angle_mask_rcv = self._compute_angle_taper(angle_rcv, self.max_angle_rad)
        angle_mask = angle_mask_src * angle_mask_rcv

        # Combine
        return dist_mask * angle_mask * offset_mask.unsqueeze(0)

    def _compute_offset_mask(self, offset: torch.Tensor) -> torch.Tensor:
        """Compute mask based on source-receiver offset."""
        # Binary mask for offset limits
        valid = (offset >= self.min_offset) & (offset <= self.max_offset)

        # Apply taper near edges
        taper_range = self.taper_width * (self.max_offset - self.min_offset)

        # Near offset taper
        near_taper_end = self.min_offset + taper_range
        in_near_taper = (offset >= self.min_offset) & (offset < near_taper_end)
        near_taper_val = torch.where(
            in_near_taper,
            0.5 * (1 - torch.cos(np.pi * (offset - self.min_offset) / taper_range)),
            torch.ones_like(offset)
        )

        # Far offset taper
        far_taper_start = self.max_offset - taper_range
        in_far_taper = (offset > far_taper_start) & (offset <= self.max_offset)
        far_taper_val = torch.where(
            in_far_taper,
            0.5 * (1 + torch.cos(np.pi * (offset - far_taper_start) / taper_range)),
            torch.ones_like(offset)
        )

        # Combine
        mask = torch.where(valid, near_taper_val * far_taper_val, torch.zeros_like(offset))

        return mask

    def _compute_distance_taper(
        self,
        distance: torch.Tensor,
        max_distance: float,
    ) -> torch.Tensor:
        """Compute cosine taper based on distance."""
        taper_start = max_distance * (1.0 - self.taper_width)

        mask = torch.ones_like(distance)

        # In taper zone
        in_taper = (distance > taper_start) & (distance <= max_distance)
        taper_val = 0.5 * (1 + torch.cos(
            np.pi * (distance - taper_start) / (max_distance - taper_start)
        ))
        mask = torch.where(in_taper, taper_val, mask)

        # Outside aperture
        mask = torch.where(distance > max_distance, torch.zeros_like(mask), mask)

        return mask

    def _compute_angle_taper(
        self,
        angle: torch.Tensor,
        max_angle: float,
    ) -> torch.Tensor:
        """Compute cosine taper based on angle."""
        taper_start = max_angle * (1.0 - self.taper_width)

        mask = torch.ones_like(angle)

        # In taper zone
        in_taper = (angle > taper_start) & (angle <= max_angle)
        taper_val = 0.5 * (1 + torch.cos(
            np.pi * (angle - taper_start) / (max_angle - taper_start)
        ))
        mask = torch.where(in_taper, taper_val, mask)

        # Outside aperture
        mask = torch.where(angle > max_angle, torch.zeros_like(mask), mask)

        return mask


def compute_aperture_indices(
    source_x: np.ndarray,
    source_y: np.ndarray,
    receiver_x: np.ndarray,
    receiver_y: np.ndarray,
    image_x: float,
    image_y: float,
    image_z: float,
    max_aperture: float,
    max_angle_deg: float,
    min_offset: float = 0.0,
    max_offset: float = 10000.0,
) -> np.ndarray:
    """
    Get indices of traces within aperture for a single image point.

    Useful for pre-filtering traces before GPU processing.

    Args:
        source_x, source_y: Source coordinates (n_traces,)
        receiver_x, receiver_y: Receiver coordinates (n_traces,)
        image_x, image_y, image_z: Image point coordinates
        max_aperture: Maximum horizontal distance
        max_angle_deg: Maximum angle from vertical
        min_offset, max_offset: Offset range

    Returns:
        Array of trace indices within aperture
    """
    max_angle_rad = np.radians(max_angle_deg)

    # Horizontal distances
    h_src = np.sqrt((image_x - source_x)**2 + (image_y - source_y)**2)
    h_rcv = np.sqrt((image_x - receiver_x)**2 + (image_y - receiver_y)**2)

    # Angles
    angle_src = np.arctan2(h_src, np.abs(image_z))
    angle_rcv = np.arctan2(h_rcv, np.abs(image_z))

    # Offset
    offset = np.sqrt((receiver_x - source_x)**2 + (receiver_y - source_y)**2)

    # Apply criteria
    valid = (
        (h_src <= max_aperture) &
        (h_rcv <= max_aperture) &
        (angle_src <= max_angle_rad) &
        (angle_rcv <= max_angle_rad) &
        (offset >= min_offset) &
        (offset <= max_offset)
    )

    return np.where(valid)[0]


def estimate_contributing_traces(
    geometry,
    image_bounds: Tuple[float, float, float, float],
    z_range: Tuple[float, float],
    aperture_params: dict,
) -> int:
    """
    Estimate number of traces contributing to migration.

    Used for memory estimation.

    Args:
        geometry: MigrationGeometry object
        image_bounds: (x_min, x_max, y_min, y_max)
        z_range: (z_min, z_max)
        aperture_params: Dict with max_aperture, max_angle, etc.

    Returns:
        Estimated number of contributing trace-image pairs
    """
    x_min, x_max, y_min, y_max = image_bounds
    z_min, z_max = z_range

    # Image center
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    cz = (z_min + z_max) / 2

    # Count traces within aperture of image center
    indices = compute_aperture_indices(
        geometry.source_x,
        geometry.source_y,
        geometry.receiver_x,
        geometry.receiver_y,
        cx, cy, cz,
        aperture_params.get('max_aperture_m', 5000.0),
        aperture_params.get('max_angle_deg', 60.0),
        aperture_params.get('min_offset_m', 0.0),
        aperture_params.get('max_offset_m', 10000.0),
    )

    return len(indices)
