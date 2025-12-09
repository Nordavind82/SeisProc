"""
Kirchhoff Pre-Stack Time Migration - GPU Implementation

Implements isotropic straight-ray Kirchhoff PSTM with:
- Output-driven migration loop
- Batch gather processing
- Fold accumulation
- Memory chunking for large outputs
- Progress tracking
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
import logging
import time

from models.velocity_model import VelocityModel
from models.migration_config import MigrationConfig, WeightMode
from models.migration_geometry import MigrationGeometry
from models.seismic_data import SeismicData
from processors.migration.base_migrator import BaseMigrator, MigrationResult
from processors.migration.traveltime import (
    TraveltimeCalculator,
    StraightRayTraveltime,
    get_traveltime_calculator,
)
from processors.migration.weights import (
    AmplitudeWeight,
    StandardWeight,
    get_amplitude_weight,
)
from processors.migration.interpolation import interpolate_batch
from processors.migration.aperture import ApertureController

logger = logging.getLogger(__name__)


class KirchhoffMigrator(BaseMigrator):
    """
    GPU-accelerated Kirchhoff Pre-Stack Time Migration.

    Implements output-driven migration loop:
    For each output image point:
        For each input trace:
            1. Compute traveltime from source to image to receiver
            2. Check aperture constraints
            3. Interpolate trace at traveltime
            4. Apply amplitude weight
            5. Accumulate to output

    Supports:
    - Straight ray (constant velocity) traveltimes
    - Multiple amplitude weighting modes
    - Aperture control (distance, angle, offset)
    - Memory chunking for large outputs
    - Progress callbacks
    """

    def __init__(
        self,
        velocity: VelocityModel,
        config: MigrationConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Kirchhoff migrator.

        Args:
            velocity: Velocity model
            config: Migration configuration
            device: Torch device (None = auto-detect)
        """
        # Store velocity and device before calling super().__init__
        self.velocity = velocity
        self.device = device or self._detect_device()

        super().__init__(velocity, config)

        # Set up components after base init
        self._setup_components()

    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _validate_inputs(self):
        """Validate velocity model and configuration."""
        if self.velocity.v0 is None or self.velocity.v0 <= 0:
            raise ValueError("Velocity model must have positive v0")

        if self.config.output_grid.n_time <= 0:
            raise ValueError("Output grid must have positive n_time")

    def _setup_components(self):
        """Set up migration components after init."""
        # Initialize traveltime calculator
        self.traveltime_calc = get_traveltime_calculator(
            self.velocity,
            mode='straight' if self.config.traveltime_mode.value == 'straight_ray' else 'curved',
            device=self.device,
        )

        # Initialize amplitude weight calculator
        self.weight_calc = get_amplitude_weight(
            self.config.weight_mode,
            device=self.device,
        )

        # Initialize aperture controller
        self.aperture = ApertureController(
            max_aperture_m=self.config.max_aperture_m,
            max_angle_deg=self.config.max_angle_deg,
            min_offset_m=self.config.min_offset_m,
            max_offset_m=self.config.max_offset_m,
            taper_width=self.config.taper_width,
            device=self.device,
        )

        # Statistics
        self._stats = {}

        logger.info(f"Initialized KirchhoffMigrator on {self.device}")
        logger.info(f"  Traveltime: {self.traveltime_calc.get_description()}")
        logger.info(f"  Weights: {self.config.weight_mode.value}")
        logger.info(f"  Output grid: {self.config.output_grid.n_time} x "
                   f"{self.config.output_grid.n_inline} x {self.config.output_grid.n_xline}")

    def migrate_gather(
        self,
        gather: SeismicData,
        geometry: MigrationGeometry,
        output_image: Optional[torch.Tensor] = None,
        output_fold: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Migrate a single gather to the output image.

        Args:
            gather: Input seismic gather
            geometry: Geometry for this gather
            output_image: Existing output to accumulate into (optional)
            output_fold: Existing fold to accumulate into (optional)

        Returns:
            Tuple of (image, fold) tensors
        """
        grid = self.config.output_grid
        n_z = grid.n_time
        n_inline = grid.n_inline
        n_xline = grid.n_xline
        dt = grid.dt
        t0 = grid.t0

        # Initialize or validate output tensors
        if output_image is None:
            output_image = torch.zeros(
                n_z, n_inline, n_xline,
                device=self.device, dtype=torch.float32
            )
        if output_fold is None:
            output_fold = torch.zeros(
                n_z, n_inline, n_xline,
                device=self.device, dtype=torch.float32
            )

        # Convert input to tensors
        traces = torch.from_numpy(gather.traces).to(self.device)
        n_samples, n_traces = traces.shape

        src_x = torch.from_numpy(geometry.source_x).to(self.device)
        src_y = torch.from_numpy(geometry.source_y).to(self.device)
        rcv_x = torch.from_numpy(geometry.receiver_x).to(self.device)
        rcv_y = torch.from_numpy(geometry.receiver_y).to(self.device)
        offset = torch.from_numpy(geometry.offset).to(self.device)

        # Get output grid coordinates
        # For time migration, convert time axis to depth using velocity
        # z_depth = t_twt * v / 2 (two-way time to one-way depth)
        t_axis = torch.from_numpy(grid.time_axis.astype(np.float32)).to(self.device)
        z_axis = t_axis * self.velocity.v0 / 2.0  # Convert time to depth (meters)

        # Process output grid in chunks for memory efficiency
        chunk_size = self._get_optimal_chunk_size(n_traces, n_z)

        for il_start in range(0, n_inline, chunk_size):
            il_end = min(il_start + chunk_size, n_inline)

            for xl_start in range(0, n_xline, chunk_size):
                xl_end = min(xl_start + chunk_size, n_xline)

                # Get image point coordinates for this chunk
                il_chunk = torch.arange(il_start, il_end, device=self.device)
                xl_chunk = torch.arange(xl_start, xl_end, device=self.device)

                # Convert to world coordinates
                img_x, img_y = self._grid_to_world(il_chunk, xl_chunk)

                # Migrate this chunk
                img_chunk, fold_chunk = self._migrate_chunk(
                    traces, n_samples, gather.sample_rate / 1000.0,
                    src_x, src_y, rcv_x, rcv_y, offset,
                    img_x, img_y, z_axis,
                )

                # Accumulate to output
                output_image[:, il_start:il_end, xl_start:xl_end] += img_chunk
                output_fold[:, il_start:il_end, xl_start:xl_end] += fold_chunk

        return output_image, output_fold

    def _migrate_chunk(
        self,
        traces: torch.Tensor,
        n_samples: int,
        dt: float,
        src_x: torch.Tensor,
        src_y: torch.Tensor,
        rcv_x: torch.Tensor,
        rcv_y: torch.Tensor,
        offset: torch.Tensor,
        img_x: torch.Tensor,
        img_y: torch.Tensor,
        z_axis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Migrate a chunk of the output image - FULLY VECTORIZED with trace batching.

        Processes all image points in parallel, traces in batches to manage memory.

        Args:
            traces: Input traces (n_samples, n_traces)
            n_samples: Number of samples
            dt: Sample interval in seconds
            src_x, src_y: Source coordinates (n_traces,)
            rcv_x, rcv_y: Receiver coordinates (n_traces,)
            offset: Source-receiver offset (n_traces,)
            img_x: Image point X coordinates (n_il,)
            img_y: Image point Y coordinates (n_xl,)
            z_axis: Depth/time axis (n_z,)

        Returns:
            Tuple of (image_chunk, fold_chunk)
        """
        n_z = len(z_axis)
        n_il = len(img_x)
        n_xl = len(img_y)
        n_traces = len(src_x)

        # Initialize output
        image_chunk = torch.zeros(n_z, n_il, n_xl, device=self.device, dtype=torch.float32)
        fold_chunk = torch.zeros(n_z, n_il, n_xl, device=self.device, dtype=torch.float32)

        # Create meshgrid of image coordinates: (n_il, n_xl)
        img_xx, img_yy = torch.meshgrid(img_x, img_y, indexing='ij')
        img_x_flat = img_xx.flatten()  # (n_points,)
        img_y_flat = img_yy.flatten()
        n_points = n_il * n_xl

        # Process traces in batches to manage memory
        # Memory estimate: n_z * n_points * batch_size * 4 bytes * 5 tensors
        # For 1000 depths, 10K points, target 1GB -> batch_size ~ 50
        mem_per_trace = n_z * n_points * 4 * 5
        target_mem = 1e9  # 1GB
        trace_batch_size = max(10, min(n_traces, int(target_mem / mem_per_trace)))

        v = self.velocity.v0
        max_angle_rad = np.radians(self.config.max_angle_deg)
        max_aperture = self.config.max_aperture_m

        # Precompute z expansion
        z_expanded = z_axis.view(n_z, 1, 1)  # (n_z, 1, 1)

        for t_start in range(0, n_traces, trace_batch_size):
            t_end = min(t_start + trace_batch_size, n_traces)
            batch_size = t_end - t_start

            # Get batch of trace data and coordinates
            traces_batch = traces[:, t_start:t_end]  # (n_samples, batch)
            sx = src_x[t_start:t_end]
            sy = src_y[t_start:t_end]
            rx = rcv_x[t_start:t_end]
            ry = rcv_y[t_start:t_end]

            # Compute distances: (n_points, batch)
            dx_src = img_x_flat.unsqueeze(1) - sx.unsqueeze(0)
            dy_src = img_y_flat.unsqueeze(1) - sy.unsqueeze(0)
            dx_rcv = img_x_flat.unsqueeze(1) - rx.unsqueeze(0)
            dy_rcv = img_y_flat.unsqueeze(1) - ry.unsqueeze(0)

            h_src = torch.sqrt(dx_src**2 + dy_src**2)
            h_rcv = torch.sqrt(dx_rcv**2 + dy_rcv**2)

            # Expand for depth dimension: (n_z, n_points, batch)
            h_src_exp = h_src.unsqueeze(0)
            h_rcv_exp = h_rcv.unsqueeze(0)

            # Distances and traveltimes
            r_src = torch.sqrt(h_src_exp**2 + z_expanded**2)
            r_rcv = torch.sqrt(h_rcv_exp**2 + z_expanded**2)
            t_total = (r_src + r_rcv) / v

            # Aperture mask
            angle_src = torch.atan2(h_src_exp, z_expanded.abs() + 1e-6)
            angle_rcv = torch.atan2(h_rcv_exp, z_expanded.abs() + 1e-6)
            mask = (angle_src < max_angle_rad) & (angle_rcv < max_angle_rad)
            mask = mask & (h_src_exp < max_aperture) & (h_rcv_exp < max_aperture)
            aperture_mask = mask.float()

            # Weights: 1/r spreading
            weights = aperture_mask / (r_src * r_rcv + 1e-6)

            # Interpolate traces at traveltimes
            sample_idx = torch.clamp(t_total / dt, 0, n_samples - 2)
            idx_floor = sample_idx.long()
            frac = sample_idx - idx_floor.float()

            # Create batch indices for gather
            batch_idx = torch.arange(batch_size, device=self.device).view(1, 1, batch_size).expand(n_z, n_points, -1)
            idx_floor_clamped = torch.clamp(idx_floor, 0, n_samples - 1)
            idx_ceil_clamped = torch.clamp(idx_floor + 1, 0, n_samples - 1)

            # Flatten for indexing
            idx_f = idx_floor_clamped.reshape(-1)
            idx_c = idx_ceil_clamped.reshape(-1)
            b_idx = batch_idx.reshape(-1)

            # Gather amplitudes
            amp_floor = traces_batch[idx_f, b_idx].reshape(n_z, n_points, batch_size)
            amp_ceil = traces_batch[idx_c, b_idx].reshape(n_z, n_points, batch_size)
            amplitudes = amp_floor + frac * (amp_ceil - amp_floor)

            # Accumulate weighted sum
            image_chunk += torch.sum(amplitudes * weights, dim=2).reshape(n_z, n_il, n_xl)
            fold_chunk += torch.sum(aperture_mask, dim=2).reshape(n_z, n_il, n_xl)

        return image_chunk, fold_chunk

    def _grid_to_world(
        self,
        il_indices: torch.Tensor,
        xl_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert grid indices to world coordinates."""
        grid = self.config.output_grid

        x = grid.x_origin + il_indices.float() * grid.d_inline
        y = grid.y_origin + xl_indices.float() * grid.d_xline

        return x, y

    def _get_optimal_chunk_size(self, n_traces: int, n_z: int) -> int:
        """Determine optimal chunk size based on available memory."""
        # Memory per image point for vectorized computation:
        # - Distances: 4 tensors * n_points * n_traces * 4 bytes
        # - Traveltimes: n_z * n_points * n_traces * 4 bytes
        # - Weights/amplitudes: n_z * n_points * n_traces * 4 bytes * 3
        # Total: ~5 * n_z * n_points * n_traces * 4 bytes

        bytes_per_point = n_z * n_traces * 4 * 5  # 5 arrays, float32

        # Target memory usage (4GB for GPU, can be higher on Apple Silicon)
        if self.device.type == 'mps':
            target_bytes = 4e9  # 4GB for Metal
        elif self.device.type == 'cuda':
            target_bytes = 6e9  # 6GB for CUDA
        else:
            target_bytes = 2e9  # 2GB for CPU

        max_points = int(target_bytes / bytes_per_point)

        # Chunk size is sqrt of max points (for 2D grid)
        chunk_size = max(1, int(np.sqrt(max_points)))

        # For vectorized code, we can handle larger chunks
        # But limit to full grid if it fits
        chunk_size = min(chunk_size, 100)  # Max 100x100 = 10K points per chunk
        chunk_size = max(chunk_size, 10)   # Min 10x10 = 100 points

        logger.debug(f"Vectorized chunk size {chunk_size}x{chunk_size} for {n_traces} traces, {n_z} depths")
        return chunk_size

    def migrate_dataset(
        self,
        gathers: List[SeismicData],
        geometries: List[MigrationGeometry],
        progress_callback: Optional[Callable[[float, str], None]] = None,
        batch_size: int = 1,
    ) -> MigrationResult:
        """
        Migrate multiple gathers to produce stacked image.

        Args:
            gathers: List of input gathers
            geometries: List of geometries (one per gather)
            progress_callback: Function called with (progress, message)
            batch_size: Number of gathers to process per GPU call

        Returns:
            MigrationResult with stacked image and fold
        """
        start_time = time.time()
        n_gathers = len(gathers)

        grid = self.config.output_grid
        n_z = grid.n_time
        n_inline = grid.n_inline
        n_xline = grid.n_xline

        # Initialize outputs
        image = torch.zeros(n_z, n_inline, n_xline, device=self.device, dtype=torch.float32)
        fold = torch.zeros(n_z, n_inline, n_xline, device=self.device, dtype=torch.float32)

        logger.info(f"Starting migration of {n_gathers} gathers")

        # Process gathers
        for i, (gather, geometry) in enumerate(zip(gathers, geometries)):
            if progress_callback:
                progress = (i / n_gathers) * 100
                progress_callback(progress, f"Migrating gather {i+1}/{n_gathers}")

            # Migrate this gather
            image, fold = self.migrate_gather(gather, geometry, image, fold)

            if i % 10 == 0:
                logger.debug(f"Completed {i+1}/{n_gathers} gathers")

        # Normalize by fold if requested
        if self.config.normalize_by_fold:
            # Avoid division by zero
            fold_safe = torch.where(fold > 0, fold, torch.ones_like(fold))
            image = image / fold_safe
            # Zero where fold is below minimum
            image = torch.where(fold >= self.config.min_fold, image, torch.zeros_like(image))

        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback(100.0, "Migration complete")

        logger.info(f"Migration complete in {elapsed:.1f}s")

        # Build result
        return MigrationResult(
            image=image.cpu().numpy(),
            fold=fold.cpu().numpy(),
            config=self.config,
            metadata={
                'n_gathers': n_gathers,
                'elapsed_seconds': elapsed,
                'traces_per_second': sum(g.n_traces for g in gathers) / elapsed,
                'velocity_v0': self.velocity.v0,
            }
        )

    def estimate_memory_gb(
        self,
        n_traces: int,
        n_samples: int,
    ) -> float:
        """
        Estimate memory requirements.

        Args:
            n_traces: Number of input traces
            n_samples: Samples per trace

        Returns:
            Estimated memory in GB
        """
        grid = self.config.output_grid

        # Output image and fold
        output_size = grid.n_time * grid.n_inline * grid.n_xline * 4 * 2  # image + fold

        # Working memory per gather (rough estimate)
        work_per_gather = n_traces * grid.n_time * 4 * 4  # 4 working arrays

        # Total in GB
        total_bytes = output_size + work_per_gather
        return total_bytes / (1024**3)

    def get_description(self) -> str:
        """Get human-readable description of migrator and parameters."""
        return (
            f"Kirchhoff Pre-Stack Time Migration\n"
            f"  Device: {self.device}\n"
            f"  Traveltime: {self.config.traveltime_mode.value}\n"
            f"  Weights: {self.config.weight_mode.value}\n"
            f"  Aperture: {self.config.max_aperture_m}m, {self.config.max_angle_deg}°\n"
            f"  Output grid: {self.config.output_grid.n_time} x "
            f"{self.config.output_grid.n_inline} x {self.config.output_grid.n_xline}"
        )


def create_kirchhoff_migrator(
    velocity: VelocityModel,
    config: MigrationConfig,
    prefer_gpu: bool = True,
) -> KirchhoffMigrator:
    """
    Factory function to create Kirchhoff migrator.

    Args:
        velocity: Velocity model
        config: Migration configuration
        prefer_gpu: If True, use GPU if available

    Returns:
        KirchhoffMigrator instance
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            logger.warning("GPU not available, falling back to CPU")
    else:
        device = torch.device('cpu')

    return KirchhoffMigrator(velocity, config, device)
