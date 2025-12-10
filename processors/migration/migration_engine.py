"""
Migration Engine - Main orchestrator for PSTM.

Combines GeometryPreprocessor and KirchhoffKernel to perform
efficient GPU-accelerated Kirchhoff pre-stack time migration.

Usage:
    engine = MigrationEngine()
    image, fold = engine.migrate_bin(traces, geometry_params, output_params)
"""

import numpy as np
import torch
from typing import Tuple, Optional, Callable, Dict, Any
import logging
import time

from processors.migration.geometry_preprocessor import (
    GeometryPreprocessor,
    PrecomputedGeometry,
)
from processors.migration.kirchhoff_kernel import (
    KirchhoffKernel,
    normalize_by_fold,
    migrate_kirchhoff_full,
    migrate_kirchhoff_time_domain,
    migrate_kirchhoff_time_domain_rms,
)
from processors.migration.velocity_model import VelocityModel, create_velocity_model

logger = logging.getLogger(__name__)


class MigrationEngine:
    """
    Main migration engine orchestrating geometry preprocessing and migration.

    This is the primary interface for performing PSTM migration.
    It manages:
    - Device selection (CPU/GPU)
    - Geometry preprocessing
    - Migration kernel execution
    - Memory management through batching

    Example:
        engine = MigrationEngine(device=torch.device('mps'))
        image, fold = engine.migrate_bin(
            traces=trace_data,
            source_x=sx, source_y=sy,
            receiver_x=rx, receiver_y=ry,
            # ... other parameters
        )
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize migration engine.

        Args:
            device: Target device. If None, auto-selects:
                    MPS (Apple Silicon) > CUDA > CPU
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        self.device = device
        self.preprocessor = GeometryPreprocessor()
        self.kernel = KirchhoffKernel(device)

        logger.info(f"MigrationEngine initialized on {device}")

    def migrate_bin(
        self,
        # Trace data
        traces: np.ndarray,
        # Geometry
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        # Output grid
        origin_x: float,
        origin_y: float,
        il_spacing: float,
        xl_spacing: float,
        azimuth_deg: float,
        n_il: int,
        n_xl: int,
        # Time axis
        dt_ms: float,
        t_min_ms: float,
        # Velocity
        velocity_mps: float,
        # Migration parameters
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        # Options
        normalize: bool = True,
        min_fold: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Migrate all traces in a bin.

        Args:
            traces: Trace data (n_samples, n_traces) float32
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
            velocity_mps: Velocity (m/s)
            max_aperture_m: Maximum aperture (m)
            max_angle_deg: Maximum angle (degrees)
            normalize: Whether to normalize by fold
            min_fold: Minimum fold for normalization
            progress_callback: Optional progress callback(fraction, message)

        Returns:
            image: Migrated image (n_samples, n_il, n_xl)
            fold: Fold count (n_samples, n_il, n_xl)
        """
        n_samples, n_traces = traces.shape
        t0 = time.time()

        logger.info(f"Starting migration: {n_traces} traces, {n_samples} samples, "
                   f"{n_il}x{n_xl} output grid")

        if progress_callback:
            progress_callback(0.0, "Preprocessing geometry...")

        # Step 1: Precompute geometry
        t1 = time.time()
        geometry = self.preprocessor.precompute(
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            origin_x=origin_x,
            origin_y=origin_y,
            il_spacing=il_spacing,
            xl_spacing=xl_spacing,
            azimuth_deg=azimuth_deg,
            n_il=n_il,
            n_xl=n_xl,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            n_samples=n_samples,
            velocity_mps=velocity_mps,
            max_aperture_m=max_aperture_m,
            max_angle_deg=max_angle_deg,
            device=self.device,
        )
        precompute_time = time.time() - t1
        logger.info(f"  Geometry precomputed in {precompute_time:.2f}s")

        if progress_callback:
            progress_callback(0.3, "Transferring traces to GPU...")

        # Step 2: Transfer traces to GPU
        t2 = time.time()
        traces_tensor = torch.from_numpy(traces.astype(np.float32)).to(self.device)
        transfer_time = time.time() - t2
        logger.info(f"  Traces transferred in {transfer_time:.2f}s")

        if progress_callback:
            progress_callback(0.4, "Running migration kernel...")

        # Step 3: Run migration
        t3 = time.time()
        image, fold = self.kernel.migrate(
            traces_tensor,
            geometry,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            n_il=n_il,
            n_xl=n_xl,
        )
        kernel_time = time.time() - t3
        logger.info(f"  Migration kernel completed in {kernel_time:.2f}s")

        if progress_callback:
            progress_callback(0.8, "Normalizing...")

        # Step 4: Normalize if requested
        if normalize:
            image = normalize_by_fold(image, fold, min_fold)

        if progress_callback:
            progress_callback(0.9, "Transferring results...")

        # Step 5: Transfer back to CPU
        t4 = time.time()
        image_np = image.cpu().numpy()
        fold_np = fold.cpu().numpy()
        result_time = time.time() - t4

        total_time = time.time() - t0
        traces_per_sec = n_traces / total_time

        logger.info(f"Migration complete: {total_time:.2f}s total, "
                   f"{traces_per_sec:.0f} traces/s")
        logger.info(f"  Breakdown: precompute={precompute_time:.2f}s, "
                   f"transfer={transfer_time:.2f}s, kernel={kernel_time:.2f}s, "
                   f"result={result_time:.2f}s")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return image_np, fold_np

    def migrate_bin_full(
        self,
        # Trace data
        traces: np.ndarray,
        # Geometry
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        # Output grid
        origin_x: float,
        origin_y: float,
        il_spacing: float,
        xl_spacing: float,
        azimuth_deg: float,
        n_il: int,
        n_xl: int,
        # Time axis
        dt_ms: float,
        t_min_ms: float,
        n_times: int = None,  # Number of output time samples (if None, uses input trace length)
        # Velocity
        velocity_mps: float = 2500.0,
        # Migration parameters
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        # Options
        normalize: bool = True,
        min_fold: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        enable_profiling: bool = False,
        use_time_dependent_aperture: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Migrate using full Kirchhoff summation (output-point centric).

        This version correctly gathers energy from ALL traces to each output point.
        Slower but produces correct results.

        Args:
            Same as migrate_bin, plus:
            n_times: Number of output time samples. If None, uses input trace length.
            enable_profiling: If True, logs detailed timing breakdown
            use_time_dependent_aperture: If True, aperture varies with depth/time

        Returns:
            image: Migrated image (n_times, n_il, n_xl)
            fold: Fold count (n_times, n_il, n_xl)
        """
        n_input_samples, n_traces = traces.shape
        # Use n_times if specified, otherwise use input trace length
        n_output_samples = n_times if n_times is not None else n_input_samples
        t0 = time.time()

        logger.info(f"Starting FULL Kirchhoff migration: {n_traces} traces, "
                   f"{n_input_samples} input samples -> {n_output_samples} output samples, {n_il}x{n_xl} output grid")

        if progress_callback:
            progress_callback(0.0, "Preparing data...")

        # Transfer data to GPU
        traces_t = torch.from_numpy(traces.astype(np.float32)).to(self.device)
        source_x_t = torch.from_numpy(source_x.astype(np.float32)).to(self.device)
        source_y_t = torch.from_numpy(source_y.astype(np.float32)).to(self.device)
        receiver_x_t = torch.from_numpy(receiver_x.astype(np.float32)).to(self.device)
        receiver_y_t = torch.from_numpy(receiver_y.astype(np.float32)).to(self.device)

        # Create output grid coordinates
        # For zero azimuth: output_x = origin_x + il * il_spacing
        #                   output_y = origin_y + xl * xl_spacing
        il_coords = torch.arange(n_il, device=self.device, dtype=torch.float32) * il_spacing + origin_x
        xl_coords = torch.arange(n_xl, device=self.device, dtype=torch.float32) * xl_spacing + origin_y

        # Create meshgrid and flatten (il-major order)
        xl_grid, il_grid = torch.meshgrid(xl_coords, il_coords, indexing='xy')
        output_x = il_grid.flatten()
        output_y = xl_grid.flatten()

        # Create OUTPUT depth axis from time (z = v * t / 2)
        # This determines the output grid size (n_output_samples)
        time_axis_ms = torch.arange(n_output_samples, device=self.device, dtype=torch.float32) * dt_ms + t_min_ms
        depth_axis = velocity_mps * (time_axis_ms / 1000.0) / 2.0  # meters

        if progress_callback:
            progress_callback(0.1, "Running Kirchhoff migration...")

        # Create a kernel progress callback that scales to 10%-90% range
        def kernel_progress(pct: float, msg: str):
            if progress_callback:
                # Map kernel progress (0-1) to (0.1-0.9)
                scaled_pct = 0.1 + pct * 0.8
                progress_callback(scaled_pct, msg)

        # Run full Kirchhoff migration
        t1 = time.time()
        image, fold = migrate_kirchhoff_full(
            traces=traces_t,
            source_x=source_x_t,
            source_y=source_y_t,
            receiver_x=receiver_x_t,
            receiver_y=receiver_y_t,
            output_x=output_x,
            output_y=output_y,
            depth_axis=depth_axis,
            velocity=velocity_mps,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            max_aperture_m=max_aperture_m,
            max_angle_deg=max_angle_deg,
            n_il=n_il,
            n_xl=n_xl,
            progress_callback=kernel_progress,
            enable_profiling=enable_profiling,
            use_time_dependent_aperture=use_time_dependent_aperture,
        )
        kernel_time = time.time() - t1
        logger.info(f"  Kirchhoff kernel completed in {kernel_time:.2f}s")

        if progress_callback:
            progress_callback(0.9, "Normalizing...")

        # Normalize if requested
        if normalize:
            image = normalize_by_fold(image, fold, min_fold)

        # Transfer back to CPU
        image_np = image.cpu().numpy()
        fold_np = fold.cpu().numpy()

        total_time = time.time() - t0
        logger.info(f"Full migration complete: {total_time:.2f}s")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return image_np, fold_np

    def migrate_bin_batched(
        self,
        traces: np.ndarray,
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
        velocity_mps: float,
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        normalize: bool = True,
        min_fold: int = 1,
        batch_size: int = 10000,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Migrate traces in batches for memory efficiency.

        Processes traces in batches to limit peak memory usage.
        Results are accumulated across batches.

        Args:
            Same as migrate_bin, plus:
            batch_size: Number of traces per batch

        Returns:
            image: Migrated image (n_samples, n_il, n_xl)
            fold: Fold count (n_samples, n_il, n_xl)
        """
        n_samples, n_traces = traces.shape
        n_batches = (n_traces + batch_size - 1) // batch_size

        logger.info(f"Batched migration: {n_traces} traces in {n_batches} batches "
                   f"of {batch_size}")

        # Initialize accumulators on CPU (to save GPU memory between batches)
        total_image = np.zeros((n_samples, n_il, n_xl), dtype=np.float32)
        total_fold = np.zeros((n_samples, n_il, n_xl), dtype=np.float32)

        t0 = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_traces)

            if progress_callback:
                progress = batch_idx / n_batches
                progress_callback(progress, f"Processing batch {batch_idx + 1}/{n_batches}")

            # Migrate this batch (without normalization - we'll normalize at end)
            batch_image, batch_fold = self.migrate_bin(
                traces=traces[:, start:end],
                source_x=source_x[start:end],
                source_y=source_y[start:end],
                receiver_x=receiver_x[start:end],
                receiver_y=receiver_y[start:end],
                origin_x=origin_x,
                origin_y=origin_y,
                il_spacing=il_spacing,
                xl_spacing=xl_spacing,
                azimuth_deg=azimuth_deg,
                n_il=n_il,
                n_xl=n_xl,
                dt_ms=dt_ms,
                t_min_ms=t_min_ms,
                velocity_mps=velocity_mps,
                max_aperture_m=max_aperture_m,
                max_angle_deg=max_angle_deg,
                normalize=False,  # Don't normalize individual batches
                progress_callback=None,
            )

            # Accumulate
            total_image += batch_image
            total_fold += batch_fold

            # Clear GPU cache between batches
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

        total_time = time.time() - t0
        logger.info(f"Batched migration complete: {total_time:.2f}s, "
                   f"{n_traces/total_time:.0f} traces/s")

        # Normalize final result
        if normalize:
            mask = total_fold >= min_fold
            total_image[mask] = total_image[mask] / total_fold[mask]

        if progress_callback:
            progress_callback(1.0, "Complete")

        return total_image, total_fold

    def migrate_bin_time_domain(
        self,
        # Trace data
        traces: np.ndarray,
        # Geometry
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        # Output grid
        origin_x: float,
        origin_y: float,
        il_spacing: float,
        xl_spacing: float,
        azimuth_deg: float,
        n_il: int,
        n_xl: int,
        # Time axis
        dt_ms: float,
        t_min_ms: float,
        n_times: int,
        # Velocity
        velocity_mps: float,
        # Migration parameters
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        tile_size: int = 100,
        # Options
        normalize: bool = True,
        min_fold: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        enable_profiling: bool = False,
        use_time_dependent_aperture: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Migrate using direct time-domain mapping (fastest for time migration).

        This method maps input time samples directly to output times without
        iterating over depths, providing significant speedup (50-100x) for
        constant-velocity time migration.

        Uses the equation: t_out = sqrt(t_in² + 4*h_eff²/v²)

        Args:
            traces: Trace data (n_samples, n_traces) float32
            source_x: Source X coordinates (n_traces,)
            source_y: Source Y coordinates (n_traces,)
            receiver_x: Receiver X coordinates (n_traces,)
            receiver_y: Receiver Y coordinates (n_traces,)
            origin_x: Grid origin X
            origin_y: Grid origin Y
            il_spacing: Inline spacing (m)
            xl_spacing: Crossline spacing (m)
            azimuth_deg: Grid azimuth (degrees)
            n_il: Number of inlines
            n_xl: Number of crosslines
            dt_ms: Sample interval (ms)
            t_min_ms: Start time (ms)
            n_times: Number of output time samples
            velocity_mps: Velocity (m/s)
            max_aperture_m: Maximum aperture (m)
            max_angle_deg: Maximum angle (degrees)
            tile_size: Number of output points per tile (default 100)
            normalize: Whether to normalize by fold
            min_fold: Minimum fold for normalization
            progress_callback: Optional progress callback(fraction, message)
            enable_profiling: If True, logs detailed timing breakdown
            use_time_dependent_aperture: If True, aperture varies with time

        Returns:
            image: Migrated image (n_times, n_il, n_xl)
            fold: Fold count (n_times, n_il, n_xl)
        """
        n_samples, n_traces = traces.shape
        t0 = time.time()

        logger.info(f"Starting TIME-DOMAIN migration: {n_traces} traces, "
                   f"{n_samples} samples -> {n_times} output times, {n_il}x{n_xl} grid")
        logger.info(f"  tile_size={tile_size}, time_dependent_aperture={use_time_dependent_aperture}")

        if progress_callback:
            progress_callback(0.0, "Preparing data...")

        # Transfer data to GPU
        traces_t = torch.from_numpy(traces.astype(np.float32)).to(self.device)
        source_x_t = torch.from_numpy(source_x.astype(np.float32)).to(self.device)
        source_y_t = torch.from_numpy(source_y.astype(np.float32)).to(self.device)
        receiver_x_t = torch.from_numpy(receiver_x.astype(np.float32)).to(self.device)
        receiver_y_t = torch.from_numpy(receiver_y.astype(np.float32)).to(self.device)

        # Create output grid coordinates
        # Apply rotation for azimuth
        cos_az = np.cos(np.radians(azimuth_deg))
        sin_az = np.sin(np.radians(azimuth_deg))

        il_offsets = torch.arange(n_il, device=self.device, dtype=torch.float32) * il_spacing
        xl_offsets = torch.arange(n_xl, device=self.device, dtype=torch.float32) * xl_spacing

        # Create meshgrid (il-major order)
        xl_grid, il_grid = torch.meshgrid(xl_offsets, il_offsets, indexing='xy')

        # Apply rotation: x = il*cos(az) - xl*sin(az), y = il*sin(az) + xl*cos(az)
        output_x = origin_x + il_grid.flatten() * cos_az - xl_grid.flatten() * sin_az
        output_y = origin_y + il_grid.flatten() * sin_az + xl_grid.flatten() * cos_az

        # Create output time axis
        time_axis_ms = torch.arange(n_times, device=self.device, dtype=torch.float32) * dt_ms + t_min_ms

        if progress_callback:
            progress_callback(0.1, "Running time-domain migration...")

        # Create a kernel progress callback
        def kernel_progress(pct: float, msg: str):
            if progress_callback:
                scaled_pct = 0.1 + pct * 0.8
                progress_callback(scaled_pct, msg)

        # Run time-domain migration
        t1 = time.time()
        image, fold = migrate_kirchhoff_time_domain(
            traces=traces_t,
            source_x=source_x_t,
            source_y=source_y_t,
            receiver_x=receiver_x_t,
            receiver_y=receiver_y_t,
            output_x=output_x,
            output_y=output_y,
            time_axis_ms=time_axis_ms,
            velocity=velocity_mps,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            max_aperture_m=max_aperture_m,
            max_angle_deg=max_angle_deg,
            n_il=n_il,
            n_xl=n_xl,
            tile_size=tile_size,
            progress_callback=kernel_progress,
            enable_profiling=enable_profiling,
            use_time_dependent_aperture=use_time_dependent_aperture,
        )
        kernel_time = time.time() - t1
        logger.info(f"  Time-domain kernel completed in {kernel_time:.2f}s")

        if progress_callback:
            progress_callback(0.9, "Normalizing...")

        # Normalize if requested
        if normalize:
            image = normalize_by_fold(image, fold, min_fold)

        # Transfer back to CPU
        image_np = image.cpu().numpy()
        fold_np = fold.cpu().numpy()

        total_time = time.time() - t0
        output_samples = n_times * n_il * n_xl
        output_rate = output_samples / total_time

        logger.info(f"Time-domain migration complete: {total_time:.2f}s, "
                   f"{output_rate:.0f} output-samples/s")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return image_np, fold_np

    def migrate_bin_time_domain_rms(
        self,
        # Trace data
        traces: np.ndarray,
        # Geometry
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        # Output grid
        origin_x: float,
        origin_y: float,
        il_spacing: float,
        xl_spacing: float,
        azimuth_deg: float,
        n_il: int,
        n_xl: int,
        # Time axis
        dt_ms: float,
        t_min_ms: float,
        n_times: int,
        # Velocity model
        velocity_model: VelocityModel,
        # Migration parameters
        max_aperture_m: float = 5000.0,
        max_angle_deg: float = 60.0,
        tile_size: int = 100,
        # Options
        normalize: bool = True,
        min_fold: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        enable_profiling: bool = False,
        use_time_dependent_aperture: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-domain migration with RMS velocity support for non-constant velocity.

        This method supports:
        - Constant velocity: v_rms = v0
        - Linear gradient: v_rms computed analytically
        - From file: v_rms computed numerically

        Uses iterative refinement for accurate time mapping with variable velocity.

        Args:
            traces: Trace data (n_samples, n_traces) float32
            source_x: Source X coordinates (n_traces,)
            source_y: Source Y coordinates (n_traces,)
            receiver_x: Receiver X coordinates (n_traces,)
            receiver_y: Receiver Y coordinates (n_traces,)
            origin_x: Grid origin X
            origin_y: Grid origin Y
            il_spacing: Inline spacing (m)
            xl_spacing: Crossline spacing (m)
            azimuth_deg: Grid azimuth (degrees)
            n_il: Number of inlines
            n_xl: Number of crosslines
            dt_ms: Sample interval (ms)
            t_min_ms: Start time (ms)
            n_times: Number of output time samples
            velocity_model: VelocityModel instance
            max_aperture_m: Maximum aperture (m)
            max_angle_deg: Maximum angle (degrees)
            tile_size: Number of output points per tile
            normalize: Whether to normalize by fold
            min_fold: Minimum fold for normalization
            progress_callback: Optional progress callback
            enable_profiling: If True, logs detailed timing breakdown
            use_time_dependent_aperture: If True, aperture varies with time

        Returns:
            image: Migrated image (n_times, n_il, n_xl)
            fold: Fold count (n_times, n_il, n_xl)
        """
        n_samples, n_traces = traces.shape
        t0 = time.time()

        logger.info(f"Starting TIME-DOMAIN (RMS) migration: {n_traces} traces, "
                   f"{n_samples} samples -> {n_times} output times, {n_il}x{n_xl} grid")
        logger.info(f"  Velocity model: {velocity_model.get_summary()}")
        logger.info(f"  tile_size={tile_size}, time_dependent_aperture={use_time_dependent_aperture}")

        if progress_callback:
            progress_callback(0.0, "Preparing data...")

        # Transfer data to GPU
        traces_t = torch.from_numpy(traces.astype(np.float32)).to(self.device)
        source_x_t = torch.from_numpy(source_x.astype(np.float32)).to(self.device)
        source_y_t = torch.from_numpy(source_y.astype(np.float32)).to(self.device)
        receiver_x_t = torch.from_numpy(receiver_x.astype(np.float32)).to(self.device)
        receiver_y_t = torch.from_numpy(receiver_y.astype(np.float32)).to(self.device)

        # Create output grid coordinates with rotation
        cos_az = np.cos(np.radians(azimuth_deg))
        sin_az = np.sin(np.radians(azimuth_deg))

        il_offsets = torch.arange(n_il, device=self.device, dtype=torch.float32) * il_spacing
        xl_offsets = torch.arange(n_xl, device=self.device, dtype=torch.float32) * xl_spacing

        xl_grid, il_grid = torch.meshgrid(xl_offsets, il_offsets, indexing='xy')

        output_x = origin_x + il_grid.flatten() * cos_az - xl_grid.flatten() * sin_az
        output_y = origin_y + il_grid.flatten() * sin_az + xl_grid.flatten() * cos_az

        # Create output time axis
        time_axis_ms = torch.arange(n_times, device=self.device, dtype=torch.float32) * dt_ms + t_min_ms

        if progress_callback:
            progress_callback(0.1, "Running time-domain migration (RMS)...")

        def kernel_progress(pct: float, msg: str):
            if progress_callback:
                scaled_pct = 0.1 + pct * 0.8
                progress_callback(scaled_pct, msg)

        # Run time-domain migration with RMS velocity
        t1 = time.time()
        image, fold = migrate_kirchhoff_time_domain_rms(
            traces=traces_t,
            source_x=source_x_t,
            source_y=source_y_t,
            receiver_x=receiver_x_t,
            receiver_y=receiver_y_t,
            output_x=output_x,
            output_y=output_y,
            time_axis_ms=time_axis_ms,
            velocity_model=velocity_model,
            dt_ms=dt_ms,
            t_min_ms=t_min_ms,
            max_aperture_m=max_aperture_m,
            max_angle_deg=max_angle_deg,
            n_il=n_il,
            n_xl=n_xl,
            tile_size=tile_size,
            progress_callback=kernel_progress,
            enable_profiling=enable_profiling,
            use_time_dependent_aperture=use_time_dependent_aperture,
        )
        kernel_time = time.time() - t1
        logger.info(f"  Time-domain (RMS) kernel completed in {kernel_time:.2f}s")

        if progress_callback:
            progress_callback(0.9, "Normalizing...")

        if normalize:
            image = normalize_by_fold(image, fold, min_fold)

        image_np = image.cpu().numpy()
        fold_np = fold.cpu().numpy()

        total_time = time.time() - t0
        output_samples = n_times * n_il * n_xl
        output_rate = output_samples / total_time

        logger.info(f"Time-domain (RMS) migration complete: {total_time:.2f}s, "
                   f"{output_rate:.0f} output-samples/s")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return image_np, fold_np

    def benchmark(
        self,
        n_traces: int = 10000,
        n_samples: int = 1501,
        n_il: int = 100,
        n_xl: int = 100,
    ) -> Dict[str, float]:
        """
        Run benchmark with synthetic data.

        Args:
            n_traces: Number of synthetic traces
            n_samples: Number of time samples
            n_il: Number of inlines
            n_xl: Number of crosslines

        Returns:
            Dictionary with timing results
        """
        logger.info(f"Running benchmark: {n_traces} traces, {n_samples} samples, "
                   f"{n_il}x{n_xl} grid")

        # Create synthetic data
        np.random.seed(42)
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)

        # Random geometry within grid
        spacing = 50.0
        origin = 2500.0
        max_coord = origin + (n_il - 1) * spacing

        source_x = np.random.uniform(origin, max_coord, n_traces).astype(np.float32)
        source_y = np.random.uniform(origin, max_coord, n_traces).astype(np.float32)
        receiver_x = source_x.copy()  # Zero-offset
        receiver_y = source_y.copy()

        # Run migration
        t0 = time.time()
        image, fold = self.migrate_bin(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            origin_x=origin,
            origin_y=origin,
            il_spacing=spacing,
            xl_spacing=spacing,
            azimuth_deg=0.0,
            n_il=n_il,
            n_xl=n_xl,
            dt_ms=2.0,
            t_min_ms=0.0,
            velocity_mps=3000.0,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
        )
        total_time = time.time() - t0

        results = {
            'n_traces': n_traces,
            'n_samples': n_samples,
            'n_il': n_il,
            'n_xl': n_xl,
            'total_time_s': total_time,
            'traces_per_second': n_traces / total_time,
            'output_points': n_samples * n_il * n_xl,
            'device': str(self.device),
        }

        logger.info(f"Benchmark results: {results['traces_per_second']:.0f} traces/s")

        return results
