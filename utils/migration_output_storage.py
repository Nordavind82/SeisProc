"""
Migration Output Storage - Zarr/Parquet format for migrated seismic data.

Saves migration results in the same format as SEG-Y import for seamless
integration with the SeisProc application navigation and display.

Storage structure (matches SEG-Y import exactly):
    output_dir/
        ├── traces.zarr/          # Migrated image data (n_samples, n_traces)
        ├── headers.parquet       # Trace headers (inline, xline, coordinates)
        ├── ensemble_index.parquet # Inline groupings for navigation
        ├── trace_index.parquet   # Trace lookup
        ├── fold.npy              # Fold map as numpy (auxiliary data)
        └── metadata.json         # Migration parameters and grid info
"""

import gc
import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MigrationOutputStorage:
    """
    Manages storage of migration output in Zarr/Parquet format.

    The output format matches SEG-Y import format so migrated data can be:
    - Loaded and displayed in the SeisProc application
    - Navigated by inline/crossline
    - Used as input to further processing

    Storage Layout:
        - traces.zarr: Migrated image, shape (n_time, n_traces) where
          n_traces = n_inline * n_xline (flattened 3D cube)
        - fold.zarr: Fold map with same shape
        - headers.parquet: One row per trace with inline, xline, cdp_x, cdp_y
        - ensemble_index.parquet: Groups traces by inline for navigation
        - metadata.json: Grid geometry, velocity info, processing params

    Example:
        >>> storage = MigrationOutputStorage('output_dir')
        >>> storage.initialize_output(n_time=1500, n_inline=100, n_xline=200,
        ...                           dt_ms=4.0, d_inline=25.0, d_xline=12.5)
        >>> storage.write_inline(inline_idx=50, image_data, fold_data)
        >>> storage.finalize()
    """

    def __init__(self, output_dir: str):
        """
        Initialize migration output storage.

        Args:
            output_dir: Directory for storing migrated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths (matching SEG-Y import structure exactly)
        self.traces_path = self.output_dir / 'traces.zarr'
        self.fold_path = self.output_dir / 'fold.npy'  # Auxiliary data, not zarr
        self.headers_path = self.output_dir / 'headers.parquet'
        self.ensemble_index_path = self.output_dir / 'ensemble_index.parquet'
        self.trace_index_path = self.output_dir / 'trace_index.parquet'
        self.metadata_path = self.output_dir / 'metadata.json'

        # Source file tracking (for app compatibility)
        self._source_file = None

        # Grid parameters (set during initialize)
        self._n_time = 0
        self._n_inline = 0
        self._n_xline = 0
        self._n_traces = 0
        self._dt_ms = 4.0
        self._t0_ms = 0.0
        self._d_inline = 25.0
        self._d_xline = 25.0
        self._inline_start = 1
        self._xline_start = 1
        self._x_origin = 0.0
        self._y_origin = 0.0
        self._grid_azimuth_deg = 0.0

        # Zarr arrays (created during initialize)
        self._traces_zarr = None
        self._fold_zarr = None

        # Processing stats
        self._stats = {
            'inlines_written': 0,
            'traces_written': 0,
            'max_fold': 0,
            'start_time': None,
            'end_time': None,
        }

    def initialize_output(
        self,
        n_time: int,
        n_inline: int,
        n_xline: int,
        dt_ms: float = 4.0,
        t0_ms: float = 0.0,
        d_inline: float = 25.0,
        d_xline: float = 25.0,
        inline_start: int = 1,
        xline_start: int = 1,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        grid_azimuth_deg: float = 0.0,
        chunk_size: int = 1000,
        source_file: Optional[str] = None,
    ):
        """
        Initialize output arrays and metadata.

        Creates Zarr arrays for traces and fold with proper chunking.
        The 3D cube (time, inline, xline) is stored as 2D (time, traces)
        where traces are ordered inline-major (all xlines for inline 1,
        then all xlines for inline 2, etc.).

        Args:
            n_time: Number of time samples
            n_inline: Number of inlines
            n_xline: Number of crosslines
            dt_ms: Sample interval in milliseconds
            t0_ms: Start time in milliseconds
            d_inline: Inline spacing in meters
            d_xline: Crossline spacing in meters
            inline_start: First inline number
            xline_start: First crossline number
            x_origin: X coordinate of origin (inline_start, xline_start)
            y_origin: Y coordinate of origin
            grid_azimuth_deg: Inline direction azimuth from north (degrees)
            chunk_size: Zarr chunk size along trace dimension
            source_file: Original input file path (for app compatibility)
        """
        self._n_time = n_time
        self._n_inline = n_inline
        self._n_xline = n_xline
        self._n_traces = n_inline * n_xline
        self._dt_ms = dt_ms
        self._t0_ms = t0_ms
        self._d_inline = d_inline
        self._d_xline = d_xline
        self._inline_start = inline_start
        self._xline_start = xline_start
        self._x_origin = x_origin
        self._y_origin = y_origin
        self._grid_azimuth_deg = grid_azimuth_deg
        self._source_file = source_file

        self._stats['start_time'] = datetime.now().isoformat()

        logger.info(f"Initializing migration output: {n_inline}x{n_xline}x{n_time} "
                   f"= {self._n_traces:,} traces")

        # Create Zarr array for traces only (transposed format: n_time, n_traces)
        # This matches the SEG-Y import format exactly
        self._traces_zarr = zarr.open(
            str(self.traces_path),
            mode='w',
            shape=(n_time, self._n_traces),
            chunks=(n_time, min(chunk_size, self._n_traces)),
            dtype=np.float32,
            compressor=None,  # No compression for write speed
            zarr_format=2
        )

        # Fold stored as numpy array in memory, saved at finalize
        self._fold_array = np.zeros((n_time, self._n_traces), dtype=np.float32)

        # Initialize traces with zeros
        self._traces_zarr[:] = 0

        logger.info(f"Created Zarr array: {self.traces_path}")

    def write_inline(
        self,
        inline_idx: int,
        image_data: np.ndarray,
        fold_data: np.ndarray,
    ):
        """
        Write one inline of migrated data.

        Args:
            inline_idx: Inline index (0-based)
            image_data: Image data, shape (n_time, n_xline)
            fold_data: Fold data, shape (n_time, n_xline)
        """
        if self._traces_zarr is None:
            raise RuntimeError("Output not initialized. Call initialize_output first.")

        # Calculate trace range for this inline
        start_trace = inline_idx * self._n_xline
        end_trace = start_trace + self._n_xline

        # Write image to Zarr, fold to numpy array
        self._traces_zarr[:, start_trace:end_trace] = image_data
        self._fold_array[:, start_trace:end_trace] = fold_data

        self._stats['inlines_written'] += 1
        self._stats['traces_written'] += self._n_xline
        self._stats['max_fold'] = max(self._stats['max_fold'], float(fold_data.max()))

    def write_full_cube(
        self,
        image_cube: np.ndarray,
        fold_cube: np.ndarray,
    ):
        """
        Write complete migration cube at once.

        Args:
            image_cube: Full image, shape (n_time, n_inline, n_xline)
            fold_cube: Full fold, shape (n_time, n_inline, n_xline)
        """
        if self._traces_zarr is None:
            raise RuntimeError("Output not initialized. Call initialize_output first.")

        n_time, n_inline, n_xline = image_cube.shape

        if n_time != self._n_time or n_inline != self._n_inline or n_xline != self._n_xline:
            raise ValueError(f"Shape mismatch: expected ({self._n_time}, {self._n_inline}, {self._n_xline}), "
                           f"got ({n_time}, {n_inline}, {n_xline})")

        # Reshape 3D cube to 2D (time, traces) - inline major order
        image_2d = image_cube.reshape(n_time, n_inline * n_xline)
        fold_2d = fold_cube.reshape(n_time, n_inline * n_xline)

        # Write image to Zarr, fold to numpy array
        self._traces_zarr[:] = image_2d
        self._fold_array[:] = fold_2d

        self._stats['inlines_written'] = n_inline
        self._stats['traces_written'] = n_inline * n_xline
        self._stats['max_fold'] = float(fold_cube.max())

        logger.info(f"Wrote full cube: {n_inline}x{n_xline}x{n_time}, max fold={self._stats['max_fold']:.0f}")

    def _build_headers(self) -> pd.DataFrame:
        """
        Build header DataFrame for all traces.

        Creates headers with inline, xline, and computed coordinates.
        Supports rotated grids using grid_azimuth_deg (inline direction from north).
        """
        import math

        headers = []

        # Convert azimuth to radians for coordinate rotation
        az_rad = math.radians(self._grid_azimuth_deg)
        sin_az = math.sin(az_rad)
        cos_az = math.cos(az_rad)

        trace_idx = 0
        for il_idx in range(self._n_inline):
            inline = self._inline_start + il_idx

            for xl_idx in range(self._n_xline):
                xline = self._xline_start + xl_idx

                # Distance along inline and crossline directions
                dist_inline = il_idx * self._d_inline
                dist_xline = xl_idx * self._d_xline

                # Compute rotated coordinates
                # Inline direction: azimuth angle from north (Y-axis)
                # Crossline direction: azimuth + 90 degrees (perpendicular to inline)
                cdp_x = self._x_origin + dist_inline * sin_az + dist_xline * cos_az
                cdp_y = self._y_origin + dist_inline * cos_az - dist_xline * sin_az

                headers.append({
                    'trace_index': trace_idx,
                    'inline': inline,
                    'xline': xline,
                    'cdp_x': cdp_x,
                    'cdp_y': cdp_y,
                    'ensemble_number': inline,  # Group by inline for navigation
                    'trace_in_ensemble': xl_idx,
                })
                trace_idx += 1

        return pd.DataFrame(headers)

    def _build_ensemble_index(self) -> pd.DataFrame:
        """
        Build ensemble index grouping traces by inline.

        This enables inline-by-inline navigation in the application.
        """
        ensembles = []

        for il_idx in range(self._n_inline):
            inline = self._inline_start + il_idx
            start_trace = il_idx * self._n_xline
            end_trace = start_trace + self._n_xline - 1

            ensembles.append({
                'ensemble_id': il_idx,
                'ensemble_value': inline,
                'inline': inline,
                'start_trace': start_trace,
                'end_trace': end_trace,
                'n_traces': self._n_xline,
            })

        return pd.DataFrame(ensembles)

    def _build_metadata(self, extra_metadata: Optional[Dict] = None) -> Dict:
        """
        Build metadata dictionary.

        Args:
            extra_metadata: Additional metadata to include (e.g., velocity info)
        """
        self._stats['end_time'] = datetime.now().isoformat()

        metadata = {
            # Standard fields (matching SEG-Y import)
            'shape': [self._n_time, self._n_traces],
            'sample_rate': self._dt_ms,
            'n_samples': self._n_time,
            'n_traces': self._n_traces,
            'duration_ms': (self._n_time - 1) * self._dt_ms,
            'nyquist_freq': 1000.0 / (2 * self._dt_ms),

            # Migration-specific fields
            'data_type': 'migration',
            'migration_info': {
                'n_inline': self._n_inline,
                'n_xline': self._n_xline,
                'n_time': self._n_time,
                'dt_ms': self._dt_ms,
                't0_ms': self._t0_ms,
                'd_inline': self._d_inline,
                'd_xline': self._d_xline,
                'inline_start': self._inline_start,
                'xline_start': self._xline_start,
                'inline_end': self._inline_start + self._n_inline - 1,
                'xline_end': self._xline_start + self._n_xline - 1,
                'x_origin': self._x_origin,
                'y_origin': self._y_origin,
                'grid_azimuth_deg': self._grid_azimuth_deg,
            },

            # Processing stats
            'processing_stats': {
                'max_fold': self._stats['max_fold'],
                'inlines_written': self._stats['inlines_written'],
                'traces_written': self._stats['traces_written'],
                'start_time': self._stats['start_time'],
                'end_time': self._stats['end_time'],
            },

            # Storage info
            'storage_info': {
                'zarr_chunks': f"({self._n_time}, chunk_size)",
                'parquet_compression': 'snappy',
                'zarr_compression': 'none',
                'fold_available': True,
            },

            # SeismicData compatibility (required by app)
            'seismic_metadata': {
                'source_file': self._source_file or str(self.output_dir),
                'source_type': 'migration',
                'processing_history': ['PSTM Kirchhoff Migration'],
                'file_info': {
                    'filename': self._source_file or 'PSTM Migration Output',
                    'n_traces': self._n_traces,
                    'n_samples': self._n_time,
                    'sample_interval': self._dt_ms,
                    'trace_length_ms': (self._n_time - 1) * self._dt_ms,
                }
            }
        }

        # Add extra metadata if provided
        if extra_metadata:
            metadata['migration_params'] = extra_metadata

        return metadata

    def finalize(
        self,
        normalize: bool = True,
        extra_metadata: Optional[Dict] = None,
    ):
        """
        Finalize output: normalize by fold, save headers and metadata.

        Args:
            normalize: If True, divide image by fold (avoid division by zero)
            extra_metadata: Additional metadata (velocity model, aperture, etc.)
        """
        if self._traces_zarr is None:
            raise RuntimeError("Output not initialized")

        logger.info("Finalizing migration output...")

        # Normalize by fold if requested
        if normalize:
            logger.info("  Normalizing image by fold...")
            # Read, normalize, write back
            image = self._traces_zarr[:]
            fold = self._fold_array

            with np.errstate(divide='ignore', invalid='ignore'):
                normalized = np.where(fold > 0, image / fold, 0.0)

            self._traces_zarr[:] = normalized.astype(np.float32)
            logger.info(f"  Normalization complete (max fold: {fold.max():.0f})")

            del image, normalized
            gc.collect()

        # Save fold as numpy file (auxiliary data, not part of main dataset)
        logger.info("  Saving fold map...")
        np.save(self.fold_path, self._fold_array)
        logger.info(f"  Fold saved to {self.fold_path}")

        # Save headers
        logger.info("  Saving headers to Parquet...")
        df_headers = self._build_headers()
        df_headers.to_parquet(
            self.headers_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"  Saved {len(df_headers):,} trace headers")

        # Save ensemble index
        logger.info("  Saving ensemble index...")
        df_ensembles = self._build_ensemble_index()
        df_ensembles.to_parquet(
            self.ensemble_index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"  Saved {len(df_ensembles)} inline ensembles")

        # Save trace index
        logger.info("  Saving trace index...")
        df_trace_idx = pd.DataFrame({
            'trace_index': np.arange(self._n_traces),
            'global_trace_id': np.arange(self._n_traces),
        })
        df_trace_idx.to_parquet(
            self.trace_index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Save metadata
        logger.info("  Saving metadata...")
        metadata = self._build_metadata(extra_metadata)
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Migration output saved to: {self.output_dir}")
        logger.info(f"  Shape: {self._n_inline}x{self._n_xline}x{self._n_time}")
        logger.info(f"  Traces: {self._n_traces:,}")
        logger.info(f"  Max fold: {self._stats['max_fold']:.0f}")

        return metadata

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        return self.output_dir


def create_migration_output_storage(
    output_dir: str,
    n_time: int,
    n_inline: int,
    n_xline: int,
    dt_ms: float = 4.0,
    **kwargs
) -> MigrationOutputStorage:
    """
    Factory function to create and initialize migration output storage.

    Args:
        output_dir: Output directory path
        n_time: Number of time samples
        n_inline: Number of inlines
        n_xline: Number of crosslines
        dt_ms: Sample interval in milliseconds
        **kwargs: Additional arguments passed to initialize_output

    Returns:
        Initialized MigrationOutputStorage
    """
    storage = MigrationOutputStorage(output_dir)
    storage.initialize_output(
        n_time=n_time,
        n_inline=n_inline,
        n_xline=n_xline,
        dt_ms=dt_ms,
        **kwargs
    )
    return storage
