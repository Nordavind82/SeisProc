"""
Coordinate-Based 3D Volume Builder

Builds 3D volumes from 2D traces using CDP coordinates with user-specified bin size.
Handles multi-fold bins (multiple traces per bin) with three reconstruction strategies:

1. noise_subtract (fastest): Build representative, filter, subtract noise model from all traces
2. residual_preserve: Store per-trace residuals, add back after filtering representative
3. multi_pass (most accurate): Run filter N times for max fold N, each trace filtered individually

Design stage uses noise_subtract for speed. Application stage can use multi_pass for accuracy.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Literal, Callable
from enum import Enum
import logging

from models.seismic_volume import SeismicVolume

logger = logging.getLogger(__name__)


class RepresentativeMethod(str, Enum):
    """Method to compute bin representative from multiple traces."""
    MEAN = 'mean'
    MEDIAN = 'median'
    FIRST = 'first'
    NEAREST = 'nearest'  # Nearest to bin center


class ReconstructionMethod(str, Enum):
    """Method to reconstruct filtered traces from filtered volume."""
    NOISE_SUBTRACT = 'noise_subtract'      # Fastest: subtract common noise model
    RESIDUAL_PRESERVE = 'residual_preserve'  # Medium: add back per-trace residuals
    MULTI_PASS = 'multi_pass'              # Accurate: filter each trace individually


@dataclass
class BinningConfig:
    """Configuration for coordinate-based volume binning."""

    # Grid definition
    bin_size_x: float = 25.0       # Bin size in X direction (meters)
    bin_size_y: float = 25.0       # Bin size in Y direction (meters)
    origin_x: Optional[float] = None  # Grid origin X (None = auto from data min)
    origin_y: Optional[float] = None  # Grid origin Y (None = auto from data min)
    rotation_deg: float = 0.0      # Grid rotation in degrees (0 = X-aligned)

    # Coordinate headers
    coord_x_key: str = 'CDP_X'
    coord_y_key: str = 'CDP_Y'
    coord_scalar_key: str = 'coordinate_scalar'

    # Multi-fold handling
    representative_method: RepresentativeMethod = RepresentativeMethod.MEDIAN
    reconstruction_method: ReconstructionMethod = ReconstructionMethod.NOISE_SUBTRACT

    # QC options
    min_fold: int = 1              # Minimum fold to include bin (bins with less are zeroed)

    def __post_init__(self):
        """Validate configuration."""
        if self.bin_size_x <= 0:
            raise ValueError(f"bin_size_x must be positive, got {self.bin_size_x}")
        if self.bin_size_y <= 0:
            raise ValueError(f"bin_size_y must be positive, got {self.bin_size_y}")
        if self.min_fold < 1:
            raise ValueError(f"min_fold must be >= 1, got {self.min_fold}")

        # Convert string to enum if needed
        if isinstance(self.representative_method, str):
            self.representative_method = RepresentativeMethod(self.representative_method)
        if isinstance(self.reconstruction_method, str):
            self.reconstruction_method = ReconstructionMethod(self.reconstruction_method)


@dataclass
class BinningGeometry:
    """Geometry information for coordinate-based binned volume."""

    # Grid parameters
    origin_x: float
    origin_y: float
    bin_size_x: float
    bin_size_y: float
    n_bins_x: int
    n_bins_y: int
    rotation_deg: float

    # Sample info
    n_samples: int
    dt: float  # seconds

    # Statistics
    total_traces: int = 0
    traces_binned: int = 0
    bins_populated: int = 0
    max_fold: int = 0
    mean_fold: float = 0.0

    # Rotation matrix (computed)
    _rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    _inv_rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))

    def __post_init__(self):
        """Compute rotation matrices."""
        theta = np.radians(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        self._rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        self._inv_rotation_matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    def world_to_bin(self, x: float, y: float) -> Tuple[int, int, float]:
        """
        Convert world coordinates to bin indices.

        Returns:
            (bin_ix, bin_iy, distance_to_bin_center)
        """
        # Translate to origin
        dx = x - self.origin_x
        dy = y - self.origin_y

        # Rotate to grid-aligned coordinates
        rx, ry = self._rotation_matrix @ [dx, dy]

        # Compute bin indices
        ix = int(np.floor(rx / self.bin_size_x))
        iy = int(np.floor(ry / self.bin_size_y))

        # Distance to bin center
        center_rx = (ix + 0.5) * self.bin_size_x
        center_ry = (iy + 0.5) * self.bin_size_y
        distance = np.sqrt((rx - center_rx)**2 + (ry - center_ry)**2)

        return ix, iy, distance

    def bin_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert bin indices to world coordinates (bin center)."""
        # Bin center in grid coordinates
        rx = (ix + 0.5) * self.bin_size_x
        ry = (iy + 0.5) * self.bin_size_y

        # Rotate back to world
        dx, dy = self._inv_rotation_matrix @ [rx, ry]

        return self.origin_x + dx, self.origin_y + dy

    @property
    def coverage_percent(self) -> float:
        """Percentage of bins that have data."""
        total_bins = self.n_bins_x * self.n_bins_y
        return (self.bins_populated / total_bins * 100) if total_bins > 0 else 0.0

    def get_summary(self) -> str:
        """Get one-line summary."""
        return (
            f"Grid: {self.n_bins_x}x{self.n_bins_y} bins @ {self.bin_size_x:.1f}x{self.bin_size_y:.1f}m | "
            f"Coverage: {self.coverage_percent:.1f}% | "
            f"Fold: max={self.max_fold}, mean={self.mean_fold:.1f}"
        )


class CoordinateVolumeBuilder:
    """
    Build 3D volume from CDP coordinates with multi-fold handling.

    Supports three reconstruction methods for handling multiple traces per bin:
    - noise_subtract: Fast, good for design/preview
    - residual_preserve: Medium, preserves per-trace differences
    - multi_pass: Accurate, filters each trace individually (N passes for max fold N)
    """

    def __init__(self, config: BinningConfig):
        """
        Initialize builder.

        Args:
            config: Binning configuration
        """
        self.config = config

        # State (populated during build)
        self.geometry: Optional[BinningGeometry] = None
        self.trace_to_bin: Dict[int, Tuple[int, int, float]] = {}  # trace_idx -> (ix, iy, distance)
        self.bin_to_traces: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}  # (ix,iy) -> [(trace_idx, dist), ...]
        self.representative_volume: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.fold_volume: Optional[np.ndarray] = None

        # Original data reference (needed for reconstruction)
        self._original_traces: Optional[np.ndarray] = None
        self._headers_df: Optional[pd.DataFrame] = None

    def build(
        self,
        traces: np.ndarray,
        headers_df: pd.DataFrame,
        sample_rate_ms: float,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> SeismicVolume:
        """
        Build 3D volume for filtering.

        Stores internal state needed for reconstruct_traces().

        Args:
            traces: 2D array (n_samples, n_traces)
            headers_df: DataFrame with coordinate headers
            sample_rate_ms: Sample rate in milliseconds
            progress_callback: Optional callback(percent, message)

        Returns:
            SeismicVolume ready for 3D filtering
        """
        n_samples, n_traces = traces.shape
        self._original_traces = traces
        self._headers_df = headers_df

        logger.info(f"Building coordinate-based volume: {n_traces} traces, {n_samples} samples")

        # 1. Extract coordinates
        if progress_callback:
            progress_callback(5, "Extracting coordinates...")
        coords_x, coords_y = self._extract_coordinates(headers_df)

        # 2. Define grid
        if progress_callback:
            progress_callback(10, "Defining grid...")
        self._define_grid(coords_x, coords_y, n_samples, sample_rate_ms)

        # 3. Assign traces to bins
        if progress_callback:
            progress_callback(15, "Assigning traces to bins...")
        self._assign_traces_to_bins(coords_x, coords_y, n_traces)

        # 4. Build representative volume
        if progress_callback:
            progress_callback(30, "Building volume...")
        volume_data = self._build_representative_volume(traces, progress_callback)

        # 5. Store residuals if needed
        if self.config.reconstruction_method == ReconstructionMethod.RESIDUAL_PRESERVE:
            if progress_callback:
                progress_callback(80, "Computing residuals...")
            self._compute_residuals(traces)

        if progress_callback:
            progress_callback(100, "Volume built")

        logger.info(f"Volume built: {self.geometry.get_summary()}")

        return SeismicVolume(
            data=volume_data,
            dt=self.geometry.dt,
            dx=self.config.bin_size_x,
            dy=self.config.bin_size_y,
            metadata={
                'built_from_coordinates': True,
                'binning_config': {
                    'bin_size_x': self.config.bin_size_x,
                    'bin_size_y': self.config.bin_size_y,
                    'origin_x': self.geometry.origin_x,
                    'origin_y': self.geometry.origin_y,
                    'rotation_deg': self.geometry.rotation_deg,
                },
                'max_fold': self.geometry.max_fold,
                'mean_fold': self.geometry.mean_fold,
            }
        )

    def _extract_coordinates(self, headers_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and scale coordinates from headers."""
        cfg = self.config

        # Find coordinate columns
        x_col = self._find_column(headers_df, cfg.coord_x_key, ['CDP_X', 'cdp_x', 'CDPX', 'cdpx'])
        y_col = self._find_column(headers_df, cfg.coord_y_key, ['CDP_Y', 'cdp_y', 'CDPY', 'cdpy'])

        if x_col is None or y_col is None:
            raise ValueError(
                f"Coordinate columns not found. Looking for X: {cfg.coord_x_key}, Y: {cfg.coord_y_key}. "
                f"Available: {list(headers_df.columns)[:15]}"
            )

        coords_x = headers_df[x_col].values.astype(np.float64)
        coords_y = headers_df[y_col].values.astype(np.float64)

        # Apply scalar if present
        scalar_col = self._find_column(headers_df, cfg.coord_scalar_key,
                                       ['coordinate_scalar', 'scalar_co', 'SourceGroupScalar'])
        if scalar_col and scalar_col in headers_df.columns:
            scalar = headers_df[scalar_col].iloc[0]
            if scalar < 0:
                coords_x = coords_x / abs(scalar)
                coords_y = coords_y / abs(scalar)
            elif scalar > 0:
                coords_x = coords_x * scalar
                coords_y = coords_y * scalar
            logger.info(f"Applied coordinate scalar: {scalar}")

        logger.info(f"Coordinates: X=[{coords_x.min():.1f}, {coords_x.max():.1f}], "
                   f"Y=[{coords_y.min():.1f}, {coords_y.max():.1f}]")

        return coords_x, coords_y

    def _find_column(self, df: pd.DataFrame, primary: str, alternatives: List[str]) -> Optional[str]:
        """Find column by primary name or alternatives."""
        if primary in df.columns:
            return primary
        for alt in alternatives:
            if alt in df.columns:
                return alt
        return None

    def _define_grid(
        self,
        coords_x: np.ndarray,
        coords_y: np.ndarray,
        n_samples: int,
        sample_rate_ms: float
    ):
        """Define grid geometry from coordinates."""
        cfg = self.config

        # Origin
        origin_x = cfg.origin_x if cfg.origin_x is not None else coords_x.min()
        origin_y = cfg.origin_y if cfg.origin_y is not None else coords_y.min()

        # Apply rotation to find extent
        theta = np.radians(cfg.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Transform all coordinates to rotated frame
        dx = coords_x - origin_x
        dy = coords_y - origin_y
        rotated = rot_matrix @ np.vstack([dx, dy])
        rx, ry = rotated[0], rotated[1]

        # Grid extent
        max_rx = rx.max() + cfg.bin_size_x  # Add one bin margin
        max_ry = ry.max() + cfg.bin_size_y

        n_bins_x = max(1, int(np.ceil(max_rx / cfg.bin_size_x)))
        n_bins_y = max(1, int(np.ceil(max_ry / cfg.bin_size_y)))

        self.geometry = BinningGeometry(
            origin_x=origin_x,
            origin_y=origin_y,
            bin_size_x=cfg.bin_size_x,
            bin_size_y=cfg.bin_size_y,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            rotation_deg=cfg.rotation_deg,
            n_samples=n_samples,
            dt=sample_rate_ms / 1000.0,
            total_traces=len(coords_x)
        )

        logger.info(f"Grid defined: {n_bins_x}x{n_bins_y} bins, origin=({origin_x:.1f}, {origin_y:.1f})")

    def _assign_traces_to_bins(self, coords_x: np.ndarray, coords_y: np.ndarray, n_traces: int):
        """Assign each trace to a bin based on coordinates."""
        self.trace_to_bin = {}
        self.bin_to_traces = {}

        for trace_idx in range(n_traces):
            x, y = coords_x[trace_idx], coords_y[trace_idx]
            ix, iy, distance = self.geometry.world_to_bin(x, y)

            # Clamp to valid range
            ix = max(0, min(ix, self.geometry.n_bins_x - 1))
            iy = max(0, min(iy, self.geometry.n_bins_y - 1))

            self.trace_to_bin[trace_idx] = (ix, iy, distance)

            key = (ix, iy)
            if key not in self.bin_to_traces:
                self.bin_to_traces[key] = []
            self.bin_to_traces[key].append((trace_idx, distance))

        # Sort traces in each bin by distance (nearest first)
        for key in self.bin_to_traces:
            self.bin_to_traces[key].sort(key=lambda x: x[1])

        # Update statistics
        folds = [len(traces) for traces in self.bin_to_traces.values()]
        self.geometry.traces_binned = sum(folds)
        self.geometry.bins_populated = len(self.bin_to_traces)
        self.geometry.max_fold = max(folds) if folds else 0
        self.geometry.mean_fold = np.mean(folds) if folds else 0.0

        logger.info(f"Traces assigned: {self.geometry.bins_populated} bins populated, "
                   f"max_fold={self.geometry.max_fold}, mean_fold={self.geometry.mean_fold:.1f}")

    def _build_representative_volume(
        self,
        traces: np.ndarray,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """Build volume with representative trace per bin."""
        n_samples = self.geometry.n_samples
        n_bins_x = self.geometry.n_bins_x
        n_bins_y = self.geometry.n_bins_y

        volume_data = np.zeros((n_samples, n_bins_x, n_bins_y), dtype=np.float32)
        self.fold_volume = np.zeros((n_bins_x, n_bins_y), dtype=np.int32)

        method = self.config.representative_method
        total_bins = len(self.bin_to_traces)

        for i, ((ix, iy), trace_list) in enumerate(self.bin_to_traces.items()):
            fold = len(trace_list)

            if fold < self.config.min_fold:
                continue

            self.fold_volume[ix, iy] = fold
            trace_indices = [t[0] for t in trace_list]
            bin_traces = traces[:, trace_indices]  # (n_samples, fold)

            if method == RepresentativeMethod.MEDIAN:
                representative = np.median(bin_traces, axis=1)
            elif method == RepresentativeMethod.MEAN:
                representative = np.mean(bin_traces, axis=1)
            elif method == RepresentativeMethod.FIRST:
                representative = bin_traces[:, 0]
            elif method == RepresentativeMethod.NEAREST:
                representative = bin_traces[:, 0]  # Already sorted by distance
            else:
                representative = np.median(bin_traces, axis=1)

            volume_data[:, ix, iy] = representative

            if progress_callback and i % 100 == 0:
                pct = 30 + (i / total_bins) * 50
                progress_callback(pct, f"Building volume: {i}/{total_bins} bins")

        self.representative_volume = volume_data
        return volume_data

    def _compute_residuals(self, traces: np.ndarray):
        """Compute per-trace residuals from representative."""
        n_samples, n_traces = traces.shape
        self.residuals = np.zeros_like(traces)

        for trace_idx in range(n_traces):
            ix, iy, _ = self.trace_to_bin[trace_idx]
            self.residuals[:, trace_idx] = traces[:, trace_idx] - self.representative_volume[:, ix, iy]

    def reconstruct_traces(
        self,
        filtered_volume: SeismicVolume,
        filter_func: Optional[Callable[[SeismicVolume], SeismicVolume]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Reconstruct all original traces from filtered volume.

        For multi_pass method, filter_func must be provided to run additional passes.

        Args:
            filtered_volume: Result of 3D filter applied to representative volume
            filter_func: Function to apply 3D filter (required for multi_pass method)
            progress_callback: Optional callback(percent, message)

        Returns:
            filtered_traces: Same shape as original traces (n_samples, n_traces)
        """
        if self._original_traces is None:
            raise ValueError("No original traces stored. Call build() first.")

        method = self.config.reconstruction_method

        if method == ReconstructionMethod.NOISE_SUBTRACT:
            return self._reconstruct_noise_subtract(filtered_volume, progress_callback)
        elif method == ReconstructionMethod.RESIDUAL_PRESERVE:
            return self._reconstruct_residual_preserve(filtered_volume, progress_callback)
        elif method == ReconstructionMethod.MULTI_PASS:
            if filter_func is None:
                raise ValueError("filter_func required for multi_pass reconstruction")
            return self._reconstruct_multi_pass(filtered_volume, filter_func, progress_callback)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

    def _reconstruct_noise_subtract(
        self,
        filtered_volume: SeismicVolume,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Reconstruct using common noise model subtraction.

        Fastest method - single pass, noise is assumed same for all traces in bin.
        """
        if progress_callback:
            progress_callback(0, "Reconstructing traces (noise subtract)...")

        original = self._original_traces
        n_samples, n_traces = original.shape
        filtered_traces = np.zeros_like(original)

        # Noise model = what the filter removed
        noise_model = self.representative_volume - filtered_volume.data

        for trace_idx in range(n_traces):
            ix, iy, _ = self.trace_to_bin[trace_idx]
            filtered_traces[:, trace_idx] = original[:, trace_idx] - noise_model[:, ix, iy]

            if progress_callback and trace_idx % 1000 == 0:
                pct = (trace_idx / n_traces) * 100
                progress_callback(pct, f"Reconstructing: {trace_idx}/{n_traces}")

        if progress_callback:
            progress_callback(100, "Reconstruction complete")

        logger.info(f"Reconstructed {n_traces} traces using noise_subtract method")
        return filtered_traces

    def _reconstruct_residual_preserve(
        self,
        filtered_volume: SeismicVolume,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Reconstruct by adding back per-trace residuals.

        Preserves individual trace differences from representative.
        """
        if self.residuals is None:
            raise ValueError("Residuals not computed. Ensure reconstruction_method was RESIDUAL_PRESERVE during build.")

        if progress_callback:
            progress_callback(0, "Reconstructing traces (residual preserve)...")

        n_samples, n_traces = self._original_traces.shape
        filtered_traces = np.zeros_like(self._original_traces)

        for trace_idx in range(n_traces):
            ix, iy, _ = self.trace_to_bin[trace_idx]
            filtered_traces[:, trace_idx] = filtered_volume.data[:, ix, iy] + self.residuals[:, trace_idx]

            if progress_callback and trace_idx % 1000 == 0:
                pct = (trace_idx / n_traces) * 100
                progress_callback(pct, f"Reconstructing: {trace_idx}/{n_traces}")

        if progress_callback:
            progress_callback(100, "Reconstruction complete")

        logger.info(f"Reconstructed {n_traces} traces using residual_preserve method")
        return filtered_traces

    def _reconstruct_multi_pass(
        self,
        first_filtered_volume: SeismicVolume,
        filter_func: Callable[[SeismicVolume], SeismicVolume],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Reconstruct using multiple filter passes.

        Most accurate method - runs filter N times for max fold N.
        Each pass uses a different trace from each multi-fold bin.
        Pass i: uses trace i from bins with fold >= i

        Args:
            first_filtered_volume: Already-filtered volume from pass 1
            filter_func: Function to apply filter to a SeismicVolume
            progress_callback: Progress callback
        """
        original = self._original_traces
        n_samples, n_traces = original.shape
        filtered_traces = np.zeros_like(original)
        max_fold = self.geometry.max_fold

        logger.info(f"Multi-pass reconstruction: {max_fold} passes for max fold {max_fold}")

        # Track which traces have been filtered
        trace_filtered = np.zeros(n_traces, dtype=bool)

        for pass_num in range(max_fold):
            if progress_callback:
                pct = (pass_num / max_fold) * 100
                progress_callback(pct, f"Pass {pass_num + 1}/{max_fold}")

            logger.info(f"Multi-pass: Pass {pass_num + 1}/{max_fold}")

            if pass_num == 0:
                # First pass already done
                filtered_vol = first_filtered_volume
            else:
                # Build volume using trace at index pass_num from each bin
                volume_data = np.zeros((n_samples, self.geometry.n_bins_x, self.geometry.n_bins_y),
                                       dtype=np.float32)

                for (ix, iy), trace_list in self.bin_to_traces.items():
                    if len(trace_list) > pass_num:
                        # Use trace at position pass_num
                        trace_idx = trace_list[pass_num][0]
                        volume_data[:, ix, iy] = original[:, trace_idx]
                    elif len(trace_list) > 0:
                        # Bin has fewer traces, use last available
                        # (keeps spatial continuity for filter)
                        trace_idx = trace_list[-1][0]
                        volume_data[:, ix, iy] = original[:, trace_idx]

                # Apply filter
                pass_volume = SeismicVolume(
                    data=volume_data,
                    dt=self.geometry.dt,
                    dx=self.config.bin_size_x,
                    dy=self.config.bin_size_y
                )
                filtered_vol = filter_func(pass_volume)

            # Extract filtered traces for this pass
            for (ix, iy), trace_list in self.bin_to_traces.items():
                if len(trace_list) > pass_num:
                    trace_idx = trace_list[pass_num][0]
                    if not trace_filtered[trace_idx]:
                        filtered_traces[:, trace_idx] = filtered_vol.data[:, ix, iy]
                        trace_filtered[trace_idx] = True

        # Verify all traces filtered
        n_filtered = trace_filtered.sum()
        if n_filtered != n_traces:
            logger.warning(f"Only {n_filtered}/{n_traces} traces were filtered in multi-pass")

        if progress_callback:
            progress_callback(100, f"Multi-pass complete: {max_fold} passes")

        logger.info(f"Reconstructed {n_filtered} traces using multi_pass method ({max_fold} passes)")
        return filtered_traces

    def get_fold_volume(self) -> Optional[np.ndarray]:
        """Get fold count per bin (n_bins_x, n_bins_y)."""
        return self.fold_volume

    def get_geometry(self) -> Optional[BinningGeometry]:
        """Get binning geometry information."""
        return self.geometry

    def get_statistics(self) -> Dict[str, Any]:
        """Get binning statistics."""
        if self.geometry is None:
            return {}

        return {
            'n_bins_x': self.geometry.n_bins_x,
            'n_bins_y': self.geometry.n_bins_y,
            'bin_size_x': self.geometry.bin_size_x,
            'bin_size_y': self.geometry.bin_size_y,
            'total_traces': self.geometry.total_traces,
            'traces_binned': self.geometry.traces_binned,
            'bins_populated': self.geometry.bins_populated,
            'coverage_percent': self.geometry.coverage_percent,
            'max_fold': self.geometry.max_fold,
            'mean_fold': self.geometry.mean_fold,
            'reconstruction_method': self.config.reconstruction_method.value,
        }


def estimate_grid_from_coordinates(
    headers_df: pd.DataFrame,
    coord_x_key: str = 'CDP_X',
    coord_y_key: str = 'CDP_Y'
) -> Dict[str, Any]:
    """
    Estimate optimal grid parameters from coordinate distribution.

    Analyzes coordinate spacing to suggest bin sizes.

    Args:
        headers_df: DataFrame with coordinate headers
        coord_x_key: X coordinate column name
        coord_y_key: Y coordinate column name

    Returns:
        Dictionary with suggested parameters
    """
    # Find columns
    x_col = coord_x_key if coord_x_key in headers_df.columns else None
    y_col = coord_y_key if coord_y_key in headers_df.columns else None

    if x_col is None or y_col is None:
        for alt in ['CDP_X', 'cdp_x', 'CDPX']:
            if alt in headers_df.columns:
                x_col = alt
                break
        for alt in ['CDP_Y', 'cdp_y', 'CDPY']:
            if alt in headers_df.columns:
                y_col = alt
                break

    if x_col is None or y_col is None:
        return {'error': 'Coordinate columns not found'}

    coords_x = headers_df[x_col].values.astype(np.float64)
    coords_y = headers_df[y_col].values.astype(np.float64)

    # Estimate spacing from nearest neighbor distances
    from scipy.spatial import cKDTree

    coords = np.column_stack([coords_x, coords_y])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest_distances = distances[:, 1]  # Exclude self

    # Filter outliers
    valid_distances = nearest_distances[nearest_distances > 0.1]
    if len(valid_distances) == 0:
        valid_distances = nearest_distances

    median_spacing = np.median(valid_distances)
    p10_spacing = np.percentile(valid_distances, 10)
    p90_spacing = np.percentile(valid_distances, 90)

    # Suggested bin size: slightly larger than median spacing
    suggested_bin_size = median_spacing * 1.1

    return {
        'coord_x_key': x_col,
        'coord_y_key': y_col,
        'x_range': (coords_x.min(), coords_x.max()),
        'y_range': (coords_y.min(), coords_y.max()),
        'n_traces': len(coords_x),
        'median_spacing': median_spacing,
        'p10_spacing': p10_spacing,
        'p90_spacing': p90_spacing,
        'suggested_bin_size': suggested_bin_size,
        'suggested_origin_x': coords_x.min(),
        'suggested_origin_y': coords_y.min(),
    }
