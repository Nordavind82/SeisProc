"""
FKK Filter with Coordinate-Based Volume Building

High-level processor that combines:
1. Coordinate-based volume building (CDP_X/CDP_Y with user bin size)
2. 3D FKK velocity cone filtering
3. Multi-fold trace reconstruction

Handles the complete workflow from 2D traces to filtered 2D traces,
with proper handling of multiple traces per bin.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple
import logging

from models.seismic_data import SeismicData
from models.seismic_volume import SeismicVolume
from models.fkk_config import FKKConfig
from processors.fkk_filter_gpu import get_fkk_filter, FKKFilterGPU, FKKFilterCPU
from utils.coordinate_volume_builder import (
    CoordinateVolumeBuilder,
    BinningConfig,
    BinningGeometry,
    ReconstructionMethod,
    RepresentativeMethod,
    estimate_grid_from_coordinates
)

logger = logging.getLogger(__name__)


@dataclass
class FKKCoordinateConfig:
    """Combined configuration for coordinate-based FKK filtering."""

    # Binning parameters
    bin_size_x: float = 25.0
    bin_size_y: float = 25.0
    origin_x: Optional[float] = None
    origin_y: Optional[float] = None
    rotation_deg: float = 0.0

    # Coordinate headers
    coord_x_key: str = 'CDP_X'
    coord_y_key: str = 'CDP_Y'

    # Multi-fold handling
    representative_method: str = 'median'  # 'mean', 'median', 'first', 'nearest'
    reconstruction_method: str = 'noise_subtract'  # 'noise_subtract', 'residual_preserve', 'multi_pass'

    # FKK filter parameters (separate FKKConfig can also be passed)
    fkk_config: Optional[FKKConfig] = None

    # Processing options
    prefer_gpu: bool = True

    def to_binning_config(self) -> BinningConfig:
        """Convert to BinningConfig for volume builder."""
        return BinningConfig(
            bin_size_x=self.bin_size_x,
            bin_size_y=self.bin_size_y,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            rotation_deg=self.rotation_deg,
            coord_x_key=self.coord_x_key,
            coord_y_key=self.coord_y_key,
            representative_method=RepresentativeMethod(self.representative_method),
            reconstruction_method=ReconstructionMethod(self.reconstruction_method),
        )


class FKKCoordinateFilter:
    """
    Apply 3D FKK filter to 2D traces using coordinate-based volume building.

    This processor handles the complete workflow:
    1. Build 3D volume from CDP coordinates with specified bin size
    2. Apply FKK velocity cone filter
    3. Reconstruct filtered traces (handling multi-fold bins)

    Three reconstruction methods available:
    - noise_subtract: Fast, good for preview/design stage
    - residual_preserve: Medium, preserves per-trace differences
    - multi_pass: Most accurate, filters each trace individually

    Usage:
        filter = FKKCoordinateFilter(coord_config, fkk_config)
        result = filter.process(seismic_data)
        # or step by step:
        filter.build_volume(traces, headers, sample_rate_ms)
        filter.apply_filter()
        filtered_traces = filter.reconstruct_traces()
    """

    def __init__(
        self,
        coord_config: FKKCoordinateConfig,
        fkk_config: Optional[FKKConfig] = None
    ):
        """
        Initialize filter.

        Args:
            coord_config: Coordinate binning and reconstruction configuration
            fkk_config: FKK filter parameters (or use coord_config.fkk_config)
        """
        self.coord_config = coord_config
        self.fkk_config = fkk_config or coord_config.fkk_config or FKKConfig()

        # Initialize components
        self.volume_builder = CoordinateVolumeBuilder(coord_config.to_binning_config())
        self.fkk_filter = get_fkk_filter(prefer_gpu=coord_config.prefer_gpu)

        # State
        self.volume: Optional[SeismicVolume] = None
        self.filtered_volume: Optional[SeismicVolume] = None
        self._sample_rate_ms: float = 2.0

    def process(
        self,
        data: SeismicData,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> SeismicData:
        """
        Apply complete FKK filtering workflow to SeismicData.

        Args:
            data: Input seismic data with traces and headers
            progress_callback: Optional progress callback(percent, message)

        Returns:
            Filtered SeismicData with same organization as input
        """
        if data.headers is None:
            raise ValueError("SeismicData must have headers with CDP coordinates")

        # Convert headers to DataFrame if needed
        if isinstance(data.headers, dict):
            headers_df = pd.DataFrame(data.headers)
        else:
            headers_df = data.headers

        sample_rate_ms = data.sample_rate * 1000  # Convert to ms

        logger.info(f"FKK Coordinate Filter: {data.traces.shape[1]} traces, "
                   f"bin_size={self.coord_config.bin_size_x}x{self.coord_config.bin_size_y}m")

        # Step 1: Build volume
        if progress_callback:
            progress_callback(5, "Building 3D volume from coordinates...")

        self.build_volume(data.traces, headers_df, sample_rate_ms,
                         progress_callback=self._wrap_callback(progress_callback, 5, 30))

        # Step 2: Apply FKK filter
        if progress_callback:
            progress_callback(35, "Applying FKK filter...")

        self.apply_filter(progress_callback=self._wrap_callback(progress_callback, 35, 70))

        # Step 3: Reconstruct traces
        if progress_callback:
            progress_callback(75, "Reconstructing traces...")

        filtered_traces = self.reconstruct_traces(
            progress_callback=self._wrap_callback(progress_callback, 75, 100)
        )

        if progress_callback:
            progress_callback(100, "Complete")

        # Build result
        stats = self.volume_builder.get_statistics()
        return SeismicData(
            traces=filtered_traces,
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata={
                **data.metadata,
                'fkk_coordinate_filter': True,
                'fkk_config': self.fkk_config.get_summary(),
                'binning': stats,
            }
        )

    def build_volume(
        self,
        traces: np.ndarray,
        headers_df: pd.DataFrame,
        sample_rate_ms: float,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> SeismicVolume:
        """
        Build 3D volume from 2D traces using coordinates.

        Args:
            traces: 2D array (n_samples, n_traces)
            headers_df: DataFrame with CDP_X, CDP_Y coordinates
            sample_rate_ms: Sample rate in milliseconds
            progress_callback: Optional progress callback

        Returns:
            SeismicVolume ready for filtering
        """
        self._sample_rate_ms = sample_rate_ms
        self.volume = self.volume_builder.build(
            traces, headers_df, sample_rate_ms, progress_callback
        )
        return self.volume

    def apply_filter(
        self,
        fkk_config: Optional[FKKConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> SeismicVolume:
        """
        Apply FKK filter to the built volume.

        Args:
            fkk_config: Optional FKK config override
            progress_callback: Optional progress callback

        Returns:
            Filtered SeismicVolume
        """
        if self.volume is None:
            raise ValueError("No volume built. Call build_volume() first.")

        config = fkk_config or self.fkk_config

        if progress_callback:
            progress_callback(0, f"Applying FKK: {config.get_summary()}")

        self.filtered_volume = self.fkk_filter.apply_filter(self.volume, config)

        if progress_callback:
            progress_callback(100, "FKK filter applied")

        return self.filtered_volume

    def reconstruct_traces(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Reconstruct filtered traces from filtered volume.

        Uses the reconstruction method specified in config:
        - noise_subtract: Fast, subtracts common noise model
        - residual_preserve: Adds back per-trace residuals
        - multi_pass: Runs filter multiple times (most accurate)

        Returns:
            Filtered traces array (n_samples, n_traces)
        """
        if self.filtered_volume is None:
            raise ValueError("No filtered volume. Call apply_filter() first.")

        # For multi_pass, we need to provide the filter function
        filter_func = None
        if self.coord_config.reconstruction_method == 'multi_pass':
            filter_func = lambda vol: self.fkk_filter.apply_filter(vol, self.fkk_config)

        return self.volume_builder.reconstruct_traces(
            self.filtered_volume,
            filter_func=filter_func,
            progress_callback=progress_callback
        )

    def get_fold_volume(self) -> Optional[np.ndarray]:
        """Get fold count per bin."""
        return self.volume_builder.get_fold_volume()

    def get_geometry(self) -> Optional[BinningGeometry]:
        """Get binning geometry."""
        return self.volume_builder.get_geometry()

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.volume_builder.get_statistics()
        stats['fkk_config'] = self.fkk_config.get_summary()
        return stats

    def compute_spectrum(self) -> Optional[np.ndarray]:
        """Compute FKK spectrum for visualization."""
        if self.volume is None:
            return None
        return self.fkk_filter.compute_spectrum(self.volume)

    def get_mask_slices(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get filter mask slices for visualization."""
        if self.volume is None:
            return None
        return self.fkk_filter.get_mask_slices(self.volume, self.fkk_config)

    @staticmethod
    def _wrap_callback(
        callback: Optional[Callable[[float, str], None]],
        start_pct: float,
        end_pct: float
    ) -> Optional[Callable[[float, str], None]]:
        """Wrap callback to map 0-100 to start_pct-end_pct range."""
        if callback is None:
            return None

        def wrapped(pct: float, msg: str):
            mapped_pct = start_pct + (pct / 100.0) * (end_pct - start_pct)
            callback(mapped_pct, msg)

        return wrapped

    @staticmethod
    def estimate_parameters(
        headers_df: pd.DataFrame,
        coord_x_key: str = 'CDP_X',
        coord_y_key: str = 'CDP_Y'
    ) -> Dict[str, Any]:
        """
        Estimate optimal binning parameters from data.

        Args:
            headers_df: DataFrame with coordinate headers
            coord_x_key: X coordinate column name
            coord_y_key: Y coordinate column name

        Returns:
            Dictionary with suggested parameters
        """
        return estimate_grid_from_coordinates(headers_df, coord_x_key, coord_y_key)


def apply_fkk_coordinate_filter(
    traces: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    bin_size: float = 25.0,
    fkk_config: Optional[FKKConfig] = None,
    reconstruction_method: str = 'noise_subtract',
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to apply FKK filter with coordinate binning.

    Args:
        traces: 2D array (n_samples, n_traces)
        headers_df: DataFrame with CDP_X, CDP_Y coordinates
        sample_rate_ms: Sample rate in milliseconds
        bin_size: Bin size in meters (used for both X and Y)
        fkk_config: FKK filter configuration
        reconstruction_method: 'noise_subtract', 'residual_preserve', or 'multi_pass'
        progress_callback: Optional progress callback

    Returns:
        Tuple of (filtered_traces, statistics_dict)
    """
    coord_config = FKKCoordinateConfig(
        bin_size_x=bin_size,
        bin_size_y=bin_size,
        reconstruction_method=reconstruction_method,
    )

    filter_obj = FKKCoordinateFilter(coord_config, fkk_config)
    filter_obj.build_volume(traces, headers_df, sample_rate_ms)
    filter_obj.apply_filter()
    filtered_traces = filter_obj.reconstruct_traces(progress_callback)

    return filtered_traces, filter_obj.get_statistics()


# Design mode shortcut - fastest method
def apply_fkk_design_mode(
    traces: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    bin_size: float = 25.0,
    fkk_config: Optional[FKKConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply FKK filter in design mode (fastest, for preview).

    Uses noise_subtract reconstruction for speed.
    """
    return apply_fkk_coordinate_filter(
        traces, headers_df, sample_rate_ms,
        bin_size=bin_size,
        fkk_config=fkk_config,
        reconstruction_method='noise_subtract'
    )


# Application mode shortcut - most accurate
def apply_fkk_application_mode(
    traces: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    bin_size: float = 25.0,
    fkk_config: Optional[FKKConfig] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply FKK filter in application mode (most accurate).

    Uses multi_pass reconstruction for best quality.
    """
    return apply_fkk_coordinate_filter(
        traces, headers_df, sample_rate_ms,
        bin_size=bin_size,
        fkk_config=fkk_config,
        reconstruction_method='multi_pass',
        progress_callback=progress_callback
    )
