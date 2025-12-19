"""
3D Spatial Denoising Processor.

Implements 3D spatial statistics for robust noise estimation by:
1. Building a 3D volume from 2D gather using CDP coordinates or headers
2. Computing spatial noise statistics in 3D (inline x crossline apertures)
3. Applying DWT-based denoising with 3D MAD thresholding
4. Extracting results back to original 2D gather organization

Supports two volume building modes:
- Coordinate-based: Uses CDP_X/CDP_Y with user-specified bin size (handles multi-fold)
- Header-based: Uses header keys like field_record/trace_number (legacy mode)

Multi-fold handling (when multiple traces fall into same bin):
- noise_subtract: Fast, compute noise model from representative and subtract from all traces
- residual_preserve: Store per-trace residuals, add back after denoising
- multi_pass: Most accurate, denoise each trace individually (N passes for max fold N)

Performance: Uses Numba JIT compilation for 5-10x faster 3D MAD computation.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
import logging
import pywt

# Try to import numba for JIT acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from models.seismic_data import SeismicData
from models.seismic_volume import SeismicVolume
from processors.base_processor import BaseProcessor
from utils.coordinate_volume_builder import (
    CoordinateVolumeBuilder,
    BinningConfig,
    BinningGeometry,
    ReconstructionMethod,
    RepresentativeMethod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Numba JIT-compiled core functions for 5-10x speedup
# =============================================================================

@jit(nopython=True, parallel=False, cache=True)
def _compute_3d_mad_numba(data_3d: np.ndarray,
                          half_il: int,
                          half_xl: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba JIT-compiled 3D MAD computation core.

    Note: parallel=False to avoid thread contention with multi-process workers.
    The parallelism happens at the process level (multiple workers), not thread level.

    This provides 5-10x speedup over pure Python implementation.

    Args:
        data_3d: 3D array (n_samples, n_inlines, n_xlines), float32
        half_il: Half aperture in inline direction
        half_xl: Half aperture in crossline direction

    Returns:
        Tuple of (median_3d, mad_3d) with same shape as input
    """
    n_samples, n_inlines, n_xlines = data_3d.shape

    median_3d = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)
    mad_3d = np.zeros((n_samples, n_inlines, n_xlines), dtype=np.float32)

    # Pre-allocate work buffer for patch values (max possible patch size)
    max_patch_size = (2 * half_il + 1) * (2 * half_xl + 1)

    for il in range(n_inlines):
        il_start = max(0, il - half_il)
        il_end = min(n_inlines, il + half_il + 1)

        for xl in range(n_xlines):
            xl_start = max(0, xl - half_xl)
            xl_end = min(n_xlines, xl + half_xl + 1)

            # Actual patch size for this position
            patch_il_size = il_end - il_start
            patch_xl_size = xl_end - xl_start
            patch_size = patch_il_size * patch_xl_size

            # Process each time sample
            for t in range(n_samples):
                # Flatten spatial patch into work buffer
                values = np.empty(patch_size, dtype=np.float32)
                idx = 0
                for i in range(il_start, il_end):
                    for j in range(xl_start, xl_end):
                        values[idx] = data_3d[t, i, j]
                        idx += 1

                # Compute median using sort (Numba-compatible)
                values_sorted = np.sort(values)
                mid = patch_size // 2
                if patch_size % 2 == 0:
                    med = (values_sorted[mid - 1] + values_sorted[mid]) * 0.5
                else:
                    med = values_sorted[mid]

                # Compute MAD (median absolute deviation)
                for k in range(patch_size):
                    values[k] = np.abs(values[k] - med)
                values_sorted = np.sort(values)
                if patch_size % 2 == 0:
                    mad = (values_sorted[mid - 1] + values_sorted[mid]) * 0.5
                else:
                    mad = values_sorted[mid]

                median_3d[t, il, xl] = med
                mad_3d[t, il, xl] = mad

    return median_3d, mad_3d


def _compute_3d_mad_python(data_3d: np.ndarray,
                           half_il: int,
                           half_xl: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure Python/NumPy fallback for 3D MAD computation.

    Used when Numba is not available.
    """
    n_samples, n_inlines, n_xlines = data_3d.shape

    median_3d = np.zeros_like(data_3d)
    mad_3d = np.zeros_like(data_3d)

    for il in range(n_inlines):
        il_start = max(0, il - half_il)
        il_end = min(n_inlines, il + half_il + 1)

        for xl in range(n_xlines):
            xl_start = max(0, xl - half_xl)
            xl_end = min(n_xlines, xl + half_xl + 1)

            # Extract spatial patch for all time samples
            patch = data_3d[:, il_start:il_end, xl_start:xl_end]

            # Reshape to (n_samples, patch_size) for per-time-sample statistics
            patch_flat = patch.reshape(n_samples, -1)

            # Compute median and MAD per time sample
            med = np.median(patch_flat, axis=1)
            mad = np.median(np.abs(patch_flat - med[:, np.newaxis]), axis=1)

            median_3d[:, il, xl] = med
            mad_3d[:, il, xl] = mad

    return median_3d, mad_3d


def compute_3d_mad(data_3d: np.ndarray,
                   aperture_inline: int = 3,
                   aperture_xline: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D spatial MAD statistics using a 2D spatial aperture.

    For each time sample and each (inline, xline) position, computes
    the median and MAD over the spatial aperture neighborhood.

    Uses Numba JIT compilation for 5-10x speedup when available.
    Falls back to pure NumPy implementation otherwise.

    Args:
        data_3d: 3D array (n_samples, n_inlines, n_xlines)
        aperture_inline: Aperture size in inline direction (odd)
        aperture_xline: Aperture size in crossline direction (odd)

    Returns:
        Tuple of (median_3d, mad_3d) with same shape as input
    """
    half_il = aperture_inline // 2
    half_xl = aperture_xline // 2

    # Ensure float32 for memory efficiency and Numba compatibility
    if data_3d.dtype != np.float32:
        data_3d = data_3d.astype(np.float32)

    if NUMBA_AVAILABLE:
        try:
            return _compute_3d_mad_numba(data_3d, half_il, half_xl)
        except Exception as e:
            logger.warning(f"Numba JIT failed, falling back to Python: {e}")
            return _compute_3d_mad_python(data_3d, half_il, half_xl)
    else:
        return _compute_3d_mad_python(data_3d, half_il, half_xl)


def get_available_headers(headers_df: pd.DataFrame, min_unique: int = 2) -> List[str]:
    """
    Get list of headers suitable for 3D volume building.

    Args:
        headers_df: DataFrame with trace headers
        min_unique: Minimum number of unique values required

    Returns:
        List of header names sorted by unique value count (descending)
    """
    if headers_df is None or headers_df.empty:
        return []

    available = []
    for col in headers_df.columns:
        if pd.api.types.is_numeric_dtype(headers_df[col]):
            n_unique = headers_df[col].nunique()
            if n_unique >= min_unique:
                available.append((col, n_unique))

    # Sort by unique count descending
    available.sort(key=lambda x: -x[1])

    return [h for h, _ in available]


@dataclass
class Denoise3DConfig:
    """Configuration for 3D spatial denoising."""

    # Volume building mode
    use_coordinates: bool = True  # True = CDP coordinates, False = header keys

    # Coordinate-based parameters (when use_coordinates=True)
    bin_size_x: float = 25.0
    bin_size_y: float = 25.0
    coord_x_key: str = 'CDP_X'
    coord_y_key: str = 'CDP_Y'

    # Header-based parameters (when use_coordinates=False)
    inline_key: str = 'field_record'
    xline_key: str = 'trace_number'

    # Multi-fold handling
    representative_method: str = 'median'  # 'mean', 'median', 'first', 'nearest'
    reconstruction_method: str = 'noise_subtract'  # 'noise_subtract', 'residual_preserve', 'multi_pass'

    # Spatial aperture
    aperture_inline: int = 3
    aperture_xline: int = 3

    # DWT parameters
    wavelet: str = 'db4'
    level: Optional[int] = None
    threshold_k: float = 3.0
    threshold_mode: str = 'soft'  # 'soft' or 'hard'


class Denoise3D(BaseProcessor):
    """
    3D Spatial Denoising using DWT with 3D MAD statistics.

    Supports two volume building modes:
    1. Coordinate-based: Uses CDP_X/CDP_Y with bin size (handles multi-fold bins)
    2. Header-based: Uses header keys for inline/xline axes (legacy mode)

    Multi-fold reconstruction methods:
    - noise_subtract: Fast, for design/preview
    - residual_preserve: Medium, preserves per-trace differences
    - multi_pass: Accurate, each trace denoised individually
    """

    def __init__(
        self,
        # Volume building mode
        use_coordinates: bool = False,
        # Coordinate-based params
        bin_size_x: float = 25.0,
        bin_size_y: float = 25.0,
        coord_x_key: str = 'CDP_X',
        coord_y_key: str = 'CDP_Y',
        # Header-based params (legacy)
        inline_key: str = 'field_record',
        xline_key: str = 'trace_number',
        # Multi-fold handling
        representative_method: str = 'median',
        reconstruction_method: str = 'noise_subtract',
        # Spatial aperture
        aperture_inline: int = 3,
        aperture_xline: int = 3,
        # DWT params
        wavelet: str = 'db4',
        level: Optional[int] = None,
        threshold_k: float = 3.0,
        threshold_mode: Literal['soft', 'hard'] = 'soft'
    ):
        """
        Initialize 3D Denoise processor.

        Args:
            use_coordinates: True = use CDP coordinates with binning, False = use headers
            bin_size_x: Bin size in X direction (meters) for coordinate mode
            bin_size_y: Bin size in Y direction (meters) for coordinate mode
            coord_x_key: X coordinate header name
            coord_y_key: Y coordinate header name
            inline_key: Header name for inline axis (header mode)
            xline_key: Header name for crossline axis (header mode)
            representative_method: How to combine traces in bin ('median', 'mean', 'first', 'nearest')
            reconstruction_method: How to reconstruct traces ('noise_subtract', 'residual_preserve', 'multi_pass')
            aperture_inline: Spatial aperture in inline direction (odd, >=1)
            aperture_xline: Spatial aperture in crossline direction (odd, >=1)
            wavelet: Wavelet name for DWT
            level: DWT decomposition level (None = auto)
            threshold_k: MAD threshold multiplier
            threshold_mode: 'soft' or 'hard' thresholding
        """
        self.use_coordinates = use_coordinates
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.coord_x_key = coord_x_key
        self.coord_y_key = coord_y_key
        self.inline_key = inline_key
        self.xline_key = xline_key
        self.representative_method = representative_method
        self.reconstruction_method = reconstruction_method
        self.aperture_inline = aperture_inline
        self.aperture_xline = aperture_xline
        self.wavelet = wavelet
        self.level = level
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode

        # Internal state
        self._volume_builder: Optional[CoordinateVolumeBuilder] = None
        self._geometry: Optional[Dict[str, Any]] = None

        super().__init__(
            use_coordinates=use_coordinates,
            bin_size_x=bin_size_x,
            bin_size_y=bin_size_y,
            coord_x_key=coord_x_key,
            coord_y_key=coord_y_key,
            inline_key=inline_key,
            xline_key=xline_key,
            representative_method=representative_method,
            reconstruction_method=reconstruction_method,
            aperture_inline=aperture_inline,
            aperture_xline=aperture_xline,
            wavelet=wavelet,
            level=level,
            threshold_k=threshold_k,
            threshold_mode=threshold_mode
        )

    def _validate_params(self):
        """Validate processor parameters."""
        if self.aperture_inline < 1:
            raise ValueError("aperture_inline must be at least 1")
        if self.aperture_inline > 1 and self.aperture_inline % 2 == 0:
            raise ValueError("aperture_inline must be odd if > 1")
        if self.aperture_xline < 1:
            raise ValueError("aperture_xline must be at least 1")
        if self.aperture_xline > 1 and self.aperture_xline % 2 == 0:
            raise ValueError("aperture_xline must be odd if > 1")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be positive")
        if self.threshold_mode not in ['soft', 'hard']:
            raise ValueError("threshold_mode must be 'soft' or 'hard'")
        if self.reconstruction_method not in ['noise_subtract', 'residual_preserve', 'multi_pass']:
            raise ValueError("reconstruction_method must be 'noise_subtract', 'residual_preserve', or 'multi_pass'")

        # Validate wavelet
        try:
            pywt.Wavelet(self.wavelet)
        except Exception:
            raise ValueError(f"Invalid wavelet: {self.wavelet}")

        # Validate bin size for coordinate mode
        if self.use_coordinates:
            if self.bin_size_x <= 0:
                raise ValueError(f"bin_size_x must be positive, got {self.bin_size_x}")
            if self.bin_size_y <= 0:
                raise ValueError(f"bin_size_y must be positive, got {self.bin_size_y}")

    def get_description(self) -> str:
        """Get processor description."""
        if self.use_coordinates:
            mode = f"coords({self.bin_size_x}x{self.bin_size_y}m)"
        else:
            mode = f"headers({self.inline_key}x{self.xline_key})"

        return (
            f"3D-Denoise: {mode}, "
            f"aperture={self.aperture_inline}x{self.aperture_xline}, "
            f"wavelet={self.wavelet}, k={self.threshold_k:.1f}, "
            f"recon={self.reconstruction_method}"
        )

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply 3D spatial denoising.

        Args:
            data: Input seismic data with headers

        Returns:
            Denoised seismic data (same trace count as input)
        """
        import time
        start_time = time.time()

        # Convert to float32 for memory efficiency (50% savings)
        traces = data.traces
        if traces.dtype != np.float32:
            traces = traces.astype(np.float32)
        else:
            traces = traces.copy()
        n_samples, n_traces = traces.shape

        # Validate headers exist
        if data.headers is None:
            raise ValueError("3D denoising requires headers. SeismicData.headers is None.")

        # Convert headers to DataFrame if needed
        if isinstance(data.headers, dict):
            headers_df = pd.DataFrame(data.headers)
        else:
            headers_df = data.headers

        if len(headers_df) != n_traces:
            raise ValueError(f"Header count ({len(headers_df)}) != trace count ({n_traces})")

        sample_rate_ms = data.sample_rate  # Already in milliseconds (SeismicData convention)

        logger.info(
            f"3D-Denoise: {n_traces} traces x {n_samples} samples | "
            f"mode={'coordinates' if self.use_coordinates else 'headers'} | "
            f"recon={self.reconstruction_method}"
        )

        if self.use_coordinates:
            denoised_traces = self._process_coordinate_mode(
                traces, headers_df, sample_rate_ms
            )
        else:
            denoised_traces = self._process_header_mode(
                traces, headers_df
            )

        # Compute metrics
        elapsed = time.time() - start_time
        throughput = n_traces / elapsed if elapsed > 0 else 0

        input_rms = np.sqrt(np.mean(traces ** 2))
        output_rms = np.sqrt(np.mean(denoised_traces ** 2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"3D-Denoise complete: {elapsed:.2f}s | "
            f"{throughput:.1f} traces/s | "
            f"Energy: {energy_ratio:.1%} retained"
        )

        if energy_ratio < 0.10:
            logger.warning("Output <10% of input energy - threshold may be too aggressive")
        elif energy_ratio > 0.95:
            logger.warning("Output >95% of input energy - minimal denoising occurred")

        # Build metadata
        metadata = {
            **data.metadata,
            'processor': self.get_description(),
            'reconstruction_method': self.reconstruction_method,
        }

        if self._volume_builder and self._volume_builder.get_geometry():
            geom = self._volume_builder.get_geometry()
            metadata['volume_shape'] = f"{n_samples}x{geom.n_bins_x}x{geom.n_bins_y}"
            metadata['volume_coverage'] = f"{geom.coverage_percent:.1f}%"
            metadata['max_fold'] = geom.max_fold
            metadata['mean_fold'] = geom.mean_fold
        elif self._geometry:
            metadata['volume_shape'] = f"{n_samples}x{self._geometry['n_inlines']}x{self._geometry['n_xlines']}"
            metadata['volume_coverage'] = f"{self._geometry['coverage']:.1f}%"

        return SeismicData(
            traces=denoised_traces,
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata=metadata
        )

    def _process_coordinate_mode(
        self,
        traces: np.ndarray,
        headers_df: pd.DataFrame,
        sample_rate_ms: float
    ) -> np.ndarray:
        """Process using coordinate-based volume building with multi-fold support."""
        n_samples, n_traces = traces.shape

        # Create binning config
        binning_config = BinningConfig(
            bin_size_x=self.bin_size_x,
            bin_size_y=self.bin_size_y,
            coord_x_key=self.coord_x_key,
            coord_y_key=self.coord_y_key,
            representative_method=RepresentativeMethod(self.representative_method),
            reconstruction_method=ReconstructionMethod(self.reconstruction_method),
        )

        # Build volume
        self._volume_builder = CoordinateVolumeBuilder(binning_config)
        volume = self._volume_builder.build(traces, headers_df, sample_rate_ms)

        geom = self._volume_builder.get_geometry()
        logger.info(f"Volume: {geom.n_bins_x}x{geom.n_bins_y} bins, "
                   f"max_fold={geom.max_fold}, coverage={geom.coverage_percent:.1f}%")

        # Apply 3D denoising to volume
        denoised_volume_data = self._denoise_volume_3d(
            volume.data,
            min(self.aperture_inline, geom.n_bins_x),
            min(self.aperture_xline, geom.n_bins_y)
        )

        denoised_volume = SeismicVolume(
            data=denoised_volume_data,
            dt=volume.dt,
            dx=volume.dx,
            dy=volume.dy
        )

        # Reconstruct traces using selected method
        if self.reconstruction_method == 'multi_pass':
            # Multi-pass needs a denoise function
            def denoise_func(vol: SeismicVolume) -> SeismicVolume:
                denoised_data = self._denoise_volume_3d(
                    vol.data,
                    min(self.aperture_inline, vol.data.shape[1]),
                    min(self.aperture_xline, vol.data.shape[2])
                )
                return SeismicVolume(
                    data=denoised_data,
                    dt=vol.dt, dx=vol.dx, dy=vol.dy
                )

            denoised_traces = self._volume_builder.reconstruct_traces(
                denoised_volume,
                filter_func=denoise_func
            )
        else:
            denoised_traces = self._volume_builder.reconstruct_traces(denoised_volume)

        return denoised_traces

    def _process_header_mode(
        self,
        traces: np.ndarray,
        headers_df: pd.DataFrame
    ) -> np.ndarray:
        """Process using header-based volume building (legacy mode, no multi-fold)."""
        n_samples, n_traces = traces.shape

        # Build volume from headers
        volume, geometry = self._build_volume_from_headers(
            traces, headers_df, self.inline_key, self.xline_key
        )
        self._geometry = geometry

        n_inlines = geometry['n_inlines']
        n_xlines = geometry['n_xlines']

        logger.info(f"Volume: {n_inlines}x{n_xlines}, coverage={geometry['coverage']:.1f}%")

        # Clamp apertures to volume size
        eff_ap_il = min(self.aperture_inline, n_inlines)
        if eff_ap_il % 2 == 0:
            eff_ap_il = max(1, eff_ap_il - 1)
        eff_ap_xl = min(self.aperture_xline, n_xlines)
        if eff_ap_xl % 2 == 0:
            eff_ap_xl = max(1, eff_ap_xl - 1)

        # Apply 3D denoising
        denoised_volume = self._denoise_volume_3d(volume, eff_ap_il, eff_ap_xl)

        # Extract back to 2D
        denoised_traces = self._extract_traces_from_volume(
            denoised_volume, headers_df, geometry
        )

        return denoised_traces

    def _build_volume_from_headers(
        self,
        traces: np.ndarray,
        headers_df: pd.DataFrame,
        inline_key: str,
        xline_key: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build a 3D volume from 2D traces using header keys (legacy)."""
        n_samples, n_traces = traces.shape

        if inline_key not in headers_df.columns:
            raise ValueError(f"Inline key '{inline_key}' not found in headers. "
                           f"Available: {list(headers_df.columns)[:10]}")
        if xline_key not in headers_df.columns:
            raise ValueError(f"Crossline key '{xline_key}' not found in headers. "
                           f"Available: {list(headers_df.columns)[:10]}")

        # Get unique values
        inline_vals = np.sort(headers_df[inline_key].unique())
        xline_vals = np.sort(headers_df[xline_key].unique())

        n_inlines = len(inline_vals)
        n_xlines = len(xline_vals)

        # Create mappings
        inline_to_idx = {v: i for i, v in enumerate(inline_vals)}
        xline_to_idx = {v: i for i, v in enumerate(xline_vals)}

        # Initialize volume
        volume = np.zeros((n_samples, n_inlines, n_xlines), dtype=traces.dtype)
        has_data = np.zeros((n_inlines, n_xlines), dtype=bool)

        # Fill volume (last trace wins for duplicates)
        for trace_idx in range(n_traces):
            il_val = headers_df.iloc[trace_idx][inline_key]
            xl_val = headers_df.iloc[trace_idx][xline_key]

            il_idx = inline_to_idx.get(il_val)
            xl_idx = xline_to_idx.get(xl_val)

            if il_idx is not None and xl_idx is not None:
                volume[:, il_idx, xl_idx] = traces[:, trace_idx]
                has_data[il_idx, xl_idx] = True

        coverage = has_data.sum() / (n_inlines * n_xlines) * 100

        geometry = {
            'inline_key': inline_key,
            'xline_key': xline_key,
            'inline_vals': inline_vals,
            'xline_vals': xline_vals,
            'inline_to_idx': inline_to_idx,
            'xline_to_idx': xline_to_idx,
            'n_inlines': n_inlines,
            'n_xlines': n_xlines,
            'has_data': has_data,
            'coverage': coverage
        }

        return volume, geometry

    def _extract_traces_from_volume(
        self,
        volume: np.ndarray,
        headers_df: pd.DataFrame,
        geometry: Dict[str, Any]
    ) -> np.ndarray:
        """Extract traces from 3D volume back to original 2D organization."""
        n_samples = volume.shape[0]
        n_traces = len(headers_df)

        inline_key = geometry['inline_key']
        xline_key = geometry['xline_key']
        inline_to_idx = geometry['inline_to_idx']
        xline_to_idx = geometry['xline_to_idx']

        traces = np.zeros((n_samples, n_traces), dtype=volume.dtype)

        for trace_idx in range(n_traces):
            il_val = headers_df.iloc[trace_idx][inline_key]
            xl_val = headers_df.iloc[trace_idx][xline_key]

            il_idx = inline_to_idx.get(il_val)
            xl_idx = xline_to_idx.get(xl_val)

            if il_idx is not None and xl_idx is not None:
                traces[:, trace_idx] = volume[:, il_idx, xl_idx]

        return traces

    def _denoise_volume_3d(
        self,
        volume: np.ndarray,
        aperture_il: int,
        aperture_xl: int
    ) -> np.ndarray:
        """
        Apply DWT denoising with 3D MAD spatial statistics.

        Args:
            volume: 3D array (n_samples, n_inlines, n_xlines)
            aperture_il: Effective inline aperture
            aperture_xl: Effective crossline aperture

        Returns:
            Denoised 3D volume
        """
        n_samples, n_inlines, n_xlines = volume.shape

        # Determine decomposition level
        if self.level is None:
            max_level = pywt.dwt_max_level(n_samples, self.wavelet)
            level = min(max_level, 5)  # Cap at 5 levels
        else:
            level = self.level

        denoised = np.zeros_like(volume)

        # Process each (inline, xline) position
        for il in range(n_inlines):
            for xl in range(n_xlines):
                trace = volume[:, il, xl]

                if np.all(trace == 0):
                    # Empty position (no data)
                    continue

                # DWT decomposition
                coeffs = pywt.wavedec(trace, self.wavelet, level=level)

                # Get spatial patch for noise estimation
                il_start = max(0, il - aperture_il // 2)
                il_end = min(n_inlines, il + aperture_il // 2 + 1)
                xl_start = max(0, xl - aperture_xl // 2)
                xl_end = min(n_xlines, xl + aperture_xl // 2 + 1)

                patch = volume[:, il_start:il_end, xl_start:xl_end]

                # Threshold each detail level using spatial MAD
                denoised_coeffs = [coeffs[0]]  # Keep approximation

                for i, coeff in enumerate(coeffs[1:], 1):
                    # Compute DWT for all traces in patch at this level
                    patch_coeffs = []
                    for pil in range(patch.shape[1]):
                        for pxl in range(patch.shape[2]):
                            p_trace = patch[:, pil, pxl]
                            if not np.all(p_trace == 0):
                                p_coeffs = pywt.wavedec(p_trace, self.wavelet, level=level)
                                if len(p_coeffs) > i:
                                    patch_coeffs.append(p_coeffs[i])

                    if patch_coeffs:
                        # Stack and compute MAD across spatial dimension (float32 for memory efficiency)
                        max_len = max(len(c) for c in patch_coeffs)
                        patch_stack = np.zeros((len(patch_coeffs), max_len), dtype=np.float32)
                        for j, c in enumerate(patch_coeffs):
                            patch_stack[j, :len(c)] = c

                        # Spatial MAD per coefficient position
                        median_coeff = np.median(patch_stack, axis=0)
                        mad = np.median(np.abs(patch_stack - median_coeff), axis=0)
                        mad_scaled = mad * 1.4826

                        # Prevent zero threshold
                        mad_scaled = np.maximum(mad_scaled, 1e-10)
                        threshold = self.threshold_k * mad_scaled[:len(coeff)]
                    else:
                        # Fallback to single-trace MAD
                        median_coeff = np.median(np.abs(coeff))
                        mad = np.median(np.abs(np.abs(coeff) - median_coeff))
                        mad_scaled = mad * 1.4826
                        threshold = self.threshold_k * max(mad_scaled, 1e-10)

                    # Apply thresholding
                    if self.threshold_mode == 'soft':
                        denoised_coeff = pywt.threshold(coeff, threshold, mode='soft')
                    else:
                        denoised_coeff = pywt.threshold(coeff, threshold, mode='hard')

                    denoised_coeffs.append(denoised_coeff)

                # Reconstruct
                rec = pywt.waverec(denoised_coeffs, self.wavelet)

                # Handle length mismatch
                if len(rec) < n_samples:
                    rec = np.pad(rec, (0, n_samples - len(rec)))
                elif len(rec) > n_samples:
                    rec = rec[:n_samples]

                denoised[:, il, xl] = rec

        return denoised


# Convenience functions for common use cases

def apply_denoise3d_design_mode(
    traces: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    bin_size: float = 25.0,
    aperture: int = 3,
    threshold_k: float = 3.0,
    wavelet: str = 'db4'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply 3D denoising in design mode (fast).

    Uses noise_subtract reconstruction for speed.
    """
    processor = Denoise3D(
        use_coordinates=True,
        bin_size_x=bin_size,
        bin_size_y=bin_size,
        aperture_inline=aperture,
        aperture_xline=aperture,
        threshold_k=threshold_k,
        wavelet=wavelet,
        reconstruction_method='noise_subtract'
    )

    from models.seismic_data import SeismicData
    data = SeismicData(
        traces=traces,
        sample_rate=sample_rate_ms / 1000.0,
        headers=headers_df
    )

    result = processor.process(data)
    stats = processor._volume_builder.get_statistics() if processor._volume_builder else {}

    return result.traces, stats


def apply_denoise3d_application_mode(
    traces: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    bin_size: float = 25.0,
    aperture: int = 3,
    threshold_k: float = 3.0,
    wavelet: str = 'db4'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply 3D denoising in application mode (accurate).

    Uses multi_pass reconstruction for best quality.
    """
    processor = Denoise3D(
        use_coordinates=True,
        bin_size_x=bin_size,
        bin_size_y=bin_size,
        aperture_inline=aperture,
        aperture_xline=aperture,
        threshold_k=threshold_k,
        wavelet=wavelet,
        reconstruction_method='multi_pass'
    )

    from models.seismic_data import SeismicData
    data = SeismicData(
        traces=traces,
        sample_rate=sample_rate_ms / 1000.0,
        headers=headers_df
    )

    result = processor.process(data)
    stats = processor._volume_builder.get_statistics() if processor._volume_builder else {}

    return result.traces, stats
