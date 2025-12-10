"""
3D Spatial Denoising Processor.

Implements 3D spatial statistics for robust noise estimation by:
1. Building a 3D volume from 2D gather using user-selected headers
2. Computing spatial noise statistics in 3D (inline x crossline apertures)
3. Applying DWT-based denoising with 3D MAD thresholding
4. Extracting results back to original 2D gather organization

Key advantages:
- Exploits spatial coherence in both inline and crossline directions
- More robust noise estimation than 1D aperture
- User-selectable headers for flexible volume organization
- DWT for fast computation with perfect reconstruction

Best suited for:
- Organized gathers (shot, CDP, receiver, etc.)
- Data with coherent signal in multiple spatial directions
- Random noise attenuation with strong spatial constraints
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple, Dict, Any, List
import logging
import pywt

from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_JOBS = max(1, multiprocessing.cpu_count() - 1)
except ImportError:
    JOBLIB_AVAILABLE = False
    N_JOBS = 1


def compute_3d_mad(data_3d: np.ndarray,
                   aperture_inline: int = 3,
                   aperture_xline: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D spatial MAD statistics using a 2D spatial aperture.

    For each time sample and each (inline, xline) position, computes
    the median and MAD over the spatial aperture neighborhood.

    Args:
        data_3d: 3D array (n_samples, n_inlines, n_xlines)
        aperture_inline: Aperture size in inline direction (odd)
        aperture_xline: Aperture size in crossline direction (odd)

    Returns:
        Tuple of (median_3d, mad_3d) with same shape as input
    """
    n_samples, n_inlines, n_xlines = data_3d.shape

    half_il = aperture_inline // 2
    half_xl = aperture_xline // 2

    median_3d = np.zeros_like(data_3d)
    mad_3d = np.zeros_like(data_3d)

    for il in range(n_inlines):
        il_start = max(0, il - half_il)
        il_end = min(n_inlines, il + half_il + 1)

        for xl in range(n_xlines):
            xl_start = max(0, xl - half_xl)
            xl_end = min(n_xlines, xl + half_xl + 1)

            # Extract spatial patch for all time samples
            patch = data_3d[:, il_start:il_end, xl_start:xl_end]  # (n_samples, patch_il, patch_xl)

            # Reshape to (n_samples, patch_size) for per-time-sample statistics
            patch_flat = patch.reshape(n_samples, -1)

            # Compute median and MAD per time sample
            med = np.median(patch_flat, axis=1)
            mad = np.median(np.abs(patch_flat - med[:, np.newaxis]), axis=1)

            median_3d[:, il, xl] = med
            mad_3d[:, il, xl] = mad

    return median_3d, mad_3d


def build_volume_from_headers(
    traces: np.ndarray,
    headers_df: pd.DataFrame,
    inline_key: str,
    xline_key: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a 3D volume from 2D traces using header keys.

    Args:
        traces: 2D array (n_samples, n_traces)
        headers_df: DataFrame with trace headers
        inline_key: Header name for inline axis
        xline_key: Header name for crossline axis

    Returns:
        Tuple of:
        - volume: 3D array (n_samples, n_inlines, n_xlines)
        - geometry: Dict with mapping information for extraction
    """
    n_samples, n_traces = traces.shape

    if inline_key not in headers_df.columns:
        raise ValueError(f"Inline key '{inline_key}' not found in headers. "
                        f"Available: {list(headers_df.columns)[:10]}")
    if xline_key not in headers_df.columns:
        raise ValueError(f"Crossline key '{xline_key}' not found in headers. "
                        f"Available: {list(headers_df.columns)[:10]}")

    if len(headers_df) != n_traces:
        raise ValueError(f"Header count ({len(headers_df)}) doesn't match trace count ({n_traces})")

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

    # Track which positions have data
    has_data = np.zeros((n_inlines, n_xlines), dtype=bool)

    # Fill volume
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

    logger.info(f"Built volume: {n_samples}×{n_inlines}×{n_xlines}, "
               f"coverage={coverage:.1f}%")

    return volume, geometry


def extract_traces_from_volume(
    volume: np.ndarray,
    headers_df: pd.DataFrame,
    geometry: Dict[str, Any]
) -> np.ndarray:
    """
    Extract traces from 3D volume back to original 2D organization.

    Args:
        volume: 3D array (n_samples, n_inlines, n_xlines)
        headers_df: DataFrame with trace headers
        geometry: Dict with mapping information from build_volume_from_headers

    Returns:
        traces: 2D array (n_samples, n_traces) in original order
    """
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


class Denoise3D(BaseProcessor):
    """
    3D Spatial Denoising using DWT with 3D MAD statistics.

    Builds a 3D volume from 2D gather data using user-specified headers,
    computes robust noise estimates using 2D spatial apertures (inline × crossline),
    and applies DWT-based thresholding.

    The result is extracted back to the original 2D gather organization.
    """

    def __init__(self,
                 inline_key: str = 'field_record',
                 xline_key: str = 'trace_number',
                 aperture_inline: int = 3,
                 aperture_xline: int = 3,
                 wavelet: str = 'db4',
                 level: Optional[int] = None,
                 threshold_k: float = 3.0,
                 threshold_mode: Literal['soft', 'hard'] = 'soft'):
        """
        Initialize 3D Denoise processor.

        Args:
            inline_key: Header name for inline (X) axis of 3D volume
            xline_key: Header name for crossline (Y) axis of 3D volume
            aperture_inline: Spatial aperture in inline direction (odd, ≥3)
            aperture_xline: Spatial aperture in crossline direction (odd, ≥3)
            wavelet: Wavelet name (default 'db4')
            level: Decomposition level (None = auto)
            threshold_k: MAD threshold multiplier
            threshold_mode: 'soft' or 'hard' thresholding
        """
        self.inline_key = inline_key
        self.xline_key = xline_key
        self.aperture_inline = aperture_inline
        self.aperture_xline = aperture_xline
        self.wavelet = wavelet
        self.level = level
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode

        super().__init__(
            inline_key=inline_key,
            xline_key=xline_key,
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

        # Validate wavelet
        try:
            pywt.Wavelet(self.wavelet)
        except Exception:
            raise ValueError(f"Invalid wavelet: {self.wavelet}")

    def get_description(self) -> str:
        """Get processor description."""
        return (f"3D-Denoise: "
                f"keys={self.inline_key}×{self.xline_key}, "
                f"aperture={self.aperture_inline}×{self.aperture_xline}, "
                f"wavelet={self.wavelet}, "
                f"k={self.threshold_k:.1f}, {self.threshold_mode}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply 3D spatial denoising.

        Requires SeismicData to have headers containing the specified
        inline_key and xline_key columns.

        Args:
            data: Input seismic data with headers

        Returns:
            Denoised seismic data (same organization as input)
        """
        import time

        start_time = time.time()

        traces = data.traces.copy()
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

        logger.info(
            f"3D-Denoise: {n_traces} traces × {n_samples} samples | "
            f"keys={self.inline_key}×{self.xline_key} | "
            f"aperture={self.aperture_inline}×{self.aperture_xline} | "
            f"k={self.threshold_k}, {self.threshold_mode}"
        )

        # Build 3D volume from headers
        try:
            volume, geometry = build_volume_from_headers(
                traces, headers_df, self.inline_key, self.xline_key
            )
        except ValueError as e:
            logger.error(f"Failed to build volume: {e}")
            raise

        n_inlines = geometry['n_inlines']
        n_xlines = geometry['n_xlines']

        # Clamp apertures to volume size
        eff_ap_il = min(self.aperture_inline, n_inlines)
        if eff_ap_il % 2 == 0:
            eff_ap_il = max(1, eff_ap_il - 1)
        eff_ap_xl = min(self.aperture_xline, n_xlines)
        if eff_ap_xl % 2 == 0:
            eff_ap_xl = max(1, eff_ap_xl - 1)

        if eff_ap_il != self.aperture_inline or eff_ap_xl != self.aperture_xline:
            logger.warning(
                f"Volume too small for aperture, using {eff_ap_il}×{eff_ap_xl}"
            )

        # Apply 3D denoising
        denoised_volume = self._denoise_volume_3d(volume, eff_ap_il, eff_ap_xl)

        # Extract back to 2D
        denoised_traces = extract_traces_from_volume(
            denoised_volume, headers_df, geometry
        )

        # Compute metrics
        elapsed = time.time() - start_time
        throughput = n_traces / elapsed if elapsed > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
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

        return SeismicData(
            traces=denoised_traces,
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata={
                **data.metadata,
                'processor': self.get_description(),
                'volume_shape': f"{n_samples}×{n_inlines}×{n_xlines}",
                'volume_coverage': f"{geometry['coverage']:.1f}%"
            }
        )

    def _denoise_volume_3d(self,
                          volume: np.ndarray,
                          aperture_il: int,
                          aperture_xl: int) -> np.ndarray:
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
                        # Stack and compute MAD across spatial dimension
                        # Pad to same length if needed
                        max_len = max(len(c) for c in patch_coeffs)
                        patch_stack = np.zeros((len(patch_coeffs), max_len))
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
