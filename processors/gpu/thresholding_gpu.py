"""
GPU-Accelerated Thresholding Operations.

Provides GPU-accelerated MAD (Median Absolute Deviation) thresholding
for time-frequency domain denoising.
"""

import torch
import numpy as np
from typing import Union, Tuple, Literal
import logging

from .utils_gpu import (
    numpy_to_tensor,
    tensor_to_numpy,
    compute_median_gpu,
    compute_mad_gpu,
    soft_threshold_gpu,
    garrote_threshold_gpu,
)

logger = logging.getLogger(__name__)


class ThresholdingGPU:
    """
    GPU-accelerated thresholding operations for TF-domain denoising.

    Implements MAD-based adaptive thresholding with soft and Garrote
    threshold functions, optimized for GPU execution.

    Threshold modes:
    - 'soft': Classical soft thresholding (partial removal)
    - 'hard': Full removal for outliers, preserve non-outliers exactly
    - 'scaled': Progressive removal based on outlier severity
    - 'adaptive': Combined hard (severe) + scaled (moderate) outliers
    """

    def __init__(
        self,
        device: torch.device,
        threshold_k: float = 3.0,
        threshold_type: Literal['soft', 'garrote'] = 'soft',
        threshold_mode: Literal['soft', 'hard', 'scaled', 'adaptive'] = 'adaptive',
        low_amp_protection: bool = True,
        low_amp_factor: float = 0.3
    ):
        """
        Initialize thresholding processor.

        Args:
            device: PyTorch device (cuda, mps, or cpu)
            threshold_k: MAD threshold multiplier (k * MAD)
            threshold_type: Type of thresholding ('soft' or 'garrote')
            threshold_mode: Noise removal mode:
                - 'soft': Classical soft thresholding (partial removal)
                - 'hard': Full removal for outliers (snap to median)
                - 'scaled': Progressive removal based on outlier severity
                - 'adaptive': Hard for severe (>2k*MAD) + scaled for moderate
            low_amp_protection: If True, prevent inflation of low-amplitude samples
            low_amp_factor: Threshold for low-amplitude protection (fraction of median)
        """
        self.device = device
        self.threshold_k = threshold_k
        self.threshold_type = threshold_type
        self.threshold_mode = threshold_mode
        self.low_amp_protection = low_amp_protection
        self.low_amp_factor = low_amp_factor

    def apply_mad_thresholding(
        self,
        tf_center: np.ndarray,
        tf_ensemble: np.ndarray,
        spatial_dim: int = 0
    ) -> np.ndarray:
        """
        Apply MAD-based thresholding to TF coefficients.

        Computes adaptive threshold from spatial ensemble using MAD,
        then applies soft or Garrote thresholding.

        Args:
            tf_center: TF coefficients for center trace (n_freqs, n_times)
            tf_ensemble: TF coefficients for spatial ensemble (n_traces, n_freqs, n_times)
            spatial_dim: Dimension along which to compute MAD (default: 0 = traces)

        Returns:
            Thresholded TF coefficients (n_freqs, n_times)
        """
        n_freqs, n_times = tf_center.shape
        n_traces = tf_ensemble.shape[0]

        logger.debug(
            f"Applying MAD thresholding on GPU: "
            f"center shape={tf_center.shape}, ensemble shape={tf_ensemble.shape}"
        )

        # Transfer to GPU
        tf_center_gpu = torch.from_numpy(tf_center).to(self.device)
        tf_ensemble_gpu = torch.from_numpy(tf_ensemble).to(self.device)

        # Pre-allocate result
        tf_denoised = torch.zeros_like(tf_center_gpu)

        # Compute spatial statistics (median and MAD) for outlier detection
        median_amp, mad, outlier_thresholds = self._compute_spatial_statistics_vectorized(
            tf_ensemble_gpu,
            spatial_dim=spatial_dim
        )
        # Shapes: median_amp (n_freqs, n_times), mad (n_freqs, n_times), outlier_thresholds (n_freqs, n_times)

        # Apply outlier-based thresholding
        # Coefficients close to median (coherent) are kept
        # Coefficients far from median (outliers) are removed
        if self.threshold_type == 'soft':
            tf_denoised = self._soft_threshold_outliers_gpu(
                tf_center_gpu, median_amp, outlier_thresholds
            )
        else:  # garrote
            tf_denoised = self._garrote_threshold_outliers_gpu(
                tf_center_gpu, median_amp, outlier_thresholds
            )

        # Transfer back to CPU
        result = tensor_to_numpy(tf_denoised)

        logger.debug(f"Thresholding completed on GPU")

        return result

    def _compute_spatial_statistics_vectorized(
        self,
        tf_ensemble: torch.Tensor,
        spatial_dim: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spatial statistics for outlier-based thresholding (fully vectorized on GPU).

        Args:
            tf_ensemble: Ensemble TF coefficients (n_traces, n_freqs, n_times)
            spatial_dim: Dimension to compute statistics along

        Returns:
            Tuple of (median, MAD, outlier_threshold)
            All shapes: (n_freqs, n_times)
        """
        # Compute magnitudes across spatial dimension
        magnitudes = torch.abs(tf_ensemble)  # (n_traces, n_freqs, n_times)

        # Compute median across spatial dimension
        median_mag = compute_median_gpu(magnitudes, dim=spatial_dim, keepdim=False)
        # Shape: (n_freqs, n_times)

        # Compute MAD
        # Expand median for broadcasting
        median_expanded = median_mag.unsqueeze(spatial_dim)  # (1, n_freqs, n_times)

        # Compute absolute deviations
        abs_dev = torch.abs(magnitudes - median_expanded)  # (n_traces, n_freqs, n_times)

        # MAD = median of absolute deviations
        mad = compute_median_gpu(abs_dev, dim=spatial_dim, keepdim=False)
        # Shape: (n_freqs, n_times)

        # Scale MAD (1.4826 factor for Gaussian consistency)
        mad_scaled = mad * 1.4826

        # Outlier threshold = k * MAD (no median offset!)
        # Coefficients with deviation > k*MAD are outliers (noise)
        outlier_thresholds = self.threshold_k * mad_scaled

        return median_mag, mad_scaled, outlier_thresholds

    def _soft_threshold_outliers_gpu(
        self,
        tf_center: torch.Tensor,
        median: torch.Tensor,
        outlier_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply soft thresholding on DEVIATION from median (outlier detection).

        Args:
            tf_center: Center trace TF coefficients (n_freqs, n_times)
            median: Spatial median (n_freqs, n_times)
            outlier_threshold: k * MAD threshold (n_freqs, n_times)

        Returns:
            Thresholded coefficients (n_freqs, n_times)
        """
        magnitude = torch.abs(tf_center)
        phase = torch.angle(tf_center)

        # Compute deviation from spatial median
        deviation = torch.abs(magnitude - median)

        # Soft threshold on deviation
        # Shrink deviations larger than k*MAD (outliers)
        new_deviation = torch.maximum(
            deviation - outlier_threshold,
            torch.zeros_like(deviation)
        )

        # Reconstruct magnitude: median ± thresholded_deviation
        signs = torch.where(magnitude >= median, 1.0, -1.0)
        new_magnitude = torch.maximum(
            median + signs * new_deviation,
            torch.zeros_like(median)
        )

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)

    def _garrote_threshold_outliers_gpu(
        self,
        tf_center: torch.Tensor,
        median: torch.Tensor,
        outlier_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Garrote thresholding on DEVIATION from median (outlier detection).

        Args:
            tf_center: Center trace TF coefficients (n_freqs, n_times)
            median: Spatial median (n_freqs, n_times)
            outlier_threshold: k * MAD threshold (n_freqs, n_times)

        Returns:
            Thresholded coefficients (n_freqs, n_times)
        """
        magnitude = torch.abs(tf_center)
        phase = torch.angle(tf_center)

        # Compute deviation from spatial median
        deviation = torch.abs(magnitude - median)

        # Garrote threshold on deviation
        new_deviation = torch.where(
            deviation > outlier_threshold,
            deviation - (outlier_threshold**2 / (deviation + 1e-10)),
            deviation
        )

        # Reconstruct magnitude: median ± thresholded_deviation
        signs = torch.where(magnitude >= median, 1.0, -1.0)
        new_magnitude = torch.maximum(
            median + signs * new_deviation,
            torch.zeros_like(median)
        )

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)

    def _hard_threshold_outliers_gpu(
        self,
        tf_center: torch.Tensor,
        median: torch.Tensor,
        outlier_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Option A: Hard threshold - full removal for outliers, preserve non-outliers exactly.

        Key difference from soft: non-outliers keep their ORIGINAL magnitude,
        not pushed toward median. This prevents inflation of low-amplitude samples.

        Args:
            tf_center: Center trace TF coefficients (n_freqs, n_times)
            median: Spatial median (n_freqs, n_times)
            outlier_threshold: k * MAD threshold (n_freqs, n_times)

        Returns:
            Thresholded coefficients (n_freqs, n_times)
        """
        magnitude = torch.abs(tf_center)
        phase = torch.angle(tf_center)

        # Compute deviation from spatial median
        deviation = torch.abs(magnitude - median)

        # Hard threshold: outliers snap to median, non-outliers keep original
        new_magnitude = torch.where(
            deviation > outlier_threshold,
            median,  # Full removal - snap to median
            magnitude  # Keep original exactly (no inflation!)
        )

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)

    def _scaled_threshold_outliers_gpu(
        self,
        tf_center: torch.Tensor,
        median: torch.Tensor,
        outlier_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Option B: Scaled progressive removal based on outlier severity.

        Removal factor increases smoothly from 0 (at threshold) to ~1 (for severe outliers).
        Non-outliers keep their original magnitude.

        Args:
            tf_center: Center trace TF coefficients (n_freqs, n_times)
            median: Spatial median (n_freqs, n_times)
            outlier_threshold: k * MAD threshold (n_freqs, n_times)

        Returns:
            Thresholded coefficients (n_freqs, n_times)
        """
        magnitude = torch.abs(tf_center)
        phase = torch.angle(tf_center)

        # Compute deviation from spatial median
        deviation = torch.abs(magnitude - median)

        # Compute outlier ratio: how many times threshold is exceeded
        outlier_ratio = deviation / (outlier_threshold + 1e-10)

        # Progressive removal factor: 0 at threshold, approaches 1 for severe outliers
        # Using smooth transition: removal = 1 - 1/(1 + (ratio-1)^2) for ratio > 1
        excess_ratio = torch.clamp(outlier_ratio - 1.0, min=0.0)
        removal_factor = 1.0 - 1.0 / (1.0 + excess_ratio ** 2)

        # New deviation: progressively reduced toward zero based on severity
        new_deviation = deviation * (1.0 - removal_factor)

        # Reconstruct magnitude
        signs = torch.where(magnitude >= median, 1.0, -1.0)
        new_magnitude = torch.where(
            outlier_ratio > 1.0,
            torch.maximum(median + signs * new_deviation, torch.zeros_like(median)),
            magnitude  # Keep original for non-outliers (no inflation!)
        )

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)

    def _adaptive_threshold_outliers_gpu(
        self,
        tf_center: torch.Tensor,
        median: torch.Tensor,
        outlier_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive threshold: combines hard (severe outliers) + scaled (moderate outliers).

        - Severe outliers (> 2*k*MAD): Hard threshold (snap to median)
        - Moderate outliers (k*MAD to 2*k*MAD): Scaled progressive removal
        - Non-outliers (< k*MAD): Keep original magnitude

        Args:
            tf_center: Center trace TF coefficients (n_freqs, n_times)
            median: Spatial median (n_freqs, n_times)
            outlier_threshold: k * MAD threshold (n_freqs, n_times)

        Returns:
            Thresholded coefficients (n_freqs, n_times)
        """
        magnitude = torch.abs(tf_center)
        phase = torch.angle(tf_center)

        # Compute deviation from spatial median
        deviation = torch.abs(magnitude - median)

        # Compute outlier ratio
        outlier_ratio = deviation / (outlier_threshold + 1e-10)

        # Severe outlier threshold (2x the normal threshold)
        severe_threshold = 2.0

        # For moderate outliers: scaled removal
        # Progressive removal between ratio 1.0 and 2.0
        moderate_removal = torch.clamp((outlier_ratio - 1.0) / (severe_threshold - 1.0), 0.0, 1.0)

        # Calculate new deviation for moderate outliers
        signs = torch.where(magnitude >= median, 1.0, -1.0)
        moderate_new_deviation = deviation * (1.0 - moderate_removal)
        moderate_magnitude = torch.maximum(
            median + signs * moderate_new_deviation,
            torch.zeros_like(median)
        )

        # Apply thresholds:
        # - Non-outliers (ratio <= 1): keep original
        # - Moderate outliers (1 < ratio <= 2): scaled removal
        # - Severe outliers (ratio > 2): snap to median
        new_magnitude = torch.where(
            outlier_ratio > severe_threshold,
            median,  # Severe: full removal
            torch.where(
                outlier_ratio > 1.0,
                moderate_magnitude,  # Moderate: scaled removal
                magnitude  # Non-outlier: keep original
            )
        )

        # Reconstruct complex value
        return new_magnitude * torch.exp(1j * phase)

    def _apply_low_amp_protection(
        self,
        original_magnitude: torch.Tensor,
        new_magnitude: torch.Tensor,
        median: torch.Tensor
    ) -> torch.Tensor:
        """
        Protect low-amplitude samples from inflation.

        Prevents the algorithm from increasing amplitude of isolated
        low-amplitude samples (which may be genuine signal gaps).

        Args:
            original_magnitude: Original magnitudes before thresholding
            new_magnitude: Magnitudes after thresholding
            median: Spatial median

        Returns:
            Protected magnitudes
        """
        # Identify low-amplitude regions
        low_amp_threshold = median * self.low_amp_factor
        low_amp_mask = original_magnitude < low_amp_threshold

        # Prevent inflation: if original was low and new is higher, keep original
        inflation_mask = low_amp_mask & (new_magnitude > original_magnitude)

        protected_magnitude = torch.where(
            inflation_mask,
            original_magnitude,  # Keep original low amplitude
            new_magnitude
        )

        return protected_magnitude

    def apply_batch_mad_thresholding(
        self,
        tf_batch: np.ndarray,
        spatial_dim: int = 0
    ) -> np.ndarray:
        """
        Apply MAD-based thresholding to batch of TF coefficients.

        Processes all traces in the batch simultaneously on GPU.
        Each trace is thresholded using the spatial ensemble statistics.

        Supports multiple threshold modes:
        - 'soft': Classical soft thresholding (partial removal)
        - 'hard': Full removal for outliers, preserve non-outliers
        - 'scaled': Progressive removal based on outlier severity
        - 'adaptive': Combined hard + scaled for different severity levels

        Args:
            tf_batch: TF coefficients for all traces (n_traces, n_freqs, n_times)
            spatial_dim: Dimension along which to compute MAD (default: 0 = traces)

        Returns:
            Thresholded TF coefficients (n_traces, n_freqs, n_times)
        """
        n_traces, n_freqs, n_times = tf_batch.shape

        logger.debug(
            f"Batch MAD thresholding on GPU: {n_traces} traces × {n_freqs} freqs × {n_times} times, "
            f"mode={self.threshold_mode}"
        )

        # Transfer entire batch to GPU
        tf_gpu = torch.from_numpy(tf_batch).to(self.device)

        # Compute spatial statistics across traces
        magnitudes = torch.abs(tf_gpu)  # (n_traces, n_freqs, n_times)
        phase = torch.angle(tf_gpu)

        # Median across traces
        median_mag = compute_median_gpu(magnitudes, dim=spatial_dim, keepdim=True)
        # Shape: (1, n_freqs, n_times)

        # MAD computation
        abs_dev = torch.abs(magnitudes - median_mag)
        mad = compute_median_gpu(abs_dev, dim=spatial_dim, keepdim=True) * 1.4826
        # Shape: (1, n_freqs, n_times)

        # Prevent MAD=0 from causing issues
        min_mad = torch.maximum(0.01 * median_mag, torch.tensor(1e-10, device=self.device))
        mad = torch.maximum(mad, min_mad)

        # Outlier threshold
        outlier_threshold = self.threshold_k * mad
        # Shape: (1, n_freqs, n_times) - broadcasts to all traces

        # Compute deviation from median
        deviation = torch.abs(magnitudes - median_mag)

        # Apply thresholding based on mode
        if self.threshold_mode == 'hard':
            # Option A: Hard threshold - outliers snap to median, non-outliers keep original
            new_magnitude = torch.where(
                deviation > outlier_threshold,
                median_mag.expand_as(magnitudes),  # Snap to median
                magnitudes  # Keep original
            )

        elif self.threshold_mode == 'scaled':
            # Option B: Progressive removal based on severity
            outlier_ratio = deviation / (outlier_threshold + 1e-10)
            excess_ratio = torch.clamp(outlier_ratio - 1.0, min=0.0)
            removal_factor = 1.0 - 1.0 / (1.0 + excess_ratio ** 2)

            new_deviation = deviation * (1.0 - removal_factor)
            signs = torch.where(magnitudes >= median_mag, 1.0, -1.0)

            new_magnitude = torch.where(
                outlier_ratio > 1.0,
                torch.maximum(median_mag + signs * new_deviation, torch.zeros_like(magnitudes)),
                magnitudes  # Keep original for non-outliers
            )

        elif self.threshold_mode == 'adaptive':
            # Combined: hard for severe, scaled for moderate, preserve non-outliers
            outlier_ratio = deviation / (outlier_threshold + 1e-10)
            severe_threshold = 2.0

            # Scaled removal for moderate outliers
            moderate_removal = torch.clamp(
                (outlier_ratio - 1.0) / (severe_threshold - 1.0), 0.0, 1.0
            )
            signs = torch.where(magnitudes >= median_mag, 1.0, -1.0)
            moderate_new_deviation = deviation * (1.0 - moderate_removal)
            moderate_magnitude = torch.maximum(
                median_mag + signs * moderate_new_deviation,
                torch.zeros_like(magnitudes)
            )

            new_magnitude = torch.where(
                outlier_ratio > severe_threshold,
                median_mag.expand_as(magnitudes),  # Severe: full removal
                torch.where(
                    outlier_ratio > 1.0,
                    moderate_magnitude,  # Moderate: scaled
                    magnitudes  # Non-outlier: keep original
                )
            )

        else:  # 'soft' (default/legacy) or threshold_type-based
            # Legacy behavior using threshold_type
            if self.threshold_type == 'soft':
                new_deviation = torch.maximum(
                    deviation - outlier_threshold,
                    torch.zeros_like(deviation)
                )
            else:  # garrote
                new_deviation = torch.where(
                    deviation > outlier_threshold,
                    deviation - (outlier_threshold**2 / (deviation + 1e-10)),
                    deviation
                )

            signs = torch.where(magnitudes >= median_mag, 1.0, -1.0)
            new_magnitude = torch.maximum(
                median_mag + signs * new_deviation,
                torch.zeros_like(magnitudes)
            )

        # Apply low-amplitude protection if enabled
        if self.low_amp_protection:
            new_magnitude = self._apply_low_amp_protection(
                magnitudes, new_magnitude, median_mag.expand_as(magnitudes)
            )

        # Reconstruct complex values
        tf_denoised = new_magnitude * torch.exp(1j * phase)

        # Transfer back to CPU
        return tensor_to_numpy(tf_denoised)

    def apply_global_threshold(
        self,
        tf_coeffs: np.ndarray,
        threshold: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Apply global threshold to TF coefficients.

        Args:
            tf_coeffs: TF coefficients (complex array)
            threshold: Threshold value(s)

        Returns:
            Thresholded coefficients
        """
        # Transfer to GPU
        tf_gpu = torch.from_numpy(tf_coeffs).to(self.device)

        if isinstance(threshold, np.ndarray):
            threshold_gpu = torch.from_numpy(threshold).to(self.device)
        else:
            threshold_gpu = threshold

        # Apply thresholding
        if self.threshold_type == 'soft':
            result_gpu = soft_threshold_gpu(tf_gpu, threshold_gpu)
        else:  # garrote
            result_gpu = garrote_threshold_gpu(tf_gpu, threshold_gpu)

        # Transfer back to CPU
        return tensor_to_numpy(result_gpu)

    def compute_spatial_statistics(
        self,
        tf_ensemble: np.ndarray,
        spatial_dim: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spatial statistics (median, MAD, threshold) on GPU.

        Args:
            tf_ensemble: TF coefficients for ensemble (n_traces, n_freqs, n_times)
            spatial_dim: Dimension to compute statistics along

        Returns:
            Tuple of (median, MAD, threshold) arrays
        """
        # Transfer to GPU
        tf_gpu = torch.from_numpy(tf_ensemble).to(self.device)

        # Compute magnitudes
        magnitudes = torch.abs(tf_gpu)

        # Compute median
        median = compute_median_gpu(magnitudes, dim=spatial_dim, keepdim=False)

        # Compute MAD
        median_expanded = median.unsqueeze(spatial_dim)
        abs_dev = torch.abs(magnitudes - median_expanded)
        mad = compute_median_gpu(abs_dev, dim=spatial_dim, keepdim=False) * 1.4826

        # Compute outlier threshold (deviation-based, not magnitude-based!)
        threshold = self.threshold_k * mad

        # Transfer back to CPU
        median_np = tensor_to_numpy(median)
        mad_np = tensor_to_numpy(mad)
        threshold_np = tensor_to_numpy(threshold)

        return median_np, mad_np, threshold_np


def apply_mad_threshold_gpu(
    tf_center: np.ndarray,
    tf_ensemble: np.ndarray,
    device: torch.device,
    threshold_k: float = 3.0,
    threshold_type: Literal['soft', 'garrote'] = 'soft',
    spatial_dim: int = 0
) -> np.ndarray:
    """
    Convenience function for MAD thresholding on GPU.

    Args:
        tf_center: TF coefficients for center trace
        tf_ensemble: TF coefficients for spatial ensemble
        device: PyTorch device
        threshold_k: MAD threshold multiplier
        threshold_type: Thresholding type
        spatial_dim: Spatial dimension for MAD computation

    Returns:
        Thresholded TF coefficients
    """
    thresholder = ThresholdingGPU(
        device=device,
        threshold_k=threshold_k,
        threshold_type=threshold_type
    )

    return thresholder.apply_mad_thresholding(
        tf_center,
        tf_ensemble,
        spatial_dim=spatial_dim
    )
