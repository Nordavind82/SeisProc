/**
 * DWT/SWT Kernel Interface
 *
 * Metal GPU kernels for Discrete and Stationary Wavelet Transform denoising.
 */

#ifndef SEISMIC_METAL_DWT_KERNEL_H
#define SEISMIC_METAL_DWT_KERNEL_H

#include "common_types.h"
#include <vector>
#include <tuple>

namespace seismic_metal {

/**
 * Apply DWT denoising to traces using Metal GPU.
 *
 * @param traces Input traces (n_samples x n_traces), row-major
 * @param n_samples Number of samples per trace
 * @param n_traces Number of traces
 * @param wavelet Wavelet name ("db4", "sym4", "coif4")
 * @param level Decomposition level (0 = auto)
 * @param threshold_k MAD threshold multiplier
 * @param mode Threshold mode ("soft" or "hard")
 * @return Tuple of (denoised traces, metrics)
 */
std::tuple<std::vector<float>, KernelMetrics> dwt_denoise(
    const float* traces,
    int n_samples,
    int n_traces,
    const std::string& wavelet = "db4",
    int level = 5,
    float threshold_k = 3.0f,
    const std::string& mode = "soft"
);

/**
 * Apply SWT denoising to traces using Metal GPU.
 *
 * SWT (Stationary Wavelet Transform) is translation-invariant and
 * produces better denoising results but is computationally more expensive.
 * The Metal kernel provides significant speedup for reconstruction.
 *
 * @param traces Input traces (n_samples x n_traces), row-major
 * @param n_samples Number of samples per trace (will be padded to power of 2)
 * @param n_traces Number of traces
 * @param wavelet Wavelet name
 * @param level Decomposition level
 * @param threshold_k MAD threshold multiplier
 * @param mode Threshold mode
 * @return Tuple of (denoised traces, metrics)
 */
std::tuple<std::vector<float>, KernelMetrics> swt_denoise(
    const float* traces,
    int n_samples,
    int n_traces,
    const std::string& wavelet = "db4",
    int level = 5,
    float threshold_k = 3.0f,
    const std::string& mode = "soft"
);

/**
 * Get wavelet filter coefficients.
 *
 * @param wavelet Wavelet name
 * @return WaveletFilter struct with coefficients
 */
WaveletFilter get_wavelet_filter(const std::string& wavelet);

/**
 * Compute optimal decomposition level for signal length.
 *
 * @param n_samples Signal length
 * @param filter_len Wavelet filter length
 * @return Recommended decomposition level
 */
int compute_max_level(int n_samples, int filter_len);

} // namespace seismic_metal

#endif // SEISMIC_METAL_DWT_KERNEL_H
