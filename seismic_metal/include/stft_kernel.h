/**
 * STFT/Gabor Kernel Interface
 *
 * Metal GPU kernels for STFT and Gabor Transform denoising.
 * Includes optimized batch processing and GPU-accelerated MAD computation.
 */

#ifndef SEISMIC_METAL_STFT_KERNEL_H
#define SEISMIC_METAL_STFT_KERNEL_H

#include "common_types.h"
#include <vector>
#include <tuple>

namespace seismic_metal {

/**
 * Apply STFT denoising to traces using Metal GPU.
 *
 * Uses batch STFT processing and GPU-accelerated MAD thresholding
 * to significantly speed up time-frequency domain denoising.
 *
 * @param traces Input traces (n_samples x n_traces), row-major
 * @param n_samples Number of samples per trace
 * @param n_traces Number of traces
 * @param nperseg STFT window size (default 64)
 * @param noverlap STFT overlap (default 32)
 * @param aperture Spatial aperture for MAD computation
 * @param threshold_k MAD threshold multiplier
 * @param fmin Minimum frequency to process (Hz, 0 = all)
 * @param fmax Maximum frequency to process (Hz, 0 = Nyquist)
 * @param sample_rate Sample rate in Hz
 * @return Tuple of (denoised traces, metrics)
 */
std::tuple<std::vector<float>, KernelMetrics> stft_denoise(
    const float* traces,
    int n_samples,
    int n_traces,
    int nperseg = 64,
    int noverlap = 32,
    int aperture = 7,
    float threshold_k = 3.0f,
    float fmin = 0.0f,
    float fmax = 0.0f,
    float sample_rate = 500.0f,
    bool low_amp_protection = true,
    float low_amp_factor = 0.3f
);

/**
 * Apply Gabor Transform denoising to traces using Metal GPU.
 *
 * Gabor Transform uses STFT with Gaussian windows for optimal
 * time-frequency localization.
 *
 * @param traces Input traces (n_samples x n_traces), row-major
 * @param n_samples Number of samples per trace
 * @param n_traces Number of traces
 * @param window_size Gabor window size (default 64)
 * @param sigma Gaussian sigma (0 = auto = window_size/6)
 * @param overlap_pct Overlap percentage (default 75)
 * @param aperture Spatial aperture
 * @param threshold_k MAD threshold multiplier
 * @param fmin Minimum frequency (Hz)
 * @param fmax Maximum frequency (Hz)
 * @param sample_rate Sample rate in Hz
 * @return Tuple of (denoised traces, metrics)
 */
std::tuple<std::vector<float>, KernelMetrics> gabor_denoise(
    const float* traces,
    int n_samples,
    int n_traces,
    int window_size = 64,
    float sigma = 0.0f,
    float overlap_pct = 75.0f,
    int aperture = 7,
    float threshold_k = 3.0f,
    float fmin = 0.0f,
    float fmax = 0.0f,
    float sample_rate = 500.0f
);

/**
 * Compute MAD threshold using GPU.
 *
 * This is a critical kernel that computes median and MAD across
 * spatial aperture for each time-frequency bin.
 *
 * @param amplitudes Amplitude values (n_traces x n_freqs x n_times)
 * @param n_traces Number of traces in aperture
 * @param n_freqs Number of frequency bins
 * @param n_times Number of time bins
 * @param threshold_k MAD multiplier
 * @return Tuple of (median, mad, threshold) arrays
 */
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
compute_mad_threshold_gpu(
    const float* amplitudes,
    int n_traces,
    int n_freqs,
    int n_times,
    float threshold_k
);

/**
 * Create Hann window for STFT.
 */
std::vector<float> create_hann_window(int size);

/**
 * Create Gaussian window for Gabor Transform.
 */
std::vector<float> create_gaussian_window(int size, float sigma = 0.0f);

} // namespace seismic_metal

#endif // SEISMIC_METAL_STFT_KERNEL_H
