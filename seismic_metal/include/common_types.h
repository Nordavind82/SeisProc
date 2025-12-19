/**
 * Common Types for SeisProc Metal Kernels
 *
 * Shared data structures and constants used across all Metal kernels.
 */

#ifndef SEISMIC_METAL_COMMON_TYPES_H
#define SEISMIC_METAL_COMMON_TYPES_H

#include <cstdint>
#include <string>

namespace seismic_metal {

/**
 * Processing parameters for denoising kernels.
 */
struct DenoiseParams {
    float threshold_k;      // MAD threshold multiplier (typically 3.0)
    int32_t aperture;       // Spatial aperture size
    int32_t mode;           // 0=soft, 1=hard threshold
    float fmin;             // Minimum frequency (Hz)
    float fmax;             // Maximum frequency (Hz)
};

/**
 * DWT/SWT processing parameters.
 */
struct DWTParams {
    int32_t level;          // Decomposition level
    int32_t filter_len;     // Wavelet filter length
    int32_t transform_type; // 0=DWT, 1=SWT
    float threshold_k;
    int32_t threshold_mode; // 0=soft, 1=hard
};

/**
 * STFT processing parameters.
 */
struct STFTParams {
    int32_t nperseg;        // Window size
    int32_t noverlap;       // Overlap samples
    int32_t window_type;    // 0=hann, 1=gaussian
    float sigma;            // Gaussian sigma (for Gabor)
    int32_t aperture;       // Spatial aperture
    float threshold_k;
};

/**
 * FKK filter parameters.
 */
struct FKKParams {
    float dt;               // Time sample interval (seconds)
    float dx;               // Inline spacing (meters)
    float dy;               // Crossline spacing (meters)
    float v_min;            // Minimum velocity to filter
    float v_max;            // Maximum velocity to filter
    int32_t filter_mode;    // 0=reject, 1=pass
    int32_t preserve_dc;    // Preserve DC component
};

/**
 * Kernel execution metrics.
 */
struct KernelMetrics {
    double kernel_time_ms;      // GPU kernel execution time
    double transfer_time_ms;    // Data transfer time
    double total_time_ms;       // Total processing time
    int64_t traces_processed;
    int64_t samples_processed;
};

/**
 * Wavelet filter coefficients.
 * Pre-computed for common wavelets (db4, sym4, coif4, etc.)
 */
struct WaveletFilter {
    std::string name;
    int32_t length;
    float lo_d[20];         // Low-pass decomposition
    float hi_d[20];         // High-pass decomposition
    float lo_r[20];         // Low-pass reconstruction
    float hi_r[20];         // High-pass reconstruction
};

// Pre-defined wavelet filters
namespace wavelets {

// Daubechies 4 (db4)
constexpr float DB4_LO_D[] = {
    -0.010597401784997278f, 0.032883011666982945f,
    0.030841381835986965f, -0.18703481171888114f,
    -0.027983769416983849f, 0.63088076792959036f,
    0.71484657055254153f, 0.23037781330885523f
};

constexpr float DB4_HI_D[] = {
    -0.23037781330885523f, 0.71484657055254153f,
    -0.63088076792959036f, -0.027983769416983849f,
    0.18703481171888114f, 0.030841381835986965f,
    -0.032883011666982945f, -0.010597401784997278f
};

constexpr int DB4_LEN = 8;

} // namespace wavelets

} // namespace seismic_metal

#endif // SEISMIC_METAL_COMMON_TYPES_H
