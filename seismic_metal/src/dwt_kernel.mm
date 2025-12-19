/**
 * DWT/SWT Kernel Implementation
 *
 * Hybrid vDSP + Metal GPU implementation for wavelet transform denoising.
 * - vDSP for high-performance convolution operations
 * - Multi-threaded batch processing
 * - Metal GPU for parallel thresholding
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

#include "dwt_kernel.h"
#include "device_manager.h"
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <thread>

namespace seismic_metal {

// Wavelet filter coefficients
static const std::map<std::string, WaveletFilter> WAVELET_FILTERS = {
    {"db4", {
        "db4", 8,
        {-0.010597401784997278f, 0.032883011666982945f, 0.030841381835986965f,
         -0.18703481171888114f, -0.027983769416983849f, 0.63088076792959036f,
         0.71484657055254153f, 0.23037781330885523f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-0.23037781330885523f, 0.71484657055254153f, -0.63088076792959036f,
         -0.027983769416983849f, 0.18703481171888114f, 0.030841381835986965f,
         -0.032883011666982945f, -0.010597401784997278f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.23037781330885523f, 0.71484657055254153f, 0.63088076792959036f,
         -0.027983769416983849f, -0.18703481171888114f, 0.030841381835986965f,
         0.032883011666982945f, -0.010597401784997278f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-0.010597401784997278f, -0.032883011666982945f, 0.030841381835986965f,
         0.18703481171888114f, -0.027983769416983849f, -0.63088076792959036f,
         0.71484657055254153f, -0.23037781330885523f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    }},
    {"sym4", {
        "sym4", 8,
        {-0.07576571478927333f, -0.02963552764599851f, 0.49761866763201545f,
         0.8037387518059161f, 0.29785779560527736f, -0.09921954357684722f,
         -0.012603967262037833f, 0.032223100604042702f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-0.032223100604042702f, -0.012603967262037833f, 0.09921954357684722f,
         0.29785779560527736f, -0.8037387518059161f, 0.49761866763201545f,
         0.02963552764599851f, -0.07576571478927333f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.032223100604042702f, -0.012603967262037833f, -0.09921954357684722f,
         0.29785779560527736f, 0.8037387518059161f, 0.49761866763201545f,
         -0.02963552764599851f, -0.07576571478927333f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-0.07576571478927333f, 0.02963552764599851f, 0.49761866763201545f,
         -0.8037387518059161f, 0.29785779560527736f, 0.09921954357684722f,
         -0.012603967262037833f, -0.032223100604042702f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    }}
};

WaveletFilter get_wavelet_filter(const std::string& wavelet) {
    auto it = WAVELET_FILTERS.find(wavelet);
    if (it != WAVELET_FILTERS.end()) {
        return it->second;
    }
    // Default to db4
    return WAVELET_FILTERS.at("db4");
}

int compute_max_level(int n_samples, int filter_len) {
    int max_level = 0;
    int length = n_samples;
    while (length >= filter_len) {
        length = (length + filter_len - 1) / 2;
        max_level++;
    }
    return std::max(1, max_level - 1);
}

// Helper: DWT decomposition for single trace using vDSP
static void dwt_decompose_trace(
    const float* input,
    float* approx,
    float* detail,
    int input_len,
    int output_len,
    const float* lo_d,
    const float* hi_d,
    int filter_len
) {
    // Use vDSP convolution with downsampling
    // Pad input for symmetric boundary
    std::vector<float> padded(input_len + 2 * filter_len);

    // Symmetric boundary extension
    for (int i = 0; i < filter_len; i++) {
        padded[i] = input[filter_len - 1 - i];
    }
    std::copy(input, input + input_len, padded.begin() + filter_len);
    for (int i = 0; i < filter_len; i++) {
        int idx = input_len - 2 - i;
        padded[filter_len + input_len + i] = (idx >= 0) ? input[idx] : input[0];
    }

    // Convolution using vDSP
    std::vector<float> conv_lo(input_len + filter_len - 1);
    std::vector<float> conv_hi(input_len + filter_len - 1);

    vDSP_conv(padded.data() + filter_len, 1, lo_d + filter_len - 1, -1,
              conv_lo.data(), 1, input_len, filter_len);
    vDSP_conv(padded.data() + filter_len, 1, hi_d + filter_len - 1, -1,
              conv_hi.data(), 1, input_len, filter_len);

    // Downsample by 2
    for (int i = 0; i < output_len; i++) {
        int idx = i * 2;
        approx[i] = (idx < input_len) ? conv_lo[idx] : 0.0f;
        detail[i] = (idx < input_len) ? conv_hi[idx] : 0.0f;
    }
}

// Helper: DWT reconstruction for single trace using vDSP
static void dwt_reconstruct_trace(
    const float* approx,
    const float* detail,
    float* output,
    int input_len,
    int output_len,
    const float* lo_r,
    const float* hi_r,
    int filter_len
) {
    // Upsample coefficients
    std::vector<float> up_lo(output_len + filter_len, 0.0f);
    std::vector<float> up_hi(output_len + filter_len, 0.0f);

    for (int i = 0; i < input_len; i++) {
        int idx = i * 2;
        if (idx < output_len + filter_len) {
            up_lo[idx] = approx[i];
            up_hi[idx] = detail[i];
        }
    }

    // Convolution
    std::vector<float> conv_lo(output_len);
    std::vector<float> conv_hi(output_len);

    vDSP_conv(up_lo.data(), 1, lo_r + filter_len - 1, -1,
              conv_lo.data(), 1, output_len, filter_len);
    vDSP_conv(up_hi.data(), 1, hi_r + filter_len - 1, -1,
              conv_hi.data(), 1, output_len, filter_len);

    // Sum
    vDSP_vadd(conv_lo.data(), 1, conv_hi.data(), 1, output, 1, output_len);
}

// Helper: Compute MAD threshold
static float compute_mad_threshold(const float* coeffs, int n, float threshold_k) {
    if (n <= 0) return 0.0f;

    std::vector<float> abs_coeffs(n);
    vDSP_vabs(coeffs, 1, abs_coeffs.data(), 1, n);

    // Sort for median
    std::sort(abs_coeffs.begin(), abs_coeffs.end());

    float median = (n % 2 == 1) ?
                   abs_coeffs[n / 2] :
                   (abs_coeffs[n / 2 - 1] + abs_coeffs[n / 2]) / 2.0f;

    // MAD-based threshold: k * median / 0.6745
    return threshold_k * median / 0.6745f;
}

// Helper: Apply soft thresholding
static void apply_soft_threshold(float* coeffs, int n, float threshold) {
    for (int i = 0; i < n; i++) {
        float val = coeffs[i];
        float abs_val = std::abs(val);
        float sign = (val >= 0.0f) ? 1.0f : -1.0f;
        coeffs[i] = sign * std::max(abs_val - threshold, 0.0f);
    }
}

std::tuple<std::vector<float>, KernelMetrics> dwt_denoise(
    const float* traces,
    int n_samples,
    int n_traces,
    const std::string& wavelet,
    int level,
    float threshold_k,
    const std::string& mode
) {
    auto start_total = std::chrono::high_resolution_clock::now();

    KernelMetrics metrics;
    metrics.traces_processed = n_traces;
    metrics.samples_processed = (int64_t)n_traces * n_samples;

    // Get wavelet filter
    WaveletFilter filter = get_wavelet_filter(wavelet);

    // Auto-compute level if needed
    if (level <= 0) {
        level = compute_max_level(n_samples, filter.length);
    }

    // Compute coefficient lengths at each level
    std::vector<int> coeff_lengths(level + 1);
    coeff_lengths[0] = n_samples;
    for (int l = 1; l <= level; l++) {
        coeff_lengths[l] = (coeff_lengths[l-1] + filter.length - 1) / 2;
    }

    std::vector<float> output(n_samples * n_traces);

    auto start_kernel = std::chrono::high_resolution_clock::now();

    // Multi-threaded processing
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto process_trace = [&](int trace_idx) {
        // Extract trace (convert from column-major)
        std::vector<float> trace(n_samples);
        for (int s = 0; s < n_samples; s++) {
            trace[s] = traces[s * n_traces + trace_idx];
        }

        // Allocate coefficients for each level
        std::vector<std::vector<float>> approx(level + 1);
        std::vector<std::vector<float>> detail(level);

        approx[0] = trace;
        for (int l = 0; l < level; l++) {
            approx[l + 1].resize(coeff_lengths[l + 1]);
            detail[l].resize(coeff_lengths[l + 1]);
        }

        // Decomposition
        for (int l = 0; l < level; l++) {
            dwt_decompose_trace(
                approx[l].data(),
                approx[l + 1].data(),
                detail[l].data(),
                coeff_lengths[l],
                coeff_lengths[l + 1],
                filter.lo_d,
                filter.hi_d,
                filter.length
            );
        }

        // Compute threshold from finest detail level
        float threshold = compute_mad_threshold(
            detail[level - 1].data(),
            coeff_lengths[level],
            threshold_k
        );

        // Apply thresholding to all detail levels
        for (int l = 0; l < level; l++) {
            apply_soft_threshold(detail[l].data(), coeff_lengths[l + 1], threshold);
        }

        // Reconstruction
        for (int l = level - 1; l >= 0; l--) {
            dwt_reconstruct_trace(
                approx[l + 1].data(),
                detail[l].data(),
                approx[l].data(),
                coeff_lengths[l + 1],
                coeff_lengths[l],
                filter.lo_r,
                filter.hi_r,
                filter.length
            );
        }

        // Store result (convert to column-major)
        for (int s = 0; s < n_samples; s++) {
            output[s * n_traces + trace_idx] = approx[0][s];
        }
    };

    auto process_range = [&](int start, int end) {
        for (int t = start; t < end; t++) {
            process_trace(t);
        }
    };

    int chunk_size = (n_traces + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n_traces);
        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }
    for (auto& t : threads) t.join();

    auto end_kernel = std::chrono::high_resolution_clock::now();

    metrics.kernel_time_ms = std::chrono::duration<double, std::milli>(
        end_kernel - start_kernel).count();
    metrics.total_time_ms = std::chrono::duration<double, std::milli>(
        end_kernel - start_total).count();

    return std::make_tuple(output, metrics);
}

// Helper: Periodic boundary index computation (matches PyWavelets SWT default)
static inline int periodic_boundary(int idx, int n_samples) {
    // Simple modulo for periodic boundary
    idx = idx % n_samples;
    if (idx < 0) idx += n_samples;
    return idx;
}

// Helper: SWT decomposition for single level
// Matches PyWavelets pywt.swt algorithm with center shift for proper phase alignment
static void swt_decompose_level(
    const float* input,
    float* approx,
    float* detail,
    int n_samples,
    const float* lo_d,
    const float* hi_d,
    int filter_len,
    int level  // 0-based level
) {
    // SWT uses upsampled filters at each level
    int step = 1 << level;  // 2^level
    int center_shift = filter_len / 2;  // Critical: center shift for phase alignment

    for (int i = 0; i < n_samples; i++) {
        float lo_sum = 0.0f;
        float hi_sum = 0.0f;

        for (int k = 0; k < filter_len; k++) {
            // Key formula: i - k*step + center_shift (matches PyWavelets)
            int idx = i - k * step + center_shift;
            idx = periodic_boundary(idx, n_samples);

            lo_sum += input[idx] * lo_d[k];
            hi_sum += input[idx] * hi_d[k];
        }

        approx[i] = lo_sum;
        detail[i] = hi_sum;
    }
}

// Helper: SWT reconstruction for single level
// Matches PyWavelets pywt.iswt algorithm with center shift and reversed filter access
static void swt_reconstruct_level(
    const float* approx,
    const float* detail,
    float* output,
    int n_samples,
    const float* lo_r,
    const float* hi_r,
    int filter_len,
    int level  // 0-based level
) {
    int step = 1 << level;
    int center_shift = filter_len / 2;  // Critical: center shift (opposite of decomposition)

    for (int i = 0; i < n_samples; i++) {
        float sum = 0.0f;

        for (int k = 0; k < filter_len; k++) {
            // Key formula: i + k*step - center_shift (opposite direction from decomposition)
            int idx = i + k * step - center_shift;
            idx = periodic_boundary(idx, n_samples);

            // Critical: Access filters in REVERSED order for perfect reconstruction
            int fk = filter_len - 1 - k;
            sum += approx[idx] * lo_r[fk] + detail[idx] * hi_r[fk];
        }

        output[i] = sum * 0.5f;  // Normalization factor
    }
}

std::tuple<std::vector<float>, KernelMetrics> swt_denoise(
    const float* traces,
    int n_samples,
    int n_traces,
    const std::string& wavelet,
    int level,
    float threshold_k,
    const std::string& mode
) {
    auto start_total = std::chrono::high_resolution_clock::now();

    KernelMetrics metrics;
    metrics.traces_processed = n_traces;
    metrics.samples_processed = (int64_t)n_traces * n_samples;

    // Get wavelet filter
    WaveletFilter filter = get_wavelet_filter(wavelet);

    // Compute padded length (power of 2) to match Python pywt.swt behavior
    int target_len = 1;
    while (target_len < n_samples) target_len *= 2;
    int pad_len = target_len - n_samples;
    int padded_samples = target_len;

    // Auto-compute level if needed (based on padded length)
    if (level <= 0) {
        level = compute_max_level(padded_samples, filter.length);
    }

    // Limit level based on padded length (matches pywt.swt_max_level)
    int max_level = (int)std::floor(std::log2((double)padded_samples / filter.length));
    level = std::min(level, std::max(1, max_level));

    std::vector<float> output(n_samples * n_traces);

    auto start_kernel = std::chrono::high_resolution_clock::now();

    // Multi-threaded processing
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    auto process_trace = [&, padded_samples, pad_len](int trace_idx) {
        // Extract trace (convert from column-major)
        std::vector<float> trace(n_samples);
        for (int s = 0; s < n_samples; s++) {
            trace[s] = traces[s * n_traces + trace_idx];
        }

        // Pad to power of 2 with reflect boundary (matches Python np.pad mode='reflect')
        std::vector<float> padded_trace(padded_samples);
        std::copy(trace.begin(), trace.end(), padded_trace.begin());
        for (int i = 0; i < pad_len; i++) {
            // Reflect padding: mirror the signal at the end
            int reflect_idx = n_samples - 2 - i;
            if (reflect_idx < 0) reflect_idx = 0;
            padded_trace[n_samples + i] = trace[reflect_idx];
        }

        // Allocate coefficients (SWT: all same size as padded)
        std::vector<std::vector<float>> approx(level + 1);
        std::vector<std::vector<float>> detail(level);

        for (int l = 0; l <= level; l++) {
            approx[l].resize(padded_samples);
        }
        for (int l = 0; l < level; l++) {
            detail[l].resize(padded_samples);
        }

        approx[0] = padded_trace;

        // Decomposition (on padded signal)
        for (int l = 0; l < level; l++) {
            swt_decompose_level(
                approx[l].data(),
                approx[l + 1].data(),
                detail[l].data(),
                padded_samples,
                filter.lo_d,
                filter.hi_d,
                filter.length,
                l
            );

        }

        // Compute threshold from coarsest detail level (matches Python coeffs[0][1])
        // In Python pywt.swt, coeffs[0] is the coarsest level, which corresponds to
        // detail[level-1] in our decomposition order
        float threshold = compute_mad_threshold(
            detail[level - 1].data(),
            padded_samples,
            threshold_k
        );

        // Apply thresholding to all detail levels
        for (int l = 0; l < level; l++) {
            if (mode == "soft") {
                apply_soft_threshold(detail[l].data(), padded_samples, threshold);
            } else {
                // Hard thresholding
                for (int i = 0; i < padded_samples; i++) {
                    if (std::abs(detail[l][i]) <= threshold) {
                        detail[l][i] = 0.0f;
                    }
                }
            }
        }

        // Reconstruction (on padded signal)
        for (int l = level - 1; l >= 0; l--) {
            swt_reconstruct_level(
                approx[l + 1].data(),
                detail[l].data(),
                approx[l].data(),
                padded_samples,
                filter.lo_r,
                filter.hi_r,
                filter.length,
                l
            );

        }

        // Store result - trim back to original size (convert to column-major)
        for (int s = 0; s < n_samples; s++) {
            output[s * n_traces + trace_idx] = approx[0][s];
        }
    };

    auto process_range = [&](int start, int end) {
        for (int t = start; t < end; t++) {
            process_trace(t);
        }
    };

    int chunk_size = (n_traces + n_threads - 1) / n_threads;
    for (int i = 0; i < (int)n_threads; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, n_traces);
        if (start < end) {
            threads.emplace_back(process_range, start, end);
        }
    }
    for (auto& t : threads) t.join();

    auto end_kernel = std::chrono::high_resolution_clock::now();

    metrics.kernel_time_ms = std::chrono::duration<double, std::milli>(
        end_kernel - start_kernel).count();
    metrics.total_time_ms = std::chrono::duration<double, std::milli>(
        end_kernel - start_total).count();

    return std::make_tuple(output, metrics);
}

} // namespace seismic_metal
