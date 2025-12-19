/**
 * MAD Threshold Kernel
 *
 * GPU-accelerated Median Absolute Deviation computation and thresholding.
 * This is the critical kernel for STFT and Gabor denoising (52-56% of processing time).
 *
 * Uses parallel sorting (bitonic sort) for efficient median computation.
 */

#include <metal_stdlib>
using namespace metal;

// Maximum traces in spatial aperture (must match threadgroup size)
constant uint MAX_APERTURE = 32;

/**
 * Compute MAD threshold for a single time-frequency bin.
 *
 * Performs:
 * 1. Parallel sort of amplitudes across spatial aperture
 * 2. Median computation
 * 3. MAD (Median Absolute Deviation) computation
 * 4. Threshold calculation: threshold = k * MAD * 1.4826
 *
 * Grid: (n_freqs, n_times, 1)
 * Threadgroup: (aperture_size, 1, 1)
 */
kernel void compute_mad_threshold(
    device const float* amplitudes [[buffer(0)]],    // [n_traces, n_freqs, n_times]
    device float* median_out [[buffer(1)]],          // [n_freqs, n_times]
    device float* mad_out [[buffer(2)]],             // [n_freqs, n_times]
    device float* threshold_out [[buffer(3)]],       // [n_freqs, n_times]
    constant uint& n_traces [[buffer(4)]],
    constant uint& n_freqs [[buffer(5)]],
    constant uint& n_times [[buffer(6)]],
    constant float& threshold_k [[buffer(7)]],
    constant uint& threadgroup_size [[buffer(8)]],   // Pass threadgroup size as constant
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint freq_idx = gid.x;
    uint time_idx = gid.y;
    uint tg_size = threadgroup_size;

    if (freq_idx >= n_freqs || time_idx >= n_times) return;

    // Shared memory for sorting
    threadgroup float local_values[MAX_APERTURE];
    threadgroup float local_devs[MAX_APERTURE];

    uint tf_offset = freq_idx * n_times + time_idx;

    // Load amplitude value for this thread's trace
    float my_value = 0.0f;
    if (tid < n_traces) {
        my_value = amplitudes[tid * n_freqs * n_times + tf_offset];
    }
    local_values[tid] = my_value;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort for median computation
    // Only sort up to n_traces elements
    uint n = min(n_traces, tg_size);

    for (uint k = 2; k <= n; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            uint ixj = tid ^ j;
            if (ixj > tid && ixj < n) {
                bool ascending = ((tid & k) == 0);
                float a = local_values[tid];
                float b = local_values[ixj];
                if ((a > b) == ascending) {
                    local_values[tid] = b;
                    local_values[ixj] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // First thread computes median
    float median_val = 0.0f;
    if (tid == 0) {
        uint mid = n / 2;
        if (n % 2 == 1) {
            median_val = local_values[mid];
        } else {
            median_val = (local_values[mid - 1] + local_values[mid]) * 0.5f;
        }
        median_out[tf_offset] = median_val;
    }

    // Broadcast median to all threads
    threadgroup float shared_median;
    if (tid == 0) {
        shared_median = median_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    median_val = shared_median;

    // Compute absolute deviations
    if (tid < n_traces) {
        float orig_val = amplitudes[tid * n_freqs * n_times + tf_offset];
        local_devs[tid] = abs(orig_val - median_val);
    } else {
        local_devs[tid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Copy deviations to local_values for sorting
    local_values[tid] = local_devs[tid];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort for MAD median
    for (uint k = 2; k <= n; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            uint ixj = tid ^ j;
            if (ixj > tid && ixj < n) {
                bool ascending = ((tid & k) == 0);
                float a = local_values[tid];
                float b = local_values[ixj];
                if ((a > b) == ascending) {
                    local_values[tid] = b;
                    local_values[ixj] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // First thread computes MAD and threshold
    if (tid == 0) {
        uint mid = n / 2;
        float mad_val;
        if (n % 2 == 1) {
            mad_val = local_values[mid];
        } else {
            mad_val = (local_values[mid - 1] + local_values[mid]) * 0.5f;
        }

        // Scale MAD for Gaussian consistency
        float mad_scaled = mad_val * 1.4826f;
        mad_out[tf_offset] = mad_scaled;

        // Compute threshold
        float thresh = max(threshold_k * mad_scaled, 1e-10f);
        threshold_out[tf_offset] = thresh;
    }
}

/**
 * Apply soft thresholding to STFT coefficients.
 *
 * For each coefficient:
 *   if |coeff| > threshold:
 *     new_coeff = sign(coeff) * (|coeff| - threshold)
 *   else:
 *     new_coeff = 0
 *
 * Grid: (n_freqs, n_times, 1)
 */
kernel void apply_soft_threshold(
    device float2* stft_coeffs [[buffer(0)]],        // Complex coefficients [n_freqs, n_times]
    device const float* median [[buffer(1)]],        // [n_freqs, n_times]
    device const float* threshold [[buffer(2)]],     // [n_freqs, n_times]
    constant uint& n_freqs [[buffer(3)]],
    constant uint& n_times [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint freq_idx = gid.x;
    uint time_idx = gid.y;

    if (freq_idx >= n_freqs || time_idx >= n_times) return;

    uint idx = freq_idx * n_times + time_idx;

    float2 coeff = stft_coeffs[idx];
    float magnitude = length(coeff);
    float phase = atan2(coeff.y, coeff.x);

    float med = median[idx];
    float thresh = threshold[idx];

    // Compute deviation from median
    float deviation = abs(magnitude - med);

    // Soft threshold
    float new_magnitude;
    if (deviation > thresh) {
        float sign_val = (magnitude >= med) ? 1.0f : -1.0f;
        float new_deviation = max(deviation - thresh, 0.0f);
        new_magnitude = max(med + sign_val * new_deviation, 0.0f);
    } else {
        new_magnitude = magnitude;
    }

    // Reconstruct complex coefficient
    stft_coeffs[idx] = float2(new_magnitude * cos(phase), new_magnitude * sin(phase));
}

/**
 * Apply hard thresholding to STFT coefficients.
 *
 * For each coefficient:
 *   if deviation > threshold:
 *     new_coeff = median (snap to median)
 *   else:
 *     new_coeff = original
 */
kernel void apply_hard_threshold(
    device float2* stft_coeffs [[buffer(0)]],
    device const float* median [[buffer(1)]],
    device const float* threshold [[buffer(2)]],
    constant uint& n_freqs [[buffer(3)]],
    constant uint& n_times [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint freq_idx = gid.x;
    uint time_idx = gid.y;

    if (freq_idx >= n_freqs || time_idx >= n_times) return;

    uint idx = freq_idx * n_times + time_idx;

    float2 coeff = stft_coeffs[idx];
    float magnitude = length(coeff);
    float phase = atan2(coeff.y, coeff.x);

    float med = median[idx];
    float thresh = threshold[idx];

    float deviation = abs(magnitude - med);

    // Hard threshold: snap outliers to median
    float new_magnitude = (deviation > thresh) ? med : magnitude;

    stft_coeffs[idx] = float2(new_magnitude * cos(phase), new_magnitude * sin(phase));
}
