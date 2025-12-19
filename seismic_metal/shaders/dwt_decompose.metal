/**
 * DWT Decomposition Kernel
 *
 * Batch Discrete Wavelet Transform decomposition for all traces.
 */

#include <metal_stdlib>
using namespace metal;

constant uint MAX_FILTER_LEN = 20;

/**
 * Single-level DWT decomposition.
 *
 * Computes approximation and detail coefficients using convolution
 * and downsampling by 2.
 *
 * Grid: (n_traces, output_len, 1)
 */
kernel void dwt_decompose_level(
    device const float* input [[buffer(0)]],         // [n_traces, input_len]
    device float* approx [[buffer(1)]],              // [n_traces, output_len]
    device float* detail [[buffer(2)]],              // [n_traces, output_len]
    device const float* lo_d [[buffer(3)]],          // Low-pass filter
    device const float* hi_d [[buffer(4)]],          // High-pass filter
    constant uint& n_traces [[buffer(5)]],
    constant uint& input_len [[buffer(6)]],
    constant uint& output_len [[buffer(7)]],
    constant uint& filter_len [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint out_idx = gid.y;

    if (trace_idx >= n_traces || out_idx >= output_len) return;

    uint trace_offset = trace_idx * input_len;
    uint out_trace_offset = trace_idx * output_len;

    // Output index corresponds to input index * 2 (downsampling)
    uint center = out_idx * 2;

    float lo_sum = 0.0f;
    float hi_sum = 0.0f;

    // Convolution with symmetric boundary extension
    for (uint k = 0; k < filter_len; k++) {
        int idx = (int)center - (int)k;

        // Symmetric boundary handling
        if (idx < 0) {
            idx = -idx - 1;
        }
        if (idx >= (int)input_len) {
            idx = 2 * (int)input_len - idx - 1;
        }
        idx = clamp(idx, 0, (int)input_len - 1);

        float val = input[trace_offset + (uint)idx];
        lo_sum += val * lo_d[k];
        hi_sum += val * hi_d[k];
    }

    approx[out_trace_offset + out_idx] = lo_sum;
    detail[out_trace_offset + out_idx] = hi_sum;
}

/**
 * Single-level DWT reconstruction.
 *
 * Reconstructs from approximation and detail coefficients using
 * upsampling by 2 and convolution.
 *
 * Grid: (n_traces, output_len, 1)
 */
kernel void dwt_reconstruct_level(
    device const float* approx [[buffer(0)]],        // [n_traces, input_len]
    device const float* detail [[buffer(1)]],        // [n_traces, input_len]
    device float* output [[buffer(2)]],              // [n_traces, output_len]
    device const float* lo_r [[buffer(3)]],          // Low-pass reconstruction
    device const float* hi_r [[buffer(4)]],          // High-pass reconstruction
    constant uint& n_traces [[buffer(5)]],
    constant uint& input_len [[buffer(6)]],          // Coefficient length
    constant uint& output_len [[buffer(7)]],         // Reconstructed length
    constant uint& filter_len [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint out_idx = gid.y;

    if (trace_idx >= n_traces || out_idx >= output_len) return;

    uint trace_offset = trace_idx * input_len;
    uint out_trace_offset = trace_idx * output_len;

    float sum = 0.0f;

    // Upsample and convolve
    for (uint k = 0; k < filter_len; k++) {
        // With upsampling by 2, only even indices have values
        int idx = ((int)out_idx - (int)k);

        if (idx >= 0 && idx % 2 == 0) {
            uint coeff_idx = (uint)idx / 2;

            if (coeff_idx < input_len) {
                float a = approx[trace_offset + coeff_idx];
                float d = detail[trace_offset + coeff_idx];
                sum += a * lo_r[k] + d * hi_r[k];
            }
        }
    }

    output[out_trace_offset + out_idx] = sum;
}

/**
 * Apply soft thresholding to wavelet coefficients.
 *
 * Grid: (n_traces, n_coeffs, 1)
 */
kernel void dwt_soft_threshold(
    device float* coeffs [[buffer(0)]],              // Coefficients to threshold
    device const float* thresholds [[buffer(1)]],    // Per-trace thresholds
    constant uint& n_traces [[buffer(2)]],
    constant uint& n_coeffs [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint coeff_idx = gid.y;

    if (trace_idx >= n_traces || coeff_idx >= n_coeffs) return;

    uint idx = trace_idx * n_coeffs + coeff_idx;
    float val = coeffs[idx];
    float thresh = thresholds[trace_idx];

    // Soft thresholding: sign(x) * max(|x| - thresh, 0)
    float abs_val = abs(val);
    float sign_val = (val >= 0.0f) ? 1.0f : -1.0f;
    coeffs[idx] = sign_val * max(abs_val - thresh, 0.0f);
}

/**
 * Compute MAD-based threshold per trace.
 *
 * Uses the finest detail coefficients (highest frequency) for noise estimation.
 *
 * Grid: (n_traces, 1, 1)
 */
kernel void compute_dwt_threshold(
    device const float* detail_coeffs [[buffer(0)]], // Finest level detail [n_traces, n_coeffs]
    device float* thresholds [[buffer(1)]],          // Output [n_traces]
    constant uint& n_traces [[buffer(2)]],
    constant uint& n_coeffs [[buffer(3)]],
    constant float& threshold_k [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint trace_idx = gid;

    if (trace_idx >= n_traces) return;

    // Load coefficients for this trace
    threadgroup float local_coeffs[512];  // Assume max 512 coeffs per trace

    uint trace_offset = trace_idx * n_coeffs;
    uint n = min(n_coeffs, 512u);

    // Copy and compute absolute values
    if (tid < n) {
        local_coeffs[tid] = abs(detail_coeffs[trace_offset + tid]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple selection sort for small arrays (can optimize with bitonic for larger)
    // Find median by partial sorting
    if (tid == 0) {
        // For small n, use simple approach
        // Sort first half+1 elements to find median
        uint target = n / 2;

        for (uint i = 0; i <= target; i++) {
            uint min_idx = i;
            for (uint j = i + 1; j < n; j++) {
                if (local_coeffs[j] < local_coeffs[min_idx]) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                float temp = local_coeffs[i];
                local_coeffs[i] = local_coeffs[min_idx];
                local_coeffs[min_idx] = temp;
            }
        }

        float median = (n % 2 == 1) ?
                       local_coeffs[target] :
                       (local_coeffs[target - 1] + local_coeffs[target]) * 0.5f;

        // MAD-based threshold: k * median / 0.6745
        float sigma = median / 0.6745f;
        thresholds[trace_idx] = threshold_k * sigma;
    }
}
