/**
 * DWT Reconstruction Kernel
 *
 * Full multi-level DWT reconstruction (waverec equivalent).
 * Separate file for clarity, though some functions overlap with dwt_decompose.metal.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * Multi-level DWT reconstruction in single pass.
 *
 * Takes all coefficients and reconstructs original signal.
 * Coefficients stored as: [approx_n, detail_n, detail_n-1, ..., detail_1]
 *
 * Grid: (n_traces, 1, 1) - one thread per trace
 */
kernel void dwt_reconstruct_full(
    device const float* all_coeffs [[buffer(0)]],    // All coefficients, concatenated
    device const uint* coeff_lengths [[buffer(1)]],  // Length of each level's coeffs
    device float* output [[buffer(2)]],              // [n_traces, n_samples]
    device const float* lo_r [[buffer(3)]],
    device const float* hi_r [[buffer(4)]],
    constant uint& n_traces [[buffer(5)]],
    constant uint& n_levels [[buffer(6)]],
    constant uint& n_samples [[buffer(7)]],
    constant uint& filter_len [[buffer(8)]],
    constant uint& total_coeff_len [[buffer(9)]],    // Total coeffs per trace
    uint gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid;
    if (trace_idx >= n_traces) return;

    // This kernel is complex - for now, use CPU reconstruction
    // and focus on optimizing the decomposition and thresholding
}

/**
 * Copy and pad array.
 *
 * Utility kernel for preparing data for reconstruction.
 *
 * Grid: (n_traces, output_len, 1)
 */
kernel void copy_with_padding(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n_traces [[buffer(2)]],
    constant uint& input_len [[buffer(3)]],
    constant uint& output_len [[buffer(4)]],
    constant float& pad_value [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint idx = gid.y;

    if (trace_idx >= n_traces || idx >= output_len) return;

    uint in_offset = trace_idx * input_len;
    uint out_offset = trace_idx * output_len;

    if (idx < input_len) {
        output[out_offset + idx] = input[in_offset + idx];
    } else {
        output[out_offset + idx] = pad_value;
    }
}

/**
 * Truncate array to target length.
 *
 * Grid: (n_traces, target_len, 1)
 */
kernel void truncate_array(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n_traces [[buffer(2)]],
    constant uint& input_len [[buffer(3)]],
    constant uint& target_len [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint idx = gid.y;

    if (trace_idx >= n_traces || idx >= target_len) return;

    uint in_offset = trace_idx * input_len;
    uint out_offset = trace_idx * target_len;

    output[out_offset + idx] = input[in_offset + idx];
}
