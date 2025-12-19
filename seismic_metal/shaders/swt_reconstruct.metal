/**
 * SWT Reconstruction Kernel
 *
 * GPU-accelerated Stationary Wavelet Transform inverse (iSWT).
 * This is the critical kernel for SWT denoising (69% of processing time).
 *
 * The SWT reconstruction is more expensive than DWT because:
 * 1. No downsampling - all levels have same length
 * 2. Must average multiple reconstructions for translation invariance
 *
 * Updated to match PyWavelets algorithm with:
 * - Periodic boundary conditions (default for pywt.swt)
 * - Center shift for proper phase alignment
 * - Reversed filter access for perfect reconstruction
 */

#include <metal_stdlib>
using namespace metal;

// Maximum filter length supported
constant uint MAX_FILTER_LEN = 20;

// Helper function for periodic boundary (matches PyWavelets SWT default)
inline int periodic_boundary(int idx, int n_samples) {
    idx = idx % n_samples;
    if (idx < 0) idx += n_samples;
    return idx;
}

/**
 * Single-level SWT reconstruction.
 *
 * Reconstructs signal from approximation and detail coefficients at one level.
 * Uses "a trous" algorithm with center shift and reversed filter access.
 * Matches PyWavelets pywt.iswt algorithm.
 *
 * Grid: (n_traces, n_samples, 1)
 */
kernel void swt_reconstruct_level(
    device const float* approx [[buffer(0)]],        // Approximation coeffs [n_traces, n_samples]
    device const float* detail [[buffer(1)]],        // Detail coeffs [n_traces, n_samples]
    device float* output [[buffer(2)]],              // Output [n_traces, n_samples]
    device const float* lo_r [[buffer(3)]],          // Low-pass reconstruction filter
    device const float* hi_r [[buffer(4)]],          // High-pass reconstruction filter
    constant uint& n_traces [[buffer(5)]],
    constant uint& n_samples [[buffer(6)]],
    constant uint& filter_len [[buffer(7)]],
    constant uint& level [[buffer(8)]],              // Current level (0-indexed)
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint sample_idx = gid.y;

    if (trace_idx >= n_traces || sample_idx >= n_samples) return;

    uint base_offset = trace_idx * n_samples;

    // Upsampling factor for this level (2^level)
    uint upsample = 1u << level;
    int center_shift = (int)filter_len / 2;

    // Convolution with upsampled filters
    float sum = 0.0f;

    for (uint k = 0; k < filter_len; k++) {
        // Key formula: i + k*step - center_shift (opposite direction from decomposition)
        int idx = (int)sample_idx + (int)(k * upsample) - center_shift;
        idx = periodic_boundary(idx, (int)n_samples);

        float a_val = approx[base_offset + idx];
        float d_val = detail[base_offset + idx];

        // Critical: Access filters in REVERSED order for perfect reconstruction
        uint fk = filter_len - 1 - k;
        sum += a_val * lo_r[fk] + d_val * hi_r[fk];
    }

    // Normalization factor
    output[base_offset + sample_idx] = sum * 0.5f;
}

/**
 * Batch SWT reconstruction for all traces.
 *
 * Performs full multi-level iSWT reconstruction in a single dispatch.
 * Matches PyWavelets algorithm with center shift and reversed filter access.
 *
 * Grid: (n_traces, ceil(n_samples/BLOCK_SIZE), 1)
 * Threadgroup: (1, BLOCK_SIZE, 1)
 */
kernel void swt_reconstruct_batch(
    device const float* coeffs_approx [[buffer(0)]],  // [n_traces, n_levels+1, n_samples]
    device const float* coeffs_detail [[buffer(1)]],  // [n_traces, n_levels, n_samples]
    device float* output [[buffer(2)]],               // [n_traces, n_samples]
    device const float* lo_r [[buffer(3)]],           // Low-pass filter
    device const float* hi_r [[buffer(4)]],           // High-pass filter
    constant uint& n_traces [[buffer(5)]],
    constant uint& n_samples [[buffer(6)]],
    constant uint& n_levels [[buffer(7)]],
    constant uint& filter_len [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint trace_idx = gid.x;
    uint sample_block = gid.y;
    uint sample_idx = sample_block * 256 + tid;  // Assuming BLOCK_SIZE=256

    if (trace_idx >= n_traces || sample_idx >= n_samples) return;

    // Start with coarsest approximation
    uint coeff_stride = n_samples;
    uint trace_offset = trace_idx * (n_levels + 1) * n_samples;
    uint detail_trace_offset = trace_idx * n_levels * n_samples;
    int center_shift = (int)filter_len / 2;

    // Start from coarsest level approximation
    float result = coeffs_approx[trace_offset + n_levels * coeff_stride + sample_idx];

    // Reconstruct from coarse to fine
    for (int level = (int)n_levels - 1; level >= 0; level--) {
        uint upsample = 1u << (uint)level;

        float sum = 0.0f;

        for (uint k = 0; k < filter_len; k++) {
            // Key formula: i + k*step - center_shift
            int idx = (int)sample_idx + (int)(k * upsample) - center_shift;
            idx = periodic_boundary(idx, (int)n_samples);

            // Get detail coefficient for this level
            float d_val = coeffs_detail[detail_trace_offset + (uint)level * coeff_stride + idx];

            // Reversed filter access
            uint fk = filter_len - 1 - k;
            sum += result * lo_r[fk] + d_val * hi_r[fk];
        }

        result = sum * 0.5f;
    }

    output[trace_idx * n_samples + sample_idx] = result;
}

/**
 * SWT decomposition single level.
 *
 * Computes approximation and detail coefficients for one level.
 * Matches PyWavelets pywt.swt algorithm with center shift for proper phase alignment.
 */
kernel void swt_decompose_level(
    device const float* input [[buffer(0)]],         // Input signal [n_traces, n_samples]
    device float* approx [[buffer(1)]],              // Output approximation [n_traces, n_samples]
    device float* detail [[buffer(2)]],              // Output detail [n_traces, n_samples]
    device const float* lo_d [[buffer(3)]],          // Low-pass decomposition filter
    device const float* hi_d [[buffer(4)]],          // High-pass decomposition filter
    constant uint& n_traces [[buffer(5)]],
    constant uint& n_samples [[buffer(6)]],
    constant uint& filter_len [[buffer(7)]],
    constant uint& level [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint sample_idx = gid.y;

    if (trace_idx >= n_traces || sample_idx >= n_samples) return;

    uint base_offset = trace_idx * n_samples;
    uint upsample = 1u << level;
    int center_shift = (int)filter_len / 2;

    float lo_sum = 0.0f;
    float hi_sum = 0.0f;

    for (uint k = 0; k < filter_len; k++) {
        // Key formula: i - k*step + center_shift (matches PyWavelets)
        int idx = (int)sample_idx - (int)(k * upsample) + center_shift;
        idx = periodic_boundary(idx, (int)n_samples);

        float val = input[base_offset + idx];
        lo_sum += val * lo_d[k];
        hi_sum += val * hi_d[k];
    }

    approx[base_offset + sample_idx] = lo_sum;
    detail[base_offset + sample_idx] = hi_sum;
}
