/**
 * STFT Inverse Transform Kernel
 *
 * Batch Inverse Short-Time Fourier Transform with overlap-add reconstruction.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * Batch ISTFT with overlap-add.
 *
 * Reconstructs time-domain signal from STFT coefficients.
 * Uses overlap-add method for seamless reconstruction.
 *
 * Grid: (n_traces, n_samples, 1)
 */
kernel void istft_overlap_add(
    device const float2* stft [[buffer(0)]],         // [n_traces, n_freqs, n_times] complex
    device float* output [[buffer(1)]],              // [n_samples, n_traces]
    device const float* window [[buffer(2)]],        // [nperseg]
    constant uint& n_samples [[buffer(3)]],
    constant uint& n_traces [[buffer(4)]],
    constant uint& nperseg [[buffer(5)]],
    constant uint& hop [[buffer(6)]],
    constant uint& n_freqs [[buffer(7)]],
    constant uint& n_times [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint sample_idx = gid.y;

    if (trace_idx >= n_traces || sample_idx >= n_samples) return;

    float sum = 0.0f;
    float window_sum = 0.0f;

    // Find all time frames that contribute to this sample
    // A frame at time t contributes to samples [t*hop, t*hop + nperseg)
    int first_frame = max(0, (int)sample_idx - (int)nperseg + 1) / (int)hop;
    int last_frame = min((int)n_times - 1, (int)sample_idx / (int)hop);

    for (int t = first_frame; t <= last_frame; t++) {
        uint frame_start = (uint)t * hop;
        uint k = sample_idx - frame_start;  // Position within frame

        if (k >= nperseg) continue;

        // Inverse DFT for this sample position
        float sample_val = 0.0f;

        for (uint f = 0; f < n_freqs; f++) {
            uint stft_idx = trace_idx * n_freqs * n_times + f * n_times + (uint)t;
            float2 coeff = stft[stft_idx];

            // Complex exponential
            float angle = 2.0f * M_PI_F * (float)f * (float)k / (float)nperseg;
            sample_val += coeff.x * cos(angle) - coeff.y * sin(angle);

            // Add conjugate symmetric part for real signal
            if (f > 0 && f < n_freqs - 1) {
                sample_val += coeff.x * cos(angle) + coeff.y * sin(angle);
            }
        }

        // Normalize by nperseg
        sample_val /= (float)nperseg;

        // Apply window and accumulate
        float w = window[k];
        sum += sample_val * w;
        window_sum += w * w;
    }

    // Normalize by window sum (overlap-add normalization)
    if (window_sum > 1e-10f) {
        sum /= window_sum;
    }

    // Output in column-major format
    output[sample_idx * n_traces + trace_idx] = sum;
}

/**
 * Single trace ISTFT (for center trace in aperture).
 *
 * More efficient when only reconstructing one trace.
 *
 * Grid: (n_samples, 1, 1)
 */
kernel void istft_single_trace(
    device const float2* stft [[buffer(0)]],         // [n_freqs, n_times] complex
    device float* output [[buffer(1)]],              // [n_samples]
    device const float* window [[buffer(2)]],
    constant uint& n_samples [[buffer(3)]],
    constant uint& nperseg [[buffer(4)]],
    constant uint& hop [[buffer(5)]],
    constant uint& n_freqs [[buffer(6)]],
    constant uint& n_times [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint sample_idx = gid;

    if (sample_idx >= n_samples) return;

    float sum = 0.0f;
    float window_sum = 0.0f;

    int first_frame = max(0, (int)sample_idx - (int)nperseg + 1) / (int)hop;
    int last_frame = min((int)n_times - 1, (int)sample_idx / (int)hop);

    for (int t = first_frame; t <= last_frame; t++) {
        uint frame_start = (uint)t * hop;
        uint k = sample_idx - frame_start;

        if (k >= nperseg) continue;

        float sample_val = 0.0f;

        for (uint f = 0; f < n_freqs; f++) {
            uint stft_idx = f * n_times + (uint)t;
            float2 coeff = stft[stft_idx];

            float angle = 2.0f * M_PI_F * (float)f * (float)k / (float)nperseg;
            sample_val += coeff.x * cos(angle) - coeff.y * sin(angle);

            if (f > 0 && f < n_freqs - 1) {
                sample_val += coeff.x * cos(angle) + coeff.y * sin(angle);
            }
        }

        sample_val /= (float)nperseg;

        float w = window[k];
        sum += sample_val * w;
        window_sum += w * w;
    }

    if (window_sum > 1e-10f) {
        sum /= window_sum;
    }

    output[sample_idx] = sum;
}
