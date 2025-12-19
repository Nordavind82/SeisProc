/**
 * STFT Forward Transform Kernel
 *
 * Batch Short-Time Fourier Transform for all traces in a single dispatch.
 * Replaces Python loop over traces with GPU parallelism.
 */

#include <metal_stdlib>
using namespace metal;

// Maximum window size
constant uint MAX_WINDOW_SIZE = 256;

/**
 * Batch STFT forward transform.
 *
 * Computes STFT for all traces simultaneously.
 * Each thread computes one (trace, freq, time) bin.
 *
 * Grid: (n_traces, n_freqs, n_times)
 */
kernel void stft_forward_batch(
    device const float* traces [[buffer(0)]],        // [n_samples, n_traces] column-major
    device float2* stft_out [[buffer(1)]],           // [n_traces, n_freqs, n_times] complex
    device const float* window [[buffer(2)]],        // [nperseg]
    constant uint& n_samples [[buffer(3)]],
    constant uint& n_traces [[buffer(4)]],
    constant uint& nperseg [[buffer(5)]],
    constant uint& hop [[buffer(6)]],                // nperseg - noverlap
    constant uint& n_freqs [[buffer(7)]],            // nperseg/2 + 1
    constant uint& n_times [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint freq_idx = gid.y;
    uint time_idx = gid.z;

    if (trace_idx >= n_traces || freq_idx >= n_freqs || time_idx >= n_times) return;

    // Start sample for this time frame
    uint start_sample = time_idx * hop;

    // DFT for this frequency bin
    float2 sum = float2(0.0f, 0.0f);
    float angle_base = -2.0f * M_PI_F * (float)freq_idx / (float)nperseg;

    for (uint k = 0; k < nperseg; k++) {
        uint sample_idx = start_sample + k;
        if (sample_idx >= n_samples) break;

        // Get sample (column-major: traces stored as [n_samples, n_traces])
        float sample = traces[sample_idx * n_traces + trace_idx];

        // Apply window
        float windowed = sample * window[k];

        // Complex exponential
        float angle = angle_base * (float)k;
        sum.x += windowed * cos(angle);
        sum.y += windowed * sin(angle);
    }

    // Output index (row-major: [n_traces, n_freqs, n_times])
    uint out_idx = trace_idx * n_freqs * n_times + freq_idx * n_times + time_idx;
    stft_out[out_idx] = sum;
}

/**
 * Create Hann window.
 *
 * Grid: (nperseg, 1, 1)
 */
kernel void create_hann_window(
    device float* window [[buffer(0)]],
    constant uint& nperseg [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nperseg) return;

    // Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
    float n = (float)gid;
    float N = (float)nperseg;
    window[gid] = 0.5f * (1.0f - cos(2.0f * M_PI_F * n / (N - 1.0f)));
}

/**
 * Create Gaussian window (for Gabor transform).
 *
 * Grid: (nperseg, 1, 1)
 */
kernel void create_gaussian_window(
    device float* window [[buffer(0)]],
    constant uint& nperseg [[buffer(1)]],
    constant float& sigma [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= nperseg) return;

    // Gaussian window centered at nperseg/2
    float n = (float)gid;
    float center = (float)(nperseg - 1) / 2.0f;
    float x = (n - center) / sigma;
    window[gid] = exp(-0.5f * x * x);
}

/**
 * Extract amplitudes from complex STFT coefficients.
 *
 * Grid: (n_traces, n_freqs, n_times)
 */
kernel void extract_amplitudes(
    device const float2* stft [[buffer(0)]],         // Complex [n_traces, n_freqs, n_times]
    device float* amplitudes [[buffer(1)]],          // Real [n_traces, n_freqs, n_times]
    constant uint& n_traces [[buffer(2)]],
    constant uint& n_freqs [[buffer(3)]],
    constant uint& n_times [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint freq_idx = gid.y;
    uint time_idx = gid.z;

    if (trace_idx >= n_traces || freq_idx >= n_freqs || time_idx >= n_times) return;

    uint idx = trace_idx * n_freqs * n_times + freq_idx * n_times + time_idx;
    float2 coeff = stft[idx];
    amplitudes[idx] = length(coeff);
}
