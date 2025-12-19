/**
 * FKK Velocity Mask Kernel
 *
 * Builds velocity cone mask for 3D FK filtering.
 * Optimized for unified memory on Apple Silicon.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * Build 3D velocity cone mask.
 *
 * Creates a mask that rejects or passes energy based on apparent velocity.
 * Velocity = frequency / horizontal_wavenumber
 *
 * Grid: (nf, nkx, nky)
 */
kernel void build_velocity_mask(
    device float* mask [[buffer(0)]],                // [nf, nkx, nky]
    constant uint& nf [[buffer(1)]],                 // Number of frequency bins
    constant uint& nkx [[buffer(2)]],                // Number of kx bins
    constant uint& nky [[buffer(3)]],                // Number of ky bins
    constant float& df [[buffer(4)]],                // Frequency resolution (Hz)
    constant float& dkx [[buffer(5)]],               // Kx resolution (1/m)
    constant float& dky [[buffer(6)]],               // Ky resolution (1/m)
    constant float& v_min [[buffer(7)]],             // Minimum velocity (m/s)
    constant float& v_max [[buffer(8)]],             // Maximum velocity (m/s)
    constant int& filter_mode [[buffer(9)]],         // 0=reject, 1=pass
    constant int& preserve_dc [[buffer(10)]],        // Preserve DC component
    uint3 gid [[thread_position_in_grid]]
) {
    uint f_idx = gid.x;
    uint kx_idx = gid.y;
    uint ky_idx = gid.z;

    if (f_idx >= nf || kx_idx >= nkx || ky_idx >= nky) return;

    // Compute frequency value
    float freq = (float)f_idx * df;

    // Compute wavenumber values (centered, after fftshift)
    float kx = ((float)kx_idx - (float)nkx / 2.0f) * dkx;
    float ky = ((float)ky_idx - (float)nky / 2.0f) * dky;

    // Horizontal wavenumber magnitude
    float k_horizontal = sqrt(kx * kx + ky * ky);

    // Compute apparent velocity (avoid division by zero)
    float velocity;
    if (k_horizontal > 1e-10f) {
        velocity = abs(freq) / k_horizontal;
    } else {
        velocity = 1e10f;  // Infinite velocity at k=0
    }

    // Determine mask value
    float mask_val;
    bool in_cone = (velocity >= v_min) && (velocity <= v_max);

    if (filter_mode == 0) {
        // Reject mode: zero out velocities in range
        mask_val = in_cone ? 0.0f : 1.0f;
    } else {
        // Pass mode: keep only velocities in range
        mask_val = in_cone ? 1.0f : 0.0f;
    }

    // Preserve DC component if requested
    if (preserve_dc && f_idx == 0 && kx_idx == nkx / 2 && ky_idx == nky / 2) {
        mask_val = 1.0f;
    }

    // Output index
    uint idx = f_idx * nkx * nky + kx_idx * nky + ky_idx;
    mask[idx] = mask_val;
}

/**
 * Build tapered velocity mask.
 *
 * Uses smooth transition at velocity boundaries to reduce ringing.
 *
 * Grid: (nf, nkx, nky)
 */
kernel void build_velocity_mask_tapered(
    device float* mask [[buffer(0)]],
    constant uint& nf [[buffer(1)]],
    constant uint& nkx [[buffer(2)]],
    constant uint& nky [[buffer(3)]],
    constant float& df [[buffer(4)]],
    constant float& dkx [[buffer(5)]],
    constant float& dky [[buffer(6)]],
    constant float& v_min [[buffer(7)]],
    constant float& v_max [[buffer(8)]],
    constant float& taper_width [[buffer(9)]],       // Taper width in m/s
    constant int& filter_mode [[buffer(10)]],
    constant int& preserve_dc [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint f_idx = gid.x;
    uint kx_idx = gid.y;
    uint ky_idx = gid.z;

    if (f_idx >= nf || kx_idx >= nkx || ky_idx >= nky) return;

    float freq = (float)f_idx * df;
    float kx = ((float)kx_idx - (float)nkx / 2.0f) * dkx;
    float ky = ((float)ky_idx - (float)nky / 2.0f) * dky;
    float k_horizontal = sqrt(kx * kx + ky * ky);

    float velocity;
    if (k_horizontal > 1e-10f) {
        velocity = abs(freq) / k_horizontal;
    } else {
        velocity = 1e10f;
    }

    // Smooth transitions using cosine taper
    float mask_val;

    if (velocity < v_min - taper_width) {
        mask_val = (filter_mode == 0) ? 1.0f : 0.0f;
    } else if (velocity < v_min) {
        float t = (velocity - (v_min - taper_width)) / taper_width;
        float taper = 0.5f * (1.0f + cos(M_PI_F * t));  // 1 to 0
        mask_val = (filter_mode == 0) ? taper : (1.0f - taper);
    } else if (velocity <= v_max) {
        mask_val = (filter_mode == 0) ? 0.0f : 1.0f;
    } else if (velocity < v_max + taper_width) {
        float t = (velocity - v_max) / taper_width;
        float taper = 0.5f * (1.0f + cos(M_PI_F * t));  // 1 to 0
        mask_val = (filter_mode == 0) ? (1.0f - taper) : taper;
    } else {
        mask_val = (filter_mode == 0) ? 1.0f : 0.0f;
    }

    if (preserve_dc && f_idx == 0 && kx_idx == nkx / 2 && ky_idx == nky / 2) {
        mask_val = 1.0f;
    }

    uint idx = f_idx * nkx * nky + kx_idx * nky + ky_idx;
    mask[idx] = mask_val;
}

/**
 * Apply 3D mask to complex spectrum.
 *
 * Grid: (nf, nkx, nky)
 */
kernel void apply_fkk_mask(
    device float2* spectrum [[buffer(0)]],           // Complex [nf, nkx, nky]
    device const float* mask [[buffer(1)]],          // Real [nf, nkx, nky]
    constant uint& nf [[buffer(2)]],
    constant uint& nkx [[buffer(3)]],
    constant uint& nky [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint f_idx = gid.x;
    uint kx_idx = gid.y;
    uint ky_idx = gid.z;

    if (f_idx >= nf || kx_idx >= nkx || ky_idx >= nky) return;

    uint idx = f_idx * nkx * nky + kx_idx * nky + ky_idx;

    float m = mask[idx];
    float2 s = spectrum[idx];

    spectrum[idx] = float2(s.x * m, s.y * m);
}
