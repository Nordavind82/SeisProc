/**
 * FKK Filter Kernel Interface
 *
 * Metal GPU kernels for 3D Frequency-Wavenumber-Wavenumber filtering.
 * Optimized for unified memory to eliminate CPU-GPU transfer overhead.
 */

#ifndef SEISMIC_METAL_FKK_KERNEL_H
#define SEISMIC_METAL_FKK_KERNEL_H

#include "common_types.h"
#include <vector>
#include <tuple>

namespace seismic_metal {

/**
 * Apply FKK filter to 3D seismic volume using Metal GPU.
 *
 * Uses unified memory for zero-copy data access on Apple Silicon.
 *
 * @param volume Input volume (nt x nx x ny), row-major
 * @param nt Number of time samples
 * @param nx Number of inline positions
 * @param ny Number of crossline positions
 * @param dt Time sample interval (seconds)
 * @param dx Inline spacing (meters)
 * @param dy Crossline spacing (meters)
 * @param v_min Minimum velocity for filter (m/s)
 * @param v_max Maximum velocity for filter (m/s)
 * @param mode Filter mode: "reject" (remove noise) or "pass" (keep signal)
 * @param preserve_dc Preserve DC component (default true)
 * @return Tuple of (filtered volume, metrics)
 */
std::tuple<std::vector<float>, KernelMetrics> fkk_filter(
    const float* volume,
    int nt,
    int nx,
    int ny,
    float dt,
    float dx,
    float dy,
    float v_min = 200.0f,
    float v_max = 1500.0f,
    const std::string& mode = "reject",
    bool preserve_dc = true
);

/**
 * Build velocity cone mask on GPU.
 *
 * Creates a 3D mask in FK domain based on velocity bounds.
 *
 * @param nf Number of frequency bins
 * @param nkx Number of kx bins
 * @param nky Number of ky bins
 * @param df Frequency resolution (Hz)
 * @param dkx Kx resolution (1/m)
 * @param dky Ky resolution (1/m)
 * @param v_min Minimum velocity
 * @param v_max Maximum velocity
 * @param mode "reject" or "pass"
 * @return Mask array (nf x nkx x nky)
 */
std::vector<float> build_velocity_mask_gpu(
    int nf,
    int nkx,
    int nky,
    float df,
    float dkx,
    float dky,
    float v_min,
    float v_max,
    const std::string& mode
);

/**
 * Compute optimal FFT padding sizes.
 *
 * Returns next power of 2 for each dimension.
 */
std::tuple<int, int, int> compute_fft_sizes(int nt, int nx, int ny);

} // namespace seismic_metal

#endif // SEISMIC_METAL_FKK_KERNEL_H
