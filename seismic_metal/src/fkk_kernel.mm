/**
 * FKK Filter Kernel Implementation
 *
 * Hybrid vDSP + Metal GPU implementation for 3D FK filtering.
 * - vDSP for high-performance FFT operations (optimized for Apple Silicon)
 * - Metal GPU for parallel mask building and application
 * Uses unified memory for zero-copy data access.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

#include "fkk_kernel.h"
#include "vdsp_fft.h"
#include "device_manager.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <complex>
#include <thread>

namespace seismic_metal {

std::tuple<int, int, int> compute_fft_sizes(int nt, int nx, int ny) {
    auto next_pow2 = [](int n) {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    };

    return std::make_tuple(next_pow2(nt), next_pow2(nx), next_pow2(ny));
}

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
) {
    if (!is_available()) {
        initialize_device();
    }

    id<MTLDevice> device = get_device();
    id<MTLCommandQueue> queue = get_command_queue();

    std::vector<float> mask(nf * nkx * nky, 1.0f);

    if (!device || !queue) {
        // CPU fallback
        int filter_mode = (mode == "reject") ? 0 : 1;

        for (int f = 0; f < nf; f++) {
            float freq = f * df;
            for (int kx_idx = 0; kx_idx < nkx; kx_idx++) {
                float kx = (kx_idx - nkx / 2.0f) * dkx;
                for (int ky_idx = 0; ky_idx < nky; ky_idx++) {
                    float ky = (ky_idx - nky / 2.0f) * dky;

                    float k_horizontal = std::sqrt(kx * kx + ky * ky);
                    float velocity = (k_horizontal > 1e-10f) ?
                                    std::abs(freq) / k_horizontal : 1e10f;

                    bool in_cone = (velocity >= v_min) && (velocity <= v_max);
                    float mask_val = (filter_mode == 0) ?
                                    (in_cone ? 0.0f : 1.0f) :
                                    (in_cone ? 1.0f : 0.0f);

                    // Preserve DC
                    if (f == 0 && kx_idx == nkx / 2 && ky_idx == nky / 2) {
                        mask_val = 1.0f;
                    }

                    mask[f * nkx * nky + kx_idx * nky + ky_idx] = mask_val;
                }
            }
        }
        return mask;
    }

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = create_pipeline("build_velocity_mask");
        if (!pipeline) {
            return mask;
        }

        size_t mask_size = nf * nkx * nky * sizeof(float);
        id<MTLBuffer> mask_buffer = [device newBufferWithLength:mask_size
                                                        options:MTLResourceStorageModeShared];

        int filter_mode = (mode == "reject") ? 0 : 1;
        int preserve_dc = 1;

        uint nf_u = nf, nkx_u = nkx, nky_u = nky;

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:mask_buffer offset:0 atIndex:0];
        [encoder setBytes:&nf_u length:sizeof(uint) atIndex:1];
        [encoder setBytes:&nkx_u length:sizeof(uint) atIndex:2];
        [encoder setBytes:&nky_u length:sizeof(uint) atIndex:3];
        [encoder setBytes:&df length:sizeof(float) atIndex:4];
        [encoder setBytes:&dkx length:sizeof(float) atIndex:5];
        [encoder setBytes:&dky length:sizeof(float) atIndex:6];
        [encoder setBytes:&v_min length:sizeof(float) atIndex:7];
        [encoder setBytes:&v_max length:sizeof(float) atIndex:8];
        [encoder setBytes:&filter_mode length:sizeof(int) atIndex:9];
        [encoder setBytes:&preserve_dc length:sizeof(int) atIndex:10];

        MTLSize grid = MTLSizeMake(nf, nkx, nky);
        MTLSize threads = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float* mask_ptr = (float*)[mask_buffer contents];
        std::copy(mask_ptr, mask_ptr + nf * nkx * nky, mask.begin());
    }

    return mask;
}

std::tuple<std::vector<float>, KernelMetrics> fkk_filter(
    const float* volume,
    int nt,
    int nx,
    int ny,
    float dt,
    float dx,
    float dy,
    float v_min,
    float v_max,
    const std::string& mode,
    bool preserve_dc
) {
    auto start_total = std::chrono::high_resolution_clock::now();

    KernelMetrics metrics;
    metrics.traces_processed = nx * ny;
    metrics.samples_processed = (int64_t)nt * nx * ny;

    // Compute FFT sizes
    auto [nt_fft, nx_fft, ny_fft] = compute_fft_sizes(nt, nx, ny);

    std::vector<float> output(nt * nx * ny);

    // For now, use Accelerate framework for FFT and GPU for mask only
    // Full GPU FFT implementation would require vDSP or custom FFT kernels

    @autoreleasepool {
        auto start_kernel = std::chrono::high_resolution_clock::now();

        // Compute frequency/wavenumber resolutions
        float df = 1.0f / (nt_fft * dt);
        float dkx = 1.0f / (nx_fft * dx);
        float dky = 1.0f / (ny_fft * dy);

        int nf = nt_fft / 2 + 1;

        // =========================================================
        // Step 1: 3D FFT using vDSP (optimized for Apple Silicon)
        // =========================================================

        // Perform 3D real-to-complex FFT
        auto spectrum = vdsp_rfft3d(volume, nt, nx, ny);

        // =========================================================
        // Step 2: Build velocity mask using Metal GPU (parallel)
        // =========================================================

        std::vector<float> mask = build_velocity_mask_gpu(
            nf, nx_fft, ny_fft, df, dkx, dky, v_min, v_max, mode
        );

        // =========================================================
        // Step 3: Apply fftshift to spatial dimensions
        // =========================================================

        // For each frequency slice, apply fftshift to kx-ky plane
        for (int f = 0; f < nf; f++) {
            fftshift_2d(spectrum.data() + f * nx_fft * ny_fft, nx_fft, ny_fft);
        }

        // =========================================================
        // Step 4: Apply mask (GPU-accelerated if available)
        // =========================================================

        id<MTLDevice> device = get_device();
        id<MTLCommandQueue> queue = get_command_queue();
        id<MTLComputePipelineState> apply_pipeline = create_pipeline("apply_fkk_mask");

        if (device && queue && apply_pipeline) {
            // GPU mask application
            size_t spectrum_size = nf * nx_fft * ny_fft * sizeof(float) * 2;
            size_t mask_size = nf * nx_fft * ny_fft * sizeof(float);

            id<MTLBuffer> spectrum_buffer = [device newBufferWithBytes:spectrum.data()
                                                                length:spectrum_size
                                                               options:MTLResourceStorageModeShared];
            id<MTLBuffer> mask_buffer = [device newBufferWithBytes:mask.data()
                                                            length:mask_size
                                                           options:MTLResourceStorageModeShared];

            uint nf_u = nf, nkx_u = nx_fft, nky_u = ny_fft;

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:apply_pipeline];
            [encoder setBuffer:spectrum_buffer offset:0 atIndex:0];
            [encoder setBuffer:mask_buffer offset:0 atIndex:1];
            [encoder setBytes:&nf_u length:sizeof(uint) atIndex:2];
            [encoder setBytes:&nkx_u length:sizeof(uint) atIndex:3];
            [encoder setBytes:&nky_u length:sizeof(uint) atIndex:4];

            MTLSize grid = MTLSizeMake(nf, nx_fft, ny_fft);
            MTLSize threads = MTLSizeMake(1, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threads];

            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Copy back filtered spectrum
            std::complex<float>* result_ptr = (std::complex<float>*)[spectrum_buffer contents];
            std::copy(result_ptr, result_ptr + nf * nx_fft * ny_fft, spectrum.begin());
        } else {
            // CPU fallback for mask application
            for (int f = 0; f < nf; f++) {
                for (int kx = 0; kx < nx_fft; kx++) {
                    for (int ky = 0; ky < ny_fft; ky++) {
                        int idx = f * nx_fft * ny_fft + kx * ny_fft + ky;
                        spectrum[idx] *= mask[idx];
                    }
                }
            }
        }

        // =========================================================
        // Step 5: Apply ifftshift to spatial dimensions
        // =========================================================

        for (int f = 0; f < nf; f++) {
            ifftshift_2d(spectrum.data() + f * nx_fft * ny_fft, nx_fft, ny_fft);
        }

        // =========================================================
        // Step 6: 3D IFFT using vDSP
        // =========================================================

        output = vdsp_irfft3d(spectrum.data(), nf, nx_fft, ny_fft, nt);

        // Crop to original size if padded
        if (nx_fft != nx || ny_fft != ny) {
            std::vector<float> cropped(nt * nx * ny);
            for (int t = 0; t < nt; t++) {
                for (int x = 0; x < nx; x++) {
                    for (int y = 0; y < ny; y++) {
                        cropped[t * nx * ny + x * ny + y] =
                            output[t * nx_fft * ny_fft + x * ny_fft + y];
                    }
                }
            }
            output = std::move(cropped);
        }

        auto end_kernel = std::chrono::high_resolution_clock::now();

        metrics.kernel_time_ms = std::chrono::duration<double, std::milli>(
            end_kernel - start_kernel).count();
        metrics.total_time_ms = std::chrono::duration<double, std::milli>(
            end_kernel - start_total).count();
    }

    return std::make_tuple(output, metrics);
}

} // namespace seismic_metal
