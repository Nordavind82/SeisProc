/**
 * Metal Device Manager
 *
 * Singleton for managing Metal device, command queue, and shader library.
 * Provides lazy initialization and resource cleanup.
 */

#ifndef SEISMIC_METAL_DEVICE_MANAGER_H
#define SEISMIC_METAL_DEVICE_MANAGER_H

#include <string>
#include <cstddef>

// Forward declarations for Objective-C types
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLLibrary;
@protocol MTLComputePipelineState;
#else
typedef void* id;
#endif

namespace seismic_metal {

/**
 * Initialize the Metal device and load shader library.
 *
 * @param shader_path Path to the .metallib shader library file.
 *                    If empty, uses default path relative to module.
 * @return true if initialization successful, false otherwise.
 */
bool initialize_device(const std::string& shader_path = "");

/**
 * Check if Metal kernels are available and initialized.
 */
bool is_available();

/**
 * Get the Metal device name (e.g., "Apple M4 Max").
 */
std::string get_device_name();

/**
 * Get recommended maximum working set size (device memory).
 */
size_t get_device_memory();

/**
 * Get maximum buffer length supported by device.
 */
size_t get_max_buffer_length();

/**
 * Get the initialized Metal device.
 * Returns nullptr if not initialized.
 */
#ifdef __OBJC__
id<MTLDevice> get_device();
id<MTLCommandQueue> get_command_queue();
id<MTLLibrary> get_shader_library();
#endif

/**
 * Create a compute pipeline state for a kernel function.
 *
 * @param function_name Name of the kernel function in the shader library.
 * @return Pipeline state, or nullptr on failure.
 */
#ifdef __OBJC__
id<MTLComputePipelineState> create_pipeline(const std::string& function_name);
#endif

/**
 * Release all Metal resources.
 * Should be called before application exit or when switching contexts.
 */
void cleanup();

/**
 * Synchronize all pending GPU operations.
 * Blocks until all commands complete.
 */
void synchronize();

} // namespace seismic_metal

#endif // SEISMIC_METAL_DEVICE_MANAGER_H
