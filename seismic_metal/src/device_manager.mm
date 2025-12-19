/**
 * Metal Device Manager Implementation
 *
 * Manages Metal device lifecycle, command queue, and shader library.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "device_manager.h"
#include <iostream>
#include <mutex>

namespace seismic_metal {

// Global state (singleton pattern)
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLLibrary> g_shader_library = nil;
static bool g_initialized = false;
static std::mutex g_init_mutex;
static std::string g_shader_path;

// Cache for compute pipeline states
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pipelines = nil;

bool initialize_device(const std::string& shader_path) {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (g_initialized) {
        return g_device != nil;
    }

    @autoreleasepool {
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            std::cerr << "[seismic_metal] Metal is not supported on this device" << std::endl;
            g_initialized = true;
            return false;
        }

        NSLog(@"[seismic_metal] Metal device: %@", g_device.name);

        // Create command queue
        g_command_queue = [g_device newCommandQueue];
        if (!g_command_queue) {
            std::cerr << "[seismic_metal] Failed to create Metal command queue" << std::endl;
            g_initialized = true;
            return false;
        }

        // Determine shader library path
        std::string lib_path = shader_path;
        if (lib_path.empty()) {
            // Try default path (same directory as module)
            #ifdef SHADER_PATH
            lib_path = SHADER_PATH;
            #else
            // Fallback: look in current directory
            lib_path = "seismic_kernels.metallib";
            #endif
        }

        // Load shader library
        NSString* path = [NSString stringWithUTF8String:lib_path.c_str()];
        NSError* error = nil;

        // First try loading from file
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            g_shader_library = [g_device newLibraryWithFile:path error:&error];
        }

        if (!g_shader_library) {
            // Try loading from bundle
            NSBundle* bundle = [NSBundle mainBundle];
            NSString* bundlePath = [bundle pathForResource:@"seismic_kernels" ofType:@"metallib"];
            if (bundlePath) {
                g_shader_library = [g_device newLibraryWithFile:bundlePath error:&error];
            }
        }

        if (!g_shader_library) {
            std::cerr << "[seismic_metal] Warning: Could not load shader library from: "
                      << lib_path << std::endl;
            if (error) {
                std::cerr << "[seismic_metal] Error: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
            // Continue without shaders - some functionality may still work
        } else {
            NSLog(@"[seismic_metal] Loaded shader library: %@", path);
            g_shader_path = lib_path;
        }

        // Initialize pipeline cache
        g_pipelines = [NSMutableDictionary dictionary];

        g_initialized = true;
        return true;
    }
}

bool is_available() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_device != nil;
}

std::string get_device_name() {
    if (!g_device) {
        return "Not initialized";
    }
    return [g_device.name UTF8String];
}

size_t get_device_memory() {
    if (!g_device) {
        return 0;
    }
    return g_device.recommendedMaxWorkingSetSize;
}

size_t get_max_buffer_length() {
    if (!g_device) {
        return 0;
    }
    return g_device.maxBufferLength;
}

id<MTLDevice> get_device() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_device;
}

id<MTLCommandQueue> get_command_queue() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_command_queue;
}

id<MTLLibrary> get_shader_library() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_shader_library;
}

id<MTLComputePipelineState> create_pipeline(const std::string& function_name) {
    if (!g_shader_library) {
        std::cerr << "[seismic_metal] Cannot create pipeline: shader library not loaded"
                  << std::endl;
        return nil;
    }

    @autoreleasepool {
        NSString* name = [NSString stringWithUTF8String:function_name.c_str()];

        // Check cache first
        id<MTLComputePipelineState> cached = g_pipelines[name];
        if (cached) {
            return cached;
        }

        // Get kernel function
        id<MTLFunction> function = [g_shader_library newFunctionWithName:name];
        if (!function) {
            std::cerr << "[seismic_metal] Kernel function not found: "
                      << function_name << std::endl;
            return nil;
        }

        // Create pipeline state
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [g_device newComputePipelineStateWithFunction:function error:&error];

        if (!pipeline) {
            std::cerr << "[seismic_metal] Failed to create pipeline for: "
                      << function_name << std::endl;
            if (error) {
                std::cerr << "[seismic_metal] Error: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
            return nil;
        }

        // Cache for reuse
        g_pipelines[name] = pipeline;

        return pipeline;
    }
}

void synchronize() {
    if (!g_command_queue) return;

    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [g_command_queue commandBuffer];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

void cleanup() {
    std::lock_guard<std::mutex> lock(g_init_mutex);

    @autoreleasepool {
        // Clear pipeline cache
        [g_pipelines removeAllObjects];
        g_pipelines = nil;

        // Release resources
        g_shader_library = nil;
        g_command_queue = nil;
        g_device = nil;

        g_initialized = false;
        g_shader_path.clear();
    }

    NSLog(@"[seismic_metal] Resources cleaned up");
}

} // namespace seismic_metal
