cmake_minimum_required(VERSION 3.18)
project(seismic_metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Ensure we're on macOS
if(NOT APPLE)
    message(FATAL_ERROR "seismic_metal requires macOS with Metal support")
endif()

# ============================================================================
# Find Dependencies
# ============================================================================

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()

find_package(pybind11 CONFIG REQUIRED)

# Find macOS frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

message(STATUS "Metal framework: ${METAL_FRAMEWORK}")
message(STATUS "Foundation framework: ${FOUNDATION_FRAMEWORK}")
message(STATUS "Accelerate framework: ${ACCELERATE_FRAMEWORK}")

# ============================================================================
# Metal Shader Compilation
# ============================================================================

set(METAL_SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(METAL_SHADERS
    ${METAL_SHADER_DIR}/mad_threshold.metal
    ${METAL_SHADER_DIR}/dwt_decompose.metal
    ${METAL_SHADER_DIR}/dwt_reconstruct.metal
    ${METAL_SHADER_DIR}/swt_reconstruct.metal
    ${METAL_SHADER_DIR}/stft_forward.metal
    ${METAL_SHADER_DIR}/stft_inverse.metal
    ${METAL_SHADER_DIR}/fkk_mask.metal
)

# Compile each .metal file to .air
set(SHADER_AIR_FILES "")
foreach(SHADER ${METAL_SHADERS})
    if(EXISTS ${SHADER})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(AIR_FILE ${CMAKE_BINARY_DIR}/${SHADER_NAME}.air)

        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun metal -c ${SHADER} -o ${AIR_FILE} -ffast-math -O3
            DEPENDS ${SHADER}
            COMMENT "Compiling Metal shader: ${SHADER_NAME}"
            VERBATIM
        )
        list(APPEND SHADER_AIR_FILES ${AIR_FILE})
    else()
        message(WARNING "Metal shader not found: ${SHADER}")
    endif()
endforeach()

# Link all .air files into .metallib
if(SHADER_AIR_FILES)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
        COMMAND xcrun metallib ${SHADER_AIR_FILES} -o ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
        DEPENDS ${SHADER_AIR_FILES}
        COMMENT "Linking Metal shader library"
        VERBATIM
    )

    add_custom_target(metal_shaders ALL
        DEPENDS ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
    )
endif()

# ============================================================================
# C++/Objective-C++ Sources
# ============================================================================

set(SOURCES
    src/device_manager.mm
    src/vdsp_fft.mm
    src/dwt_kernel.mm
    src/stft_kernel.mm
    src/fkk_kernel.mm
    src/bindings.cpp
)

# Create Python module
pybind11_add_module(seismic_metal ${SOURCES})

# Add dependency on shader compilation
if(TARGET metal_shaders)
    add_dependencies(seismic_metal metal_shaders)
endif()

# Include directories
target_include_directories(seismic_metal PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link frameworks
target_link_libraries(seismic_metal PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

# Compiler options
target_compile_options(seismic_metal PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-O3 -ffast-math -fobjc-arc -Wall -Wextra>
)

# Embed shader library path
target_compile_definitions(seismic_metal PRIVATE
    SHADER_PATH="${CMAKE_BINARY_DIR}/seismic_kernels.metallib"
)

# ============================================================================
# Post-Build Steps
# ============================================================================

# Copy metallib to output directory
if(EXISTS ${CMAKE_BINARY_DIR}/seismic_kernels.metallib OR TARGET metal_shaders)
    add_custom_command(TARGET seismic_metal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
                $<TARGET_FILE_DIR:seismic_metal>/seismic_kernels.metallib
        COMMENT "Copying Metal library to output directory"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS seismic_metal
    LIBRARY DESTINATION .
)

if(EXISTS ${CMAKE_BINARY_DIR}/seismic_kernels.metallib)
    install(FILES ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
        DESTINATION .
    )
endif()

# ============================================================================
# Status Summary
# ============================================================================

message(STATUS "")
message(STATUS "seismic_metal configuration:")
message(STATUS "  Python: ${Python3_EXECUTABLE}")
message(STATUS "  pybind11: ${pybind11_DIR}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
