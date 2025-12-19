#!/bin/bash
#
# Build script for Seismic Metal Kernels
#
# This script compiles the Metal shaders and C++ code into a Python extension.
# Requires: CMake 3.18+, Xcode command line tools, pybind11
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
METAL_DIR="$PROJECT_ROOT/seismic_metal"
BUILD_DIR="$METAL_DIR/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."

    # Check for macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        echo_error "This script only works on macOS (requires Metal)"
        exit 1
    fi

    # Check for Apple Silicon or Metal-capable Mac
    if ! system_profiler SPDisplaysDataType | grep -q "Metal"; then
        echo_error "Metal GPU not detected"
        exit 1
    fi

    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        echo_error "CMake not found. Install with: brew install cmake"
        exit 1
    fi

    cmake_version=$(cmake --version | head -1 | cut -d' ' -f3)
    echo_info "CMake version: $cmake_version"

    # Check for xcrun (Metal compiler)
    if ! command -v xcrun &> /dev/null; then
        echo_error "Xcode command line tools not found. Install with: xcode-select --install"
        exit 1
    fi

    # Check for pybind11
    if ! python3 -c "import pybind11" 2>/dev/null; then
        echo_warn "pybind11 not found. Installing..."
        pip install pybind11
    fi

    echo_info "All prerequisites satisfied"
}

# Compile Metal shaders
compile_shaders() {
    echo_info "Compiling Metal shaders..."

    SHADER_DIR="$METAL_DIR/shaders"
    OUTPUT_DIR="$BUILD_DIR/shaders"

    mkdir -p "$OUTPUT_DIR"

    # Compile each shader to .air (intermediate)
    for shader in "$SHADER_DIR"/*.metal; do
        if [[ -f "$shader" ]]; then
            name=$(basename "$shader" .metal)
            echo_info "  Compiling $name.metal -> $name.air"
            xcrun -sdk macosx metal -c "$shader" -o "$OUTPUT_DIR/$name.air"
        fi
    done

    # Link all .air files into metallib
    echo_info "  Linking shaders into seismic_kernels.metallib"
    xcrun -sdk macosx metallib "$OUTPUT_DIR"/*.air -o "$OUTPUT_DIR/seismic_kernels.metallib"

    echo_info "Metal shaders compiled successfully"
}

# Build C++ extension
build_extension() {
    echo_info "Building C++ extension..."

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure with CMake
    echo_info "  Running CMake configuration..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=$(which python3)

    # Build
    echo_info "  Building..."
    cmake --build . --config Release -j$(sysctl -n hw.ncpu)

    echo_info "C++ extension built successfully"
}

# Install extension
install_extension() {
    echo_info "Installing extension..."

    # Find the built .so file
    SO_FILE=$(find "$BUILD_DIR" -name "seismic_metal*.so" -type f | head -1)

    if [[ -z "$SO_FILE" ]]; then
        echo_error "Built extension not found"
        exit 1
    fi

    # Copy to Python module directory
    PYTHON_DIR="$METAL_DIR/python"
    mkdir -p "$PYTHON_DIR"
    cp "$SO_FILE" "$PYTHON_DIR/"

    # Copy metallib
    METALLIB="$BUILD_DIR/shaders/seismic_kernels.metallib"
    if [[ -f "$METALLIB" ]]; then
        cp "$METALLIB" "$PYTHON_DIR/"
    fi

    echo_info "Extension installed to $PYTHON_DIR"

    # Verify import
    echo_info "Verifying installation..."
    cd "$PROJECT_ROOT"
    if python3 -c "from seismic_metal.python import seismic_metal; print('Import successful:', seismic_metal.get_device_info())" 2>/dev/null; then
        echo_info "Installation verified successfully!"
    else
        echo_warn "Import verification failed - extension may need additional setup"
    fi
}

# Clean build
clean() {
    echo_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    echo_info "Clean complete"
}

# Print usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build everything (default)"
    echo "  shaders   Compile Metal shaders only"
    echo "  cpp       Build C++ extension only"
    echo "  install   Install built extension"
    echo "  clean     Remove build directory"
    echo "  help      Show this help"
}

# Main
main() {
    case "${1:-build}" in
        build)
            check_prerequisites
            compile_shaders
            build_extension
            install_extension
            echo_info "Build complete!"
            ;;
        shaders)
            check_prerequisites
            compile_shaders
            ;;
        cpp)
            check_prerequisites
            build_extension
            ;;
        install)
            install_extension
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
