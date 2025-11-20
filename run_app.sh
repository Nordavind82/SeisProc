#!/bin/bash
# Seismic QC Tool Launcher
# This script ensures proper environment setup and launches the application

set -e

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found!"
    echo "Please run: ./setup_env.sh"
    exit 1
fi

# Set Qt/Display environment variables for stability
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none
export MPLBACKEND=Agg

echo "Starting Seismic QC Tool..."
echo "Environment:"
echo "  - QT Platform: $QT_QPA_PLATFORM"
echo "  - OpenGL Integration: disabled"
echo "  - Matplotlib Backend: $MPLBACKEND"
echo ""

# Launch application
python main.py

# Deactivate when done
deactivate
