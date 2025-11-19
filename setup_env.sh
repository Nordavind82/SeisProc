#!/bin/bash
# Virtual Environment Setup Script for Seismic Denoise App
# This script creates and configures a Python virtual environment

set -e  # Exit on error

VENV_DIR="venv"
PYTHON_CMD="python3"

echo "================================================"
echo "Seismic Denoise App - Environment Setup"
echo "================================================"
echo ""

# Check if Python 3 is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Display Python version
PYTHON_VERSION=$($PYTHON_CMD --version)
echo "Using: $PYTHON_VERSION"
echo ""

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
echo "Creating virtual environment in '$VENV_DIR'..."
$PYTHON_CMD -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Display installed packages
echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Installed packages:"
pip list
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python main.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""