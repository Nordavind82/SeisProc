#!/bin/bash
# Quick activation script for the virtual environment
# Usage: source activate.sh
source venv/bin/activate
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Virtual environment activated!"
    echo "Run 'python main.py' to start the application"
else
    echo "Virtual environment not found!"
    echo "Please run './setup_env.sh' first to create the environment"
    return 1
fi
