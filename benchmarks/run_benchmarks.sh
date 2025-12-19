#!/bin/bash
#
# Run SeisProc processor benchmarks
#
# Usage:
#   ./run_benchmarks.sh                 # Full benchmark (100 gathers, 4000 traces)
#   ./run_benchmarks.sh --quick         # Quick test (10 gathers, 500 traces)
#   ./run_benchmarks.sh --generate      # Regenerate benchmark data
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default settings
DATA_DIR="benchmark_data"
GATHERS=3
QUICK=false
GENERATE=false
OUTPUT="benchmark_results.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            GATHERS=2
            shift
            ;;
        --generate)
            GENERATE=true
            shift
            ;;
        --gathers)
            GATHERS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "SeisProc Processor Benchmarking"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Gathers to profile: $GATHERS"
echo "Quick mode: $QUICK"
echo ""

# Build command
CMD="python -m benchmarks.profile_processors"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --gathers $GATHERS"
CMD="$CMD --output $OUTPUT"

if [ "$QUICK" = true ]; then
    CMD="$CMD --quick"
fi

if [ "$GENERATE" = true ] || [ ! -d "$DATA_DIR" ]; then
    CMD="$CMD --generate"
fi

echo "Running: $CMD"
echo ""

# Run benchmark
$CMD

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Results saved to: $OUTPUT"
echo "========================================"
