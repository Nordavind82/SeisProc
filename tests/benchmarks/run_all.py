"""
Run all SeisProc performance benchmarks.

Usage:
    python -m tests.benchmarks.run_all [--quick] [--save DIR] [--baseline DIR]
"""
import sys
import time
import json
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.benchmark_stransform import (
    run_benchmark_suite as run_stransform_benchmarks,
    print_summary as print_stransform_summary,
    check_regression as check_stransform_regression
)
from tests.benchmarks.benchmark_segy import (
    run_segy_benchmark_suite,
    print_segy_summary
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all SeisProc benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmarks with smaller sizes")
    parser.add_argument("--save", type=str,
                        help="Directory to save benchmark results")
    parser.add_argument("--baseline", type=str,
                        help="Directory containing baseline results for regression checking")
    args = parser.parse_args()

    print("="*80)
    print("SEISPROC PERFORMANCE BENCHMARK SUITE")
    print("="*80)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print("="*80)

    all_passed = True

    # S-Transform benchmarks
    print("\n" + "="*80)
    print("SECTION 1: S-TRANSFORM / TF-DENOISE BENCHMARKS")
    print("="*80)

    if args.quick:
        stransform_results = run_stransform_benchmarks([(1000, 50), (2000, 100)])
    else:
        stransform_results = run_stransform_benchmarks()

    print_stransform_summary(stransform_results)

    # SEG-Y benchmarks
    print("\n" + "="*80)
    print("SECTION 2: SEG-Y I/O BENCHMARKS")
    print("="*80)

    if args.quick:
        segy_results = run_segy_benchmark_suite([(1000, 100), (2000, 500)])
    else:
        segy_results = run_segy_benchmark_suite()

    print_segy_summary(segy_results)

    # Save results
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save S-Transform results
        stransform_path = save_dir / f"stransform_{timestamp}.json"
        with open(stransform_path, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [asdict(r) for r in stransform_results]
            }, f, indent=2)
        print(f"\nS-Transform results saved to: {stransform_path}")

        # Save SEG-Y results
        segy_path = save_dir / f"segy_{timestamp}.json"
        with open(segy_path, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [asdict(r) for r in segy_results]
            }, f, indent=2)
        print(f"SEG-Y results saved to: {segy_path}")

        # Also save as 'latest' for easy baseline comparison
        latest_stransform = save_dir / "stransform_latest.json"
        latest_segy = save_dir / "segy_latest.json"

        with open(latest_stransform, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [asdict(r) for r in stransform_results]
            }, f, indent=2)

        with open(latest_segy, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [asdict(r) for r in segy_results]
            }, f, indent=2)

    # Check for regression
    if args.baseline:
        baseline_dir = Path(args.baseline)

        print("\n" + "="*80)
        print("REGRESSION CHECK")
        print("="*80)

        stransform_baseline = baseline_dir / "stransform_latest.json"
        if stransform_baseline.exists():
            if not check_stransform_regression(stransform_results, stransform_baseline):
                all_passed = False
        else:
            print(f"No S-Transform baseline found at {stransform_baseline}")

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"S-Transform tests: {len(stransform_results)}")
    print(f"SEG-Y tests: {len(segy_results)}")

    if args.baseline:
        if all_passed:
            print("Regression check: PASSED")
        else:
            print("Regression check: FAILED")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
