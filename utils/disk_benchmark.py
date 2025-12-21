#!/usr/bin/env python3
"""
Disk I/O Benchmark Utility

Run this to measure actual disk read/write speeds for your storage device.
Useful for diagnosing I/O bottlenecks in seismic processing.

Usage:
    python utils/disk_benchmark.py /path/to/test/directory
    python utils/disk_benchmark.py  # Uses current directory
"""

import os
import sys
import time
import tempfile
import numpy as np
from pathlib import Path


def benchmark_sequential_write(path: Path, size_mb: int = 500, block_size_mb: int = 10) -> float:
    """Benchmark sequential write speed."""
    test_file = path / f"benchmark_write_{os.getpid()}.tmp"
    block_size = block_size_mb * 1024 * 1024
    n_blocks = size_mb // block_size_mb
    data = np.random.rand(block_size // 8).astype(np.float64)  # 8 bytes per float64

    try:
        start = time.perf_counter()
        with open(test_file, 'wb') as f:
            for _ in range(n_blocks):
                f.write(data.tobytes())
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.perf_counter() - start
        speed = size_mb / elapsed
        return speed
    finally:
        if test_file.exists():
            test_file.unlink()


def benchmark_sequential_read(path: Path, size_mb: int = 500, block_size_mb: int = 10) -> float:
    """Benchmark sequential read speed."""
    test_file = path / f"benchmark_read_{os.getpid()}.tmp"
    block_size = block_size_mb * 1024 * 1024
    n_blocks = size_mb // block_size_mb
    data = np.random.rand(block_size // 8).astype(np.float64)

    try:
        # Create test file first
        with open(test_file, 'wb') as f:
            for _ in range(n_blocks):
                f.write(data.tobytes())
            f.flush()
            os.fsync(f.fileno())

        # Clear cache (best effort - may require root)
        try:
            os.system('sync')
        except:
            pass

        # Read benchmark
        start = time.perf_counter()
        with open(test_file, 'rb') as f:
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
        elapsed = time.perf_counter() - start
        speed = size_mb / elapsed
        return speed
    finally:
        if test_file.exists():
            test_file.unlink()


def benchmark_zarr_pattern(path: Path, n_samples: int = 2000, n_traces: int = 10000,
                           gather_size: int = 100) -> dict:
    """
    Benchmark disk I/O in a pattern similar to seismic processing.
    Creates a zarr-like access pattern (column slices).
    """
    import zarr

    test_zarr = path / f"benchmark_zarr_{os.getpid()}.zarr"

    try:
        # Create test array
        print(f"  Creating test array ({n_samples} x {n_traces})...")
        arr = zarr.open(
            str(test_zarr),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, gather_size),
            dtype='float32'
        )

        # Fill with random data
        for i in range(0, n_traces, gather_size):
            end = min(i + gather_size, n_traces)
            arr[:, i:end] = np.random.randn(n_samples, end - i).astype(np.float32)

        # Benchmark reads (gather access pattern)
        n_reads = min(100, n_traces // gather_size)
        indices = np.random.choice(n_traces // gather_size, n_reads, replace=False)

        print(f"  Benchmarking {n_reads} random gather reads...")
        read_times = []
        bytes_per_read = n_samples * gather_size * 4  # float32

        for idx in indices:
            start_col = idx * gather_size
            end_col = min(start_col + gather_size, n_traces)

            t_start = time.perf_counter()
            _ = np.array(arr[:, start_col:end_col])
            t_end = time.perf_counter()
            read_times.append((t_end - t_start) * 1000)  # ms

        # Benchmark writes
        print(f"  Benchmarking {n_reads} random gather writes...")
        write_times = []
        test_data = np.random.randn(n_samples, gather_size).astype(np.float32)

        for idx in indices:
            start_col = idx * gather_size
            end_col = min(start_col + gather_size, n_traces)

            t_start = time.perf_counter()
            arr[:, start_col:end_col] = test_data[:, :end_col - start_col]
            t_end = time.perf_counter()
            write_times.append((t_end - t_start) * 1000)  # ms

        return {
            'n_samples': n_samples,
            'n_traces': n_traces,
            'gather_size': gather_size,
            'bytes_per_gather': bytes_per_read,
            'n_operations': n_reads,
            'read_avg_ms': np.mean(read_times),
            'read_max_ms': np.max(read_times),
            'read_min_ms': np.min(read_times),
            'read_std_ms': np.std(read_times),
            'read_speed_mbs': (bytes_per_read / (1024*1024)) / (np.mean(read_times) / 1000),
            'write_avg_ms': np.mean(write_times),
            'write_max_ms': np.max(write_times),
            'write_min_ms': np.min(write_times),
            'write_std_ms': np.std(write_times),
            'write_speed_mbs': (bytes_per_read / (1024*1024)) / (np.mean(write_times) / 1000),
        }
    finally:
        import shutil
        if test_zarr.exists():
            shutil.rmtree(test_zarr)


def main():
    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
    else:
        test_path = Path.cwd()

    if not test_path.exists():
        print(f"Error: Path does not exist: {test_path}")
        sys.exit(1)

    print("=" * 60)
    print("DISK I/O BENCHMARK")
    print("=" * 60)
    print(f"Test path: {test_path}")
    print(f"Free space: {os.statvfs(test_path).f_bavail * os.statvfs(test_path).f_frsize / (1024**3):.1f} GB")
    print()

    # Sequential benchmarks
    print("1. Sequential Write Benchmark (500 MB)...")
    write_speed = benchmark_sequential_write(test_path, size_mb=500)
    print(f"   Sequential Write: {write_speed:.1f} MB/s")

    print("\n2. Sequential Read Benchmark (500 MB)...")
    read_speed = benchmark_sequential_read(test_path, size_mb=500)
    print(f"   Sequential Read: {read_speed:.1f} MB/s")

    # Zarr pattern benchmark
    print("\n3. Zarr/Seismic Pattern Benchmark...")
    try:
        zarr_results = benchmark_zarr_pattern(test_path)
        print(f"   Gather size: {zarr_results['bytes_per_gather'] / (1024*1024):.2f} MB")
        print(f"   Read:  Avg={zarr_results['read_avg_ms']:.1f}ms, "
              f"Max={zarr_results['read_max_ms']:.1f}ms, "
              f"Speed={zarr_results['read_speed_mbs']:.1f} MB/s")
        print(f"   Write: Avg={zarr_results['write_avg_ms']:.1f}ms, "
              f"Max={zarr_results['write_max_ms']:.1f}ms, "
              f"Speed={zarr_results['write_speed_mbs']:.1f} MB/s")
    except ImportError:
        print("   Skipped (zarr not available)")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Analysis
    if write_speed < 100 or read_speed < 100:
        print("WARNING: Very slow disk I/O detected (<100 MB/s)")
        print("  - Check if disk is connected via USB 2.0 instead of USB 3/Thunderbolt")
        print("  - Check for disk errors or heavy background activity")
    elif write_speed < 300 or read_speed < 300:
        print("NOTE: Moderate disk I/O speed (100-300 MB/s)")
        print("  - Typical for HDD or slow SSD")
        print("  - Processing may be I/O bound")
    else:
        print("Good disk I/O speed detected (>300 MB/s)")
        print("  - If processing is still slow, bottleneck is elsewhere (CPU/GPU)")


if __name__ == "__main__":
    main()
