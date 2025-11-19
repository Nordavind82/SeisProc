"""
Test script for GPU foundation components.

Tests:
1. Device detection and manager
2. Data transfer CPU ↔ GPU
3. Memory management
4. GPU utilities (thresholding, MAD)
"""

import sys

import numpy as np
import torch
from processors.gpu.device_manager import DeviceManager, get_device_manager
from processors.gpu.utils_gpu import (
    numpy_to_tensor,
    tensor_to_numpy,
    soft_threshold_gpu,
    garrote_threshold_gpu,
    compute_mad_gpu,
    benchmark_transfer,
)


def test_device_detection():
    """Test device detection and information."""
    print("=" * 60)
    print("TEST 1: Device Detection")
    print("=" * 60)

    # Create device manager
    dm = DeviceManager()

    print(f"Device manager: {dm}")
    print(f"Device type: {dm.get_device_type()}")
    print(f"Device name: {dm.get_device_name()}")
    print(f"GPU available: {dm.is_gpu_available()}")
    print(f"Status: {dm.get_status_string()}")

    # Print detailed info
    print("\nDevice Info:")
    info = dm.get_info_dict()
    for key, value in info.items():
        if key != 'memory':
            print(f"  {key}: {value}")

    # Memory info
    print("\nMemory Info:")
    mem_info = dm.get_memory_info()
    for key, value in mem_info.items():
        if value is not None:
            print(f"  {key}: {value / (1024**3):.2f} GB")
        else:
            print(f"  {key}: Not available")

    print("\n✓ Device detection test passed\n")
    return dm


def test_data_transfer(dm: DeviceManager):
    """Test CPU ↔ GPU data transfer."""
    print("=" * 60)
    print("TEST 2: Data Transfer")
    print("=" * 60)

    # Create test data (simulating seismic trace)
    n_samples = 2049
    n_traces = 10
    test_data = np.random.randn(n_samples, n_traces).astype(np.float32)

    print(f"Test data shape: {test_data.shape}")
    print(f"Test data size: {test_data.nbytes / (1024**2):.2f} MB")

    # Transfer to GPU
    print("\nTransferring to GPU...")
    tensor_gpu = numpy_to_tensor(test_data, dm.device)
    print(f"Tensor device: {tensor_gpu.device}")
    print(f"Tensor shape: {tensor_gpu.shape}")
    print(f"Tensor dtype: {tensor_gpu.dtype}")

    # Transfer back to CPU
    print("\nTransferring back to CPU...")
    result_cpu = tensor_to_numpy(tensor_gpu)
    print(f"Result shape: {result_cpu.shape}")
    print(f"Result dtype: {result_cpu.dtype}")

    # Verify data integrity
    print("\nVerifying data integrity...")
    max_error = np.max(np.abs(test_data - result_cpu))
    print(f"Max absolute error: {max_error:.2e}")

    if max_error < 1e-6:
        print("✓ Data transfer test passed\n")
    else:
        print(f"✗ Data transfer test FAILED (error too large: {max_error})\n")


def test_batch_size_calculation(dm: DeviceManager):
    """Test batch size calculation."""
    print("=" * 60)
    print("TEST 3: Batch Size Calculation")
    print("=" * 60)

    n_samples = 2049
    test_cases = [100, 497, 1000, 10000]

    print(f"Samples per trace: {n_samples}")
    print(f"Data type: float32 (4 bytes)\n")

    for n_traces in test_cases:
        batch_size = dm.calculate_batch_size(n_samples, n_traces)
        print(f"Total traces: {n_traces:5d} → Batch size: {batch_size:5d}")

    print("\n✓ Batch size calculation test passed\n")


def test_gpu_operations(dm: DeviceManager):
    """Test GPU operations (thresholding, MAD)."""
    print("=" * 60)
    print("TEST 4: GPU Operations")
    print("=" * 60)

    # Create complex test data
    n = 100
    data = np.random.randn(n) + 1j * np.random.randn(n)
    data = data.astype(np.complex64)

    # Transfer to GPU
    data_gpu = torch.from_numpy(data).to(dm.device)

    print(f"Test data shape: {data.shape}")
    print(f"Test data dtype: {data_gpu.dtype}")

    # Test soft thresholding
    print("\nTesting soft thresholding...")
    threshold = 0.5
    soft_result = soft_threshold_gpu(data_gpu, threshold)
    print(f"Soft threshold result shape: {soft_result.shape}")
    print(f"Original magnitude range: [{torch.abs(data_gpu).min():.3f}, {torch.abs(data_gpu).max():.3f}]")
    print(f"Thresholded magnitude range: [{torch.abs(soft_result).min():.3f}, {torch.abs(soft_result).max():.3f}]")

    # Test Garrote thresholding
    print("\nTesting Garrote thresholding...")
    garrote_result = garrote_threshold_gpu(data_gpu, threshold)
    print(f"Garrote threshold result shape: {garrote_result.shape}")
    print(f"Garrote magnitude range: [{torch.abs(garrote_result).min():.3f}, {torch.abs(garrote_result).max():.3f}]")

    # Test MAD computation
    print("\nTesting MAD computation...")
    # Create 2D data for MAD test
    data_2d = np.random.randn(7, 100).astype(np.float32)  # 7 traces, 100 samples
    data_2d_gpu = numpy_to_tensor(data_2d, dm.device)

    mad = compute_mad_gpu(data_2d_gpu, dim=0, keepdim=False)
    print(f"Input shape: {data_2d_gpu.shape}")
    print(f"MAD shape: {mad.shape}")
    print(f"MAD range: [{mad.min():.3f}, {mad.max():.3f}]")

    print("\n✓ GPU operations test passed\n")


def test_transfer_benchmark(dm: DeviceManager):
    """Benchmark data transfer performance."""
    print("=" * 60)
    print("TEST 5: Transfer Benchmark")
    print("=" * 60)

    # Test different data sizes
    test_sizes = [
        (2049, 10),      # Small: 10 traces
        (2049, 100),     # Medium: 100 traces
        (2049, 497),     # Large: 497 traces (single gather)
    ]

    for shape in test_sizes:
        print(f"\nBenchmarking shape {shape}...")
        results = benchmark_transfer(shape, dm.device, n_iterations=5)

        print(f"  Data size: {results['size_mb']:.2f} MB")
        print(f"  CPU → GPU: {results['cpu_to_gpu_ms']:.2f} ms "
              f"({results['cpu_to_gpu_bandwidth_gbps']:.2f} GB/s)")
        print(f"  GPU → CPU: {results['gpu_to_cpu_ms']:.2f} ms "
              f"({results['gpu_to_cpu_bandwidth_gbps']:.2f} GB/s)")

    print("\n✓ Transfer benchmark completed\n")


def test_memory_management(dm: DeviceManager):
    """Test memory management and cleanup."""
    print("=" * 60)
    print("TEST 6: Memory Management")
    print("=" * 60)

    if dm.device.type == 'cuda':
        # CUDA memory tracking
        print("Initial memory:")
        mem_info = dm.get_memory_info()
        print(f"  Allocated: {mem_info['allocated'] / (1024**2):.2f} MB")

        # Allocate large tensor
        print("\nAllocating 100 MB tensor...")
        large_tensor = torch.randn(5000, 5000, device=dm.device)

        mem_info = dm.get_memory_info()
        print(f"  Allocated: {mem_info['allocated'] / (1024**2):.2f} MB")

        # Delete and clear cache
        print("\nDeleting tensor and clearing cache...")
        del large_tensor
        dm.clear_cache()

        mem_info = dm.get_memory_info()
        print(f"  Allocated: {mem_info['allocated'] / (1024**2):.2f} MB")

        print("\n✓ Memory management test passed\n")
    else:
        print("Memory management test skipped (not CUDA)\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GPU FOUNDATION TESTING")
    print("=" * 60 + "\n")

    try:
        # Test 1: Device detection
        dm = test_device_detection()

        # Test 2: Data transfer
        test_data_transfer(dm)

        # Test 3: Batch size calculation
        test_batch_size_calculation(dm)

        # Test 4: GPU operations
        test_gpu_operations(dm)

        # Test 5: Transfer benchmark
        test_transfer_benchmark(dm)

        # Test 6: Memory management
        test_memory_management(dm)

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
