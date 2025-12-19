# MLX C++ Kernel Acceleration Plan for SeisProc

## Executive Summary

This document outlines tangible tasks for accelerating SeisProc processing tools through MLX C++ precompiled kernels, following the successful approach implemented in the PSTM application.

**Target Processors (User Priority):**
1. DWT-Denoise (including SWT mode)
2. STFT-Denoise
3. FKK-Filter (GPU)
4. Gabor-Denoise

---

## Actual Benchmark Results (4000 traces x 1600 samples)

These are **real profiling results** from the benchmark suite:

### Throughput Summary

| Processor | Traces/s | Msamples/s | Time (ms) | Bottleneck |
|-----------|----------|------------|-----------|------------|
| **DWT** | 16,462 | 26.3 | 243 | Evenly distributed |
| **SWT** | 3,087 | 4.9 | 1,296 | **wavelet_reconstruction (69%)** |
| **STFT** | 1,932 | 3.1 | 2,071 | **mad_computation (52%)** |
| **Gabor** | 1,137 | 1.8 | 3,518 | **mad_computation (56%)** |
| **FKK** | 144,762 | 231.6 | 27 | cpu_to_gpu transfer (48%) |

### Component Breakdown

**DWT-Denoise:**
```
wavelet_decomposition    31.4%   (76 ms)
wavelet_reconstruction   24.2%   (59 ms)
thresholding            22.3%   (54 ms)
mad_computation         22.1%   (54 ms)
```

**SWT-Denoise (5.3x slower than DWT):**
```
wavelet_reconstruction   69.4%   (899 ms)  ← CRITICAL BOTTLENECK
wavelet_decomposition    19.6%   (254 ms)
thresholding             6.3%   (82 ms)
mad_computation          4.7%   (61 ms)
```

**STFT-Denoise:**
```
mad_computation          52.1%   (1,062 ms)  ← MAIN BOTTLENECK
stft_forward            20.8%   (424 ms)
istft_inverse           18.6%   (379 ms)
thresholding             8.5%   (174 ms)
```

**Gabor-Denoise (slowest overall):**
```
mad_computation          56.2%   (1,959 ms)  ← MAIN BOTTLENECK
gabor_inverse           17.1%   (595 ms)
gabor_forward           16.6%   (577 ms)
soft_thresholding       10.1%   (353 ms)
```

**FKK-Filter (warm GPU):**
```
cpu_to_gpu              47.9%   (13 ms)   ← Transfer overhead
mask_building           24.2%   (7 ms)
ifft_3d                  9.1%   (2.5 ms)
fft_3d                   8.9%   (2.5 ms)
```

### Key Insights for C++ Acceleration

1. **SWT**: The `iswt` reconstruction is the killer (69%). Batch Metal kernel for reconstruction would give **~5x speedup**.

2. **STFT/Gabor**: MAD computation dominates (~52-56%). This is `np.median()` called 4000 times per gather. GPU parallel median would give **3-4x speedup**.

3. **FKK**: Already very fast on GPU (231 Msamples/s). Main gain from eliminating CPU↔GPU transfer via unified memory.

4. **DWT**: Already fast (26 Msamples/s), but batch Metal FFT could still give **2-3x improvement**.

---

## Phase 0: Benchmarking Infrastructure

Before implementing C++ kernels, we must identify the exact bottlenecks through profiling.

### Task 0.1: Create Benchmarking Framework

```python
# File: benchmarks/kernel_profiler.py

import numpy as np
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Callable
import json

@dataclass
class ProfileResult:
    name: str
    total_time_ms: float
    iterations: int
    time_per_iter_ms: float
    samples_per_sec: float
    traces_per_sec: float

@contextmanager
def profile_section(name: str, results: Dict[str, float]):
    """Context manager for profiling code sections."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000  # ms
    results[name] = results.get(name, 0) + elapsed

def generate_test_data(n_traces: int = 500, n_samples: int = 2000,
                       noise_level: float = 0.3) -> np.ndarray:
    """Generate synthetic seismic data for benchmarking."""
    # Create reflectivity with realistic frequency content
    t = np.linspace(0, 2.0, n_samples)  # 2 seconds

    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        # Ricker wavelets at random times
        for _ in range(10):
            t0 = np.random.uniform(0.2, 1.8)
            f0 = np.random.uniform(20, 60)
            wavelet = (1 - 2 * (np.pi * f0 * (t - t0))**2) * \
                      np.exp(-(np.pi * f0 * (t - t0))**2)
            traces[:, i] += wavelet * np.random.uniform(0.5, 1.5)

        # Add noise
        traces[:, i] += np.random.randn(n_samples) * noise_level

    return traces

def run_benchmark(func: Callable, data: np.ndarray,
                  n_iterations: int = 5, warmup: int = 1) -> ProfileResult:
    """Run benchmark with warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup):
        _ = func(data)

    # Timed runs
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = func(data)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    n_samples, n_traces = data.shape
    avg_time = np.mean(times)

    return ProfileResult(
        name=func.__name__,
        total_time_ms=sum(times),
        iterations=n_iterations,
        time_per_iter_ms=avg_time,
        samples_per_sec=(n_samples * n_traces) / (avg_time / 1000),
        traces_per_sec=n_traces / (avg_time / 1000)
    )
```

---

## Phase 1: DWT/SWT Denoise Acceleration

### Current Bottleneck Analysis

**File:** `processors/dwt_denoise.py`

**Critical Loop (lines 209-226):**
```python
for i in range(n_traces):                    # Python loop - BOTTLENECK
    coeffs = pywt.wavedec(traces[:, i], ...)  # Per-trace FFT
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # ... thresholding ...
    denoised[:, i] = pywt.waverec(...)        # Per-trace IFFT
```

**Identified Bottlenecks:**
1. Python `for` loop over traces (eliminable with C++)
2. `pywt.wavedec` called n_traces times (batch-able)
3. `np.median` for MAD (parallelizable)
4. `pywt.waverec` called n_traces times (batch-able)

### Task 1.1: DWT Kernel Benchmarking

```python
# File: benchmarks/benchmark_dwt.py

import numpy as np
import time
import pywt
from benchmarks.kernel_profiler import generate_test_data, profile_section

def benchmark_dwt_components(n_traces=500, n_samples=2000, level=5, wavelet='db4'):
    """Profile each component of DWT denoising."""
    traces = generate_test_data(n_traces, n_samples)
    results = {}

    # Component 1: Wavelet Decomposition (wavedec)
    with profile_section("wavedec_loop", results):
        all_coeffs = []
        for i in range(n_traces):
            coeffs = pywt.wavedec(traces[:, i], wavelet, level=level)
            all_coeffs.append(coeffs)

    # Component 2: MAD computation per trace
    with profile_section("mad_computation", results):
        thresholds = []
        for coeffs in all_coeffs:
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresholds.append(3.0 * sigma)

    # Component 3: Thresholding
    with profile_section("thresholding", results):
        thresholded_coeffs = []
        for coeffs, thresh in zip(all_coeffs, thresholds):
            new_coeffs = [coeffs[0]]  # Keep approximation
            for c in coeffs[1:]:
                new_coeffs.append(pywt.threshold(c, thresh, mode='soft'))
            thresholded_coeffs.append(new_coeffs)

    # Component 4: Reconstruction (waverec)
    with profile_section("waverec_loop", results):
        denoised = np.zeros_like(traces)
        for i, coeffs in enumerate(thresholded_coeffs):
            denoised[:, i] = pywt.waverec(coeffs, wavelet)[:n_samples]

    # Print results
    total = sum(results.values())
    print(f"\n=== DWT Benchmark ({n_traces} traces × {n_samples} samples) ===")
    print(f"{'Component':<25} {'Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for name, time_ms in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<25} {time_ms:<15.2f} {100*time_ms/total:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.2f}")
    print(f"\nThroughput: {n_traces / (total/1000):.1f} traces/s")
    print(f"            {n_traces * n_samples / (total/1000) / 1e6:.2f} Msamples/s")

    return results

def benchmark_swt_components(n_traces=500, n_samples=2048, level=5, wavelet='db4'):
    """Profile SWT variant (requires power-of-2 samples)."""
    # SWT requires power-of-2 length
    traces = generate_test_data(n_traces, n_samples)
    results = {}

    # Component 1: SWT Decomposition
    with profile_section("swt_decomp", results):
        all_coeffs = []
        for i in range(n_traces):
            coeffs = pywt.swt(traces[:, i], wavelet, level=level)
            all_coeffs.append(coeffs)

    # Component 2: MAD computation
    with profile_section("mad_computation", results):
        thresholds = []
        for coeffs in all_coeffs:
            # SWT returns list of (cA, cD) tuples
            sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
            thresholds.append(3.0 * sigma)

    # Component 3: Thresholding
    with profile_section("thresholding", results):
        thresholded_coeffs = []
        for coeffs, thresh in zip(all_coeffs, thresholds):
            new_coeffs = []
            for cA, cD in coeffs:
                cD_thresh = pywt.threshold(cD, thresh, mode='soft')
                new_coeffs.append((cA, cD_thresh))
            thresholded_coeffs.append(new_coeffs)

    # Component 4: Inverse SWT
    with profile_section("iswt_recon", results):
        denoised = np.zeros_like(traces)
        for i, coeffs in enumerate(thresholded_coeffs):
            denoised[:, i] = pywt.iswt(coeffs, wavelet)

    total = sum(results.values())
    print(f"\n=== SWT Benchmark ({n_traces} traces × {n_samples} samples) ===")
    print(f"{'Component':<25} {'Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for name, time_ms in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<25} {time_ms:<15.2f} {100*time_ms/total:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.2f}")

    return results

if __name__ == "__main__":
    # Run benchmarks at different scales
    for n_traces in [100, 500, 1000]:
        benchmark_dwt_components(n_traces=n_traces)

    print("\n" + "="*60)
    print("SWT BENCHMARKS (translation-invariant variant)")
    print("="*60)

    for n_traces in [100, 500, 1000]:
        benchmark_swt_components(n_traces=n_traces)
```

### Task 1.2: DWT C++ Kernel Implementation Tasks

| Task ID | Description | Priority | Estimated Speedup |
|---------|-------------|----------|-------------------|
| DWT-1 | Implement batch `wavedec` in Metal (all traces in single dispatch) | HIGH | 10-20x |
| DWT-2 | Implement parallel MAD computation across traces | HIGH | 5-10x |
| DWT-3 | Implement batch thresholding (soft/hard) | MEDIUM | 3-5x |
| DWT-4 | Implement batch `waverec` in Metal | HIGH | 10-20x |
| DWT-5 | SWT variant with padding handling | MEDIUM | 15-25x |

### Task 1.3: DWT Metal Kernel Structure

```
seismic_metal/
├── shaders/
│   ├── dwt_decompose.metal      # Batch wavelet decomposition
│   ├── dwt_threshold.metal      # Parallel MAD + thresholding
│   └── dwt_reconstruct.metal    # Batch reconstruction
├── src/
│   ├── dwt_kernel.mm            # Objective-C Metal device code
│   └── dwt_bindings.cpp         # pybind11 bindings
└── include/
    └── dwt_kernel.h             # C++ API
```

---

## Phase 2: STFT Denoise Acceleration

### Current Bottleneck Analysis

**File:** `processors/stft_denoise.py`

**Critical Path (lines 202-209):**
```python
for trace_idx in range(n_traces):           # Python loop - BOTTLENECK
    ensemble = traces[:, start_idx:end_idx]  # Spatial aperture
    denoised_traces[:, trace_idx] = self._process_ensemble(...)
```

**Inside `_process_ensemble` (lines 261-328):**
```python
freqs, times, stft_batch = signal.stft(ensemble.T, ...)  # Batch STFT
median_amp = np.median(all_amplitudes, axis=0)           # MAD stats
_, denoised_trace = signal.istft(stft_denoised, ...)     # ISTFT
```

**Identified Bottlenecks:**
1. Python loop over traces (eliminable)
2. `signal.stft` per ensemble (could batch more)
3. `np.median` for spatial MAD (parallelizable)
4. Thresholding logic (vectorized but could be GPU)
5. `signal.istft` per trace (batch-able)

### Task 2.1: STFT Kernel Benchmarking

```python
# File: benchmarks/benchmark_stft.py

import numpy as np
import time
from scipy import signal
from scipy.ndimage import uniform_filter1d
from benchmarks.kernel_profiler import generate_test_data, profile_section

def benchmark_stft_components(n_traces=500, n_samples=2000,
                               aperture=7, nperseg=64, noverlap=32):
    """Profile each component of STFT denoising."""
    traces = generate_test_data(n_traces, n_samples)
    results = {}
    half_ap = aperture // 2

    # Pre-allocate output
    denoised = np.zeros_like(traces)

    # Profile main processing loop
    with profile_section("total_processing", results):
        for trace_idx in range(n_traces):
            start_idx = max(0, trace_idx - half_ap)
            end_idx = min(n_traces, trace_idx + half_ap + 1)
            ensemble = traces[:, start_idx:end_idx]

            # Sub-profile: STFT
            t0 = time.perf_counter()
            freqs, times, stft_batch = signal.stft(
                ensemble.T, nperseg=nperseg, noverlap=noverlap, axis=-1
            )
            results["stft_forward"] = results.get("stft_forward", 0) + \
                                       (time.perf_counter() - t0) * 1000

            # Sub-profile: MAD computation
            t0 = time.perf_counter()
            all_amplitudes = np.abs(stft_batch)
            median_amp = np.median(all_amplitudes, axis=0)
            mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)
            mad_scaled = mad * 1.4826
            results["mad_computation"] = results.get("mad_computation", 0) + \
                                          (time.perf_counter() - t0) * 1000

            # Sub-profile: Thresholding
            t0 = time.perf_counter()
            center_idx = trace_idx - start_idx
            stft_center = stft_batch[center_idx]
            magnitudes = np.abs(stft_center)
            phases = np.angle(stft_center)
            threshold = 3.0 * mad_scaled
            threshold = np.maximum(threshold, 1e-10)
            deviations = np.abs(magnitudes - median_amp)
            new_magnitudes = np.where(deviations > threshold, median_amp, magnitudes)
            stft_denoised = new_magnitudes * np.exp(1j * phases)
            results["thresholding"] = results.get("thresholding", 0) + \
                                       (time.perf_counter() - t0) * 1000

            # Sub-profile: ISTFT
            t0 = time.perf_counter()
            _, denoised_trace = signal.istft(stft_denoised, nperseg=nperseg,
                                              noverlap=noverlap)
            denoised[:len(denoised_trace), trace_idx] = denoised_trace[:n_samples]
            results["istft_inverse"] = results.get("istft_inverse", 0) + \
                                        (time.perf_counter() - t0) * 1000

    # Print results
    total = results["total_processing"]
    print(f"\n=== STFT Benchmark ({n_traces} traces × {n_samples} samples) ===")
    print(f"Aperture: {aperture}, nperseg: {nperseg}, noverlap: {noverlap}")
    print(f"{'Component':<25} {'Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for name, time_ms in sorted(results.items(), key=lambda x: -x[1]):
        if name != "total_processing":
            print(f"{name:<25} {time_ms:<15.2f} {100*time_ms/total:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.2f}")
    print(f"\nThroughput: {n_traces / (total/1000):.1f} traces/s")

    return results

def benchmark_stft_batch_potential(n_traces=500, n_samples=2000, nperseg=64):
    """
    Estimate potential speedup from batching all traces in single STFT.

    Current: STFT per ensemble (aperture traces at a time)
    Potential: Single batch STFT for all traces
    """
    traces = generate_test_data(n_traces, n_samples)
    results = {}

    # Current approach: per-trace STFT
    with profile_section("per_trace_stft", results):
        for i in range(n_traces):
            _, _, _ = signal.stft(traces[:, i], nperseg=nperseg)

    # Potential: single batch STFT
    with profile_section("batch_stft", results):
        _, _, stft_all = signal.stft(traces.T, nperseg=nperseg, axis=-1)

    print(f"\n=== STFT Batching Potential ===")
    print(f"Per-trace STFT: {results['per_trace_stft']:.2f} ms")
    print(f"Batch STFT:     {results['batch_stft']:.2f} ms")
    print(f"Speedup:        {results['per_trace_stft']/results['batch_stft']:.1f}x")

    return results

if __name__ == "__main__":
    for n_traces in [100, 500, 1000]:
        benchmark_stft_components(n_traces=n_traces)

    print("\n" + "="*60)
    benchmark_stft_batch_potential()
```

### Task 2.2: STFT C++ Kernel Implementation Tasks

| Task ID | Description | Priority | Estimated Speedup |
|---------|-------------|----------|-------------------|
| STFT-1 | Implement batch STFT in Metal (all traces, single dispatch) | HIGH | 15-25x |
| STFT-2 | Implement GPU spatial MAD computation | HIGH | 10-15x |
| STFT-3 | Implement GPU thresholding (soft/hard/scaled/adaptive) | MEDIUM | 5-10x |
| STFT-4 | Implement batch ISTFT in Metal | HIGH | 15-25x |
| STFT-5 | Fuse STFT+threshold+ISTFT into single kernel | LOW | Additional 2-3x |

### Task 2.3: STFT Metal Kernel Structure

```
seismic_metal/
├── shaders/
│   ├── stft_forward.metal       # Batch STFT (windowing + FFT)
│   ├── stft_mad_threshold.metal # Spatial MAD + thresholding
│   └── stft_inverse.metal       # Batch ISTFT (IFFT + overlap-add)
├── src/
│   ├── stft_kernel.mm
│   └── stft_bindings.cpp
└── include/
    └── stft_kernel.h
```

---

## Phase 3: FKK Filter Acceleration

### Current Implementation Analysis

**File:** `processors/fkk_filter_gpu.py`

**Already GPU-accelerated via PyTorch:**
- 3D FFT: `torch.fft.rfft`, `torch.fft.fft` (lines 454-457)
- Mask application: element-wise multiply (line 465)
- 3D IFFT: `torch.fft.ifft`, `torch.fft.irfft` (lines 468-471)

**Potential Bottlenecks:**
1. Padding operations (`pad_copy_3d`, `pad_copy_temporal`) - pure NumPy
2. Mask building (`build_velocity_cone_mask`) - GPU but Python orchestrated
3. AGC application - CPU vectorized
4. Data transfer CPU↔GPU

### Task 3.1: FKK Kernel Benchmarking

```python
# File: benchmarks/benchmark_fkk.py

import numpy as np
import torch
import time
from benchmarks.kernel_profiler import profile_section

def generate_test_volume(nt=512, nx=64, ny=64):
    """Generate synthetic 3D seismic volume."""
    volume = np.random.randn(nt, nx, ny).astype(np.float32)
    # Add some coherent energy
    for i in range(nx):
        for j in range(ny):
            t0 = 100 + i * 2 + j * 1.5
            volume[int(t0):int(t0)+50, i, j] += 2.0
    return volume

def benchmark_fkk_components(nt=512, nx=64, ny=64, device='mps'):
    """Profile FKK filter components."""
    volume = generate_test_volume(nt, nx, ny)
    results = {}

    # Check device availability
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"Using device: {device}")

    # Component 1: Padding (CPU)
    with profile_section("padding_cpu", results):
        pad_x, pad_y = 10, 10
        pad_t = 50
        # Temporal padding
        padded = np.pad(volume, ((pad_t, pad_t), (0, 0), (0, 0)), mode='reflect')
        # Spatial padding
        padded = np.pad(padded, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode='reflect')
        # FFT padding to power-of-2
        nt_fft = 2**int(np.ceil(np.log2(padded.shape[0])))
        nx_fft = 2**int(np.ceil(np.log2(padded.shape[1])))
        ny_fft = 2**int(np.ceil(np.log2(padded.shape[2])))
        padded = np.pad(padded, (
            (0, nt_fft - padded.shape[0]),
            (0, nx_fft - padded.shape[1]),
            (0, ny_fft - padded.shape[2])
        ), mode='constant')

    # Component 2: CPU→GPU transfer
    with profile_section("cpu_to_gpu_transfer", results):
        data_gpu = torch.from_numpy(padded).float().to(device)
        if device != 'cpu':
            torch.mps.synchronize() if device == 'mps' else torch.cuda.synchronize()

    # Component 3: 3D FFT
    with profile_section("fft_3d", results):
        spectrum = torch.fft.rfft(data_gpu, dim=0)
        spectrum = torch.fft.fft(spectrum, dim=1)
        spectrum = torch.fft.fft(spectrum, dim=2)
        spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))
        if device != 'cpu':
            torch.mps.synchronize() if device == 'mps' else torch.cuda.synchronize()

    # Component 4: Build velocity cone mask
    with profile_section("mask_building", results):
        dt, dx, dy = 0.002, 12.5, 12.5  # Example geometry
        nf = spectrum.shape[0]
        f_axis = torch.fft.rfftfreq(nt_fft, dt, device=device, dtype=torch.float32)
        kx_axis = torch.fft.fftshift(torch.fft.fftfreq(nx_fft, dx, device=device, dtype=torch.float32))
        ky_axis = torch.fft.fftshift(torch.fft.fftfreq(ny_fft, dy, device=device, dtype=torch.float32))

        f_grid, kx_grid, ky_grid = torch.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')
        k_horizontal = torch.sqrt(kx_grid**2 + ky_grid**2)
        k_safe = torch.where(k_horizontal > 1e-10, k_horizontal,
                             torch.tensor(1e-10, device=device))
        velocity = torch.abs(f_grid) / k_safe

        # Simple velocity filter
        v_min, v_max = 200, 2000
        mask = torch.ones_like(velocity)
        mask[(velocity >= v_min) & (velocity <= v_max)] = 0.0
        mask[0, :, :] = 1.0  # Preserve DC

        if device != 'cpu':
            torch.mps.synchronize() if device == 'mps' else torch.cuda.synchronize()

    # Component 5: Apply mask
    with profile_section("mask_application", results):
        spectrum_filtered = spectrum * mask
        if device != 'cpu':
            torch.mps.synchronize() if device == 'mps' else torch.cuda.synchronize()

    # Component 6: 3D IFFT
    with profile_section("ifft_3d", results):
        spectrum_filtered = torch.fft.ifftshift(spectrum_filtered, dim=(1, 2))
        spectrum_filtered = torch.fft.ifft(spectrum_filtered, dim=2)
        spectrum_filtered = torch.fft.ifft(spectrum_filtered, dim=1)
        result = torch.fft.irfft(spectrum_filtered, n=nt_fft, dim=0)
        if device != 'cpu':
            torch.mps.synchronize() if device == 'mps' else torch.cuda.synchronize()

    # Component 7: GPU→CPU transfer
    with profile_section("gpu_to_cpu_transfer", results):
        output = result.real.cpu().numpy()

    # Component 8: Unpadding
    with profile_section("unpadding", results):
        output = output[:nt + 2*pad_t, :nx + 2*pad_x, :ny + 2*pad_y]
        output = output[pad_t:-pad_t, pad_x:-pad_x, pad_y:-pad_y]

    # Print results
    total = sum(results.values())
    print(f"\n=== FKK Benchmark ({nt}×{nx}×{ny} volume, device={device}) ===")
    print(f"FFT size: {nt_fft}×{nx_fft}×{ny_fft}")
    print(f"{'Component':<25} {'Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for name, time_ms in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<25} {time_ms:<15.2f} {100*time_ms/total:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.2f}")
    print(f"\nThroughput: {nt*nx*ny / (total/1000) / 1e6:.2f} Msamples/s")

    return results

if __name__ == "__main__":
    # Compare CPU vs GPU
    print("="*60)
    print("CPU BENCHMARK")
    print("="*60)
    benchmark_fkk_components(device='cpu')

    print("\n" + "="*60)
    print("GPU (MPS) BENCHMARK")
    print("="*60)
    benchmark_fkk_components(device='mps')

    # Scale test
    print("\n" + "="*60)
    print("SCALE TEST (GPU)")
    print("="*60)
    for size in [(256, 32, 32), (512, 64, 64), (1024, 128, 128)]:
        benchmark_fkk_components(*size, device='mps')
```

### Task 3.2: FKK C++ Kernel Implementation Tasks

| Task ID | Description | Priority | Estimated Speedup |
|---------|-------------|----------|-------------------|
| FKK-1 | Implement GPU padding kernels (eliminate CPU padding) | MEDIUM | 2-3x |
| FKK-2 | Fuse mask building into single Metal kernel | MEDIUM | 2-4x |
| FKK-3 | Implement Metal 3D FFT (replace PyTorch FFT) | LOW | Marginal (PyTorch already optimized) |
| FKK-4 | Zero-copy memory integration (eliminate transfers) | HIGH | 3-5x |
| FKK-5 | AGC GPU implementation | MEDIUM | 2-3x |

**Note:** FKK already uses GPU. Main gains from eliminating CPU↔GPU transfers and padding overhead.

---

## Phase 4: Gabor Denoise Acceleration

### Current Bottleneck Analysis

**File:** `processors/gabor_denoise.py`

**Nearly identical to STFT** with Gaussian window:
```python
for trace_idx in range(n_traces):                    # Python loop
    ensemble = traces[:, start_idx:end_idx]
    denoised_traces[:, trace_idx] = self._process_gabor(...)  # STFT + threshold
```

**`_process_gabor` uses:**
- `signal.stft` with Gaussian window
- MAD computation
- Thresholding
- `signal.istft`

### Task 4.1: Gabor Kernel Benchmarking

```python
# File: benchmarks/benchmark_gabor.py

import numpy as np
import time
from scipy import signal
from benchmarks.kernel_profiler import generate_test_data, profile_section

def create_gaussian_window(nperseg, sigma=None):
    """Create Gaussian window for Gabor Transform."""
    if sigma is None:
        sigma = nperseg / 6.0
    return signal.windows.gaussian(nperseg, std=sigma)

def benchmark_gabor_components(n_traces=500, n_samples=2000,
                                aperture=7, window_size=64, overlap_pct=75):
    """Profile Gabor denoising components."""
    traces = generate_test_data(n_traces, n_samples)
    results = {}

    half_ap = aperture // 2
    nperseg = window_size
    noverlap = int(nperseg * overlap_pct / 100)
    gabor_window = create_gaussian_window(nperseg)

    with profile_section("total_processing", results):
        for trace_idx in range(n_traces):
            start_idx = max(0, trace_idx - half_ap)
            end_idx = min(n_traces, trace_idx + half_ap + 1)
            ensemble = traces[:, start_idx:end_idx]

            # Gabor Transform (STFT with Gaussian window)
            t0 = time.perf_counter()
            _, _, stft_batch = signal.stft(
                ensemble.T, nperseg=nperseg, noverlap=noverlap,
                window=gabor_window, axis=-1
            )
            results["gabor_forward"] = results.get("gabor_forward", 0) + \
                                        (time.perf_counter() - t0) * 1000

            # MAD computation
            t0 = time.perf_counter()
            all_amplitudes = np.abs(stft_batch)
            median_amp = np.median(all_amplitudes, axis=0)
            mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)
            results["mad_computation"] = results.get("mad_computation", 0) + \
                                          (time.perf_counter() - t0) * 1000

            # Thresholding
            t0 = time.perf_counter()
            center_idx = trace_idx - start_idx
            stft_center = stft_batch[center_idx]
            magnitudes = np.abs(stft_center)
            phases = np.angle(stft_center)
            threshold = 3.0 * mad * 1.4826
            threshold = np.maximum(threshold, 1e-10)

            # Soft threshold
            deviations = np.abs(magnitudes - median_amp)
            signs = np.where(magnitudes >= median_amp, 1, -1)
            new_deviations = np.maximum(deviations - threshold, 0)
            new_magnitudes = np.maximum(median_amp + signs * new_deviations, 0)
            stft_denoised = new_magnitudes * np.exp(1j * phases)
            results["thresholding"] = results.get("thresholding", 0) + \
                                       (time.perf_counter() - t0) * 1000

            # Inverse Gabor Transform
            t0 = time.perf_counter()
            _, denoised_trace = signal.istft(
                stft_denoised, nperseg=nperseg, noverlap=noverlap,
                window=gabor_window
            )
            results["gabor_inverse"] = results.get("gabor_inverse", 0) + \
                                        (time.perf_counter() - t0) * 1000

    total = results["total_processing"]
    print(f"\n=== Gabor Benchmark ({n_traces} traces × {n_samples} samples) ===")
    print(f"Aperture: {aperture}, window: {window_size}, overlap: {overlap_pct}%")
    print(f"{'Component':<25} {'Time (ms)':<15} {'Percentage':<10}")
    print("-" * 50)
    for name, time_ms in sorted(results.items(), key=lambda x: -x[1]):
        if name != "total_processing":
            print(f"{name:<25} {time_ms:<15.2f} {100*time_ms/total:<10.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total:<15.2f}")
    print(f"\nThroughput: {n_traces / (total/1000):.1f} traces/s")

    return results

def compare_stft_vs_gabor(n_traces=500, n_samples=2000):
    """Compare STFT vs Gabor processing time."""
    from benchmarks.benchmark_stft import benchmark_stft_components

    print("\n" + "="*60)
    print("STFT vs GABOR COMPARISON")
    print("="*60)

    stft_results = benchmark_stft_components(n_traces, n_samples)
    gabor_results = benchmark_gabor_components(n_traces, n_samples)

    print(f"\nSTFT total:  {stft_results['total_processing']:.2f} ms")
    print(f"Gabor total: {gabor_results['total_processing']:.2f} ms")

    return stft_results, gabor_results

if __name__ == "__main__":
    for n_traces in [100, 500, 1000]:
        benchmark_gabor_components(n_traces=n_traces)
```

### Task 4.2: Gabor C++ Kernel Implementation Tasks

**Same as STFT with Gaussian window support:**

| Task ID | Description | Priority | Estimated Speedup |
|---------|-------------|----------|-------------------|
| GAB-1 | Reuse STFT Metal kernels with Gaussian window | HIGH | 15-25x |
| GAB-2 | Pre-compute Gaussian window coefficients | LOW | Marginal |
| GAB-3 | Implement combined STFT+Gabor kernel selector | MEDIUM | Code reuse |

---

## Phase 5: Infrastructure & Integration

### Task 5.1: Build System Setup

```bash
# Directory structure for Metal kernels
seismic_metal/
├── CMakeLists.txt
├── include/
│   ├── common_types.h           # Shared data structures
│   ├── dwt_kernel.h
│   ├── stft_kernel.h
│   └── fkk_kernel.h
├── src/
│   ├── bindings.cpp             # Main pybind11 module
│   ├── device_manager.mm        # Metal device singleton
│   ├── dwt_kernel.mm
│   ├── stft_kernel.mm
│   └── fkk_kernel.mm
├── shaders/
│   ├── dwt_decompose.metal
│   ├── dwt_threshold.metal
│   ├── dwt_reconstruct.metal
│   ├── stft_forward.metal
│   ├── stft_mad_threshold.metal
│   └── stft_inverse.metal
├── python/
│   └── __init__.py              # Python module interface
└── tests/
    ├── test_dwt_kernel.py
    ├── test_stft_kernel.py
    └── test_fkk_kernel.py
```

### Task 5.2: CMakeLists.txt Template

```cmake
cmake_minimum_required(VERSION 3.18)
project(seismic_metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Find macOS frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Compile Metal shaders
set(METAL_SHADERS
    shaders/dwt_decompose.metal
    shaders/dwt_threshold.metal
    shaders/dwt_reconstruct.metal
    shaders/stft_forward.metal
    shaders/stft_mad_threshold.metal
    shaders/stft_inverse.metal
)

foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/${SHADER_NAME}.air
        COMMAND xcrun metal -c ${CMAKE_SOURCE_DIR}/${SHADER}
                -o ${CMAKE_BINARY_DIR}/${SHADER_NAME}.air
                -ffast-math -O3
        DEPENDS ${CMAKE_SOURCE_DIR}/${SHADER}
    )
    list(APPEND SHADER_AIR_FILES ${CMAKE_BINARY_DIR}/${SHADER_NAME}.air)
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
    COMMAND xcrun metallib ${SHADER_AIR_FILES}
            -o ${CMAKE_BINARY_DIR}/seismic_kernels.metallib
    DEPENDS ${SHADER_AIR_FILES}
)

# Build Python module
set(SOURCES
    src/device_manager.mm
    src/dwt_kernel.mm
    src/stft_kernel.mm
    src/fkk_kernel.mm
    src/bindings.cpp
)

pybind11_add_module(seismic_metal ${SOURCES})

target_include_directories(seismic_metal PRIVATE include)
target_link_libraries(seismic_metal PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_options(seismic_metal PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-O3 -ffast-math -fobjc-arc>
)

# Embed shader path
target_compile_definitions(seismic_metal PRIVATE
    SHADER_PATH="${CMAKE_BINARY_DIR}/seismic_kernels.metallib"
)
```

### Task 5.3: Integration Pattern (Kernel Factory)

```python
# File: processors/kernel_factory.py

from enum import Enum
from typing import Protocol, Optional
import logging

logger = logging.getLogger(__name__)

class ComputeBackend(Enum):
    METAL_CPP = "metal_cpp"      # Native Metal C++ kernels
    PYTORCH_GPU = "pytorch_gpu"  # PyTorch GPU
    NUMBA_CPU = "numba_cpu"      # Numba JIT
    NUMPY = "numpy"              # Pure NumPy reference

class DWTKernel(Protocol):
    """Protocol for DWT kernel implementations."""
    def decompose(self, traces: np.ndarray, wavelet: str, level: int) -> list: ...
    def threshold(self, coeffs: list, k: float, mode: str) -> list: ...
    def reconstruct(self, coeffs: list, wavelet: str) -> np.ndarray: ...

def get_dwt_kernel(backend: Optional[ComputeBackend] = None) -> DWTKernel:
    """Get best available DWT kernel."""
    if backend is None:
        # Auto-select best available
        try:
            from seismic_metal import dwt_kernel
            logger.info("Using Metal C++ DWT kernel")
            return dwt_kernel
        except ImportError:
            pass

        try:
            import pywt
            logger.info("Using PyWavelets DWT kernel")
            from processors.dwt_denoise import DWTDenoise
            return DWTDenoise
        except ImportError:
            pass

    raise RuntimeError("No DWT kernel available")
```

---

## Summary: Implementation Priority

### High Priority (Largest Impact)
1. **DWT-1, DWT-4**: Batch wavelet transform (wavedec/waverec) - 10-20x speedup
2. **STFT-1, STFT-4**: Batch STFT/ISTFT - 15-25x speedup
3. **FKK-4**: Zero-copy memory integration - 3-5x speedup

### Medium Priority
4. **DWT-2, STFT-2**: GPU MAD computation - 5-15x speedup
5. **DWT-5**: SWT variant - 15-25x speedup
6. **FKK-1**: GPU padding kernels - 2-3x speedup

### Low Priority (Diminishing Returns)
7. **STFT-5**: Fused kernel - Additional 2-3x
8. **FKK-3**: Metal FFT - Marginal (PyTorch already good)
9. **GAB-2**: Pre-computed windows - Marginal

---

## Expected Overall Speedup

| Processor | Current (traces/s) | Projected (traces/s) | Speedup |
|-----------|-------------------|---------------------|---------|
| DWT       | ~500-1000         | ~15,000-30,000      | 15-30x  |
| SWT       | ~200-400          | ~8,000-15,000       | 20-40x  |
| STFT      | ~100-300          | ~3,000-8,000        | 15-30x  |
| Gabor     | ~100-300          | ~3,000-8,000        | 15-30x  |
| FKK       | ~10-50 vol/s      | ~30-100 vol/s       | 3-5x    |

---

## Next Steps

1. Run benchmarking scripts to establish baseline metrics
2. Implement DWT Metal kernels (highest ROI)
3. Port STFT/Gabor (share kernel code)
4. Optimize FKK memory transfers
5. Create unified kernel factory with fallbacks
