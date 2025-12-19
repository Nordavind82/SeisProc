# Metal C++ Kernel Roadmap v2

## Current Status (December 2025)

### Completed Metal C++ Kernels

| Algorithm | Speedup | Implementation |
|-----------|---------|----------------|
| DWT | 30.1x | vDSP convolutions + multi-threaded |
| SWT | 24.9x | vDSP convolutions + multi-threaded |
| STFT | 7.5x | vDSP FFT + multi-threaded MAD |
| Gabor | 7.0x | vDSP FFT + multi-threaded MAD |
| FKK (3D) | 9.7x | vDSP 3D FFT + Metal GPU mask |

---

## Phase 2: Next Priority Kernels

### 2.1 FK Filter (2D) - HIGH PRIORITY

**Current:** SciPy FFT + NumPy (CPU only)
**Target:** 10-20x speedup

**Implementation Plan:**
```cpp
// seismic_metal/src/fk_filter_kernel.mm
std::tuple<std::vector<float>, KernelMetrics> fk_filter(
    const float* gather,      // [n_samples, n_traces]
    int n_samples,
    int n_traces,
    float dt,                 // sample interval
    float dx,                 // trace spacing
    float v_min,              // min velocity
    float v_max,              // max velocity
    const std::string& mode   // "reject" or "pass"
);
```

**Key Functions:**
1. `vdsp_rfft2d()` - 2D real-to-complex FFT
2. `build_fk_mask_gpu()` - Metal shader for velocity cone
3. `vdsp_irfft2d()` - 2D inverse FFT

**Metal Shader:**
```metal
kernel void build_fk_velocity_mask(
    device float* mask [[buffer(0)]],
    constant uint& nf [[buffer(1)]],
    constant uint& nk [[buffer(2)]],
    constant float& df [[buffer(3)]],
    constant float& dk [[buffer(4)]],
    constant float& v_min [[buffer(5)]],
    constant float& v_max [[buffer(6)]],
    constant int& filter_mode [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
);
```

---

### 2.2 NMO Correction - HIGH PRIORITY

**Current:** Numba JIT (10-20x over Python)
**Target:** 2-5x over Numba (20-100x over Python)

**Implementation Plan:**
```cpp
// seismic_metal/src/nmo_kernel.mm
std::tuple<std::vector<float>, KernelMetrics> nmo_correction(
    const float* traces,      // [n_samples, n_traces]
    const float* offsets,     // [n_traces] - source-receiver offset
    const float* velocities,  // [n_samples] - interval velocity
    int n_samples,
    int n_traces,
    float dt,
    float stretch_mute        // stretch mute percentage
);
```

**Key Optimizations:**
1. `vDSP_vsqrt` - Batch sqrt for traveltime computation
2. `vDSP_vlint` - Vector linear interpolation
3. Metal kernel for sinc interpolation (8-point)

**Metal Shader (Sinc Interpolation):**
```metal
kernel void sinc_interpolate_batch(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* times [[buffer(2)]],
    constant uint& n_samples [[buffer(3)]],
    constant uint& n_traces [[buffer(4)]],
    constant float& dt [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // 8-point sinc interpolation with Kaiser window
    const int HALF_WIDTH = 4;
    float sinc_sum = 0.0f;
    float weight_sum = 0.0f;

    float t = times[gid.y * n_traces + gid.x];
    int center = int(t / dt);
    float frac = t / dt - float(center);

    for (int k = -HALF_WIDTH; k <= HALF_WIDTH; k++) {
        int idx = center + k;
        if (idx >= 0 && idx < int(n_samples)) {
            float x = frac - float(k);
            float sinc = (abs(x) < 1e-6f) ? 1.0f : sin(M_PI_F * x) / (M_PI_F * x);
            float kaiser = kaiser_window(x, HALF_WIDTH);
            float weight = sinc * kaiser;
            sinc_sum += input[idx * n_traces + gid.x] * weight;
            weight_sum += weight;
        }
    }
    output[gid.y * n_traces + gid.x] = sinc_sum / weight_sum;
}
```

---

### 2.3 Deconvolution - HIGH PRIORITY

**Current:** Numba autocorrelation (5-10x)
**Target:** 2-3x over Numba

**Implementation Plan:**
```cpp
// seismic_metal/src/deconvolution_kernel.mm
std::tuple<std::vector<float>, KernelMetrics> wiener_deconvolution(
    const float* traces,
    int n_samples,
    int n_traces,
    int operator_length,
    float white_noise,        // prewhitening factor
    float mu                  // damping factor
);
```

**Key Optimizations:**
1. `vDSP_conv` - Autocorrelation via correlation
2. Accelerate LAPACK - `sposv_` for Toeplitz solve
3. Multi-threaded batch processing

---

### 2.4 Stockwell/S-Transform - MEDIUM PRIORITY

**Current:** Numba + joblib (O(N²) complexity)
**Target:** 5-10x speedup

**Implementation Plan:**
```cpp
// seismic_metal/src/stockwell_kernel.mm
std::tuple<std::vector<std::complex<float>>, KernelMetrics> stockwell_transform(
    const float* trace,
    int n_samples,
    float dt,
    int min_freq_idx,
    int max_freq_idx
);
```

**Key Optimizations:**
1. Precompute Gaussian windows for all frequencies
2. `vDSP_fft` for frequency-domain multiplication
3. Metal kernel for batch Gaussian convolution

---

### 2.5 AGC (Automatic Gain Control) - MEDIUM PRIORITY

**Current:** SciPy uniform_filter
**Target:** 3-5x speedup

**Implementation Plan:**
```cpp
// seismic_metal/src/agc_kernel.mm
std::tuple<std::vector<float>, KernelMetrics> agc_apply(
    const float* traces,
    int n_samples,
    int n_traces,
    int window_length,        // samples
    float target_rms
);
```

**Metal Shader (Sliding Window RMS):**
```metal
kernel void compute_sliding_rms(
    device const float* input [[buffer(0)]],
    device float* rms [[buffer(1)]],
    constant uint& n_samples [[buffer(2)]],
    constant uint& n_traces [[buffer(3)]],
    constant uint& window_half [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_idx = gid.x;
    uint sample_idx = gid.y;

    float sum_sq = 0.0f;
    int count = 0;

    for (int k = -int(window_half); k <= int(window_half); k++) {
        int idx = int(sample_idx) + k;
        if (idx >= 0 && idx < int(n_samples)) {
            float val = input[idx * n_traces + trace_idx];
            sum_sq += val * val;
            count++;
        }
    }

    rms[sample_idx * n_traces + trace_idx] = sqrt(sum_sq / float(count));
}
```

---

## Phase 3: Advanced Kernels

### 3.1 Bandpass Filter
- `vDSP_biquad` for IIR filtering
- Forward-backward for zero-phase

### 3.2 CDP Stacker
- Metal parallel median (bitonic sort)
- Weighted stack with GPU reduction

### 3.3 SST (Synchrosqueezing Transform)
- Frequency reassignment on GPU
- Phase derivative computation

### 3.4 EMD/EEMD
- Parallel ensemble processing
- Extrema finding optimization

---

## Implementation Priority Matrix

| Kernel | Priority | Effort | Expected Speedup | Dependencies |
|--------|----------|--------|------------------|--------------|
| FK Filter (2D) | P1 | 2 days | 10-20x | vDSP FFT |
| NMO Correction | P1 | 3 days | 20-100x | Metal sinc |
| Deconvolution | P1 | 2 days | 10-30x | LAPACK |
| S-Transform | P2 | 3 days | 5-10x | vDSP FFT |
| AGC | P2 | 1 day | 3-5x | Metal RMS |
| Bandpass | P2 | 1 day | 2-3x | vDSP biquad |
| CDP Stacker | P3 | 2 days | 3-5x | Metal sort |
| SST | P3 | 3 days | 5-10x | Complex |

---

## Architecture: Hybrid vDSP + Metal

The optimal approach continues to be:

1. **vDSP (CPU)** for:
   - FFT operations (lower overhead than GPU for < 4K samples)
   - Convolution/correlation
   - BLAS/LAPACK operations
   - Vector math (sqrt, sin, cos)

2. **Metal GPU** for:
   - Parallel mask building
   - Batch thresholding
   - Large matrix operations (> 1M elements)
   - Independent per-element operations

3. **Multi-threading** for:
   - Trace-level parallelism
   - Embarrassingly parallel operations
   - Memory-bound operations

---

## Files to Create

```
seismic_metal/
├── include/
│   ├── fk_filter_kernel.h
│   ├── nmo_kernel.h
│   ├── deconvolution_kernel.h
│   ├── stockwell_kernel.h
│   └── agc_kernel.h
├── src/
│   ├── fk_filter_kernel.mm
│   ├── nmo_kernel.mm
│   ├── deconvolution_kernel.mm
│   ├── stockwell_kernel.mm
│   └── agc_kernel.mm
├── shaders/
│   ├── fk_velocity_mask.metal
│   ├── sinc_interpolate.metal
│   ├── sliding_rms.metal
│   └── gaussian_conv.metal
└── python/
    └── __init__.py  # Add new kernel exports
```

---

## Performance Targets Summary

| Algorithm | Current (Python) | With Metal C++ | Target Speedup |
|-----------|------------------|----------------|----------------|
| FK Filter | ~50 ms | ~3 ms | 15x |
| NMO | ~100 ms (Numba) | ~5 ms | 20x |
| Deconvolution | ~200 ms (Numba) | ~20 ms | 10x |
| S-Transform | ~500 ms | ~50 ms | 10x |
| AGC | ~30 ms | ~5 ms | 6x |
| Bandpass | ~20 ms | ~8 ms | 2.5x |

**Combined with existing kernels, overall seismic processing pipeline speedup: 10-50x**
