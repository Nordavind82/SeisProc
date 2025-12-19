# Surface-Consistent Seismic Processing Application Architecture

## Executive Summary

This document describes the architecture for a **Surface-Consistent Deconvolution and Amplitude Correction** application, designed to leverage the existing SeisProc framework with MLX C++ Metal kernels for Apple Silicon GPU acceleration.

**Key Features:**
- Surface-consistent decomposition into source, receiver, offset, and CMP factors
- Robust L1/L2 norm solvers for ill-conditioned systems
- Modern sparse inversion methods for improved resolution
- Interactive QC with maps and seismic displays
- "Seismic Equalizer" for manual spectral shaping
- Out-of-core processing for datasets exceeding memory
- GPU-accelerated spectral estimation and decomposition

---

## 1. Mathematical Foundations

### 1.1 Classical Surface-Consistent Model

The surface-consistent assumption states that the recorded seismic trace can be decomposed into multiplicative (or convolutional in log domain) components:

```
Recorded(t) = Source(s) * Receiver(r) * Offset(x) * CMP(c) * Earth(t)
```

In the **log-spectral domain** (for amplitude) or **cepstral domain** (for wavelets):

```
log|S(f)| = log|A_s(f)| + log|A_r(f)| + log|A_x(f)| + log|A_c(f)| + log|E(f)|
```

Where:
- `A_s(f)` = Source amplitude spectrum (varies by source location)
- `A_r(f)` = Receiver amplitude spectrum (varies by receiver location)
- `A_x(f)` = Offset-dependent amplitude (geometrical spreading, attenuation)
- `A_c(f)` = CMP-dependent amplitude (reflectivity variations)
- `E(f)` = Earth response (the signal we want to preserve)

### 1.2 Decomposition as Linear System

For each trace `i` with source index `s[i]`, receiver index `r[i]`, offset bin `x[i]`, and CMP index `c[i]`:

```
d_i = m_s[i] + m_r[i] + m_x[i] + m_c[i]
```

Where `d_i` is the log-amplitude (or spectral attribute) at trace `i`, and `m_*` are the unknown factor values.

This forms a **sparse linear system**:

```
d = G * m

Where:
- d: [N_traces × N_frequencies] observed log-amplitudes
- G: [N_traces × (N_sources + N_receivers + N_offsets + N_cmps)] sparse design matrix
- m: Unknown surface-consistent factors
```

### 1.3 Classical Least-Squares Solution

The standard approach minimizes the L2 norm:

```
min_m ||d - G*m||_2^2
```

Solved via **Conjugate Gradient** for the normal equations:

```
G^T * G * m = G^T * d
```

**Limitations:**
- Sensitive to outliers (dead traces, cultural noise)
- Non-unique solutions due to null space (need constraints)
- Assumes Gaussian noise distribution

### 1.4 Modern Robust Methods

#### 1.4.1 L1 Norm (Robust to Outliers)

```
min_m ||d - G*m||_1
```

Solved via **Iteratively Reweighted Least Squares (IRLS)**:

```python
for iteration in range(max_iter):
    residuals = d - G @ m
    weights = 1.0 / (np.abs(residuals) + epsilon)
    W = diag(weights)
    m = solve(G.T @ W @ G, G.T @ W @ d)
```

**Advantages:**
- Robust to outliers and erratic data
- Better handling of dead/noisy traces
- Reference: [Dai 2018 - Seismic deconvolution with erratic data](https://onlinelibrary.wiley.com/doi/full/10.1111/1365-2478.12689)

#### 1.4.2 L1/L2 Hybrid (RSCD - Robust Surface Consistent Deconvolution)

```
min_m  lambda * ||d - G*m||_1 + (1-lambda) * ||d - G*m||_2^2
```

- `lambda = 1`: Pure L1 (most robust)
- `lambda = 0`: Pure L2 (classical)
- `lambda = 0.5-0.8`: Balanced (recommended)

Reference: [Robust Surface-Consistent Deconvolution](https://www.researchgate.net/publication/301434080_Robust_Surface-Consistent_Deconvolution_Creating_Inversion_Ready_Land_Data)

#### 1.4.3 Sparse Regularization with L1 Model Norm

For improved resolution and sparse solutions:

```
min_m ||d - G*m||_2^2 + alpha * ||L*m||_1
```

Where `L` is a roughness operator (first/second derivative).

Solved via **FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)**:

```python
# FISTA for sparse surface-consistent inversion
def fista_solve(G, d, alpha, L, max_iter=100):
    m = np.zeros(G.shape[1])
    y = m.copy()
    t = 1.0
    lipschitz = np.linalg.norm(G.T @ G, 2)

    for k in range(max_iter):
        grad = G.T @ (G @ y - d)
        z = y - (1/lipschitz) * grad

        # Soft thresholding (proximal operator for L1)
        m_new = soft_threshold(L @ z, alpha/lipschitz)
        m_new = L_inv @ m_new  # Back-transform

        # FISTA momentum update
        t_new = (1 + np.sqrt(1 + 4*t*t)) / 2
        y = m_new + ((t-1)/t_new) * (m_new - m)
        m, t = m_new, t_new

    return m
```

Reference: [Sparse-spike seismic inversion (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11358434/)

#### 1.4.4 Phase-Consistent Decomposition

For deconvolution (not just amplitude), we need phase estimation:

**Minimum Phase Assumption:**
```python
# Hilbert transform relationship for minimum phase
phase = -hilbert(log_amplitude)
```

**Mixed Phase via Cepstral Analysis:**
```python
# Separate minimum and maximum phase components
cepstrum = ifft(log(abs(fft(trace))))
min_phase_cepstrum = cepstrum.copy()
min_phase_cepstrum[n//2:] = 0  # Causal part only
min_phase_cepstrum[0] /= 2
```

**Modern: Phase retrieval via optimization:**
```
min_phase ||fft(wavelet) - target_amplitude * exp(i*phase)||_2^2
         + beta * TV(phase)  # Total variation regularization
```

### 1.5 Spectral Estimation Methods

#### 1.5.1 Windowed FFT (Classical)
```python
def estimate_spectrum(trace, window_start, window_end, taper='hann'):
    window = trace[window_start:window_end]
    window *= get_taper(len(window), taper)
    spectrum = np.fft.rfft(window)
    return np.abs(spectrum), np.angle(spectrum)
```

#### 1.5.2 Multitaper Spectral Estimation (Improved Resolution)
```python
from scipy.signal import windows

def multitaper_spectrum(trace, window, n_tapers=5, bandwidth=4.0):
    """
    Thomson's multitaper method for improved spectral estimation.
    Reduces variance while maintaining resolution.
    """
    tapers = windows.dpss(len(window), bandwidth, n_tapers)
    spectra = []
    for taper in tapers:
        windowed = trace[window] * taper
        spectra.append(np.abs(np.fft.rfft(windowed))**2)
    return np.mean(spectra, axis=0)
```

#### 1.5.3 Wavelet-Based Spectral Decomposition (Time-Frequency)
```python
def cwt_spectrum(trace, freqs, sample_rate, wavelet='morlet'):
    """
    Continuous wavelet transform for time-frequency analysis.
    Better for non-stationary signals.
    """
    scales = pywt.frequency2scale(wavelet, freqs / sample_rate)
    coeffs, _ = pywt.cwt(trace, scales, wavelet, 1/sample_rate)
    return np.abs(coeffs)
```

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Surface-Consistent Processing App                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │   Data I/O   │   │   Spectral   │   │ Decomposition│   │  Filter  │ │
│  │   Layer      │───│  Estimation  │───│   Engine     │───│ Design   │ │
│  │ (Zarr/PQ)    │   │   (GPU)      │   │  (GPU/CPU)   │   │  & QC    │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────┘ │
│         │                  │                  │                │        │
│         │                  │                  │                │        │
│  ┌──────┴──────────────────┴──────────────────┴────────────────┴──────┐ │
│  │                        Metal C++ Kernel Layer                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │ │
│  │  │ FFT Batch   │  │ Sparse Mat  │  │  IRLS       │  │ Filter    │  │ │
│  │  │ Kernels     │  │ Operations  │  │  Solver     │  │ Apply     │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                          PyQt6 UI Layer                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐ │ │
│  │  │ Workflow │  │ Factor   │  │ Seismic  │  │ Equalizer│  │ Export│ │ │
│  │  │ Wizard   │  │ QC Maps  │  │ Viewer   │  │ Panel    │  │ Panel │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └───────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
SeisProc/
├── processors/
│   └── surface_consistent/
│       ├── __init__.py
│       ├── spectral_estimator.py      # Windowed FFT, multitaper, CWT
│       ├── decomposition_engine.py    # L2/L1/IRLS solvers
│       ├── filter_designer.py         # Decon filter computation
│       ├── amplitude_corrector.py     # Scalar correction application
│       ├── phase_estimator.py         # Min-phase, mixed-phase
│       └── config.py                  # SCConfig dataclass
│
├── views/
│   └── surface_consistent/
│       ├── __init__.py
│       ├── sc_main_panel.py           # Main workflow panel
│       ├── spectral_window_picker.py  # Interactive window selection
│       ├── factor_qc_maps.py          # Source/receiver/offset/CMP maps
│       ├── seismic_equalizer.py       # Manual spectral shaping
│       ├── filter_qc_viewer.py        # Before/after/diff comparison
│       └── production_dialog.py       # Batch processing setup
│
├── seismic_metal/
│   └── shaders/
│       ├── sc_fft_batch.metal         # Batch FFT for all traces
│       ├── sc_sparse_matvec.metal     # Sparse G*m and G^T*d
│       ├── sc_irls_weights.metal      # IRLS weight computation
│       ├── sc_soft_threshold.metal    # FISTA proximal operator
│       └── sc_filter_apply.metal      # Batch convolution
│
└── models/
    └── surface_consistent/
        ├── __init__.py
        ├── factor_model.py            # Source/Receiver/Offset/CMP factors
        ├── spectral_data.py           # Per-trace spectra storage
        └── design_matrix.py           # Sparse G matrix representation
```

---

## 3. Data Flow Architecture

### 3.1 Complete Workflow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: SPECTRAL ESTIMATION                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Zarr Dataset                                                      │
│  (traces.zarr + headers.parquet)                                         │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────┐                            │
│  │ User Selects Estimation Window          │                            │
│  │ - Time gate (ms): [t_start, t_end]      │                            │
│  │ - Taper type: Hann/Kaiser/Tukey         │                            │
│  │ - Method: FFT/Multitaper/CWT            │                            │
│  │ - Frequency range: [f_min, f_max]       │                            │
│  └─────────────────────────────────────────┘                            │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────┐                            │
│  │ GPU Batch Spectral Estimation           │◄──── Metal Kernel          │
│  │ - Process 10,000+ traces in parallel    │                            │
│  │ - Output: [N_traces × N_freqs] complex  │                            │
│  └─────────────────────────────────────────┘                            │
│         │                                                                │
│         ▼                                                                │
│  Output: spectral_data.zarr                                              │
│  - amplitude[n_traces, n_freqs]                                          │
│  - phase[n_traces, n_freqs] (optional)                                   │
│  - frequencies[n_freqs]                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: SURFACE-CONSISTENT DECOMPOSITION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  spectral_data.zarr + headers.parquet                                    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────┐                            │
│  │ User Selects Decomposition Factors      │                            │
│  │ ☑ Source (header: FFID/SOURCE)          │                            │
│  │ ☑ Receiver (header: CHAN/RECEIVER)      │                            │
│  │ ☐ Offset (header: OFFSET, bins: 50m)    │                            │
│  │ ☐ CMP (header: CDP)                     │                            │
│  │ Solver: L2 / L1 / IRLS / FISTA          │                            │
│  └─────────────────────────────────────────┘                            │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────┐                            │
│  │ Build Sparse Design Matrix G            │                            │
│  │ - COO format for construction           │                            │
│  │ - CSR format for G*m                    │                            │
│  │ - CSC format for G^T*d                  │                            │
│  │ - Size: [N_traces × N_factors]          │                            │
│  └─────────────────────────────────────────┘                            │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────┐                            │
│  │ Iterative Solver (per frequency)        │◄──── Metal/Accelerate      │
│  │ - Conjugate Gradient (L2)               │                            │
│  │ - IRLS (L1/L2 hybrid)                   │                            │
│  │ - FISTA (sparse regularization)         │                            │
│  │ - Convergence monitoring + progress     │                            │
│  └─────────────────────────────────────────┘                            │
│         │                                                                │
│         ▼                                                                │
│  Output: factors.zarr                                                    │
│  - source_factors[n_sources, n_freqs]                                    │
│  - receiver_factors[n_receivers, n_freqs]                                │
│  - offset_factors[n_offset_bins, n_freqs]                                │
│  - cmp_factors[n_cmps, n_freqs] (optional)                               │
│  - residuals[n_traces, n_freqs]                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: FACTOR QC IN MAPS                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Factor QC Panel                                │  │
│  ├───────────────────┬───────────────────┬───────────────────────────┤  │
│  │                   │                   │                           │  │
│  │   Source Map      │   Receiver Map    │   Offset Curve            │  │
│  │   (color=dB)      │   (color=dB)      │   (amplitude vs offset)   │  │
│  │                   │                   │                           │  │
│  │   ● ● ● ● ●      │   ■ ■ ■ ■ ■ ■    │      ╱                    │  │
│  │   ● ● ● ● ●      │   ■ ■ ■ ■ ■ ■    │    ╱                      │  │
│  │   ● ● ● ● ●      │   ■ ■ ■ ■ ■ ■    │  ╱────────                 │  │
│  │                   │                   │                           │  │
│  ├───────────────────┴───────────────────┴───────────────────────────┤  │
│  │                                                                    │  │
│  │   Frequency Slider: [████████░░░░░░░░░░░░] 35 Hz                   │  │
│  │                                                                    │  │
│  │   Statistics:                                                      │  │
│  │   - Source std:   2.3 dB  (target < 3 dB)                          │  │
│  │   - Receiver std: 1.8 dB  (target < 3 dB)                          │  │
│  │   - Residual RMS: 0.5 dB                                           │  │
│  │                                                                    │  │
│  │   Actions:                                                         │  │
│  │   [Edit Outliers] [Smooth Factors] [Re-solve with constraints]    │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: FILTER DESIGN & QC                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   Filter Design Panel                              │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Mode: ○ Amplitude Only  ● Deconvolution (amplitude + phase)      │  │
│  │                                                                    │  │
│  │  Target Spectrum:                                                  │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │     ╱╲                                                      │   │  │
│  │  │    ╱  ╲    Flat (whitening)                                 │   │  │
│  │  │   ╱    ╲   ──────────────                                   │   │  │
│  │  │  ╱      ╲____                                               │   │  │
│  │  │ ╱                                                           │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  │  [Load Target] [Draw Target] [Bandpass Target]                    │  │
│  │                                                                    │  │
│  │  Phase: ○ Minimum Phase  ○ Zero Phase  ● Mixed Phase (α=0.3)     │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │               Seismic Equalizer (Manual Shaping)                   │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Frequency Bands (dB adjustment):                                  │  │
│  │                                                                    │  │
│  │   5Hz   15Hz   25Hz   35Hz   50Hz   70Hz   100Hz  150Hz           │  │
│  │    │     │      │      │      │      │      │      │              │  │
│  │    ▲     │      │      ▼      │      │      ▲      │    +6 dB     │  │
│  │    █     │      │      █      │      │      █      │              │  │
│  │    █     █      █      █      █      █      █      █    0 dB      │  │
│  │    █     █      █      █      █      █      █      █              │  │
│  │    █     █      █      █      █      █      █      █    -6 dB     │  │
│  │                                                                    │  │
│  │  Presets: [Flat] [Bass Boost] [High Cut] [Custom...]              │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              QC Viewer (Selected Inlines)                          │  │
│  ├────────────────────┬────────────────────┬─────────────────────────┤  │
│  │                    │                    │                         │  │
│  │   INPUT            │   PROCESSED        │   DIFFERENCE            │  │
│  │                    │                    │                         │  │
│  │   ▓▓▓░░▓▓▓░░      │   ▓▓▓▓▓▓▓▓▓▓      │   ░░░░░░░░░░            │  │
│  │   ▓░▓░░▓░▓░░      │   ▓▓▓▓▓▓▓▓▓▓      │   ░░░░░░░░░░            │  │
│  │   ░░▓▓▓░░▓▓▓      │   ▓▓▓▓▓▓▓▓▓▓      │   ░░░░░░░░░░            │  │
│  │                    │                    │                         │  │
│  ├────────────────────┴────────────────────┴─────────────────────────┤  │
│  │  Inline Selector: [◄] IL 1000 [►]   [Add to QC Set] [Remove]      │  │
│  │                                                                    │  │
│  │  Spectra Comparison:                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │   ── Input    ── Processed    ── Target                     │  │  │
│  │  │   ╱╲                                                        │  │  │
│  │  │  ╱  ╲      ════════════════                                 │  │  │
│  │  │ ╱    ╲____                                                  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: PRODUCTION APPLICATION                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   Production Dialog                                │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Input:  /data/survey/raw_traces.zarr                [Browse]     │  │
│  │  Output: /data/survey/sc_processed.zarr              [Browse]     │  │
│  │                                                                    │  │
│  │  Apply:                                                            │  │
│  │  ☑ Surface-consistent amplitude correction                        │  │
│  │  ☑ Surface-consistent deconvolution                               │  │
│  │  ☑ Seismic equalizer adjustments                                  │  │
│  │                                                                    │  │
│  │  Processing:                                                       │  │
│  │  - Chunk size: [10000] traces                                      │  │
│  │  - Workers: [4] (Metal GPU)                                        │  │
│  │  - Memory limit: [8] GB                                            │  │
│  │                                                                    │  │
│  │  Progress:                                                         │  │
│  │  [████████████████████░░░░░░░░░░░░░░░░░░] 45% - ETA: 12:34         │  │
│  │  Processing chunk 45/100 (450,000 / 1,000,000 traces)              │  │
│  │                                                                    │  │
│  │  [Start Processing] [Pause] [Cancel]                               │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Components Design

### 4.1 Spectral Estimator

```python
# processors/surface_consistent/spectral_estimator.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np

class SpectralMethod(Enum):
    FFT = "fft"
    MULTITAPER = "multitaper"
    CWT = "cwt"

@dataclass
class SpectralConfig:
    """Configuration for spectral estimation."""
    method: SpectralMethod = SpectralMethod.FFT
    window_start_ms: float = 200.0
    window_end_ms: float = 1500.0
    taper_type: str = 'hann'  # hann, kaiser, tukey, cosine
    taper_percent: float = 10.0
    freq_min_hz: float = 5.0
    freq_max_hz: float = 120.0

    # Multitaper specific
    n_tapers: int = 5
    bandwidth: float = 4.0

    # CWT specific
    wavelet: str = 'morlet'
    n_voices: int = 16

class SpectralEstimator:
    """
    GPU-accelerated spectral estimation for surface-consistent processing.

    Supports:
    - Windowed FFT (fastest, standard)
    - Multitaper (lower variance, better for noisy data)
    - CWT (time-frequency, best for non-stationary)
    """

    def __init__(self, config: SpectralConfig, backend: str = 'auto'):
        self.config = config
        self.backend = backend
        self._metal_module = None

    def estimate_all_traces(
        self,
        traces: np.ndarray,
        sample_rate_ms: float,
        progress_callback: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate spectra for all traces.

        Args:
            traces: [n_samples, n_traces] seismic data
            sample_rate_ms: Sample interval in ms
            progress_callback: Progress reporting function

        Returns:
            Tuple of (amplitudes, phases, frequencies)
            - amplitudes: [n_traces, n_freqs] log-amplitudes (dB)
            - phases: [n_traces, n_freqs] phases (radians)
            - frequencies: [n_freqs] frequency axis (Hz)
        """
        n_samples, n_traces = traces.shape

        # Convert window to samples
        win_start = int(self.config.window_start_ms / sample_rate_ms)
        win_end = int(self.config.window_end_ms / sample_rate_ms)
        win_start = max(0, min(win_start, n_samples - 1))
        win_end = max(win_start + 1, min(win_end, n_samples))

        window_length = win_end - win_start

        # Compute frequency axis
        sample_rate_hz = 1000.0 / sample_rate_ms
        freqs = np.fft.rfftfreq(window_length, 1.0 / sample_rate_hz)

        # Filter to requested frequency range
        freq_mask = (freqs >= self.config.freq_min_hz) & (freqs <= self.config.freq_max_hz)
        freqs = freqs[freq_mask]
        n_freqs = len(freqs)

        # Try GPU backend
        if self._try_metal_estimation(traces, win_start, win_end, freq_mask):
            return self._metal_result

        # CPU fallback with chunking
        amplitudes = np.zeros((n_traces, n_freqs), dtype=np.float32)
        phases = np.zeros((n_traces, n_freqs), dtype=np.float32)

        # Create taper
        taper = self._create_taper(window_length)

        chunk_size = 5000
        for chunk_start in range(0, n_traces, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_traces)

            for i in range(chunk_start, chunk_end):
                window = traces[win_start:win_end, i] * taper
                spectrum = np.fft.rfft(window)
                spectrum = spectrum[freq_mask]

                amplitudes[i] = 20 * np.log10(np.abs(spectrum) + 1e-10)
                phases[i] = np.angle(spectrum)

            if progress_callback:
                progress_callback(chunk_end, n_traces)

        return amplitudes, phases, freqs

    def _create_taper(self, length: int) -> np.ndarray:
        """Create windowing taper."""
        if self.config.taper_type == 'hann':
            return np.hanning(length).astype(np.float32)
        elif self.config.taper_type == 'kaiser':
            return np.kaiser(length, 8.0).astype(np.float32)
        elif self.config.taper_type == 'tukey':
            from scipy.signal.windows import tukey
            return tukey(length, alpha=0.1).astype(np.float32)
        else:
            return np.ones(length, dtype=np.float32)

    def _try_metal_estimation(self, traces, win_start, win_end, freq_mask) -> bool:
        """Try GPU-accelerated spectral estimation."""
        try:
            if self._metal_module is None:
                from seismic_metal import spectral_kernel
                self._metal_module = spectral_kernel

            result = self._metal_module.batch_fft_estimate(
                traces, win_start, win_end, freq_mask,
                self.config.taper_type
            )
            self._metal_result = result
            return True
        except ImportError:
            return False
        except Exception as e:
            import logging
            logging.warning(f"Metal spectral estimation failed: {e}")
            return False
```

### 4.2 Decomposition Engine

```python
# processors/surface_consistent/decomposition_engine.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, lsqr

class SolverType(Enum):
    CG_L2 = "cg_l2"           # Conjugate Gradient (L2 norm)
    IRLS_L1 = "irls_l1"       # Iteratively Reweighted LS (L1 norm)
    IRLS_HYBRID = "irls_hybrid"  # L1/L2 hybrid
    FISTA = "fista"           # Fast IST for sparse regularization

@dataclass
class DecompositionConfig:
    """Configuration for surface-consistent decomposition."""
    # Factor selection
    use_source: bool = True
    use_receiver: bool = True
    use_offset: bool = False
    use_cmp: bool = False

    # Header mappings
    source_header: str = 'FFID'
    receiver_header: str = 'CHAN'
    offset_header: str = 'OFFSET'
    cmp_header: str = 'CDP'

    # Offset binning
    offset_bin_size: float = 50.0  # meters

    # Solver settings
    solver: SolverType = SolverType.CG_L2
    max_iterations: int = 100
    tolerance: float = 1e-6

    # L1/IRLS settings
    l1_weight: float = 0.8  # lambda in L1/L2 hybrid (1.0 = pure L1)
    irls_epsilon: float = 1e-4  # Regularization for weights

    # FISTA settings
    sparsity_weight: float = 0.01  # alpha in L1 regularization

    # Constraints
    reference_source: Optional[int] = None  # Fix this source to 0 dB
    reference_receiver: Optional[int] = None

class DecompositionEngine:
    """
    Surface-consistent decomposition engine.

    Decomposes trace attributes into source, receiver, offset, and CMP factors
    using various optimization methods.
    """

    def __init__(self, config: DecompositionConfig):
        self.config = config
        self._G = None  # Design matrix
        self._factor_indices = {}

    def build_design_matrix(
        self,
        headers: Dict[str, np.ndarray],
        n_traces: int
    ) -> sparse.csr_matrix:
        """
        Build sparse design matrix G.

        Each row corresponds to a trace, each column to a factor.
        G[i,j] = 1 if trace i uses factor j.

        Returns:
            Sparse CSR matrix G of shape [n_traces, n_factors]
        """
        rows = []
        cols = []
        data = []

        col_offset = 0

        # Source factors
        if self.config.use_source:
            sources = headers[self.config.source_header]
            unique_sources = np.unique(sources)
            source_map = {s: i for i, s in enumerate(unique_sources)}

            self._factor_indices['source'] = {
                'start': col_offset,
                'end': col_offset + len(unique_sources),
                'mapping': source_map,
                'values': unique_sources
            }

            for i, s in enumerate(sources):
                rows.append(i)
                cols.append(col_offset + source_map[s])
                data.append(1.0)

            col_offset += len(unique_sources)

        # Receiver factors
        if self.config.use_receiver:
            receivers = headers[self.config.receiver_header]
            unique_receivers = np.unique(receivers)
            receiver_map = {r: i for i, r in enumerate(unique_receivers)}

            self._factor_indices['receiver'] = {
                'start': col_offset,
                'end': col_offset + len(unique_receivers),
                'mapping': receiver_map,
                'values': unique_receivers
            }

            for i, r in enumerate(receivers):
                rows.append(i)
                cols.append(col_offset + receiver_map[r])
                data.append(1.0)

            col_offset += len(unique_receivers)

        # Offset factors (binned)
        if self.config.use_offset:
            offsets = np.abs(headers[self.config.offset_header])
            offset_bins = (offsets / self.config.offset_bin_size).astype(int)
            unique_bins = np.unique(offset_bins)
            bin_map = {b: i for i, b in enumerate(unique_bins)}

            self._factor_indices['offset'] = {
                'start': col_offset,
                'end': col_offset + len(unique_bins),
                'mapping': bin_map,
                'values': unique_bins * self.config.offset_bin_size
            }

            for i, b in enumerate(offset_bins):
                rows.append(i)
                cols.append(col_offset + bin_map[b])
                data.append(1.0)

            col_offset += len(unique_bins)

        # CMP factors
        if self.config.use_cmp:
            cmps = headers[self.config.cmp_header]
            unique_cmps = np.unique(cmps)
            cmp_map = {c: i for i, c in enumerate(unique_cmps)}

            self._factor_indices['cmp'] = {
                'start': col_offset,
                'end': col_offset + len(unique_cmps),
                'mapping': cmp_map,
                'values': unique_cmps
            }

            for i, c in enumerate(cmps):
                rows.append(i)
                cols.append(col_offset + cmp_map[c])
                data.append(1.0)

            col_offset += len(unique_cmps)

        n_factors = col_offset

        self._G = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_traces, n_factors),
            dtype=np.float32
        )

        return self._G

    def decompose(
        self,
        data: np.ndarray,
        freq_idx: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, np.ndarray]:
        """
        Decompose data into surface-consistent factors.

        Args:
            data: [n_traces] or [n_traces, n_freqs] log-amplitude data
            freq_idx: If provided, only decompose this frequency
            progress_callback: Progress reporting function

        Returns:
            Dictionary with factor arrays:
            - 'source': [n_sources] or [n_sources, n_freqs]
            - 'receiver': [n_receivers] or [n_receivers, n_freqs]
            - 'offset': [n_offset_bins] or [n_offset_bins, n_freqs]
            - 'cmp': [n_cmps] or [n_cmps, n_freqs]
            - 'residual': [n_traces] or [n_traces, n_freqs]
        """
        if self._G is None:
            raise ValueError("Must call build_design_matrix() first")

        # Handle single frequency or full decomposition
        if data.ndim == 1:
            return self._decompose_single(data)

        n_traces, n_freqs = data.shape

        # Initialize results
        results = {}
        for name, info in self._factor_indices.items():
            n_factors = info['end'] - info['start']
            results[name] = np.zeros((n_factors, n_freqs), dtype=np.float32)
        results['residual'] = np.zeros((n_traces, n_freqs), dtype=np.float32)

        # Decompose each frequency
        for f in range(n_freqs):
            if freq_idx is not None and f != freq_idx:
                continue

            freq_result = self._decompose_single(data[:, f])

            for name in self._factor_indices:
                results[name][:, f] = freq_result[name]
            results['residual'][:, f] = freq_result['residual']

            if progress_callback:
                progress_callback(f + 1, n_freqs)

        return results

    def _decompose_single(self, d: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose single frequency/attribute."""
        G = self._G

        # Select solver
        if self.config.solver == SolverType.CG_L2:
            m = self._solve_cg_l2(G, d)
        elif self.config.solver == SolverType.IRLS_L1:
            m = self._solve_irls(G, d, lambda_l1=1.0)
        elif self.config.solver == SolverType.IRLS_HYBRID:
            m = self._solve_irls(G, d, lambda_l1=self.config.l1_weight)
        elif self.config.solver == SolverType.FISTA:
            m = self._solve_fista(G, d)
        else:
            raise ValueError(f"Unknown solver: {self.config.solver}")

        # Extract factors
        results = {}
        for name, info in self._factor_indices.items():
            results[name] = m[info['start']:info['end']]

        # Compute residual
        results['residual'] = d - G @ m

        return results

    def _solve_cg_l2(self, G: sparse.csr_matrix, d: np.ndarray) -> np.ndarray:
        """Solve using Conjugate Gradient (L2 norm)."""
        # Normal equations: G^T G m = G^T d
        GtG = G.T @ G
        Gtd = G.T @ d

        # Add small regularization for stability
        GtG = GtG + sparse.eye(GtG.shape[0]) * 1e-6

        m, info = cg(GtG, Gtd, maxiter=self.config.max_iterations,
                     tol=self.config.tolerance)

        return m

    def _solve_irls(
        self,
        G: sparse.csr_matrix,
        d: np.ndarray,
        lambda_l1: float = 0.8
    ) -> np.ndarray:
        """
        Solve using Iteratively Reweighted Least Squares.

        Minimizes: lambda * ||d - Gm||_1 + (1-lambda) * ||d - Gm||_2^2
        """
        n_factors = G.shape[1]
        m = np.zeros(n_factors, dtype=np.float32)
        eps = self.config.irls_epsilon

        for iteration in range(self.config.max_iterations):
            # Compute residuals
            residuals = d - G @ m

            # Compute weights for L1 term
            if lambda_l1 > 0:
                weights_l1 = 1.0 / (np.abs(residuals) + eps)
            else:
                weights_l1 = np.zeros_like(residuals)

            # Combined weights
            weights = lambda_l1 * weights_l1 + (1 - lambda_l1) * np.ones_like(residuals)
            W = sparse.diags(weights)

            # Weighted normal equations
            GtWG = G.T @ W @ G
            GtWd = G.T @ W @ d

            # Add regularization
            GtWG = GtWG + sparse.eye(n_factors) * 1e-6

            # Solve
            m_new, _ = cg(GtWG, GtWd, x0=m, maxiter=50, tol=1e-4)

            # Check convergence
            if np.linalg.norm(m_new - m) < self.config.tolerance * np.linalg.norm(m_new):
                m = m_new
                break

            m = m_new

        return m

    def _solve_fista(self, G: sparse.csr_matrix, d: np.ndarray) -> np.ndarray:
        """
        Solve using FISTA with L1 regularization.

        Minimizes: ||d - Gm||_2^2 + alpha * ||m||_1
        """
        n_factors = G.shape[1]

        # Estimate Lipschitz constant
        GtG = G.T @ G
        L = sparse.linalg.norm(GtG, 'fro')  # Approximate

        m = np.zeros(n_factors, dtype=np.float32)
        y = m.copy()
        t = 1.0
        alpha = self.config.sparsity_weight

        for iteration in range(self.config.max_iterations):
            # Gradient step
            grad = G.T @ (G @ y - d)
            z = y - (1.0 / L) * grad

            # Proximal operator (soft thresholding)
            threshold = alpha / L
            m_new = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

            # FISTA momentum
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = m_new + ((t - 1) / t_new) * (m_new - m)

            # Check convergence
            if np.linalg.norm(m_new - m) < self.config.tolerance * np.linalg.norm(m_new):
                m = m_new
                break

            m = m_new
            t = t_new

        return m

    def get_factor_info(self) -> Dict[str, Dict]:
        """Get information about computed factors."""
        return self._factor_indices.copy()
```

### 4.3 Seismic Equalizer Widget

```python
# views/surface_consistent/seismic_equalizer.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QPushButton, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np

class SeismicEqualizerWidget(QWidget):
    """
    Interactive frequency-band equalizer for manual spectral shaping.

    Allows users to adjust amplitude in multiple frequency bands,
    similar to an audio equalizer but for seismic data.
    """

    # Signal emitted when equalizer settings change
    settings_changed = pyqtSignal(dict)  # {freq_hz: gain_db}

    # Preset equalizer curves
    PRESETS = {
        'Flat': {},
        'Bass Boost': {10: 3, 15: 2, 20: 1},
        'High Boost': {60: 1, 80: 2, 100: 3},
        'Low Cut': {5: -6, 10: -3, 15: -1},
        'High Cut': {80: -1, 100: -3, 120: -6},
        'Notch 50Hz': {45: 0, 50: -12, 55: 0},
        'Broadband Boost': {20: 2, 40: 2, 60: 2, 80: 2},
    }

    def __init__(
        self,
        freq_bands: list = None,
        parent: QWidget = None
    ):
        super().__init__(parent)

        # Default frequency bands (Hz)
        self.freq_bands = freq_bands or [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120]

        # Current gains (dB) per band
        self.gains = {f: 0.0 for f in self.freq_bands}

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_all)
        preset_layout.addWidget(reset_btn)

        layout.addLayout(preset_layout)

        # Equalizer sliders
        eq_group = QGroupBox("Frequency Bands (dB)")
        eq_layout = QGridLayout(eq_group)

        self.sliders = {}
        self.labels = {}

        for col, freq in enumerate(self.freq_bands):
            # Frequency label
            freq_label = QLabel(f"{freq}")
            freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            eq_layout.addWidget(freq_label, 0, col)

            # Vertical slider (-12 to +12 dB)
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-120, 120)  # -12.0 to +12.0 dB (x10)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
            slider.setTickInterval(30)  # 3 dB ticks
            slider.valueChanged.connect(lambda v, f=freq: self._on_slider_changed(f, v))
            eq_layout.addWidget(slider, 1, col, Qt.AlignmentFlag.AlignHCenter)
            self.sliders[freq] = slider

            # Value label
            value_label = QLabel("0.0")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            eq_layout.addWidget(value_label, 2, col)
            self.labels[freq] = value_label

        # Set row stretch for slider row
        eq_layout.setRowStretch(1, 1)

        layout.addWidget(eq_group)

        # dB scale labels
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("+12 dB"))
        scale_layout.addStretch()
        scale_layout.addWidget(QLabel("0 dB"))
        scale_layout.addStretch()
        scale_layout.addWidget(QLabel("-12 dB"))
        layout.addLayout(scale_layout)

    def _on_slider_changed(self, freq: int, value: int):
        """Handle slider value change."""
        gain_db = value / 10.0  # Convert to dB
        self.gains[freq] = gain_db
        self.labels[freq].setText(f"{gain_db:+.1f}")

        # Emit settings changed signal
        self.settings_changed.emit(self.gains.copy())

    def _on_preset_selected(self, preset_name: str):
        """Apply preset equalizer curve."""
        preset = self.PRESETS.get(preset_name, {})

        # Reset all to 0
        for freq in self.freq_bands:
            self.gains[freq] = preset.get(freq, 0.0)
            self.sliders[freq].setValue(int(self.gains[freq] * 10))
            self.labels[freq].setText(f"{self.gains[freq]:+.1f}")

        self.settings_changed.emit(self.gains.copy())

    def _reset_all(self):
        """Reset all bands to 0 dB."""
        for freq in self.freq_bands:
            self.gains[freq] = 0.0
            self.sliders[freq].setValue(0)
            self.labels[freq].setText("0.0")

        self.settings_changed.emit(self.gains.copy())

    def get_filter_response(self, freqs: np.ndarray) -> np.ndarray:
        """
        Get interpolated filter response at arbitrary frequencies.

        Args:
            freqs: Array of frequencies (Hz)

        Returns:
            Array of gain values (linear scale, not dB)
        """
        # Interpolate between equalizer bands
        eq_freqs = np.array(self.freq_bands, dtype=np.float32)
        eq_gains_db = np.array([self.gains[f] for f in self.freq_bands], dtype=np.float32)

        # Interpolate (with extrapolation at edges)
        gains_db = np.interp(freqs, eq_freqs, eq_gains_db)

        # Convert to linear scale
        gains_linear = 10 ** (gains_db / 20.0)

        return gains_linear.astype(np.float32)

    def set_gains(self, gains: dict):
        """Set equalizer gains from dictionary."""
        for freq, gain in gains.items():
            if freq in self.sliders:
                self.gains[freq] = gain
                self.sliders[freq].setValue(int(gain * 10))
                self.labels[freq].setText(f"{gain:+.1f}")

        self.settings_changed.emit(self.gains.copy())
```

### 4.4 Factor QC Maps Widget

```python
# views/surface_consistent/factor_qc_maps.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox, QSplitter, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np
import pyqtgraph as pg

class FactorQCMapsWidget(QWidget):
    """
    Interactive QC maps for surface-consistent factors.

    Displays source, receiver, and offset factors as color-coded maps,
    with frequency selection and statistics.
    """

    # Signal when user selects a factor for editing
    factor_selected = pyqtSignal(str, int)  # factor_type, factor_index

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self.factors = {}  # {name: [n_factors, n_freqs]}
        self.factor_coords = {}  # {name: {'x': [...], 'y': [...]}}
        self.frequencies = None
        self.current_freq_idx = 0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Frequency selector
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency:"))

        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 100)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        freq_layout.addWidget(self.freq_slider)

        self.freq_label = QLabel("-- Hz")
        freq_layout.addWidget(self.freq_label)

        layout.addLayout(freq_layout)

        # Map displays in splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Source map
        self.source_group = QGroupBox("Source Factors")
        source_layout = QVBoxLayout(self.source_group)
        self.source_plot = pg.PlotWidget()
        self.source_plot.setAspectLocked(True)
        self.source_scatter = pg.ScatterPlotItem()
        self.source_plot.addItem(self.source_scatter)
        source_layout.addWidget(self.source_plot)
        self.source_stats = QLabel("Mean: -- dB, Std: -- dB")
        source_layout.addWidget(self.source_stats)
        splitter.addWidget(self.source_group)

        # Receiver map
        self.receiver_group = QGroupBox("Receiver Factors")
        receiver_layout = QVBoxLayout(self.receiver_group)
        self.receiver_plot = pg.PlotWidget()
        self.receiver_plot.setAspectLocked(True)
        self.receiver_scatter = pg.ScatterPlotItem()
        self.receiver_plot.addItem(self.receiver_scatter)
        receiver_layout.addWidget(self.receiver_plot)
        self.receiver_stats = QLabel("Mean: -- dB, Std: -- dB")
        receiver_layout.addWidget(self.receiver_stats)
        splitter.addWidget(self.receiver_group)

        # Offset curve
        self.offset_group = QGroupBox("Offset Response")
        offset_layout = QVBoxLayout(self.offset_group)
        self.offset_plot = pg.PlotWidget()
        self.offset_plot.setLabel('bottom', 'Offset', 'm')
        self.offset_plot.setLabel('left', 'Amplitude', 'dB')
        self.offset_curve = self.offset_plot.plot(pen='y', symbol='o')
        offset_layout.addWidget(self.offset_plot)
        self.offset_stats = QLabel("Trend: -- dB/km")
        offset_layout.addWidget(self.offset_stats)
        splitter.addWidget(self.offset_group)

        layout.addWidget(splitter)

        # Overall statistics
        stats_layout = QHBoxLayout()
        self.residual_stats = QLabel("Residual RMS: -- dB")
        stats_layout.addWidget(self.residual_stats)
        stats_layout.addStretch()

        self.quality_indicator = QLabel("Quality: --")
        stats_layout.addWidget(self.quality_indicator)

        layout.addLayout(stats_layout)

    def set_factors(
        self,
        factors: dict,
        factor_info: dict,
        frequencies: np.ndarray,
        source_coords: tuple = None,
        receiver_coords: tuple = None
    ):
        """
        Set factor data for display.

        Args:
            factors: {'source': [n_src, n_freq], 'receiver': [...], 'offset': [...]}
            factor_info: Factor index information from decomposition
            frequencies: Frequency axis (Hz)
            source_coords: (x, y) arrays for source positions
            receiver_coords: (x, y) arrays for receiver positions
        """
        self.factors = factors
        self.factor_info = factor_info
        self.frequencies = frequencies

        # Store coordinates
        if source_coords:
            self.factor_coords['source'] = {'x': source_coords[0], 'y': source_coords[1]}
        if receiver_coords:
            self.factor_coords['receiver'] = {'x': receiver_coords[0], 'y': receiver_coords[1]}

        # Update frequency slider
        self.freq_slider.setRange(0, len(frequencies) - 1)
        self.freq_slider.setValue(len(frequencies) // 4)  # Start at ~25% of bandwidth

        self._update_display()

    def _on_freq_changed(self, idx: int):
        """Handle frequency slider change."""
        self.current_freq_idx = idx
        if self.frequencies is not None:
            self.freq_label.setText(f"{self.frequencies[idx]:.1f} Hz")
        self._update_display()

    def _update_display(self):
        """Update all map displays for current frequency."""
        if not self.factors:
            return

        freq_idx = self.current_freq_idx

        # Update source map
        if 'source' in self.factors and 'source' in self.factor_coords:
            values = self.factors['source'][:, freq_idx]
            coords = self.factor_coords['source']

            # Normalize colors
            vmin, vmax = np.percentile(values, [5, 95])
            colors = self._values_to_colors(values, vmin, vmax)

            self.source_scatter.setData(
                x=coords['x'], y=coords['y'],
                brush=colors, size=10
            )

            self.source_stats.setText(
                f"Mean: {np.mean(values):.1f} dB, Std: {np.std(values):.1f} dB"
            )

        # Update receiver map
        if 'receiver' in self.factors and 'receiver' in self.factor_coords:
            values = self.factors['receiver'][:, freq_idx]
            coords = self.factor_coords['receiver']

            vmin, vmax = np.percentile(values, [5, 95])
            colors = self._values_to_colors(values, vmin, vmax)

            self.receiver_scatter.setData(
                x=coords['x'], y=coords['y'],
                brush=colors, size=8
            )

            self.receiver_stats.setText(
                f"Mean: {np.mean(values):.1f} dB, Std: {np.std(values):.1f} dB"
            )

        # Update offset curve
        if 'offset' in self.factors and 'offset' in self.factor_info:
            values = self.factors['offset'][:, freq_idx]
            offsets = self.factor_info['offset']['values']

            self.offset_curve.setData(x=offsets, y=values)

            # Compute trend
            if len(offsets) > 1:
                slope, _ = np.polyfit(offsets, values, 1)
                trend_db_km = slope * 1000
                self.offset_stats.setText(f"Trend: {trend_db_km:.2f} dB/km")

        # Update residual stats
        if 'residual' in self.factors:
            residual = self.factors['residual'][:, freq_idx]
            rms = np.sqrt(np.mean(residual ** 2))
            self.residual_stats.setText(f"Residual RMS: {rms:.2f} dB")

            # Quality indicator
            if rms < 1.0:
                quality = "Excellent"
                color = "green"
            elif rms < 2.0:
                quality = "Good"
                color = "blue"
            elif rms < 3.0:
                quality = "Fair"
                color = "orange"
            else:
                quality = "Poor"
                color = "red"

            self.quality_indicator.setText(f"Quality: {quality}")
            self.quality_indicator.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _values_to_colors(self, values: np.ndarray, vmin: float, vmax: float):
        """Convert values to color brushes using colormap."""
        from PyQt6.QtGui import QColor

        # Normalize to 0-1
        norm = (values - vmin) / (vmax - vmin + 1e-10)
        norm = np.clip(norm, 0, 1)

        # Apply colormap (RdBu diverging)
        colors = []
        for v in norm:
            if v < 0.5:
                # Blue to white
                r = int(255 * (2 * v))
                g = int(255 * (2 * v))
                b = 255
            else:
                # White to red
                r = 255
                g = int(255 * (2 * (1 - v)))
                b = int(255 * (2 * (1 - v)))
            colors.append(pg.mkBrush(QColor(r, g, b)))

        return colors
```

---

## 5. Metal C++ Kernel Design

### 5.1 Batch FFT Kernel

```metal
// seismic_metal/shaders/sc_fft_batch.metal

#include <metal_stdlib>
using namespace metal;

// Cooley-Tukey radix-2 DIT FFT for power-of-2 lengths
// Processes multiple traces in parallel

struct FFTParams {
    uint n_samples;      // Window length (power of 2)
    uint n_traces;       // Number of traces to process
    uint win_start;      // Window start sample
    uint win_end;        // Window end sample
    uint n_freqs;        // Output frequencies
};

// Complex number operations
struct Complex {
    float real;
    float imag;

    Complex operator+(Complex other) const {
        return {real + other.real, imag + other.imag};
    }

    Complex operator-(Complex other) const {
        return {real - other.real, imag - other.imag};
    }

    Complex operator*(Complex other) const {
        return {
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        };
    }

    float magnitude() const {
        return sqrt(real * real + imag * imag);
    }

    float phase() const {
        return atan2(imag, real);
    }
};

// Hann window function
float hann_window(uint idx, uint length) {
    return 0.5f * (1.0f - cos(2.0f * M_PI_F * float(idx) / float(length - 1)));
}

// Bit-reversal permutation index
uint bit_reverse(uint x, uint log2n) {
    uint result = 0;
    for (uint i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

kernel void batch_fft_estimate(
    device const float* traces [[buffer(0)]],        // [n_samples, n_traces]
    device float* amplitudes [[buffer(1)]],          // [n_traces, n_freqs]
    device float* phases [[buffer(2)]],              // [n_traces, n_freqs]
    constant FFTParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint trace_idx = gid.x;
    if (trace_idx >= params.n_traces) return;

    uint window_length = params.win_end - params.win_start;
    uint fft_length = params.n_samples;  // Padded to power of 2
    uint log2n = 0;
    for (uint n = fft_length; n > 1; n >>= 1) log2n++;

    // Allocate threadgroup memory for FFT
    threadgroup Complex fft_buffer[2048];  // Max FFT size

    // Load and window the trace
    for (uint i = tid.x; i < fft_length; i += 32) {
        if (i < window_length) {
            uint src_idx = (params.win_start + i) * params.n_traces + trace_idx;
            float windowed = traces[src_idx] * hann_window(i, window_length);
            fft_buffer[bit_reverse(i, log2n)] = {windowed, 0.0f};
        } else {
            fft_buffer[bit_reverse(i, log2n)] = {0.0f, 0.0f};
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey iterative FFT
    for (uint stage = 1; stage <= log2n; stage++) {
        uint m = 1 << stage;
        uint m2 = m >> 1;

        float angle = -M_PI_F / float(m2);
        Complex w_m = {cos(angle), sin(angle)};

        for (uint k = tid.x; k < fft_length; k += 32) {
            if ((k % m) < m2) {
                uint j = k + m2;
                Complex w = {1.0f, 0.0f};

                for (uint i = 0; i < (k % m2); i++) {
                    w = w * w_m;
                }

                Complex t = w * fft_buffer[j];
                Complex u = fft_buffer[k];

                fft_buffer[k] = u + t;
                fft_buffer[j] = u - t;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Extract positive frequencies and compute amplitude/phase
    for (uint f = tid.x; f < params.n_freqs; f += 32) {
        Complex spectrum = fft_buffer[f];

        float amp = spectrum.magnitude();
        float ph = spectrum.phase();

        // Convert amplitude to dB
        amp = 20.0f * log10(amp + 1e-10f);

        uint out_idx = trace_idx * params.n_freqs + f;
        amplitudes[out_idx] = amp;
        phases[out_idx] = ph;
    }
}
```

### 5.2 IRLS Solver Kernel

```metal
// seismic_metal/shaders/sc_irls_weights.metal

#include <metal_stdlib>
using namespace metal;

// Compute IRLS weights for L1 norm minimization
// w_i = 1 / (|r_i| + epsilon)

struct IRLSParams {
    uint n_elements;
    float epsilon;
    float lambda_l1;  // Weight for L1 vs L2 (0 = L2, 1 = L1)
};

kernel void compute_irls_weights(
    device const float* residuals [[buffer(0)]],  // [n_elements]
    device float* weights [[buffer(1)]],          // [n_elements]
    constant IRLSParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;

    float r = residuals[gid];
    float abs_r = abs(r);

    // L1 weight
    float w_l1 = 1.0f / (abs_r + params.epsilon);

    // L2 weight (constant)
    float w_l2 = 1.0f;

    // Hybrid weight
    weights[gid] = params.lambda_l1 * w_l1 + (1.0f - params.lambda_l1) * w_l2;
}

// Soft thresholding for FISTA
kernel void soft_threshold(
    device float* x [[buffer(0)]],
    constant float& threshold [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    constant uint& n_elements [[buffer(2)]]
) {
    if (gid >= n_elements) return;

    float val = x[gid];
    float sign_val = (val > 0) ? 1.0f : ((val < 0) ? -1.0f : 0.0f);
    float abs_val = abs(val);

    x[gid] = sign_val * max(abs_val - threshold, 0.0f);
}
```

### 5.3 Sparse Matrix-Vector Kernel

```metal
// seismic_metal/shaders/sc_sparse_matvec.metal

#include <metal_stdlib>
using namespace metal;

// CSR sparse matrix-vector multiplication: y = A * x
// Used for G * m in surface-consistent decomposition

struct CSRParams {
    uint n_rows;
    uint n_cols;
    uint nnz;  // Number of non-zeros
};

kernel void sparse_matvec_csr(
    device const float* values [[buffer(0)]],      // [nnz] non-zero values
    device const uint* col_indices [[buffer(1)]],  // [nnz] column indices
    device const uint* row_ptrs [[buffer(2)]],     // [n_rows + 1] row pointers
    device const float* x [[buffer(3)]],           // [n_cols] input vector
    device float* y [[buffer(4)]],                 // [n_rows] output vector
    constant CSRParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_rows) return;

    uint row_start = row_ptrs[gid];
    uint row_end = row_ptrs[gid + 1];

    float sum = 0.0f;
    for (uint i = row_start; i < row_end; i++) {
        uint col = col_indices[i];
        sum += values[i] * x[col];
    }

    y[gid] = sum;
}

// Transpose CSR matrix-vector: y = A^T * x
// Used for G^T * d in surface-consistent decomposition
// Note: CSR^T is effectively CSC

kernel void sparse_matvec_csr_transpose(
    device const float* values [[buffer(0)]],
    device const uint* col_indices [[buffer(1)]],
    device const uint* row_ptrs [[buffer(2)]],
    device const float* x [[buffer(3)]],           // [n_rows] input
    device atomic_float* y [[buffer(4)]],          // [n_cols] output (atomic for reduction)
    constant CSRParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_rows) return;

    uint row_start = row_ptrs[gid];
    uint row_end = row_ptrs[gid + 1];

    float x_val = x[gid];

    for (uint i = row_start; i < row_end; i++) {
        uint col = col_indices[i];
        float contribution = values[i] * x_val;
        atomic_fetch_add_explicit(&y[col], contribution, memory_order_relaxed);
    }
}
```

---

## 6. Out-of-Core Processing Strategy

### 6.1 Chunked Processing Pipeline

```python
# processors/surface_consistent/chunked_processor.py

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import zarr

@dataclass
class ChunkConfig:
    """Configuration for chunked processing."""
    chunk_size: int = 10000  # Traces per chunk
    memory_limit_gb: float = 8.0
    n_workers: int = 4
    overlap_traces: int = 0  # For spatial operations

class ChunkedSCProcessor:
    """
    Chunked surface-consistent processing for datasets larger than memory.

    Strategy:
    1. Spectral estimation: Process in trace chunks, store to Zarr
    2. Decomposition: Build full design matrix (sparse), solve iteratively
    3. Application: Process in chunks, apply pre-computed factors
    """

    def __init__(self, config: ChunkConfig):
        self.config = config

    def process_spectral_estimation(
        self,
        input_zarr: zarr.Array,
        output_zarr: zarr.Array,
        estimator,
        sample_rate_ms: float,
        progress_callback: Optional[Callable] = None
    ):
        """
        Chunked spectral estimation.

        Args:
            input_zarr: [n_samples, n_traces] input traces
            output_zarr: [n_traces, n_freqs] output spectra (pre-allocated)
            estimator: SpectralEstimator instance
            sample_rate_ms: Sample interval
            progress_callback: Progress function
        """
        n_samples, n_traces = input_zarr.shape
        chunk_size = self.config.chunk_size

        total_chunks = (n_traces + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start_trace = chunk_idx * chunk_size
            end_trace = min(start_trace + chunk_size, n_traces)

            # Load chunk
            chunk = input_zarr[:, start_trace:end_trace]

            # Estimate spectra
            amplitudes, phases, freqs = estimator.estimate_all_traces(
                chunk, sample_rate_ms
            )

            # Store results
            output_zarr[start_trace:end_trace, :] = amplitudes

            if progress_callback:
                progress_callback(
                    end_trace, n_traces,
                    f"Spectral estimation: chunk {chunk_idx + 1}/{total_chunks}"
                )

    def process_decomposition(
        self,
        spectral_zarr: zarr.Array,
        headers: dict,
        engine,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Surface-consistent decomposition.

        For decomposition, we need the full design matrix G.
        The matrix is sparse, so it fits in memory even for large surveys.

        We solve per-frequency, which can be parallelized.
        """
        n_traces, n_freqs = spectral_zarr.shape

        # Build design matrix (sparse, fits in memory)
        G = engine.build_design_matrix(headers, n_traces)

        # Pre-allocate results
        results = {}
        for name, info in engine.get_factor_info().items():
            n_factors = info['end'] - info['start']
            results[name] = np.zeros((n_factors, n_freqs), dtype=np.float32)

        # Process each frequency (can be parallelized)
        for f in range(n_freqs):
            # Load this frequency for all traces
            d = spectral_zarr[:, f]

            # Decompose
            freq_result = engine._decompose_single(d)

            for name in engine.get_factor_info():
                results[name][:, f] = freq_result[name]

            if progress_callback:
                progress_callback(f + 1, n_freqs, f"Decomposition: freq {f + 1}/{n_freqs}")

        return results

    def apply_correction(
        self,
        input_zarr: zarr.Array,
        output_zarr: zarr.Array,
        factors: dict,
        headers: dict,
        factor_info: dict,
        frequencies: np.ndarray,
        sample_rate_ms: float,
        mode: str = 'amplitude',  # 'amplitude' or 'deconvolution'
        progress_callback: Optional[Callable] = None
    ):
        """
        Apply surface-consistent correction in chunks.

        Args:
            input_zarr: Input traces
            output_zarr: Output traces (pre-allocated)
            factors: Pre-computed factors
            headers: Trace headers
            factor_info: Factor index information
            frequencies: Frequency axis
            sample_rate_ms: Sample interval
            mode: 'amplitude' for scalar correction, 'deconvolution' for filter
            progress_callback: Progress function
        """
        n_samples, n_traces = input_zarr.shape
        chunk_size = self.config.chunk_size

        # Pre-compute per-trace correction spectra
        # This maps each trace to its combined source + receiver + offset factor
        trace_corrections = self._compute_trace_corrections(
            factors, headers, factor_info
        )

        total_chunks = (n_traces + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start_trace = chunk_idx * chunk_size
            end_trace = min(start_trace + chunk_size, n_traces)

            # Load chunk
            chunk = input_zarr[:, start_trace:end_trace]

            # Get corrections for this chunk
            chunk_corrections = trace_corrections[start_trace:end_trace]

            if mode == 'amplitude':
                # Apply amplitude correction in frequency domain
                corrected = self._apply_amplitude_correction(
                    chunk, chunk_corrections, frequencies, sample_rate_ms
                )
            else:
                # Apply deconvolution filter
                corrected = self._apply_deconvolution(
                    chunk, chunk_corrections, frequencies, sample_rate_ms
                )

            # Store results
            output_zarr[:, start_trace:end_trace] = corrected

            if progress_callback:
                progress_callback(
                    end_trace, n_traces,
                    f"Applying correction: chunk {chunk_idx + 1}/{total_chunks}"
                )

    def _compute_trace_corrections(self, factors, headers, factor_info):
        """Pre-compute combined correction for each trace."""
        n_traces = len(headers[list(headers.keys())[0]])
        n_freqs = factors[list(factors.keys())[0]].shape[1]

        corrections = np.zeros((n_traces, n_freqs), dtype=np.float32)

        # Add source contribution
        if 'source' in factors and 'source' in factor_info:
            info = factor_info['source']
            header_key = info.get('header', 'FFID')
            if header_key in headers:
                source_vals = headers[header_key]
                mapping = info['mapping']
                for i, s in enumerate(source_vals):
                    if s in mapping:
                        corrections[i] += factors['source'][mapping[s]]

        # Add receiver contribution
        if 'receiver' in factors and 'receiver' in factor_info:
            info = factor_info['receiver']
            header_key = info.get('header', 'CHAN')
            if header_key in headers:
                receiver_vals = headers[header_key]
                mapping = info['mapping']
                for i, r in enumerate(receiver_vals):
                    if r in mapping:
                        corrections[i] += factors['receiver'][mapping[r]]

        # Add offset contribution
        if 'offset' in factors and 'offset' in factor_info:
            info = factor_info['offset']
            bin_size = info.get('bin_size', 50.0)
            if 'OFFSET' in headers:
                offsets = np.abs(headers['OFFSET'])
                offset_bins = (offsets / bin_size).astype(int)
                mapping = info['mapping']
                for i, b in enumerate(offset_bins):
                    if b in mapping:
                        corrections[i] += factors['offset'][mapping[b]]

        return corrections

    def _apply_amplitude_correction(self, traces, corrections, freqs, sample_rate_ms):
        """Apply amplitude correction via frequency domain multiplication."""
        n_samples, n_traces = traces.shape

        # FFT
        spectra = np.fft.rfft(traces, axis=0)

        # Interpolate corrections to FFT frequencies
        fft_freqs = np.fft.rfftfreq(n_samples, sample_rate_ms / 1000.0)

        corrected_spectra = spectra.copy()
        for i in range(n_traces):
            # Interpolate correction to FFT frequency grid
            correction_interp = np.interp(fft_freqs, freqs, corrections[i])

            # Convert from dB to linear and invert (we want to remove the effect)
            correction_linear = 10 ** (-correction_interp / 20.0)

            corrected_spectra[:, i] *= correction_linear

        # IFFT
        corrected = np.fft.irfft(corrected_spectra, n=n_samples, axis=0)

        return corrected.astype(np.float32)

    def _apply_deconvolution(self, traces, corrections, freqs, sample_rate_ms):
        """Apply deconvolution filter."""
        # Similar to amplitude but includes phase (minimum phase assumption)
        n_samples, n_traces = traces.shape

        spectra = np.fft.rfft(traces, axis=0)
        fft_freqs = np.fft.rfftfreq(n_samples, sample_rate_ms / 1000.0)

        corrected_spectra = spectra.copy()
        for i in range(n_traces):
            # Interpolate correction
            correction_interp = np.interp(fft_freqs, freqs, corrections[i])

            # Amplitude correction (inverse)
            amp_correction = 10 ** (-correction_interp / 20.0)

            # Minimum phase from amplitude (Hilbert transform)
            log_amp = np.log(amp_correction + 1e-10)
            phase_correction = -np.imag(np.fft.hilbert(log_amp))

            # Complex correction
            complex_correction = amp_correction * np.exp(1j * phase_correction)

            corrected_spectra[:, i] *= complex_correction

        corrected = np.fft.irfft(corrected_spectra, n=n_samples, axis=0)

        return corrected.astype(np.float32)
```

---

## 7. UI/UX Design Specifications

### 7.1 Main Workflow Panel

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Surface-Consistent Processing Workflow                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐│
│  │ 1.Input │───▶│2.Spectra│───▶│3.Decomp │───▶│4.Filter │───▶│5.Apply ││
│  │  Data   │    │Estimate │    │& QC     │    │Design   │    │& Export││
│  └────●────┘    └─────────┘    └─────────┘    └─────────┘    └────────┘│
│       │                                                                  │
│       ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Step 1: Input Data                             │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  Dataset: /surveys/north_sea_2024/cdp_sorted.zarr      [Browse]   │  │
│  │                                                                    │  │
│  │  Statistics:                                                       │  │
│  │  - Traces: 1,234,567                                               │  │
│  │  - Samples: 2,500 (5,000 ms @ 2ms)                                 │  │
│  │  - Sources: 1,024                                                  │  │
│  │  - Receivers: 4,096                                                │  │
│  │  - Offset range: 0 - 6,000 m                                       │  │
│  │  - CMPs: 2,048                                                     │  │
│  │                                                                    │  │
│  │  Required Headers: ☑ FFID  ☑ CHAN  ☑ OFFSET  ☑ CDP                │  │
│  │                    ☑ SX    ☑ SY    ☑ GX      ☑ GY                 │  │
│  │                                                                    │  │
│  │  [Validate Headers] [View Sample Gather]                           │  │
│  │                                                                    │  │
│  │                                         [Next: Spectral Estimate ▶]│  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Spectral Estimation Panel

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     Step 2: Spectral Estimation                           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Estimation Window:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │    Time Gate:  [200] ──────●────────────────●────── [2500] ms       │ │
│  │                        window start      window end                  │ │
│  │                                                                      │ │
│  │    ┌────────────────────────────────────────────────────────────┐   │ │
│  │    │                    Seismic Preview                          │   │ │
│  │    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   │ │
│  │    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   │ │
│  │    │  ████████████████████████████████████████████████████████ │   │ │
│  │    │  ████████████████████████████████████████████████████████ │   │ │
│  │    │  ████████████████████████████████████████████████████████ │   │ │
│  │    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   │ │
│  │    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   │ │
│  │    └────────────────────────────────────────────────────────────┘   │ │
│  │                      ▲ window region highlighted ▲                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  Method:  ● FFT (fastest)  ○ Multitaper (robust)  ○ CWT (time-varying)   │
│                                                                           │
│  Taper:   ● Hann  ○ Kaiser  ○ Tukey  ○ Cosine                            │
│                                                                           │
│  Frequency Range: [5] ────────────────────────── [120] Hz                │
│                                                                           │
│  Output Preview:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Average Spectrum (10 random traces)                                 │ │
│  │          ╱╲                                                          │ │
│  │         ╱  ╲                                                         │ │
│  │        ╱    ╲____                                                    │ │
│  │       ╱          ╲_______                                            │ │
│  │  ────╱                   ╲────────────────────                       │ │
│  │  0   20   40   60   80   100  120  Hz                                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  Estimated processing time: ~5 minutes (GPU)                              │
│  Output size: 1.2 GB (spectral_data.zarr)                                 │
│                                                                           │
│  [◀ Back]                                          [Run Estimation ▶]     │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Next Step | `Ctrl+N` |
| Previous Step | `Ctrl+B` |
| Run Current Step | `Ctrl+R` |
| Cancel Processing | `Esc` |
| Save Project | `Ctrl+S` |
| Toggle QC View | `Q` |
| Frequency Up/Down | `↑` / `↓` |
| Reset Equalizer | `Ctrl+0` |

---

## 8. Performance Targets

### 8.1 Benchmark Targets (M4 Max, 64GB RAM)

| Operation | Target Throughput | Memory Usage |
|-----------|-------------------|--------------|
| Spectral Estimation | 50,000 traces/s | < 2 GB |
| Decomposition (per freq) | 1M traces in < 1s | Sparse matrix only |
| Filter Application | 30,000 traces/s | < 4 GB |
| QC Map Rendering | 60 FPS | < 500 MB |

### 8.2 Scalability Targets

| Dataset Size | Processing Time | Memory |
|--------------|-----------------|--------|
| 100K traces | < 30 seconds | < 4 GB |
| 1M traces | < 5 minutes | < 8 GB |
| 10M traces | < 1 hour | < 16 GB (chunked) |
| 100M traces | < 10 hours | < 16 GB (chunked) |

---

## 9. References

### Classical Methods
- [Surface-consistent deconvolution - SEG Wiki](https://wiki.seg.org/wiki/Surface-consistent_deconvolution)
- [Surface consistent corrections | GEOPHYSICS](https://library.seg.org/doi/10.1190/1.1441133)
- [A new, simple approach to surface-consistent scaling | CSEG](https://csegrecorder.com/articles/view/a-new-simple-approach-to-surface-consistent-scaling)

### Modern Robust Methods
- [Robust Surface-Consistent Deconvolution](https://www.researchgate.net/publication/301434080_Robust_Surface-Consistent_Deconvolution_Creating_Inversion_Ready_Land_Data)
- [A new perspective of surface consistent deconvolution (2024)](https://onepetro.org/SEGAM/proceedings-abstract/IMAGE24/IMAGE24/SEG-2024-4095553/621307)
- [Surface-Consistent Sparse Multichannel Blind Deconvolution](https://ieeexplore.ieee.org/document/7387763/)

### Sparse Inversion
- [Sparse-spike seismic inversion with FISTA (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11358434/)
- [Seismic deconvolution with erratic data - L1 norm](https://onlinelibrary.wiley.com/doi/full/10.1111/1365-2478.12689)
- [Data-driven inversion with reweighted L1 norm (2023)](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1191077/full)

### Phase Estimation
- [Deconvolution of marine seismic data using L1 norm](https://academic.oup.com/gji/article/72/1/93/666866)
- [L1 norm inversion in attenuating media](https://www.earthdoc.org/content/journals/10.1111/1365-2478.12002)

---

## 10. Implementation Roadmap

### Phase 1: Core Infrastructure (Foundation)
1. Data models for factors and spectral data
2. Spectral estimator with GPU acceleration
3. Basic L2 decomposition engine
4. Chunked processing framework

### Phase 2: Solver Enhancement
1. IRLS L1/L2 hybrid solver
2. FISTA sparse regularization
3. Phase estimation methods
4. Constraint handling

### Phase 3: UI Components
1. Workflow wizard panel
2. Spectral window picker
3. Factor QC maps
4. Seismic equalizer

### Phase 4: QC and Visualization
1. Filter design panel
2. Before/after comparison viewer
3. Spectral comparison plots
4. Quality metrics dashboard

### Phase 5: Production Features
1. Batch processing dialog
2. Progress monitoring
3. Export options
4. Project save/load

---

*Document Version: 1.0*
*Last Updated: December 2024*
