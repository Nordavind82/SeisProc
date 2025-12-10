# Time-Frequency Transforms for Seismic Denoising

## Overview

This document outlines additional time-frequency transform methods for seismic data denoising, building on the existing STFT, S-Transform, and DWT implementations in SeisProc.

## Current Implementation Status

| Transform | Status | Location |
|-----------|--------|----------|
| STFT | Implemented | `processors/tf_denoise.py` |
| S-Transform | Implemented | `processors/tf_denoise.py` |
| DWT | Implemented | `processors/dwt_denoise.py` |
| SWT | Implemented | `processors/dwt_denoise.py` |

## Performance Baseline

| Method | Speed vs STFT | SNR Improvement | Energy Retained |
|--------|--------------|-----------------|-----------------|
| DWT (db4) | 6.5x faster | +5.8 dB | 70-75% |
| SWT (db4) | 1.4x faster | +6.3 dB | 65-70% |
| STFT | 1.0x baseline | +0.2 dB | 98-99% |

---

## Proposed Transforms

### Tier 1: Quick Wins

#### 1. Gabor Transform
- **Description**: STFT with Gaussian windows providing optimal time-frequency localization (minimum uncertainty principle)
- **Implementation Effort**: Low (1-2 hours)
- **Speed**: Similar to STFT
- **Advantages**:
  - Better time-frequency resolution trade-off than rectangular STFT
  - Smooth spectral estimates, reduced spectral leakage
  - Optimal for analyzing chirp-like signals
- **Seismic Applications**:
  - AVO analysis
  - Spectral decomposition
  - Thin bed detection
- **Dependencies**: None (scipy has Gaussian window)
- **Implementation Notes**:
  - Modify existing STFT to use `scipy.signal.windows.gaussian`
  - Add window width parameter (sigma) for user control

#### 2. Wavelet Packets (WPT)
- **Description**: Full binary tree decomposition of both approximation and detail coefficients at each level
- **Implementation Effort**: Low (2-4 hours)
- **Speed**: 2-3x slower than DWT
- **Advantages**:
  - Adaptive frequency band selection
  - Better frequency resolution than standard DWT
  - Optimal for narrowband noise removal
- **Seismic Applications**:
  - Ground roll removal (specific frequency bands)
  - Powerline noise (50/60 Hz harmonics)
  - Narrowband interference
- **Dependencies**: PyWavelets (`pywt.WaveletPacket`)
- **Implementation Notes**:
  - Add WPT mode to existing DWTDenoise processor
  - Allow user to select decomposition depth and best-basis selection

---

### Tier 2: Medium Effort

#### 3. Curvelet Transform
- **Description**: Multi-scale, multi-directional decomposition optimal for curve-like singularities
- **Implementation Effort**: Medium (1 day)
- **Speed**: Medium (comparable to SWT)
- **Advantages**:
  - Specifically designed for seismic-like data with curved features
  - Optimal sparse representation for reflectors
  - Excellent for coherent noise separation
  - Preserves edges and discontinuities
- **Seismic Applications**:
  - Coherent noise attenuation
  - Fault detection and enhancement
  - Multiple removal
  - Migration artifacts reduction
  - Seismic interpolation
- **Dependencies**: `curvelops` or `pylops.signalprocessing.FDCT2D`
- **Implementation Notes**:
  - Requires 2D (or 3D) implementation for full benefit
  - Can work on individual gathers or stacked sections
  - Threshold in curvelet domain, inverse transform

#### 4. Empirical Mode Decomposition (EMD/EEMD)
- **Description**: Data-driven adaptive decomposition into Intrinsic Mode Functions (IMFs)
- **Implementation Effort**: Medium (1 day)
- **Speed**: Slower than DWT (iterative sifting process)
- **Advantages**:
  - No predefined basis functions - fully adaptive
  - Handles non-linear and non-stationary signals
  - Separates signals by instantaneous frequency
  - No spectral leakage between modes
- **Seismic Applications**:
  - Non-stationary noise removal
  - Mode mixing separation
  - Trend/drift removal
  - Ground roll (often captured in low-frequency IMFs)
- **Dependencies**: `PyEMD` or `emd` library
- **Variants**:
  - EMD: Original algorithm (mode mixing issues)
  - EEMD: Ensemble EMD (noise-assisted, more robust)
  - CEEMDAN: Complete EEMD with Adaptive Noise (best quality)
- **Implementation Notes**:
  - Process trace-by-trace
  - Allow user to select which IMFs to remove/keep
  - Consider parallel processing for large gathers

#### 5. Synchrosqueezing Transform (SST)
- **Description**: Time-frequency reassignment method that sharpens STFT/CWT representations
- **Implementation Effort**: Medium (1 day)
- **Speed**: 2-3x slower than STFT
- **Advantages**:
  - Near-perfect instantaneous frequency estimation
  - Sharper time-frequency localization than STFT
  - Invertible (unlike some reassignment methods)
  - Excellent for mode separation
- **Seismic Applications**:
  - Instantaneous frequency/phase attributes
  - Thin layer detection
  - Spectral decomposition with high resolution
  - Time-varying frequency analysis
- **Dependencies**: `ssqueezepy` library
- **Implementation Notes**:
  - Can be based on STFT (STFT-SST) or CWT (CWT-SST)
  - CWT-SST generally provides better results
  - Threshold in synchrosqueezed domain

---

## Comparison Matrix

| Transform | Speed | Random Noise | Coherent Noise | Frequency Selective | Adaptive |
|-----------|-------|--------------|----------------|---------------------|----------|
| STFT | Fast | Medium | Poor | Yes | No |
| DWT | Very Fast | Good | Poor | Limited | No |
| SWT | Fast | Very Good | Poor | Limited | No |
| **Gabor** | Fast | Good | Poor | Yes | No |
| **WPT** | Medium | Very Good | Medium | Yes | Partial |
| **Curvelet** | Medium | Good | **Excellent** | Limited | No |
| **EMD** | Slow | Good | Medium | No | **Yes** |
| **SST** | Medium | Good | Medium | Yes | Partial |

---

## Recommended Priority

### Phase 1: Quick Implementations
1. Gabor Transform - Minimal code change, immediate benefit
2. Wavelet Packets - Leverages existing infrastructure

### Phase 2: High-Value Additions
3. Curvelet Transform - Unique coherent noise capability
4. EMD/EEMD - Adaptive decomposition for complex noise

### Phase 3: Specialized Tools
5. Synchrosqueezing - Advanced attributes and analysis

---

## Architecture Considerations

### Processor Design Pattern
Each transform should follow the existing `BaseProcessor` pattern:
- Inherit from `BaseProcessor`
- Implement `_validate_params()`, `process()`, `get_description()`
- Return new `SeismicData` object (immutable processing)
- Support progress callbacks for long operations

### UI Integration
Each processor needs:
- Entry in algorithm dropdown (`control_panel.py`)
- Parameter group with relevant controls
- Apply button connected to `_on_apply_clicked()`
- Registration in `PROCESSOR_REGISTRY`

### Testing Requirements
- Unit tests for transform correctness
- SNR improvement verification
- Performance benchmarks vs existing methods
- Edge case handling (short signals, single trace, pure noise)

---

## Dependencies to Add

```txt
# requirements.txt additions
PyWavelets>=1.4.0      # Already added for DWT
PyEMD>=1.0.0           # For EMD/EEMD
ssqueezepy>=0.6.0      # For Synchrosqueezing
curvelops>=0.1.0       # For Curvelet (or use pylops)
```

---

## References

1. Mallat, S. (2008). A Wavelet Tour of Signal Processing
2. Huang, N.E. et al. (1998). The empirical mode decomposition and the Hilbert spectrum
3. Candes, E. & Donoho, D. (2004). New tight frames of curvelets
4. Daubechies, I. & Maes, S. (1996). A nonlinear squeezing of the CWT
5. Chopra, S. & Marfurt, K. (2007). Seismic Attributes for Prospect Identification
