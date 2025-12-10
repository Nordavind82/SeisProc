# Time-Frequency Transform Implementation Tasks

## Overview

Implementation tasks for adding new time-frequency transforms to SeisProc, ordered by priority and effort.

---

## Task 1: Gabor Transform Denoising

**Priority**: High
**Effort**: 1-2 hours
**Depends on**: Existing STFT implementation

### Description
Add Gabor transform as a variant of STFT using Gaussian windows for optimal time-frequency localization.

### Subtasks

- [ ] **1.1** Create `processors/gabor_denoise.py`
  - Implement `GaborDenoise` class extending `BaseProcessor`
  - Use `scipy.signal.windows.gaussian` for window function
  - Add sigma (window width) parameter for user control
  - Implement MAD-based thresholding similar to TFDenoise

- [ ] **1.2** Add Gabor parameters to UI
  - Add "Gabor Transform" option to algorithm dropdown in `control_panel.py`
  - Create `_create_gabor_group()` with parameters:
    - Window size (samples)
    - Sigma (Gaussian width parameter)
    - Overlap percentage
    - Threshold k (MAD multiplier)
    - Threshold mode (soft/hard)

- [ ] **1.3** Wire up processor creation
  - Update `_on_algorithm_changed()` for Gabor index
  - Update `_on_apply_clicked()` to create GaborDenoise processor

- [ ] **1.4** Register processor
  - Add to `processors/__init__.py`
  - Add to `PROCESSOR_REGISTRY`

- [ ] **1.5** Create tests
  - `tests/test_gabor_denoise.py`
  - Test Gaussian window generation
  - Test reconstruction quality
  - Compare SNR improvement vs STFT

---

## Task 2: Wavelet Packets (WPT) Denoising

**Priority**: High
**Effort**: 2-4 hours
**Depends on**: Existing DWT implementation, PyWavelets

### Description
Add Wavelet Packet Transform mode to enable full binary tree decomposition with adaptive frequency band selection.

### Subtasks

- [ ] **2.1** Extend `processors/dwt_denoise.py`
  - Add `wpt` and `wpt_spatial` transform types
  - Implement `_process_wpt()` method using `pywt.WaveletPacket`
  - Add best-basis selection option (entropy-based)
  - Implement coefficient thresholding at all nodes

- [ ] **2.2** Update DWT UI controls
  - Add "WPT (Wavelet Packets)" to transform type dropdown
  - Add "WPT-Spatial" option
  - Add decomposition depth control
  - Add best-basis selection checkbox

- [ ] **2.3** Update processor creation
  - Handle new transform types in `_on_apply_clicked()`

- [ ] **2.4** Create/extend tests
  - Add WPT tests to `tests/test_dwt_denoise.py`
  - Test frequency band selectivity
  - Compare with standard DWT

---

## Task 3: Curvelet Transform Denoising

**Priority**: High
**Effort**: 1 day
**Depends on**: `curvelops` or `pylops` library

### Description
Implement Curvelet transform for coherent noise attenuation, leveraging its optimal representation of curve-like features (seismic reflectors).

### Subtasks

- [ ] **3.1** Add dependency
  - Add `curvelops>=0.1.0` to `requirements.txt`
  - Alternative: use `pylops.signalprocessing.FDCT2D`

- [ ] **3.2** Create `processors/curvelet_denoise.py`
  - Implement `CurveletDenoise` class
  - Support 2D processing (gather-wise)
  - Parameters: scales, angles, threshold_k
  - Implement thresholding in curvelet domain
  - Handle gather padding for transform requirements

- [ ] **3.3** Add Curvelet UI controls
  - Add "Curvelet Transform" to algorithm dropdown
  - Create `_create_curvelet_group()` with:
    - Number of scales
    - Number of angles per scale
    - Threshold k (MAD multiplier)
    - Threshold mode (soft/hard)
    - Processing mode (2D gather / trace-by-trace)

- [ ] **3.4** Wire up and register
  - Update `_on_algorithm_changed()`
  - Update `_on_apply_clicked()`
  - Add to processor registry

- [ ] **3.5** Create tests
  - `tests/test_curvelet_denoise.py`
  - Test coherent noise attenuation (linear events)
  - Test reflector preservation
  - Benchmark performance

---

## Task 4: EMD/EEMD Denoising

**Priority**: Medium
**Effort**: 1 day
**Depends on**: `PyEMD` library

### Description
Implement Empirical Mode Decomposition for adaptive, data-driven signal decomposition into Intrinsic Mode Functions (IMFs).

### Subtasks

- [ ] **4.1** Add dependency
  - Add `EMD-signal>=1.0.0` or `PyEMD>=1.0.0` to `requirements.txt`

- [ ] **4.2** Create `processors/emd_denoise.py`
  - Implement `EMDDenoise` class
  - Support EMD, EEMD, and CEEMDAN variants
  - Parameters: method, num_imfs, ensemble_size (for EEMD)
  - Allow selection of IMFs to remove (low-freq, high-freq, specific)
  - Implement parallel processing for trace-by-trace

- [ ] **4.3** Add EMD UI controls
  - Add "EMD Decomposition" to algorithm dropdown
  - Create `_create_emd_group()` with:
    - Method selection (EMD/EEMD/CEEMDAN)
    - Number of IMFs to compute
    - IMFs to remove (checkboxes or range)
    - Ensemble size (for EEMD variants)
    - Noise amplitude (for EEMD)

- [ ] **4.4** Wire up and register
  - Update control panel handlers
  - Add to processor registry

- [ ] **4.5** Create tests
  - `tests/test_emd_denoise.py`
  - Test IMF separation quality
  - Test non-stationary signal handling
  - Compare EMD vs EEMD vs CEEMDAN

---

## Task 5: Synchrosqueezing Transform Denoising

**Priority**: Medium
**Effort**: 1 day
**Depends on**: `ssqueezepy` library

### Description
Implement Synchrosqueezing Transform for high-resolution time-frequency analysis with sharp instantaneous frequency estimation.

### Subtasks

- [ ] **5.1** Add dependency
  - Add `ssqueezepy>=0.6.0` to `requirements.txt`

- [ ] **5.2** Create `processors/sst_denoise.py`
  - Implement `SSTDenoise` class
  - Support STFT-based and CWT-based SST
  - Parameters: wavelet (for CWT), nv (voices), threshold_k
  - Implement thresholding in synchrosqueezed domain
  - Handle reconstruction from thresholded coefficients

- [ ] **5.3** Add SST UI controls
  - Add "Synchrosqueezing Transform" to algorithm dropdown
  - Create `_create_sst_group()` with:
    - Base transform (STFT/CWT)
    - Wavelet selection (for CWT mode)
    - Number of voices
    - Threshold k
    - Threshold mode

- [ ] **5.4** Wire up and register
  - Update control panel handlers
  - Add to processor registry

- [ ] **5.5** Create tests
  - `tests/test_sst_denoise.py`
  - Test instantaneous frequency estimation
  - Test reconstruction quality
  - Compare time-frequency resolution vs STFT

---

## Summary Table

| Task | Transform | Priority | Effort | Dependencies |
|------|-----------|----------|--------|--------------|
| 1 | Gabor | High | 1-2 hrs | scipy (existing) |
| 2 | Wavelet Packets | High | 2-4 hrs | PyWavelets (existing) |
| 3 | Curvelet | High | 1 day | curvelops (new) |
| 4 | EMD/EEMD | Medium | 1 day | PyEMD (new) |
| 5 | Synchrosqueezing | Medium | 1 day | ssqueezepy (new) |

---

## Checklist for Each Implementation

For each new transform processor:

- [ ] Create processor class in `processors/`
- [ ] Implement `BaseProcessor` interface
- [ ] Add to `processors/__init__.py`
- [ ] Add to `PROCESSOR_REGISTRY`
- [ ] Create UI parameter group in `control_panel.py`
- [ ] Add to algorithm dropdown
- [ ] Update `_on_algorithm_changed()`
- [ ] Update `_on_apply_clicked()`
- [ ] Create test file in `tests/`
- [ ] Verify SNR improvement
- [ ] Benchmark performance
- [ ] Update documentation
