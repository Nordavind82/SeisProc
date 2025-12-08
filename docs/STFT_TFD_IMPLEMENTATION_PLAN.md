# STFT TFD Algorithm - Implementation Plan

## Overview

This document provides a detailed plan to fix bugs, clean up code, and optimize the STFT-based Time-Frequency Denoising algorithm in `processors/tf_denoise.py`.

---

## Current Issues Summary

| Issue | Severity | Type | Location |
|-------|----------|------|----------|
| MAD=0 causes no attenuation | **CRITICAL** | Bug | `_process_with_stft()` L796-818 |
| fmin/fmax ignored in STFT | **MEDIUM** | Design | `_process_with_stft()` |
| Unused helper functions | **LOW** | Cleanup | L336-397 |
| Per-trace STFT loop (slow) | **MEDIUM** | Performance | `_process_with_stft()` L774-776 |
| Frequency loop not vectorized | **MEDIUM** | Performance | `_process_with_stft()` L792 |

---

## Implementation Plan

### Phase 1: Fix Critical Bug (MAD=0)

**File**: `processors/tf_denoise.py`
**Location**: `_process_with_stft()` method, lines 796-801

#### Current Code (BUGGY):
```python
# Line 797-801
median_amp = np.median(spatial_amplitudes, axis=0)
mad = np.median(np.abs(spatial_amplitudes - median_amp), axis=0)

# MAD-based outlier threshold
outlier_threshold = self.threshold_k * mad
```

#### Problem:
When MAD=0 (identical traces), threshold=0, so:
- `new_deviation = max(deviation - 0, 0) = deviation`
- `new_magnitude = median + deviation = original` (no attenuation!)

#### Fix Strategy:
Use robust MAD with minimum floor based on signal level:

```python
# REPLACE lines 797-801 with:

# Compute spatial median and MAD (vectorized across time)
median_amp = np.median(spatial_amplitudes, axis=0)  # shape: (n_times,)
mad = np.median(np.abs(spatial_amplitudes - median_amp), axis=0)  # shape: (n_times,)

# CRITICAL FIX: Prevent MAD=0 from causing zero threshold
# Use adaptive floor: 1% of median or small epsilon
min_mad = np.maximum(0.01 * median_amp, 1e-10)
mad = np.maximum(mad, min_mad)

# MAD-based outlier threshold
outlier_threshold = self.threshold_k * mad
```

#### Why This Works:
- When MAD=0 (identical traces), floor kicks in
- Floor is proportional to signal level (1% of median)
- Prevents division issues and ensures outliers are detected
- Epsilon fallback (1e-10) handles zero-signal case

---

### Phase 2: Add Frequency Filtering to STFT

**File**: `processors/tf_denoise.py`
**Location**: `_process_with_stft()` method

#### Current Behavior:
- fmin/fmax parameters exist but are ignored in STFT path
- All frequencies processed regardless of settings

#### Fix Strategy:
Add frequency masking to process only the specified range:

```python
def _process_with_stft(self, ensemble, center_idx, sample_rate=None):
    """
    Process ensemble using STFT.

    Args:
        ensemble: Spatial aperture traces (n_samples, n_traces)
        center_idx: Index of center trace in ensemble
        sample_rate: Sample rate in Hz (for frequency filtering)
    """
    n_samples, n_traces = ensemble.shape
    center_trace = ensemble[:, center_idx]

    # Compute STFT for all traces
    nperseg = min(64, n_samples // 4)
    noverlap = nperseg // 2

    # ... existing STFT computation ...

    # Get frequency axis in Hz
    if sample_rate is not None:
        # freqs from scipy.stft are normalized (0 to 0.5)
        # Convert to Hz: freq_hz = freq_norm * sample_rate
        freq_hz = freqs * sample_rate

        # Create frequency mask
        freq_mask = (freq_hz >= self.fmin) & (freq_hz <= self.fmax)
    else:
        freq_mask = np.ones(n_freqs, dtype=bool)

    # Only process frequencies in range
    for f in range(n_freqs):
        if not freq_mask[f]:
            # Outside range: keep original (or zero for noise-only bands)
            stft_denoised[f, :] = stft_center[f, :]
            continue

        # ... existing thresholding logic ...
```

#### Caller Update:
In `process()` method, pass sample_rate:

```python
# Line ~598 in process()
denoised_traces[:, trace_idx] = self._process_with_stft(
    ensemble, center_in_ensemble,
    sample_rate=2.0 * data.nyquist_freq  # Convert to Hz
)
```

---

### Phase 3: Performance Optimization - Batch STFT

**File**: `processors/tf_denoise.py`
**Location**: `_process_with_stft()` lines 774-776

#### Current Code (SLOW):
```python
# Sequential STFT - one trace at a time
stft_ensemble = []
for i in range(n_traces):
    stft, freqs, times = stft_transform(ensemble[:, i], nperseg=nperseg)
    stft_ensemble.append(stft)
stft_ensemble = np.array(stft_ensemble)
```

#### Optimized Code:
```python
# Batch STFT using scipy's ability to handle 2D input
from scipy import signal

nperseg = min(64, n_samples // 4)
noverlap = nperseg // 2

# Transpose for batch processing: (n_traces, n_samples)
ensemble_T = ensemble.T

# Compute all STFTs at once
# scipy.signal.stft can process along last axis
freqs, times, stft_batch = signal.stft(
    ensemble_T,
    nperseg=nperseg,
    noverlap=noverlap,
    axis=-1  # Process along samples axis
)
# Result shape: (n_traces, n_freqs, n_times)
stft_ensemble = stft_batch
```

**Expected Speedup**: ~2-3x for STFT computation phase

---

### Phase 4: Performance Optimization - Full Vectorization

**File**: `processors/tf_denoise.py`
**Location**: `_process_with_stft()` lines 792-830

#### Current Code:
```python
# Loop over frequencies (partially vectorized)
for f in range(n_freqs):
    spatial_amplitudes = np.abs(stft_ensemble[:, f, :])
    # ... process one frequency at a time
```

#### Fully Vectorized Code:
```python
# Process ALL frequencies at once (fully vectorized)
# stft_ensemble shape: (n_traces, n_freqs, n_times)
# stft_center shape: (n_freqs, n_times)

# Compute spatial statistics for all freqs at once
all_amplitudes = np.abs(stft_ensemble)  # (n_traces, n_freqs, n_times)

# Median across spatial dimension (axis=0)
median_amp = np.median(all_amplitudes, axis=0)  # (n_freqs, n_times)

# MAD across spatial dimension
mad = np.median(np.abs(all_amplitudes - median_amp), axis=0)  # (n_freqs, n_times)

# Apply minimum floor (vectorized)
min_mad = np.maximum(0.01 * median_amp, 1e-10)
mad = np.maximum(mad, min_mad)

# Threshold
outlier_threshold = self.threshold_k * mad  # (n_freqs, n_times)

# Center trace processing
magnitudes = np.abs(stft_center)  # (n_freqs, n_times)
phases = np.angle(stft_center)    # (n_freqs, n_times)

# Deviation from median
deviations = np.abs(magnitudes - median_amp)  # (n_freqs, n_times)

# Soft thresholding (vectorized)
if self.threshold_type == 'soft':
    new_deviations = np.maximum(deviations - outlier_threshold, 0)
    signs = np.where(magnitudes >= median_amp, 1, -1)
    new_magnitudes = np.maximum(median_amp + signs * new_deviations, 0)
else:  # garrote
    new_deviations = np.where(
        deviations > outlier_threshold,
        deviations - (outlier_threshold**2 / (deviations + 1e-10)),
        deviations
    )
    signs = np.where(magnitudes >= median_amp, 1, -1)
    new_magnitudes = np.maximum(median_amp + signs * new_deviations, 0)

# Apply frequency mask if specified
if hasattr(self, '_freq_mask') and self._freq_mask is not None:
    # Keep original values outside frequency range
    new_magnitudes = np.where(self._freq_mask[:, np.newaxis], new_magnitudes, magnitudes)

# Reconstruct
stft_denoised = new_magnitudes * np.exp(1j * phases)
```

**Expected Speedup**: ~5-10x by eliminating Python loop

---

### Phase 5: Code Cleanup - Remove Unused Functions

**File**: `processors/tf_denoise.py`
**Location**: Lines 336-397

#### Functions to Remove:
1. `compute_mad_threshold()` (lines 336-354) - Not used in STFT/S-Transform paths
2. `soft_threshold()` (lines 357-374) - Not used (inline vectorized version used)
3. `garrote_threshold()` (lines 377-397) - Not used (inline vectorized version used)

#### Action:
Delete these functions entirely OR refactor to use them in the main code.

**Recommendation**: Delete them since:
- Inline vectorized code is faster
- They implement different algorithms (magnitude vs deviation thresholding)
- Keeping them causes confusion about what's actually used

---

### Phase 6: Unified Thresholding Module (Optional Enhancement)

Create a dedicated thresholding module for reuse:

**New File**: `processors/thresholding.py`

```python
"""
Vectorized thresholding functions for TF-domain denoising.
"""
import numpy as np
from typing import Literal


def compute_robust_mad(
    spatial_amplitudes: np.ndarray,
    axis: int = 0,
    min_floor_ratio: float = 0.01,
    epsilon: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute robust MAD with adaptive floor to prevent zero threshold.

    Args:
        spatial_amplitudes: Array of amplitudes (traces × freqs × times) or (traces × times)
        axis: Axis representing spatial dimension
        min_floor_ratio: Minimum MAD as fraction of median
        epsilon: Absolute minimum floor

    Returns:
        median: Median amplitudes
        mad: Robust MAD values (never zero)
    """
    median = np.median(spatial_amplitudes, axis=axis)
    mad = np.median(np.abs(spatial_amplitudes - np.expand_dims(median, axis)), axis=axis)

    # Apply adaptive floor
    min_mad = np.maximum(min_floor_ratio * median, epsilon)
    mad = np.maximum(mad, min_mad)

    return median, mad


def soft_threshold_deviation(
    magnitudes: np.ndarray,
    median: np.ndarray,
    threshold: np.ndarray
) -> np.ndarray:
    """
    Soft thresholding on deviation from median (not magnitude).

    Args:
        magnitudes: Original magnitudes
        median: Spatial median values
        threshold: Threshold values (k * MAD)

    Returns:
        New magnitudes after thresholding
    """
    deviations = np.abs(magnitudes - median)
    new_deviations = np.maximum(deviations - threshold, 0)
    signs = np.where(magnitudes >= median, 1, -1)
    new_magnitudes = np.maximum(median + signs * new_deviations, 0)
    return new_magnitudes


def garrote_threshold_deviation(
    magnitudes: np.ndarray,
    median: np.ndarray,
    threshold: np.ndarray
) -> np.ndarray:
    """
    Garrote thresholding on deviation from median.

    Args:
        magnitudes: Original magnitudes
        median: Spatial median values
        threshold: Threshold values (k * MAD)

    Returns:
        New magnitudes after thresholding
    """
    deviations = np.abs(magnitudes - median)
    new_deviations = np.where(
        deviations > threshold,
        deviations - (threshold**2 / (deviations + 1e-10)),
        deviations
    )
    signs = np.where(magnitudes >= median, 1, -1)
    new_magnitudes = np.maximum(median + signs * new_deviations, 0)
    return new_magnitudes
```

---

## Implementation Order

### Priority 1 (Critical - Do First):
1. **Fix MAD=0 bug** - 5 lines of code change
   - Location: `_process_with_stft()` lines 797-801
   - Time: ~10 minutes

### Priority 2 (Important):
2. **Full vectorization** - Remove frequency loop
   - Location: `_process_with_stft()` lines 792-830
   - Time: ~30 minutes

3. **Batch STFT** - Remove trace loop
   - Location: `_process_with_stft()` lines 774-776
   - Time: ~20 minutes

### Priority 3 (Enhancement):
4. **Add frequency filtering**
   - Location: `_process_with_stft()` and `process()`
   - Time: ~30 minutes

### Priority 4 (Cleanup):
5. **Remove unused functions**
   - Location: Lines 336-397
   - Time: ~5 minutes

6. **Optional: Create thresholding module**
   - New file: `processors/thresholding.py`
   - Time: ~20 minutes

---

## Testing Plan

After each phase, run these tests:

```python
# Test 1: Clean signal preservation
# Expected: >95% energy preserved

# Test 2: Random noise attenuation
# Expected: Correlation with clean signal improves

# Test 3: Impulse noise attenuation (CRITICAL)
# Expected: >50% attenuation of impulse spikes

# Test 4: Edge trace handling
# Expected: No artifacts at gather edges

# Test 5: Performance benchmark
# Expected: Improvement over baseline
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `processors/tf_denoise.py` | Main algorithm fixes |
| `processors/thresholding.py` | New file (optional) |
| `processors/tf_denoise_gpu.py` | Apply same fixes to GPU version |

---

## Rollback Plan

Before making changes:
1. Create backup: `cp processors/tf_denoise.py processors/tf_denoise.py.bak`
2. Use git branch: `git checkout -b fix/stft-tfd-bugs`
3. Commit after each phase

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Impulse attenuation | 0% | >50% |
| Processing speed | ~700 traces/sec | ~2000+ traces/sec |
| Code clarity | Confusing (unused funcs) | Clean |
| API consistency | fmin/fmax ignored | Properly used |
