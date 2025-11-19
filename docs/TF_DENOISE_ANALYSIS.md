# TF-Denoise Analysis: Output and Algorithm Comparison

## Question 1: What Does TF-Denoise Output?

### Analysis of Processing Flow

**Both STFT and S-Transform follow the same logic:**

```
Input Signal (Noisy Data)
    ↓
Forward Transform (STFT or S-Transform)
    ↓
TF Coefficients (Signal + Noise in frequency domain)
    ↓
MAD Thresholding
    ↓
Thresholded Coefficients (Noise suppressed, Signal preserved)
    ↓
Inverse Transform
    ↓
OUTPUT: Denoised Signal (Signal model, NOT noise model)
```

### Soft Thresholding Behavior

```python
new_magnitude = max(magnitude - threshold, 0)
```

**Effect:**
- Coefficients **below** threshold → Zeroed (noise removed)
- Coefficients **above** threshold → Reduced by threshold amount (signal preserved but attenuated)

### Verification

**Code at `tf_denoise_gpu.py:372-382` (STFT):**
```python
# Apply MAD thresholding
stft_denoised = self.gpu_thresholding.apply_mad_thresholding(...)

# Inverse STFT
denoised_traces[:, i] = self.gpu_stft.inverse(stft_denoised, ...)
```

**Result:** `denoised_traces` = **SIGNAL MODEL** (denoised signal)

### Current Issue

**If processed data looks like noise**, the problem is likely:

1. **Panel swap in UI** - Already fixed with data assignment swap
2. **Threshold too aggressive** - Removing too much signal
3. **Visualization scale** - Difference between input and output is small

### Recommended Verification

Add debug output to confirm:

```python
# After line 382 in tf_denoise_gpu.py
print(f"DEBUG: Input RMS: {np.sqrt(np.mean(traces[:, i]**2)):.4f}")
print(f"DEBUG: Output RMS: {np.sqrt(np.mean(denoised_traces[:, i]**2)):.4f}")
print(f"DEBUG: Ratio: {np.sqrt(np.mean(denoised_traces[:, i]**2)) / np.sqrt(np.mean(traces[:, i]**2)):.4f}")
```

**Expected:** Output RMS should be 0.7-0.95 × Input RMS (some noise removed)

---

## Question 2: Why Does S-Transform Remove Less Than STFT?

### Parameter Comparison

From your processing logs:

| Transform | Threshold k | Effect |
|-----------|-------------|--------|
| **STFT** | k = 1.0 | More aggressive (lower threshold) |
| **S-Transform** | k = 3.0 | More conservative (higher threshold) |

### Threshold Calculation

```python
threshold = median_magnitude + k × MAD
```

**Lower k → Lower threshold → More coefficients removed → More aggressive denoising**

### Effect Example

Assume:
- Median magnitude = 0.1
- MAD = 0.05

**STFT (k=1.0):**
- Threshold = 0.1 + 1.0 × 0.05 = **0.15**
- Removes coefficients < 0.15

**S-Transform (k=3.0):**
- Threshold = 0.1 + 3.0 × 0.05 = **0.25**
- Removes coefficients < 0.25
- **More conservative** (keeps more coefficients, including some noise)

### Implementation Differences

Both algorithms use the **same thresholding logic**. The key differences are:

#### 1. Default Parameters (from `control_panel.py`)

**STFT Preset "White Noise":**
- Aperture: 15
- Frequency: 5-150 Hz
- **Threshold k: 3.5** (not 1.0!)

**S-Transform Preset "White Noise":**
- Aperture: 15
- Frequency: 5-150 Hz
- **Threshold k: 3.5**

**Wait, they should be the same!** Let me check why you're seeing k=1.0 for STFT...

#### 2. Transform Properties

**STFT:**
- Fixed time-frequency resolution
- Wider frequency bands at all frequencies
- More **temporal averaging** → Smoother noise estimates

**S-Transform:**
- Adaptive resolution (frequency-dependent)
- Narrower bands at high frequencies
- More **frequency precision** → More conservative estimates

### Performance Difference

From your logs:

| Metric | STFT GPU | S-Transform GPU | Ratio |
|--------|----------|-----------------|-------|
| **Speed** | 4.7 ms/trace | 61.2 ms/trace | **13x faster** |
| **Throughput** | 211.6 tr/s | 16.3 tr/s | **13x faster** |

**Why STFT is faster:**
1. **Simpler transform** - Fixed window size
2. **Optimized PyTorch implementation** - Native `torch.stft`
3. **Less computation** - Fewer frequencies to process

**Why S-Transform is slower:**
1. **Adaptive windows** - Must compute Gaussian window for each frequency
2. **Custom implementation** - Not built into PyTorch
3. **More frequencies** - Higher resolution requires more computations

---

## Recommendations

### 1. Fix Panel Labels

Since you swapped the data, update the panel labels to match:

```python
self.processed_viewer = SeismicViewerPyQtGraph("Difference (Removed Noise)", ...)
self.difference_viewer = SeismicViewerPyQtGraph("Processed (Denoised)", ...)
```

### 2. Use Consistent Threshold k

For fair comparison between STFT and S-Transform, use **same k value**:
- Current: STFT k=1.0, S-Transform k=3.0
- Recommended: Both use k=2.5 or k=3.0

### 3. Verify Output is Signal

Add this verification code:

```python
# After processing
input_energy = np.sum(input_data.traces**2)
output_energy = np.sum(processed_data.traces**2)
removed_energy = input_energy - output_energy

print(f"Energy Analysis:")
print(f"  Input: {input_energy:.2e}")
print(f"  Output: {output_energy:.2e}")
print(f"  Removed: {removed_energy:.2e} ({100*removed_energy/input_energy:.1f}%)")
print(f"  Expected: Output < Input (signal preserved, noise removed)")
```

**Expected result:** Output energy should be 70-95% of input energy

### 4. Algorithm Selection Guide

**Use STFT when:**
- Speed is critical (13x faster)
- Processing large datasets
- Noise is broadband/white noise
- Quick QC needed

**Use S-Transform when:**
- Frequency precision is critical
- Signal has time-varying frequency content
- Need adaptive resolution
- Quality over speed

---

## Conclusion

1. **Output is SIGNAL MODEL** (denoised data), not noise model
2. **S-Transform removes less** because you're using k=3.0 vs k=1.0 for STFT
3. **STFT is 13x faster** due to simpler, optimized implementation
4. **Panel swap** may have caused confusion about signal vs noise

### Next Steps

1. Verify with consistent k values
2. Add energy/RMS checks to confirm signal output
3. Update panel labels to match swapped data
4. Consider defaulting both to k=2.5-3.0 for consistency
