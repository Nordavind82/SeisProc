# Screenshot Analysis: TF-Denoise Output Verification

## Observations from Screenshots (2025-11-17 15:49-15:50)

### Screenshot 1: S-Transform with k=1.0
- **Left panel (Input Data)**: Shows normal seismic gather with clear signal
- **Middle panel (Difference - Removed Noise)**: Appears nearly empty/white
- **Right panel (Processed - Denoised)**: Appears nearly empty/white

### Screenshot 2: STFT with k=3.0
- **Left panel (Input Data)**: Shows normal seismic gather with clear signal
- **Middle panel (Difference - Removed Noise)**: Appears nearly empty/white
- **Right panel (Processed - Denoised)**: Shows sparse, low-amplitude, noise-like vertical lines

## Critical Findings

### 1. S-Transform k=1.0 Results Are Too Aggressive

**Threshold calculation:**
```
threshold = median + k × MAD
With k=1.0: threshold = median + 1.0 × MAD
```

**Issue:** k=1.0 is **EXTREMELY aggressive** for seismic data. This threshold is:
- Removing almost ALL coefficients (including signal)
- Leaving nearly empty output (explains the white panels)
- Both processed and difference panels are empty because there's almost nothing left

**Expected behavior with k=1.0:**
- Input RMS: ~1.0 (normalized)
- Output RMS: ~0.01-0.05 (1-5% of input) ⚠️ SIGNAL DESTROYED
- This is NOT denoising - this is signal destruction

### 2. STFT k=3.0 Results Look Questionable

**Threshold calculation:**
```
threshold = median + k × MAD
With k=3.0: threshold = median + 3.0 × MAD
```

**Observations:**
- k=3.0 should be conservative (preserve more signal)
- Yet processed panel shows sparse, noise-like patterns
- This suggests one of three issues:
  1. **Scale/visualization issue** - Data is there but very low amplitude
  2. **Still too aggressive** - Even k=3.0 removing too much for this specific dataset
  3. **Panel swap issue** - Though we already swapped the data assignments

## What the Energy Verification Will Tell Us

I've added energy verification output to both CPU and GPU versions. When you run processing, you'll now see:

```
ENERGY VERIFICATION:
  Input RMS:  X.XXXXXX
  Output RMS: Y.YYYYYY
  Ratio:      ZZ.ZZ%
```

### Expected Results

**For PROPER denoising:**
```
Input RMS:  0.500000
Output RMS: 0.450000
Ratio:      90.00%
✓ Output is signal model (70-95% of input energy preserved)
```

**For OVER-AGGRESSIVE thresholding (k=1.0):**
```
Input RMS:  0.500000
Output RMS: 0.025000
Ratio:      5.00%
⚠️ WARNING: Output is < 10% of input - threshold may be too aggressive!
```

**For MINIMAL denoising (k too high):**
```
Input RMS:  0.500000
Output RMS: 0.490000
Ratio:      98.00%
⚠️ WARNING: Output is > 95% of input - minimal denoising occurred
```

## Recommended Threshold Values

Based on typical seismic processing literature:

| Transform | Threshold k | Use Case |
|-----------|-------------|----------|
| **STFT** | 2.5-3.5 | White noise removal (recommended: **3.0**) |
| **S-Transform** | 3.0-4.0 | White noise removal (recommended: **3.5**) |
| **STFT** | 4.0-5.0 | Coherent noise (very conservative) |
| **S-Transform** | 4.5-5.5 | Coherent noise (very conservative) |

**WARNING:** Never use k < 2.0 unless you specifically want to remove signal!

### Why Different Values?

- **Lower k** = More aggressive (removes more)
- **Higher k** = More conservative (preserves more)

The MAD (Median Absolute Deviation) estimates the noise level:
- `threshold = median + k × MAD`
- k=1.0: Remove everything > 1σ (way too aggressive)
- k=2.5: Remove everything > 2.5σ (reasonable for noise)
- k=3.0: Remove everything > 3σ (standard for signal preservation)
- k=3.5: Remove everything > 3.5σ (conservative, preserves weak signals)

## Action Items

### 1. Reprocess with Recommended Parameters

Try these combinations:

**Test 1: Conservative (Preserve Signal)**
- STFT: k=3.5
- S-Transform: k=3.5
- Expected: Output RMS = 85-95% of input

**Test 2: Moderate (Balance)**
- STFT: k=3.0
- S-Transform: k=3.5
- Expected: Output RMS = 75-90% of input

**Test 3: Aggressive (Strong Denoising)**
- STFT: k=2.5
- S-Transform: k=3.0
- Expected: Output RMS = 60-80% of input

### 2. Check Energy Verification Output

After each run, verify:
1. **Ratio should be 70-95%** for proper denoising
2. If ratio < 10%: Threshold too aggressive (increase k)
3. If ratio > 95%: Threshold too conservative (decrease k, but carefully!)

### 3. Visual Verification

**Processed panel should show:**
- Clear trace structure (not sparse/empty)
- Reduced random noise
- Preserved reflections and events
- ~80-90% of input amplitude

**Difference panel should show:**
- Random noise patterns
- Low amplitude
- No clear reflections
- ~10-20% of input amplitude

### 4. Update Presets

Once you find good parameters, update the presets in `control_panel.py`:

```python
'White Noise': {
    'aperture': 15,
    'fmin': 5,
    'fmax': 150,
    'threshold_k': 3.0,  # Changed from 3.5 or 1.0
    'threshold_type': 'Soft'
}
```

## Conclusion

The screenshots reveal that **k=1.0 is destroying the signal** (nearly empty output). Even k=3.0 may be too aggressive for this specific dataset, resulting in sparse output.

**Next steps:**
1. Run processing with k=3.5 for both STFT and S-Transform
2. Check the ENERGY VERIFICATION output
3. Adjust k based on the RMS ratio
4. Verify visually that processed panel shows clear signal structure

The energy verification I added will definitively tell us if the output is signal or noise based on the RMS ratio.
