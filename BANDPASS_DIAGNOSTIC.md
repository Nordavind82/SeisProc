# Bandpass Filter Diagnostic Guide

## Current Implementation

The bandpass filter uses:
- **Algorithm:** Zero-phase Butterworth filter
- **Method:** `scipy.signal.sosfiltfilt()` (forward-backward filtering)
- **Advantages:** No phase shift, stable second-order sections (SOS)

## Potential Issues to Check

### 1. Viewer Assignment
**Already Fixed:** Lines 481-482 in `main_window.py` correctly assign:
- `processed_viewer` ← Filtered signal ✅
- `difference_viewer` ← Removed frequencies ✅

### 2. Filter Parameters
Check if parameters are valid:
- **Low frequency** < **High frequency**
- **High frequency** < **Nyquist frequency**
- **Order:** Typical range 2-6 (higher = steeper rolloff but potential instability)

### 3. Expected Behavior

**What You Should See:**
```
┌─────────────┬─────────────────────────┬─────────────────────────┐
│   Input     │   Processed (Filtered)  │   Difference (Removed)  │
├─────────────┼─────────────────────────┼─────────────────────────┤
│             │                         │                         │
│ All freqs   │ Only freqs in pass band │ Freqs outside pass band │
│ (0-Nyquist) │ (low_freq - high_freq)  │ (0-low, high-Nyquist)   │
│             │                         │                         │
└─────────────┴─────────────────────────┴─────────────────────────┘
```

**Example: 10-50 Hz Bandpass**
- **Input:** Broadband signal (0-250 Hz)
- **Processed:** Only 10-50 Hz content (signal events in this band)
- **Difference:** 0-10 Hz + 50-250 Hz (low freq noise + high freq noise)

### 4. Common Issues

#### Issue: Processed looks same as input
**Cause:** Pass band includes all signal energy
**Solution:** Adjust frequency range to exclude noise frequencies

#### Issue: Processed is very weak
**Cause:** Pass band excludes most signal energy
**Solution:** Widen frequency range or check signal spectrum

#### Issue: Difference has signal events
**Cause:** Expected behavior - removing frequencies will remove signal too
**Note:** Bandpass is NOT adaptive - it removes ALL energy outside pass band

#### Issue: Ringing artifacts
**Cause:** Filter order too high or frequencies too close to Nyquist
**Solution:** Reduce filter order (try 2-4) or lower high frequency

### 5. How to Diagnose

1. **Check frequency content of input:**
   - Look at amplitude spectrum
   - Identify where signal vs noise energy is

2. **Choose appropriate pass band:**
   - Include frequencies with signal
   - Exclude frequencies with dominant noise

3. **Verify Nyquist:**
   - Sample rate = 1000 Hz → Nyquist = 500 Hz
   - High freq must be < 500 Hz

4. **Test with simple parameters:**
   - Start with wide band (e.g., 5-100 Hz)
   - Gradually narrow to isolate specific frequencies

### 6. Energy Conservation

Bandpass should conserve energy:
```
Energy(Input) ≈ Energy(Processed) + Energy(Difference)
```

If this doesn't hold:
- Check for numerical issues
- Check for data clipping
- Verify filter stability

## What "Mixed Results" Might Mean

Without more details, possible interpretations:

1. **Viewers are swapped** (already fixed)
2. **Partial filtering** - only some frequencies removed
3. **Ringing/artifacts** - filter introduces oscillations
4. **Unexpected frequency content** - signal not in expected band
5. **Energy imbalance** - doesn't add up to input

## Recommended Tests

### Test 1: Full Band (No Filtering)
- Low: 1 Hz
- High: Nyquist - 10 Hz
- Expected: Processed ≈ Input, Difference ≈ 0

### Test 2: Low-Pass (Remove High Frequencies)
- Low: 1 Hz
- High: 50 Hz
- Expected: Processed has low freqs only, Difference has high freqs

### Test 3: High-Pass (Remove Low Frequencies)
- Low: 50 Hz
- High: Nyquist - 10 Hz
- Expected: Processed has high freqs only, Difference has low freqs

### Test 4: Narrow Band
- Low: 40 Hz
- High: 60 Hz
- Expected: Processed has very limited content, Difference has most energy

---

**If issue persists, please provide:**
1. Screenshot showing the "mixed results"
2. Parameters used (low freq, high freq, order)
3. Sample rate of data
4. Description of what you expected vs what you see
