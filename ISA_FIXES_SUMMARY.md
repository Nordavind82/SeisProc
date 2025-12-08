# ISA Fixes Summary

## Issues Fixed

### 1. ✓ Grayscale Colormap Not Working

**Problem:** Selecting "gray" colormap didn't change the seismic data visualization.

**Root Cause:**
- ISA window was sending 'gray' to the viewer
- Seismic viewer expected 'grayscale'
- Name mismatch caused colormap not to apply

**Solution:**
```python
# Before:
self.colormap_combo.addItems(['seismic', 'gray', 'RdBu', ...])

# After:
self.colormap_combo.addItems(['seismic', 'grayscale', 'viridis', 'plasma', 'jet', 'inferno'])
```

**Result:** Grayscale (and all colormaps) now work correctly ✓

---

### 2. ✓ Linear/Log Scale Logic Error

**Problem:**
- "Linear (dB)" option was confusing (dB is logarithmic)
- "Logarithmic" option applied log Y-axis to dB data (double log!)
- No true linear amplitude option

**Root Cause:** Poor understanding of scale terminology:
- **dB scale** = logarithmic amplitude representation (20*log10)
- **Linear scale** = actual amplitude values
- Axis scale should be linear for both, unless log frequency axis requested

**Solution:**

**Before (Incorrect):**
```
Y-axis options:
○ Linear (dB)      → dB values, but name confusing
○ Logarithmic      → Applied log Y-axis (wrong!)
```

**After (Correct):**
```python
Y-axis (Amplitude):
○ dB scale (20*log10)     → dB values, linear Y-axis
○ Linear amplitude        → Linear values, linear Y-axis

X-axis (Frequency):
○ Linear frequency        → Linear X-axis
○ Log frequency          → Log X-axis
```

**Implementation:**
```python
# Y-axis data selection
if self.spectrum_y_scale == 'db':
    y_data = amplitudes_db          # dB values
    y_label = 'Amplitude (dB)'
else:
    y_data = 10 ** (amplitudes_db / 20.0)  # Convert to linear
    y_label = 'Amplitude (Linear)'

# Plot with LINEAR Y-axis (default)
self.spectrum_ax.plot(plot_freqs, y_data, 'b-')

# X-axis scale selection (separate)
if self.spectrum_x_scale == 'log':
    self.spectrum_ax.set_xscale('log')  # Only X-axis is log
else:
    self.spectrum_ax.set_xscale('linear')
```

**Result:**
- dB scale: Shows dB values on linear Y-axis ✓
- Linear scale: Shows linear amplitude values on linear Y-axis ✓
- No double-log confusion ✓

---

### 3. ✓ Added X-axis Log Scale Option

**Problem:** No way to view spectrum with logarithmic frequency axis.

**Use Case:**
- Wide frequency range (e.g., 1-250 Hz)
- Need detail in low frequencies (1-10 Hz)
- Linear scale compresses low frequencies

**Solution:** Added separate X-axis scale control

**Implementation:**
```python
# X-axis scale selection
if self.spectrum_x_scale == 'log':
    self.spectrum_ax.set_xscale('log')
    # Handle log(0) issue
    if self.freq_min == 0:
        self.spectrum_ax.set_xlim(left=max(0.1, plot_freqs[1]))
else:
    self.spectrum_ax.set_xscale('linear')
```

**Special Handling:**
- Log scale cannot display 0 frequency
- Automatically starts from 0.1 Hz or first non-zero frequency
- Prevents matplotlib errors

**Result:** Log frequency axis works correctly ✓

---

## Scale Combinations

All 4 combinations now work correctly:

### 1. dB + Linear Frequency (Default)
```
Y: dB values [-200 to 0 dB range]
X: Linear frequency [0 to Nyquist Hz]
Use: Standard seismic QC
```

### 2. dB + Log Frequency
```
Y: dB values [-200 to 0 dB range]
X: Log frequency [0.1 to Nyquist Hz]
Use: Wide frequency range, low freq detail
```

### 3. Linear + Linear Frequency
```
Y: Linear amplitude [0 to max]
X: Linear frequency [0 to Nyquist Hz]
Use: Direct amplitude comparison
```

### 4. Linear + Log Frequency
```
Y: Linear amplitude [0 to max]
X: Log frequency [0.1 to Nyquist Hz]
Use: Wide range with amplitude proportions visible
```

---

## Updated Control Panel

```
┌─────────────────────────────────────┐
│ Spectrum Display                    │
│                                     │
│ Y-axis (Amplitude):                 │
│   ● dB scale (20*log10)        ← FIX│
│   ○ Linear amplitude           ← FIX│
│                                     │
│ X-axis (Frequency):            ← NEW│
│   ● Linear frequency                │
│   ○ Log frequency                   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Data Colormap                       │
│   Colormap: [grayscale ▼]      ← FIX│
│                                     │
│   Options:                          │
│   • seismic                         │
│   • grayscale  ← Fixed              │
│   • viridis                         │
│   • plasma                          │
│   • jet                             │
│   • inferno                         │
└─────────────────────────────────────┘
```

---

## Technical Details

### dB to Linear Conversion
```python
# Correct formula
linear_amplitude = 10 ** (dB / 20.0)

# Examples:
   0 dB →   1.0 linear
  20 dB →  10.0 linear
  40 dB → 100.0 linear
 -20 dB →   0.1 linear
```

### Log Frequency Handling
```python
# Problem: Can't compute log(0)
frequencies = [0, 1, 2, 3, ..., 250]

# Solution: Start from first non-zero
if freq_min == 0 and x_scale == 'log':
    start_freq = max(0.1, frequencies[1])
    ax.set_xlim(left=start_freq)
```

### Colormap Matching
```python
# ISA sends:        Viewer expects:
'seismic'     →     'seismic'      ✓
'grayscale'   →     'grayscale'    ✓ (was 'gray' ✗)
'viridis'     →     'viridis'      ✓
'plasma'      →     'plasma'       ✓
'jet'         →     'jet'          ✓
'inferno'     →     'inferno'      ✓
```

---

## Testing

All fixes verified:

```bash
python test_isa_fixes.py
```

**Test Results:**
- ✓ Colormap names match viewer
- ✓ dB scale uses linear Y-axis
- ✓ Linear scale uses linear Y-axis
- ✓ dB to linear conversion correct
- ✓ Log frequency handles zero correctly
- ✓ All 4 scale combinations work

---

## Files Modified

### views/isa_window.py
**Changes:**
1. Fixed colormap names: 'gray' → 'grayscale'
2. Renamed variables: `spectrum_scale` → `spectrum_y_scale`, added `spectrum_x_scale`
3. Updated radio buttons: "Linear (dB)" → "dB scale (20*log10)"
4. Added X-axis log scale controls
5. Fixed `_update_spectrum()` logic:
   - Removed incorrect `set_yscale('log')` call
   - Added `set_xscale('log')` for X-axis
   - Correct dB/linear conversion

**Lines changed:** ~50 lines

---

## Comparison: Before vs After

| Feature | Before | After |
|---------|--------|-------|
| **Colormaps** | 'gray' (broken) | 'grayscale' (works) ✓ |
| **Y-axis dB** | dB + linear axis | dB + linear axis ✓ |
| **Y-axis Linear** | dB + log axis (wrong!) | Linear amp + linear axis ✓ |
| **X-axis options** | Linear only | Linear or Log ✓ |
| **Scale clarity** | Confusing labels | Clear labels ✓ |

---

## Usage Examples

### Example 1: Standard Seismic QC
```
Y-axis: dB scale (20*log10)
X-axis: Linear frequency
Result: Traditional dB spectrum, 0-Nyquist Hz
```

### Example 2: Low Frequency Detail
```
Y-axis: dB scale (20*log10)
X-axis: Log frequency
Result: More detail in 1-10 Hz range
```

### Example 3: Direct Amplitude
```
Y-axis: Linear amplitude
X-axis: Linear frequency
Result: See actual amplitude values, not dB
```

### Example 4: Wide Range Analysis
```
Y-axis: Linear amplitude
X-axis: Log frequency
Result: View 1-250 Hz with proportional amplitudes
```

---

## Key Takeaways

1. **dB is not a scale type** - it's an amplitude representation
   - dB uses linear axis
   - Don't apply log axis to dB values

2. **Linear amplitude ≠ linear scale** - but both use linear axis
   - Linear amplitude: actual values (not dB)
   - Still plotted on linear axis

3. **Log scale applies to frequency (X-axis)**
   - Expands low frequencies
   - Compresses high frequencies
   - Requires special handling for zero

4. **Colormap names must match exactly**
   - Case sensitive
   - 'gray' ≠ 'grayscale'

---

## Conclusion

All three issues fixed:
1. ✓ Grayscale colormap works
2. ✓ Linear Y-axis is truly linear
3. ✓ X-axis log scale added

The ISA window now provides correct and flexible spectrum visualization options for seismic QC workflows.
