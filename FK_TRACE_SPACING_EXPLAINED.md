# FK Filtering: Trace Spacing, Wavenumber, and Velocity Relationship

## Overview

Accurate trace spacing is **critical** for FK filtering because it directly affects:
1. Wavenumber (k) calculation
2. Velocity line positioning in FK domain
3. Filter accuracy and effectiveness

## How Trace Spacing is Calculated

### Method 1: From GroupX Coordinates (Most Accurate)

**Source**: SEG-Y header `GroupX` (receiver group X coordinate in meters)

**Calculation**:
```python
# Extract GroupX for all traces in gather
group_x = gather_headers['GroupX'].values  # [100, 125, 150, 175, ...]

# Calculate spacing between consecutive traces
spacings = np.abs(np.diff(group_x))  # [25, 25, 25, ...]

# Use median (robust to outliers)
median_spacing = np.median(spacings[spacings > 0])  # 25.0 meters

# Sanity check (reject if unreasonable)
if 0 < median_spacing < 1000:
    trace_spacing = median_spacing
```

**Why Median?**
- Handles irregular trace spacing
- Robust to missing traces or bad data
- Common in real seismic data

**Example**:
```
Trace:    1     2     3     4     5     6
GroupX:  100   125   150   180   205   230
Spacing:    25    25    30    25    25
Median:  25 meters ✓
```

### Method 2: From d3 Header (Explicit Spacing)

**Source**: SEG-Y header `d3` (trace spacing in meters)

**Calculation**:
```python
# Read trace spacing from first trace header
d3 = gather_headers['d3'].iloc[0]  # e.g., 25.0

if 0 < d3 < 1000:
    trace_spacing = d3
```

**When Used**:
- If GroupX not available
- If d3 header is populated correctly
- Common in processed data

### Method 3: Default Fallback

**Default**: 25.0 meters (typical for land seismic)

**Used When**:
- No GroupX coordinates
- No d3 header
- Header values out of range
- Any calculation error

### Sub-Gather Specific Spacing

For sub-gathers (when boundary header changes):

```python
# Each sub-gather may have different spacing
for subgather in subgathers:
    # Extract headers for just this sub-gather
    sg_headers = headers_df[start:end]

    # Calculate spacing for this specific sub-gather
    sg_spacing = calculate_subgather_trace_spacing(sg_headers, subgather)

    # Use sub-gather specific spacing for FK filter
    processor = FKFilter(trace_spacing=sg_spacing)
```

**Why Sub-Gather Specific?**
- Different receiver lines may have different spacing
- Irregular geometry (marine, 3D)
- Accurate FK filtering requires local spacing

---

## FK Domain Mathematics

### Wavenumber Calculation

From `fft.fftfreq()` in NumPy:

```python
wavenumbers = fft.fftfreq(n_traces, trace_spacing)
```

**Formula**:
```
k[i] = i / (n_traces * trace_spacing)  for i = 0, 1, ..., n_traces/2
k[i] = (i - n_traces) / (n_traces * trace_spacing)  for i = n_traces/2+1, ..., n_traces-1
```

**Units**: cycles/meter (or 1/meter)

**Nyquist Wavenumber**:
```
k_nyquist = 1 / (2 * trace_spacing)
```

**Example** (trace_spacing = 25m):
```
k_nyquist = 1 / (2 * 25) = 0.02 cycles/m
```

### Velocity-Wavenumber-Frequency Relationship

**Fundamental Equation**:
```
v = f / k

where:
  v = apparent velocity (m/s)
  f = temporal frequency (Hz)
  k = spatial wavenumber (cycles/m)
```

**In FK Domain**:
```
f = v * k    (for positive k)
f = v * |k|  (for negative k, creates V-shape)
```

**Example**:
```
Velocity v = 2000 m/s
Wavenumber k = 0.01 cycles/m
Frequency f = 2000 * 0.01 = 20 Hz
```

### Velocity Lines in FK Plot

For a given velocity `v`, the line in FK domain:

```python
# Positive k side
k_pos = np.linspace(0, k_max, 100)
f_pos = v * k_pos

# Negative k side (mirror)
k_neg = np.linspace(k_min, 0, 100)
f_neg = v * np.abs(k_neg)
```

**Result**: V-shaped line with apex at (k=0, f=0)

---

## Impact of Incorrect Trace Spacing

### Example: Actual spacing = 25m, but assumed = 50m

**Wavenumber Error**:
```
Correct:  k_nyquist = 1/(2*25) = 0.020 cycles/m
Wrong:    k_nyquist = 1/(2*50) = 0.010 cycles/m
Error:    50% too low!
```

**Velocity Error**:
```
For f = 20 Hz, k = 0.01 cycles/m:

Correct velocity:  v = 20 / 0.01 = 2000 m/s
Displayed (wrong): v = 20 / 0.005 = 4000 m/s (2x too high!)
```

**Filter Error**:
```
Design filter for 1000-3000 m/s ground roll:
- Actually filters 500-1500 m/s (wrong!)
- Removes signal instead of noise!
```

### Visual Impact

**Correct Spacing**:
```
FK Plot:
  ▲ f (Hz)
  │         ╱  ← 3000 m/s
  │       ╱
  │     ╱  ← 1000 m/s
  │   ╱
  │ ╱
  ○────────────→ k (cycles/m)
 Origin (0,0)

Velocity lines radiate from origin (0,0) with slope v
```

**Wrong Spacing (2x too large)**:
```
FK Plot:
  ▲ f (Hz)
  │               ╱  ← Appears as 6000 m/s (actually 3000)
  │             ╱
  │         ╱  ← Appears as 2000 m/s (actually 1000)
  │       ╱
  │   ╱
  ○────────────→ k (cycles/m)
 Origin (0,0)

Everything compressed horizontally!
k-axis range is halved, so slopes appear doubled
```

---

## Verification Methods

### 1. Check Header Values

```python
# In Python/IPython
import pandas as pd

# Load gather headers
headers = gather_navigator.get_current_gather()[1]

# Check GroupX
print("GroupX values:", headers['GroupX'].values[:10])
print("Spacings:", np.diff(headers['GroupX'].values[:10]))

# Check d3
if 'd3' in headers.columns:
    print("d3 value:", headers['d3'].iloc[0])
```

### 2. Verify FK Spectrum

**Known Ground Roll Velocity**: ~500-1500 m/s typically

**Check**:
1. Open FK Designer
2. Look at low-frequency, low-wavenumber energy
3. Check if velocity lines match expected ground roll velocities
4. If lines are at wrong velocities → spacing might be wrong

### 3. Cross-Check with Geometry

**Method**:
```python
# For common shot gather
# Receivers typically at regular intervals
# Check acquisition geometry:
receiver_interval = 25m  # From survey parameters

# Should match calculated spacing
if abs(trace_spacing - receiver_interval) > 1.0:
    print("WARNING: Spacing mismatch!")
```

---

## Common Trace Spacing Values

### Land Seismic
- **Typical**: 25m (most common)
- **High resolution**: 12.5m
- **3D**: 25-50m (inline and crossline may differ)

### Marine Seismic
- **Typical**: 12.5m or 25m
- **Streamer**: Usually regular spacing
- **OBC**: May be irregular

### Special Cases
- **Irregular geometry**: Use median spacing
- **Missing traces**: Spacing may vary
- **Multi-component**: Check each component

---

## Best Practices

### 1. Always Verify Spacing

```python
# Print spacing when opening FK Designer
print(f"Using trace spacing: {trace_spacing:.2f} m")
print(f"Nyquist wavenumber: {1/(2*trace_spacing):.4f} cycles/m")
```

### 2. Use Sub-Gathers for Irregular Geometry

```python
# If geometry changes (e.g., receiver line changes)
use_subgathers = True
boundary_header = 'ReceiverLine'

# Each sub-gather gets its own spacing calculation
```

### 3. Sanity Check Velocities

```python
# Expected velocities for different waves:
Ground roll: 300-1500 m/s
Refraction:  1500-3000 m/s
Reflection:  2000-6000 m/s
Air wave:    330 m/s

# If FK plot shows ground roll at 5000 m/s → spacing is wrong!
```

### 4. Add Manual Override (Future Enhancement)

```python
# Allow user to manually specify spacing if auto-detection fails
trace_spacing = user_input or auto_calculated_spacing
```

---

## Code Flow Summary

### Full Gather FK Filtering

```python
# 1. Calculate trace spacing
trace_spacing = _get_trace_spacing()
# → Tries GroupX, then d3, then default 25m

# 2. Pass to FK filter
processor = FKFilter(trace_spacing=trace_spacing)

# 3. Compute wavenumbers
wavenumbers = fft.fftfreq(n_traces, trace_spacing)
# → k = [0, 1/L, 2/L, ..., k_nyquist, -k_nyquist, ...]
# where L = n_traces * trace_spacing (aperture length)

# 4. Plot velocity lines
f = v * |k|  # Using calculated wavenumbers
```

### Sub-Gather FK Filtering

```python
# 1. Detect sub-gathers based on header changes
subgathers = detect_subgathers(headers, 'ReceiverLine')

# 2. For EACH sub-gather:
for sg in subgathers:
    # Calculate spacing for THIS sub-gather
    sg_spacing = calculate_subgather_trace_spacing(headers, sg)

    # Apply FK filter with sub-gather specific spacing
    processor = FKFilter(trace_spacing=sg_spacing)

    # Each sub-gather may have different k-axis!
```

---

## Mathematical Proof

### Why trace_spacing affects wavenumber:

**Spatial sampling theorem** (analogous to Nyquist in time):

```
Δx = trace_spacing (spatial sample interval)
k_nyquist = 1 / (2 * Δx)  (spatial Nyquist)

For Δx = 25m:
  k_nyquist = 1/(2*25) = 0.02 cycles/m
  λ_min = 1/k_nyquist = 50m (minimum resolvable wavelength)

For Δx = 12.5m:
  k_nyquist = 1/(2*12.5) = 0.04 cycles/m
  λ_min = 25m (better spatial resolution!)
```

### Why velocity depends on correct k:

```
Phase velocity:  v_p = ω/k = 2πf/k

If k is wrong by factor α:
  k_wrong = α * k_true
  v_wrong = 2πf/(α*k_true) = v_true/α

Example: α = 2 (spacing 2x too large)
  v_wrong = v_true/2  (velocity appears half!)
```

---

## Troubleshooting

### Problem: Velocity lines don't match expected velocities

**Cause**: Incorrect trace spacing

**Solutions**:
1. Check GroupX header: `print(headers['GroupX'])`
2. Check d3 header: `print(headers['d3'])`
3. Verify with acquisition geometry
4. Manually override if needed

### Problem: FK spectrum looks stretched/compressed

**Cause**: Wrong trace spacing → wrong k-axis scaling

**Solutions**:
1. Recalculate spacing from headers
2. Check for irregular geometry
3. Use sub-gathers if spacing varies

### Problem: Filter removes wrong velocities

**Cause**: Velocity lines plotted at wrong positions due to spacing error

**Solutions**:
1. Verify trace spacing before designing filter
2. Test on synthetic data with known velocities
3. Compare with other processing software

---

## Summary

**Trace Spacing Calculation**:
```
Priority:
1. GroupX coordinates (median of differences) ← Most accurate
2. d3 header (explicit spacing)
3. Default 25m (fallback)
```

**Critical Relationships**:
```
k = fft.fftfreq(n_traces, trace_spacing)
k_nyquist = 1 / (2 * trace_spacing)
v = f / k
```

**Impact on FK Filtering**:
- Correct spacing → accurate velocity lines → effective filtering ✅
- Wrong spacing → wrong velocities → filtering wrong components ❌

**Best Practice**: Always verify calculated spacing matches acquisition geometry!
