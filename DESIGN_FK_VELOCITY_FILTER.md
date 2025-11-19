# FK Velocity Filter - Proper Design

## Theory

### Velocity in FK Domain

For a linear event with apparent velocity `v`:
```
f = v × |k|
```

Where:
- `f` = temporal frequency (Hz)
- `k` = spatial wavenumber (cycles/distance_unit)
- `v` = apparent velocity (distance_unit/s)
- `|k|` = absolute value (creates symmetric wedge)

### Velocity Wedges

In FK space, constant velocity creates a **wedge** (V-shape):
- Vertex at origin (0, 0)
- Slopes: ±v
- For positive k: f = v × k
- For negative k: f = v × |k| = v × (-k)

### Filter Zones

**Pass Mode** (keep velocities v_min ≤ v ≤ v_max):

```
        f
        ↑
        |     REJECT (v > v_max)
        |    /   \
   v_max|   /     \
        |  /  PASS \
   v_min| /  ZONE  \
        |/ REJECT   \
        +------------→ k
```

**Regions:**
1. **Inside v_max wedge**: v > v_max → REJECT (too fast)
2. **Between wedges**: v_min ≤ v ≤ v_max → PASS (keep)
3. **Outside v_min wedge**: v < v_min → REJECT (too slow)

### Taper Zones

Apply **cosine taper** for smooth transition:

```
v_min zone:
  v1 = v_min - taper_width
  v2 = v_min + taper_width

  v < v1:           weight = 0 (full reject)
  v1 ≤ v < v_min:   weight = 0 → 0.5 (taper up)
  v_min ≤ v < v2:   weight = 0.5 → 1 (taper up)
  v ≥ v2:           weight = 1 (full pass)
```

## Implementation

### 1. Calculate Apparent Velocity at Each FK Point

```python
# For each point (f, k) in FK grid:
v_app = |f| / |k|  (with k != 0)

# Handle special cases:
if k = 0: v_app = ∞ (horizontal event)
if f = 0: v_app = 0 (DC component)
```

### 2. Create Filter Weights

For **Pass mode** with v_min and v_max enabled:

```python
# Initialize
weights = np.ones_like(v_app)

# Define taper boundaries
v1 = v_min - taper_width  # Full reject below
v2 = v_min + taper_width  # Full pass above
v3 = v_max - taper_width  # Full pass below
v4 = v_max + taper_width  # Full reject above

# Apply v_min cutoff
weights[v_app < v1] = 0.0  # Full reject

# Taper zone for v_min
mask = (v_app >= v1) & (v_app < v2)
weights[mask] = 0.5 * (1 - cos(π * (v_app - v1) / (2 * taper_width)))

# Apply v_max cutoff
weights[v_app > v4] = 0.0  # Full reject

# Taper zone for v_max
mask = (v_app > v3) & (v_app <= v4)
weights[mask] = 0.5 * (1 + cos(π * (v_app - v3) / (2 * taper_width)))

# Passband v2 ≤ v ≤ v3: weights = 1 (already initialized)
```

### 3. Visualization - Draw Boundary Lines

For **displaying** filter boundaries on FK plot:

```python
# Create dense k array for smooth lines
k_display = np.linspace(k_min, k_max, 500)

# For each velocity boundary:
for v in [v1, v2, v_min, v3, v4, v_max]:
    # Positive k side
    k_pos = k_display[k_display >= 0]
    f_pos = v * k_pos
    plot(k_pos, f_pos)

    # Negative k side (mirror)
    k_neg = k_display[k_display <= 0]
    f_neg = v * np.abs(k_neg)
    plot(k_neg, f_neg)
```

**NOT:**
```python
# WRONG - only 2 points!
k_range = [k_min, k_max]
```

## Correct Code Structure

### Filter Calculation (processors/fk_filter.py)

```python
def _create_velocity_filter(self, f_grid, k_grid):
    """Create velocity filter weights on FK grid."""

    # Calculate apparent velocity at each point
    with np.errstate(divide='ignore', invalid='ignore'):
        v_app = np.abs(f_grid) / np.abs(k_grid)
        v_app[k_grid == 0] = np.inf

    # Initialize weights
    if self.mode == 'pass':
        weights = np.ones_like(v_app)
    else:
        weights = np.zeros_like(v_app)

    # Define taper zones
    v1 = self.v_min - self.taper_width
    v2 = self.v_min + self.taper_width
    v3 = self.v_max - self.taper_width
    v4 = self.v_max + self.taper_width

    if self.mode == 'pass':
        # Reject below v1
        weights[v_app < v1] = 0.0

        # Taper v1 to v2
        mask = (v_app >= v1) & (v_app < v2)
        weights[mask] = 0.5 * (1.0 - np.cos(
            np.pi * (v_app[mask] - v1) / (2 * self.taper_width)
        ))

        # Reject above v4
        weights[v_app > v4] = 0.0

        # Taper v3 to v4
        mask = (v_app > v3) & (v_app <= v4)
        weights[mask] = 0.5 * (1.0 + np.cos(
            np.pi * (v_app[mask] - v3) / (2 * self.taper_width)
        ))

    # Preserve DC
    weights[0, 0] = 1.0

    return weights
```

### Line Drawing (views/fk_designer_dialog.py)

```python
def _draw_velocity_lines(self, freqs, wavenumbers):
    """Draw velocity boundary lines for visualization."""

    # Create dense k array for smooth lines (500 points)
    k_min, k_max = wavenumbers.min(), wavenumbers.max()
    k_display = np.linspace(k_min, k_max, 500)
    f_max = freqs.max()

    # Define velocities to draw
    velocities = []

    if self.taper_width > 0 and self.v_min_enabled:
        v1 = self.v_min - self.taper_width
        if v1 > 0:
            velocities.append((v1, 'gray', Qt.PenStyle.DashLine))
        v2 = self.v_min + self.taper_width
        velocities.append((v2, 'gray', Qt.PenStyle.DashLine))

    if self.v_min_enabled:
        velocities.append((self.v_min, 'yellow', Qt.PenStyle.SolidLine))

    if self.taper_width > 0 and self.v_max_enabled:
        v3 = self.v_max - self.taper_width
        velocities.append((v3, 'gray', Qt.PenStyle.DashLine))
        v4 = self.v_max + self.taper_width
        if v4 > 0:
            velocities.append((v4, 'gray', Qt.PenStyle.DashLine))

    if self.v_max_enabled:
        velocities.append((self.v_max, 'yellow', Qt.PenStyle.SolidLine))

    # Draw each velocity line
    for v, color, style in velocities:
        if v <= 0:
            continue

        # Positive k: f = v * k
        k_pos = k_display[k_display >= 0]
        f_pos = v * k_pos
        f_pos = np.clip(f_pos, 0, f_max)

        pen = pg.mkPen(color, width=2, style=style)
        self.fk_plot.plot(k_pos, f_pos, pen=pen)

        # Negative k: f = v * |k|
        k_neg = k_display[k_display <= 0]
        f_neg = v * np.abs(k_neg)
        f_neg = np.clip(f_neg, 0, f_max)

        self.fk_plot.plot(k_neg, f_neg, pen=pen)
```

## Validation

### Test Case 1: Simple Pass Filter

```
Data: dx = 25 m, n_traces = 100
k_nyquist = 1/(2*25) = 0.02 cycles/m = 20 mcycles/m

Filter: v_min = 1500 m/s, v_max = 6000 m/s, taper = 300 m/s

At k = 0.02 cycles/m:
  v_min line: f = 1500 × 0.02 = 30 Hz
  v_max line: f = 6000 × 0.02 = 120 Hz

Expected: Yellow lines at f=30Hz and f=120Hz at k_max
```

### Test Case 2: Ground Roll Removal

```
Data: dx = 220 ft, k_nyquist = 0.00227 cycles/ft ≈ 2.27 mcycles/ft

Filter: v_min = 1500 ft/s (reject below), taper = 300 ft/s

At k = 0.002 cycles/ft:
  v_min line: f = 1500 × 0.002 = 3 Hz
  Taper starts at: v1 = 1200 ft/s → f = 2.4 Hz

Expected: Yellow line at ~3 Hz, gray dashed at ~2.4 Hz
```

## Key Fixes Needed

1. **Line drawing**: Use `np.linspace(k_min, k_max, 500)` not `[k_min, k_max]`
2. **Apparent velocity**: Use `|f| / |k|` not `f / k`
3. **Taper formula**: Verify cosine taper is correctly applied
4. **Units**: Ensure v, k, f all in consistent units (no conversions!)

## Summary

The filter **calculation** appears correct in current code.
The **visualization** is broken - only draws 2 points instead of smooth lines.
Fix: Use proper k array with many points for line drawing.
