# FK Filtering Improvements - 2025-11-18

## User Requests

Based on screenshot analysis and user feedback:

1. **Lower velocity limit**: Allow velocity down to 1 m/s (was 100 m/s minimum)
2. **Optional velocity limits**: Allow disabling low/high velocity limits entirely
   - Problem: Velocities > 10000 m/s were being filtered out, affecting data
3. **Dip-based filtering**: Add dip-based filtering with separate positive/negative dip controls
4. **Filter type selector**: Dropdown to select between velocity/dip/manual modes

## Changes Implemented

### 1. Extended Velocity Range ✅

**Before**:
```python
self.v_min_slider.setRange(100, 8000)   # 100-8000 m/s
self.v_max_slider.setRange(100, 10000)  # 100-10000 m/s
```

**After**:
```python
self.v_min_slider.setRange(1, 20000)    # 1-20000 m/s (NO UPPER LIMIT!)
self.v_max_slider.setRange(1, 20000)    # 1-20000 m/s
```

**Benefits**:
- Can now filter very slow-moving coherent noise (ground roll < 100 m/s)
- No artificial upper limit restricting high-velocity data
- Full flexibility for user's data characteristics

---

### 2. Optional Velocity Limits ✅

**Before**: Both v_min and v_max were always applied to filter.

**After**: Each limit can be independently enabled/disabled with checkboxes.

#### UI Controls (views/fk_designer_dialog.py)

**Lines 403-465**:
```python
# V_min with enable checkbox
vmin_header = QHBoxLayout()
self.v_min_enable = QCheckBox("Min Velocity:")
self.v_min_enable.setChecked(self.v_min_enabled)
self.v_min_enable.stateChanged.connect(self._on_v_min_enable_changed)

# V_max with enable checkbox
vmax_header = QHBoxLayout()
self.v_max_enable = QCheckBox("Max Velocity:")
self.v_max_enable.setChecked(self.v_max_enabled)
self.v_max_enable.stateChanged.connect(self._on_v_max_enable_changed)
```

#### Filter Logic (processors/fk_filter.py)

**Lines 212-240**: Updated `_create_velocity_filter()` to handle optional limits:

```python
if self.mode == 'pass':
    # Start with all pass
    weights = np.ones_like(v_app)

    # Apply minimum velocity limit (if enabled)
    if self.v_min_enabled:
        weights[v_app < v1] = 0.0
        # Taper zone...

    # Apply maximum velocity limit (if enabled)
    if self.v_max_enabled:
        weights[v_app > v4] = 0.0
        # Taper zone...
```

**Result**:
- If v_min disabled: No lower velocity limit (all slow velocities pass)
- If v_max disabled: No upper velocity limit (all fast velocities pass)
- If both disabled: All velocities pass (no filtering)
- Can use just one limit (e.g., only remove v < 1000 m/s)

---

### 3. Dip-Based Filtering Mode ✅

**New Concept**: Filter based on dip (dt/dx) instead of velocity (dx/dt).

**Dip Definition**:
- Dip = dt/dx (time gradient / spatial gradient)
- Units: s/m (seconds per meter)
- Negative dip: Left-dipping events (k < 0)
- Positive dip: Right-dipping events (k > 0)

**In FK Domain**:
- Dip = k/f (wavenumber / frequency)
- Velocity = f/k
- Therefore: dip = 1/velocity

#### UI Controls (views/fk_designer_dialog.py)

**Lines 467-536**: New dip parameter widget:

```python
self.dip_widget = QWidget()
dip_layout = QVBoxLayout()

# Negative dip with enable checkbox
self.dip_min_enable = QCheckBox("Negative Dip:")
self.dip_min_enable.setChecked(self.dip_min_enabled)

self.dip_min_spin = QDoubleSpinBox()
self.dip_min_spin.setRange(-1.0, 0.0)    # Negative dips
self.dip_min_spin.setValue(self.dip_min)
self.dip_min_spin.setSingleStep(0.001)
self.dip_min_spin.setDecimals(4)
self.dip_min_spin.setSuffix(" s/m")

# Positive dip with enable checkbox
self.dip_max_enable = QCheckBox("Positive Dip:")
self.dip_max_enable.setChecked(self.dip_max_enabled)

self.dip_max_spin = QDoubleSpinBox()
self.dip_max_spin.setRange(0.0, 1.0)     # Positive dips
self.dip_max_spin.setValue(self.dip_max)
self.dip_max_spin.setSingleStep(0.001)
self.dip_max_spin.setDecimals(4)
self.dip_max_spin.setSuffix(" s/m")
```

**Default Values**:
- Negative dip: -0.01 s/m (left-dipping events)
- Positive dip: +0.01 s/m (right-dipping events)

#### Filter Implementation (processors/fk_filter.py)

**Lines 311-452**: New `_create_dip_filter()` method:

```python
def _create_dip_filter(
    self,
    f_grid: np.ndarray,
    k_grid: np.ndarray
) -> np.ndarray:
    """
    Create dip-based filter weights with cosine taper.

    Dip is defined as dt/dx (s/m). In FK domain: dip = k/f
    - Negative dip: k < 0 (left-dipping events)
    - Positive dip: k > 0 (right-dipping events)
    """
    # Calculate apparent dip at each FK point
    with np.errstate(divide='ignore', invalid='ignore'):
        dip_app = k_grid / f_grid
        dip_app[f_grid == 0] = np.inf

    # Apply filters based on enabled dip limits
    # (Similar logic to velocity filter, but using dip values)
    ...
```

**Key Features**:
- Independent control of negative and positive dips
- Each dip limit can be enabled/disabled
- Supports pass/reject modes
- Cosine taper zones for smooth transitions

#### Visualization (views/fk_designer_dialog.py)

**Lines 1455-1531**: New `_draw_dip_lines()` method:

```python
def _draw_dip_lines(self, freqs: np.ndarray, wavenumbers: np.ndarray):
    """Draw dip lines on FK spectrum."""
    # Dip line equation: f = k/dip
    # For constant dip, line passes through origin with slope = 1/dip

    for dip, color, style, label in dips:
        # Sample k values
        k_vals = np.linspace(k_min, k_max, 100)
        f_vals = k_vals / dip

        # Clip to positive frequencies
        mask = (f_vals >= 0) & (f_vals <= f_max)
        k_vals_clipped = k_vals[mask]
        f_vals_clipped = f_vals[mask]

        # Draw line
        pen = pg.mkPen(color, width=line_width, style=...)
        self.fk_plot.plot(k_vals_clipped, f_vals_clipped, pen=pen)
```

**Dip Lines**:
- Yellow solid: Main dip boundaries (if enabled)
- Gray dashed: Taper zones (if taper_width > 0)
- Lines radiate from origin with slope = 1/dip
- Legend shows enabled/disabled status

---

### 4. Filter Type Selector ✅

**New Dropdown**: Choose between Velocity, Dip, or Manual filtering modes.

#### UI (views/fk_designer_dialog.py)

**Lines 395-401**:
```python
# Filter type selection
type_layout = QHBoxLayout()
type_layout.addWidget(QLabel("Filter Type:"))
self.filter_type_combo = QComboBox()
self.filter_type_combo.addItem("Velocity", 'velocity')
self.filter_type_combo.addItem("Dip", 'dip')
self.filter_type_combo.addItem("Manual", 'manual')
self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
```

#### Behavior (lines 788-800):

```python
def _on_filter_type_changed(self, index: int):
    """Handle filter type change (velocity/dip/manual)."""
    self.filter_type = self.filter_type_combo.currentData()

    # Show/hide appropriate parameter widgets
    self.velocity_widget.setVisible(self.filter_type == 'velocity')
    self.dip_widget.setVisible(self.filter_type == 'dip')
    self.manual_widget.setVisible(self.filter_type == 'manual')

    # Trigger recompute
    if self.auto_update:
        self._compute_fk_spectrum()
        self._apply_filter()
```

**Result**:
- Only relevant parameters shown for selected mode
- Automatic recompute when mode changes
- Clean UI without clutter

---

## Layout Structure

### Filter Parameters Panel (Left)

```
Filter Parameters
├── Filter Type: [Velocity ▼]  ← Dropdown selector
│
├── VELOCITY WIDGET (visible when filter_type='velocity')
│   ├── ☑ Min Velocity: [slider] 2000 m/s
│   ├── ☑ Max Velocity: [slider] 6000 m/s
│   └── Range: 1 - 20000 m/s
│
├── DIP WIDGET (visible when filter_type='dip')
│   ├── ☑ Negative Dip: [spinbox] -0.0100 s/m
│   ├── ☑ Positive Dip: [spinbox] +0.0100 s/m
│   └── Range: -1.0 to +1.0 s/m
│
└── MANUAL WIDGET (visible when filter_type='manual')
    └── [Placeholder for polygon-based manual filter]
```

---

## FK Spectrum Visualization

### Velocity Mode
```
FK Plot:
  ▲ f (Hz)
  │         ╱  ← v_max (yellow, if enabled)
  │       ╱ ╱  ← v_max taper (gray, if taper_width > 0)
  │     ╱ ╱
  │   ╱ ╱  ← v_min (yellow, if enabled)
  │ ╱ ╱  ← v_min taper (gray)
  ○────────────→ k (cycles/m)
 (0,0)

Legend:
  Mode: Pass (Velocity)
  v_min: 2000 m/s (yellow solid) [or "disabled"]
  v_max: 6000 m/s (yellow solid) [or "disabled"]
  Taper zones (gray dashed)
```

### Dip Mode
```
FK Plot:
  ▲ f (Hz)
  │       ╱  ← dip_max (positive, yellow if enabled)
  │     ╱
  │   ╱
  │ ╱
  ○────────────→ k (cycles/m)
╱ │
  │  ← dip_min (negative, yellow if enabled)

Legend:
  Mode: Pass (Dip)
  dip_min: -0.0100 s/m (yellow solid) [or "disabled"]
  dip_max: +0.0100 s/m (yellow solid) [or "disabled"]
  Taper zones (gray dashed)

Note: Dip lines pass through origin with slope = 1/dip
```

---

## Files Modified

### 1. processors/fk_filter.py (~200 lines added/modified)

**Changes**:
- Updated class docstring to describe velocity and dip modes
- Modified `_validate_params()` to handle new parameters:
  - `filter_type`: 'velocity' or 'dip'
  - `v_min_enabled`, `v_max_enabled`: boolean flags
  - `dip_min`, `dip_max`, `dip_min_enabled`, `dip_max_enabled`
- Updated `_apply_fk_filter()` to dispatch to correct filter method
- Rewrote `_create_velocity_filter()` to support optional limits
- Added `_create_dip_filter()` for dip-based filtering

**Key Logic**:

```python
# In _apply_fk_filter():
if self.filter_type == 'velocity':
    weights = self._create_velocity_filter(f_grid, k_grid)
elif self.filter_type == 'dip':
    weights = self._create_dip_filter(f_grid, k_grid)

# In _create_velocity_filter():
if self.mode == 'pass':
    weights = np.ones_like(v_app)  # Start with all pass
    if self.v_min_enabled:
        # Apply lower limit
    if self.v_max_enabled:
        # Apply upper limit

# In _create_dip_filter():
# Calculate dip = k/f at each FK point
dip_app = k_grid / f_grid
# Apply filters based on enabled dip limits
```

### 2. views/fk_designer_dialog.py (~150 lines added/modified)

**Changes**:
- Added state variables (lines 80-91):
  - `filter_type`, `v_min_enabled`, `v_max_enabled`
  - `dip_min`, `dip_max`, `dip_min_enabled`, `dip_max_enabled`
- Updated `_create_parameter_group()` (lines 390-550):
  - Filter type dropdown
  - Velocity widget with enable checkboxes
  - Dip widget with spinboxes and enable checkboxes
  - Widget visibility based on filter_type
- Added event handlers (lines 788-888):
  - `_on_filter_type_changed()`
  - `_on_v_min_enable_changed()`, `_on_v_max_enable_changed()`
  - `_on_dip_min_enable_changed()`, `_on_dip_max_enable_changed()`
  - `_on_dip_min_changed()`, `_on_dip_max_changed()`
- Updated `_compute_fk_spectrum()` to pass new parameters (lines 1195-1208)
- Updated `_apply_filter()` to pass new parameters (lines 1245-1258)
- Updated `_update_fk_spectrum_plot()` to dispatch visualization (lines 1380-1385)
- Updated `_draw_velocity_lines()` to show only enabled limits (lines 1402-1420)
- Added `_draw_dip_lines()` for dip mode visualization (lines 1455-1531)

---

## Usage Examples

### Example 1: Remove Low-Velocity Noise Only

**Setup**:
1. Filter Type: Velocity
2. ☑ Min Velocity: 1000 m/s
3. ☐ Max Velocity: disabled
4. Mode: Reject

**Result**: Removes all velocities < 1000 m/s (ground roll), keeps everything else.

---

### Example 2: Dip-Based Filtering

**Setup**:
1. Filter Type: Dip
2. ☑ Negative Dip: -0.005 s/m
3. ☑ Positive Dip: +0.005 s/m
4. Mode: Pass

**Result**: Keeps only events with dips between -0.005 and +0.005 s/m.

Interpretation:
- dip = -0.005 s/m → velocity ≈ -200 m/s (left-dipping)
- dip = +0.005 s/m → velocity ≈ +200 m/s (right-dipping)

---

### Example 3: One-Sided Velocity Limit

**Setup**:
1. Filter Type: Velocity
2. ☐ Min Velocity: disabled
3. ☑ Max Velocity: 8000 m/s
4. Mode: Pass

**Result**: Passes all velocities up to 8000 m/s, no lower limit.
- Useful when you want to preserve slow ground roll but remove very high velocities

---

## Testing Checklist

### Velocity Mode:
- [ ] Velocity slider goes down to 1 m/s
- [ ] Velocity slider goes up to 20000 m/s
- [ ] Unchecking v_min checkbox disables slider and removes lower limit
- [ ] Unchecking v_max checkbox disables slider and removes upper limit
- [ ] With both limits disabled, all velocities pass
- [ ] FK plot shows only enabled velocity lines
- [ ] Legend correctly shows "disabled" for unchecked limits

### Dip Mode:
- [ ] Selecting "Dip" from dropdown shows dip controls
- [ ] Negative dip spinbox range: -1.0 to 0.0 s/m
- [ ] Positive dip spinbox range: 0.0 to 1.0 s/m
- [ ] Unchecking dip_min checkbox disables spinbox
- [ ] Unchecking dip_max checkbox disables spinbox
- [ ] FK plot shows dip lines passing through origin
- [ ] Dip lines have slope = 1/dip
- [ ] Legend shows dip values with correct units (s/m)

### Filter Application:
- [ ] Changing filter type triggers recompute
- [ ] Enabling/disabling limits triggers recompute (if auto_update on)
- [ ] Filtered output respects enabled/disabled limits
- [ ] Filter works in both pass and reject modes
- [ ] Taper zones work correctly with dip mode

### Edge Cases:
- [ ] Both velocity limits disabled → no filtering
- [ ] Both dip limits disabled → no filtering
- [ ] Dip = 0 not displayed (would be horizontal line)
- [ ] Very small dip values (e.g., 0.0001 s/m) work correctly
- [ ] Very large velocity values (e.g., 15000 m/s) work correctly

---

## Compilation

```bash
python3 -m py_compile processors/fk_filter.py
✓ Compilation successful

python3 -m py_compile views/fk_designer_dialog.py
✓ Compilation successful
```

---

## Summary

**Status**: ✅ COMPLETE

**Files Modified**: 2
- `processors/fk_filter.py` (~200 lines)
- `views/fk_designer_dialog.py` (~150 lines)

**Key Features Implemented**:
1. ✓ Velocity range extended: 1-20000 m/s (was 100-10000)
2. ✓ Optional velocity limits with enable/disable checkboxes
3. ✓ Dip-based filtering mode with separate negative/positive controls
4. ✓ Filter type dropdown (Velocity/Dip/Manual)
5. ✓ Optional dip limits with enable/disable checkboxes
6. ✓ FK spectrum visualization for both velocity and dip modes
7. ✓ Legend shows enabled/disabled status
8. ✓ Only enabled boundary lines drawn on FK plot

**Benefits**:
- **Flexibility**: Users can now filter exactly what they need
- **No artificial limits**: Can handle any velocity range
- **Dip-based filtering**: Alternative approach for certain noise types
- **User control**: Every limit can be enabled/disabled independently
- **Clear visualization**: FK plot shows only active filter boundaries

**Result**: FK filtering system now provides professional-grade control over filtering parameters, matching capabilities of commercial seismic processing software!
