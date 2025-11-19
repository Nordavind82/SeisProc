# Universal Units System Design

## Problem
Current system mixes units:
- SEGY coordinates in feet/meters (unknown)
- Code assumes meters after scalar
- Velocities stored in ft/s but used as m/s
- FK wavenumbers in cycles/m regardless of data units
- Unit conversions at display time cause errors

## Solution: Native Units Throughout

### Principle
**ALL calculations use the SAME units as the SEGY data. NO conversions.**

---

## Implementation

### 1. SEGY Import - Unit Specification

**File:** `views/segy_import_dialog.py`

Add unit selection dropdown:
```python
# In _create_file_selection_group():
self.coord_units_combo = QComboBox()
self.coord_units_combo.addItems(['meters', 'feet'])
self.coord_units_combo.setToolTip("Unit of coordinates AFTER applying scalar")
```

Store in SeismicData metadata:
```python
metadata = {
    'coordinate_units': self.coord_units_combo.currentText(),  # 'meters' or 'feet'
    'segy_file': segy_path,
    ...
}
```

### 2. SeismicData Model

**File:** `models/seismic_data.py`

Metadata already supports arbitrary keys. Add helper:
```python
@property
def coordinate_units(self) -> str:
    """Get coordinate units ('meters' or 'feet')."""
    return self.metadata.get('coordinate_units', 'meters')

@property
def unit_symbol(self) -> str:
    """Get unit symbol ('m' or 'ft')."""
    return 'm' if self.coordinate_units == 'meters' else 'ft'
```

### 3. Trace Spacing Calculation

**File:** `utils/trace_spacing.py`

**NO CHANGES** - already returns spacing in native units (whatever SEGY has after scalar)

Just document:
```python
def calculate_trace_spacing_with_stats(...):
    """
    Calculate trace spacing in NATIVE units (meters or feet, depending on SEGY).

    Returns:
        TraceSpacingStats with spacing in same units as input coordinates
    """
```

### 4. FK Filter - Remove Unit Conversions

**File:** `processors/fk_filter.py`

**NO CHANGES NEEDED** - already uses trace_spacing as-is:
```python
wavenumbers = fft.fftfreq(n_traces, trace_spacing)
# If trace_spacing in feet → wavenumbers in cycles/foot
# If trace_spacing in meters → wavenumbers in cycles/meter
```

Already correct!

### 5. FK Designer Dialog - Critical Fixes

**File:** `views/fk_designer_dialog.py`

#### A. Store velocities in NATIVE units

**CURRENT (BROKEN):**
```python
self.v_min = 2000.0  # Assumed m/s, but slider might be ft/s
format_velocity(self.v_min)  # Converts m/s → display units
```

**FIX:**
```python
# Velocities stored in NATIVE units (same as data)
self.v_min = 2000.0  # In whatever units data uses
# NO conversion functions!
```

#### B. Remove format_velocity() calls

**Lines to change:**
- Line 450: `self.v_min_label.setText(format_velocity(self.v_min))`
- Line 469: `self.v_max_label.setText(format_velocity(self.v_max))`
- Line 546: `self.taper_label.setText(format_velocity(self.taper_width))`
- Line 835: `self.v_min_label.setText(format_velocity(self.v_min))`
- Line 852: `self.v_max_label.setText(format_velocity(self.v_max))`
- Line 899: `self.taper_label.setText(format_velocity(self.taper_width))`
- Line 1455: `legend_text += f"v_min: {format_velocity(self.v_min)}`
- Line 1459: `legend_text += f"v_max: {format_velocity(self.v_max)}`

**Replace with:**
```python
unit = self.working_data.unit_symbol
self.v_min_label.setText(f"{self.v_min:.0f} {unit}/s")
```

#### C. Fix trace spacing display

**Lines 1025, 1862:** Hardcoded "m"
```python
# BEFORE:
f"Spacing: {stats.spacing:.1f}m"

# AFTER:
unit = self.working_data.unit_symbol
f"Spacing: {stats.spacing:.1f}{unit}"
```

#### D. Velocity line drawing

**Already correct!** Line 1453:
```python
f_pos = v * k_pos  # Both in same native units
```

This works if:
- v in ft/s, k in cycles/ft → f in Hz ✓
- v in m/s, k in cycles/m → f in Hz ✓

#### E. FK plot axis label

**Line 703:**
```python
# BEFORE:
self.fk_plot.setLabel('bottom', 'Wavenumber', units='cycles/m')

# AFTER:
unit = self.working_data.unit_symbol
self.fk_plot.setLabel('bottom', 'Wavenumber', units=f'cycles/{unit}')
```

### 6. Remove AppSettings.spatial_units

**File:** `models/app_settings.py`

**DELETE:**
- `spatial_units` preference
- `spatial_units_changed` signal
- All related methods

**Rationale:** Units are determined by DATA, not user preference.

### 7. Remove UnitConverter Usage

**Files to update:**
- `views/fk_designer_dialog.py` - Remove all `format_velocity()` calls
- Remove imports of `UnitConverter`, `format_velocity`, `format_distance`

**Keep UnitConverter for:**
- Manual conversions if user wants to switch units (future feature)
- But NOT for automatic conversions in calculations

---

## Migration Path

### Phase 1: Add Unit Specification (Immediate)
1. Add unit selector to SEGY import dialog
2. Store in metadata
3. Add helper properties to SeismicData

### Phase 2: Remove Conversions from FK (Critical Fix)
1. Replace all `format_velocity()` with direct formatting
2. Update axis labels to use native units
3. Fix trace spacing label

### Phase 3: Clean Up (Optional)
1. Remove unused AppSettings.spatial_units
2. Mark UnitConverter as deprecated for FK pipeline

---

## Testing

### Test Case 1: Feet Data
```
SEGY: npr3_field.sgy
Coordinates: 788173, 788393, ... (feet after scalar)
Spacing: 220 feet
User selects: "feet"

Expected:
- Trace spacing: "220 ft"
- Wavenumbers: "±7.5 mcycles/ft"
- Velocity 900 ft/s at k=0.0072 → f=6.5 Hz
- FK axis: "cycles/ft"
```

### Test Case 2: Meter Data
```
SEGY: [meter data]
Coordinates: meters after scalar
Spacing: 25 meters
User selects: "meters"

Expected:
- Trace spacing: "25 m"
- Wavenumbers: "±20 mcycles/m"
- Velocity 1500 m/s at k=0.020 → f=30 Hz
- FK axis: "cycles/m"
```

---

## Benefits

1. **Correctness:** No unit mixing bugs
2. **Simplicity:** No conversion logic
3. **Performance:** No runtime conversions
4. **Clarity:** Units explicit in labels
5. **Flexibility:** Works with any unit system

## Key Insight

**The data has natural units. Use them. Don't convert.**
