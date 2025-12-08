# FK Filter Unit Consistency Analysis & Implementation Tasks

**Generated:** 2025-12-07
**Status:** Critical Issues Found
**Scope:** Meters/Feet/Distance unit handling throughout FK filter implementation

---

## Executive Summary

The FK filter implementation has **CRITICAL UNIT CONSISTENCY BUGS** that will cause incorrect filtering results when working with feet-based SEGY data. All velocity values are hardcoded to meters/second with no conversion support, despite UI tooltips claiming "m/s or ft/s" support.

### Impact Assessment

| Scenario | Current Behavior | Correct Behavior |
|----------|------------------|------------------|
| Data in meters | Works correctly | Works correctly |
| Data in feet | **BROKEN** - velocities wrong by 3.28x | Should convert velocities |
| Mixed units config | **BROKEN** - config may be wrong units | Should store/convert units |
| Sub-gather spacing | **BROKEN** - scalars not applied | Should apply SEGY scalars |

---

## Bug Summary Table

| ID | Issue | Severity | File | Lines |
|----|-------|----------|------|-------|
| B1 | Sub-gather spacing ignores SEGY scalars | **CRITICAL** | `utils/subgather_detector.py` | 147-152 |
| B2 | Velocity values hardcoded to m/s | **CRITICAL** | `processors/fk_filter.py` | 53-85 |
| B3 | Slider ranges assume metric only | **HIGH** | `views/fk_designer_dialog.py` | 490, 514 |
| B4 | Tooltips falsely claim ft/s support | **MEDIUM** | `views/fk_designer_dialog.py` | 492-519 |
| B5 | Wavenumber hardcoded to cycles/m | **HIGH** | `processors/fk_filter.py` | 151 |
| B6 | Dip calculation assumes meters | **MEDIUM** | `processors/fk_filter.py` | 397 |
| B7 | Unit conversion utilities unused | **MEDIUM** | `utils/unit_conversion.py` | 217-237 |
| B8 | Velocity label shows wrong unit | **MEDIUM** | `views/fk_designer_dialog.py` | 176-179 |
| B9 | Presets all metric only | **HIGH** | `processors/fk_filter.py` | 568-595 |
| B10 | Config doesn't store unit info | **MEDIUM** | `models/fk_config.py` | 43-45 |

---

## Detailed Bug Analysis

### B1: Sub-gather Spacing Ignores SEGY Scalars (CRITICAL)

**Location:** `utils/subgather_detector.py`, lines 147-152

**Current Code:**
```python
if 'GroupX' in sg_headers.columns and len(sg_headers) > 1:
    group_x = sg_headers['GroupX'].values
    spacings = np.abs(np.diff(group_x))  # ← NO SCALAR APPLIED!
    median_spacing = np.median(spacings[spacings > 0])
```

**Problem:** SEGY coordinates are often stored with scalars (e.g., scalar=-100 means divide by 100). Without applying the scalar, spacing could be off by 10x, 100x, or 1000x.

**Impact:** FK filter will use completely wrong trace spacing, resulting in:
- Incorrect wavenumber axis
- Velocity filter applied at wrong frequencies
- Filtered output contains wrong events

---

### B2: Velocity Values Hardcoded to m/s (CRITICAL)

**Location:** `processors/fk_filter.py`, lines 53-85

**Current Code:**
```python
# Line 24-25: Only m/s documented
v_min: Minimum velocity to pass (m/s)
v_max: Maximum velocity to pass (m/s)

# Line 555-557: Always shows m/s
def get_description(self) -> str:
    return f"FK Filter: {self.v_min:.0f}-{self.v_max:.0f} m/s"  # HARDCODED
```

**Problem:** When data is in feet, user enters "2000 ft/s" thinking that's what they get, but the filter actually uses 2000 m/s (≈6562 ft/s).

---

### B3: Slider Ranges Assume Metric (HIGH)

**Location:** `views/fk_designer_dialog.py`, lines 490, 514

**Current Code:**
```python
self.v_min_slider.setRange(1, 20000)  # 1-20000 m/s
self.v_max_slider.setRange(1, 20000)  # 1-20000 m/s
```

**Problem:**
- Maximum 20000 m/s ≈ 65617 ft/s (reasonable for metric)
- But if units are feet, user needs range up to ~65000 ft/s
- Current max of 20000 limits feet-mode to 20000 ft/s ≈ 6096 m/s

---

### B4: Tooltips Falsely Claim ft/s Support (MEDIUM)

**Location:** `views/fk_designer_dialog.py`, lines 492-495, 516-519

**Current Code:**
```python
self.v_min_slider.setToolTip(
    "Minimum velocity boundary (m/s or ft/s).\n"  # ← LIE
    ...
)
```

**Problem:** Documentation promises feature that doesn't exist.

---

### B5: Wavenumber Hardcoded to cycles/m (HIGH)

**Location:** `processors/fk_filter.py`, line 151

**Current Code:**
```python
wavenumbers = fft.fftfreq(n_traces, trace_spacing)  # cycles/m (ASSUMED)
```

**Problem:** If trace_spacing is in feet, wavenumber is actually cycles/feet, but all subsequent calculations assume cycles/m.

---

### B6-B10: Additional Issues

See detailed analysis in Bug Summary Table above.

---

## Implementation Tasks

### Phase 1: Critical Bug Fixes (Priority: IMMEDIATE)

#### Task 1.1: Fix Sub-gather Trace Spacing Scalar Application

**File:** `utils/subgather_detector.py`
**Effort:** 2 hours
**Risk:** Low

**Changes Required:**

```python
# Line 126-163: Update calculate_subgather_trace_spacing()

def calculate_subgather_trace_spacing(
    sg_headers: pd.DataFrame,
    default_spacing: float = 25.0
) -> float:
    """Calculate trace spacing for a sub-gather with proper SEGY scalar handling."""

    # Get scalar value (check multiple possible column names)
    scalar = 1.0
    for scalar_col in ['scalar_coord', 'scalco', 'ScalarCoord']:
        if scalar_col in sg_headers.columns:
            scalar_val = sg_headers[scalar_col].iloc[0]
            if scalar_val != 0:
                scalar = 1.0 / abs(scalar_val) if scalar_val < 0 else scalar_val
            break

    # Try GroupX with scalar
    if 'GroupX' in sg_headers.columns and len(sg_headers) > 1:
        group_x = sg_headers['GroupX'].values * scalar  # ← APPLY SCALAR
        spacings = np.abs(np.diff(group_x))
        median_spacing = np.median(spacings[spacings > 0])
        if median_spacing > 0:
            return float(median_spacing)

    # ... similar fixes for other coordinate sources
```

**Tests to Add:**
- [ ] Test with scalar_coord = -100 (centimeter coordinates)
- [ ] Test with scalar_coord = 1 (meter coordinates)
- [ ] Test with scalar_coord = 0 (no scaling)
- [ ] Test with missing scalar column

---

#### Task 1.2: Add Unit Awareness to FK Processor

**File:** `processors/fk_filter.py`
**Effort:** 4 hours
**Risk:** Medium

**Changes Required:**

```python
# Add to __init__ (around line 53):
def __init__(
    self,
    v_min: float = 1500.0,
    v_max: float = 4500.0,
    taper_width: float = 200.0,
    mode: str = 'pass',
    filter_type: str = 'velocity',
    coordinate_units: str = 'meters',  # NEW PARAMETER
    ...
):
    self.coordinate_units = coordinate_units

    # Convert velocities if in feet
    self._v_min_internal = self._to_metric_velocity(v_min)
    self._v_max_internal = self._to_metric_velocity(v_max)
    self._taper_internal = self._to_metric_velocity(taper_width)

def _to_metric_velocity(self, velocity: float) -> float:
    """Convert velocity to m/s for internal calculations."""
    if self.coordinate_units == 'feet':
        return velocity * 0.3048  # ft/s to m/s
    return velocity

def _to_display_velocity(self, velocity_ms: float) -> float:
    """Convert m/s velocity to display units."""
    if self.coordinate_units == 'feet':
        return velocity_ms / 0.3048  # m/s to ft/s
    return velocity_ms
```

**Update get_description (line 555):**
```python
def get_description(self) -> str:
    unit = 'ft' if self.coordinate_units == 'feet' else 'm'
    v_min_display = self._to_display_velocity(self._v_min_internal)
    v_max_display = self._to_display_velocity(self._v_max_internal)
    return f"FK Filter: {v_min_display:.0f}-{v_max_display:.0f} {unit}/s"
```

---

#### Task 1.3: Update FK Designer Dialog for Unit Support

**File:** `views/fk_designer_dialog.py`
**Effort:** 6 hours
**Risk:** Medium

**Changes Required:**

1. **Add unit detection on data load (around line 200):**
```python
def _load_data(self):
    ...
    self.coordinate_units = self.working_data.coordinate_units
    self._update_velocity_ranges()
```

2. **Dynamic slider ranges (new method):**
```python
def _update_velocity_ranges(self):
    """Update velocity slider ranges based on coordinate units."""
    if self.coordinate_units == 'feet':
        max_velocity = 65000  # ~20000 m/s in ft/s
        default_taper_max = 6500  # ~2000 m/s in ft/s
    else:
        max_velocity = 20000  # m/s
        default_taper_max = 2000  # m/s

    self.v_min_slider.setRange(1, max_velocity)
    self.v_max_slider.setRange(1, max_velocity)
    self.taper_slider.setRange(0, default_taper_max)
```

3. **Update tooltips (lines 492-519):**
```python
def _update_tooltips(self):
    unit = 'ft' if self.coordinate_units == 'feet' else 'm'
    self.v_min_slider.setToolTip(
        f"Minimum velocity boundary ({unit}/s).\n"
        "In 'Pass' mode: Energy with apparent velocity below this is rejected.\n"
        "In 'Reject' mode: Energy with apparent velocity below this is kept."
    )
```

4. **Fix velocity label (line 176-179):**
```python
def _get_velocity_label(self, velocity: float) -> str:
    """Get velocity label with correct units (no conversion needed - value is in display units)."""
    unit = 'ft' if self.coordinate_units == 'feet' else 'm'
    return f"{velocity:.0f} {unit}/s"
```

---

### Phase 2: Configuration & Persistence (Priority: HIGH)

#### Task 2.1: Add Unit Info to FKFilterConfig

**File:** `models/fk_config.py`
**Effort:** 2 hours
**Risk:** Low

**Changes Required:**

```python
@dataclass
class FKFilterConfig:
    name: str
    v_min: float
    v_max: float
    taper_width: float
    mode: str = 'pass'
    filter_type: str = 'velocity'
    coordinate_units: str = 'meters'  # NEW FIELD
    ...

    def to_processor_params(self, trace_spacing: float) -> Dict:
        """Convert to processor parameters, handling unit conversion if needed."""
        return {
            'v_min': self.v_min,
            'v_max': self.v_max,
            'taper_width': self.taper_width,
            'mode': self.mode,
            'trace_spacing': trace_spacing,
            'coordinate_units': self.coordinate_units,  # NEW
        }

    def convert_to_units(self, target_units: str) -> 'FKFilterConfig':
        """Create a copy with velocities converted to target units."""
        if target_units == self.coordinate_units:
            return self

        factor = 0.3048 if target_units == 'meters' else 1/0.3048
        return FKFilterConfig(
            name=self.name,
            v_min=self.v_min * factor,
            v_max=self.v_max * factor,
            taper_width=self.taper_width * factor,
            mode=self.mode,
            filter_type=self.filter_type,
            coordinate_units=target_units,
            ...
        )
```

**Migration:** Add version field to detect old configs without units.

---

#### Task 2.2: Update Preset Definitions

**File:** `processors/fk_filter.py`
**Effort:** 1 hour
**Risk:** Low

**Changes Required:**

```python
# Line 560-596: Store presets in metric, convert on use
FK_PRESETS_METRIC = {
    'Ground Roll Removal': {
        'v_min': 1500,  # m/s
        'v_max': 6000,  # m/s
        'taper_width': 300,  # m/s
        'mode': 'pass',
    },
    ...
}

def get_preset(name: str, coordinate_units: str = 'meters') -> Dict:
    """Get preset with values converted to specified units."""
    preset = FK_PRESETS_METRIC.get(name, {}).copy()
    if coordinate_units == 'feet':
        for key in ['v_min', 'v_max', 'taper_width']:
            if key in preset:
                preset[key] = preset[key] / 0.3048  # m/s to ft/s
    return preset
```

---

### Phase 3: Legacy Code Cleanup (Priority: MEDIUM)

#### Task 3.1: Remove Debug Output

**Files:** Multiple
**Effort:** 2 hours
**Risk:** Low

**Locations to Clean:**

| File | Lines | Action |
|------|-------|--------|
| `fk_filter.py` | 188-372 | Remove or add DEBUG flag |
| `fk_designer_dialog.py` | 85-100 | Remove or add DEBUG flag |
| `fk_designer_dialog.py` | 1343-1372 | Remove or add DEBUG flag |
| `fk_designer_dialog.py` | 1554-1627 | Remove or add DEBUG flag |
| `fk_designer_dialog.py` | 1982-2030 | Remove or add DEBUG flag |

**Implementation:**
```python
# Add at top of each file:
import os
DEBUG_FK = os.environ.get('SEISPROC_DEBUG_FK', '').lower() == 'true'

# Replace print statements:
if DEBUG_FK:
    print(f"Debug: ...")
```

---

#### Task 3.2: Use Existing Unit Conversion Utilities

**File:** `utils/unit_conversion.py`
**Effort:** 1 hour
**Risk:** Low

**Current Unused Functions:**
- `convert_wavenumber()` - lines 217-237
- `convert_velocity()` - should be added

**Action:**
1. Add `convert_velocity()` function
2. Update FK code to use these utilities instead of inline conversions

```python
# Add to utils/unit_conversion.py:
METERS_TO_FEET = 3.28084

def convert_velocity(velocity: float, from_units: str, to_units: str) -> float:
    """Convert velocity between m/s and ft/s."""
    if from_units == to_units:
        return velocity
    if from_units == 'meters' and to_units == 'feet':
        return velocity / 0.3048
    if from_units == 'feet' and to_units == 'meters':
        return velocity * 0.3048
    raise ValueError(f"Unknown units: {from_units} -> {to_units}")
```

---

#### Task 3.3: Standardize Coordinate Unit Detection

**File:** New utility or enhance existing
**Effort:** 3 hours
**Risk:** Medium

**Current Issues:**
- Unit detection scattered across multiple files
- Inconsistent metadata key names
- No validation

**Solution:** Create centralized unit detection:

```python
# utils/coordinate_units.py (new file)

def detect_coordinate_units(headers: pd.DataFrame, metadata: dict = None) -> str:
    """
    Detect coordinate units from SEGY headers or metadata.

    Priority:
    1. Explicit metadata setting
    2. SEGY byte 3255-3256 (Coordinate Units)
    3. Heuristic based on coordinate magnitudes
    4. Default to meters
    """
    # Check metadata first
    if metadata and 'coordinate_units' in metadata:
        return metadata['coordinate_units']

    # Check SEGY coordinate units field
    if 'CoordinateUnits' in headers.columns:
        cu = headers['CoordinateUnits'].iloc[0]
        if cu == 1:
            return 'meters'
        elif cu == 2:
            return 'feet'

    # Heuristic: large coordinates (>100000) likely feet or unscaled
    # Small coordinates (<10000) likely meters with scalar applied
    # This is imperfect but better than nothing

    return 'meters'  # Default
```

---

#### Task 3.4: Fix Manual Filter Mode Stub

**File:** `views/fk_designer_dialog.py`
**Effort:** Decision needed
**Risk:** Low

**Current State:** UI shows "Manual" option but feature not implemented.

**Options:**
1. **Remove from UI** (1 hour) - Hide until implemented
2. **Implement feature** (20+ hours) - Full polygon drawing support

**Recommendation:** Remove from UI for now:
```python
# Line 461: Comment out or remove
# self.filter_type_combo.addItem("Manual", 'manual')
```

---

### Phase 4: Testing & Validation (Priority: HIGH)

#### Task 4.1: Create Unit Test Suite for FK Units

**File:** `tests/test_fk_units.py` (new)
**Effort:** 4 hours

**Test Cases:**

```python
class TestFKUnitConsistency:

    def test_metric_velocity_passthrough(self):
        """Verify metric velocities pass through unchanged."""
        processor = FKFilter(v_min=1500, v_max=4500, coordinate_units='meters')
        assert processor._v_min_internal == 1500
        assert processor._v_max_internal == 4500

    def test_feet_velocity_conversion(self):
        """Verify feet velocities convert to metric internally."""
        processor = FKFilter(v_min=4921, v_max=14764, coordinate_units='feet')
        # 4921 ft/s ≈ 1500 m/s, 14764 ft/s ≈ 4500 m/s
        assert abs(processor._v_min_internal - 1500) < 1
        assert abs(processor._v_max_internal - 4500) < 1

    def test_subgather_spacing_with_scalar(self):
        """Verify sub-gather spacing applies SEGY scalar."""
        headers = pd.DataFrame({
            'GroupX': [100000, 102500, 105000],  # Centimeters
            'scalar_coord': [-100, -100, -100]   # Divide by 100
        })
        spacing = calculate_subgather_trace_spacing(headers)
        assert abs(spacing - 25.0) < 0.1  # Should be 25 meters

    def test_config_unit_conversion(self):
        """Verify config can convert between units."""
        config_m = FKFilterConfig(
            name='test', v_min=1500, v_max=4500,
            taper_width=300, coordinate_units='meters'
        )
        config_ft = config_m.convert_to_units('feet')
        assert abs(config_ft.v_min - 4921) < 1
        assert config_ft.coordinate_units == 'feet'
```

---

#### Task 4.2: Integration Test with Real SEGY Files

**File:** `tests/test_fk_integration.py`
**Effort:** 3 hours

**Test Cases:**
- [ ] Load metric SEGY, apply FK filter, verify results
- [ ] Load feet SEGY, apply FK filter, verify results
- [ ] Save config in metric, load in feet session, verify conversion
- [ ] Test all presets with both unit systems

---

## Implementation Order

### Week 1: Critical Fixes
1. [ ] Task 1.1: Fix sub-gather scalar application
2. [ ] Task 1.2: Add unit awareness to FK processor
3. [ ] Task 4.1: Create unit tests

### Week 2: UI & Config
4. [ ] Task 1.3: Update FK Designer for unit support
5. [ ] Task 2.1: Add unit info to FKFilterConfig
6. [ ] Task 2.2: Update preset definitions

### Week 3: Cleanup & Testing
7. [ ] Task 3.1: Remove debug output
8. [ ] Task 3.2: Use existing unit conversion utilities
9. [ ] Task 3.3: Standardize coordinate unit detection
10. [ ] Task 3.4: Fix manual filter mode stub
11. [ ] Task 4.2: Integration testing

---

## Code Review Checklist

Before merging any changes, verify:

- [ ] All velocities converted correctly between m/s and ft/s
- [ ] Slider ranges appropriate for current unit system
- [ ] Tooltips accurate for current unit system
- [ ] Presets work correctly in both unit systems
- [ ] Saved configs include unit information
- [ ] Loading old configs (without units) defaults correctly
- [ ] Sub-gather spacing correctly scaled
- [ ] Wavenumber axis labels show correct units
- [ ] Debug output conditional on flag
- [ ] All new code has unit tests
- [ ] Integration tests pass with metric and feet SEGY files

---

## Constants Reference

```python
# Conversion factors
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 0.3048

# Velocity conversions
# m/s to ft/s: multiply by 3.28084
# ft/s to m/s: multiply by 0.3048

# Example velocities:
# Ground roll: 300-500 m/s = 984-1640 ft/s
# Air wave: 330 m/s = 1083 ft/s
# P-wave: 1500-6000 m/s = 4921-19685 ft/s
# S-wave: 500-3000 m/s = 1640-9843 ft/s
```

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `utils/subgather_detector.py` | Add SEGY scalar application |
| `processors/fk_filter.py` | Add unit conversion, update description |
| `views/fk_designer_dialog.py` | Dynamic ranges, tooltips, unit awareness |
| `models/fk_config.py` | Add coordinate_units field, conversion method |
| `utils/unit_conversion.py` | Add velocity conversion function |
| `utils/coordinate_units.py` | New file for unit detection |
| `tests/test_fk_units.py` | New unit test file |
| `tests/test_fk_integration.py` | New integration test file |

---

*Report generated by Claude Code analysis*
