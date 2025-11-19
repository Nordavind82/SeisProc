# Spatial Units Implementation Guide

## Overview

This document provides a complete guide for implementing meters/feet spatial units throughout the application.

## ✅ Completed Components

### 1. Core Infrastructure
- ✅ **`models/app_settings.py`** - Global settings singleton with QSettings persistence
- ✅ **`utils/unit_conversion.py`** - Unit conversion utilities
- ✅ **`views/segy_import_dialog.py`** - Spatial units selector added to import dialog

### 2. Key Features Implemented
- Persistent storage of user's unit preference
- Conversion functions (meters ↔ feet)
- Formatting utilities for display
- Signal emission when units change

## 📋 Remaining Implementation Tasks

### Priority 1: FK Designer Dialog (`views/fk_designer_dialog.py`)

#### Locations to Update:

**Initial Label Creation:**
- Line 452: `QLabel(f"{self.v_min:.0f} m/s")` → Use `format_velocity()`
- Line 471: `QLabel(f"{self.v_max:.0f} m/s")` → Use `format_velocity()`
- Line 548: `QLabel(f"{self.taper_width:.0f} m/s")` → Use `format_velocity()`

**Slider Change Handlers:**
- Line 837: `setText(f"{self.v_min:.0f} m/s")` → Use `format_velocity()`
- Line 854: `setText(f"{self.v_max:.0f} m/s")` → Use `format_velocity()`
- Line 901: `setText(f"{self.taper_width:.0f} m/s")` → Use `format_velocity()`

**Legend Text:**
- Lines 1457, 1461: Update legend text to use `format_velocity()`

**Trace Spacing Display:**
- Line 1011: "Spacing: X.Xm" → Use `format_distance()`
- Method `_update_trace_spacing_display()` (line ~1300) → Use `format_distance()`

**FK Spectrum Axes:**
- Line 151: Wavenumber axis label → Use `UnitConverter.get_wavenumber_label()`

**Dip Labels:**
- Dip parameter displays → Use `UnitConverter.get_dip_label()`

#### Implementation Pattern:

```python
# Before:
self.v_min_label.setText(f"{self.v_min:.0f} m/s")

# After:
self.v_min_label.setText(format_velocity(self.v_min))

# Before:
legend_text += f"v_min: {self.v_min:.0f} m/s (yellow solid)\n"

# After:
legend_text += f"v_min: {format_velocity(self.v_min)} (yellow solid)\n"

# Before:
print(f"Spacing: {spacing:.1f}m")

# After:
print(f"Spacing: {format_distance(spacing)}")
```

#### Connect to Settings Changes:

Add in `__init__`:
```python
# Connect to settings changes
get_settings().spatial_units_changed.connect(self._on_units_changed)
```

Add new method:
```python
def _on_units_changed(self, new_units: str):
    """Handle spatial units change."""
    # Update all labels
    self.v_min_label.setText(format_velocity(self.v_min))
    self.v_max_label.setText(format_velocity(self.v_max))
    self.taper_label.setText(format_velocity(self.taper_width))
    self._update_trace_spacing_display()
    self._update_fk_spectrum()  # Refresh axes labels
```

### Priority 2: Trace Spacing Utilities (`utils/trace_spacing.py`)

#### Update `format_spacing_stats()`:

```python
def format_spacing_stats(stats: TraceSpacingStats, units: Optional[str] = None) -> str:
    """Format spacing statistics with proper units."""
    if stats.n_traces < 2:
        return "Insufficient traces for spacing calculation"

    lines = []
    lines.append(f"Trace Spacing Statistics ({stats.n_traces} traces):")
    lines.append(f"  Mean:     {format_distance(stats.mean, decimals=2, units=units)}")
    lines.append(f"  Median:   {format_distance(stats.median, decimals=2, units=units)}")
    lines.append(f"  Std Dev:  {format_distance(stats.std, decimals=3, units=units)}")
    lines.append(f"  Min:      {format_distance(stats.min, decimals=2, units=units)}")
    lines.append(f"  Max:      {format_distance(stats.max, decimals=2, units=units)}")
    # ... rest of formatting
```

### Priority 3: FK Filter Config (`models/fk_config.py`)

#### Update `get_summary()`:

```python
def get_summary(self, units: Optional[str] = None) -> str:
    """Get human-readable summary with proper units."""
    parts = []

    if self.filter_mode == 'velocity':
        parts.append(f"Mode: Velocity")
        if self.v_min is not None:
            parts.append(f"v_min={format_velocity(self.v_min, decimals=0, units=units)}")
        # ... rest
```

#### Store Units in Config:

Add field to FKFilterConfig:
```python
@dataclass
class FKFilterConfig:
    # ... existing fields ...
    display_units: str = 'meters'  # Units used when config was created
```

### Priority 4: Control Panel (`views/control_panel.py`)

#### Add Units Display:

Update FK filter configuration list to show units:
```python
# When displaying FK config summaries
summary = config.get_summary(units=get_settings().get_spatial_units())
```

#### Add Settings Menu Action:

```python
def _create_settings_action(self):
    """Add settings menu or button."""
    settings_btn = QPushButton("⚙ Settings")
    settings_btn.clicked.connect(self._show_settings)
```

```python
def _show_settings(self):
    """Show settings dialog."""
    from views.settings_dialog import SettingsDialog
    dialog = SettingsDialog(self)
    if dialog.exec():
        # Settings changed, update UI
        self._refresh_all_displays()
```

### Priority 5: Settings Dialog (NEW FILE)

Create `views/settings_dialog.py`:

```python
"""
Application settings dialog.
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QComboBox, QGroupBox)
from PyQt6.QtCore import Qt
from models.app_settings import get_settings, AppSettings


class SettingsDialog(QDialog):
    """Application settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.resize(500, 300)

        layout = QVBoxLayout()

        # Spatial units group
        units_group = QGroupBox("Spatial Units")
        units_layout = QVBoxLayout()

        units_layout.addWidget(QLabel(
            "Select the units used for displaying coordinates, "
            "distances, offsets, and velocities:"
        ))

        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Units:"))

        self.units_combo = QComboBox()
        self.units_combo.addItem("Meters (m, m/s)", AppSettings.METERS)
        self.units_combo.addItem("Feet (ft, ft/s)", AppSettings.FEET)

        # Set current value
        current_units = get_settings().get_spatial_units()
        index = 0 if current_units == AppSettings.METERS else 1
        self.units_combo.setCurrentIndex(index)

        units_row.addWidget(self.units_combo)
        units_row.addStretch()

        units_layout.addLayout(units_row)

        units_layout.addWidget(QLabel(
            "Note: Changing units will update all displays throughout the application."
        ))

        units_group.setLayout(units_layout)
        layout.addWidget(units_group)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def accept(self):
        """Save settings and close."""
        selected_units = self.units_combo.currentData()
        get_settings().set_spatial_units(selected_units)
        super().accept()

    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        get_settings().reset_to_defaults()
        self.units_combo.setCurrentIndex(0)
```

### Priority 6: Processor Parameters (`processors/fk_filter.py`)

Update parameter validation to use units:

```python
def apply_fk_filter(...):
    """
    Apply FK filter with spatial unit awareness.

    Args:
        trace_spacing: ALWAYS in METERS (internal representation)
        ... other params ...
    """
    # Validation with helpful error messages
    if trace_spacing < 0.1 or trace_spacing > 1000:
        spacing_display = format_distance(trace_spacing)
        raise ValueError(
            f"Invalid trace spacing: {spacing_display}. "
            f"Expected range: {format_distance(0.1)} to {format_distance(1000)}"
        )
```

### Priority 7: Header Display

Update any header display dialogs to show units for spatial headers:

```python
# When displaying headers
SPATIAL_HEADERS = ['source_x', 'source_y', 'receiver_x', 'receiver_y',
                   'cdp_x', 'cdp_y', 'offset']

def format_header_value(header_name: str, value: float) -> str:
    """Format header value with units if applicable."""
    if header_name in SPATIAL_HEADERS:
        return format_distance(value)
    else:
        return str(value)
```

## 🔄 Unit Conversion Rules

### Internal Representation (ALWAYS)
- **All internal calculations**: METERS
- **All stored values**: METERS
- **SEGY import**: Convert to METERS (via scalar)
- **Processor parameters**: METERS

### Display Conversion (CONDITIONAL)
- **UI labels**: Convert based on `get_settings().get_spatial_units()`
- **User input**: Convert from display units to meters
- **Formatted output**: Use `format_distance()` and `format_velocity()`

### Conversion Constants
```python
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 1.0 / 3.28084
```

## 📊 Testing Checklist

### Unit Conversion Tests
- [ ] Meters to feet: 100m = 328.084ft
- [ ] Feet to meters: 1000ft = 304.8m
- [ ] Velocity: 1500 m/s = 4921 ft/s
- [ ] Wavenumber: k_m / 3.28084 = k_ft
- [ ] Settings persistence across app restarts

### UI Update Tests
- [ ] SEGY import: Select feet, verify app uses feet
- [ ] FK Designer: Change units in settings, verify labels update
- [ ] Trace spacing: Display matches selected units
- [ ] Config summaries: Show correct units
- [ ] Settings dialog: Changes apply immediately

### Edge Cases
- [ ] Very small values (0.1m = 0.328ft)
- [ ] Very large values (10000m = 32808ft)
- [ ] Decimal precision maintained
- [ ] Unit labels clear and consistent

## 🚀 Implementation Steps

### Phase 1: Core (✅ DONE)
1. ✅ Create `models/app_settings.py`
2. ✅ Create `utils/unit_conversion.py`
3. ✅ Add unit selector to SEGY import dialog
4. ✅ Save unit selection to settings

### Phase 2: FK Designer (IN PROGRESS)
1. Import unit conversion utilities
2. Update all velocity labels
3. Update trace spacing display
4. Update FK spectrum axis labels
5. Connect to settings change signal
6. Test with both meters and feet

### Phase 3: Supporting Components
1. Update `utils/trace_spacing.py` formatting
2. Update `models/fk_config.py` summaries
3. Create `views/settings_dialog.py`
4. Add settings menu to main app
5. Update processor error messages

### Phase 4: Polish & Testing
1. Test all UI components
2. Test settings persistence
3. Test unit conversions
4. Update documentation
5. Add tooltips where helpful

## 📝 Code Snippets

### Common Pattern for Labels

```python
# Create label
self.velocity_label = QLabel()
self._update_velocity_label()

def _update_velocity_label(self):
    """Update velocity label with current units."""
    self.velocity_label.setText(format_velocity(self.velocity))

# Connect to settings
get_settings().spatial_units_changed.connect(lambda: self._update_velocity_label())
```

### Common Pattern for User Input

```python
# User enters value in display units
user_value = float(self.input_field.text())

# Convert to internal (meters)
internal_value = UnitConverter.from_display_units(user_value)

# Store/use internal value
self.spacing = internal_value  # Always in meters

# Display back to user
self.label.setText(format_distance(self.spacing))
```

## 🎯 Summary

**Principle**: Store internally in METERS, display in user's preferred units.

**Key Files Modified**:
- `models/app_settings.py` (new)
- `utils/unit_conversion.py` (new)
- `views/segy_import_dialog.py` (updated)
- `views/fk_designer_dialog.py` (needs update)
- `views/settings_dialog.py` (new)
- `utils/trace_spacing.py` (needs update)
- `models/fk_config.py` (needs update)
- `views/control_panel.py` (needs update)

**Total Impact**:
- ~100-150 lines changed in FK Designer Dialog
- ~50 lines in trace spacing utilities
- ~30 lines in FK config
- ~100 lines for settings dialog (new)
- ~20 lines in control panel

**User Experience**:
1. Select units during SEGY import
2. Change units anytime in Settings
3. All displays update immediately
4. Units persist across sessions
5. Internal calculations unaffected
