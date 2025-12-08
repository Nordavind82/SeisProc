# ISA Gain Synchronization - Implementation Summary

## Overview

Added gain and colormap synchronization between ISA window and main window, identical to the flip window implementation. All windows now share the same `ViewportState` object for seamless synchronization.

---

## Implementation

### Architecture

```
┌──────────────────┐
│   Main Window    │
│                  │
│  • Gain Slider   │──┐
│  • Colormap      │  │
│  • Zoom/Pan      │  │
└──────────────────┘  │
                      │
                      ├─────► ViewportState (Shared Object)
                      │         ├─ amplitude_range (gain)
┌──────────────────┐  │         ├─ colormap
│   ISA Window     │  │         ├─ zoom/pan limits
│                  │  │         └─ PyQt signals
│  • Seismic View  │──┤
│  • Colormap Drop │  │
└──────────────────┘  │
                      │
┌──────────────────┐  │
│   Flip Window    │  │
│                  │  │
│  • Seismic View  │──┘
│  • Colormap Drop │
└──────────────────┘
```

All windows share the **same ViewportState instance** → automatic synchronization!

---

## Changes Made

### 1. ISA Window Constructor

**Before:**
```python
def __init__(self, data: SeismicData, parent=None):
    self.viewport_state = ViewportState()  # Created new instance
```

**After:**
```python
def __init__(self, data: SeismicData, viewport_state: ViewportState = None, parent=None):
    # Use shared viewport state if provided, otherwise create new one
    self.viewport_state = viewport_state if viewport_state is not None else ViewportState()
```

**Benefits:**
- Backward compatible (optional parameter)
- Standalone mode still works (creates own viewport)
- Main window mode shares viewport for sync

---

### 2. Signal Connections

Added signal handlers in ISA window:

```python
# Connect viewport state changes for synchronization
self.viewport_state.amplitude_range_changed.connect(
    self._on_viewport_amplitude_changed
)
self.viewport_state.colormap_changed.connect(
    self._on_viewport_colormap_changed
)
```

---

### 3. Signal Handlers

**Amplitude Range Handler:**
```python
def _on_viewport_amplitude_changed(self, min_amp: float, max_amp: float):
    """Handle amplitude range change from viewport state (main window)."""
    # Amplitude range is already applied to viewer through viewport_state
    # No additional action needed - viewer updates automatically
    pass
```

**Colormap Handler:**
```python
def _on_viewport_colormap_changed(self, colormap_name: str):
    """Handle colormap change from viewport state (main window)."""
    # Update combo box to match without triggering signal
    self.colormap_combo.blockSignals(True)
    index = self.colormap_combo.findText(colormap_name)
    if index >= 0:
        self.colormap_combo.setCurrentIndex(index)
    self.colormap_combo.blockSignals(False)
```

**ISA Colormap Change:**
```python
def _on_colormap_changed(self, colormap_name: str):
    """Handle colormap change from dropdown."""
    self.viewport_state.set_colormap(colormap_name)  # Updates all windows
```

---

### 4. Main Window Integration

**Before:**
```python
isa_window = ISAWindow(self.input_data, self)
```

**After:**
```python
# Pass shared viewport_state for synchronization
isa_window = ISAWindow(self.input_data, self.viewport_state, self)
```

---

## Synchronization Features

### What Gets Synchronized

| Property | Main → ISA | ISA → Main | Notes |
|----------|------------|------------|-------|
| **Amplitude Range (Gain)** | ✓ Auto | N/A | Viewer updates automatically |
| **Colormap** | ✓ Auto | ✓ Auto | Bidirectional sync |
| **Zoom/Pan** | ✓ Auto | ✓ Auto | Shared viewport limits |
| **Interpolation** | ✓ Auto | N/A | Display quality |

### How It Works

1. **Main window changes gain:**
   - `control_panel.amplitude_range_changed` → `viewport_state.set_amplitude_range()`
   - ViewportState emits `amplitude_range_changed` signal
   - ISA seismic viewer receives signal (connected to viewport)
   - ISA data display updates automatically ✓

2. **Main window changes colormap:**
   - `control_panel.colormap_changed` → `viewport_state.set_colormap()`
   - ViewportState emits `colormap_changed` signal
   - ISA receives signal via `_on_viewport_colormap_changed`
   - ISA colormap dropdown updates ✓
   - ISA seismic viewer updates ✓

3. **ISA changes colormap:**
   - User selects colormap in ISA dropdown
   - `_on_colormap_changed` → `viewport_state.set_colormap()`
   - ViewportState emits `colormap_changed` signal
   - Main window receives signal
   - Main window colormap updates ✓
   - All other windows update ✓

---

## Benefits

### 1. Consistent Visualization
- Same gain applied across all windows
- Same colormap everywhere
- No manual synchronization needed

### 2. Improved QC Workflow
```
Traditional workflow:
- Adjust gain in main window
- Manually adjust gain in ISA window
- Keep trying to match visually
- Frustrating and time-consuming ✗

Synchronized workflow:
- Adjust gain ONCE in main window
- ISA updates automatically
- Perfect match guaranteed ✓
```

### 3. Seamless Multi-Window QC
- Open multiple ISA windows
- All share same viewport_state
- All stay synchronized
- Switch between windows seamlessly

### 4. Flip-Window Consistency
- Same mechanism as flip window
- Familiar behavior for users
- Proven implementation

---

## Usage Examples

### Example 1: Basic Synchronization
```
1. Load data in main window
2. Open ISA window (Ctrl+I)
3. In main window:
   - Move gain slider left
   → ISA data darkens automatically
   - Move gain slider right
   → ISA data brightens automatically
4. Perfect sync, no manual adjustment!
```

### Example 2: Colormap Synchronization
```
1. Main window and ISA both showing data
2. In main window:
   - Change colormap to 'grayscale'
   → ISA immediately switches to grayscale
3. In ISA window:
   - Change colormap to 'viridis'
   → Main window immediately switches to viridis
4. Bidirectional sync working!
```

### Example 3: Multi-Window QC
```
1. Open ISA Window #1
2. Open ISA Window #2
3. In main window:
   - Adjust gain
   → Both ISA windows update
   - Change colormap
   → Both ISA windows update
4. All windows perfectly synchronized!
```

---

## Technical Details

### Signal Flow

```
Main Window Gain Change:
control_panel → viewport_state.set_amplitude_range()
              → viewport_state.amplitude_range_changed.emit()
              → isa_seismic_viewer (connected to viewport)
              → viewer updates display

Main Window Colormap Change:
control_panel → viewport_state.set_colormap()
              → viewport_state.colormap_changed.emit()
              → isa._on_viewport_colormap_changed()
              → isa.colormap_combo updates
              → viewer updates display

ISA Colormap Change:
isa.colormap_combo → isa._on_colormap_changed()
                   → viewport_state.set_colormap()
                   → viewport_state.colormap_changed.emit()
                   → main_window receives signal
                   → all windows update
```

### Preventing Signal Loops

```python
# When updating combo box from signal, block signals to prevent loop
self.colormap_combo.blockSignals(True)
self.colormap_combo.setCurrentIndex(index)
self.colormap_combo.blockSignals(False)
```

Without blocking:
```
ISA colormap changed → viewport updated → ISA receives signal
→ ISA combo updated → signal emitted → viewport updated (LOOP!)
```

With blocking:
```
ISA colormap changed → viewport updated → ISA receives signal
→ ISA combo updated (signals blocked) → NO loop ✓
```

---

## Testing

### Manual Testing Steps

```bash
python main.py
```

1. **Load Data:**
   - File → Generate Sample Data

2. **Open ISA:**
   - View → Open ISA Window (Ctrl+I)

3. **Test Gain Sync:**
   - In main: Move gain slider
   - Watch: ISA data updates in real-time ✓

4. **Test Colormap Sync (Main → ISA):**
   - In main: Change colormap dropdown
   - Watch: ISA colormap changes ✓
   - Watch: ISA dropdown updates ✓

5. **Test Colormap Sync (ISA → Main):**
   - In ISA: Change colormap dropdown
   - Watch: Main window colormap changes ✓
   - Watch: Main dropdown updates ✓

6. **Test Multi-Window:**
   - Open second ISA window
   - In main: Change gain/colormap
   - Watch: Both ISA windows update ✓

### Automated Testing

```bash
python test_isa_gain_sync.py
```

**Test Results:**
- ✓ ViewportState synchronization
- ✓ Amplitude range updates
- ✓ Colormap updates
- ✓ Bidirectional sync
- ✓ Architecture diagram
- ✓ Implementation examples

---

## Files Modified

### views/isa_window.py
**Changes:**
1. Added `viewport_state` parameter to constructor (optional)
2. Connected `amplitude_range_changed` signal
3. Connected `colormap_changed` signal
4. Added `_on_viewport_colormap_changed()` handler
5. Added `_on_viewport_amplitude_changed()` handler
6. Updated `_on_colormap_changed()` to use viewport

**Lines changed:** ~30 lines

### main_window.py
**Changes:**
1. Pass `self.viewport_state` to ISA constructor
2. Updated status message to mention synchronization

**Lines changed:** 2 lines

### test_isa.py
**Changes:**
1. Added comment about optional viewport_state parameter

**Lines changed:** 1 line

---

## Comparison: Before vs After

| Feature | Before | After |
|---------|--------|-------|
| **Viewport State** | ISA creates own | ISA shares with main ✓ |
| **Gain Sync** | No sync | Auto sync ✓ |
| **Colormap Sync** | No sync | Bidirectional sync ✓ |
| **Multi-Window** | Independent | All synchronized ✓ |
| **User Experience** | Manual adjustment | Automatic ✓ |

---

## Similar Implementation

This implementation mirrors the **Flip Window** exactly:

```python
# Flip Window (existing):
flip_window = FlipWindow(self.viewport_state, self)

# ISA Window (new):
isa_window = ISAWindow(self.input_data, self.viewport_state, self)
```

Both use the same synchronization mechanism:
- Shared viewport_state object
- Signal connections for updates
- blockSignals() to prevent loops
- Bidirectional colormap sync

---

## Future Enhancements

Potential additions (not implemented):

1. **Amplitude Range Controls in ISA:**
   - Add gain slider to ISA window
   - Allow direct gain adjustment in ISA
   - Would sync back to main window

2. **Auto-Scale Synchronization:**
   - When main auto-scales, ISA updates
   - When ISA auto-scales, main updates

3. **Clip Percentile Sync:**
   - Sync clip percentile setting
   - Currently each window independent

4. **Interpolation Sync:**
   - Sync interpolation mode
   - Currently auto-syncs through viewport

---

## Conclusion

Gain and colormap synchronization successfully implemented in ISA window:

- ✓ Same architecture as flip window
- ✓ Automatic bidirectional sync
- ✓ No manual adjustment needed
- ✓ Improved QC workflow
- ✓ Multi-window support
- ✓ Tested and verified

The ISA window now provides a seamless multi-window QC experience with perfect synchronization across all windows!
