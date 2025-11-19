# FK Designer UI Improvements V2 - 2025-11-18

## User Requests

1. **Move FK display options to left panel**
2. **Make left panel scrollable**
3. **Make vertical splitter more visible and easier to drag**

## Changes Implemented

### 1. FK Display Options Moved to Left Panel ✅

**Before**: FK display controls (smoothing, gain, colormap, etc.) were in a horizontal bar at the top of the right panel, taking up vertical space.

**After**: Moved to left control panel with vertical layout.

**Benefits**:
- More vertical space for FK plot
- All controls in one place (left panel)
- Better organization

**Implementation**:

```python
def _create_controls_panel(self) -> QWidget:
    """Create left panel with filter controls (scrollable)."""
    # ...

    # FK Display Options (moved from right panel)
    fk_display_group = self._create_fk_display_controls()
    layout.addWidget(fk_display_group)

    # Preset selection
    preset_group = self._create_preset_group()
    layout.addWidget(preset_group)

    # ... other controls
```

**Vertical Layout** (suitable for left panel):
```python
def _create_fk_display_controls(self) -> QGroupBox:
    """Create FK display option controls (vertical layout)."""
    layout = QVBoxLayout()

    # Smoothing: [Label] [Slider] [Value]
    smooth_layout = QHBoxLayout()
    smooth_layout.addWidget(QLabel("Smoothing:"))
    smooth_layout.addWidget(self.fk_smoothing_slider)
    smooth_layout.addWidget(self.fk_smoothing_label)
    layout.addLayout(smooth_layout)

    # Gain: [Label] [Slider] [Value]
    gain_layout = QHBoxLayout()
    gain_layout.addWidget(QLabel("Gain:"))
    gain_layout.addWidget(self.fk_gain_slider)
    gain_layout.addWidget(self.fk_gain_label)
    layout.addLayout(gain_layout)

    # Colormap: [Label] [Dropdown]
    cmap_layout = QHBoxLayout()
    cmap_layout.addWidget(QLabel("Colormap:"))
    cmap_layout.addWidget(self.fk_colormap_combo)
    layout.addLayout(cmap_layout)

    # Checkboxes
    layout.addWidget(self.fk_show_filtered_check)
    layout.addWidget(self.fk_interactive_check)

    # Reset button
    layout.addWidget(reset_btn)
```

---

### 2. Scrollable Left Panel ✅

**Before**: Fixed height left panel. If controls didn't fit, they were cut off.

**After**: Left panel wrapped in QScrollArea with vertical scrollbar.

**Benefits**:
- All controls accessible even on small screens
- No clipping of controls
- Smooth scrolling

**Implementation**:

```python
def _create_controls_panel(self) -> QWidget:
    """Create left panel with filter controls (scrollable)."""
    from PyQt6.QtWidgets import QScrollArea

    # Create scroll area
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)  # Content resizes with scroll area
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # No horizontal scroll
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)  # No frame

    # Content widget with all controls
    content = QWidget()
    layout = QVBoxLayout()
    # ... add all control groups
    content.setLayout(layout)

    # Set content as scroll widget
    scroll.setWidget(content)

    return scroll
```

**Behavior**:
- Scrollbar appears automatically when content exceeds available height
- Mouse wheel scrolling works
- Touch/trackpad scrolling works
- Horizontal scrollbar disabled (controls fit width)

---

### 3. Enhanced Splitter Visibility and Dragging ✅

**Problem**: Splitter handles were thin (3px) and hard to find with cursor.

**Solution**: Wider handles (6px) with visual styling.

#### Horizontal Splitter (Left ↔ Right)

**Before**:
```python
main_splitter.setHandleWidth(3)  # Thin, hard to grab
# No visual styling
```

**After**:
```python
main_splitter.setHandleWidth(6)  # Wider handle

# Visual styling for better visibility
main_splitter.setStyleSheet("""
    QSplitter::handle {
        background-color: #c0c0c0;  /* Light gray */
        border: 1px solid #808080;   /* Darker border */
    }
    QSplitter::handle:hover {
        background-color: #a0a0ff;  /* Light blue on hover */
    }
    QSplitter::handle:pressed {
        background-color: #8080ff;  /* Darker blue when dragging */
    }
""")

# Prevent collapsing controls panel
main_splitter.setCollapsible(0, False)
```

**Visual States**:
- **Normal**: Light gray (#c0c0c0) with border
- **Hover**: Light blue (#a0a0ff) - cursor indicates draggable
- **Pressed**: Darker blue (#8080ff) - actively dragging

#### Vertical Splitter (FK Plot ↔ Previews)

**Before**:
```python
display_splitter.setHandleWidth(3)  # Thin
# No styling
```

**After**:
```python
display_splitter.setHandleWidth(6)  # Wider

# Same visual styling as horizontal splitter
display_splitter.setStyleSheet("""
    QSplitter::handle {
        background-color: #c0c0c0;
        border: 1px solid #808080;
    }
    QSplitter::handle:hover {
        background-color: #a0a0ff;
    }
    QSplitter::handle:pressed {
        background-color: #8080ff;
    }
""")
```

**Result**:
- ✓ Easy to find with cursor
- ✓ Visual feedback (changes color on hover/press)
- ✓ 6px handle (2x wider than before)
- ✓ Consistent styling across both splitters

---

## Layout Structure

### Complete Layout Hierarchy:

```
FK Designer Dialog
├── Header (compact: "Gather X: Y traces × Z samples")
├── Main Horizontal Splitter (20% ↔ 80%, resizable)
│   ├── LEFT PANEL (Scrollable QScrollArea)
│   │   ├── FK Display Options ← MOVED HERE!
│   │   │   ├── Smoothing: [slider] (Off/Light/Medium/...)
│   │   │   ├── Gain: [slider] (0.0001x - 10000x)
│   │   │   ├── Colormap: [dropdown]
│   │   │   ├── ☐ Show Filtered FK
│   │   │   ├── ☐ Interactive Boundaries
│   │   │   └── [Reset Display]
│   │   ├── Quick Presets
│   │   ├── Sub-Gathers (if headers available)
│   │   ├── AGC Options
│   │   ├── Filter Parameters
│   │   └── Quality Metrics
│   │
│   └── RIGHT PANEL (Vertical Splitter 60% ↕ 40%, resizable)
│       ├── FK Spectrum (top)
│       │   ├── FK Plot (large!)
│       │   └── Spacing: X m (from source) [ⓘ Details]
│       │
│       └── Previews (bottom)
│           ├── Input
│           ├── Filtered
│           └── Rejected
│
└── Bottom Buttons ([Preview] [Apply] [Save] [Cancel])
```

---

## User Interaction

### Scrolling Left Panel:

**When to scroll**:
- Controls exceed available vertical space
- Window is resized smaller

**How to scroll**:
- Mouse wheel
- Scroll bar (appears automatically)
- Trackpad gestures
- Click and drag scrollbar

**Example**: If window height is 800px but controls need 1000px, scrollbar appears automatically.

### Dragging Horizontal Splitter (Left ↔ Right):

1. **Find handle**: Move cursor between left panel and FK plot
2. **Visual feedback**: Handle turns light blue (#a0a0ff)
3. **Drag**: Cursor changes to ↔, click and drag
4. **While dragging**: Handle turns darker blue (#8080ff)
5. **Release**: New proportions saved

**Typical adjustments**:
- Drag left: Wider controls, narrower FK plot
- Drag right: Narrower controls, wider FK plot (more room for data!)

### Dragging Vertical Splitter (FK Plot ↔ Previews):

1. **Find handle**: Move cursor between FK plot and previews
2. **Visual feedback**: Handle turns light blue
3. **Drag**: Cursor changes to ↕, click and drag up/down
4. **While dragging**: Handle turns darker blue
5. **Release**: New proportions saved

**Typical adjustments**:
- Drag up: Larger previews, smaller FK plot
- Drag down: Larger FK plot, smaller previews (detailed analysis)

---

## Visual Comparison

### Before:

```
┌──────────────────────────────────────────────────────┐
│ Gather 1: 48 traces × 1501 samples                   │
├────────┬─────────────────────────────────────────────┤
│        │ FK Display: [Smooth][Gain][Cmap][✓][Reset] │ ← Horizontal bar
│ Filter │ ┌─────────────────────────────────────────┐ │
│ Params │ │                                         │ │
│ (20%)  │ │    FK Spectrum                          │ │
│        │ │    (compressed by controls bar)         │ │
│ Preset │ └─────────────────────────────────────────┘ │
│        │ Spacing: 220.0 m (from receiver_x) [ⓘ]    │
│ Sub-   │ ───────────────────────────────────────────│ ← Thin splitter
│ Gather │ ┌──────┐  ┌──────┐  ┌──────┐              │
│        │ │Input │  │Filter│  │Reject│              │
│ AGC    │ └──────┘  └──────┘  └──────┘              │
│        │                                             │
│ Params │                                             │
│        │                                             │
│Metrics │                                             │
│        │                                             │
│(maybe  │                                             │
│ cut    │                                             │
│ off!)  │                                             │
└────────┴─────────────────────────────────────────────┘
  ↑ thin, hard to find
```

### After:

```
┌──────────────────────────────────────────────────────┐
│ Gather 1: 48 traces × 1501 samples                   │
├────────┬─────────────────────────────────────────────┤
│[Scroll]│ ┌─────────────────────────────────────────┐ │
│↕       │ │                                         │ │
│FK Disp │ │                                         │ │
│ Smooth │ │    FK Spectrum (FULL HEIGHT!)           │ │
│ Gain   │ │                                         │ │
│ Cmap   │ │                                         │ │
│ ✓✓     │ │                                         │ │
│ Reset  │ │                                         │ │
│        │ └─────────────────────────────────────────┘ │
│Preset  │ Spacing: 220.0 m (from receiver_x) [ⓘ]    │
│        │ ═══════════════════════════════════════════│ ← Visible handle!
│Sub-    │ ┌────────┐  ┌────────┐  ┌────────┐        │   (gray → blue)
│Gather  │ │        │  │        │  │        │        │
│        │ │ Input  │  │Filtered│  │Rejected│        │
│AGC     │ │        │  │        │  │        │        │
│        │ └────────┘  └────────┘  └────────┘        │
│Params  │                                             │
│        │                                             │
│Metrics │                                             │
│        │ (All visible - scrollbar if needed)         │
└────────┴─────────────────────────────────────────────┘
    ↑ 6px, styled, easy to grab
```

---

## Code Changes Summary

### Files Modified:

**views/fk_designer_dialog.py** (~100 lines changed)

### Main Changes:

1. **Lines 142-177**: `_init_ui()`
   - Horizontal splitter styling (6px, colors, hover)
   - setCollapsible(0, False) to prevent control panel collapse

2. **Lines 191-237**: `_create_controls_panel()`
   - Wrapped in QScrollArea
   - Added FK display controls at top
   - All controls in scrollable content widget

3. **Lines 462-497**: `_create_display_panel()`
   - Removed FK display controls (moved to left)
   - Vertical splitter styling (6px, colors, hover)
   - Returns splitter directly (no extra wrapper)

4. **Lines 499-556**: `_create_fk_display_controls()`
   - Changed from horizontal to vertical layout
   - Each control row: [Label][Widget][Value]
   - Suitable for narrow left panel

---

## Benefits

### For User:

**Better Organization**:
- All controls in left panel
- Scrollable - nothing cut off
- Logical grouping

**More Space for Data**:
- FK plot not compressed by controls bar
- Full height available for visualization
- Larger preview plots

**Easier Resizing**:
- Splitter handles easy to find (6px wide)
- Visual feedback (color changes)
- Cursor changes to ↔ or ↕
- Smooth dragging experience

**Flexible Layout**:
- Adjust left/right split for workflow
- Adjust top/bottom split for task
- Scroll controls when needed

### For Workflow:

**Design Mode**:
1. Adjust FK display options in left panel
2. See immediate effect on large FK plot
3. Scroll down to adjust filter parameters
4. All in one place!

**Analysis Mode**:
1. Drag vertical splitter down (maximize FK plot)
2. Detailed analysis of FK spectrum
3. Adjust display options as needed
4. Drag splitter up to check previews

**Comparison Mode**:
1. Drag vertical splitter up (enlarge previews)
2. Compare input/filtered/rejected clearly
3. Adjust horizontal splitter for more room
4. Scroll controls to adjust parameters

---

## Testing Checklist

### Scrolling:
- [ ] Left panel scrolls with mouse wheel
- [ ] Scrollbar appears when controls exceed height
- [ ] No horizontal scrollbar
- [ ] All controls accessible via scrolling

### Horizontal Splitter:
- [ ] Handle visible (gray bar between panels)
- [ ] Cursor changes to ↔ when hovering
- [ ] Handle turns light blue on hover
- [ ] Handle turns darker blue when dragging
- [ ] Dragging adjusts left/right proportions
- [ ] Controls panel doesn't collapse completely

### Vertical Splitter:
- [ ] Handle visible (gray bar between FK and previews)
- [ ] Cursor changes to ↕ when hovering
- [ ] Handle turns light blue on hover
- [ ] Handle turns darker blue when dragging
- [ ] Dragging adjusts FK plot and preview heights

### FK Display Options:
- [ ] Located in left panel (top of controls)
- [ ] Smoothing slider works
- [ ] Gain slider works
- [ ] Colormap dropdown works
- [ ] Show Filtered FK checkbox works
- [ ] Interactive Boundaries checkbox works
- [ ] Reset Display button works

### Overall:
- [ ] FK plot has more vertical space
- [ ] All controls accessible
- [ ] Layout feels natural
- [ ] Easy to customize for workflow

---

## Compilation

```bash
python3 -m py_compile views/fk_designer_dialog.py
✓ Compilation successful
```

---

## Summary

**Status**: ✅ COMPLETE
**Files Modified**: 1 (`views/fk_designer_dialog.py`)
**Lines Changed**: ~100

**Key Improvements**:
1. ✓ FK display options moved to left panel (better organization)
2. ✓ Left panel scrollable (all controls accessible)
3. ✓ Splitter handles wider (6px vs 3px)
4. ✓ Splitter handles styled (gray → blue on hover/press)
5. ✓ Visual feedback when dragging
6. ✓ More vertical space for FK plot

**Result**: FK Designer now has:
- **Better organization** (all controls in scrollable left panel)
- **More space for data** (FK plot not compressed)
- **Easier resizing** (visible, styled splitter handles)
- **Professional feel** (visual feedback, smooth interaction)

The interface is now more intuitive and provides better control over the layout!
