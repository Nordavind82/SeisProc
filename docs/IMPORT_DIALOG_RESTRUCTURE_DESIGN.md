# SEG-Y Import Dialog Restructuring Design

## Overview

Restructure the SEG-Y Import Dialog from a vertical stack of QGroupBox widgets to a tabbed interface that:
1. Improves organization and reduces visual clutter
2. Enables addition of new configuration sections (QC Lines)
3. Maintains all existing functionality
4. Improves user workflow with logical groupings

---

## Current Structure

```
┌─────────────────────────────────────────────────────────────┐
│ SEG-Y File Selection (QGroupBox)                            │
│   [File path] [Browse...] [File Info]                       │
├─────────────────────────────────────────────────────────────┤
│ Trace Header Mapping Configuration (QGroupBox) - LARGE      │
│   [Add] [Remove] [Save] [Load] [Standard]                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Header Table (5 columns, many rows)                 │   │
│   └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│ Computed Headers (QGroupBox)                                │
│   [Add] [Remove]                                            │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Computed Header Table (max height 150)              │   │
│   └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│ Import Configuration (QGroupBox)                            │
│   Ensemble Keys: [___________]                              │
│   Spatial Units: [Meters ▼]                                 │
├─────────────────────────────────────────────────────────────┤
│ Header Preview (QGroupBox)                                  │
│   [Preview Headers]                                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Preview Text (max height 150)                       │   │
│   └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                              [Import SEG-Y] [Cancel]        │
└─────────────────────────────────────────────────────────────┘
```

**Problems:**
- Vertical scrolling required on smaller screens
- Header mapping table dominates the view
- No logical grouping of related settings
- No space for new features (QC configuration)

---

## Proposed Structure

```
┌─────────────────────────────────────────────────────────────┐
│ SEG-Y File Selection (Always Visible)                       │
│   [File path_______________] [Browse...] [File Info]        │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────┬────────────┬──────────┬─────────┐              │
│ │ Headers │ Import     │ QC Lines │ Preview │              │
│ └─────────┴────────────┴──────────┴─────────┘              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ │              Tab Content Area                           │ │
│ │                                                         │ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Summary: 1,234,567 traces | CDP sort | 5 QC lines          │
├─────────────────────────────────────────────────────────────┤
│                              [Import SEG-Y] [Cancel]        │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab Definitions

### Tab 1: Headers

Contains header mapping and computed headers configuration.

```
┌─────────────────────────────────────────────────────────────┐
│ Trace Header Mapping                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Add Custom] [Remove] ──── [Save...] [Load...] [Reset]  │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Header Name │ Byte Pos │ Format │ Description │ Sample  │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ TRACE_NO    │ 1        │ i4     │ Trace num   │ 1,2,3   │ │
│ │ CDP         │ 21       │ i4     │ CDP number  │ 100,... │ │
│ │ ...         │ ...      │ ...    │ ...         │ ...     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Computed Headers                                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Add Computed] [Remove]                                 │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Name        │ Expression           │ Format │ Desc      │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ recv_line   │ round(recv_sta/1000) │ i4     │ Rec line  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Syntax: +, -, *, /, round, floor, ceil, abs, sqrt, ...      │
└─────────────────────────────────────────────────────────────┘
```

### Tab 2: Import Settings

Contains ensemble configuration and output settings.

```
┌─────────────────────────────────────────────────────────────┐
│ Ensemble Configuration                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Ensemble Keys: [cdp________________] ▼ [Auto-detect]    │ │
│ │   Define how traces are grouped into gathers            │ │
│ │                                                         │ │
│ │ Common presets: ○ CDP  ○ Shot  ○ Inline/Xline  ○ Custom │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Coordinate Settings                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Spatial Units: [Meters ▼]                               │ │
│ │   Used for coordinates, offsets, and distances          │ │
│ │                                                         │ │
│ │ Coordinate Scalar: [Auto-detect from header ▼]          │ │
│ │   □ Apply scalar to X/Y coordinates                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Output Settings                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Output Directory: [________________________] [Browse]   │ │
│ │   □ Use default (same as SEG-Y location)                │ │
│ │                                                         │ │
│ │ Compression: [Blosc LZ4 ▼]                              │ │
│ │ Chunk Size:  [10000 traces ▼]                           │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Tab 3: QC Lines (NEW)

Configure which inlines to flag for QC processing.

```
┌─────────────────────────────────────────────────────────────┐
│ QC Line Configuration                                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ □ Enable QC line marking during import                  │ │
│ │   Marks specific inlines for later QC stacking/batch    │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Inline Selection                                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Available range: 100 - 5000 (from header scan)          │ │
│ │                                                         │ │
│ │ QC Inlines: [100, 500, 1000, 2000, 3000, 4000, 5000__]  │ │
│ │   Format: comma-separated, ranges with hyphen           │ │
│ │                                                         │ │
│ │ Quick Select:                                           │ │
│ │   [Every 10th] [Every 50th] [Every 100th] [Clear]       │ │
│ │                                                         │ │
│ │ Selected: 7 QC lines                                    │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ QC Processing Options (applied after import)                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ □ Auto-run QC stacking after import                     │ │
│ │   Uses default velocity and stacking parameters         │ │
│ │                                                         │ │
│ │ □ Open QC Batch dialog after import                     │ │
│ │   Configure processing chain for before/after compare   │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Tab 4: Preview

Header preview and file statistics.

```
┌─────────────────────────────────────────────────────────────┐
│ Header Preview                                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Preview First 10 Traces] [Preview Random 10] [Clear]   │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Trace │ CDP   │ Offset │ Inline │ Xline │ X      │ Y    │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ 1     │ 100   │ 150    │ 100    │ 200   │ 5000.0 │ 3000 │ │
│ │ 2     │ 100   │ 175    │ 100    │ 200   │ 5000.0 │ 3000 │ │
│ │ ...   │ ...   │ ...    │ ...    │ ...   │ ...    │ ...  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ File Statistics                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Traces: 1,234,567        Samples: 2000                  │ │
│ │ Sample Interval: 4.0 ms  Trace Length: 8000 ms          │ │
│ │ Data Format: IEEE Float  File Size: 9.8 GB              │ │
│ │                                                         │ │
│ │ Header Ranges (from scan):                              │ │
│ │   CDP: 100 - 5000        Inline: 100 - 500              │ │
│ │   Offset: 0 - 6000 m     Xline: 200 - 800               │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Step 1: Create Tab Container Structure

Modify `_init_ui()` to use QTabWidget:

```python
def _init_ui(self):
    layout = QVBoxLayout()

    # File selection - always visible at top
    layout.addWidget(self._create_file_selection_group())

    # Tab widget for main content
    self.tab_widget = QTabWidget()

    # Create tabs
    self.tab_widget.addTab(self._create_headers_tab(), "Headers")
    self.tab_widget.addTab(self._create_import_settings_tab(), "Import Settings")
    self.tab_widget.addTab(self._create_qc_lines_tab(), "QC Lines")
    self.tab_widget.addTab(self._create_preview_tab(), "Preview")

    layout.addWidget(self.tab_widget)

    # Summary bar
    layout.addWidget(self._create_summary_bar())

    # Action buttons - always visible at bottom
    layout.addLayout(self._create_action_buttons())

    self.setLayout(layout)
```

### Step 2: Refactor Existing Groups into Tab Content

**Headers Tab (`_create_headers_tab`):**
- Combine `_create_header_mapping_group()` and `_create_computed_headers_group()`
- Use QSplitter for adjustable sizing between the two tables

**Import Settings Tab (`_create_import_settings_tab`):**
- Refactor `_create_ensemble_group()` into ensemble section
- Add new coordinate settings section
- Add new output settings section

**Preview Tab (`_create_preview_tab`):**
- Refactor `_create_preview_group()`
- Add file statistics display
- Improve preview table formatting

### Step 3: Create New QC Lines Tab

**New file section or method `_create_qc_lines_tab`:**
- Reuse inline selection logic from `QCStackingDialog`
- Add enable/disable checkbox
- Add quick select buttons
- Store QC config in import metadata

### Step 4: Add Summary Bar

New widget showing import summary:

```python
def _create_summary_bar(self):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(5, 2, 5, 2)

    self.summary_label = QLabel("Select a SEG-Y file to begin")
    self.summary_label.setStyleSheet("color: #666;")
    layout.addWidget(self.summary_label)

    layout.addStretch()

    # Validation indicator
    self.validation_label = QLabel()
    layout.addWidget(self.validation_label)

    return widget

def _update_summary(self):
    if not self.segy_file:
        self.summary_label.setText("Select a SEG-Y file to begin")
        return

    parts = []
    parts.append(f"{self.reader.n_traces:,} traces")
    parts.append(f"{self.reader.n_samples} samples")

    if self.ensemble_keys_edit.text():
        parts.append(f"Sort: {self.ensemble_keys_edit.text()}")

    if hasattr(self, 'qc_lines_enabled') and self.qc_lines_enabled.isChecked():
        n_qc = len(self._parse_qc_inlines())
        parts.append(f"{n_qc} QC lines")

    self.summary_label.setText(" | ".join(parts))
```

### Step 5: Update Metadata Storage

Store QC configuration in the output metadata:

```python
# In _import_segy() or DataStorage
metadata = {
    # ... existing metadata ...
    'qc_config': {
        'enabled': self.qc_lines_enabled.isChecked(),
        'inline_numbers': self._parse_qc_inlines(),
        'auto_stack': self.auto_stack_check.isChecked(),
    } if self.qc_lines_enabled.isChecked() else None
}
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `views/segy_import_dialog.py` | Major restructure - add tabs, refactor groups |
| `utils/segy_import/data_storage.py` | Add QC config to metadata |
| `main_window.py` | Handle post-import QC actions |

---

## Migration Strategy

1. **Phase A: Tab Structure** (non-breaking)
   - Add QTabWidget wrapper
   - Move existing groups into tabs
   - No functionality changes

2. **Phase B: Enhanced Tabs** (additive)
   - Add Import Settings improvements
   - Add Preview enhancements
   - Add Summary bar

3. **Phase C: QC Lines Tab** (new feature)
   - Add QC Lines tab
   - Add metadata storage
   - Add post-import QC actions

---

## UI/UX Considerations

1. **Tab Order**: Most commonly used tabs first
   - Headers (primary configuration)
   - Import Settings (secondary configuration)
   - QC Lines (optional feature)
   - Preview (verification)

2. **Default Tab**: Headers tab opens by default

3. **Tab Validation**:
   - Show warning icon on tabs with validation errors
   - Prevent import if required fields missing

4. **Keyboard Navigation**:
   - Ctrl+1/2/3/4 to switch tabs
   - Tab key moves through fields within tab

5. **State Persistence**:
   - Remember last-used tab
   - Remember QC lines configuration

---

## Success Criteria

1. All existing functionality preserved
2. QC Lines tab fully integrated
3. Dialog fits on 900x700 minimum resolution
4. Tab switching is instant (no lag)
5. Summary bar updates in real-time
6. Import metadata includes QC configuration
