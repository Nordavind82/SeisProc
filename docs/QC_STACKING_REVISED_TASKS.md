# QC Stacking Implementation - Revised Task List

## Overview
Revised implementation plan based on critical analysis of existing codebase infrastructure.

**Key Simplifications:**
- Leverage existing `VelocityModel` class (no recreation)
- Use existing `ensemble_index.parquet` for QC gather identification (no new index files)
- Reuse `ParallelProcessingCoordinator` pattern for orchestration
- Reuse `SeismicViewerPyQtGraph` components for QC viewer

---

## Phase 1: Core Infrastructure

### Task 1.1: Velocity I/O Module
**File:** `utils/velocity_io.py` (NEW)

- [ ] `read_velocity_ascii(filepath)` - Parse ASCII velocity files
  - Format 1: Time-Velocity pairs (single location)
  - Format 2: CDP-Time-Velocity triplets (spatially varying)
  - Auto-detect format from file structure
- [ ] `read_velocity_segy(filepath)` - Extract velocities from SEG-Y
  - Read trace data as velocity values
  - Extract CDP/inline/xline from headers
  - Build spatial velocity field
- [ ] `VelocityFileInfo` dataclass for file metadata preview
- [ ] Unit tests for both formats

**Estimated complexity:** Medium

### Task 1.2: NMO Processor
**File:** `processors/nmo_processor.py` (NEW)

- [ ] `NMOConfig` dataclass:
  - `velocity_type`: 'rms' or 'interval' (auto-convert if interval)
  - `stretch_mute_factor`: float (default 1.5)
  - `interpolation`: 'linear' or 'sinc'
- [ ] `NMOProcessor(BaseProcessor)`:
  - `__init__(config, velocity_model)`
  - `apply_nmo(traces, offsets, t0_axis)` → corrected traces
  - `apply_inverse_nmo(traces, offsets, t0_axis)` → restored traces
  - `compute_stretch_mute(offsets, t0_axis)` → mute mask
- [ ] Vectorized implementation using numpy
- [ ] Serialization support for multiprocess workers
- [ ] Unit tests with synthetic hyperbolic events

**Estimated complexity:** Medium

### Task 1.3: CDP Stacker
**File:** `processors/cdp_stacker.py` (NEW)

- [ ] `StackConfig` dataclass:
  - `method`: 'mean', 'median', 'weighted'
  - `min_fold`: int (minimum traces to stack)
  - `normalize`: bool
- [ ] `CDPStacker`:
  - `stack_gather(traces, headers)` → (stacked_trace, output_header)
  - `compute_fold(traces)` → fold count
  - Header preservation logic (CDP, mean offset, fold)
- [ ] Integration with NMOProcessor for NMO+stack workflow
- [ ] Unit tests

**Estimated complexity:** Low-Medium

---

## Phase 2: QC Stacking Workflow

### Task 2.1: QC Stacking Dialog
**File:** `views/qc_stacking_dialog.py` (NEW)

- [ ] Multi-page wizard dialog:
  - **Page 1 - Line Selection:**
    - Manual inline entry (comma-separated, ranges: "100,200,300-350")
    - Dataset info display (available inline range)
    - Estimated trace count
  - **Page 2 - Velocity Configuration:**
    - File type selector (ASCII/SEG-Y)
    - File browser with preview
    - Velocity type selector (RMS/Interval)
    - Simple T-V plot preview
  - **Page 3 - Stacking Parameters:**
    - NMO stretch mute (spinbox, default 1.5)
    - Stack method (combo: mean/median)
    - Minimum fold cutoff
  - **Page 4 - Output:**
    - Output directory selection
    - Output naming
    - Execute button
- [ ] Validation between pages
- [ ] Configuration save/load

**Estimated complexity:** Medium-High

### Task 2.2: QC Stacking Engine
**File:** `processors/qc_stacking_engine.py` (NEW)

- [ ] `QCStackingEngine`:
  - `__init__(dataset_path, velocity_model, config)`
  - `get_contributing_gathers(inline_nums)` → ensemble_ids
    - Query existing `ensemble_index.parquet`
    - Filter by INLINE_NO or CDP range
  - `stack_inline(inline_num, progress_callback)` → stacked section
  - `run(inline_nums, progress_callback)` → full QC stack output
- [ ] Progress reporting via signals (match existing pattern)
- [ ] Output to Zarr with metadata
- [ ] Memory-efficient chunked processing

**Estimated complexity:** Medium

### Task 2.3: Main Window Integration
**File:** `main_window.py` (MODIFY)

- [ ] Add "QC Stacking..." menu item under Processing menu
- [ ] Connect to QCStackingDialog
- [ ] Handle output dataset creation
- [ ] Optional auto-load of result

**Estimated complexity:** Low

---

## Phase 3: QC Stack Viewer

### Task 3.1: QC Stack Viewer Window
**File:** `views/qc_stack_viewer.py` (NEW)

- [ ] `QCStackViewerWindow(QMainWindow)`:
  - Three `SeismicViewerPyQtGraph` panels (Before/After/Difference)
  - Shared `ViewportState` for synchronization
  - Toolbar with display mode toggle
- [ ] Display modes:
  - Side-by-side (all three panels)
  - Before only / After only / Difference only
  - Flip mode (keyboard toggle between before/after)
- [ ] Inline navigation:
  - Slider + spinbox for inline selection
  - Prev/Next buttons
- [ ] Amplitude controls:
  - Independent scaling per panel
  - Shared amplitude scale option
  - Difference-optimized scale (±N std)

**Estimated complexity:** Medium

### Task 3.2: QC Statistics Panel
**File:** `views/qc_stack_viewer.py` (part of above)

- [ ] Statistics display widget:
  - Global RMS difference
  - Correlation coefficient (before vs after)
  - Max absolute difference (location + value)
  - Fold information
- [ ] Per-trace RMS difference plot (line graph below panels)
- [ ] Export statistics to text file

**Estimated complexity:** Medium

### Task 3.3: Viewer Integration
**File:** `main_window.py` (MODIFY)

- [ ] Add "QC Stack Viewer..." menu item under View menu
- [ ] File browser for loading before/after stacks
- [ ] Recent QC stacks list

**Estimated complexity:** Low

---

## Phase 4: QC Batch Processing (Future)

### Task 4.1: QC Batch Dialog
**File:** `views/qc_batch_dialog.py` (NEW)

- [ ] Inline selection (reuse from QCStackingDialog)
- [ ] Processing chain configuration widget
- [ ] Velocity model selection
- [ ] Output options (gathers, stacks, or both)

**Estimated complexity:** High

### Task 4.2: Processing Chain Widget
**File:** `views/processing_chain_widget.py` (NEW)

- [ ] List of available processors
- [ ] Drag-drop ordering
- [ ] Parameter editor per processor
- [ ] Chain preview/summary

**Estimated complexity:** High

### Task 4.3: QC Batch Engine
**File:** `processors/qc_batch_engine.py` (NEW)

- [ ] Integrate with existing ParallelProcessingCoordinator
- [ ] Selective gather processing
- [ ] Before/after stack generation
- [ ] Comparison metadata output

**Estimated complexity:** High

---

## Phase 5: Import Enhancement (Optional)

### Task 5.1: QC Tab in Import Dialog
**File:** `views/segy_import_dialog.py` (MODIFY)

- [ ] Add "QC Lines" tab
- [ ] Inline range input
- [ ] Auto-detect available inline range
- [ ] Store QC config in metadata.json

**Estimated complexity:** Low-Medium

---

## Implementation Status (2025-12-13)

### Phase 1: Core Infrastructure - COMPLETE
- [x] Task 1.1: Velocity I/O Module (`utils/velocity_io.py`)
  - Extended existing file with CDP-TV, ILXL-TV, SEG-Y support
  - Added VelocityFileFormat enum and VelocityFileInfo dataclass
  - Added preview functions and conversion utilities

- [x] Task 1.2: NMO Processor (`processors/nmo_processor.py`)
  - NMOConfig dataclass with stretch_mute, interpolation options
  - NMOProcessor with apply_nmo, apply_inverse_nmo methods
  - Vectorized implementation with sinc interpolation option

- [x] Task 1.3: CDP Stacker (`processors/cdp_stacker.py`)
  - StackConfig dataclass with method, min_fold options
  - CDPStacker with stack_gather, stack_gather_with_nmo methods
  - Mean, median, and weighted stacking support

### Phase 2: QC Stacking Workflow - COMPLETE
- [x] Task 2.1: QC Stacking Dialog (`views/qc_stacking_dialog.py`)
  - Multi-tab wizard with line selection, velocity config, parameters, output
  - Inline range parsing with quick select buttons
  - Velocity file preview

- [x] Task 2.2: QC Stacking Engine (`processors/qc_stacking_engine.py`)
  - QCStackingEngine orchestrates full workflow
  - QCStackingWorker for background execution
  - Progress signals for UI updates

- [x] Task 2.3: Main Window Integration (`main_window.py`)
  - Added "Processing → QC Stacking..." menu item
  - Added "_open_qc_stacking()" handler

### Phase 3: QC Stack Viewer - COMPLETE
- [x] Task 3.1: QC Stack Viewer (`views/qc_stack_viewer.py`)
  - QCStackViewerWindow with three synchronized panels
  - Before/After/Difference display modes
  - Flip mode for rapid A/B comparison
  - Statistics panel with RMS, correlation, SNR

- [x] Task 3.3: Viewer Integration (`main_window.py`)
  - Added "View → QC Stack Viewer" menu item
  - Added "_open_qc_stack_viewer()" handler

### Exports Updated
- [x] `processors/__init__.py` updated with NMOProcessor, CDPStacker exports

### Phase 4: QC Batch Processing - COMPLETE
- [x] Task 4.1: QC Batch Dialog (`views/qc_batch_dialog.py`)
  - QCBatchConfig dataclass with full workflow configuration
  - Multi-tab wizard: Lines, Processing, NMO/Velocity, Output
  - Config save/load functionality
  - Output selection (before/after gathers, stacks, difference)

- [x] Task 4.2: Processing Chain Widget (`views/processing_chain_widget.py`)
  - ProcessorInfo registry with param specs
  - ChainItemWidget with move up/down/remove controls
  - ProcessorParamEditor with type-specific widgets
  - Chain configuration save/load

- [x] Task 4.3: QC Batch Engine (`processors/qc_batch_engine.py`)
  - QCBatchEngine orchestrates full batch workflow
  - QCBatchWorker for background execution
  - BatchProgress and BatchResult dataclasses
  - Optional NMO integration with velocity model
  - Before/After stack generation with difference

- [x] Main Window Integration (`main_window.py`)
  - Added "Processing → QC Batch Processing..." menu item
  - Added "_open_qc_batch_processing()" handler with progress dialog

### Phase 5: Import Enhancement - COMPLETE
- [x] Task 5.1: QC Tab in Import Dialog (`views/segy_import_dialog.py`)
  - Restructured dialog to use QTabWidget with 4 tabs
  - Tab 1: Headers (header mapping + computed headers with QSplitter)
  - Tab 2: Import Settings (ensemble config + coordinate settings)
  - Tab 3: QC Lines (NEW - inline selection with quick-select buttons)
  - Tab 4: Preview (header preview + file statistics)
  - Added summary bar showing configuration overview
  - QC config stored in metadata.json after import

---

## Execution Order

```
Phase 1 (Foundation):
  1.1 Velocity I/O ─────┐
  1.2 NMO Processor ────┼──► Phase 2
  1.3 CDP Stacker ──────┘

Phase 2 (Core Workflow):
  2.1 QC Stacking Dialog ──┐
  2.2 QC Stacking Engine ──┼──► Phase 3
  2.3 Main Window Integration

Phase 3 (Visualization):
  3.1 QC Stack Viewer Window ──┐
  3.2 QC Statistics Panel ─────┼──► Complete MVP
  3.3 Viewer Integration ──────┘

Phase 4 (Advanced) - COMPLETE:
  4.1 QC Batch Dialog ────────┐
  4.2 Processing Chain Widget ┼──► Complete Advanced QC
  4.3 QC Batch Engine ────────┘

Phase 5 (Enhancement) - COMPLETE:
  5.1 Import QC Tab (dialog restructured with tabs)
```

---

## Files Summary

| File | Action | Priority | Status |
|------|--------|----------|--------|
| `utils/velocity_io.py` | EXTEND | P1 | DONE |
| `processors/nmo_processor.py` | CREATE | P1 | DONE |
| `processors/cdp_stacker.py` | CREATE | P1 | DONE |
| `views/qc_stacking_dialog.py` | CREATE | P2 | DONE |
| `processors/qc_stacking_engine.py` | CREATE | P2 | DONE |
| `views/qc_stack_viewer.py` | CREATE | P3 | DONE |
| `views/processing_chain_widget.py` | CREATE | P4 | DONE |
| `views/qc_batch_dialog.py` | CREATE | P4 | DONE |
| `processors/qc_batch_engine.py` | CREATE | P4 | DONE |
| `main_window.py` | MODIFY | P2/P3/P4 | DONE |
| `processors/__init__.py` | MODIFY | P1/P4 | DONE |
| `views/segy_import_dialog.py` | RESTRUCTURE | P5 | DONE |

---

## Success Criteria

1. **Velocity I/O:** Successfully load ASCII and SEG-Y velocity files ✓
2. **NMO:** Correct moveout with visible flattening of hyperbolic events ✓
3. **Stacking:** CDP stacks match expected fold and amplitude ✓
4. **Dialog:** User can configure and execute QC stacking workflow ✓
5. **Viewer:** Synchronized before/after/difference display with flip capability ✓
6. **Batch Processing:** Apply processor chain to selected gathers with before/after comparison ✓
7. **Import Enhancement:** Tabbed import dialog with QC lines configuration ✓
