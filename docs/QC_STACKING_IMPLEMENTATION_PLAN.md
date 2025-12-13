# QC Stacking Implementation Plan

## Overview

This document outlines the implementation of comprehensive QC (Quality Control) stacking functionality for the SeisProc application. The feature set enables users to:
- Define QC inlines during import for targeted quality analysis
- Create CDP stacks using NMO correction with velocity models from various sources
- Perform selective batch processing on QC-relevant gathers only
- Compare before/after processing results with difference analysis

---

## Understanding of Requirements

### 1. Import Phase - QC Line Selection
**User Need:** During SEG-Y import, specify inline numbers that will serve as QC reference lines.

**Implementation Approach:**
- Add QC inline selection UI to the import dialog
- Create a separate index structure (`qc_inlines_index.parquet`) mapping QC inlines to their contributing CDP gathers
- Optionally extract and store QC line traces as a separate lightweight dataset for quick access
- Store QC configuration in metadata.json for persistence

### 2. CDP Stacking Tool with NMO
**User Need:** Stack CDP gathers using NMO correction with velocities from ASCII text files or SEG-Y files, with proper interpolation/extrapolation.

**Implementation Approach:**
- Create `NMOProcessor` for normal moveout correction
- Create `CDPStackProcessor` that combines NMO + stack
- Implement velocity readers for:
  - ASCII text files (time-velocity pairs, or CDP-time-velocity triplets)
  - SEG-Y velocity files (velocity cubes or 2D sections)
- Implement velocity interpolation:
  - Temporal: interpolate V(t) for any time sample
  - Spatial: interpolate V(cdp) or V(inline, xline) for any CDP location
  - Extrapolation: constant extrapolation at boundaries with optional linear extension

### 3. Processing - QC Stacking Procedure
**User Need:** In the Processing menu, add a procedure where users select QC lines and velocity models, then generate stacks for those lines.

**Implementation Approach:**
- Create `QCStackingDialog` for configuration:
  - Line selection (from predefined QC lines or manual entry)
  - Velocity model selection (file browser + preview)
  - Stacking parameters (mute, stretch limit, fold cutoff)
- Execute stacking workflow:
  - Identify all CDP gathers contributing to selected lines
  - Apply NMO using interpolated velocities
  - Stack traces within each CDP
  - Assemble into inline sections
- Output QC stack dataset with metadata

### 4. Processing - Batch Processing for QC Lines
**User Need:** User selects inlines, app identifies contributing gathers, processes only those gathers, outputs both processed gathers and stacked QC inlines.

**Implementation Approach:**
- Create `QCBatchProcessingDialog`:
  - Inline selection interface
  - Processing chain configuration (reuse existing processors)
  - Velocity model for final stacking
  - Output options (gathers, stacks, or both)
- Implement selective gather identification:
  - Query ensemble index to find gathers contributing to selected inlines
  - Build processing subset (significantly faster than full dataset)
- Execute batch workflow:
  - Process identified gathers through configured processing chain
  - Output processed gathers (optional)
  - Apply NMO + stack for QC inline output
  - Generate before/after comparison data

### 5. View - QC Stacks Viewer
**User Need:** Load QC stacks, calculate differences to assess noise impact, flip between before/after/difference views using the same viewer functionality as gathers.

**Implementation Approach:**
- Create `QCStackViewer` (reuse `SeismicViewerPyQtGraph` components)
- Three-panel layout similar to main window:
  - Before stack (original or previous processing stage)
  - After stack (current processing result)
  - Difference (noise removed / artifacts introduced)
- Add controls:
  - Load stack files (before/after)
  - Auto-compute difference
  - Synchronized navigation (inline scroll)
  - Amplitude normalization options (individual vs. shared scale)
- Flip/toggle functionality for rapid A/B comparison

---

## Tangible Implementation Tasks

### Phase 1: Core Infrastructure (Foundation)

#### Task 1.1: Velocity Model I/O
- [ ] Create `utils/velocity_io.py` with readers:
  - `read_velocity_ascii(filepath)` - Parse ASCII velocity files (T-V pairs, CDP-T-V triplets)
  - `read_velocity_segy(filepath)` - Extract velocities from SEG-Y
  - `VelocityField` class for 2D/3D velocity storage
- [ ] Implement interpolation in `models/velocity_model.py`:
  - `interpolate_temporal(time_samples)` - V(t) at any time
  - `interpolate_spatial(cdp_x, cdp_y)` or `interpolate_spatial(inline, xline)`
  - `extrapolate_boundaries(method='constant'|'linear')`
- [ ] Add unit tests for velocity I/O and interpolation

#### Task 1.2: NMO Processor
- [ ] Create `processors/nmo_processor.py`:
  - `NMOProcessor(velocity_model, stretch_mute=1.5)`
  - `process(gather_data, offsets)` - Apply NMO correction
  - Handle stretch muting (configurable limit)
  - Support both forward NMO and inverse NMO
- [ ] Implement efficient NMO calculation:
  - Vectorized time shift computation: `t_nmo = sqrt(t0^2 + (offset/v)^2)`
  - Sinc interpolation for accurate amplitude preservation
  - Optional linear interpolation for speed
- [ ] Add unit tests with synthetic gathers

#### Task 1.3: CDP Stacker
- [ ] Create `processors/cdp_stacker.py`:
  - `CDPStacker(velocity_model, nmo_params, stack_params)`
  - `stack_gather(gather_data, headers)` - NMO + stack single CDP
  - `stack_line(gathers, headers, inline_num)` - Stack entire inline
- [ ] Implement stacking options:
  - Mean stack (default)
  - Median stack (robust to outliers)
  - Weighted stack (by fold or SNR)
  - Diversity stack (noise-adaptive)
- [ ] Add fold calculation and output

### Phase 2: Import Enhancement

#### Task 2.1: QC Inline Selection UI
- [ ] Modify `views/segy_import_dialog.py`:
  - Add "QC Lines" tab/section to import wizard
  - Inline number input (comma-separated or range notation: "100,200,300-350")
  - Preview inline locations on survey geometry (if available)
  - Validation against actual inline range in data
- [ ] Store QC configuration in import config

#### Task 2.2: QC Index Generation
- [ ] Modify `utils/segy_import/multiprocess_import/coordinator.py`:
  - Add `_build_qc_index()` method after ensemble index
  - Create `qc_inlines_index.parquet`:
    - Columns: qc_inline, ensemble_ids[], start_trace, end_trace, n_traces
  - Map each QC inline to contributing CDP gathers
- [ ] Update `metadata.json` with QC configuration:
  - `qc_inlines`: List of QC inline numbers
  - `qc_index_path`: Path to QC index file

#### Task 2.3: QC Data Extraction (Optional)
- [ ] Add option to extract QC line traces to separate dataset:
  - Smaller Zarr file with only QC-relevant traces
  - Faster loading for QC-only workflows
  - Link to parent dataset for full processing

### Phase 3: QC Stacking Procedure

#### Task 3.1: QC Stacking Dialog
- [ ] Create `views/qc_stacking_dialog.py`:
  - Multi-page wizard or tabbed dialog
  - Page 1: Line Selection
    - List of predefined QC lines (from import)
    - Manual inline entry
    - Multi-select capability
  - Page 2: Velocity Configuration
    - File type selector (ASCII/SEG-Y)
    - File browser with preview
    - Interpolation method selection
    - Velocity QC display (time-velocity plot)
  - Page 3: Stacking Parameters
    - NMO stretch mute limit (default 1.5)
    - Minimum fold cutoff
    - Stack method (mean/median/weighted)
    - Output format and location
  - Page 4: Preview & Execute
    - Estimated processing time
    - Output summary
    - Execute button with progress

#### Task 3.2: QC Stacking Engine
- [ ] Create `processors/qc_stacking_engine.py`:
  - `QCStackingEngine(config)` - Orchestrates QC stacking workflow
  - `run(progress_callback)` - Execute stacking
  - Methods:
    - `_load_velocity_model()` - Load and validate velocities
    - `_identify_gathers(inline_nums)` - Find contributing CDPs
    - `_stack_inline(inline_num)` - Process single inline
    - `_assemble_output()` - Combine into output dataset
- [ ] Implement parallel inline processing (one inline per worker)
- [ ] Output format: Zarr + metadata for viewer compatibility

#### Task 3.3: Integration with Main Window
- [ ] Add "QC Stacking..." menu item under Processing menu
- [ ] Connect to QC Stacking Dialog
- [ ] Handle output dataset creation and optional auto-load

### Phase 4: Batch Processing for QC Lines

#### Task 4.1: QC Batch Processing Dialog
- [ ] Create `views/qc_batch_dialog.py`:
  - Inline selection (similar to QC Stacking Dialog)
  - Processing chain configuration:
    - Reuse existing processor selection UI
    - Chain multiple processors (e.g., bandpass → FK filter → denoise)
    - Parameter configuration for each processor
  - Velocity model for final stacking
  - Output configuration:
    - Checkbox: Output processed gathers
    - Checkbox: Output stacked QC lines
    - Output directory selection

#### Task 4.2: Selective Gather Identification
- [ ] Create `utils/qc_gather_selector.py`:
  - `QCGatherSelector(dataset_path, qc_index_path)`
  - `get_contributing_gathers(inline_nums)` → List of ensemble_ids
  - `get_gather_inline_mapping()` → Dict mapping gather_id to inline contributions
  - Efficient query using parquet filtering

#### Task 4.3: QC Batch Processing Engine
- [ ] Create `processors/qc_batch_engine.py`:
  - `QCBatchEngine(config)`:
    - `inline_nums`: List of target inlines
    - `processing_chain`: List of processor configs
    - `velocity_model`: For final stacking
    - `output_config`: What to output
  - `run(progress_callback)`:
    - Phase 1: Identify contributing gathers (fast)
    - Phase 2: Process gathers through chain (main work)
    - Phase 3: Stack processed gathers to QC inlines
    - Phase 4: Write outputs
  - Memory-efficient chunked processing (reuse existing pattern)

#### Task 4.4: Before/After Data Management
- [ ] Implement comparison data generation:
  - Store "before" stack (original data, no processing)
  - Store "after" stack (processed data)
  - Metadata linking the two for viewer

### Phase 5: QC Stack Viewer

#### Task 5.1: QC Stack Viewer Widget
- [ ] Create `views/qc_stack_viewer.py`:
  - Reuse `SeismicViewerPyQtGraph` for display
  - Three-panel layout (Before / After / Difference)
  - Shared `ViewportState` for synchronization
  - Inline navigation controls (slider + spinbox)

#### Task 5.2: QC Stack Data Manager
- [ ] Create `models/qc_stack_data.py`:
  - `QCStackDataset` class:
    - `before_stack`: SeismicData or LazySeismicData
    - `after_stack`: SeismicData or LazySeismicData
    - `difference`: Computed on-demand or cached
    - `inline_index`: Available inlines and their positions
  - Methods:
    - `load_inline(inline_num)` → (before, after, diff) traces
    - `compute_difference(normalize=True)` → difference data
    - `get_statistics()` → RMS difference, correlation, etc.

#### Task 5.3: QC Stack Viewer Controls
- [ ] Add control panel for QC Stack Viewer:
  - Before/After file loaders
  - Inline navigation (dropdown + prev/next buttons)
  - Display mode toggle:
    - "Before" only
    - "After" only
    - "Difference" only
    - "Side-by-side" (all three)
    - "Flip" mode (toggle between before/after on keypress)
  - Normalization options:
    - Independent scaling per panel
    - Shared amplitude scale
    - Difference-optimized scale (±N std)
  - Statistics display (RMS diff, % change, fold)

#### Task 5.4: Flip/Compare Functionality
- [ ] Implement rapid A/B comparison:
  - Keyboard shortcut (Space or Tab) to flip between before/after
  - Animated transition option (fade or instant)
  - Cursor tracking across flip (highlight same location)
- [ ] Add difference metrics overlay:
  - Per-trace RMS difference
  - Time-windowed statistics
  - Anomaly highlighting (large differences)

#### Task 5.5: Integration
- [ ] Add "QC Stacks" menu item under View menu
- [ ] Create `QCStackViewerWindow` as standalone window
- [ ] Recent QC stacks list for quick access
- [ ] Optional: Dock into main window as additional tab

### Phase 6: Testing & Documentation

#### Task 6.1: Unit Tests
- [ ] Test velocity I/O (ASCII, SEG-Y formats)
- [ ] Test NMO correction accuracy
- [ ] Test CDP stacking (synthetic data)
- [ ] Test QC index generation
- [ ] Test selective gather identification

#### Task 6.2: Integration Tests
- [ ] End-to-end QC import workflow
- [ ] End-to-end QC stacking workflow
- [ ] End-to-end batch processing workflow
- [ ] Viewer comparison functionality

#### Task 6.3: Documentation
- [ ] Update user guide with QC workflows
- [ ] Add tooltips and help text to dialogs
- [ ] Create example velocity file formats

---

## Data Flow Diagrams

### QC Import Flow
```
SEG-Y File
    ↓
[Import Dialog]
    ├── Header Mapping
    ├── Ensemble Config
    └── QC Lines Selection ← NEW
            ↓
[Parallel Import]
    ├── traces.zarr
    ├── headers.parquet
    ├── ensemble_index.parquet
    ├── qc_inlines_index.parquet  ← NEW
    └── metadata.json (with qc_config)
```

### QC Stacking Flow
```
[QC Stacking Dialog]
    ├── Select QC Inlines
    ├── Load Velocity Model (ASCII/SEG-Y)
    └── Configure Stack Parameters
            ↓
[QC Stacking Engine]
    ├── Query qc_inlines_index → Gather IDs
    ├── For each CDP gather:
    │   ├── Load traces
    │   ├── Apply NMO (interpolated velocity)
    │   └── Stack → single trace
    └── Assemble inline sections
            ↓
[Output]
    ├── qc_stacks.zarr (stacked inlines)
    └── qc_metadata.json
```

### QC Batch Processing Flow
```
[QC Batch Dialog]
    ├── Select QC Inlines
    ├── Configure Processing Chain
    ├── Load Velocity Model
    └── Select Outputs
            ↓
[QC Batch Engine]
    ├── Identify contributing gathers (subset)
    ├── Stack original gathers → "Before" stack
    ├── Process gathers through chain
    ├── Optionally output processed gathers
    ├── Stack processed gathers → "After" stack
    └── Package before/after for viewer
            ↓
[Outputs]
    ├── processed_gathers/ (optional)
    ├── qc_before_stack.zarr
    ├── qc_after_stack.zarr
    └── qc_comparison_metadata.json
```

### QC Viewer Flow
```
[QC Stack Viewer]
    ├── Load Before Stack
    ├── Load After Stack
    └── Compute Difference (on-demand)
            ↓
[Three Synchronized Viewers]
    ├── Before Panel (SeismicViewerPyQtGraph)
    ├── After Panel (SeismicViewerPyQtGraph)
    └── Difference Panel (SeismicViewerPyQtGraph)
            ↓
[Controls]
    ├── Inline Navigation
    ├── Display Mode (flip/side-by-side)
    └── Normalization Options
```

---

## File Structure (New Files)

```
SeisProc/
├── models/
│   ├── qc_stack_data.py          # QC stack dataset management
│   └── velocity_model.py          # Enhanced with interpolation
├── processors/
│   ├── nmo_processor.py           # NMO correction
│   ├── cdp_stacker.py             # CDP stacking
│   ├── qc_stacking_engine.py      # QC stacking orchestrator
│   └── qc_batch_engine.py         # QC batch processing orchestrator
├── views/
│   ├── qc_stacking_dialog.py      # QC stacking configuration UI
│   ├── qc_batch_dialog.py         # QC batch processing UI
│   ├── qc_stack_viewer.py         # QC stack comparison viewer
│   └── velocity_preview_widget.py # Velocity visualization
├── utils/
│   ├── velocity_io.py             # Velocity file I/O
│   ├── qc_gather_selector.py      # Selective gather identification
│   └── qc_index_builder.py        # QC index generation
└── docs/
    └── QC_STACKING_IMPLEMENTATION_PLAN.md  # This file
```

---

## Priority & Dependencies

### High Priority (Core Functionality)
1. **Task 1.1** Velocity I/O → Required by all stacking
2. **Task 1.2** NMO Processor → Required by stacking
3. **Task 1.3** CDP Stacker → Core stacking logic
4. **Task 3.1-3.3** QC Stacking Procedure → Main user workflow

### Medium Priority (Enhanced Workflow)
5. **Task 2.1-2.3** Import Enhancement → Better UX but can work without
6. **Task 4.1-4.4** Batch Processing → Advanced workflow
7. **Task 5.1-5.5** QC Viewer → Essential for QC analysis

### Lower Priority (Polish)
8. **Task 6.1-6.3** Testing & Documentation

### Dependency Graph
```
Velocity I/O (1.1)
    ↓
NMO Processor (1.2) ←──────────────────┐
    ↓                                   │
CDP Stacker (1.3)                       │
    ↓                                   │
┌───┴───────────────┐                   │
↓                   ↓                   │
QC Stacking      QC Batch              │
(3.1-3.3)        (4.1-4.4)             │
    ↓                   ↓               │
    └───────┬───────────┘               │
            ↓                           │
      QC Viewer (5.1-5.5) ──────────────┘

Import Enhancement (2.1-2.3) → Independent, enhances all workflows
```

---

## Estimated Complexity

| Phase | Tasks | Complexity | Notes |
|-------|-------|------------|-------|
| 1. Core Infrastructure | 1.1-1.3 | Medium | NMO math is well-established |
| 2. Import Enhancement | 2.1-2.3 | Low-Medium | Extends existing dialog |
| 3. QC Stacking | 3.1-3.3 | Medium | New dialog + engine |
| 4. Batch Processing | 4.1-4.4 | Medium-High | Complex workflow orchestration |
| 5. QC Viewer | 5.1-5.5 | Medium | Reuses existing viewer code |
| 6. Testing & Docs | 6.1-6.3 | Low | Standard practice |

---

## Success Criteria

1. **Import**: User can specify QC inlines during import, system creates appropriate indices
2. **Velocity**: Application correctly reads ASCII and SEG-Y velocity files, interpolates to any CDP/time location
3. **NMO**: Correct moveout correction with configurable stretch muting
4. **Stacking**: CDP stacks match expected results (verified against synthetic data)
5. **QC Stacking**: User can select lines, load velocities, and generate QC stacks in under 5 minutes for typical datasets
6. **Batch Processing**: Selective processing reduces runtime by 80%+ compared to full dataset processing
7. **Viewer**: Smooth flip between before/after, difference clearly shows processing impact
