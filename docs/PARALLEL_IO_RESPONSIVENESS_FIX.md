# Parallel I/O Responsiveness Fix - Architectural Design

## Problem Statement

When parallel SEGY import/export completes the worker phase, the application becomes unresponsive during file merging/finalizing stages. Linux displays "wait or force quit" dialogs, and the UI freezes for 30-90+ seconds, confusing users who don't know what's happening.

## Root Cause Analysis

### 1. Main Thread Blocking

The `coordinator.run()` method is called **synchronously from the main UI thread**:

```python
# In main_window.py / segy_import_dialog.py
result = coordinator.run(progress_callback=update_progress)  # BLOCKS main thread
```

UI responsiveness depends entirely on `QApplication.processEvents()` being called inside the progress callback. When the callback isn't invoked frequently, the UI freezes.

### 2. Missing Progress Callbacks During Critical Phases

**Export Coordinator (coordinator.py:229):**
```python
# BUG: merger.merge() has progress_callback param but it's NOT passed!
merger.merge(segment_paths, self.config.output_path)  # No progress updates
```

**Import Coordinator (lines 165-171):**
```python
# These operations have ZERO progress callbacks:
headers_path = self._merge_headers(output_dir, worker_results)    # Can be slow
self._create_indices(output_dir, n_traces)                        # Creates index
self._build_ensemble_index(output_dir, headers_path)              # O(n) scan
self._cleanup_segment_files(output_dir, worker_results)           # File I/O
self._save_metadata(output_dir, file_info, len(segments))
```

### 3. Blocking Operations Without Feedback

| Operation | Time (Large File) | Progress Updates |
|-----------|------------------|------------------|
| Export: Merge segments | 30-60s | NONE |
| Export: Cleanup temp files | 5-15s | NONE |
| Import: Merge headers | 10-30s | NONE |
| Import: Build ensemble index | 15-45s | NONE |
| Import: Create indices | 5-10s | NONE |

---

## Proposed Architecture

### Solution Overview: Sequential Stage Dialogs

Keep the current `QProgressDialog` for parallel computation (it works well), and add **separate dialogs for each post-processing stage**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARALLEL EXPORT FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Parallel Export        Stage 2: Merge Segments        │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │ Exporting traces... │   →     │ Merging files...    │        │
│  │ [████████████░░░░░] │         │ [████████░░░░░░░░░] │        │
│  │ 500K / 1M traces    │         │ 1.2 / 2.4 GB        │        │
│  │ Workers: 6 active   │         │ Segment 3 of 6      │        │
│  │ ETA: 2:30           │         │ ETA: 0:45           │        │
│  └─────────────────────┘         └─────────────────────┘        │
│           ↓                               ↓                     │
│  Stage 3: Cleanup                                               │
│  ┌─────────────────────┐                                        │
│  │ Cleaning up...      │                                        │
│  │ [██████████████░░░] │                                        │
│  │ File 5 of 12        │                                        │
│  └─────────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PARALLEL IMPORT FLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: Parallel Import        Stage 2: Merge Headers         │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │ Importing traces... │   →     │ Merging headers...  │        │
│  │ [████████████░░░░░] │         │ [████████░░░░░░░░░] │        │
│  │ 500K / 1M traces    │         │ Segment 3 of 6      │        │
│  │ Workers: 6 active   │         │                     │        │
│  └─────────────────────┘         └─────────────────────┘        │
│           ↓                               ↓                     │
│  Stage 3: Build Index            Stage 4: Cleanup               │
│  ┌─────────────────────┐         ┌─────────────────────┐        │
│  │ Building ensemble   │   →     │ Cleaning up...      │        │
│  │ index...            │         │ [██████████████░░░] │        │
│  │ [██████░░░░░░░░░░░] │         │ File 4 of 6         │        │
│  │ 300K / 1M traces    │         │                     │        │
│  └─────────────────────┘         └─────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Keep current dialog** - The existing `QProgressDialog` for parallel workers is accurate and well-designed
2. **One dialog per stage** - Each post-processing stage gets its own dialog with appropriate units
3. **Sequential flow** - Close previous dialog, open next one for each stage transition
4. **Stage-appropriate metrics** - Each dialog shows relevant progress (traces, bytes, files, etc.)

### Component 1: Stage-Specific Progress Helper

Create a utility class for easy stage dialog management:

```python
# utils/progress_helper.py
from PyQt6.QtWidgets import QProgressDialog, QApplication
from PyQt6.QtCore import Qt

class StageProgressDialog:
    """
    Creates and manages stage-specific progress dialogs.

    Usage:
        with StageProgressDialog("Merging files...", total_bytes, parent) as dialog:
            for chunk in chunks:
                process(chunk)
                dialog.update(bytes_done, f"{bytes_done/1e9:.2f} / {total_bytes/1e9:.2f} GB")
    """

    def __init__(self, title: str, maximum: int, parent=None,
                 cancel_text: str = "Cancel", unit: str = ""):
        self.title = title
        self.maximum = maximum
        self.parent = parent
        self.cancel_text = cancel_text
        self.unit = unit
        self.dialog = None
        self._cancelled = False

    def __enter__(self):
        self.dialog = QProgressDialog(self.title, self.cancel_text, 0, self.maximum, self.parent)
        self.dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.dialog.setMinimumDuration(0)  # Show immediately
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.show()
        QApplication.processEvents()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dialog:
            self.dialog.close()
        return False

    def update(self, value: int, label: str = None):
        """Update progress and process events to keep UI responsive."""
        if self.dialog.wasCanceled():
            self._cancelled = True
            return False

        self.dialog.setValue(value)
        if label:
            self.dialog.setLabelText(label)
        QApplication.processEvents()
        return True

    @property
    def was_cancelled(self) -> bool:
        return self._cancelled


def create_merge_dialog(total_bytes: int, parent=None) -> StageProgressDialog:
    """Create dialog for segment merging stage."""
    return StageProgressDialog(
        title="Merging segment files...",
        maximum=total_bytes,
        parent=parent,
        unit="bytes"
    )

def create_index_dialog(total_traces: int, parent=None) -> StageProgressDialog:
    """Create dialog for index building stage."""
    return StageProgressDialog(
        title="Building ensemble index...",
        maximum=total_traces,
        parent=parent,
        unit="traces"
    )

def create_cleanup_dialog(total_files: int, parent=None) -> StageProgressDialog:
    """Create dialog for cleanup stage."""
    return StageProgressDialog(
        title="Cleaning up temporary files...",
        maximum=total_files,
        parent=parent,
        unit="files"
    )

def create_header_merge_dialog(total_segments: int, parent=None) -> StageProgressDialog:
    """Create dialog for header merge stage."""
    return StageProgressDialog(
        title="Merging header files...",
        maximum=total_segments,
        parent=parent,
        unit="segments"
    )
```

### Component 2: Split Coordinator into Stages

Refactor coordinators to return control between stages, allowing the caller to manage separate dialogs:

#### Option A: Stage-Based Methods (Recommended)

Split `run()` into separate stage methods that can be called sequentially:

```python
# utils/parallel_export/coordinator.py

class ParallelExportCoordinator:
    """Refactored to support stage-based progress dialogs."""

    def run_parallel_export(
        self,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None
    ) -> Tuple[List[str], int, int]:
        """
        Stage 1: Run parallel workers only.

        Returns:
            Tuple of (segment_paths, n_traces, n_samples) for next stage
        """
        # ... existing phases 1-4 (init, vectorize, partition, workers) ...
        return segment_paths, n_traces, n_samples

    def run_merge(
        self,
        segment_paths: List[str],
        n_samples: int,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Stage 2: Merge segment files.

        Args:
            segment_paths: Paths from run_parallel_export()
            n_samples: Samples per trace
            progress_callback: callback(bytes_written, total_bytes)

        Returns:
            Total traces in merged file
        """
        merger = SEGYSegmentMerger(n_samples, self.data_format)
        return merger.merge(segment_paths, self.config.output_path, progress_callback)

    def run_cleanup(
        self,
        segment_paths: List[str],
        segments: List[TraceSegment],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stage 3: Cleanup temporary files.

        Args:
            segment_paths: Segment file paths to delete
            segments: Segment info (for header slice paths)
            progress_callback: callback(files_cleaned, total_files)
        """
        total_files = len(segment_paths) + len(segments)
        cleaned = 0

        for path in segment_paths:
            try:
                os.remove(path)
            except Exception:
                pass
            cleaned += 1
            if progress_callback:
                progress_callback(cleaned, total_files)

        for segment in segments:
            try:
                os.remove(self.temp_dir / f'headers_segment_{segment.segment_id}.pkl')
            except Exception:
                pass
            cleaned += 1
            if progress_callback:
                progress_callback(cleaned, total_files)

    # Keep run() for backwards compatibility - calls all stages internally
    def run(self, progress_callback=None) -> ExportResult:
        """Full pipeline (backwards compatible)."""
        ...
```

#### Option B: Callback Registry (Alternative)

Register separate callbacks for each stage:

```python
class ParallelExportCoordinator:
    def __init__(self, config: ExportConfig):
        self.config = config
        # Stage-specific callbacks
        self._callbacks = {
            'export': None,      # (ExportProgress) -> None
            'merge': None,       # (bytes_written, total_bytes) -> None
            'cleanup': None,     # (files_done, total_files) -> None
        }

    def set_stage_callback(self, stage: str, callback: Callable):
        """Register callback for specific stage."""
        self._callbacks[stage] = callback

    def run(self) -> ExportResult:
        """Run with stage-specific callbacks."""
        # During export phase
        if self._callbacks['export']:
            self._callbacks['export'](progress)

        # During merge phase
        merger.merge(paths, output, progress_callback=self._callbacks['merge'])

        # During cleanup phase
        for i, path in enumerate(paths):
            os.remove(path)
            if self._callbacks['cleanup']:
                self._callbacks['cleanup'](i + 1, total)
```

### Component 3: Caller Integration (main_window.py)

Show how the caller manages sequential dialogs:

```python
# main_window.py - on_export_parallel_segy()

def on_export_parallel_segy(self):
    """Export with separate progress dialogs per stage."""

    config = ExportConfig(...)
    coordinator = ParallelExportCoordinator(config)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: Parallel Export (existing dialog - keep as-is)
    # ═══════════════════════════════════════════════════════════════
    export_dialog = QProgressDialog(
        "Exporting traces...", "Cancel", 0, n_traces, self
    )
    export_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    export_dialog.setMinimumDuration(0)

    def on_export_progress(prog: ExportProgress):
        if export_dialog.wasCanceled():
            coordinator.cancel()
            return
        export_dialog.setValue(prog.current_traces)
        export_dialog.setLabelText(
            f"Exporting traces ({prog.active_workers} workers)...\n"
            f"{prog.current_traces:,} / {prog.total_traces:,}\n"
            f"ETA: {prog.eta_seconds:.0f}s"
        )
        QApplication.processEvents()

    try:
        # Run parallel export stage
        segment_paths, n_traces, n_samples = coordinator.run_parallel_export(
            progress_callback=on_export_progress
        )
    finally:
        export_dialog.close()

    if coordinator.was_cancelled:
        return

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Merge Segments (NEW separate dialog)
    # ═══════════════════════════════════════════════════════════════
    merge_stats = coordinator.get_merge_stats(segment_paths)
    total_bytes = merge_stats['output_size_bytes']

    merge_dialog = QProgressDialog(
        "Merging segment files...", "Cancel", 0, total_bytes, self
    )
    merge_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    merge_dialog.setMinimumDuration(0)

    def on_merge_progress(bytes_written: int, total: int):
        if merge_dialog.wasCanceled():
            return False
        merge_dialog.setValue(bytes_written)
        merge_dialog.setLabelText(
            f"Merging segment files...\n"
            f"{bytes_written / 1e9:.2f} / {total / 1e9:.2f} GB"
        )
        QApplication.processEvents()
        return True

    try:
        coordinator.run_merge(segment_paths, n_samples, progress_callback=on_merge_progress)
    finally:
        merge_dialog.close()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: Cleanup (NEW separate dialog)
    # ═══════════════════════════════════════════════════════════════
    total_files = len(segment_paths) + coordinator.n_workers

    cleanup_dialog = QProgressDialog(
        "Cleaning up temporary files...", None, 0, total_files, self
    )
    cleanup_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    cleanup_dialog.setMinimumDuration(0)
    cleanup_dialog.setCancelButton(None)  # No cancel for cleanup

    def on_cleanup_progress(done: int, total: int):
        cleanup_dialog.setValue(done)
        cleanup_dialog.setLabelText(f"Cleaning up... {done}/{total} files")
        QApplication.processEvents()

    try:
        coordinator.run_cleanup(segment_paths, segments, progress_callback=on_cleanup_progress)
    finally:
        cleanup_dialog.close()

    # Done!
    QMessageBox.information(self, "Export Complete", f"Exported {n_traces:,} traces")
```

### Component 4: Import Coordinator Stage Split

Similar refactoring for import:

```python
# utils/segy_import/multiprocess_import/coordinator.py

class ParallelImportCoordinator:

    def run_parallel_import(self, progress_callback=None) -> Tuple[List[WorkerResult], int]:
        """Stage 1: Parallel import workers."""
        # ... phases 1-4 ...
        return worker_results, n_traces

    def run_merge_headers(
        self,
        worker_results: List[WorkerResult],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Stage 2: Merge header parquet files.

        Args:
            worker_results: Results from run_parallel_import()
            progress_callback: callback(segments_merged, total_segments)

        Returns:
            Path to merged headers.parquet
        """
        sorted_results = sorted(worker_results, key=lambda r: r.segment_id)
        all_dfs = []

        for i, result in enumerate(sorted_results):
            if result.headers_path and Path(result.headers_path).exists():
                df = pd.read_parquet(result.headers_path)
                all_dfs.append(df)

            if progress_callback:
                progress_callback(i + 1, len(sorted_results))

        # ... rest of merge logic ...
        return str(final_path)

    def run_build_index(
        self,
        headers_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stage 3: Build ensemble index.

        Args:
            headers_path: Path to merged headers
            progress_callback: callback(traces_scanned, total_traces)
        """
        # Load ensemble column
        headers_df = pd.read_parquet(headers_path, columns=[self.config.ensemble_key])
        ensemble_col = headers_df[self.config.ensemble_key].values
        n_total = len(ensemble_col)

        # Report every 1% or 10K traces, whichever is smaller
        report_interval = min(10000, max(1, n_total // 100))

        ensembles = []
        current_value = ensemble_col[0]
        start_trace = 0
        ensemble_id = 0

        for i in range(1, n_total):
            if ensemble_col[i] != current_value:
                ensembles.append({...})
                ensemble_id += 1
                current_value = ensemble_col[i]
                start_trace = i

            # Progress callback
            if i % report_interval == 0 and progress_callback:
                progress_callback(i, n_total)

        # ... save to parquet ...

    def run_cleanup(
        self,
        worker_results: List[WorkerResult],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """Stage 4: Cleanup segment header files."""
        total = len(worker_results)
        for i, result in enumerate(worker_results):
            if result.headers_path:
                Path(result.headers_path).unlink(missing_ok=True)
            if progress_callback:
                progress_callback(i + 1, total)
```

### Component 5: Import Dialog Flow

```python
# views/segy_import_dialog.py

def _import_streaming(self, output_dir, file_info, ensemble_keys):
    """Import with separate stage dialogs."""

    coordinator = ParallelImportCoordinator(config)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: Parallel Import (existing dialog)
    # ═══════════════════════════════════════════════════════════════
    import_dialog = QProgressDialog("Importing...", "Cancel", 0, n_traces, self)
    # ... existing import dialog code ...

    worker_results, n_traces = coordinator.run_parallel_import(on_import_progress)
    import_dialog.close()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Merge Headers (NEW)
    # ═══════════════════════════════════════════════════════════════
    n_segments = len(worker_results)
    merge_dialog = QProgressDialog(
        "Merging header files...", None, 0, n_segments, self
    )
    merge_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    merge_dialog.setCancelButton(None)

    def on_merge_progress(done, total):
        merge_dialog.setValue(done)
        merge_dialog.setLabelText(f"Merging headers... segment {done}/{total}")
        QApplication.processEvents()

    headers_path = coordinator.run_merge_headers(worker_results, on_merge_progress)
    merge_dialog.close()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: Build Ensemble Index (NEW)
    # ═══════════════════════════════════════════════════════════════
    index_dialog = QProgressDialog(
        "Building ensemble index...", None, 0, n_traces, self
    )
    index_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    index_dialog.setCancelButton(None)

    def on_index_progress(done, total):
        index_dialog.setValue(done)
        index_dialog.setLabelText(
            f"Building index... {done:,}/{total:,} traces scanned"
        )
        QApplication.processEvents()

    coordinator.run_build_index(headers_path, on_index_progress)
    index_dialog.close()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: Cleanup (NEW)
    # ═══════════════════════════════════════════════════════════════
    cleanup_dialog = QProgressDialog(
        "Cleaning up...", None, 0, n_segments, self
    )
    cleanup_dialog.setCancelButton(None)

    def on_cleanup_progress(done, total):
        cleanup_dialog.setValue(done)
        QApplication.processEvents()

    coordinator.run_cleanup(worker_results, on_cleanup_progress)
    cleanup_dialog.close()
```

---

## Implementation Plan

### Phase 1: Refactor Coordinators (Core Changes)

**Export Coordinator:**
1. Split `run()` into stage methods:
   - `run_parallel_export()` - returns segment_paths, n_traces, n_samples, segments
   - `run_merge()` - accepts progress_callback(bytes_written, total_bytes)
   - `run_cleanup()` - accepts progress_callback(files_done, total_files)
2. Keep `run()` as backwards-compatible wrapper
3. Add `get_merge_stats()` method for pre-calculating merge size

**Import Coordinator:**
1. Split `run()` into stage methods:
   - `run_parallel_import()` - returns worker_results, n_traces
   - `run_merge_headers()` - accepts progress_callback(segments_done, total)
   - `run_build_index()` - accepts progress_callback(traces_scanned, total)
   - `run_cleanup()` - accepts progress_callback(files_done, total)
2. Keep `run()` as backwards-compatible wrapper

### Phase 2: Update UI Callers

**main_window.py - on_export_parallel_segy():**
1. Keep existing dialog for parallel export stage
2. Add merge dialog with GB progress
3. Add cleanup dialog with file count

**segy_import_dialog.py - _import_streaming():**
1. Keep existing dialog for parallel import stage
2. Add header merge dialog with segment count
3. Add index building dialog with trace count
4. Add cleanup dialog with file count

### Phase 3: Progress Helper Utilities (Optional)

Create `utils/progress_helper.py` with:
- `StageProgressDialog` context manager
- Factory functions for common stage types

---

## Stage Dialog Specifications

### Export Stages

| Stage | Dialog Title | Progress Unit | Max Value | Cancel? |
|-------|--------------|---------------|-----------|---------|
| 1. Parallel Export | "Exporting traces..." | traces | n_traces | Yes |
| 2. Merge Segments | "Merging segment files..." | bytes | total_bytes | Yes |
| 3. Cleanup | "Cleaning up..." | files | n_files | No |

### Import Stages

| Stage | Dialog Title | Progress Unit | Max Value | Cancel? |
|-------|--------------|---------------|-----------|---------|
| 1. Parallel Import | "Importing traces..." | traces | n_traces | Yes |
| 2. Merge Headers | "Merging header files..." | segments | n_segments | No |
| 3. Build Index | "Building ensemble index..." | traces | n_traces | No |
| 4. Cleanup | "Cleaning up..." | files | n_files | No |

### Dialog Label Formats

```
Export Merge:     "Merging segment files...\n1.2 / 2.4 GB"
Import Merge:     "Merging headers... segment 3/6"
Build Index:      "Building index... 500,000/1,000,000 traces scanned"
Cleanup:          "Cleaning up... 4/12 files"
```

---

## Code Locations to Modify

### Export System

| File | Change Description |
|------|-------------------|
| `utils/parallel_export/coordinator.py` | Split `run()` into `run_parallel_export()`, `run_merge()`, `run_cleanup()` |
| `utils/parallel_export/coordinator.py` | Add `get_merge_stats()` method |
| `utils/parallel_export/merger.py` | Already has progress_callback - no change needed |
| `main_window.py` | Add separate dialogs for merge and cleanup stages |

### Import System

| File | Change Description |
|------|-------------------|
| `utils/segy_import/multiprocess_import/coordinator.py` | Split `run()` into stage methods |
| `utils/segy_import/multiprocess_import/coordinator.py` | Add progress to `_build_ensemble_index()` |
| `views/segy_import_dialog.py` | Add separate dialogs for merge, index, cleanup stages |

### Optional New File

| File | Purpose |
|------|---------|
| `utils/progress_helper.py` | `StageProgressDialog` context manager (convenience) |

---

## Testing Checklist

- [ ] Export 1M+ trace file - separate dialogs appear for each stage
- [ ] Export merge dialog shows GB progress accurately
- [ ] Import 1M+ trace file - separate dialogs for all 4 stages
- [ ] Index building dialog updates smoothly (every 1%)
- [ ] No UI freeze at any stage transition
- [ ] Cancel works during parallel stages
- [ ] Post-processing stages complete without "force quit" dialog
- [ ] Cleanup dialog shows file count progress
- [ ] Works on Linux (primary), Windows, macOS

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Dialog flickering on fast stages | Use `setMinimumDuration(500)` for short stages |
| Stage method API breaking changes | Keep `run()` as backwards-compatible wrapper |
| Progress callback overhead | Report at intervals (1%, 10K traces, etc.) not every item |
| Error handling across stages | Wrap each stage in try/finally, cleanup on any failure |

---

## Summary

The architecture uses **separate progress dialogs per stage** rather than one combined dialog:

1. **Keep existing dialog** - The current parallel worker dialog is accurate and works well
2. **Add stage dialogs** - Merge, index building, and cleanup each get their own dialog
3. **Stage-appropriate metrics** - Each dialog shows relevant units (GB, traces, files)
4. **No UI freeze** - `QApplication.processEvents()` called during all stages
5. **Clear user feedback** - User always knows what's happening

### Key Changes:

| Component | Before | After |
|-----------|--------|-------|
| Export | 1 dialog, freeze during merge | 3 dialogs: export → merge → cleanup |
| Import | 1 dialog, freeze during index | 4 dialogs: import → merge → index → cleanup |
| Coordinator API | Single `run()` method | Stage methods + backwards-compatible `run()` |
