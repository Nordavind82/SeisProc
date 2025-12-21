# Phase 4: Job Integration Migration Plan

## Executive Summary

Migrate all parallel processing operations from direct `ParallelProcessingCoordinator` usage to the Phase 4 Job Management system (`ProcessingJobAdapter` + `JobManager`).

**Impact:**
- Full job visibility in Job Dashboard
- Toast notifications on complete/failure  
- Job history and analytics
- Multi-level cancellation with pause/resume
- Unified processing API

---

## Current State

### Files Involved

| File | Current Usage | Target State |
|------|--------------|--------------|
| `main_window.py:_batch_process_parallel()` | Direct `ParallelProcessingCoordinator` | Use `ProcessingJobAdapter` |
| `main_window.py:_parallel_export_segy()` | Direct multiprocess export | Use `SEGYExportJobAdapter` |
| `views/pstm_wizard_dialog.py` | Custom processing | Use `ProcessingJobAdapter` |
| `processors/qc_batch_engine.py` | Custom thread | Use `ProcessingJobAdapter` |

### Code to Migrate (~2000 lines affected)

```
main_window.py:
  - _batch_process_parallel()     [lines 2379-2996] → ProcessingJobAdapter
  - _parallel_export_segy()       [lines 2999-3400] → SEGYExportJobAdapter
  - Legacy: _batch_process_all_gathers() [deprecate further]
  - Legacy: _batch_process_and_export_streaming() [deprecate further]

views/pstm_wizard_dialog.py:
  - Migration processing          → ProcessingJobAdapter

processors/qc_batch_engine.py:  
  - QCBatchWorker thread          → ProcessingJobAdapter
```

---

## Migration Steps

### Phase A: Core Infrastructure (2 tasks)

#### A.1: Enhance ProcessingJobAdapter for Qt Integration

**File:** `utils/ray_orchestration/processing_job_adapter.py`

```python
# Add Qt signal emission support
class ProcessingJobAdapter:
    def __init__(self, processing_config, job_name=None, qt_bridge=None):
        ...
        self._qt_bridge = qt_bridge  # Optional Qt bridge for UI updates
    
    def _emit_progress(self, progress: JobProgress):
        """Emit progress to JobManager and Qt bridge."""
        self._manager.update_progress(self._job.id, progress)
        if self._qt_bridge:
            self._qt_bridge.emit_progress(self._job.id, progress.to_dict())
```

**Changes:**
1. Add optional `qt_bridge` parameter
2. Add `_emit_progress()` method that calls both JobManager and Qt bridge
3. Ensure progress updates flow to Dashboard

#### A.2: Add High-Level Processing API

**File:** `utils/ray_orchestration/processing_api.py` (NEW)

```python
"""
High-level processing API for SeisProc.

Provides simple functions that handle all job management internally.
"""

def run_parallel_processing(
    input_dir: str,
    output_dir: str,
    processor_config: dict,
    n_workers: int = None,
    progress_callback: Callable = None,
    qt_parent: QWidget = None,
) -> ProcessingResult:
    """
    Run parallel processing with full job management.
    
    Handles:
    - Job creation and tracking
    - Progress updates to Dashboard
    - Toast notifications
    - Job history storage
    - Cancellation support
    """
    from .processing_job_adapter import ProcessingJobAdapter
    from .qt_bridge import get_job_bridge
    from utils.parallel_processing import ProcessingConfig
    
    bridge = get_job_bridge() if qt_parent else None
    
    config = ProcessingConfig(
        input_storage_dir=input_dir,
        output_storage_dir=output_dir,
        processor_config=processor_config,
        n_workers=n_workers or get_optimal_workers(),
    )
    
    adapter = ProcessingJobAdapter(config, qt_bridge=bridge)
    return adapter.run(progress_callback=progress_callback)


def run_segy_export(
    input_dir: str,
    output_file: str,
    template_segy: str,
    n_workers: int = None,
    progress_callback: Callable = None,
) -> ExportResult:
    """Run parallel SEG-Y export with full job management."""
    from .segy_job_adapter import SEGYExportJobAdapter
    ...
```

---

### Phase B: Main Window Migration (4 tasks)

#### B.1: Migrate _batch_process_parallel()

**File:** `main_window.py`

**Before (current):**
```python
def _batch_process_parallel(self):
    ...
    coordinator = ParallelProcessingCoordinator(config)
    result = coordinator.run(progress_callback=on_progress)
```

**After (migrated):**
```python
def _batch_process_parallel(self):
    ...
    from utils.ray_orchestration.processing_api import run_parallel_processing
    
    # Create adapter with Qt bridge for Dashboard updates
    adapter = ProcessingJobAdapter(
        config, 
        job_name=f"Batch: {self.last_processor.get_description()}",
        qt_bridge=self._get_job_bridge()
    )
    
    # Submit and run (job appears in Dashboard immediately)
    job = adapter.submit()
    
    # Show in Dashboard
    if self._job_integration:
        self._job_integration.show_job_dashboard()
    
    # Run with progress
    result = adapter.run(progress_callback=on_progress)
    
    # Toast notification handled automatically by AlertToastBridge
```

**Key Changes:**
1. Replace direct `ParallelProcessingCoordinator` with `ProcessingJobAdapter`
2. Job appears in Dashboard during processing
3. Remove manual QProgressDialog (Dashboard shows progress)
4. Toast notifications automatic via AlertManager

#### B.2: Migrate _parallel_export_segy()

**File:** `main_window.py`

Similar migration using `SEGYExportJobAdapter`.

#### B.3: Add Helper Methods to MainWindow

```python
def _get_job_bridge(self):
    """Get the job bridge for Qt signal integration."""
    if hasattr(self, '_job_integration') and self._job_integration:
        return get_job_bridge()
    return None

def _show_processing_in_dashboard(self, job: Job):
    """Show job dashboard and highlight the job."""
    if self._job_integration:
        self._job_integration.show_job_dashboard()
```

#### B.4: Update Progress Display Strategy

**Option A: Keep QProgressDialog (Minimal Change)**
- Keep existing dialog for modal blocking
- Also emit to Dashboard for visibility
- User can close dialog, processing continues

**Option B: Dashboard-Only (Recommended)**
- Remove QProgressDialog
- Show Job Dashboard automatically
- Non-blocking processing
- User can continue using app while processing

---

### Phase C: Secondary Migrations (3 tasks)

#### C.1: PSTM Wizard Dialog

**File:** `views/pstm_wizard_dialog.py`

Wrap migration processing with `ProcessingJobAdapter`.

#### C.2: QC Batch Engine

**File:** `processors/qc_batch_engine.py`

Replace `QCBatchWorker` thread with `ProcessingJobAdapter`.

#### C.3: Individual Processor Apply

Consider whether single-gather processing should also appear in Dashboard (probably not - too noisy).

---

### Phase D: Legacy Code Removal (2 tasks)

#### D.1: Deprecate Legacy Methods

Add deprecation warnings to:
```python
def _batch_process_all_gathers(self):
    """DEPRECATED: Use _batch_process_parallel() instead."""
    warnings.warn(
        "_batch_process_all_gathers is deprecated. "
        "Use Parallel Batch Process (Ctrl+Shift+B)",
        DeprecationWarning
    )
    # Keep working for now, remove in next version
```

#### D.2: Remove Legacy Code (Future)

After 1-2 release cycles, remove:
- `_batch_process_all_gathers()` 
- `_batch_process_and_export_streaming()`
- Associated menu actions

---

### Phase E: Testing (3 tasks)

#### E.1: Unit Tests

- Test `ProcessingJobAdapter` with mock coordinator
- Test progress emission to Qt bridge
- Test cancellation flow

#### E.2: Integration Tests

- Test full processing flow with Dashboard
- Test toast notifications on complete/fail
- Test job history storage

#### E.3: UI Tests

- Manual testing of Dashboard during processing
- Verify cancel from Dashboard works
- Verify job appears in history after completion

---

## Implementation Order

| Priority | Task | Estimated Changes | Dependencies |
|----------|------|-------------------|--------------|
| 1 | A.1: Enhance ProcessingJobAdapter | ~50 lines | None |
| 2 | A.2: Create processing_api.py | ~100 lines | A.1 |
| 3 | B.1: Migrate _batch_process_parallel | ~80 lines changed | A.1, A.2 |
| 4 | B.3: Add helper methods | ~20 lines | B.1 |
| 5 | B.4: Update progress display | ~30 lines | B.1 |
| 6 | B.2: Migrate _parallel_export_segy | ~60 lines | A.1 |
| 7 | E.1: Unit tests | ~100 lines | A.1, A.2 |
| 8 | E.2: Integration tests | ~150 lines | B.1, B.2 |
| 9 | C.1: PSTM wizard | ~40 lines | A.1 |
| 10 | C.2: QC batch engine | ~60 lines | A.1 |
| 11 | D.1: Deprecation warnings | ~20 lines | B.1 |
| 12 | D.2: Legacy removal | -500 lines | After release |

**Total New Code:** ~500 lines
**Total Removed Code:** ~500 lines (later)
**Net Change:** Neutral, but cleaner architecture

---

## Files Modified Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `utils/ray_orchestration/processing_job_adapter.py` | Enhance | +50 |
| `utils/ray_orchestration/processing_api.py` | Create | +100 |
| `utils/ray_orchestration/__init__.py` | Export | +5 |
| `main_window.py` | Migrate | ~150 (net) |
| `views/pstm_wizard_dialog.py` | Migrate | ~40 |
| `processors/qc_batch_engine.py` | Migrate | ~60 |
| `tests/test_processing_job_integration.py` | Create | +250 |

---

## Rollback Plan

If issues arise:
1. Keep old methods available with `_legacy` suffix
2. Add feature flag: `use_job_manager_for_processing`
3. Default to new system, flag falls back to old

```python
if self.app_settings.get('use_job_manager_for_processing', True):
    self._batch_process_parallel_v2()
else:
    self._batch_process_parallel_legacy()
```

---

## Success Criteria

- [ ] All processing jobs appear in Job Dashboard
- [ ] Progress updates in real-time during processing
- [ ] Cancel from Dashboard stops processing
- [ ] Toast notification on job completion
- [ ] Toast notification on job failure
- [ ] Jobs saved to history with statistics
- [ ] Analytics show processing job data
- [ ] No performance regression (< 5% overhead)
- [ ] All existing tests pass
- [ ] New integration tests pass

---

## Timeline Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| A: Infrastructure | A.1, A.2 | 1 session |
| B: Main Window | B.1-B.4 | 1-2 sessions |
| C: Secondary | C.1, C.2 | 1 session |
| D: Legacy | D.1 | 0.5 session |
| E: Testing | E.1, E.2 | 1 session |
| **Total** | | **4-5 sessions** |

