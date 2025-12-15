# Polars Migration & Ensemble Index Pre-loading Design

## Executive Summary

This document outlines the design for:
1. **Polars Migration**: Replace Pandas with Polars for Parquet I/O (6x faster, 50% less memory)
2. **Ensemble Index Pre-loading**: Share ensemble index via fork COW (save ~80MB per worker)

**Expected Impact:**
- Header loading: 2.5s → 0.4s (6x faster)
- Memory per worker: Reduced by ~80-170MB
- Total dataset memory footprint: 11 workers × 80MB = 880MB saved

---

## 1. Current Architecture Analysis

### 1.1 Header Loading Locations (54 locations)

| Category | File | Count | Current Method |
|----------|------|-------|----------------|
| Parallel Processing | coordinator.py, worker.py | 3 | pd.read_parquet (shared headers mechanism) |
| SEGY Import | multiprocess_import/coordinator.py | 8 | pd.read_parquet (various modes) |
| Data Storage | data_storage.py | 3 | pd.read_parquet |
| Lazy Data Model | lazy_seismic_data.py | 2 | pd.read_parquet with filtering |
| Main Window | main_window.py | 2 | pd.read_parquet |
| Export | parallel_export/coordinator.py | 2 | pd.read_parquet |
| QC Engines | qc_stacking_engine.py, qc_batch_engine.py | 2 | pd.read_parquet |
| Trace Sorter | trace_sorter.py | 1 | pd.read_parquet |
| Views/Dialogs | export_options_dialog.py, pstm_wizard_dialog.py | 5 | pd.read_parquet |
| Tests | Various test files | 6 | pd.read_parquet |

### 1.2 Ensemble Index Locations

| Operation | File | Lines | Current Method |
|-----------|------|-------|----------------|
| Create | data_storage.py | 693-789 | df.groupby().agg() + to_parquet |
| Create | multiprocess_import/coordinator.py | 865-981 | pd.read_parquet + groupby |
| Load | lazy_seismic_data.py | 64 | pd.read_parquet (cached) |
| Load | parallel_processing/worker.py | 635 | pd.read_parquet (per worker) |
| Access | gather_navigator.py | 81-128 | DataFrame operations |

### 1.3 Current Shared Headers Mechanism

```
COORDINATOR (before fork):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Detect needed columns (sort_key, offset for mute)            │
│ 2. pd.read_parquet(headers_path, columns=cols_to_load)          │
│ 3. set_shared_headers(df, columns)  → sets module-level global  │
│ 4. ProcessPoolExecutor created → fork() inherits globals        │
└─────────────────────────────────────────────────────────────────┘

WORKERS (after fork):
┌─────────────────────────────────────────────────────────────────┐
│ 1. get_shared_headers() → accesses module-level global          │
│ 2. If columns match needed → use shared (zero-copy via COW)     │
│ 3. If columns missing → fall back to pd.read_parquet            │
└─────────────────────────────────────────────────────────────────┘
```

**Problem: Ensemble index is NOT pre-loaded - each worker loads independently (~80MB × 11 workers = 880MB wasted)**

---

## 2. Proposed Architecture

### 2.1 Polars Integration Layer

Create a thin abstraction layer to handle Pandas ↔ Polars conversion:

```
utils/
└── parquet_io.py (NEW)
    ├── read_parquet() → Polars read + optional Pandas conversion
    ├── read_parquet_lazy() → Polars lazy scan
    ├── write_parquet() → Polars write
    └── ParquetReader class for streaming
```

**Why abstraction layer?**
- Single point to change for library swap
- Can use Polars internally while returning Pandas for compatibility
- Easy to benchmark/A/B test
- Gradual migration without breaking changes

### 2.2 Shared Data Pre-loading (Extended)

Extend the existing shared headers mechanism to include ensemble index:

```
COORDINATOR (before fork):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Pre-load headers (existing)                                  │
│ 2. Pre-load ensemble_index (NEW)                                │
│ 3. Pre-load both as numpy arrays for minimal memory (NEW)       │
│ 4. Store in module-level globals                                │
│ 5. fork() → workers inherit via COW                             │
└─────────────────────────────────────────────────────────────────┘

WORKERS (after fork):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Access pre-loaded data via get_shared_*() functions          │
│ 2. No file I/O needed if pre-loaded data sufficient             │
│ 3. COW sharing → near-zero memory overhead per worker           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Module Structure

```
utils/
├── parquet_io.py (NEW)
│   ├── read_parquet()           # Primary API
│   ├── read_parquet_lazy()      # Lazy loading for large files
│   ├── read_parquet_columns()   # Schema introspection
│   ├── write_parquet()          # Polars-optimized write
│   └── POLARS_AVAILABLE         # Feature flag
│
├── parallel_processing/
│   ├── shared_data.py (NEW)     # Centralized shared data management
│   │   ├── SharedDataManager    # Class to manage pre-loaded data
│   │   ├── set_shared_headers()
│   │   ├── get_shared_headers()
│   │   ├── set_shared_ensemble_index()  (NEW)
│   │   ├── get_shared_ensemble_index()  (NEW)
│   │   └── clear_shared_data()
│   │
│   ├── coordinator.py           # Modified to use new modules
│   └── worker.py                # Modified to use new modules
```

---

## 3. Pros and Cons Analysis

### 3.1 Polars Migration

| Aspect | Pros | Cons |
|--------|------|------|
| **Performance** | 6x faster parquet read, multi-threaded | First-time import overhead |
| **Memory** | 50% less RAM, Arrow-native | Adds ~50MB to package |
| **API** | Method chaining, lazy evaluation | Different from Pandas (learning curve) |
| **Compatibility** | Easy to_pandas() conversion | Some DataFrame features missing |
| **Maintenance** | Active development, good docs | Another dependency to track |
| **Risk** | Low - can keep Pandas fallback | Code duplication if maintaining both |

### 3.2 Ensemble Index Pre-loading

| Aspect | Pros | Cons |
|--------|------|------|
| **Memory** | Save ~80MB per worker | Must pre-load before knowing exact needs |
| **Speed** | No file I/O in workers | Slightly longer startup |
| **Complexity** | Simple extension of existing pattern | More module-level state to manage |
| **Compatibility** | Works only with fork context | Spawn context (Windows) gets no benefit |
| **Cleanup** | Needs explicit clear_shared_data() | Memory leak if not cleared |

### 3.3 Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Polars API differences | Medium | Abstraction layer, return Pandas when needed |
| Fork-only benefit | Low | Already using fork on Linux (primary target) |
| Memory leak | Medium | Explicit cleanup in finally blocks |
| Test coverage | Medium | Add unit tests for new module |
| Backward compatibility | Low | Keep pd.read_parquet as fallback |

---

## 4. Implementation Plan

### Phase 1: Foundation (Low Risk)

#### Task 1.1: Create parquet_io.py Module
**Location:** `/scratch/Python_Apps/SeisProc/utils/parquet_io.py`

**What:** Create abstraction layer for Parquet I/O

**Why:**
- Single point for Polars integration
- Can benchmark Polars vs Pandas
- Easy rollback if issues

**Implementation:**
```python
# utils/parquet_io.py
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import Polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    logger.info("Polars not available, using Pandas for Parquet I/O")


def read_parquet(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    use_polars: bool = True,
    return_pandas: bool = True
) -> Union[pd.DataFrame, 'pl.DataFrame']:
    """
    Read Parquet file with Polars (if available) or Pandas fallback.

    Args:
        path: Path to parquet file
        columns: Optional list of columns to load
        use_polars: Whether to use Polars (if available)
        return_pandas: Convert result to Pandas DataFrame

    Returns:
        DataFrame (Pandas or Polars based on settings)
    """
    path = Path(path)

    if POLARS_AVAILABLE and use_polars:
        try:
            df = pl.read_parquet(path, columns=columns)
            if return_pandas:
                return df.to_pandas()
            return df
        except Exception as e:
            logger.warning(f"Polars read failed, falling back to Pandas: {e}")

    # Pandas fallback
    return pd.read_parquet(path, columns=columns)


def read_parquet_schema(path: Union[str, Path]) -> List[str]:
    """Get column names from Parquet file without loading data."""
    if POLARS_AVAILABLE:
        schema = pl.read_parquet_schema(path)
        return list(schema.keys())
    else:
        import pyarrow.parquet as pq
        return pq.read_schema(path).names
```

**Effort:** 1-2 hours
**Risk:** Very Low

---

#### Task 1.2: Create shared_data.py Module
**Location:** `/scratch/Python_Apps/SeisProc/utils/parallel_processing/shared_data.py`

**What:** Centralize shared data management for fork COW

**Why:**
- Clean separation from worker.py
- Easier testing
- Single place for pre-loading logic

**Implementation:**
```python
# utils/parallel_processing/shared_data.py
"""
Shared data management for fork copy-on-write memory sharing.

This module provides a centralized way to pre-load data before forking
worker processes, allowing them to share memory via COW.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Module-level globals for fork COW sharing
_SHARED_HEADERS_DF: Optional[pd.DataFrame] = None
_SHARED_HEADERS_COLUMNS: Optional[List[str]] = None
_SHARED_ENSEMBLE_INDEX: Optional[pd.DataFrame] = None
_SHARED_ENSEMBLE_ARRAYS: Optional[Dict[str, np.ndarray]] = None


# === Headers (existing functionality, moved here) ===

def set_shared_headers(headers_df: pd.DataFrame, columns: List[str]) -> None:
    """Pre-load headers for fork COW sharing."""
    global _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS
    _SHARED_HEADERS_DF = headers_df
    _SHARED_HEADERS_COLUMNS = columns
    logger.debug(f"Shared headers set: {len(headers_df)} rows, columns={columns}")


def get_shared_headers() -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Get pre-loaded shared headers."""
    return _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS


# === Ensemble Index (NEW) ===

def set_shared_ensemble_index(ensemble_df: pd.DataFrame) -> None:
    """
    Pre-load ensemble index for fork COW sharing.

    Also converts to numpy arrays for faster access during processing.
    """
    global _SHARED_ENSEMBLE_INDEX, _SHARED_ENSEMBLE_ARRAYS
    _SHARED_ENSEMBLE_INDEX = ensemble_df

    # Pre-convert to numpy for faster access
    _SHARED_ENSEMBLE_ARRAYS = {
        'start_trace': ensemble_df['start_trace'].to_numpy(),
        'end_trace': ensemble_df['end_trace'].to_numpy(),
    }

    # Include ensemble_key if present
    if 'ensemble_key' in ensemble_df.columns:
        _SHARED_ENSEMBLE_ARRAYS['ensemble_key'] = ensemble_df['ensemble_key'].to_numpy()

    logger.debug(f"Shared ensemble index set: {len(ensemble_df)} gathers")


def get_shared_ensemble_index() -> Optional[pd.DataFrame]:
    """Get pre-loaded ensemble index as DataFrame."""
    return _SHARED_ENSEMBLE_INDEX


def get_shared_ensemble_arrays() -> Optional[Dict[str, np.ndarray]]:
    """Get pre-loaded ensemble index as numpy arrays for fast access."""
    return _SHARED_ENSEMBLE_ARRAYS


# === Cleanup ===

def clear_shared_data() -> None:
    """Clear all shared data (call after processing complete)."""
    global _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS
    global _SHARED_ENSEMBLE_INDEX, _SHARED_ENSEMBLE_ARRAYS

    _SHARED_HEADERS_DF = None
    _SHARED_HEADERS_COLUMNS = None
    _SHARED_ENSEMBLE_INDEX = None
    _SHARED_ENSEMBLE_ARRAYS = None

    logger.debug("Shared data cleared")


def get_shared_data_memory_usage() -> int:
    """Get approximate memory usage of shared data in bytes."""
    total = 0

    if _SHARED_HEADERS_DF is not None:
        total += _SHARED_HEADERS_DF.memory_usage(deep=True).sum()

    if _SHARED_ENSEMBLE_INDEX is not None:
        total += _SHARED_ENSEMBLE_INDEX.memory_usage(deep=True).sum()

    if _SHARED_ENSEMBLE_ARRAYS is not None:
        for arr in _SHARED_ENSEMBLE_ARRAYS.values():
            total += arr.nbytes

    return total
```

**Effort:** 1-2 hours
**Risk:** Low

---

### Phase 2: Integration (Medium Risk)

#### Task 2.1: Update Parallel Processing Coordinator
**Location:** `/scratch/Python_Apps/SeisProc/utils/parallel_processing/coordinator.py`

**What:**
- Use parquet_io for header loading
- Add ensemble index pre-loading
- Use shared_data module

**Changes Required:**
```python
# Lines 908-964: Modify pre-loading section

# BEFORE (current):
from utils.parallel_processing.worker import set_shared_headers
...
shared_headers = pd.read_parquet(headers_path, columns=cols_to_load)
set_shared_headers(shared_headers, cols_to_load)

# AFTER (new):
from utils.parquet_io import read_parquet
from utils.parallel_processing.shared_data import (
    set_shared_headers,
    set_shared_ensemble_index,
    clear_shared_data
)
...
# Pre-load headers (faster with Polars)
shared_headers = read_parquet(headers_path, columns=cols_to_load)
set_shared_headers(shared_headers, cols_to_load)

# Pre-load ensemble index (NEW)
ensemble_index = read_parquet(task.ensemble_index_path)
set_shared_ensemble_index(ensemble_index)

# In finally block:
clear_shared_data()  # Cleanup
```

**Why:**
- Polars reads 6x faster → faster job startup
- Ensemble pre-load → save 80MB per worker
- Centralized cleanup → no memory leaks

**Effort:** 2-3 hours
**Risk:** Medium (touches critical path)

---

#### Task 2.2: Update Parallel Processing Worker
**Location:** `/scratch/Python_Apps/SeisProc/utils/parallel_processing/worker.py`

**What:**
- Use shared_data module instead of local globals
- Add ensemble index access from shared data

**Changes Required:**
```python
# Lines 32-76: Remove local globals, import from shared_data

# BEFORE:
_SHARED_HEADERS_DF = None
_SHARED_HEADERS_COLUMNS = None

def set_shared_headers(headers_df, columns):
    global _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS
    ...

# AFTER:
from utils.parallel_processing.shared_data import (
    get_shared_headers,
    get_shared_ensemble_index,
    get_shared_ensemble_arrays
)

# Line 635: Use shared ensemble index

# BEFORE:
ensemble_index = pd.read_parquet(task.ensemble_index_path)

# AFTER:
ensemble_index = get_shared_ensemble_index()
if ensemble_index is None:
    # Fallback if not pre-loaded (spawn context)
    ensemble_index = read_parquet(task.ensemble_index_path)
```

**Why:**
- Workers get ensemble index via COW (no file I/O)
- 80MB saved per worker
- Faster startup per worker

**Effort:** 2-3 hours
**Risk:** Medium

---

#### Task 2.3: Update SEGY Import Coordinator
**Location:** `/scratch/Python_Apps/SeisProc/utils/segy_import/multiprocess_import/coordinator.py`

**What:** Replace pd.read_parquet with parquet_io

**Lines to modify:** 363, 419, 422, 425, 816, 877, 882, 887

**Changes Required:**
```python
# Add import
from utils.parquet_io import read_parquet

# Replace each pd.read_parquet call:
# BEFORE:
headers_df = pd.read_parquet(headers_path, columns=[ensemble_key])

# AFTER:
headers_df = read_parquet(headers_path, columns=[ensemble_key])
```

**Why:**
- Import phase is not memory-critical but benefits from speed
- Consistent API across codebase

**Effort:** 1-2 hours
**Risk:** Low

---

#### Task 2.4: Update LazySeismicData
**Location:** `/scratch/Python_Apps/SeisProc/models/lazy_seismic_data.py`

**What:** Use parquet_io for header/ensemble loading

**Lines to modify:** 64, 332, 337

**Changes Required:**
```python
# Add import
from utils.parquet_io import read_parquet

# Line 64: Ensemble index loading
# BEFORE:
self._ensemble_index = pd.read_parquet(ensemble_index_path)

# AFTER:
self._ensemble_index = read_parquet(ensemble_index_path)

# Lines 332-337: Header filtering
# BEFORE:
return pd.read_parquet(self._headers_path, filters=...)

# AFTER:
return read_parquet(self._headers_path, filters=...)
```

**Why:**
- Faster ensemble loading on startup
- Consistent API

**Effort:** 1 hour
**Risk:** Low

---

### Phase 3: Extended Migration (Lower Priority)

#### Task 3.1: Update Data Storage
**Location:** `/scratch/Python_Apps/SeisProc/utils/segy_import/data_storage.py`

**Lines:** 853, 895, 911

**Effort:** 1 hour
**Risk:** Low

---

#### Task 3.2: Update Main Window
**Location:** `/scratch/Python_Apps/SeisProc/main_window.py`

**Lines:** 618, 2572

**Effort:** 1 hour
**Risk:** Low

---

#### Task 3.3: Update Export Coordinator
**Location:** `/scratch/Python_Apps/SeisProc/utils/parallel_export/coordinator.py`

**Lines:** 183, 387

**Effort:** 1 hour
**Risk:** Low

---

#### Task 3.4: Update QC Engines
**Locations:**
- `/scratch/Python_Apps/SeisProc/processors/qc_stacking_engine.py` (line 222)
- `/scratch/Python_Apps/SeisProc/processors/qc_batch_engine.py` (line 388)

**Effort:** 1 hour
**Risk:** Low

---

#### Task 3.5: Update Trace Sorter
**Location:** `/scratch/Python_Apps/SeisProc/utils/trace_sorter.py`

**Line:** 434

**Effort:** 30 minutes
**Risk:** Low

---

#### Task 3.6: Update Views/Dialogs
**Locations:**
- `views/export_options_dialog.py` (lines 92, 777, 781)
- `views/pstm_wizard_dialog.py` (line 1032)
- `views/migration_monitor_dialog.py` (line 88)

**Effort:** 1-2 hours
**Risk:** Low

---

### Phase 4: Testing & Validation

#### Task 4.1: Unit Tests for parquet_io
**What:** Test Polars read, Pandas fallback, schema reading

**Effort:** 2 hours

---

#### Task 4.2: Unit Tests for shared_data
**What:** Test pre-loading, access, cleanup, memory usage

**Effort:** 2 hours

---

#### Task 4.3: Integration Tests
**What:** Test parallel processing with new shared data mechanism

**Effort:** 2-3 hours

---

#### Task 4.4: Performance Benchmarks
**What:** Compare before/after for:
- Header loading time
- Worker memory usage
- Total processing time

**Effort:** 2 hours

---

## 5. Old Code Cleanup

### 5.1 Code to Remove

| Location | Lines | What | When Safe to Remove |
|----------|-------|------|---------------------|
| worker.py | 41-76 | Local shared headers globals | After Task 2.2 complete |
| coordinator.py | N/A | Direct pd.read_parquet calls | After Task 2.1 complete |

### 5.2 Deprecation Strategy

1. **Phase 1-2:** New code uses parquet_io, old code still works
2. **Phase 3:** Add deprecation warnings to direct pd.read_parquet calls
3. **Phase 4:** Remove deprecated code after testing

---

## 6. Dependency Management

### 6.1 New Dependencies

```toml
# pyproject.toml or requirements.txt
polars >= 0.20.0  # Parquet I/O acceleration
```

### 6.2 Optional Dependency Pattern

```python
# utils/parquet_io.py
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    # Graceful fallback to Pandas
```

---

## 7. Rollback Plan

If issues discovered:

1. **parquet_io.py:** Set `use_polars=False` as default
2. **shared_data.py:** Skip ensemble pre-loading, fall back to per-worker loading
3. **Full rollback:** Revert to previous commit (changes are isolated to new modules)

---

## 8. Task Summary

| ID | Task | Effort | Priority | Risk | Dependencies |
|----|------|--------|----------|------|--------------|
| 1.1 | Create parquet_io.py | 1-2h | High | Very Low | None |
| 1.2 | Create shared_data.py | 1-2h | High | Low | None |
| 2.1 | Update coordinator.py | 2-3h | High | Medium | 1.1, 1.2 |
| 2.2 | Update worker.py | 2-3h | High | Medium | 1.2 |
| 2.3 | Update SEGY import coordinator | 1-2h | Medium | Low | 1.1 |
| 2.4 | Update LazySeismicData | 1h | Medium | Low | 1.1 |
| 3.1-3.6 | Extended migration | 5-6h | Low | Low | 1.1 |
| 4.1-4.4 | Testing | 8-9h | High | N/A | 2.1, 2.2 |

**Total Effort:** ~20-25 hours

**Recommended Order:**
1. Task 1.1 + 1.2 (foundation)
2. Task 2.1 + 2.2 (critical path)
3. Task 4.1 + 4.2 (validate foundation)
4. Task 4.3 + 4.4 (validate integration)
5. Tasks 2.3, 2.4, 3.x (extended migration)

---

## 9. Success Metrics

| Metric | Before | Target | How to Measure |
|--------|--------|--------|----------------|
| Header load time (22M rows) | 2.5s | 0.4s | time.time() around read_parquet |
| Memory per worker | ~800MB | ~720MB | psutil.Process().memory_info() |
| Job startup time | ~5s | ~3s | Time from start to first gather |
| Worker ensemble load | 80MB each | 0 (shared) | Memory profiling |

---

## Document Information

- **Version:** 1.0
- **Date:** 2025-12-13
- **Author:** Claude Code Assistant
- **Related Files:**
  - `utils/parallel_processing/coordinator.py`
  - `utils/parallel_processing/worker.py`
  - `utils/segy_import/multiprocess_import/coordinator.py`
  - `models/lazy_seismic_data.py`
