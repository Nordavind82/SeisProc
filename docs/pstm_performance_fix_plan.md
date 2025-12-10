# PSTM Performance Fix Plan

## Problem Statement

Current PSTM performance: **~14 traces/second** (22 seconds for 310 traces)
Target performance: **~5,000-10,000 traces/second** (industry standard for GPU migration)

**Root Cause**: Optimizations were implemented as separate modules but not properly integrated into the execution flow. The migration is called per-chunk (~310 traces) rather than per-dataset, causing massive overhead.

---

## Task List

### Phase 1: Cleanup & Diagnostics (Priority: High)

#### Task 1.1: Remove Dead/Unused Code ✅ COMPLETED
**Files deleted:**
- [x] `processors/migration/orchestrator.py` - removed
- [x] `processors/migration/symmetry.py` - removed (unused optimization)
- [x] `tests/test_symmetry.py` - removed

**Result:** ~1500 lines of dead code removed

---

#### Task 1.2: Add Performance Instrumentation ✅ COMPLETED
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Implemented:**
- Added `_profile_tiles` flag to enable/disable detailed profiling
- Added `enable_profiling()` method
- `_migrate_tile()` now logs timing breakdown when profiling is enabled
- Profiling auto-enabled for first chunk in migration_monitor_dialog.py

**Expected Output:**
```
DEBUG: Tile (0-128, 0-128) profile: setup=0.5ms, trace_sel=2.1ms, migration=450.3ms, total=453.0ms, traces=310
```

---

#### Task 1.3: Add GPU Utilization Monitoring
**File:** `processors/migration/gpu_memory.py`

Add method to check actual GPU usage:
```python
def get_device_utilization(self) -> dict:
    """Get current device memory and compute utilization."""
```

**Expected Output:** Logs showing GPU memory used vs available, kernel execution time vs idle time

---

### Phase 2: Fix Core Performance Issues (Priority: Critical)

#### Task 2.1: Eliminate Per-Chunk Structure Rebuilding ✅ COMPLETED
**Files:** `processors/migration/optimized_kirchhoff_migrator.py`, `views/migration_monitor_dialog.py`

**Implemented:**
1. Added `initialize_for_bin(all_geometry)` method - builds spatial index once for entire bin
2. Added `reset_for_new_bin()` method - resets structures between bins
3. Added `_spatial_index_initialized` flag to skip rebuilding
4. Updated `migration_monitor_dialog.py` to call `initialize_for_bin()` before chunk loop
5. Added log message showing bin initialization time

**Result:** Spatial index built once per bin instead of 101 times. Expected time savings: ~10-15 seconds per bin.

---

#### Task 2.2: Implement True Batch Processing
**File:** `views/migration_monitor_dialog.py`

**Problem:** Currently processes 310 traces per call, accumulates results externally.

**Fix:** Change processing flow:
```
Current (slow):
  For each chunk (310 traces):
    migrator.migrate_gather(chunk_traces)  # 22 seconds each
    accumulate externally

New (fast):
  migrator.initialize_for_bin(all_bin_geometry)  # Build structures once
  For each chunk (310 traces):
    migrator.migrate_gather(chunk_traces, accumulate=True)  # Fast, ~0.5s each
  result = migrator.get_accumulated_result()
```

**Changes to migration_monitor_dialog.py:**
- Lines 380-490: Restructure bin processing loop
- Add initialization call before chunk loop
- Remove external accumulation (let migrator handle it)

**Expected Output:** 10-20x speedup in migration time per chunk

---

#### Task 2.3: Optimize Tile Size and Count ✅ COMPLETED
**Files:** `processors/migration/gpu_memory.py`, `processors/migration/optimized_kirchhoff_migrator.py`

**Implemented:**
1. Increased `min_tile_size` from 10 to 64
2. Increased `max_tile_size` from 200 to 256
3. Increased default fallback tile size from 50 to 128

**Result:** Fewer tiles created, reduced kernel launch overhead. Expected tile count reduction: ~75%.

**Expected Output:** Reduce tile count from 48 to ~12, reduce kernel launch overhead by 75%

---

#### Task 2.4: Disable or Fix LUT for MPS
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Problem:** Traveltime LUT may be slower than direct computation on MPS (Apple Silicon).

**Fix:**
1. Add benchmark to compare LUT vs direct on first run
2. Auto-disable LUT if direct is faster
3. Or: Simplify LUT to avoid bilinear interpolation overhead

**Changes:**
- `_build_traveltime_lut()`: Add benchmark
- `_migrate_tile()`: Add fallback path

**Expected Output:** Traveltime computation 2-3x faster on MPS

---

### Phase 3: Algorithmic Improvements (Priority: Medium)

#### Task 3.1: Implement Vectorized Inner Loop
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Problem:** Lines 409-477 use Python for-loop over trace batches.

**Fix:** Vectorize to process all traces in single GPU operation:
```python
# Instead of:
for t_start in range(0, n_subset, trace_batch_size):
    # process batch

# Do:
# Process all traces at once using torch operations
```

**Expected Output:** Eliminate Python loop overhead, 2-5x speedup in `_migrate_tile()`

---

#### Task 3.2: Implement Output-Driven Trace Selection
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Problem:** All traces processed for every tile, even if they can't contribute.

**Fix:** Use spatial index to select only relevant traces per tile:
```python
def _migrate_tile(self, ...):
    # Get tile bounds in world coordinates
    # Query spatial index for traces within aperture
    # Process only relevant traces
```

**Expected Output:** 3-5x reduction in traces processed per tile (geometry dependent)

---

#### Task 3.3: Implement Depth-Grouped Processing
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Problem:** DepthAdaptiveAperture computed but not used to reduce work.

**Fix:** Process depths in groups with similar aperture:
- Shallow depths: Process few traces (small aperture)
- Deep depths: Process more traces (large aperture)

**Expected Output:** 1.5-2x speedup for typical geometries

---

### Phase 4: Memory Optimization (Priority: Medium)

#### Task 4.1: Implement Tensor Reuse
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Problem:** New tensors allocated every tile/batch iteration.

**Fix:** Pre-allocate workspace tensors and reuse:
```python
def _allocate_workspace(self, max_traces, max_points, n_z):
    self._ws_h_src = torch.zeros(...)
    self._ws_h_rcv = torch.zeros(...)
    # etc.
```

**Expected Output:** Reduce memory allocation overhead by 50-80%

---

#### Task 4.2: Implement Streaming Output
**File:** `processors/migration/optimized_kirchhoff_migrator.py`

**Problem:** Full output grid kept in memory during processing.

**Fix:** Write output tiles to disk as they complete:
```python
def _migrate_tile(..., output_writer):
    # Compute tile
    # Write to disk immediately
    # Free GPU memory
```

**Expected Output:** Enable processing of larger grids, reduce peak memory by 50%

---

### Phase 5: Testing & Validation (Priority: High)

#### Task 5.1: Create Performance Benchmark Script
**File:** `tests/benchmarks/benchmark_pstm.py`

Create script that:
1. Generates synthetic data (1000, 10000, 100000 traces)
2. Runs migration with different settings
3. Reports traces/second for each configuration
4. Compares CPU vs MPS vs CUDA (if available)

**Expected Output:**
```
PSTM Benchmark Results:
-----------------------
Traces | CPU    | MPS    | Config
1000   | 500/s  | 2000/s | LUT=True, Tiles=64
1000   | 600/s  | 1800/s | LUT=False, Tiles=64
1000   | 450/s  | 3500/s | LUT=False, Tiles=128
...
```

---

#### Task 5.2: Create Regression Test for Output Quality
**File:** `tests/test_pstm_quality.py`

Ensure optimizations don't change output:
1. Run migration with baseline (all optimizations off)
2. Run migration with optimizations
3. Compare outputs (L2 error < 1%)

**Expected Output:** Test passes confirming output quality maintained

---

#### Task 5.3: Update Existing Tests
**Files:**
- `tests/test_optimized_kirchhoff.py`
- `tests/test_traveltime_lut.py`
- `tests/test_trace_index.py`
- `tests/test_aperture_adaptive.py`
- `tests/test_gpu_memory.py`

Review and update tests to reflect actual usage patterns.

**Expected Output:** All tests pass, coverage > 80% for migration code

---

### Phase 6: Documentation & Cleanup (Priority: Low)

#### Task 6.1: Update Optimization Plan Document
**File:** `docs/pstm_optimization_plan.md`

Update to reflect:
- What was implemented vs what works
- Actual measured speedups
- Known issues and workarounds

**Expected Output:** Accurate documentation matching reality

---

#### Task 6.2: Add Performance Tuning Guide
**File:** `docs/pstm_performance_tuning.md`

Document:
- Recommended settings for different hardware
- How to diagnose performance issues
- Expected performance numbers

**Expected Output:** User guide for performance tuning

---

## Implementation Priority Order

| Order | Task | Impact | Effort | Risk |
|-------|------|--------|--------|------|
| 1 | 2.1 Eliminate per-chunk rebuilding | High | Low | Low |
| 2 | 2.2 Implement batch processing | High | Medium | Medium |
| 3 | 1.2 Add instrumentation | Medium | Low | Low |
| 4 | 2.3 Optimize tile size | Medium | Low | Low |
| 5 | 2.4 Fix LUT for MPS | Medium | Medium | Low |
| 6 | 3.1 Vectorized inner loop | High | High | Medium |
| 7 | 3.2 Output-driven trace selection | High | Medium | Medium |
| 8 | 5.1 Create benchmark | Medium | Low | Low |
| 9 | 1.1 Remove dead code | Low | Low | Low |
| 10 | 4.1 Tensor reuse | Medium | Medium | Low |

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Traces/second | 14 | 2,000 | 10,000 |
| Time for 31K traces | 37 min | 15 sec | 3 sec |
| Memory usage | Unknown | < 4GB | < 2GB |
| GPU utilization | Unknown | > 50% | > 80% |

---

## Quick Wins (Can implement immediately)

1. **Increase tile size** in `gpu_memory.py`: Change default from 64 to 128
2. **Skip spatial index rebuild** if geometry hasn't changed
3. **Disable symmetry** (already done, just verify)
4. **Test with LUT disabled** to see if it's actually helping

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `processors/migration/optimized_kirchhoff_migrator.py` | Major refactoring |
| `processors/migration/gpu_memory.py` | Tile size optimization |
| `views/migration_monitor_dialog.py` | Batch processing flow |
| `processors/migration/traveltime_lut.py` | MPS optimization |
| `tests/benchmarks/benchmark_pstm.py` | New file |
| `tests/test_pstm_quality.py` | New file |

## Files to Delete

| File | Reason |
|------|--------|
| `processors/migration/orchestrator.py` | Already deleted - unused |
| `processors/migration/symmetry.py` | Unused, disabled by default |
| `tests/test_symmetry.py` | Tests for unused code |
