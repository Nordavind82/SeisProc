# PSTM Optimization Plan v2

## Current Performance Baseline
- **Input:** 31,012 traces × 1,600 samples
- **Output:** 463×321 = 148,623 output points × 1,001 depths
- **Rate:** ~40,773 output-samples/s
- **ETA:** ~5,800s (1.6 hours) for one offset bin
- **Total operations:** 4.6 trillion

## Target Performance
- **Goal:** < 5 minutes for 31K traces (20x speedup minimum)
- **Stretch goal:** < 1 minute (100x speedup)

---

## Phase 1: Diagnostics & Profiling

### Task 1.1: Add Detailed Timing Instrumentation
**File:** `processors/migration/kirchhoff_kernel.py`

Add timing breakdown for each operation in `migrate_kirchhoff_full()`:
- Distance computation time
- sqrt() calls time
- Aperture mask creation time
- Trace interpolation time
- Weighted sum time
- Memory transfer overhead

**Test:** `tests/test_kernel_profiling.py`
- Test that profiling can be enabled/disabled
- Test that timing dict is returned with all expected keys
- Test that timings sum to approximately total time

**Validation:** Run migration and inspect timing breakdown in logs.

**STOP POINT:** Review timing breakdown before proceeding.

---

### Task 1.2: Add Aperture Statistics Logging
**File:** `processors/migration/kirchhoff_kernel.py`

Log aperture effectiveness:
- Number of traces passing aperture at each depth
- Min/max/avg traces per output point
- Percentage of trace-output pairs actually computed

**Test:** `tests/test_aperture_statistics.py`
- Test aperture stats are computed correctly for synthetic data
- Test that shallow depths have fewer traces passing
- Test that stats dict contains expected keys

**Validation:** Run migration and check if aperture is filtering effectively.

**STOP POINT:** Review aperture statistics to understand filtering effectiveness.

---

## Phase 2: Quick Wins (Easy, High Impact)

### Task 2.1: Implement Time-Dependent Aperture
**File:** `processors/migration/kirchhoff_kernel.py`

Replace constant aperture with time-dependent:
```python
# Current: aperture_m = 3000 (constant)
# New: aperture_m(t) = v * t / 2 * tan(max_angle)
```

At t=100ms with v=1700m/s and angle=60°:
- Current: 3000m aperture → processes all traces
- New: 147m aperture → processes ~1% of traces

**Test:** `tests/test_time_dependent_aperture.py`
- Test aperture at t=0 is 0
- Test aperture at t=1000ms matches expected formula
- Test aperture at t=2000ms is larger than at t=1000ms
- Test migration still produces correct focused image
- Test speedup is measurable (>2x for shallow-heavy data)

**Validation:**
1. Run migration on test data
2. Compare image quality: should be nearly identical
3. Compare timing: expect 5-10x speedup

**STOP POINT:** Validate image quality and speedup before proceeding.

---

### Task 2.2: Pre-filter Traces by Maximum Aperture
**File:** `processors/migration/kirchhoff_kernel.py`

Before processing each tile of output points:
1. Compute max possible aperture for deepest time
2. Find traces within max aperture of tile center
3. Only process those traces for the entire tile

**Test:** `tests/test_trace_prefilter.py`
- Test that trace count is reduced for corner output points
- Test that no valid traces are excluded (gold standard comparison)
- Test that speedup scales with aperture reduction

**Validation:**
1. Log trace counts before/after filtering
2. Verify no change in output (bit-exact or within tolerance)
3. Measure speedup

**STOP POINT:** Validate correctness and measure speedup.

---

## Phase 3: Algorithm Optimization (Medium Complexity)

### Task 3.1: Direct Time-Domain Mapping (Eliminate Depth Loop)
**File:** `processors/migration/kirchhoff_time.py` (new file)

For constant velocity PSTM, use direct time mapping:
```python
# Instead of: for each depth z, compute t = (r_src + r_rcv) / v
# Use: for each output time t_out, find input time t_in directly
# t_in² = t_out² + 4*h²/v²  (for zero-offset approximation)
# or full: t_in = t_src + t_rcv where t_src = sqrt(t_out² + h_src²/v²), etc.
```

This eliminates the depth loop entirely - process time samples directly.

**Test:** `tests/test_time_domain_migration.py`
- Test time mapping formula is correct
- Test output matches depth-domain migration (within tolerance)
- Test diffractor focuses correctly
- Test performance improvement (expect 50-100x)

**Validation:**
1. Compare output image to depth-domain version
2. Measure timing improvement
3. Check image quality on real data

**STOP POINT:** This is a major algorithm change - thorough validation required.

---

### Task 3.2: Spatial Indexing for Output-to-Trace Mapping
**File:** `processors/migration/spatial_index.py` (new file)

Build KD-tree of trace midpoints, query for each output region:
1. Build KD-tree from all trace midpoints once
2. For each tile of output points, query traces within max aperture
3. Only process relevant traces

**Test:** `tests/test_spatial_index.py`
- Test KD-tree builds correctly
- Test query returns correct traces for known geometry
- Test no valid traces are missed (compare to brute force)
- Test speedup for large trace counts

**Validation:**
1. Verify identical output to non-indexed version
2. Measure speedup (expect 5-20x depending on geometry)

**STOP POINT:** Validate correctness before combining with other optimizations.

---

## Phase 4: Memory & Compute Optimization

### Task 4.1: Batch Trace Processing
**File:** `processors/migration/kirchhoff_kernel.py`

Process traces in batches to optimize memory access:
1. Sort traces by midpoint location
2. Process in spatial batches that fit in GPU cache
3. Reuse distance computations within batch

**Test:** `tests/test_batch_processing.py`
- Test batched output matches non-batched
- Test memory usage is bounded
- Test cache efficiency improves

**Validation:** Profile memory bandwidth and GPU utilization.

**STOP POINT:** Verify no regression in output quality.

---

### Task 4.2: Vectorized sqrt Using Fast Approximation
**File:** `processors/migration/fast_math.py` (new file)

Replace exact sqrt with fast approximation where precision isn't critical:
1. Use `rsqrt()` (reciprocal sqrt) which is faster on GPU
2. Or use Newton-Raphson iteration with fewer steps
3. Only for distance computation, not final traveltime

**Test:** `tests/test_fast_math.py`
- Test approximation error is within tolerance (< 0.1%)
- Test output image difference is negligible
- Test speedup is measurable

**Validation:** Compare image quality and measure speedup.

**STOP POINT:** Ensure approximation doesn't degrade image quality.

---

## Phase 5: Advanced Optimizations (Complex)

### Task 5.1: Hybrid DMO-Based Approximation
**File:** `processors/migration/dmo_migration.py` (new file)

Implement fast DMO-based PSTM as alternative:
1. Apply NMO correction
2. Apply DMO (dip moveout)
3. Stack
4. Apply post-stack time migration

This is much faster but slightly less accurate for steep dips.

**Test:** `tests/test_dmo_migration.py`
- Test NMO correction is correct
- Test DMO operator is correct
- Test output is reasonable for gentle dips
- Test speedup (expect 100x+)

**Validation:** Compare to full Kirchhoff on test data with known dips.

**STOP POINT:** Major feature - needs extensive validation.

---

### Task 5.2: Multi-Resolution Migration
**File:** `processors/migration/multiresolution.py` (new file)

Coarse-to-fine approach:
1. First pass: migrate at 4x coarser grid (16x fewer points)
2. Identify regions with significant energy
3. Second pass: full resolution only where needed

**Test:** `tests/test_multiresolution.py`
- Test coarse migration identifies energy regions
- Test fine migration fills in detail
- Test final output matches full resolution (within tolerance)

**Validation:** Compare output quality and measure speedup.

**STOP POINT:** Validate image quality is acceptable.

---

## Implementation Order

```
Phase 1: Diagnostics (understand the problem)
├── 1.1 Timing instrumentation [FIRST]
└── 1.2 Aperture statistics [SECOND]

Phase 2: Quick wins (low risk, high reward)
├── 2.1 Time-dependent aperture [HIGH PRIORITY]
└── 2.2 Trace pre-filtering [HIGH PRIORITY]

Phase 3: Algorithm optimization (medium risk, highest reward)
├── 3.1 Time-domain mapping [HIGHEST IMPACT - 50-100x]
└── 3.2 Spatial indexing [5-20x]

Phase 4: Memory/compute optimization (low risk, medium reward)
├── 4.1 Batch processing
└── 4.2 Fast math

Phase 5: Advanced (high complexity, optional)
├── 5.1 DMO approximation
└── 5.2 Multi-resolution
```

---

## Expected Cumulative Speedup

| After Task | Expected Speedup | Time for 31K traces |
|------------|------------------|---------------------|
| Baseline   | 1x               | 96 minutes          |
| 1.1-1.2    | 1x (diagnostics) | 96 minutes          |
| 2.1        | 5-10x            | 10-20 minutes       |
| 2.2        | 2x more          | 5-10 minutes        |
| 3.1        | 10-20x more      | 30s - 1 minute      |
| 3.2        | 2x more          | 15-30 seconds       |

---

## Test Commands

```bash
# Run specific test file
python -m pytest tests/test_kernel_profiling.py -v

# Run all optimization tests
python -m pytest tests/test_*aperture*.py tests/test_*prefilter*.py -v

# Run with timing output
python -m pytest tests/test_time_domain_migration.py -v -s

# Run benchmark comparison
python -m pytest tests/test_migration_benchmark.py -v --benchmark
```

---

## Validation Checklist for Each Task

- [ ] All existing tests still pass
- [ ] New tests pass
- [ ] Output image visually similar to baseline
- [ ] Peak location unchanged for diffractor test
- [ ] Fold values reasonable
- [ ] Speedup measured and documented
- [ ] No memory leaks (check GPU memory)
- [ ] Works on both MPS (Apple) and CUDA

---

## Notes

1. **Always keep the original implementation available** for A/B comparison
2. **Add feature flags** to enable/disable optimizations
3. **Log timing and statistics** so we can track improvements
4. **Test on real data** after synthetic tests pass
5. **Stop and validate** after each task before moving to next
