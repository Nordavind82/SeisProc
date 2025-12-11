# Time-Domain Migration Bug Analysis

## Problem Observed
Data accumulates in a diagonal pattern at corner of output grid instead of being distributed across the full survey area. With uniform offset range (300-330m), expect uniform fold coverage.

## Screenshot Analysis
- Energy concentrated in lower-left corner with diagonal striping pattern
- Rest of grid appears empty (gray = zero)
- Pattern suggests systematic error in output coordinate mapping

## Root Cause Hypothesis

The scatter_add accumulation uses linear indices computed as:
```python
linear_idx = out_idx_floor * (n_il * n_xl) + tile_il_exp * n_xl + tile_xl_exp
```

**Critical Bug Found:** `tile_il` and `tile_xl` are computed from **output point indices**, not from the actual inline/crossline grid positions of trace midpoints.

```python
# Current (WRONG):
tile_il = torch.arange(out_start, out_end, device=device) // n_xl
tile_xl = torch.arange(out_start, out_end, device=device) % n_xl
```

This computes the IL/XL of the **output tile points**, not where the **input traces** should map to.

In time-domain migration:
1. Each input trace has a midpoint (x, y)
2. That midpoint maps to an (inline, crossline) on the output grid
3. The amplitude should scatter to that (inline, crossline) location

**But the current code:**
1. Takes a tile of output points
2. Computes which traces are nearby
3. Scatters ALL those traces to the TILE's output points (tile_il, tile_xl)
4. This means traces are mapped to wrong locations!

## The Fundamental Algorithm Error

Time-domain migration should be **trace-centric** (scatter from trace to its midpoint), but the code is **output-centric** (gather traces into output tiles).

### Correct Algorithm (Trace-Centric Scatter):
```
For each input trace:
    midpoint = (source + receiver) / 2
    (il, xl) = grid_coords(midpoint)  # Where this trace contributes
    For each input time sample:
        t_out = NMO(t_in, offset, velocity)
        output[t_out, il, xl] += weighted_amplitude
```

### Current Algorithm (Output-Centric - WRONG for scatter):
```
For each output tile (set of output points):
    Find nearby traces
    For each sample:
        Compute t_out for all traces
        Scatter ALL traces to THIS TILE's (il, xl)  # BUG!
```

## Evidence
- `tile_il_exp` and `tile_xl_exp` are derived from the tile's output indices
- They do NOT represent where each trace's midpoint falls on the grid
- All traces within aperture of a tile get scattered to that tile's output coordinates

## Required Fix

The scatter indices must use the **trace midpoint's** grid coordinates, not the tile's output coordinates:

```python
# For each trace, compute its midpoint grid location
trace_il = ((mid_x - origin_x) * cos_az + (mid_y - origin_y) * sin_az) / il_spacing
trace_xl = (-(mid_x - origin_x) * sin_az + (mid_y - origin_y) * cos_az) / xl_spacing

# Then scatter_add uses trace_il, trace_xl (not tile_il, tile_xl)
```

## Impact
- All data accumulates at tile center locations instead of trace midpoint locations
- Results in diagonal striping pattern matching tile boundaries
- Fold is artificially concentrated rather than distributed

## Complexity Assessment
This is a fundamental algorithm restructure, not a simple parameter fix. The entire tiling approach needs reconsideration:

**Option A:** Convert to trace-centric approach (scatter from each trace)
- More natural for time-domain migration
- Each trace processed independently
- Simpler logic but different parallelization strategy

**Option B:** Fix output-centric to use trace midpoint coordinates
- Keep tile structure for memory management
- Pass trace midpoint grid coords to kernel
- Scatter to trace's grid location, not tile's grid location

## Fix Implemented

**Files Modified:**
- `processors/migration/kirchhoff_kernel.py` - lines 1418-1440, 1555-1558, 1654-1663
- `processors/migration/migration_engine.py` - lines 768-772

**Changes:**
1. Added grid parameters to `migrate_kirchhoff_time_domain_rms()`:
   - `origin_x`, `origin_y`: Grid origin coordinates
   - `il_spacing`, `xl_spacing`: Grid cell sizes
   - `azimuth_deg`: Grid rotation angle

2. Compute trace midpoint grid indices once at start:
```python
cos_az = np.cos(np.radians(azimuth_deg))
sin_az = np.sin(np.radians(azimuth_deg))
dx_from_origin = mid_x - origin_x
dy_from_origin = mid_y - origin_y
il_offset = dx_from_origin * cos_az + dy_from_origin * sin_az
xl_offset = -dx_from_origin * sin_az + dy_from_origin * cos_az
trace_il_all = (il_offset / il_spacing).long().clamp(0, n_il - 1)
trace_xl_all = (xl_offset / xl_spacing).long().clamp(0, n_xl - 1)
```

3. Extract filtered trace indices per tile:
```python
trace_il_f = trace_il_all[filtered_indices]
trace_xl_f = trace_xl_all[filtered_indices]
```

4. Use trace indices in scatter (replaced tile indices):
```python
# OLD (WRONG):
tile_il_exp = tile_il.view(1, n_tile, 1).expand(...)
tile_xl_exp = tile_xl.view(1, n_tile, 1).expand(...)

# NEW (CORRECT):
trace_il_exp = trace_il_f.view(1, 1, n_filtered).expand(...)
trace_xl_exp = trace_xl_f.view(1, 1, n_filtered).expand(...)
```

**Result:** Each trace now scatters to its own midpoint grid location, producing uniform coverage for uniform offset data.

**Tests:** All 22 tests pass (13 time-domain + 9 engine)

---

## Additional Issue Discovered: Grid/Data Mismatch (2025-12-11)

### Symptom
After fixing scatter indices, output was still all zeros.

### Root Cause Analysis
Debug logging revealed:
```
IL float idx: [-347.6, 637.4] (grid 0-462)
XL float idx: [-404.4, 237.7] (grid 0-320)
```

**Most traces fall OUTSIDE the output grid!** The wizard computed origin/grid from the FULL dataset, but user is migrating only a SUBSET (offset 300-330m).

### Data Comparison
| Dataset | Midpoint Y Range |
|---------|------------------|
| Full dataset | 5,110,605 - 5,119,957 |
| 300-330m offset | 5,110,700 - 5,119,851 |
| **Output grid** | **5,116,498 - 5,126,085** |

The grid starts at Y=5,116,498 but most data is below Y=5,116,498!

### Grid Coordinate Check
```
Data corner (618911, 5110700) -> IL=-347.6, XL=-153.6  (OUTSIDE)
Data corner (627156, 5110700) -> IL=80.7, XL=-404.4   (OUTSIDE)
Data corner (618911, 5119851) -> IL=209.0, XL=84.1    (INSIDE)
Data corner (627156, 5119851) -> IL=637.4, XL=-166.7  (OUTSIDE)
```

Only 1 of 4 corners is inside the grid!

### Solution Options
1. **Wizard fix**: Recompute grid when user selects offset range (proper fix)
2. **User workaround**: Adjust wizard origin/grid to cover the offset subset
3. **Kernel improvement**: Add warning when traces fall outside grid (implemented)

### Warning Added
The kernel now logs a warning when traces fall outside the grid:
```
WARNING: 85.3% of traces (125000/146446) fall OUTSIDE the output grid!
  Traces in IL range: 45000, in XL range: 30000, in both: 21446
  Check that output grid origin/extent matches the input trace geometry
```

### Immediate Workaround
For offset 300-330m subset, adjust wizard parameters:
- Origin: (~618900, ~5110700) instead of (618813, 5116498)
- Or select full offset range to match the grid
