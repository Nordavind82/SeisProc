# PSTM Optimization Implementation Plan

## Overview

This plan implements six major optimizations for the Kirchhoff PSTM migrator to achieve 10-30x speedup while maintaining full output quality:

1. **Traveltime Table Pre-computation** - Replace on-the-fly sqrt calculations with table lookups
2. **Depth-Dependent Aperture Optimization** - Reduce contributing traces at shallow depths
3. **Output-Driven Trace Selection** - Spatial indexing to skip irrelevant traces
4. **Incremental/Streaming Migration** - Reduce memory footprint via accumulation
5. **Symmetry Exploitation** - Leverage source-receiver reciprocity
6. **GPU Memory Tiling** - Optimize data transfer patterns

---

## Phase 1: Traveltime Table Pre-computation (Days 1-2)

### Objective
Replace expensive `sqrt(h² + z²) / v` computations with fast table interpolation.

### Current State
- `_migrate_chunk()` computes traveltimes on-the-fly for every (point, depth, trace) combination
- Existing `TraveltimeTable` and `TraveltimeTableBuilder` classes exist but aren't integrated into main loop

### Implementation

#### 1.1 Create Optimized Traveltime Lookup Table

**File:** `processors/migration/traveltime_lut.py`

```
TraveltimeLUT:
    - Pre-compute t(h, z) for discrete (offset, depth) pairs
    - Table dimensions: n_offsets × n_depths (e.g., 500 × 1000 = 500K entries)
    - Offset range: 0 to max_aperture_m (5000m default)
    - Depth range: 0 to max_time * v / 2

    Methods:
    - build(velocity_model, max_offset, max_depth, n_offsets, n_depths)
    - lookup_batch(h_array, z_array) -> t_array  # Bilinear interpolation
    - to_device(device) -> move table to GPU
    - save/load for disk caching
```

#### 1.2 Integrate into KirchhoffMigrator

**File:** `processors/migration/kirchhoff_migrator.py`

- Add `traveltime_lut` attribute initialized in `_setup_components()`
- Replace traveltime calculation in `_migrate_chunk()`:
  ```python
  # Before:
  r_src = torch.sqrt(h_src_exp**2 + z_expanded**2)
  t_src = r_src / v

  # After:
  t_src = self.traveltime_lut.lookup_batch(h_src_exp, z_expanded)
  ```

#### 1.3 Tests

**File:** `tests/test_traveltime_lut.py`

- Test accuracy: compare LUT vs analytical for random points (error < 0.1%)
- Test interpolation quality at table boundaries
- Test GPU/CPU parity
- Benchmark: measure speedup vs on-the-fly calculation

### Expected Speedup: 2-3x

---

## Phase 2: Depth-Dependent Aperture Optimization (Days 3-4)

### Objective
Reduce computation at shallow depths where aperture is naturally smaller.

### Concept
- At depth z, max horizontal distance for angle θ is: `h_max = z * tan(θ)`
- Shallow depths → small aperture → fewer contributing traces
- Deep depths → large aperture but fewer depth samples contribute

### Implementation

#### 2.1 Create Depth-Adaptive Aperture Calculator

**File:** `processors/migration/aperture_adaptive.py`

```
DepthAdaptiveAperture:
    - Pre-compute effective aperture per depth level
    - aperture(z) = min(max_aperture, z * tan(max_angle))

    Methods:
    - compute_depth_apertures(z_axis) -> array of aperture per depth
    - get_contributing_trace_mask(z_idx, h_src, h_rcv) -> sparse mask
    - estimate_active_traces_per_depth() -> planning statistics
```

#### 2.2 Modify _migrate_chunk for Depth-Adaptive Processing

Process depths in groups with similar aperture requirements:
- Group 1: z < 500m, aperture ~500m (few traces)
- Group 2: 500m < z < 2000m, aperture ~2000m
- Group 3: z > 2000m, full aperture

```python
def _migrate_chunk_adaptive(self, ...):
    depth_groups = self.aperture_adaptive.group_depths_by_aperture(z_axis)

    for group in depth_groups:
        z_subset = z_axis[group.indices]
        trace_mask = group.contributing_traces  # Pre-computed
        traces_subset = traces[:, trace_mask]

        # Process only relevant traces for this depth group
        self._migrate_depth_group(traces_subset, z_subset, ...)
```

#### 2.3 Tests

**File:** `tests/test_aperture_adaptive.py`

- Verify aperture calculation matches theory
- Test that shallow depths correctly select fewer traces
- Verify output matches non-optimized version (within tolerance)
- Benchmark trace reduction ratio per depth

### Expected Speedup: 1.5-2x (geometry dependent)

---

## Phase 3: Output-Driven Trace Selection (Days 5-7)

### Objective
For each output chunk, only process traces that can geometrically contribute.

### Concept
- Build spatial index of trace midpoints (CDP locations)
- For output chunk covering (x_min, x_max, y_min, y_max):
  - Query index for traces within max_aperture of chunk bounds
  - Skip traces that cannot contribute to any point in chunk

### Implementation

#### 3.1 Create Trace Spatial Index

**File:** `processors/migration/trace_index.py`

```
TraceSpatialIndex:
    - Build KD-tree or grid-based index of trace midpoints
    - Support query by bounding box + buffer

    Methods:
    - build(source_x, source_y, receiver_x, receiver_y)
    - query_traces_for_region(x_min, x_max, y_min, y_max, buffer) -> trace_indices
    - query_traces_for_point(x, y, max_dist) -> trace_indices
    - get_trace_bounds() -> (x_min, x_max, y_min, y_max)
```

#### 3.2 Integrate into Migration Loop

**File:** `processors/migration/kirchhoff_migrator.py`

```python
def migrate_gather(self, gather, geometry, ...):
    # Build spatial index once per gather
    if self.trace_index is None or self._geometry_changed(geometry):
        self.trace_index = TraceSpatialIndex()
        self.trace_index.build(geometry.source_x, geometry.source_y,
                               geometry.receiver_x, geometry.receiver_y)

    for il_start, il_end in chunk_inline_ranges:
        for xl_start, xl_end in chunk_xline_ranges:
            # Get output chunk bounds in world coordinates
            x_min, x_max, y_min, y_max = self._chunk_to_world_bounds(...)

            # Query relevant traces
            relevant_traces = self.trace_index.query_traces_for_region(
                x_min, x_max, y_min, y_max,
                buffer=self.config.max_aperture_m
            )

            if len(relevant_traces) == 0:
                continue  # Skip empty chunk

            # Process only relevant traces
            self._migrate_chunk(traces[:, relevant_traces], ...)
```

#### 3.3 Tests

**File:** `tests/test_trace_index.py`

- Test spatial index correctness (all contributing traces found)
- Test no false negatives (verify against brute force)
- Test query performance for various chunk sizes
- Benchmark trace reduction ratio for realistic geometries

### Expected Speedup: 3-5x (highly geometry dependent)

---

## Phase 4: Incremental/Streaming Migration (Days 8-9)

### Objective
Reduce peak memory usage by accumulating directly to output instead of materializing full intermediate tensors.

### Current Problem
- Creating `(n_z, n_points, n_traces)` tensor consumes huge memory
- For 1000 depths × 10K points × 500 traces = 20GB float32

### Implementation

#### 4.1 Restructure _migrate_chunk for Streaming

**File:** `processors/migration/kirchhoff_migrator.py`

```python
def _migrate_chunk_streaming(self, traces, ..., img_x, img_y, z_axis):
    """Process one trace at a time, accumulate directly to output."""
    n_z, n_il, n_xl = len(z_axis), len(img_x), len(img_y)
    n_traces = traces.shape[1]

    # Initialize output (kept in GPU memory)
    image_chunk = torch.zeros(n_z, n_il, n_xl, device=self.device)
    fold_chunk = torch.zeros(n_z, n_il, n_xl, device=self.device)

    # Pre-compute image point grid (reused for all traces)
    img_xx, img_yy = torch.meshgrid(img_x, img_y, indexing='ij')

    # Stream through traces in small batches
    stream_batch_size = 16  # Tune based on GPU memory

    for t_start in range(0, n_traces, stream_batch_size):
        t_end = min(t_start + stream_batch_size, n_traces)

        # Compute contributions from this trace batch
        contrib, fold_contrib = self._compute_trace_contribution(
            traces[:, t_start:t_end],
            src_x[t_start:t_end], src_y[t_start:t_end],
            rcv_x[t_start:t_end], rcv_y[t_start:t_end],
            img_xx, img_yy, z_axis
        )

        # Accumulate (in-place addition)
        image_chunk += contrib
        fold_chunk += fold_contrib

    return image_chunk, fold_chunk
```

#### 4.2 Memory-Efficient Contribution Calculation

```python
def _compute_trace_contribution(self, trace_batch, sx, sy, rx, ry, img_xx, img_yy, z_axis):
    """Compute contribution from small batch of traces to all image points."""
    batch_size = trace_batch.shape[1]
    n_z = len(z_axis)
    n_il, n_xl = img_xx.shape

    # Distances: (n_il, n_xl, batch)
    h_src = torch.sqrt((img_xx.unsqueeze(-1) - sx)**2 +
                       (img_yy.unsqueeze(-1) - sy)**2)
    h_rcv = torch.sqrt((img_xx.unsqueeze(-1) - rx)**2 +
                       (img_yy.unsqueeze(-1) - ry)**2)

    # Process each depth level to avoid huge 4D tensor
    contrib = torch.zeros(n_z, n_il, n_xl, device=self.device)
    fold_contrib = torch.zeros(n_z, n_il, n_xl, device=self.device)

    for z_idx, z in enumerate(z_axis):
        t_src = self.traveltime_lut.lookup(h_src, z)  # (n_il, n_xl, batch)
        t_rcv = self.traveltime_lut.lookup(h_rcv, z)
        t_total = t_src + t_rcv

        # Aperture and weights for this depth
        mask = self._compute_mask_single_depth(h_src, h_rcv, z)
        weights = mask / (torch.sqrt(h_src**2 + z**2) * torch.sqrt(h_rcv**2 + z**2) + 1e-6)

        # Interpolate and accumulate
        amps = self._interpolate_traces(trace_batch, t_total)  # (n_il, n_xl, batch)

        contrib[z_idx] = torch.sum(amps * weights, dim=-1)
        fold_contrib[z_idx] = torch.sum(mask, dim=-1)

    return contrib, fold_contrib
```

#### 4.3 Tests

**File:** `tests/test_streaming_migration.py`

- Verify streaming output matches batch output (within floating point tolerance)
- Measure peak memory usage vs batch version
- Benchmark throughput comparison
- Test with various trace counts and grid sizes

### Expected Memory Reduction: 10-100x
### Expected Speedup: Neutral to slight improvement (better cache locality)

---

## Phase 5: Symmetry Exploitation (Days 10-11)

### Objective
Leverage source-receiver reciprocity to reduce traveltime computations by ~50%.

### Concept
For isotropic media: `t(source → image → receiver) = t(receiver → image → source)`

When processing trace pairs with swapped source/receiver positions:
- Compute traveltime once, use for both traces
- Particularly valuable for dense acquisition geometries

### Implementation

#### 5.1 Create Trace Pair Matcher

**File:** `processors/migration/symmetry.py`

```
TraceSymmetryMatcher:
    - Identify trace pairs where (s1,r1) ≈ (r2,s2)
    - Group traces for shared traveltime computation

    Methods:
    - find_reciprocal_pairs(src_x, src_y, rcv_x, rcv_y, tolerance=1.0)
        -> list of (trace_i, trace_j) pairs
    - group_by_midpoint(src_x, src_y, rcv_x, rcv_y)
        -> dict mapping midpoint_bin to trace_indices
    - create_symmetric_index(geometry) -> optimized structure for lookup
```

#### 5.2 Optimize Traveltime Calculation

```python
def _compute_traveltimes_with_symmetry(self, geometry, img_x, img_y, z_axis):
    """Compute traveltimes exploiting reciprocity."""
    # Group traces by midpoint
    midpoint_groups = self.symmetry_matcher.group_by_midpoint(geometry)

    # For each group, compute traveltimes to/from midpoint
    # Then adjust for actual source/receiver offset

    traveltimes = {}
    for midpoint, trace_indices in midpoint_groups.items():
        # Compute midpoint-to-image traveltime once
        t_mid_to_img = self.traveltime_lut.lookup_batch(
            distance_to_midpoint, z_axis
        )

        # Adjust for each trace's actual source/receiver position
        for trace_idx in trace_indices:
            offset = geometry.offset[trace_idx]
            # Small correction based on offset from midpoint
            traveltimes[trace_idx] = t_mid_to_img + self._offset_correction(offset, z_axis)

    return traveltimes
```

#### 5.3 Tests

**File:** `tests/test_symmetry.py`

- Verify reciprocal pair detection accuracy
- Test traveltime equivalence for swapped pairs
- Benchmark computation reduction for various geometries
- Verify output quality maintained

### Expected Speedup: 1.3-1.5x (depends on acquisition geometry)

---

## Phase 6: GPU Memory Tiling (Days 12-14)

### Objective
Optimize GPU memory transfer patterns by keeping trace data resident and tiling output.

### Current Problem
- Each chunk iteration re-transfers trace data to GPU
- For 500 traces × 1600 samples × 4 bytes = 3.2 MB per transfer
- Thousands of transfers add up

### Implementation

#### 6.1 Create GPU Memory Manager

**File:** `processors/migration/gpu_memory.py`

```
GPUMemoryManager:
    - Pin trace data in GPU memory
    - Manage output tile allocation
    - Handle memory pressure gracefully

    Methods:
    - allocate_trace_buffer(n_traces, n_samples) -> GPU tensor
    - allocate_output_tile(tile_shape) -> GPU tensor
    - transfer_traces_to_gpu(traces_cpu) -> traces_gpu (pinned)
    - get_optimal_tile_size(total_traces, n_samples, available_memory)
    - sync_output_to_cpu(output_gpu) -> output_cpu
```

#### 6.2 Restructure Migration for Tiled Output

```python
def migrate_gather_tiled(self, gather, geometry, ...):
    """Tile output grid, keep traces resident in GPU."""

    # Transfer all trace data to GPU once
    traces_gpu = self.gpu_memory.transfer_traces_to_gpu(gather.traces)
    geometry_gpu = self._geometry_to_gpu(geometry)

    # Compute optimal tile size based on remaining GPU memory
    tile_size = self.gpu_memory.get_optimal_tile_size(
        gather.n_traces, gather.n_samples, available_memory
    )

    # Allocate output in tiles
    output_image = torch.zeros(n_z, n_inline, n_xline, device='cpu')
    output_fold = torch.zeros(n_z, n_inline, n_xline, device='cpu')

    # Process output in tiles
    for tile in self._generate_output_tiles(n_inline, n_xline, tile_size):
        # Allocate GPU tile
        tile_image = self.gpu_memory.allocate_output_tile(tile.shape)
        tile_fold = self.gpu_memory.allocate_output_tile(tile.shape)

        # Migrate this tile (all traces contribute)
        self._migrate_tile(
            traces_gpu, geometry_gpu,
            tile.il_range, tile.xl_range, z_axis,
            tile_image, tile_fold
        )

        # Transfer tile result to CPU output
        output_image[:, tile.il_slice, tile.xl_slice] = tile_image.cpu()
        output_fold[:, tile.il_slice, tile.xl_slice] = tile_fold.cpu()

        # Free GPU tile memory
        del tile_image, tile_fold
        torch.cuda.empty_cache()  # or torch.mps.empty_cache()

    return output_image, output_fold
```

#### 6.3 Tests

**File:** `tests/test_gpu_memory.py`

- Test memory allocation and deallocation
- Verify tiled output matches non-tiled
- Benchmark GPU memory usage and transfer times
- Test behavior under memory pressure

### Expected Speedup: 1.2-1.5x (reduces CPU-GPU transfer overhead)

---

## Phase 7: Integration & Wizard Updates (Days 15-17)

### 7.1 Unified Optimized Migrator

**File:** `processors/migration/kirchhoff_migrator_optimized.py`

Create new class combining all optimizations:

```python
class OptimizedKirchhoffMigrator(BaseMigrator):
    """
    Production Kirchhoff PSTM with all optimizations enabled.

    Optimizations:
    - Traveltime lookup tables (2-3x)
    - Depth-adaptive aperture (1.5-2x)
    - Output-driven trace selection (3-5x)
    - Streaming accumulation (memory efficient)
    - Symmetry exploitation (1.3-1.5x)
    - GPU memory tiling (1.2-1.5x)

    Combined expected speedup: 10-30x
    """

    def __init__(self, velocity, config, optimization_config=None):
        super().__init__(velocity, config)

        self.opt_config = optimization_config or OptimizationConfig()

        # Initialize optimization components
        self.traveltime_lut = TraveltimeLUT()
        self.aperture_adaptive = DepthAdaptiveAperture()
        self.trace_index = TraceSpatialIndex()
        self.symmetry_matcher = TraceSymmetryMatcher()
        self.gpu_memory = GPUMemoryManager()

        self._build_traveltime_table()
```

### 7.2 Optimization Configuration

**File:** `models/optimization_config.py`

```python
@dataclass
class OptimizationConfig:
    """Configuration for migration optimizations."""

    # Traveltime table
    traveltime_table_enabled: bool = True
    traveltime_n_offsets: int = 500
    traveltime_n_depths: int = 1000

    # Depth-adaptive aperture
    depth_adaptive_enabled: bool = True
    min_depth_for_full_aperture: float = 2000.0  # meters

    # Trace selection
    trace_selection_enabled: bool = True
    spatial_index_type: str = 'kdtree'  # 'kdtree' or 'grid'

    # Streaming
    streaming_enabled: bool = True
    stream_batch_size: int = 32

    # Symmetry
    symmetry_enabled: bool = True
    symmetry_tolerance_m: float = 1.0

    # GPU tiling
    gpu_tiling_enabled: bool = True
    target_gpu_memory_gb: float = 4.0
```

### 7.3 Wizard Integration

**File:** `views/pstm_wizard_dialog.py`

Add "Performance" page to wizard:

```python
class PerformancePage(QWizardPage):
    """Page for configuring optimization settings."""

    def __init__(self, wizard):
        super().__init__()
        self.setTitle("Performance Optimization")
        self.setSubTitle("Configure processing optimizations")

        layout = QVBoxLayout()

        # Preset selector
        preset_group = QGroupBox("Optimization Preset")
        preset_layout = QVBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Maximum Quality (slowest)",
            "Balanced (recommended)",
            "Maximum Speed (preview)",
            "Custom..."
        ])
        preset_layout.addWidget(self.preset_combo)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Estimated performance
        self.perf_estimate_label = QLabel()
        layout.addWidget(self.perf_estimate_label)

        # Advanced options (collapsed by default)
        self.advanced_group = QGroupBox("Advanced Options")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        # ... add individual optimization toggles ...

        layout.addWidget(self.advanced_group)
        self.setLayout(layout)

    def _update_estimate(self):
        """Update performance estimate based on settings."""
        n_traces = self.wizard()._config.get('n_traces', 0)
        base_time = n_traces / 100  # Base estimate: 100 traces/sec

        speedup = 1.0
        if self.preset_combo.currentIndex() == 2:  # Max speed
            speedup = 20.0
        elif self.preset_combo.currentIndex() == 1:  # Balanced
            speedup = 10.0

        estimated_time = base_time / speedup

        if estimated_time < 60:
            time_str = f"{estimated_time:.0f} seconds"
        elif estimated_time < 3600:
            time_str = f"{estimated_time/60:.1f} minutes"
        else:
            time_str = f"{estimated_time/3600:.1f} hours"

        self.perf_estimate_label.setText(
            f"Estimated processing time: {time_str}\n"
            f"({n_traces:,} traces at ~{100*speedup:.0f} traces/sec)"
        )
```

### 7.4 MigrationMonitorDialog Updates

**File:** `views/migration_monitor_dialog.py`

- Use OptimizedKirchhoffMigrator when optimizations enabled
- Add optimization status to progress display
- Show cache hit rates and memory usage

---

## Phase 8: Testing & Validation (Days 18-20)

### 8.1 Unit Tests

Each optimization has dedicated test file:
- `tests/test_traveltime_lut.py`
- `tests/test_aperture_adaptive.py`
- `tests/test_trace_index.py`
- `tests/test_streaming_migration.py`
- `tests/test_symmetry.py`
- `tests/test_gpu_memory.py`

### 8.2 Integration Tests

**File:** `tests/test_optimized_migrator.py`

- Compare optimized vs reference output (L2 error < 1%)
- Test all optimization combinations
- Memory usage validation
- Performance regression tests

### 8.3 Benchmark Suite

**File:** `tests/benchmarks/benchmark_migration.py`

```python
def benchmark_configurations():
    """Benchmark various optimization configurations."""

    configurations = [
        ("Baseline", {}),
        ("TT Table Only", {"traveltime_table_enabled": True}),
        ("Depth Adaptive", {"depth_adaptive_enabled": True}),
        ("Trace Selection", {"trace_selection_enabled": True}),
        ("All Optimizations", {...}),
    ]

    for name, config in configurations:
        migrator = OptimizedKirchhoffMigrator(velocity, mig_config, config)

        start = time.time()
        result = migrator.migrate_gather(gather, geometry)
        elapsed = time.time() - start

        print(f"{name}: {elapsed:.2f}s ({gather.n_traces/elapsed:.0f} traces/s)")
```

---

## Deliverables Summary

### New Files
1. `processors/migration/traveltime_lut.py` - Traveltime lookup table
2. `processors/migration/aperture_adaptive.py` - Depth-dependent aperture
3. `processors/migration/trace_index.py` - Spatial trace index
4. `processors/migration/symmetry.py` - Reciprocity exploitation
5. `processors/migration/gpu_memory.py` - GPU memory manager
6. `processors/migration/kirchhoff_migrator_optimized.py` - Unified optimized migrator
7. `models/optimization_config.py` - Optimization settings

### Modified Files
1. `processors/migration/kirchhoff_migrator.py` - Add streaming mode
2. `views/pstm_wizard_dialog.py` - Add Performance page
3. `views/migration_monitor_dialog.py` - Use optimized migrator

### Test Files
1. `tests/test_traveltime_lut.py`
2. `tests/test_aperture_adaptive.py`
3. `tests/test_trace_index.py`
4. `tests/test_streaming_migration.py`
5. `tests/test_symmetry.py`
6. `tests/test_gpu_memory.py`
7. `tests/test_optimized_migrator.py`
8. `tests/benchmarks/benchmark_migration.py`

---

## Timeline Summary

| Phase | Days | Deliverable | Expected Speedup |
|-------|------|-------------|------------------|
| 1. Traveltime Tables | 1-2 | traveltime_lut.py | 2-3x |
| 2. Depth-Adaptive Aperture | 3-4 | aperture_adaptive.py | 1.5-2x |
| 3. Trace Selection | 5-7 | trace_index.py | 3-5x |
| 4. Streaming Migration | 8-9 | Refactored _migrate_chunk | Memory 10-100x |
| 5. Symmetry | 10-11 | symmetry.py | 1.3-1.5x |
| 6. GPU Tiling | 12-14 | gpu_memory.py | 1.2-1.5x |
| 7. Integration | 15-17 | Wizard + unified migrator | - |
| 8. Testing | 18-20 | Full test suite | - |

**Total: ~20 working days**

**Combined Expected Speedup: 10-30x**

---

## Success Criteria

1. **Performance**: 870K traces processed in < 30 minutes per bin (vs current ~2 hours)
2. **Quality**: Output image L2 error < 1% vs reference implementation
3. **Memory**: Peak GPU memory < 6GB for any dataset size
4. **Stability**: All tests pass, no memory leaks
5. **Usability**: Wizard provides clear performance estimates and optimization presets
