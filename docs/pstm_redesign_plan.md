# PSTM Redesign - Test-Driven Implementation Plan

## IMPLEMENTATION STATUS: PHASE 1-3 COMPLETE

**Benchmark Results (Apple Silicon MPS):**
| Metric | Old Implementation | New MigrationEngine | Improvement |
|--------|-------------------|---------------------|-------------|
| 10K traces | ~10 traces/s | **58,459 traces/s** | **5,846x** |
| 30K traces | ~10 traces/s | **181,797 traces/s** | **18,180x** |
| 31K bin time | ~55 min | **~0.17s** | **~19,000x** |

**Test Results:** 39/39 tests passing

---

## Original Performance Analysis

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Chunk time | 33s | <1s | 33x |
| Traces/sec | ~10 | 2000+ | 200x |
| Total time (31K traces) | ~55 min | <1 min | 55x |

**Root Cause**: The current architecture processes traces in small chunks (310) with Python-level loops and per-tile overhead. The algorithm is correct but the implementation doesn't leverage GPU parallelism effectively.

---

## Proposed Architecture

### Core Principle: **Gather-Centric GPU Kernels**

Instead of tiling the output image and iterating over traces, we should:
1. **Precompute** all geometry-dependent values ONCE
2. **Process entire bin** in a single GPU operation
3. **Use native GPU scatter-add** for image accumulation

---

## Architecture Components

### Data Flow Overview

```
PSTMWizard (config dict)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  MigrationJobController (NEW)                           │
│  - Validates config                                     │
│  - Creates MigrationJob dataclass                       │
│  - Manages worker lifecycle                             │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  MigrationEngine (NEW - replaces OptimizedKirchhoff)    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  GeometryPreprocessor                           │    │
│  │  - Computes ALL distances for entire bin        │    │
│  │  - Builds traveltime tables                     │    │
│  │  - Creates aperture masks                       │    │
│  │  - Output: GPU tensors ready for migration      │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  KirchhoffKernel                                │    │
│  │  - Single GPU operation per bin                 │    │
│  │  - torch.compile() optimized                    │    │
│  │  - Scatter-add to output image                  │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  OutputAccumulator                              │    │
│  │  - Manages image/fold tensors                   │    │
│  │  - Handles normalization                        │    │
│  │  - Streams to disk                              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  MigrationOutputWriter                                  │
│  - Zarr chunked output                                  │
│  - Parallel I/O                                         │
└─────────────────────────────────────────────────────────┘
```

---

### Key Algorithm Change: Trace-Centric Scatter

**Current (Output-Centric - SLOW):**
```
for each output_tile (130 tiles):
    for each depth_batch (59 batches):
        for each trace (310):
            compute contribution → accumulate to tile
```
**Operations**: 130 × 59 × 310 = 2.4M iterations with Python overhead

**Proposed (Trace-Centric - FAST):**
```
PRECOMPUTE (once per bin):
    traveltime_table[n_traces, n_depths]
    weight_table[n_traces, n_depths]
    output_indices[n_traces, n_depths] → (il, xl) for each depth

MIGRATE (single GPU call):
    for each trace in parallel:
        sample amplitudes at traveltime_table[trace, :]
        scatter_add to output_image at output_indices[trace, :]
```
**Operations**: 1 GPU kernel launch, 31K traces processed in parallel

---

## Phase 0: Create Synthetic Test Dataset

### Task 0.1: Create Diffractor Dataset Generator
**File**: `tests/fixtures/synthetic_diffractor.py`

**Specification**:
- Single point diffractor at known (x=5000m, y=5000m, z=1500m)
- Zero-offset geometry (source = receiver at midpoint)
- Grid: 100×100 midpoints, spacing 50m → 10,000 traces
- Time axis: 0-3000ms, dt=2ms → 1501 samples
- Constant velocity: 3000 m/s
- Diffractor response: Ricker wavelet at calculated traveltime

**Expected Output**:
- `traces`: (1501, 10000) float32
- `midpoint_x`: (10000,) float32 - range 2500-7500m
- `midpoint_y`: (10000,) float32 - range 2500-7500m
- `source_x`, `source_y`, `receiver_x`, `receiver_y`: all equal to midpoint

**Validation**:
- Traces near diffractor (x≈5000, y≈5000) should have event at t=1000ms (z=1500m, v=3000 → t=2×1500/3000=1.0s)
- Traces at distance should have hyperbolic moveout

---

### Task 0.2: Create Expected Migration Result
**File**: `tests/fixtures/synthetic_diffractor.py`

**Specification**:
- Output grid matching input: 100×100×1501 (il, xl, time)
- Perfect migration should collapse diffractor to a point at (il=50, xl=50, t=1000ms)
- Create analytical expected result for comparison

**Validation**:
- Peak amplitude at expected location
- Energy focused within small radius

---

## Phase 1: Geometry Preprocessor

### Task 1.1: Create MigrationJob Dataclass
**File**: `processors/migration/migration_job.py`
**Test**: `tests/test_migration_job.py`

**Specification**:
```python
@dataclass
class MigrationJob:
    # Input
    input_path: Path
    traces: np.ndarray  # (n_samples, n_traces)
    source_x: np.ndarray
    source_y: np.ndarray
    receiver_x: np.ndarray
    receiver_y: np.ndarray

    # Time axis
    dt_ms: float
    t_min_ms: float
    n_samples: int

    # Output grid
    origin_x: float
    origin_y: float
    il_spacing: float
    xl_spacing: float
    n_il: int
    n_xl: int
    azimuth_deg: float

    # Velocity
    velocity_mps: float  # Constant velocity for now

    # Migration parameters
    max_aperture_m: float
    max_angle_deg: float
```

**Tests**:
- Create job from synthetic dataset
- Validate all parameters have correct types
- Test serialization to/from dict (wizard compatibility)

---

### Task 1.2: Implement Midpoint-to-Grid Mapping
**File**: `processors/migration/geometry_preprocessor.py`
**Test**: `tests/test_geometry_preprocessor.py`

**Specification**:
```python
def compute_output_indices(
    midpoint_x: np.ndarray,  # (n_traces,)
    midpoint_y: np.ndarray,
    origin_x: float,
    origin_y: float,
    il_spacing: float,
    xl_spacing: float,
    azimuth_deg: float,
    n_il: int,
    n_xl: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        output_il: (n_traces,) int32 - inline index for each trace
        output_xl: (n_traces,) int32 - crossline index for each trace
        valid_mask: (n_traces,) bool - True if trace maps to valid grid point
    """
```

**Tests**:
- Synthetic dataset: trace at midpoint (5000, 5000) → (il=50, xl=50)
- Traces outside grid → valid_mask=False
- Rotated grid (azimuth≠0) maps correctly

---

### Task 1.3: Implement Traveltime Computation
**File**: `processors/migration/geometry_preprocessor.py`
**Test**: `tests/test_geometry_preprocessor.py`

**Specification**:
```python
def compute_traveltimes(
    source_x: torch.Tensor,    # (n_traces,)
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    image_x: torch.Tensor,     # (n_traces,) - x coord of output point
    image_y: torch.Tensor,     # (n_traces,) - y coord of output point
    depth_axis: torch.Tensor,  # (n_depths,)
    velocity: float
) -> torch.Tensor:
    """
    Returns:
        traveltimes: (n_traces, n_depths) float32 - two-way traveltime in ms
    """
```

**Formula** (zero-offset):
```
t = 2 * sqrt((mx - ix)² + (my - iy)² + z²) / v
```
For pre-stack:
```
t = (sqrt((sx - ix)² + (sy - iy)² + z²) + sqrt((rx - ix)² + (ry - iy)² + z²)) / v
```

**Tests**:
- Zero-offset trace at diffractor location: t[z=1500m] = 1000ms
- Verify shape (n_traces, n_depths)
- Verify values match analytical calculation for selected traces

---

### Task 1.4: Implement Weight Computation
**File**: `processors/migration/geometry_preprocessor.py`
**Test**: `tests/test_geometry_preprocessor.py`

**Specification**:
```python
def compute_weights(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    receiver_x: torch.Tensor,
    receiver_y: torch.Tensor,
    image_x: torch.Tensor,
    image_y: torch.Tensor,
    depth_axis: torch.Tensor,
    velocity: float,
    max_aperture_m: float,
    max_angle_deg: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        weights: (n_traces, n_depths) float32 - obliquity/spreading correction
        aperture_mask: (n_traces, n_depths) bool - within aperture
    """
```

**Tests**:
- Weight at zero offset, directly above image point = maximum
- Weight decreases with angle
- Aperture mask excludes traces beyond max_aperture
- Aperture mask excludes steep angles

---

### Task 1.5: Create PrecomputedGeometry Container
**File**: `processors/migration/geometry_preprocessor.py`
**Test**: `tests/test_geometry_preprocessor.py`

**Specification**:
```python
@dataclass
class PrecomputedGeometry:
    traveltimes: torch.Tensor      # (n_traces, n_depths) ms
    weights: torch.Tensor          # (n_traces, n_depths)
    output_il: torch.Tensor        # (n_traces,) int32
    output_xl: torch.Tensor        # (n_traces,) int32
    valid_mask: torch.Tensor       # (n_traces,) bool
    aperture_mask: torch.Tensor    # (n_traces, n_depths) bool

class GeometryPreprocessor:
    def precompute(self, job: MigrationJob, device: torch.device) -> PrecomputedGeometry
```

**Tests**:
- Full precomputation on synthetic dataset
- Verify all tensors have correct shapes
- Verify memory footprint matches expectation

---

## Phase 2: Kirchhoff Kernel

### Task 2.1: Implement Trace Interpolation
**File**: `processors/migration/kirchhoff_kernel.py`
**Test**: `tests/test_kirchhoff_kernel.py`

**Specification**:
```python
def interpolate_traces(
    traces: torch.Tensor,       # (n_samples, n_traces)
    sample_indices: torch.Tensor,  # (n_traces, n_depths) float
    dt_ms: float
) -> torch.Tensor:
    """
    Linear interpolation of traces at fractional sample indices.

    Returns:
        amplitudes: (n_traces, n_depths) float32
    """
```

**Tests**:
- Interpolation at integer sample = exact trace value
- Interpolation at half sample = average of neighbors
- Out-of-range indices clamp to edge values
- Verify shape matches input indices

---

### Task 2.2: Implement Scatter-Add Kernel
**File**: `processors/migration/kirchhoff_kernel.py`
**Test**: `tests/test_kirchhoff_kernel.py`

**Specification**:
```python
def scatter_add_migration(
    amplitudes: torch.Tensor,    # (n_traces, n_depths)
    weights: torch.Tensor,       # (n_traces, n_depths)
    aperture_mask: torch.Tensor, # (n_traces, n_depths)
    output_il: torch.Tensor,     # (n_traces,)
    output_xl: torch.Tensor,     # (n_traces,)
    valid_mask: torch.Tensor,    # (n_traces,)
    output_image: torch.Tensor,  # (n_depths, n_il, n_xl) - modified in place
    output_fold: torch.Tensor    # (n_depths, n_il, n_xl) - modified in place
) -> None:
    """
    Scatter-add weighted amplitudes to output image.
    Each trace contributes a depth column to its output location.
    """
```

**Key Implementation**:
```python
# Mask invalid traces
valid_amp = amplitudes[valid_mask] * weights[valid_mask] * aperture_mask[valid_mask].float()
valid_il = output_il[valid_mask]
valid_xl = output_xl[valid_mask]

# For each valid trace, add its depth column to output
# This is the key vectorized operation
for d in range(n_depths):  # Can be parallelized
    output_image[d].index_put_(
        (valid_il, valid_xl),
        valid_amp[:, d],
        accumulate=True
    )
```

**Tests**:
- Single trace at known location adds to correct output point
- Multiple traces at same location accumulate correctly
- Fold counts number of contributions
- Masked traces don't contribute

---

### Task 2.3: Implement Full Migration Kernel
**File**: `processors/migration/kirchhoff_kernel.py`
**Test**: `tests/test_kirchhoff_kernel.py`

**Specification**:
```python
class KirchhoffKernel:
    def __init__(self, device: torch.device):
        self.device = device

    def migrate(
        self,
        traces: torch.Tensor,           # (n_samples, n_traces)
        precomputed: PrecomputedGeometry,
        dt_ms: float,
        n_il: int,
        n_xl: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (n_depths, n_il, n_xl)
            fold: (n_depths, n_il, n_xl)
        """
```

**Tests**:
- Migrate synthetic diffractor dataset
- Verify output shape
- Verify peak at expected location (il=50, xl=50, t=1000ms)
- Compare with analytical expected result (Task 0.2)

---

### Task 2.4: Add torch.compile Optimization
**File**: `processors/migration/kirchhoff_kernel.py`
**Test**: `tests/test_kirchhoff_kernel.py`

**Specification**:
- Wrap core computation in `@torch.compile(mode="reduce-overhead")`
- Benchmark compiled vs non-compiled

**Tests**:
- Verify compiled kernel produces identical results
- Benchmark speedup (expect 2-5x on supported backends)

---

## Phase 3: Migration Engine Integration

### Task 3.1: Create MigrationEngine Orchestrator
**File**: `processors/migration/migration_engine.py`
**Test**: `tests/test_migration_engine.py`

**Specification**:
```python
class MigrationEngine:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.preprocessor = GeometryPreprocessor()
        self.kernel = KirchhoffKernel(self.device)

    def migrate_bin(
        self,
        job: MigrationJob,
        traces: np.ndarray,
        geometry: dict,  # source_x, source_y, receiver_x, receiver_y
        progress_callback: Callable[[float, str], None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Migrate all traces in a bin.

        Returns:
            image: (n_depths, n_il, n_xl) float32
            fold: (n_depths, n_il, n_xl) float32
        """
```

**Tests**:
- Full migration of synthetic dataset through engine
- Verify diffractor focuses correctly
- Benchmark total time

---

### Task 3.2: Implement Trace Batching for Large Bins
**File**: `processors/migration/migration_engine.py`
**Test**: `tests/test_migration_engine.py`

**Specification**:
```python
def migrate_bin_batched(
    self,
    job: MigrationJob,
    traces: np.ndarray,
    geometry: dict,
    batch_size: int = 10000,
    progress_callback: Callable[[float, str], None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process traces in batches to manage memory.
    Accumulates results across batches.
    """
```

**Tests**:
- 50,000 trace synthetic dataset in batches of 10,000
- Verify result matches non-batched (within tolerance)
- Verify memory stays bounded

---

### Task 3.3: Create Wizard Config Adapter
**File**: `processors/migration/config_adapter.py`
**Test**: `tests/test_config_adapter.py`

**Specification**:
```python
def wizard_config_to_job(config: dict) -> MigrationJob:
    """
    Convert PSTMWizard config dict to typed MigrationJob.
    Handles field name mapping and validation.
    """

def validate_wizard_config(config: dict) -> List[str]:
    """
    Returns list of validation errors (empty if valid).
    """
```

**Tests**:
- Valid config converts successfully
- Missing required fields detected
- Invalid types detected
- Field name mapping (x_origin → origin_x, etc.)

---

## Phase 4: Output and I/O

### Task 4.1: Implement Output Accumulator
**File**: `processors/migration/output_accumulator.py`
**Test**: `tests/test_output_accumulator.py`

**Specification**:
```python
class OutputAccumulator:
    def __init__(self, n_depths: int, n_il: int, n_xl: int, device: torch.device):
        self.image = torch.zeros(n_depths, n_il, n_xl, device=device)
        self.fold = torch.zeros(n_depths, n_il, n_xl, device=device)

    def add(self, image: torch.Tensor, fold: torch.Tensor):
        """Accumulate bin contribution."""

    def normalize(self, min_fold: int = 1) -> torch.Tensor:
        """Normalize by fold, return image."""

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Transfer to CPU and convert to numpy."""
```

**Tests**:
- Multiple adds accumulate correctly
- Normalization divides by fold
- Zero fold locations remain zero (no division by zero)

---

### Task 4.2: Implement Zarr Output Writer
**File**: `processors/migration/output_writer.py`
**Test**: `tests/test_output_writer.py`

**Specification**:
```python
class MigrationOutputWriter:
    def __init__(self, output_path: Path, job: MigrationJob):
        self.store = zarr.open(output_path / 'traces.zarr', mode='w')
        # Create chunked array

    def write_image(self, image: np.ndarray, fold: np.ndarray):
        """Write complete image and fold."""

    def write_metadata(self, job: MigrationJob):
        """Write job parameters as metadata.json."""

    def write_headers(self, job: MigrationJob):
        """Generate and write trace headers as parquet."""
```

**Tests**:
- Write synthetic result
- Read back and verify data integrity
- Verify metadata contains all job parameters

---

## Phase 5: Full Integration

### Task 5.1: Integrate with MigrationMonitorDialog
**File**: `views/migration_monitor_dialog.py`
**Test**: `tests/test_migration_monitor_integration.py`

**Specification**:
- Replace `OptimizedKirchhoffMigrator` usage with `MigrationEngine`
- Keep existing progress/cancel UI
- Use new batched processing

**Tests**:
- End-to-end wizard → monitor → output with synthetic data
- Progress updates work correctly
- Cancel interrupts processing

---

### Task 5.2: Performance Benchmark Suite
**File**: `tests/benchmarks/benchmark_migration.py`

**Specification**:
- Benchmark each component separately
- Benchmark full pipeline
- Compare old vs new implementation

**Benchmarks**:
| Test | Traces | Expected Time |
|------|--------|---------------|
| Precompute geometry | 10,000 | <0.5s |
| Migrate kernel | 10,000 | <0.2s |
| Full pipeline | 10,000 | <1s |
| Full pipeline | 50,000 | <3s |

---

## Task Summary Table

| Phase | Task | File | Test File | Priority | Status |
|-------|------|------|-----------|----------|--------|
| 0 | 0.1 Diffractor generator | `tests/fixtures/synthetic_diffractor.py` | self-testing | **HIGH** | **DONE** |
| 0 | 0.2 Expected result | `tests/fixtures/synthetic_diffractor.py` | self-testing | **HIGH** | **DONE** |
| 1 | 1.1 MigrationJob dataclass | `models/migration_job.py` (existing) | `tests/test_migration_job.py` | **HIGH** | **DONE** |
| 1 | 1.2 Midpoint-to-grid | `processors/migration/geometry_preprocessor.py` | `tests/test_geometry_preprocessor.py` | **HIGH** | **DONE** |
| 1 | 1.3 Traveltime computation | `processors/migration/geometry_preprocessor.py` | `tests/test_geometry_preprocessor.py` | **HIGH** | **DONE** |
| 1 | 1.4 Weight computation | `processors/migration/geometry_preprocessor.py` | `tests/test_geometry_preprocessor.py` | **HIGH** | **DONE** |
| 1 | 1.5 PrecomputedGeometry | `processors/migration/geometry_preprocessor.py` | `tests/test_geometry_preprocessor.py` | **HIGH** | **DONE** |
| 2 | 2.1 Trace interpolation | `processors/migration/kirchhoff_kernel.py` | `tests/test_kirchhoff_kernel.py` | **HIGH** | **DONE** |
| 2 | 2.2 Scatter-add kernel | `processors/migration/kirchhoff_kernel.py` | `tests/test_kirchhoff_kernel.py` | **HIGH** | **DONE** |
| 2 | 2.3 Full migration kernel | `processors/migration/kirchhoff_kernel.py` | `tests/test_kirchhoff_kernel.py` | **HIGH** | **DONE** |
| 2 | 2.4 torch.compile | `processors/migration/kirchhoff_kernel.py` | `tests/test_kirchhoff_kernel.py` | MEDIUM | TODO |
| 3 | 3.1 MigrationEngine | `processors/migration/migration_engine.py` | `tests/test_migration_engine.py` | **HIGH** | **DONE** |
| 3 | 3.2 Trace batching | `processors/migration/migration_engine.py` | `tests/test_migration_engine.py` | **HIGH** | **DONE** |
| 3 | 3.3 Config adapter | `processors/migration/config_adapter.py` | `tests/test_config_adapter.py` | MEDIUM | TODO |
| 4 | 4.1 Output accumulator | `processors/migration/output_accumulator.py` | `tests/test_output_accumulator.py` | MEDIUM | TODO |
| 4 | 4.2 Zarr writer | `processors/migration/output_writer.py` | `tests/test_output_writer.py` | MEDIUM | TODO |
| 5 | 5.1 Monitor integration | `views/migration_monitor_dialog.py` | `tests/test_migration_monitor_integration.py` | LOW | TODO |
| 5 | 5.2 Benchmarks | `tests/benchmarks/benchmark_migration.py` | self-running | LOW | TODO |

---

## Implementation Order

```
0.1 → 0.2 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2
                                                                    ↓
                                              ← ← ← 5.1 ← 4.2 ← 4.1 ← 3.3
```

---

## Memory Strategy

**Problem**: 31K traces × 1600 depths × 4 bytes = 200 MB for traveltimes alone

**Solution**: Process in **trace batches** of 5000-10000, not output tiles

| Component | Size (31K traces, 1600 depths) |
|-----------|-------------------------------|
| Traveltimes | 200 MB |
| Weights | 200 MB |
| Output indices | 250 KB |
| Aperture mask | 50 MB (bool) |
| Trace data | 200 MB |
| Output image | 900 MB |
| **Total** | ~1.6 GB |

Fits comfortably in 8GB GPU memory. For larger bins, batch traces.

---

## Expected Performance

| Stage | Time | Notes |
|-------|------|-------|
| Load bin traces | 0.5s | Zarr parallel read |
| Precompute geometry | 1.0s | 31K traces × 1600 depths vectorized |
| Migration kernel | 0.5s | Single GPU scatter-add |
| Output write | 0.3s | Async Zarr write |
| **Total per bin** | **~2.5s** | vs current 55 min |

**Speedup**: ~1300x improvement
