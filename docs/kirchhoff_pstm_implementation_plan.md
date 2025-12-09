# 3D Anisotropic Curved Ray Pre-Stack Kirchhoff Time Migration
## Implementation Roadmap

**Document Version:** 2.0
**Created:** December 2024
**Updated:** December 2024
**Target Application:** SeisProc

---

## Executive Summary

This document outlines a phased implementation approach for GPU-accelerated 3D Pre-Stack Kirchhoff Time Migration (PSTM). The strategy prioritizes delivering a working MVP early, then incrementally adding sophistication through data management, anisotropy, curved rays, and advanced features.

**Total Phases:** 7
**MVP Completion:** Phase 2
**Production-Ready with Data Management:** Phase 3
**Full Feature Set:** Phase 7

---

## Phase 1: Foundation & Infrastructure

**Goal:** Establish core data structures, interfaces, header management, and testing framework

### 1.1 Data Models

| Task | Description | Files |
|------|-------------|-------|
| 1.1.1 | Create `VelocityModel` dataclass supporting 1D v(t) functions | `models/velocity_model.py` |
| 1.1.2 | Create `MigrationConfig` dataclass with basic parameters (aperture, angles, output grid) | `models/migration_config.py` |
| 1.1.3 | Create `MigrationGeometry` class for survey geometry (source/receiver coordinates) | `models/migration_geometry.py` |
| 1.1.4 | Add coordinate extraction utilities for SEG-Y headers (SX, SY, GX, GY) | `utils/geometry_utils.py` |

### 1.2 Header Definition & Mapping System

| Task | Description | Files |
|------|-------------|-------|
| 1.2.1 | Create `HeaderSchema` class defining required PSTM headers | `models/header_schema.py` |
| 1.2.2 | Define standard header requirements: SX, SY, GX, GY, OFFSET, AZIMUTH, INLINE, XLINE, CDP_X, CDP_Y | Same file |
| 1.2.3 | Create `HeaderMapping` class for user-defined input→app header mapping | `models/header_mapping.py` |
| 1.2.4 | Implement header mapping UI model (source header name → target field) | Same file |
| 1.2.5 | Add computed header support (e.g., compute OFFSET from SX,SY,GX,GY if missing) | `utils/header_calculator.py` |
| 1.2.6 | Add computed AZIMUTH calculation from coordinates | Same file |
| 1.2.7 | Header mapping validation (check required headers present/computable) | `models/header_mapping.py` |
| 1.2.8 | Header mapping serialization (save/load mapping configurations) | Same file |

### 1.3 Base Classes & Interfaces

| Task | Description | Files |
|------|-------------|-------|
| 1.3.1 | Create abstract `BaseMigrator` class with standard interface | `processors/migration/base_migrator.py` |
| 1.3.2 | Define `TraveltimeCalculator` interface (abstract base) | `processors/migration/traveltime.py` |
| 1.3.3 | Define `AmplitudeWeight` interface for weight computation | `processors/migration/weights.py` |
| 1.3.4 | Create `MigrationResult` container for output image + metadata | `models/migration_result.py` |

### 1.4 Testing Infrastructure

| Task | Description | Files |
|------|-------------|-------|
| 1.4.1 | Create synthetic point diffractor generator (known solution) | `tests/fixtures/synthetic_diffractor.py` |
| 1.4.2 | Create synthetic dipping reflector generator | `tests/fixtures/synthetic_reflector.py` |
| 1.4.3 | Create constant velocity test dataset | `tests/fixtures/synthetic_constant_v.py` |
| 1.4.4 | Set up migration accuracy metrics (focusing, positioning error) | `tests/utils/migration_metrics.py` |
| 1.4.5 | Create synthetic dataset with realistic headers (offset, azimuth, CDP) | `tests/fixtures/synthetic_prestack.py` |

### 1.5 Deliverables Checklist
- [ ] All dataclasses serializable via `to_dict()` / `from_dict()`
- [ ] Header mapping system functional
- [ ] Computed headers working (OFFSET, AZIMUTH from coordinates)
- [ ] Unit tests for all data models
- [ ] Documentation strings complete
- [ ] Integration with existing `DeviceManager`

---

## Phase 2: Isotropic Straight-Ray MVP

**Goal:** Working 3D PSTM with constant velocity — **THIS IS THE MVP**

### 2.1 Traveltime Computation (Straight Ray)

| Task | Description | Files |
|------|-------------|-------|
| 2.1.1 | Implement isotropic straight-ray traveltime: `t = sqrt(x^2 + y^2 + z^2) / V` | `processors/migration/traveltime_straight.py` |
| 2.1.2 | GPU kernel for batch traveltime computation (source leg) | Same file |
| 2.1.3 | GPU kernel for batch traveltime computation (receiver leg) | Same file |
| 2.1.4 | Vectorized total traveltime: `t_total = t_src + t_rcv` | Same file |
| 2.1.5 | Unit test: compare GPU vs analytical formula | `tests/test_traveltime_straight.py` |

### 2.2 Trace Interpolation

| Task | Description | Files |
|------|-------------|-------|
| 2.2.1 | Implement linear interpolation kernel (fast, baseline) | `processors/migration/interpolation.py` |
| 2.2.2 | Implement sinc interpolation kernel (8-point, accurate) | Same file |
| 2.2.3 | Batch interpolation: extract amplitudes at arbitrary times | Same file |
| 2.2.4 | Unit test: interpolation accuracy vs scipy | `tests/test_interpolation.py` |

### 2.3 Aperture & Geometry

| Task | Description | Files |
|------|-------------|-------|
| 2.3.1 | Implement circular aperture mask (max distance from image point) | `processors/migration/aperture.py` |
| 2.3.2 | Implement angle aperture mask (max angle from vertical) | Same file |
| 2.3.3 | Implement offset mute (max source-receiver offset) | Same file |
| 2.3.4 | Cosine taper at aperture boundaries | Same file |

### 2.4 Core Migration Engine

| Task | Description | Files |
|------|-------------|-------|
| 2.4.1 | Implement single-gather migration kernel (output-driven loop) | `processors/migration/kirchhoff_iso_gpu.py` |
| 2.4.2 | Implement fold accumulation (count contributing traces per output sample) | Same file |
| 2.4.3 | Implement stack normalization (divide by fold) | Same file |
| 2.4.4 | Memory chunking strategy for large output grids | Same file |
| 2.4.5 | Progress callback integration | Same file |

### 2.5 Multi-Gather Processing

| Task | Description | Files |
|------|-------------|-------|
| 2.5.1 | Implement gather loop with image accumulation | `processors/migration/kirchhoff_iso_gpu.py` |
| 2.5.2 | Batch gather processing (multiple gathers per GPU call) | Same file |
| 2.5.3 | Memory estimation function | Same file |
| 2.5.4 | CPU fallback implementation (NumPy/SciPy based) | `processors/migration/kirchhoff_iso_cpu.py` |

### 2.6 Integration & Testing

| Task | Description | Files |
|------|-------------|-------|
| 2.6.1 | Create `KirchhoffPSTMProcessor` (BaseProcessor wrapper) | `processors/kirchhoff_pstm.py` |
| 2.6.2 | Factory function: `get_kirchhoff_migrator(prefer_gpu=True)` | Same file |
| 2.6.3 | Integration test: point diffractor focusing | `tests/test_kirchhoff_mvp.py` |
| 2.6.4 | Integration test: flat reflector positioning | Same file |
| 2.6.5 | Integration test: dipping reflector | Same file |
| 2.6.6 | Performance benchmark: GPU vs CPU speedup | `tests/benchmarks/benchmark_kirchhoff.py` |

### 2.7 MVP Deliverables Checklist
- [ ] Migrate synthetic shot gather with constant velocity
- [ ] Output focused diffractor image
- [ ] GPU acceleration working (>10x speedup over CPU)
- [ ] Progress reporting functional
- [ ] Memory usage under control (chunked processing)
- [ ] Basic example script working

---

## Phase 3: Data Preparation & Binning System

**Goal:** Production-ready input data handling with offset/azimuth binning and proper output management

### 3.1 Offset-Azimuth Binning Definition

| Task | Description | Files |
|------|-------------|-------|
| 3.1.1 | Create `OffsetAzimuthBin` dataclass (offset_min, offset_max, azimuth_min, azimuth_max) | `models/binning.py` |
| 3.1.2 | Create `BinningTable` class to hold list of bins (the migration job definition) | Same file |
| 3.1.3 | Implement bin overlap validation (warn on gaps/overlaps in offset-azimuth space) | Same file |
| 3.1.4 | Add bin naming convention (e.g., "offset_0000-0200_az_000-360") | Same file |
| 3.1.5 | Implement common offset binning preset (uniform offset ranges, full azimuth) | `utils/binning_presets.py` |
| 3.1.6 | Implement OVT (Offset Vector Tile) binning preset | Same file |
| 3.1.7 | Implement narrow azimuth binning preset | Same file |
| 3.1.8 | BinningTable serialization to/from JSON/YAML | `models/binning.py` |

**Example BinningTable Structure:**
```
Line | Offset Min | Offset Max | Azimuth Min | Azimuth Max | Output Name
-----|------------|------------|-------------|-------------|-------------
  1  |     0      |    200     |      0      |     360     | near_offset
  2  |    200     |    400     |      0      |     360     | mid_offset_1
  3  |    400     |    700     |      0      |     360     | mid_offset_2
  4  |    700     |   1200     |      0      |     360     | far_offset
```

### 3.2 Data Indexing & Subsetting

| Task | Description | Files |
|------|-------------|-------|
| 3.2.1 | Create `DatasetIndexer` class to scan input SEG-Y and build trace index | `utils/dataset_indexer.py` |
| 3.2.2 | Index structure: trace_number → (offset, azimuth, inline, xline, file_position) | Same file |
| 3.2.3 | Implement bin assignment: assign each trace to one or more bins | Same file |
| 3.2.4 | Create `BinnedDataset` class holding index + bin assignments | `models/binned_dataset.py` |
| 3.2.5 | Implement efficient trace retrieval by bin (sorted file access) | Same file |
| 3.2.6 | Memory-efficient indexing for large datasets (streaming scan) | `utils/dataset_indexer.py` |
| 3.2.7 | Index persistence (save/load index to avoid re-scanning) | Same file |
| 3.2.8 | Index validation (check for missing coordinates, invalid values) | Same file |

### 3.3 Input Data Sorting Support

| Task | Description | Files |
|------|-------------|-------|
| 3.3.1 | Detect input data sort order (common offset, common shot, common receiver, OVT) | `utils/sort_detector.py` |
| 3.3.2 | Implement common offset gather reader | `io/gather_readers.py` |
| 3.3.3 | Implement common shot gather reader | Same file |
| 3.3.4 | Implement OVT (Offset Vector Tile) gather reader | Same file |
| 3.3.5 | Abstract `GatherIterator` interface for unified access | Same file |
| 3.3.6 | Streaming gather reader (don't load entire dataset to memory) | Same file |

### 3.4 Migration Job Configuration

| Task | Description | Files |
|------|-------------|-------|
| 3.4.1 | Create `MigrationJob` class combining: velocity + config + binning + header mapping | `models/migration_job.py` |
| 3.4.2 | Job validation (check all components compatible) | Same file |
| 3.4.3 | Job serialization (save complete job definition) | Same file |
| 3.4.4 | Job template system (save/load common configurations) | Same file |

### 3.5 Output Volume Management

| Task | Description | Files |
|------|-------------|-------|
| 3.5.1 | Create `MigrationOutputManager` for handling multiple output volumes | `io/migration_output.py` |
| 3.5.2 | One output volume per bin (e.g., migrated_near_offset.sgy, migrated_far_offset.sgy) | Same file |
| 3.5.3 | Output header population: preserve INLINE, XLINE from output grid | Same file |
| 3.5.4 | Output header: write OFFSET (bin center), AZIMUTH (bin center) to output traces | Same file |
| 3.5.5 | Output header: write CDP_X, CDP_Y from output grid coordinates | Same file |
| 3.5.6 | Full stack output option (sum all bins into single volume) | Same file |
| 3.5.7 | Output fold volume (traces contributing per output sample per bin) | Same file |

### 3.6 Execution Orchestration

| Task | Description | Files |
|------|-------------|-------|
| 3.6.1 | Create `MigrationOrchestrator` to manage bin-by-bin execution | `processors/migration/orchestrator.py` |
| 3.6.2 | Sequential bin processing mode (one bin at a time, lower memory) | Same file |
| 3.6.3 | Parallel bin processing mode (multiple bins simultaneously if memory allows) | Same file |
| 3.6.4 | Progress reporting per bin and overall | Same file |
| 3.6.5 | Checkpoint/resume at bin level (restart from last completed bin) | Same file |
| 3.6.6 | Resource estimation (memory/time per bin based on trace count) | Same file |

### 3.7 Header Mapping UI Components

| Task | Description | Files |
|------|-------------|-------|
| 3.7.1 | Create `HeaderMappingWidget` (PyQt6) - table showing required vs available headers | `views/header_mapping_widget.py` |
| 3.7.2 | Dropdown selectors for mapping input headers to required fields | Same file |
| 3.7.3 | Auto-detect common header names (guess mapping from standard names) | `utils/header_autodetect.py` |
| 3.7.4 | Computed header indicators (show which headers will be calculated) | `views/header_mapping_widget.py` |
| 3.7.5 | Validation feedback (green check / red X for each required header) | Same file |

### 3.8 Binning UI Components

| Task | Description | Files |
|------|-------------|-------|
| 3.8.1 | Create `BinningTableWidget` (PyQt6) - editable table for bin definitions | `views/binning_table_widget.py` |
| 3.8.2 | Add/remove/edit bin rows | Same file |
| 3.8.3 | Offset-Azimuth coverage visualization (2D plot showing bin boundaries) | `views/binning_coverage_plot.py` |
| 3.8.4 | Bin preset selector dropdown | `views/binning_table_widget.py` |
| 3.8.5 | Import binning from CSV/text file | Same file |
| 3.8.6 | Data coverage overlay (show actual data distribution vs bin boundaries) | `views/binning_coverage_plot.py` |

### 3.9 Testing

| Task | Description | Files |
|------|-------------|-------|
| 3.9.1 | Unit test: bin assignment correctness | `tests/test_binning.py` |
| 3.9.2 | Unit test: dataset indexing | `tests/test_dataset_indexer.py` |
| 3.9.3 | Integration test: common offset migration workflow | `tests/test_common_offset_workflow.py` |
| 3.9.4 | Integration test: output header correctness | `tests/test_output_headers.py` |
| 3.9.5 | Performance test: large dataset indexing | `tests/benchmarks/benchmark_indexing.py` |

### 3.10 Deliverables Checklist
- [ ] Offset-azimuth binning system functional
- [ ] Header mapping UI working
- [ ] Dataset indexing for large files
- [ ] Multiple output volumes generated correctly
- [ ] Output headers properly populated
- [ ] Sequential and parallel bin processing
- [ ] Checkpoint/resume working
- [ ] Common offset workflow end-to-end tested

---

## Phase 4: 1D Velocity Model (V of Z/T)

**Goal:** Support depth/time-varying velocity for realistic imaging

### 4.1 Velocity Model Extensions

| Task | Description | Files |
|------|-------------|-------|
| 4.1.1 | Extend `VelocityModel` for 1D v(z) or v(t) arrays | `models/velocity_model.py` |
| 4.1.2 | Implement velocity interpolation at arbitrary depths | Same file |
| 4.1.3 | RMS-to-interval velocity conversion utility | `utils/velocity_utils.py` |
| 4.1.4 | Interval-to-RMS velocity conversion utility | Same file |
| 4.1.5 | Velocity model I/O (simple text format) | `utils/velocity_io.py` |

### 4.2 Straight-Ray with V(z)

| Task | Description | Files |
|------|-------------|-------|
| 4.2.1 | Modify traveltime kernel to use depth-varying velocity | `processors/migration/traveltime_straight.py` |
| 4.2.2 | Implement effective velocity computation for straight ray | Same file |
| 4.2.3 | GPU tensor for velocity profile lookup | Same file |
| 4.2.4 | Unit test: v(z) traveltime vs ray tracing reference | `tests/test_traveltime_vz.py` |

### 4.3 Amplitude Weights (Basic)

| Task | Description | Files |
|------|-------------|-------|
| 4.3.1 | Implement geometrical spreading weight: `1 / (r_s * r_r)` | `processors/migration/weights.py` |
| 4.3.2 | Implement obliquity factor: `cos(theta)` | Same file |
| 4.3.3 | Combined weight computation kernel | Same file |
| 4.3.4 | Unit test: weight symmetry and limits | `tests/test_weights.py` |

### 4.4 Testing

| Task | Description | Files |
|------|-------------|-------|
| 4.4.1 | Test with linear velocity gradient v(z) = v0 + kz | `tests/test_kirchhoff_vz.py` |
| 4.4.2 | Compare imaging with/without velocity gradient | Same file |
| 4.4.3 | Verify amplitude preservation | Same file |

### 4.5 Deliverables Checklist
- [ ] Migration with 1D velocity function
- [ ] Geometrical spreading correction
- [ ] Improved amplitude handling
- [ ] Velocity model file I/O

---

## Phase 5: Curved Ray Traveltimes

**Goal:** Accurate traveltimes for strong velocity gradients

### 5.1 Curved Ray Theory Implementation

| Task | Description | Files |
|------|-------------|-------|
| 5.1.1 | Implement analytic curved ray for constant gradient: `v(z) = v0 + kz` | `processors/migration/traveltime_curved.py` |
| 5.1.2 | Formula: `t = (1/k) * arccosh(1 + k*r^2/(2*v0*z))` | Same file |
| 5.1.3 | Handle edge cases (k approaches 0 straight ray limit, z approaches 0 surface) | Same file |
| 5.1.4 | GPU kernel for curved ray traveltime batch computation | Same file |
| 5.1.5 | Unit test: curved vs straight ray comparison | `tests/test_traveltime_curved.py` |
| 5.1.6 | Unit test: k approaches 0 limit matches straight ray | Same file |

### 5.2 Ray Parameter & Emergence Angle

| Task | Description | Files |
|------|-------------|-------|
| 5.2.1 | Compute ray parameter p from curved ray geometry | `processors/migration/traveltime_curved.py` |
| 5.2.2 | Compute emergence angle at surface | Same file |
| 5.2.3 | Store angles for obliquity weight computation | Same file |

### 5.3 Improved Amplitude Weights

| Task | Description | Files |
|------|-------------|-------|
| 5.3.1 | Update geometrical spreading for curved rays | `processors/migration/weights.py` |
| 5.3.2 | Implement obliquity using emergence angles | Same file |
| 5.3.3 | Optional: Jacobian for true-amplitude migration | Same file |

### 5.4 Switchable Traveltime Mode

| Task | Description | Files |
|------|-------------|-------|
| 5.4.1 | Add `traveltime_mode` parameter to config: 'straight' or 'curved' | `models/migration_config.py` |
| 5.4.2 | Factory pattern to select appropriate calculator | `processors/migration/traveltime.py` |
| 5.4.3 | Default to curved when velocity gradient available | Same file |

### 5.5 Testing

| Task | Description | Files |
|------|-------------|-------|
| 5.5.1 | Compare curved ray migration vs straight ray | `tests/test_kirchhoff_curved.py` |
| 5.5.2 | Test with strong gradient (>0.5 m/s per meter) | Same file |
| 5.5.3 | Verify improved focusing at depth | Same file |
| 5.5.4 | Benchmark: curved ray computational overhead | `tests/benchmarks/benchmark_curved_ray.py` |

### 5.6 Deliverables Checklist
- [ ] Curved ray traveltimes for constant gradient
- [ ] Smooth transition straight to curved at k approaches 0
- [ ] Emergence angle computation
- [ ] Improved deep imaging accuracy

---

## Phase 6: VTI Anisotropy

**Goal:** Support Vertical Transversely Isotropic media (Thomsen parameters)

### 6.1 Anisotropy Model

| Task | Description | Files |
|------|-------------|-------|
| 6.1.1 | Create `AnisotropyModel` dataclass (epsilon, delta, eta) | `models/anisotropy_model.py` |
| 6.1.2 | Support scalar, 1D(z), 2D(x,z), 3D(x,y,z) parameter fields | Same file |
| 6.1.3 | Auto-compute eta from epsilon and delta: `eta = (epsilon-delta)/(1+2*delta)` | Same file |
| 6.1.4 | Anisotropy parameter I/O utilities | `utils/anisotropy_io.py` |
| 6.1.5 | Validation: physical bounds on Thomsen parameters | `models/anisotropy_model.py` |

### 6.2 VTI Traveltime Corrections

| Task | Description | Files |
|------|-------------|-------|
| 6.2.1 | Implement VTI phase velocity: `V(theta) = V0 * sqrt(1 + 2*delta*sin^2(theta)*cos^2(theta) + 2*epsilon*sin^4(theta))` | `processors/migration/traveltime_vti.py` |
| 6.2.2 | Implement weak anisotropy approximation (Alkhalifah) | Same file |
| 6.2.3 | Combine with curved ray traveltimes | Same file |
| 6.2.4 | GPU kernel for VTI traveltime computation | Same file |
| 6.2.5 | Unit test: isotropic limit (epsilon=delta=0) | `tests/test_traveltime_vti.py` |
| 6.2.6 | Unit test: VTI moveout curves | Same file |

### 6.3 Anelliptic Approximation

| Task | Description | Files |
|------|-------------|-------|
| 6.3.1 | Implement Fomel's anelliptic approximation using eta | `processors/migration/traveltime_vti.py` |
| 6.3.2 | Compare accuracy: exact vs anelliptic | `tests/test_anelliptic.py` |

### 6.4 Integration

| Task | Description | Files |
|------|-------------|-------|
| 6.4.1 | Add `anisotropy` parameter to `KirchhoffPSTMGPU` | `processors/kirchhoff_pstm.py` |
| 6.4.2 | Update config with anisotropy enable/disable flag | `models/migration_config.py` |
| 6.4.3 | Graceful fallback to isotropic when anisotropy=None | `processors/kirchhoff_pstm.py` |

### 6.5 Testing

| Task | Description | Files |
|------|-------------|-------|
| 6.5.1 | Test with typical shale anisotropy (epsilon=0.2, delta=0.1) | `tests/test_kirchhoff_vti.py` |
| 6.5.2 | Compare isotropic vs VTI migration on same data | Same file |
| 6.5.3 | Verify hockey-stick elimination in CIGs | Same file |

### 6.6 Deliverables Checklist
- [ ] VTI anisotropy support (Thomsen epsilon, delta)
- [ ] Accurate non-hyperbolic moveout handling
- [ ] Improved imaging in anisotropic formations
- [ ] Anisotropy parameter file I/O

---

## Phase 7: Production Features

**Goal:** Antialiasing, optimization, full UI integration, and production hardening

### 7.1 Antialiasing

| Task | Description | Files |
|------|-------------|-------|
| 7.1.1 | Implement local dip estimation from neighboring traces | `processors/migration/antialias.py` |
| 7.1.2 | Compute alias frequency: `f_alias = v_apparent / (2*dx)` | Same file |
| 7.1.3 | Implement triangle filter (linear interpolation AA) | Same file |
| 7.1.4 | Implement operator aliasing protection | Same file |
| 7.1.5 | Add `antialias_enabled` flag to config | `models/migration_config.py` |
| 7.1.6 | Unit test: verify no aliasing artifacts | `tests/test_antialias.py` |

### 7.2 2D/3D Velocity Models

| Task | Description | Files |
|------|-------------|-------|
| 7.2.1 | Extend `VelocityModel` for 2D v(x,z) grids | `models/velocity_model.py` |
| 7.2.2 | Extend for 3D v(x,y,z) cubes | Same file |
| 7.2.3 | Implement trilinear interpolation for velocity lookup | Same file |
| 7.2.4 | GPU texture memory for velocity cube (if beneficial) | `processors/migration/velocity_texture.py` |
| 7.2.5 | SEG-Y velocity cube I/O | `utils/velocity_io.py` |

### 7.3 Performance Optimization

| Task | Description | Files |
|------|-------------|-------|
| 7.3.1 | Profile GPU kernel occupancy and optimize | Various |
| 7.3.2 | Implement traveltime table caching (vs on-the-fly) | `processors/migration/traveltime_cache.py` |
| 7.3.3 | Optimize memory access patterns (coalescing) | Various |
| 7.3.4 | Multi-GPU support (optional, data parallelism) | `processors/migration/multi_gpu.py` |
| 7.3.5 | Mixed precision (FP16) option for memory savings | Various |

### 7.4 Output Options

| Task | Description | Files |
|------|-------------|-------|
| 7.4.1 | Common Image Gather (CIG) output option | `processors/kirchhoff_pstm.py` |
| 7.4.2 | Offset-domain CIG output | Same file |
| 7.4.3 | Angle-domain CIG output (subsurface angle) | Same file |
| 7.4.4 | Illumination/fold map output | Same file |

### 7.5 Full UI Integration

| Task | Description | Files |
|------|-------------|-------|
| 7.5.1 | Create `MigrationWizard` - step-by-step workflow dialog | `views/migration_wizard.py` |
| 7.5.2 | Step 1: Input data selection & header mapping | Same file |
| 7.5.3 | Step 2: Binning table definition | Same file |
| 7.5.4 | Step 3: Velocity model selection | Same file |
| 7.5.5 | Step 4: Migration parameters (aperture, angles, output grid) | Same file |
| 7.5.6 | Step 5: Output configuration | Same file |
| 7.5.7 | Step 6: Review & execute | Same file |
| 7.5.8 | Create `MigrationConfigPanel` (PyQt6 widget) for quick access | `views/migration_config_panel.py` |
| 7.5.9 | Velocity model viewer/editor widget | `views/velocity_viewer.py` |
| 7.5.10 | Anisotropy parameter editor | `views/anisotropy_panel.py` |
| 7.5.11 | Migration progress dialog with per-bin progress and cancel support | `views/migration_progress.py` |
| 7.5.12 | Result viewer (inline/crossline/time slice) | `views/migration_result_viewer.py` |
| 7.5.13 | Integration with main window menu | `main_window.py` |

### 7.6 Production Hardening

| Task | Description | Files |
|------|-------------|-------|
| 7.6.1 | Comprehensive input validation | Various |
| 7.6.2 | Graceful error handling and recovery | Various |
| 7.6.3 | Logging throughout (DEBUG, INFO, WARNING, ERROR) | Various |
| 7.6.4 | Memory overflow protection | Various |
| 7.6.5 | Enhanced checkpoint/resume for long migrations | `processors/migration/checkpoint.py` |
| 7.6.6 | Output SEG-Y writing with proper headers | `utils/migration_export.py` |
| 7.6.7 | Job queue for batch processing multiple datasets | `processors/migration/job_queue.py` |

### 7.7 Documentation & Examples

| Task | Description | Files |
|------|-------------|-------|
| 7.7.1 | API documentation (docstrings complete) | Various |
| 7.7.2 | User guide: running migration | `docs/user_guide_migration.md` |
| 7.7.3 | User guide: header mapping | `docs/user_guide_header_mapping.md` |
| 7.7.4 | User guide: binning strategies | `docs/user_guide_binning.md` |
| 7.7.5 | Example: synthetic data migration | `examples/migration_synthetic.py` |
| 7.7.6 | Example: common offset migration workflow | `examples/migration_common_offset.py` |
| 7.7.7 | Example: OVT migration workflow | `examples/migration_ovt.py` |
| 7.7.8 | Example: field data migration workflow | `examples/migration_field_data.py` |
| 7.7.9 | Parameter tuning guide | `docs/migration_parameters.md` |

### 7.8 Final Deliverables Checklist
- [ ] Antialiasing functional
- [ ] 2D/3D velocity model support
- [ ] Full UI wizard integration
- [ ] CIG output options
- [ ] Production-ready error handling
- [ ] Complete documentation
- [ ] All example workflows tested

---

## Dependency Graph

```
Phase 1 (Foundation + Header Mapping)
    |
    v
Phase 2 (MVP - Isotropic Straight Ray) <-- USABLE FOR TESTING
    |
    v
Phase 3 (Data Prep & Binning) <-- PRODUCTION-READY DATA HANDLING
    |
    +------------------+------------------+
    |                  |                  |
    v                  v                  |
Phase 4 (V of Z)    Phase 5 (Curved Ray) |
    |                  |                  |
    +--------+---------+                  |
             |                            |
             v                            |
      Phase 6 (VTI Anisotropy)            |
             |                            |
             +----------------------------+
             |
             v
      Phase 7 (Production Features + Full UI)
```

**Key Milestones:**
- **Phase 2**: First working migration (synthetic/test data)
- **Phase 3**: Production data handling (real surveys)
- **Phase 7**: Complete product

---

## Data Flow Diagram

```
                    +------------------+
                    |   Input SEG-Y    |
                    | (sorted by offset|
                    |  or OVT or shot) |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Header Mapping   |
                    | (user defines    |
                    |  SX,SY,GX,GY,    |
                    |  OFFSET, etc.)   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Dataset Indexer  |
                    | (scan headers,   |
                    |  compute OFFSET, |
                    |  AZIMUTH if      |
                    |  needed)         |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Binning Table    |
                    | Line 1: 0-200m   |
                    | Line 2: 200-400m |
                    | Line 3: 400-700m |
                    | ...              |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Orchestrator     |
                    | (process bins    |
                    |  sequential or   |
                    |  parallel)       |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
   +-----------+      +-----------+      +-----------+
   | Bin 1     |      | Bin 2     |      | Bin 3     |
   | Migration |      | Migration |      | Migration |
   +-----------+      +-----------+      +-----------+
          |                  |                  |
          v                  v                  v
   +-----------+      +-----------+      +-----------+
   | Output 1  |      | Output 2  |      | Output 3  |
   | near.sgy  |      | mid.sgy   |      | far.sgy   |
   | OFFSET=100|      | OFFSET=300|      | OFFSET=550|
   | headers   |      | headers   |      | headers   |
   +-----------+      +-----------+      +-----------+
          |                  |                  |
          +------------------+------------------+
                             |
                             v (optional)
                    +------------------+
                    | Full Stack       |
                    | (sum all bins)   |
                    +------------------+
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU memory exhaustion | Chunked processing, memory estimation pre-flight check |
| Numerical instability | Careful handling of edge cases (z approaching 0, k approaching 0, theta approaching 90 degrees) |
| Performance bottleneck | Profile early, optimize critical kernels |
| Accuracy issues | Extensive testing against analytic solutions |
| MPS (Apple Silicon) limitations | CPU fallback, thorough MPS testing |
| Large dataset indexing slow | Streaming index build, index persistence |
| Header mapping errors | Validation feedback, auto-detection, preview before run |
| Bin definition errors | Coverage visualization, overlap warnings |

---

## Success Criteria

### MVP (Phase 2)
- [ ] Process 1000-trace gather in <10 seconds (GPU)
- [ ] Point diffractor focuses to <2 samples spread
- [ ] Flat reflector positioned within 1 sample accuracy
- [ ] Memory usage predictable and bounded

### Production Data Handling (Phase 3)
- [ ] Index 1M+ trace dataset in <5 minutes
- [ ] Header mapping UI intuitive and validated
- [ ] Binning visualization clear and accurate
- [ ] Output headers match industry standards
- [ ] Checkpoint/resume working across bins

### Full Implementation (Phase 7)
- [ ] VTI anisotropy improves CIG flatness by >50%
- [ ] Curved ray improves deep imaging (>3 s) focusing
- [ ] Antialiasing eliminates migration artifacts
- [ ] Production workflow handles 10 GB+ datasets
- [ ] UI wizard complete and intuitive
- [ ] All documentation complete

---

## Appendix A: Required Headers Reference

### Input Headers (User Must Map)

| Header | Description | Required | Can Compute |
|--------|-------------|----------|-------------|
| SX | Source X coordinate | Yes | No |
| SY | Source Y coordinate | Yes | No |
| GX | Receiver (Group) X coordinate | Yes | No |
| GY | Receiver (Group) Y coordinate | Yes | No |
| OFFSET | Source-receiver offset (m) | Preferred | Yes (from SX,SY,GX,GY) |
| AZIMUTH | Source-receiver azimuth (deg) | Preferred | Yes (from SX,SY,GX,GY) |
| INLINE | Inline number | Optional* | No |
| XLINE | Crossline number | Optional* | No |
| CDP_X | CDP X coordinate | Optional | Yes (midpoint) |
| CDP_Y | CDP Y coordinate | Optional | Yes (midpoint) |

*INLINE/XLINE required if data is sorted by these.

### Output Headers (Written by PSTM)

| Header | Description | Value |
|--------|-------------|-------|
| INLINE | Output inline number | From output grid |
| XLINE | Output crossline number | From output grid |
| CDP_X | Output CDP X coordinate | From output grid geometry |
| CDP_Y | Output CDP Y coordinate | From output grid geometry |
| OFFSET | Bin center offset | From binning table |
| AZIMUTH | Bin center azimuth | From binning table |
| FOLD | Number of contributing traces | Computed during migration |

---

## Appendix B: Key Formulas Reference

### Straight Ray Traveltime (Isotropic)
```
t = sqrt(x^2 + y^2 + z^2) / V
```

### Curved Ray Traveltime (Constant Gradient)
```
v(z) = v0 + k*z

t = (1/k) * arccosh(1 + k*(x^2 + z^2)/(2*v0*z))

Limit as k approaches 0: t approaches sqrt(x^2 + z^2) / v0
```

### VTI Phase Velocity (Thomsen)
```
V(theta) = V0 * sqrt(1 + 2*delta*sin^2(theta)*cos^2(theta) + 2*epsilon*sin^4(theta))

where theta = angle from vertical
```

### Anelliptic Parameter
```
eta = (epsilon - delta) / (1 + 2*delta)
```

### Geometrical Spreading Weight
```
W = cos(theta_s) * cos(theta_r) / (r_s * r_r * V)
```

### Alias Frequency
```
f_alias = V_apparent / (2 * dx)

V_apparent = dx / dt (from local dip)
```

### Offset and Azimuth Computation
```
OFFSET = sqrt((GX - SX)^2 + (GY - SY)^2)

AZIMUTH = atan2(GY - SY, GX - SX) * 180 / pi
         (converted to 0-360 range)
```

### CDP (Midpoint) Computation
```
CDP_X = (SX + GX) / 2
CDP_Y = (SY + GY) / 2
```

---

## Appendix C: Binning Presets

### Common Offset Binning (Example)
```yaml
name: "Common Offset - 10 bins"
description: "Standard offset binning for land 3D"
bins:
  - offset_min: 0,    offset_max: 200,  azimuth_min: 0, azimuth_max: 360, name: "near"
  - offset_min: 200,  offset_max: 400,  azimuth_min: 0, azimuth_max: 360, name: "off_200"
  - offset_min: 400,  offset_max: 600,  azimuth_min: 0, azimuth_max: 360, name: "off_400"
  - offset_min: 600,  offset_max: 800,  azimuth_min: 0, azimuth_max: 360, name: "off_600"
  - offset_min: 800,  offset_max: 1000, azimuth_min: 0, azimuth_max: 360, name: "off_800"
  - offset_min: 1000, offset_max: 1500, azimuth_min: 0, azimuth_max: 360, name: "off_1000"
  - offset_min: 1500, offset_max: 2000, azimuth_min: 0, azimuth_max: 360, name: "off_1500"
  - offset_min: 2000, offset_max: 2500, azimuth_min: 0, azimuth_max: 360, name: "off_2000"
  - offset_min: 2500, offset_max: 3000, azimuth_min: 0, azimuth_max: 360, name: "off_2500"
  - offset_min: 3000, offset_max: 5000, azimuth_min: 0, azimuth_max: 360, name: "far"
```

### OVT Binning (Example)
```yaml
name: "OVT - 4x4 tiles"
description: "Offset Vector Tile binning for wide-azimuth"
bins:
  # Near offsets, 4 azimuth sectors
  - offset_min: 0,   offset_max: 1000, azimuth_min: 0,   azimuth_max: 90,  name: "ovt_near_az0"
  - offset_min: 0,   offset_max: 1000, azimuth_min: 90,  azimuth_max: 180, name: "ovt_near_az90"
  - offset_min: 0,   offset_max: 1000, azimuth_min: 180, azimuth_max: 270, name: "ovt_near_az180"
  - offset_min: 0,   offset_max: 1000, azimuth_min: 270, azimuth_max: 360, name: "ovt_near_az270"
  # Mid offsets, 4 azimuth sectors
  - offset_min: 1000, offset_max: 2000, azimuth_min: 0,   azimuth_max: 90,  name: "ovt_mid_az0"
  # ... (continue pattern)
```

### Narrow Azimuth Binning (Example)
```yaml
name: "Narrow Azimuth - Inline/Crossline"
description: "Separate inline and crossline azimuths"
bins:
  - offset_min: 0, offset_max: 5000, azimuth_min: 80,  azimuth_max: 100, name: "inline_pos"
  - offset_min: 0, offset_max: 5000, azimuth_min: 260, azimuth_max: 280, name: "inline_neg"
  - offset_min: 0, offset_max: 5000, azimuth_min: 350, azimuth_max: 10,  name: "xline_pos"
  - offset_min: 0, offset_max: 5000, azimuth_min: 170, azimuth_max: 190, name: "xline_neg"
```

---

*End of Document*
