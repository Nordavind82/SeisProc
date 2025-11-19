# FK Filter Enhancements Plan

## Overview
Enhance FK filter with sub-gather boundaries, navigation, and AGC pre-conditioning for improved filtering on complex gathers.

## 1. Sub-Gather Boundary Detection

### Purpose
Allow users to split large gathers into smaller sub-gathers based on header changes. Example: Common shot gather split by receiver line changes.

### Implementation Strategy

#### 1.1 Header-Based Boundary Detection
```
Input:
- Gather data with headers
- User-selected header key (e.g., "ReceiverLineNumber", "GroupX", etc.)

Process:
1. Scan selected header column for value changes
2. Detect transition points where header value changes
3. Create sub-gather boundaries at transition points
4. Store sub-gather metadata (start_trace, end_trace, header_value)

Output:
- List of SubGather objects with boundaries and identifying information
```

#### 1.2 SubGather Data Structure
```python
@dataclass
class SubGather:
    """Represents a sub-gather within a larger gather."""
    sub_id: int              # 0-based index within parent gather
    start_trace: int         # Start trace index (absolute)
    end_trace: int           # End trace index (absolute, inclusive)
    n_traces: int            # Number of traces
    boundary_header: str     # Header used for boundary (e.g., "ReceiverLine")
    boundary_value: Any      # Value of boundary header for this sub-gather
    description: str         # Human-readable description
```

#### 1.3 Boundary Detection Algorithm
```
Algorithm: detect_sub_gathers(headers_df, boundary_header)

1. Extract boundary column values
2. Find indices where value changes: np.where(np.diff(values) != 0)
3. Create boundary list: [0] + change_indices + [n_traces]
4. For each boundary pair:
   - Create SubGather with start/end traces
   - Store boundary value
   - Generate description
5. Return list of SubGather objects

Edge Cases:
- Single value (no changes) → Return single sub-gather = full gather
- Empty header → Raise error with suggestion
- Non-unique changes → Group by unique values with warning
```

### 1.4 UI for Boundary Selection (Design Mode Only)

**Location**: FK Designer Dialog

**Controls**:
```
┌─────────────────────────────────────────────┐
│ Sub-Gather Boundaries (Optional)            │
├─────────────────────────────────────────────┤
│ ☐ Split gather by header changes           │
│                                             │
│   Boundary Header: [Dropdown ▼]            │
│   Available: ReceiverLine, GroupX,          │
│              SourceLine, Offset, etc.       │
│                                             │
│   Detected: 4 sub-gathers                   │
│   Current: 1/4 (ReceiverLine=101)          │
│                                             │
│   [◄ Prev] [Next ►] [View All]             │
└─────────────────────────────────────────────┘
```

**Workflow**:
1. User checks "Split gather by header changes"
2. Selects boundary header from dropdown (populated from available headers)
3. System detects boundaries and updates count
4. User navigates between sub-gathers to design filter
5. FK spectrum and preview update for current sub-gather
6. Configuration saved with boundary settings

## 2. Sub-Gather Navigation in Design Mode

### 2.1 Navigation Controls

**UI Elements**:
- Previous/Next buttons for sequential navigation
- Sub-gather index display (e.g., "2/4")
- Sub-gather description label
- "View All" button to see all sub-gathers stacked

**Behavior**:
- Navigation updates:
  - FK spectrum (computed for current sub-gather only)
  - Input/Filtered/Rejected previews (show current sub-gather)
  - Quality metrics (for current sub-gather)
  - Trace count and spacing info
- Filter parameters apply to current sub-gather
- When saved, configuration stores:
  - Boundary header name
  - Whether sub-gather mode is enabled
  - Filter parameters (same for all sub-gathers)

### 2.2 Data Management

**Memory Optimization**:
```
Strategy: Lazy sub-gather extraction

1. Store full gather data once
2. Create views/slices for each sub-gather (no copying)
3. Compute FK spectrum only for current sub-gather
4. Cache FK spectra as user navigates (LRU cache, max 3)
5. Preview plots use array slicing (gather.traces[:, start:end])
```

**Trace Spacing Handling**:
```
Issue: Sub-gathers may have different trace spacing
Solution:
1. Calculate trace spacing per sub-gather
2. Display warning if spacing varies significantly
3. Use median spacing for FK filter
4. Allow manual override
```

## 3. Apply Mode with Sub-Gathers

### 3.1 Processing Strategy

**Principle**: Process each sub-gather independently, reassemble full gather

**Algorithm**:
```
apply_fk_with_subgathers(full_gather, config):

1. If config has sub-gather boundaries:
   a. Detect sub-gathers using config.boundary_header
   b. For each sub-gather:
      - Extract traces (view, no copy)
      - Calculate trace spacing
      - Create FK processor with config params
      - Process sub-gather → filtered_sub
      - Store in output array at original position
   c. Reassemble full filtered gather

2. Else:
   - Process full gather as before

3. Return full filtered gather

Performance:
- Use in-place operations where possible
- Vectorize sub-gather processing if all same size
- Parallelize sub-gather processing (optional, if >4 sub-gathers)
```

### 3.2 Output Display

**Viewer Behavior**:
```
Input Viewer:    Full gather (all traces)
Filtered Viewer: Full gather (all sub-gathers processed)
Rejected Viewer: Full gather (difference)

Visual Indicators (Optional):
- Overlay vertical lines at sub-gather boundaries
- Color-code sub-gathers in trace headers
- Display boundary info in hover tooltip
```

### 3.3 Configuration Storage

**FKFilterConfig Extension**:
```python
@dataclass
class FKFilterConfig:
    # ... existing fields ...

    # Sub-gather settings
    use_subgathers: bool = False
    boundary_header: Optional[str] = None  # e.g., "ReceiverLine"

    # AGC settings (see next section)
    apply_agc: bool = False
    agc_window_ms: float = 500.0
```

## 4. AGC (Automatic Gain Control) Pre-Conditioning

### 4.1 Purpose and Theory

**Why AGC for FK Filtering?**
```
Problem:
- Amplitude decay with time/offset affects FK spectrum
- Weak late arrivals invisible in FK domain
- Strong early arrivals dominate filter design

Solution:
- Apply AGC before FK to equalize amplitudes
- Design filter on amplitude-balanced data
- Remove AGC after filtering to restore original character

Process:
Input → AGC → FK Filter → Inverse AGC → Output
```

**AGC Algorithm (Sliding Window RMS)**:
```
For each trace:
  For each sample at time t:
    window = samples[t - w/2 : t + w/2]  # w = window length
    rms = sqrt(mean(window²))
    scale[t] = target_rms / (rms + epsilon)
    agc_trace[t] = trace[t] * scale[t]
```

### 4.2 Fast Vectorized AGC Implementation

**Performance Requirements**:
- Process 1000 traces × 2000 samples in <100ms
- Memory efficient (no large intermediate arrays)
- Fully vectorized (NumPy operations)

**Implementation Strategy**:
```python
def apply_agc_vectorized(traces, window_samples, target_rms=1.0, epsilon=1e-10):
    """
    Fast vectorized AGC using uniform filter for RMS calculation.

    Args:
        traces: 2D array (n_samples, n_traces)
        window_samples: Window length in samples (odd number)
        target_rms: Target RMS value
        epsilon: Small value to prevent division by zero

    Returns:
        agc_traces: AGC-applied traces (same shape)
        scale_factors: Scale factors for inversion (same shape)
    """

    # Strategy: Use scipy.ndimage.uniform_filter for sliding window

    1. Compute squared traces: traces²
    2. Apply uniform filter to get mean of squares in window
       mean_sq = uniform_filter(traces², size=(window_samples, 1), mode='reflect')
    3. Compute RMS: rms = sqrt(mean_sq)
    4. Compute scale factors: scale = target_rms / (rms + epsilon)
    5. Apply scaling: agc_traces = traces * scale
    6. Return agc_traces and scale (for inverse)

    Performance:
    - uniform_filter is C-optimized in scipy
    - Vectorized operations on full array
    - No Python loops
    - Memory: 2× input size (for squared and RMS arrays)
```

**Alternative: CuPy GPU Implementation** (if GPU available):
```python
def apply_agc_gpu(traces, window_samples, target_rms=1.0):
    """GPU-accelerated AGC using CuPy."""

    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter

    # Transfer to GPU
    traces_gpu = cp.asarray(traces)

    # Same algorithm as CPU but on GPU
    traces_sq = traces_gpu ** 2
    mean_sq = uniform_filter(traces_sq, size=(window_samples, 1))
    rms = cp.sqrt(mean_sq)
    scale = target_rms / (rms + epsilon)
    agc_traces = traces_gpu * scale

    # Transfer back
    return cp.asnumpy(agc_traces), cp.asnumpy(scale)
```

### 4.3 AGC Inverse (Restoration)

**Strategy**: Store scale factors, apply inverse

```python
def remove_agc(agc_traces, scale_factors):
    """
    Remove AGC by applying inverse scaling.

    Args:
        agc_traces: AGC-applied traces
        scale_factors: Scale factors from apply_agc

    Returns:
        original_traces: Approximately restored traces
    """
    return agc_traces / (scale_factors + epsilon)
```

**Note**: AGC is not perfectly reversible due to:
- Epsilon in denominator
- Numerical precision
- But close enough for FK filtering purposes (errors <1%)

### 4.4 AGC UI Controls (Design Mode)

**Location**: FK Designer Dialog, below filter parameters

```
┌─────────────────────────────────────────────┐
│ AGC Pre-Conditioning (Optional)             │
├─────────────────────────────────────────────┤
│ ☑ Apply AGC before FK filtering            │
│                                             │
│   AGC Window (ms): [500] ▲▼                │
│   Range: 50-2000 ms                         │
│                                             │
│   ℹ AGC equalizes amplitudes for better    │
│     FK filtering, then is removed from      │
│     output                                  │
│                                             │
│   Preview: ○ Without AGC  ● With AGC       │
└─────────────────────────────────────────────┘
```

**Workflow**:
1. User checks "Apply AGC before FK filtering"
2. Sets AGC window length (default 500ms)
3. Preview radio buttons:
   - "Without AGC": Show FK spectrum of raw data
   - "With AGC": Show FK spectrum of AGC-applied data
4. User designs filter based on preview choice
5. Configuration saved with AGC settings

### 4.5 AGC Processing Flow

**Design Mode**:
```
User Preview:
  Raw Gather → (Optional: AGC) → FK Spectrum Display
                                ↓
                         User adjusts filter
                                ↓
                    Filter applied to preview
                         (with or without AGC)
```

**Apply Mode**:
```
Full Processing Chain:

For each (sub-)gather:
  1. Input gather
  ↓
  2. [If apply_agc] Apply AGC → store scale factors
  ↓
  3. Apply FK filter
  ↓
  4. [If apply_agc] Remove AGC using scale factors
  ↓
  5. Output gather (original amplitude character preserved)
```

## 5. Performance Optimization

### 5.1 FK Filter Optimization

**Current Bottlenecks**:
- 2D FFT (scipy.fft.fft2)
- Filter weight calculation
- 2D inverse FFT

**Optimizations**:

#### Use FFT Wisdom (Pre-planning)
```python
import scipy.fft as fft

# One-time setup for common gather sizes
def setup_fft_wisdom(common_shapes):
    """Pre-plan FFT for common gather sizes."""
    for shape in common_shapes:
        dummy = np.random.randn(*shape)
        fft.fft2(dummy)  # Creates wisdom/plan
```

#### Use FFTW Backend (if available)
```python
# Install: pip install pyfftw
import pyfftw

# Use FFTW for 2D FFT (faster than scipy)
def fft2_optimized(data):
    return pyfftw.interfaces.scipy_fft.fft2(data)
```

#### Vectorize Filter Weight Calculation
```python
# Current: Multiple np.where calls
# Optimized: Single vectorized computation

def create_velocity_filter_optimized(f_grid, k_grid, v_min, v_max, taper, mode):
    """Fully vectorized filter weight computation."""

    # All computations in NumPy without conditionals
    v_app = np.abs(f_grid / (k_grid + 1e-10))

    # Define boundaries
    v1, v2, v3, v4 = (
        v_min - taper,
        v_min + taper,
        v_max - taper,
        v_max + taper
    )

    # Vectorized weight calculation using continuous functions
    # (Use smooth functions instead of hard thresholds where possible)
    weights = compute_taper_weights_vectorized(v_app, v1, v2, v3, v4, mode)

    return weights
```

### 5.2 Memory Optimization

**Strategy**: Minimize copies, use views

```python
# Bad: Creates copy
filtered_subgather = full_gather[:, start:end].copy()
fk_processor.process(filtered_subgather)

# Good: Use view, modify in-place
subgather_view = full_gather[:, start:end]
filtered_view = fk_processor.process(subgather_view)  # Returns new array
full_gather_filtered[:, start:end] = filtered_view
```

**Memory Budget for Large Gathers**:
```
Example: 1000 traces × 2000 samples × float32 (4 bytes)

Arrays needed:
- Input gather:           8 MB
- Filtered gather:        8 MB
- FK spectrum (complex):  16 MB
- AGC scales (if used):   8 MB
- Temporary arrays:       ~16 MB

Total:                    ~56 MB per gather

Optimization:
- Process sub-gathers sequentially to avoid holding all FK spectra
- Reuse FK spectrum buffer
- Clear intermediate arrays explicitly
```

### 5.3 Parallel Processing (Optional)

**When to parallelize**:
- Multiple sub-gathers (>4)
- Batch processing of multiple gathers
- Not critical for single gather

**Strategy**: Use ProcessPoolExecutor for sub-gathers
```python
from concurrent.futures import ProcessPoolExecutor

def process_subgather_parallel(subgather_list, processor_config):
    """Process multiple sub-gathers in parallel."""

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_single_subgather, sg, processor_config)
            for sg in subgather_list
        ]
        results = [f.result() for f in futures]

    return results
```

**Note**: Only parallelize if sub-gather processing >500ms each

## 6. Implementation Plan

### Phase 1: Sub-Gather Detection and Navigation (Design Mode)

**Files to Modify**:
1. `models/fk_config.py`
   - Add sub-gather fields to FKFilterConfig
   - Add SubGather dataclass

2. `views/fk_designer_dialog.py`
   - Add boundary header selection UI
   - Add sub-gather navigation controls
   - Implement detect_sub_gathers() method
   - Update FK spectrum computation for current sub-gather
   - Update preview plots for current sub-gather

3. `processors/fk_filter.py`
   - No changes (works on whatever gather passed to it)

**Testing**:
- Load gather with multiple receiver lines
- Select ReceiverLine as boundary header
- Verify sub-gather detection
- Navigate between sub-gathers
- Check FK spectrum updates correctly
- Save configuration with boundary settings

### Phase 2: AGC Implementation

**Files to Create**:
1. `processors/agc.py`
   - `apply_agc_vectorized()` function
   - `remove_agc()` function
   - `AGCProcessor` class (optional, for standalone use)

**Files to Modify**:
2. `models/fk_config.py`
   - Add AGC fields to FKFilterConfig

3. `views/fk_designer_dialog.py`
   - Add AGC controls UI
   - Add AGC preview toggle
   - Apply AGC before FK spectrum computation (if enabled)
   - Apply AGC before filter preview (if enabled)

4. `processors/fk_filter.py`
   - Add optional AGC integration
   - Or keep AGC separate and apply in MainWindow

**Testing**:
- Enable AGC in designer
- Verify FK spectrum changes with AGC
- Test filter with/without AGC
- Verify inverse AGC restores amplitudes
- Measure performance on large gathers

### Phase 3: Apply Mode with Sub-Gathers

**Files to Modify**:
1. `main_window.py`
   - Update `_on_fk_config_selected()` to handle sub-gathers
   - Implement sub-gather detection in apply mode
   - Process each sub-gather independently
   - Reassemble full gather
   - Apply AGC chain if configured

2. `processors/fk_filter.py`
   - Add method to process gather with sub-gather boundaries
   - `process_with_subgathers(gather, boundary_header, config)`

**Testing**:
- Design filter with sub-gathers
- Apply to same gather (verify correct)
- Apply to different gather with same structure
- Verify full gather output
- Test with AGC enabled
- Test batch processing with auto-process

### Phase 4: Performance Optimization

**Actions**:
1. Profile FK filter performance
   - Identify bottlenecks (likely FFT)
   - Implement FFTW if beneficial
   - Optimize filter weight calculation

2. Profile AGC performance
   - Verify vectorized implementation is fast
   - Test on large gathers (2000×1000)
   - Optimize if needed

3. Memory profiling
   - Check memory usage on large gathers
   - Implement memory-efficient sub-gather processing
   - Add warnings for very large gathers

**Benchmarks to Achieve**:
```
Target Performance:

AGC:
- 1000 traces × 2000 samples: < 100ms (CPU)
- 1000 traces × 2000 samples: < 20ms (GPU)

FK Filter:
- 1000 traces × 2000 samples: < 500ms (CPU)
- 1000 traces × 2000 samples: < 100ms (GPU, future)

Full Chain (AGC + FK + Inverse AGC):
- 1000 traces × 2000 samples: < 800ms (CPU)

Memory:
- Peak usage < 100MB for 1000×2000 gather
```

## 7. UI/UX Improvements

### 7.1 FK Designer Dialog Layout Update

**New Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│ FK Filter Designer - Gather 5                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ┌─ Left Panel ───────────┐  ┌─ Right Panel ──────────────┐ │
│ │                         │  │                            │ │
│ │ [Preset Selection]      │  │ FK Spectrum (Log Amp)      │ │
│ │                         │  │ (with velocity lines)      │ │
│ │ ┌─ Sub-Gathers ───────┐ │  │                            │ │
│ │ │ ☐ Split by header   │ │  │                            │ │
│ │ │ Header: [RecLine ▼] │ │  └────────────────────────────┘ │
│ │ │ Sub: 2/4 (RL=102)   │ │                                 │
│ │ │ [◄ Prev] [Next ►]   │ │  ┌─ Preview ─────────────────┐ │
│ │ └─────────────────────┘ │  │ Input | Filtered | Reject │ │
│ │                         │  │                            │ │
│ │ ┌─ AGC ───────────────┐ │  │                            │ │
│ │ │ ☑ Apply AGC         │ │  │                            │ │
│ │ │ Window: [500] ms    │ │  └────────────────────────────┘ │
│ │ │ Preview: ● With AGC │ │                                 │
│ │ └─────────────────────┘ │  ┌─ Metrics ─────────────────┐ │
│ │                         │  │ Energy preserved: 65.2%   │ │
│ │ [Filter Parameters]     │  │ Energy rejected: 34.8%    │ │
│ │ - v_min, v_max          │  └────────────────────────────┘ │
│ │ - taper, mode           │                                 │
│ │                         │                                 │
│ └─────────────────────────┘                                 │
│                                                             │
│ Config Name: [Ground_Roll_RL102_____________]              │
│                                    [Save] [Close]           │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Information Labels

**Sub-Gather Info Display**:
```
When sub-gathers enabled:
"Working on: Gather 5, Sub-gather 2/4 (ReceiverLine=102)
 30 traces, 2000 samples, spacing: 25.0 m"

When sub-gathers disabled:
"Working on: Gather 5
 120 traces, 2000 samples, spacing: 25.0 m"
```

**AGC Info Tooltip**:
```
ℹ AGC (Automatic Gain Control):
  Equalizes trace amplitudes before FK filtering
  Helps reveal weak events in FK spectrum
  Removed after filtering to preserve original character

  Window: Sliding window length for RMS calculation
  Typical: 300-1000 ms
```

## 8. Configuration File Examples

### 8.1 Simple FK Filter (No Sub-Gathers, No AGC)
```json
{
  "name": "Ground_Roll_Removal_v1",
  "filter_type": "velocity_fan",
  "v_min": 1500.0,
  "v_max": 6000.0,
  "taper_width": 300.0,
  "mode": "pass",
  "created": "2025-01-18T10:30:00",
  "created_on_gather": 5,
  "description": "Created on gather 5",
  "author": "user",
  "use_subgathers": false,
  "boundary_header": null,
  "apply_agc": false,
  "agc_window_ms": 500.0
}
```

### 8.2 FK Filter with Sub-Gathers and AGC
```json
{
  "name": "Ground_Roll_PerReceiverLine_AGC",
  "filter_type": "velocity_fan",
  "v_min": 1800.0,
  "v_max": 5500.0,
  "taper_width": 250.0,
  "mode": "pass",
  "created": "2025-01-18T10:45:00",
  "created_on_gather": 12,
  "description": "Per receiver line with AGC, created on gather 12",
  "author": "user",
  "use_subgathers": true,
  "boundary_header": "ReceiverLineNumber",
  "apply_agc": true,
  "agc_window_ms": 600.0
}
```

## 9. Error Handling and Edge Cases

### 9.1 Sub-Gather Edge Cases

**Case 1: Boundary header not found**
```
Error: "Header 'ReceiverLine' not found in gather.
Available headers: FFID, SourceX, SourceY, GroupX, GroupY, Offset"

Solution: Show dropdown with only available headers
```

**Case 2: No boundaries detected**
```
Warning: "Header 'ReceiverLine' has constant value (101).
No sub-gathers detected. Processing as single gather."

Solution: Automatically disable sub-gather mode
```

**Case 3: Too many sub-gathers**
```
Warning: "Detected 47 sub-gathers based on 'Offset'.
This may not be meaningful for FK filtering.
Consider using a different boundary header."

Solution: Allow but warn; suggest alternatives
```

**Case 4: Sub-gathers too small**
```
Warning: "Sub-gather 3 has only 5 traces.
FK filtering requires at least 8 traces.
Consider different boundary or disable sub-gathers."

Solution: Skip sub-gathers <8 traces with warning
```

### 9.2 AGC Edge Cases

**Case 1: AGC window > trace length**
```
Warning: "AGC window (2000ms) exceeds trace length (1500ms).
Using trace length as window size."

Solution: Clamp window to trace length
```

**Case 2: Zero/near-zero RMS in window**
```
Issue: Division by epsilon causes huge gains

Solution:
- Clip maximum gain (e.g., 100x)
- Or use adaptive epsilon based on trace RMS
```

**Case 3: AGC on already-gained data**
```
Warning: "Data appears to have strong amplitude variations.
AGC may have already been applied. Applying AGC again
may distort results."

Solution: Detect via amplitude histogram, warn user
```

## 10. Testing Strategy

### 10.1 Unit Tests

**AGC Tests** (`tests/test_agc.py`):
```python
def test_agc_preserves_shape()
def test_agc_inverse_restores()
def test_agc_equalizes_amplitudes()
def test_agc_performance()  # <100ms for 1000×2000
def test_agc_edge_cases()   # window > trace, zero RMS, etc.
```

**Sub-Gather Tests** (`tests/test_subgather.py`):
```python
def test_detect_subgathers_single_boundary()
def test_detect_subgathers_multiple_boundaries()
def test_detect_subgathers_no_boundaries()
def test_subgather_trace_spacing()
def test_subgather_too_small()
```

**FK Filter Tests** (`tests/test_fk_with_enhancements.py`):
```python
def test_fk_with_subgathers()
def test_fk_with_agc()
def test_fk_with_both()
def test_fk_energy_conservation()
def test_fk_performance()  # <500ms for 1000×2000
```

### 10.2 Integration Tests

**Workflow Tests**:
1. Load synthetic gather with receiver line changes
2. Open FK Designer
3. Enable sub-gathers with ReceiverLine boundary
4. Navigate between sub-gathers
5. Enable AGC
6. Design filter
7. Save configuration
8. Apply to full gather
9. Verify output correctness
10. Measure performance

### 10.3 Performance Benchmarks

**Benchmark Suite**:
```python
# Create synthetic gathers of various sizes
sizes = [
    (500, 1000),   # Small
    (1000, 2000),  # Medium
    (2000, 4000),  # Large
]

for n_traces, n_samples in sizes:
    gather = create_synthetic_gather(n_traces, n_samples)

    # Benchmark AGC
    time_agc = benchmark_agc(gather)

    # Benchmark FK
    time_fk = benchmark_fk(gather)

    # Benchmark full chain
    time_full = benchmark_agc_fk_chain(gather)

    # Verify targets met
    assert time_agc < 100e-3    # 100ms
    assert time_fk < 500e-3     # 500ms
    assert time_full < 800e-3   # 800ms
```

## 11. Documentation Updates

### 11.1 User Documentation

**Topics to Add**:
1. "Using Sub-Gathers for Complex Acquisitions"
   - When to use sub-gathers
   - How to select boundary header
   - Navigating sub-gathers in designer

2. "AGC Pre-Conditioning for FK Filters"
   - What is AGC and why use it
   - Choosing AGC window length
   - When to enable/disable

3. "FK Filter Best Practices"
   - Sub-gather vs full gather filtering
   - AGC recommendations
   - Performance tips

### 11.2 Technical Documentation

**Add to FK_FILTER_DESIGN.md**:
- Sub-gather processing algorithm
- AGC theory and implementation
- Performance characteristics
- API reference for new functions

## 12. Future Enhancements (Out of Scope)

**Potential Future Work**:
1. GPU-accelerated FK filtering (CuPy/CUDA)
2. Adaptive FK filters (different params per sub-gather)
3. Real-time FK spectrum preview (during navigation)
4. 3D FK filtering (for 3D surveys)
5. Machine learning-based filter parameter suggestion
6. Dip-based FK filtering (not just velocity)
7. Custom AGC curves (not just RMS)

---

## Summary

This enhancement plan adds sophisticated sub-gather handling and AGC pre-conditioning to FK filtering while maintaining performance and usability. Key deliverables:

1. ✅ Sub-gather boundary detection based on header changes
2. ✅ Sub-gather navigation in Design mode
3. ✅ Sub-gather-aware Apply mode (process independently, show full gather)
4. ✅ Fast vectorized AGC implementation (<100ms for 1000×2000)
5. ✅ AGC integration with FK filtering (apply before, remove after)
6. ✅ Performance optimization for FK filter (<500ms for 1000×2000)
7. ✅ Comprehensive UI/UX for new features
8. ✅ Robust error handling and edge cases
9. ✅ Full test coverage

**Implementation Priority**:
1. Phase 1: Sub-gathers (most impactful for complex data)
2. Phase 2: AGC (enhances filter effectiveness)
3. Phase 3: Apply mode integration (completes workflow)
4. Phase 4: Performance optimization (polish)
