# 3D FKK Filter - Implementation Summary

## Philosophy

**Simple. Accurate. GPU-efficient. Integrated with existing workflow.**

- Uses loaded seismic data (not separate 3D volume loading)
- User selects headers to define inline/crossline axes
- Design/Apply workflow like 2D FK filter
- Show slices, not full volumes
- Velocity cone filtering done right
- Reuses existing patterns

---

## User Workflow

### Design Mode
1. User selects "3D FKK Filter" from Algorithm dropdown
2. Clicks "Open 3D FKK Designer..."
3. **Header Selection Dialog** appears:
   - Shows available headers from loaded data
   - User selects inline key (e.g., ReceiverLine, SourceLine, CDP)
   - User selects crossline key (e.g., ReceiverStation, Channel)
   - Preview shows estimated volume size and coverage
4. System builds 3D volume from traces using selected headers
5. **FKK Designer Dialog** opens with:
   - Input slices (time slice, inline)
   - Filtered output slices
   - FKK spectrum views (kx-ky, f-kx)
   - Velocity cone controls
6. User designs filter, clicks Accept
7. Filtered result is extracted back to 2D gather format for display

### Apply Mode
1. User selects preset or configures velocity/mode parameters
2. Clicks "Apply FKK Filter"
3. Uses previously built volume (from Design mode)
4. Filter applied, result displayed

---

## UI Layout - FKK Designer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  File   View   Process   Help                                      [3D FKK]    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─── INPUT DATA ───────────────────┐  ┌─── FKK SPECTRUM ──────────────────┐   │
│  │                                  │  │                                   │   │
│  │   ┌────────────────────────┐     │  │   ┌────────────────────────┐      │   │
│  │   │    Time Slice (X-Y)    │     │  │   │    Kx-Ky Slice @ f     │      │   │
│  │   │      @ t = 500ms       │     │  │   │     (shows cone)       │      │   │
│  │   └────────────────────────┘     │  │   └────────────────────────┘      │   │
│  │         [t: ◄ 500ms ►]           │  │         [f: ◄ 25 Hz ►]            │   │
│  │                                  │  │                                   │   │
│  │   ┌────────────────────────┐     │  │   ┌────────────────────────┐      │   │
│  │   │   Inline (T-X) @ Y=50  │     │  │   │    F-Kx Slice @ ky=0   │      │   │
│  │   └────────────────────────┘     │  │   └────────────────────────┘      │   │
│  │         [Y: ◄ 50 ►]              │  │         [ky: ◄ 0 ►]               │   │
│  └──────────────────────────────────┘  └───────────────────────────────────┘   │
│                                                                                 │
│  ┌─── FILTERED OUTPUT ──────────────┐  ┌─── FILTER CONTROLS ───────────────┐   │
│  │                                  │  │                                   │   │
│  │   ┌────────────────────────┐     │  │  Mode:  ○ Reject   ● Pass         │   │
│  │   │    Time Slice (X-Y)    │     │  │                                   │   │
│  │   └────────────────────────┘     │  │  V min: [====●====] 500 m/s       │   │
│  │                                  │  │  V max: [========●] 3000 m/s      │   │
│  │   ┌────────────────────────┐     │  │                                   │   │
│  │   │   Inline (T-X) @ Y=50  │     │  │  Azimuth: [0°] to [360°]          │   │
│  │   └────────────────────────┘     │  │  Taper:  [====●====] 0.1          │   │
│  │                                  │  │                                   │   │
│  │  View: ○ Filtered  ○ Difference  │  │  [Compute] [Apply] [Export]       │   │
│  └──────────────────────────────────┘  └───────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  GPU: 8.2/12.0 GB │ Volume: 256×256×512 │ Status: Ready                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implemented Components

### 1. Data Models

**`models/seismic_volume.py`**
- `SeismicVolume` dataclass with slice accessors
- `create_synthetic_volume()` for testing

**`models/fkk_config.py`**
- `FKKConfig` dataclass with velocity cone parameters
- `FKK_PRESETS` for common use cases

### 2. Volume Builder

**`utils/volume_builder.py`**
- `get_available_volume_headers()` - find suitable headers
- `estimate_volume_size()` - preview before building
- `build_volume_from_gathers()` - construct 3D from 2D traces
- `extract_traces_from_volume()` - convert back to 2D format

### 3. GPU Processor

**`processors/fkk_filter_gpu.py`**
- `FKKFilterGPU` with PyTorch backend
- `FKKFilterCPU` fallback with scipy
- `get_fkk_filter()` auto-selection

### 4. UI Components

**`views/volume_header_dialog.py`**
- Header selection for inline/crossline axes
- Volume size preview
- Coverage estimation

**`views/fkk_designer_dialog.py`**
- Slice-based visualization
- FKK spectrum views
- Interactive parameter controls
- Real-time preview with debouncing

### 5. Control Panel Integration

**`views/control_panel.py`**
- "3D FKK Filter" in algorithm dropdown
- Design/Apply mode selection
- Preset and parameter controls
- `fkk_design_requested` and `fkk_apply_requested` signals

### 6. Main Window Integration

**`main_window.py`**
- Signal handlers for FKK design/apply
- Volume building from loaded data
- Filtered result extraction to 2D view

---

## File Structure

```
models/
    seismic_volume.py      # SeismicVolume dataclass
    fkk_config.py          # FKKConfig dataclass

processors/
    fkk_filter_gpu.py      # FKKFilterGPU/CPU processors

views/
    control_panel.py       # Updated with 3D FKK controls
    volume_header_dialog.py # Header selection dialog
    fkk_designer_dialog.py # Main FKK designer UI

utils/
    volume_builder.py      # Volume building utilities
```

---

## Key Features

1. **No hardcoded headers** - User selects from available headers
2. **Design/Apply workflow** - Same pattern as 2D FK filter
3. **Uses loaded data** - No separate 3D volume loading
4. **GPU acceleration** - PyTorch backend with CPU fallback
5. **Memory efficient** - Slice-based visualization
6. **Integrated results** - Filtered data displayed in main viewer

---

## GPU Memory Formula

For volume of shape `(nt, nx, ny)`:

```
Memory ≈ nt × nx × ny × 4 bytes      (input, float32)
       + (nt/2+1) × nx × ny × 8      (spectrum, complex64)
       + (nt/2+1) × nx × ny × 4      (mask, float32)
       + nt × nx × ny × 4            (output, float32)

Total ≈ 2.5 × input size
```

---

## Velocity-Wavenumber Relationship

```
v = f / k_horizontal
k_horizontal = sqrt(kx² + ky²)

For velocity cone filter:
- Reject: Remove points where v_min ≤ v ≤ v_max
- Pass: Keep only points where v_min ≤ v ≤ v_max
```
