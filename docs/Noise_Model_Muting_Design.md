# Noise Model Muting Feature Design

## Executive Summary

This document describes the architecture and UI/UX design for a **Noise Model Muting** feature in SeisProc. The feature allows geophysicists to preserve signal in specific zones (e.g., shallow reflections, ground roll areas) by muting parts of the estimated noise model before subtraction from the original data.

### Core Concept

```
Traditional TF-Denoise:
    Output = Input - NoiseModel

With Mute Applied:
    MutedNoise = NoiseModel × (1 - MuteMask)  # Zero out protected zones
    Output = Input - MutedNoise               # Signal preserved where muted
```

---

## 1. Feature Overview

### 1.1 Problem Statement

Current TF-Denoise processing removes all detected incoherent energy, which may include:
- Weak but valid reflections that appear on only one or few traces
- Signal in ground roll zones that gets partially attenuated
- Near-surface reflections mixed with coherent noise

Geophysicists need a way to **protect specific time-offset zones** where they know signal exists, preventing the noise model from affecting those areas.

### 1.2 Solution Approach

Implement a **mute function applied to the noise model** (not the input data):

1. User defines mute geometry (top/bottom, velocity, taper)
2. A mute mask is computed for each gather based on offset
3. The noise model is multiplied by (1 - mask), zeroing protected areas
4. Clean output = Input - (NoiseModel × InverseMask)

This preserves the original data in muted zones while still denoising unmuted areas.

---

## 2. Architecture Design

### 2.1 New Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Architecture Overview                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐ │
│  │  MuteConfig     │────▶│  MuteMaskBuilder │────▶│ NoiseModel    │ │
│  │  (dataclass)    │     │  (utility)       │     │ Muter         │ │
│  └─────────────────┘     └──────────────────┘     └───────────────┘ │
│         │                                                 │          │
│         │                                                 ▼          │
│         │              ┌──────────────────────────────────────┐     │
│         │              │      TFDenoise Processor             │     │
│         └─────────────▶│  + mute_config parameter             │     │
│                        │  + _apply_noise_mute() method        │     │
│                        └──────────────────────────────────────┘     │
│                                        │                             │
│                                        ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    UI Components                             │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │    │
│  │  │ MuteDrawingTool │  │ MuteControlPanel│  │ MutePreview  │ │    │
│  │  │ (viewer overlay)│  │ (parameters)    │  │ (live mask)  │ │    │
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
utils/
├── mute/
│   ├── __init__.py
│   ├── config.py          # MuteConfig dataclass
│   ├── mask_builder.py    # MuteMaskBuilder class
│   └── noise_muter.py     # Apply mute to noise model

processors/
├── tf_denoise.py          # Modified: add mute_config parameter
└── tf_denoise_gpu.py      # Modified: add mute_config parameter

views/
├── mute_control_panel.py  # New: mute parameter controls
├── mute_drawing_tool.py   # New: interactive mute line drawing
└── seismic_viewer_pyqtgraph.py  # Modified: support overlay drawing

main_window.py             # Modified: integrate mute controls
```

### 2.3 Data Classes

```python
# utils/mute/config.py

from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum

class MuteType(Enum):
    TOP = "top"       # Mute above the curve (early times)
    BOTTOM = "bottom" # Mute below the curve (late times)

class MuteShape(Enum):
    LINEAR = "linear"         # t = t0 + offset/velocity
    HYPERBOLIC = "hyperbolic" # t = sqrt(t0² + (offset/velocity)²)

@dataclass
class MuteConfig:
    """Configuration for noise model muting."""
    enabled: bool = False

    # Mute geometry
    mute_type: MuteType = MuteType.TOP
    mute_shape: MuteShape = MuteShape.LINEAR

    # Velocity parameters (m/s or ft/s depending on project)
    velocity: float = 1500.0  # Apparent velocity for mute curve

    # Time intercept (ms) - time at zero offset
    t0_ms: float = 0.0

    # Taper parameters
    taper_length_ms: float = 50.0  # Taper zone in milliseconds
    taper_type: Literal["linear", "cosine", "hanning"] = "cosine"

    # Optional: manual picks (for non-parametric mute)
    manual_picks: Optional[list] = None  # [(offset, time_ms), ...]

    def to_dict(self) -> dict:
        """Serialize for multiprocessing transfer."""
        return {
            'enabled': self.enabled,
            'mute_type': self.mute_type.value,
            'mute_shape': self.mute_shape.value,
            'velocity': self.velocity,
            't0_ms': self.t0_ms,
            'taper_length_ms': self.taper_length_ms,
            'taper_type': self.taper_type,
            'manual_picks': self.manual_picks
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MuteConfig':
        """Deserialize from dict."""
        return cls(
            enabled=d['enabled'],
            mute_type=MuteType(d['mute_type']),
            mute_shape=MuteShape(d['mute_shape']),
            velocity=d['velocity'],
            t0_ms=d['t0_ms'],
            taper_length_ms=d['taper_length_ms'],
            taper_type=d['taper_type'],
            manual_picks=d.get('manual_picks')
        )
```

### 2.4 Mask Builder

```python
# utils/mute/mask_builder.py

class MuteMaskBuilder:
    """
    Builds mute masks for seismic gathers.

    The mask is a 2D array (n_samples × n_traces) with values 0-1:
    - 0 = fully muted (noise model zeroed, signal preserved)
    - 1 = fully active (noise model applied normally)
    - 0-1 = taper zone (gradual transition)
    """

    def __init__(self, config: MuteConfig, sample_rate_ms: float):
        self.config = config
        self.sample_rate_ms = sample_rate_ms

    def compute_mute_time(self, offset: float) -> float:
        """
        Compute mute time in ms for a given offset.

        For LINEAR:   t = t0 + |offset| / velocity
        For HYPERBOLIC: t = sqrt(t0² + (offset/velocity)²)
        """
        abs_offset = abs(offset)

        if self.config.mute_shape == MuteShape.LINEAR:
            return self.config.t0_ms + abs_offset / self.config.velocity * 1000
        else:  # HYPERBOLIC
            t0_sec = self.config.t0_ms / 1000
            return np.sqrt(t0_sec**2 + (abs_offset / self.config.velocity)**2) * 1000

    def build_mask(
        self,
        n_samples: int,
        n_traces: int,
        offsets: np.ndarray
    ) -> np.ndarray:
        """
        Build the mute mask for a gather.

        Args:
            n_samples: Number of time samples
            n_traces: Number of traces
            offsets: Array of offset values for each trace

        Returns:
            Mask array (n_samples × n_traces) with values 0-1
        """
        mask = np.ones((n_samples, n_traces), dtype=np.float32)
        time_axis = np.arange(n_samples) * self.sample_rate_ms

        for i, offset in enumerate(offsets):
            mute_time = self.compute_mute_time(offset)

            # Build taper for this trace
            trace_mask = self._build_trace_mask(
                time_axis, mute_time, self.config.mute_type
            )
            mask[:, i] = trace_mask

        return mask

    def _build_trace_mask(
        self,
        time_axis: np.ndarray,
        mute_time: float,
        mute_type: MuteType
    ) -> np.ndarray:
        """Build mask for a single trace with taper."""
        taper_len = self.config.taper_length_ms

        if mute_type == MuteType.TOP:
            # Mute above the curve (early times protected)
            # mask = 0 for t < mute_time, 1 for t > mute_time + taper
            taper_start = mute_time
            taper_end = mute_time + taper_len

            mask = np.ones_like(time_axis)
            mask[time_axis < taper_start] = 0.0

            in_taper = (time_axis >= taper_start) & (time_axis < taper_end)
            if np.any(in_taper):
                t_norm = (time_axis[in_taper] - taper_start) / taper_len
                mask[in_taper] = self._apply_taper(t_norm)

        else:  # BOTTOM
            # Mute below the curve (late times protected)
            taper_start = mute_time - taper_len
            taper_end = mute_time

            mask = np.ones_like(time_axis)
            mask[time_axis > taper_end] = 0.0

            in_taper = (time_axis > taper_start) & (time_axis <= taper_end)
            if np.any(in_taper):
                t_norm = (taper_end - time_axis[in_taper]) / taper_len
                mask[in_taper] = self._apply_taper(t_norm)

        return mask

    def _apply_taper(self, t_norm: np.ndarray) -> np.ndarray:
        """Apply taper function. t_norm goes from 0 to 1."""
        if self.config.taper_type == "linear":
            return t_norm
        elif self.config.taper_type == "cosine":
            return 0.5 * (1 - np.cos(np.pi * t_norm))
        else:  # hanning
            return 0.5 * (1 - np.cos(np.pi * t_norm))
```

---

## 3. UI/UX Design

### 3.1 Mute Control Panel

Located in the main control panel, collapsible group box:

```
┌─────────────────────────────────────────────────────────────┐
│ ▼ Noise Model Mute                                      [?] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [✓] Enable Mute Drawing                                    │
│      (Shows mute line overlay on viewers)                   │
│                                                             │
│  ─────────────────────────────────────────                  │
│  Mute Type:        ◉ Top (protect early)                    │
│                    ○ Bottom (protect late)                  │
│                                                             │
│  ─────────────────────────────────────────                  │
│  Mute Shape:       ◉ Linear                                 │
│                    ○ Hyperbolic                             │
│                                                             │
│  ─────────────────────────────────────────                  │
│  Parameters:                                                │
│                                                             │
│  T0 (ms):          [    0.0    ] ▲▼                        │
│                    Time at zero offset                      │
│                                                             │
│  Velocity (m/s):   [  1500.0   ] ▲▼                        │
│                    Apparent velocity                        │
│                                                             │
│  ─────────────────────────────────────────                  │
│  Taper:                                                     │
│                                                             │
│  Length (ms):      [   50.0    ] ▲▼                        │
│                                                             │
│  Type:             [  Cosine   ▼]                          │
│                    Linear | Cosine | Hanning                │
│                                                             │
│  ─────────────────────────────────────────                  │
│  [ Preview Mute ]  [ Clear ]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Visual Mute Line Drawing

When "Enable Mute Drawing" is checked:

1. **Mute line overlay** appears on all three viewers (Input, Processed, Difference)
2. **Color coding**:
   - Muted zone (signal preserved): Semi-transparent **green overlay**
   - Active zone (noise removed): No overlay
   - Taper zone: Gradient from green to transparent
3. **Interactive adjustment**:
   - Drag T0 marker (at zero offset) to adjust intercept time
   - Drag velocity handle (at far offset) to adjust slope/curvature
   - Real-time preview updates as parameters change

```
┌────────────────────────────────────────────────────────────────────┐
│                    Seismic Viewer with Mute Overlay                 │
├────────────────────────────────────────────────────────────────────┤
│     Trace Number                                                    │
│     0    50   100   150   200   250   300   350   400              │
│  0  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│ 100 ░░░░░░░░░░████████████████████████████████████████████████████  │
│ 200 ░░░░░░░████████████████████████████████████████████████████████  │
│ 300 ░░░░████████████████████████████████████████████████████████████  │
│ 400 ░░██████████████████████████████████████████████████████████████  │
│     ▲                                                               │
│     │  ░░ = Muted zone (green semi-transparent)                    │
│ T   │  ██ = Active zone (normal display)                           │
│ i   │                                                               │
│ m   │     ╭── Mute line (white/yellow dashed)                      │
│ e   │     │                                                         │
│     │     ▼                                                         │
│ 500 ████████████████████████████████████████████████████████████████  │
│ 600 ████████████████████████████████████████████████████████████████  │
│                                                                      │
│  ○ T0 drag handle    ─────── Mute curve ───────  ○ Velocity handle  │
└────────────────────────────────────────────────────────────────────┘

Legend:
  ░ = Protected zone (noise model will be zeroed here)
  █ = Normal processing zone (noise model applied)
  ─ = Mute line (draggable)
  ○ = Interactive handles
```

### 3.3 Batch Processing Integration

When submitting "Parallel Batch Process...", add option in dialog:

```
┌─────────────────────────────────────────────────────────────────────┐
│              Parallel Batch Processing Options                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Dataset: 22,847 gathers, 12,301,564 traces                          │
│  Processor: TF-Denoise (aperture=7, k=3.0)                           │
│  Workers: 14 CPU cores                                               │
│  Estimated time: ~15 minutes                                         │
│                                                                       │
│  ─────────────────────────────────────────────────────────────────   │
│                                                                       │
│  ▼ In-Gather Sorting                                                 │
│    [✓] Sort traces within each gather                                │
│    Sort by: [ offset        ▼]  Direction: [Ascending ▼]            │
│                                                                       │
│  ─────────────────────────────────────────────────────────────────   │
│                                                                       │
│  ▼ Output Mode                                                       │
│                                                                       │
│    ◉ Standard (Input - NoiseModel)                                   │
│        Normal TF-Denoise output                                      │
│                                                                       │
│    ○ With Noise Model Mute Applied                                   │
│        Preserves signal in muted zones                               │
│        ┌───────────────────────────────────────────────────────┐    │
│        │  Current Mute Settings:                                │    │
│        │    Type: Top (protect early times)                     │    │
│        │    Shape: Linear                                       │    │
│        │    T0: 50.0 ms, Velocity: 1800 m/s                    │    │
│        │    Taper: 40 ms (cosine)                               │    │
│        │                                                        │    │
│        │  [ Edit Mute Settings... ]                             │    │
│        └───────────────────────────────────────────────────────┘    │
│                                                                       │
│    ○ Export Noise Model Only                                         │
│        Outputs only the estimated noise (for QC)                     │
│                                                                       │
│    ○ Export Both (Input and Muted Noise Model)                       │
│        Two outputs for detailed analysis                             │
│                                                                       │
│  ─────────────────────────────────────────────────────────────────   │
│                                                                       │
│                              [ Cancel ]  [ Process ]                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 Workflow for Geophysicists

#### Typical Workflow:

1. **Load data** and navigate to a representative gather
2. **Apply TF-Denoise** to see the default result
3. **Examine difference panel** - identify areas where signal was incorrectly removed
4. **Enable mute drawing** - checkbox in Mute control panel
5. **Adjust mute parameters**:
   - Set T0 at the earliest time you want to protect
   - Adjust velocity to match the mute curve to your target
   - Choose appropriate taper length (typically 30-100 ms)
6. **Preview on current gather** - click "Preview Mute"
7. **Navigate to other gathers** to verify mute works across dataset
8. **Run Parallel Batch Processing** with "With Noise Model Mute Applied"

#### Quick Parameter Guide:

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| T0 | 0-500 ms | Time intercept at zero offset |
| Velocity | 1000-5000 m/s | Ground roll: 300-800 m/s; Reflections: 1500-4000 m/s |
| Taper Length | 20-100 ms | Shorter = sharper transition; Longer = smoother |

---

## 4. Processing Integration

### 4.1 TFDenoise Processor Modifications

```python
# processors/tf_denoise.py - Modified

class TFDenoise(BaseProcessor):
    def __init__(
        self,
        aperture: int = 7,
        fmin: float = 1.0,
        fmax: float = 100.0,
        threshold_k: float = 3.0,
        threshold_type: str = 'hard',
        threshold_mode: str = 'adaptive',
        transform_type: str = 'stransform',
        low_amp_protection: bool = True,
        mute_config: Optional[MuteConfig] = None,  # NEW
        **kwargs
    ):
        self.mute_config = mute_config
        # ... existing init ...

    def process(self, data: SeismicData) -> SeismicData:
        # ... existing processing to get denoised output ...

        # Apply noise model mute if configured
        if self.mute_config and self.mute_config.enabled:
            output = self._apply_noise_mute(data, output)

        return output

    def _apply_noise_mute(
        self,
        input_data: SeismicData,
        denoised_data: SeismicData
    ) -> SeismicData:
        """
        Apply mute to noise model and recompute output.

        1. Compute noise model: noise = input - denoised
        2. Build mute mask from offsets
        3. Apply mask: muted_noise = noise * mask
        4. Recompute output: output = input - muted_noise
        """
        from utils.mute import MuteMaskBuilder

        # Get offsets from headers
        if data.headers and 'offset' in data.headers:
            offsets = data.headers['offset']
        else:
            # Fallback: assume uniform offset spacing
            offsets = np.arange(data.n_traces) * 100  # Arbitrary

        # Build mask
        builder = MuteMaskBuilder(self.mute_config, data.sample_rate)
        mask = builder.build_mask(data.n_samples, data.n_traces, offsets)

        # Compute noise model
        noise = input_data.traces - denoised_data.traces

        # Apply mute (where mask=0, noise is zeroed, preserving input)
        muted_noise = noise * mask

        # Recompute output
        output_traces = input_data.traces - muted_noise

        return SeismicData(
            traces=output_traces,
            sample_rate=denoised_data.sample_rate,
            headers=denoised_data.headers,
            metadata={
                **denoised_data.metadata,
                'mute_applied': True,
                'mute_config': self.mute_config.to_dict()
            }
        )

    def to_dict(self) -> dict:
        """Serialize for multiprocessing."""
        d = super().to_dict()
        if self.mute_config:
            d['params']['mute_config'] = self.mute_config.to_dict()
        return d
```

### 4.2 Parallel Processing Integration

The mute configuration flows through the existing pipeline:

```
ProcessingConfig
    └── processor_config (dict)
            └── mute_config (dict)
                    └── MuteConfig (reconstructed in worker)
```

Each worker:
1. Reconstructs MuteConfig from dict
2. Reconstructs TFDenoise with mute_config
3. Processes gathers - mute is applied per-gather using that gather's offsets

### 4.3 Output Modes

Implement as enum in ProcessingConfig:

```python
class OutputMode(Enum):
    STANDARD = "standard"              # Input - NoiseModel
    MUTED = "muted"                    # Input - MutedNoiseModel
    NOISE_ONLY = "noise_only"          # NoiseModel
    BOTH = "both"                      # Two outputs
```

---

## 5. Implementation Phases

### Phase 1: Core Mute Infrastructure (Priority: High)
- [ ] Create `utils/mute/` module with config, mask_builder, noise_muter
- [ ] Add mute_config parameter to TFDenoise processor
- [ ] Implement `_apply_noise_mute()` method
- [ ] Add serialization support for mute_config

### Phase 2: UI Controls (Priority: High)
- [ ] Create MuteControlPanel widget
- [ ] Integrate into main control panel as collapsible group
- [ ] Connect signals for parameter changes
- [ ] Add "Preview Mute" functionality for single gather

### Phase 3: Visual Overlay (Priority: Medium)
- [ ] Implement mute line drawing on PyQtGraph viewer
- [ ] Add interactive handles for T0 and velocity
- [ ] Show muted zone overlay (semi-transparent)
- [ ] Synchronize across all three viewers

### Phase 4: Batch Processing Integration (Priority: High)
- [ ] Add output mode selector to batch processing dialog
- [ ] Pass mute config through parallel processing pipeline
- [ ] Implement noise-only export option
- [ ] Add mute status to processing metadata

### Phase 5: Advanced Features (Priority: Low)
- [ ] Manual picks mode (click to define mute line)
- [ ] Multi-zone muting (multiple mute lines)
- [ ] Import/export mute definitions
- [ ] Mute templates library

---

## 6. Testing Strategy

### Unit Tests
- MuteMaskBuilder: verify mask shapes for linear/hyperbolic
- Taper functions: verify smooth transitions
- Serialization: round-trip MuteConfig to/from dict

### Integration Tests
- TFDenoise with mute: verify output preserves input in muted zones
- Parallel processing: verify mute applied correctly across workers
- Offset handling: verify correct behavior with various offset distributions

### Visual QC
- Side-by-side comparison: with mute vs without
- Difference display: should show zeros in muted zones
- Varying offsets: verify mute curve tracks correctly

---

## 7. Performance Considerations

### Memory Impact
- Mask array: n_samples × n_traces × 4 bytes (float32)
- For 1600 samples × 500 traces: ~3.2 MB per gather
- Negligible compared to trace data

### Computational Cost
- Mask building: O(n_samples × n_traces) - very fast
- Additional subtraction: O(n_samples × n_traces) - negligible
- Overall impact: < 1% processing time increase

### Parallelization
- Mask building is per-gather, naturally parallel
- No shared state between workers
- No synchronization needed

---

## 8. Error Handling

### Offset Availability
```python
if 'offset' not in headers:
    if strict_mode:
        raise ValueError("Mute requires offset header")
    else:
        # Fallback: use trace number as pseudo-offset
        warnings.warn("Offset header not found, using trace index")
        offsets = np.arange(n_traces) * estimated_dx
```

### Parameter Validation
- Velocity must be > 0
- T0 must be >= 0
- Taper length must be > 0
- Mute time must not exceed trace length

### Edge Cases
- Zero-offset traces: apply T0 directly
- Negative offsets: use absolute value
- Variable gather sizes: handle dynamically per-gather

---

## 9. Future Enhancements

### Multi-Zone Muting
Allow multiple mute definitions (e.g., top mute for first breaks + bottom mute for multiples):
```python
mute_configs: List[MuteConfig] = [
    MuteConfig(type=TOP, velocity=1800, t0=50),
    MuteConfig(type=BOTTOM, velocity=2500, t0=2000)
]
```

### Adaptive Muting
Automatically detect zones where signal is being removed:
- Analyze residual coherence
- Suggest mute parameters based on data statistics

### Mute QC Tools
- Amplitude statistics in muted vs active zones
- Residual energy comparison
- Before/after signal-to-noise metrics

---

## 10. Summary

This design provides a comprehensive solution for noise model muting that:

1. **Preserves signal** in user-defined zones
2. **Integrates seamlessly** with existing TF-Denoise workflow
3. **Supports batch processing** at scale
4. **Provides intuitive UI** for geophysicists
5. **Maintains performance** with minimal overhead

The feature fills a critical gap in the current processing workflow, giving geophysicists control over where denoising is applied while maintaining the power of the S-Transform based approach.
