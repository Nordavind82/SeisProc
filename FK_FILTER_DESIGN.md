# FK Filter Design Document
## Frequency-Wavenumber Domain Filtering for Seismic QC

**Status:** Design Phase (No Implementation)
**Purpose:** Filter seismic data based on apparent velocity (dip) and frequency
**Application:** Remove coherent linear noise (ground roll, air wave, multiples)

---

## Workflow Overview

FK filtering uses a **two-mode workflow** integrated into the main control panel:

```
┌──────────────────────────────────────────────────────────────┐
│                  FK FILTER WORKFLOW                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   1. DESIGN MODE                  2. APPLY MODE             │
│   ┌─────────────────┐             ┌────────────────┐        │
│   │                 │             │                │        │
│   │  Design filter  │────────────→│  Browse with   │        │
│   │  on one gather  │   Save      │  saved filter  │        │
│   │                 │   config    │  on all        │        │
│   │  • Interactive  │             │  gathers       │        │
│   │  • FK spectrum  │             │                │        │
│   │  • Preview      │             │  • Auto-apply  │        │
│   │  • Tune params  │             │  • QC all data │        │
│   │  • Save config  │             │  • Re-edit if  │        │
│   │                 │             │    needed      │        │
│   └─────────────────┘             └────────────────┘        │
│           │                              │                  │
│           │                              │                  │
│           └──────── [Edit] ──────────────┘                  │
│               Loop back to refine                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- ✅ **Separate design from application** - tune once, apply everywhere
- ✅ **Save multiple configurations** - different filters for different purposes
- ✅ **Seamless browsing** - filter applied automatically when navigating
- ✅ **Consistent with other algorithms** - same Design/Apply pattern
- ✅ **Easy iteration** - edit saved configs anytime

---

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Filter Types](#filter-types)
3. [Implementation Algorithm](#implementation-algorithm)
4. [UI/UX Design](#uiux-design)
5. [Parameter Guidance](#parameter-guidance)
6. [Performance Considerations](#performance-considerations)

---

## Mathematical Foundation

### 1. FK Domain Transform

The FK transform converts seismic data from time-space (t-x) domain to frequency-wavenumber (f-k) domain.

#### Forward Transform (t-x → f-k)

For a 2D seismic gather with N traces and M time samples:

```
Input: D(t, x)  where t = time, x = spatial position (trace offset/distance)

FK Transform:
F(f, k) = ∬ D(t, x) · e^(-2πi(ft + kx)) dt dx

Discrete form (2D FFT):
F[f_n, k_m] = Σ_t Σ_x D[t, x] · e^(-2πi(f_n·t + k_m·x))
```

Where:
- **f** = temporal frequency (Hz)
- **k** = spatial wavenumber (cycles/meter or cycles/trace)
- **F(f, k)** = Complex FK spectrum

#### Inverse Transform (f-k → t-x)

```
D(t, x) = ∬ F(f, k) · e^(2πi(ft + kx)) df dk

Discrete form (2D IFFT):
D[t, x] = Σ_f Σ_k F[f, k] · e^(2πi(f·t + k·x))
```

### 2. Relationship to Apparent Velocity

The FK domain has a direct relationship to **apparent velocity** (moveout):

```
Apparent velocity: v_app = f / k

Or equivalently:
k = f / v_app
```

**Physical meaning:**
- Events with constant apparent velocity appear as **straight lines** through origin in FK domain
- Line slope = 1/v_app
- Steeper slope = higher velocity
- Shallower slope = lower velocity

**Example:**
- Ground roll: v_app = 300-800 m/s (slow, shallow slope)
- Reflections: v_app = 2000-4000 m/s (fast, steep slope)
- Air wave: v_app = 330 m/s (very slow, very shallow slope)

### 3. FK Spectrum Properties

#### Quadrant Symmetry

For **real-valued** seismic data, the FK spectrum has **conjugate symmetry**:

```
F(-f, -k) = F*(f, k)  where * = complex conjugate
```

This means:
- Only need to filter positive frequencies
- Negative frequencies automatically handled
- Reduces computation by ~2x

#### Nyquist Limits

```
Frequency Nyquist:   f_nyq = 1 / (2·Δt)   where Δt = sample interval
Wavenumber Nyquist:  k_nyq = 1 / (2·Δx)   where Δx = trace spacing

Spatial aliasing occurs when:
k > k_nyq  or  v_app < f / k_nyq
```

**Aliasing example:**
- Sample interval Δt = 2 ms → f_nyq = 250 Hz
- Trace spacing Δx = 25 m → k_nyq = 0.02 cycles/m
- At f = 100 Hz: v_app_min = 100/0.02 = 5000 m/s
- Events slower than 5000 m/s are **spatially aliased**

---

## Filter Types

### 1. Velocity Fan Filter (Pie Slice)

**Purpose:** Remove events with specific apparent velocities

**Geometry:** Wedge-shaped regions in FK domain

```
Filter Definition:

For each point (f, k) in FK spectrum:
  v_app = |f / k|

  IF v_min ≤ v_app ≤ v_max:
      Keep (pass)
  ELSE:
      Reject (zero)
```

**Taper:** Use cosine taper to avoid Gibbs ringing:
```
Taper width: v_taper (m/s)

IF v_app < v_min - v_taper:
    Weight = 0  (full reject)
ELIF v_app < v_min + v_taper:
    Weight = 0.5 * (1 - cos(π * (v_app - v_min + v_taper) / (2*v_taper)))
ELIF v_app < v_max - v_taper:
    Weight = 1  (full pass)
ELIF v_app < v_max + v_taper:
    Weight = 0.5 * (1 + cos(π * (v_app - v_max + v_taper) / (2*v_taper)))
ELSE:
    Weight = 0  (full reject)

Apply:
F_filtered(f, k) = F(f, k) · Weight(f, k)
```

**Use cases:**
- **Pass:** 2000-6000 m/s (keep reflections)
- **Reject:** 200-800 m/s (remove ground roll)

### 2. Dip Filter (Slope Filter)

**Purpose:** Filter based on event dip (slope in t-x domain)

**Relationship to FK:**
```
Dip (ms/trace) = dt/dx = Δt/Δx · (k/f)

Or: k = f · (dip / v_app)
```

**Filter Definition:**
```
For dip range [dip_min, dip_max]:
  Pass events with slopes in this range
  Reject all others

Equivalent to velocity filter with:
  v = f / (k · dip)
```

### 3. Polygonal FK Filter

**Purpose:** Maximum flexibility - define arbitrary reject zones

**Geometry:** User-defined polygons in FK domain

```
Define polygon vertices: [(f1,k1), (f2,k2), ..., (fn,kn)]

For each FK point:
  IF point inside polygon:
      Weight = 0 (reject)
  ELSE:
      Weight = 1 (pass)

Apply taper around polygon boundaries
```

**Use cases:**
- Complex noise patterns
- Multiple velocity components
- Frequency-dependent velocity filtering

### 4. Top Mute (Low Velocity Reject)

**Purpose:** Remove all events slower than threshold velocity

**Geometry:** Reject region below v_app line

```
v_threshold = 1500 m/s

For each (f, k):
  v_app = f / k

  IF v_app < v_threshold:
      Reject
  ELSE:
      Pass

Equivalent to: |k| > |f| / v_threshold
```

**Use cases:**
- Remove ground roll (v < 1000 m/s)
- Remove air wave (v ≈ 330 m/s)
- Remove direct wave

---

## Implementation Algorithm

### Step-by-Step Process

```
INPUT:
  - SeismicData: D(t, x)  [n_samples × n_traces]
  - Filter parameters: v_min, v_max, taper_width, etc.

STEP 1: Prepare Data
  - Extract gather from input
  - Check trace spacing (constant Δx required)
  - Pad to power-of-2 for FFT efficiency (optional)

STEP 2: Forward 2D FFT
  F(f, k) = FFT2D(D(t, x))

  Note: Use numpy.fft.fft2() or scipy.fft.fft2()

STEP 3: Create Filter Weights
  For each point (f_i, k_j) in spectrum:
    - Calculate v_app = |f_i / k_j|
    - Determine weight W(f_i, k_j) based on filter type
    - Apply taper for smooth transitions

STEP 4: Apply Filter
  F_filtered(f, k) = F(f, k) · W(f, k)

STEP 5: Inverse 2D FFT
  D_filtered(t, x) = IFFT2D(F_filtered(f, k))

STEP 6: Extract Real Part
  D_out = Real(D_filtered)

  Note: Imaginary part should be ~0 for real input

OUTPUT:
  - Processed gather
  - Rejected gather = Input - Processed
```

### Trace Spacing Handling

**Critical:** FK transform requires **uniform trace spacing**

```
IF traces have irregular spacing:
  OPTION 1: Interpolate to regular grid
    - Use linear/cubic interpolation
    - Resample to constant Δx

  OPTION 2: Bin traces
    - Group traces into spatial bins
    - Use median/mean within each bin

  OPTION 3: Skip FK filtering
    - Warn user about irregular spacing
    - Suggest alternative filters
```

### Edge Handling

**Issue:** FFT assumes periodic boundary conditions

**Solutions:**

1. **Taper edges:**
   ```
   Apply cosine taper to first and last 10% of traces
   Reduces edge artifacts
   ```

2. **Pad traces:**
   ```
   Add zeros or reflection-padded traces at edges
   Pad to 2× length, filter, then trim
   ```

3. **Accept artifacts:**
   ```
   Edge traces may have artifacts
   Document in user guidance
   ```

---

## UI/UX Design

### Two-Mode Workflow

FK filtering uses a **Design → Apply** workflow similar to other processors, but with an interactive design phase:

```
Main Control Panel (Algorithm Selection)
    ↓
Select: "FK Filter"
    ↓
Mode Selection: ○ Design  ○ Apply
```

---

### Mode 1: Design Mode

**Purpose:** Interactively design and save FK filter parameters on current gather

**Entry:** User selects "FK Filter" + "Design" mode in control panel

**Single Button:** `[Open FK Filter Designer...]`

When clicked, opens the **FK Filter Design Window**:

```
┌─────────────────────────────────────────────────────────────────┐
│              FK Filter Designer (Design Mode)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Working on: Gather 42 (CDP 1250, 48 traces)                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Presets ▼]  [Velocity Fan]  [Dip Filter]  [Polygon]   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                Interactive FK Spectrum                    │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Frequency (Hz)                                     │  │  │
│  │  │  250├─────────────────────────────────────────────┤  │  │
│  │  │     │  ╱│╲              ╱│╲                        │  │  │
│  │  │  200│ ╱ │ ╲            ╱ │ ╲    [PASS ZONE]       │  │  │
│  │  │     │╱  │  ╲═════════╱  │  ╲                      │  │  │
│  │  │  150│   │ ░░╲░░░░░░░   │ ░░╲    ← Drag lines     │  │  │
│  │  │     │   │░░░░╲░░REJECT░│░░░░╲                     │  │  │
│  │  │  100│   │░░░░░╲░░░░░░ │░░░░░░╲                    │  │  │
│  │  │     │   │GROUND░ROLL  │  │  │ ╲                   │  │  │
│  │  │   50│   │░░░░░░░░░░░░ │  │  │  ╲                  │  │  │
│  │  │    0└───┼─────────────┼──┼──┼───╲────────────────┤  │  │
│  │  │     -0.02  -0.01    0  0.01 0.02                   │  │  │
│  │  │              Wavenumber (cycles/m)                  │  │  │
│  │  │                                                     │  │  │
│  │  │  🖱️ Click & drag velocity lines to adjust         │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  Display: [Input Spectrum ▼] [Log Scale] Clip: [98%]    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Filter Parameters                        │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                           │  │
│  │  Pass Band Velocities:                                   │  │
│  │    Minimum:  [2000] m/s    ◄──────────────►             │  │
│  │    Maximum:  [6000] m/s    ◄──────────────►             │  │
│  │                                                           │  │
│  │  Taper Width: [300] m/s                                  │  │
│  │                                                           │  │
│  │  ☑ Preview Live (auto-update on parameter change)       │  │
│  │                                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Side-by-Side Preview                         │  │
│  │  ┌────────────┬────────────┬────────────────────────┐    │  │
│  │  │   Input    │ Filtered   │  Rejected (Difference) │    │  │
│  │  │            │            │                        │    │  │
│  │  │  [gather]  │  [gather]  │      [gather]          │    │  │
│  │  │            │            │                        │    │  │
│  │  └────────────┴────────────┴────────────────────────┘    │  │
│  │                                                           │  │
│  │  Quality Metrics:                                         │  │
│  │    Energy preserved: 65%  │  Energy rejected: 35%        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Configuration Name: [Ground_Roll_Removal_v1   ]               │
│                                                                  │
│  [Save Configuration]  [Export FK Spectrum]  [Close]           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Design Workflow:**

1. **Open designer** on current gather
2. **Select preset** or manually adjust parameters
3. **Interactive tuning:**
   - Drag velocity lines in FK spectrum
   - Adjust sliders for v_min, v_max, taper
   - See live preview if enabled
4. **Review results:**
   - Check side-by-side: Input → Filtered → Rejected
   - Verify energy metrics
   - Inspect FK spectrum of filtered/rejected
5. **Name and save** configuration
6. **Configuration stored** for Apply mode

---

### Mode 2: Apply Mode

**Purpose:** Apply saved FK filter configuration while browsing gathers

**Entry:** User selects "FK Filter" + "Apply" mode in control panel

**Control Panel Integration:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Processing Control Panel                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Algorithm: [FK Filter        ▼]                                │
│                                                                  │
│  Mode: ○ Design  ● Apply                                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Saved Configurations:                                    │  │
│  │                                                            │  │
│  │  ● Ground_Roll_Removal_v1                                 │  │
│  │    Pass: 2000-6000 m/s, Taper: 300 m/s                   │  │
│  │    Created: 2025-11-18 10:30                              │  │
│  │                                                            │  │
│  │  ○ AirWave_Rejection                                      │  │
│  │    Reject: <400 m/s                                       │  │
│  │    Created: 2025-11-18 09:15                              │  │
│  │                                                            │  │
│  │  ○ Steep_Reflections_Only                                 │  │
│  │    Pass: >4000 m/s                                        │  │
│  │    Created: 2025-11-17 14:20                              │  │
│  │                                                            │  │
│  │  [Load...] [Edit Selected] [Delete] [Export...]          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ☑ Auto-process on gather change                                │
│  ☑ Show FK spectrum overlay                                     │
│                                                                  │
│  Current Gather: 42 / 839                                       │
│  Status: Filtered using "Ground_Roll_Removal_v1"                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Apply Workflow:**

1. **Select saved configuration** from list
2. **Enable auto-process** (optional)
3. **Browse gathers** using navigator:
   - Previous/Next buttons
   - Jump to gather
   - Ensemble navigation
4. **FK filter applied automatically** to each gather
5. **Three viewers show:**
   - Input (left)
   - Processed/Filtered (middle)
   - Difference/Rejected (right)
6. **FK spectrum overlay** (optional) shows filter zones
7. **Seamless browsing** - filter applied on-the-fly

---

### Configuration Management

**Saved Configurations Store:**

```json
{
  "name": "Ground_Roll_Removal_v1",
  "filter_type": "velocity_fan",
  "parameters": {
    "v_min": 2000,
    "v_max": 6000,
    "taper_width": 300,
    "mode": "pass"
  },
  "metadata": {
    "created": "2025-11-18T10:30:00",
    "created_on_gather": 42,
    "description": "Removes ground roll (300-1200 m/s)",
    "author": "user"
  }
}
```

**Buttons in Apply Mode:**

- **[Load...]** - Import configuration from file
- **[Edit Selected]** - Opens Design mode with selected config
- **[Delete]** - Remove saved configuration
- **[Export...]** - Save configuration to file (for sharing)

### Quick Presets (in Design Mode)

Available presets for common filtering tasks:

```
Presets Dropdown:
  ├─ Ground Roll Removal     (Pass: 1500-6000 m/s, Taper: 300 m/s)
  ├─ Air Wave Removal        (Reject: <400 m/s, Taper: 100 m/s)
  ├─ Reflection Pass         (Pass: 2000-5000 m/s, Taper: 200 m/s)
  ├─ Steep Dip Only          (Pass: >4000 m/s, Taper: 400 m/s)
  └─ Custom                  (User-defined parameters)
```

Selecting a preset:
- Loads predefined parameters
- Updates FK spectrum visualization
- User can fine-tune before saving

### Processing Indicator (in Apply Mode)

When browsing gathers with FK filter applied:

```
Status Bar: "Processing gather 42/839 with FK filter... 15ms"

Optional Progress (if processing is slow):
┌──────────────────────────────────────────┐
│  FK Filtering: Gather 5/25               │
│  ████████░░░░░░░░░░ 40%                  │
│  Est. remaining: 15s                      │
│  [Cancel]                                 │
└──────────────────────────────────────────┘
```

---

## Parameter Guidance

### Choosing Velocity Ranges

**Typical seismic velocities:**

| Event Type | Apparent Velocity | FK Filtering |
|------------|------------------|--------------|
| **Air wave** | 330 m/s | Reject: <400 m/s |
| **Ground roll** | 300-800 m/s | Reject: 200-1000 m/s |
| **Direct wave** | 500-2000 m/s | Reject: <1500 m/s |
| **Refractions** | 1500-4000 m/s | Context dependent |
| **Reflections** | 2000-6000 m/s | **Pass: 1500-6000 m/s** |
| **Multiples** | Variable | Complex pattern |
| **Aliased** | Appears as negative v | Reject: very high v |

### Taper Width Selection

**Rule of thumb:** Taper width = 10-20% of velocity range

```
Example:
  Pass: 2000-6000 m/s
  Range = 4000 m/s
  Taper = 10% × 4000 = 400 m/s

Narrow taper (50-200 m/s):
  ✅ Sharp transition
  ❌ Possible ringing artifacts
  Use: Clean separation, low noise

Wide taper (400-800 m/s):
  ✅ Smooth transition, no artifacts
  ❌ Less selective
  Use: Overlapping velocities, noisy data
```

### Trace Spacing Requirements

**Minimum number of traces:** 12-24 traces per gather

```
Fewer traces:
  ❌ Poor wavenumber resolution
  ❌ Spatial aliasing more severe
  ❌ Filter less effective

More traces:
  ✅ Better resolution
  ✅ More accurate velocity discrimination
  ✅ Can filter finer dips

Recommended: 24-48 traces for good results
```

**Uniform spacing:** CRITICAL

```
IF max(Δx) / min(Δx) > 1.2:
  ⚠️  WARNING: Spacing variation >20%
  → FK filter may produce artifacts
  → Consider interpolation or binning
```

### Frequency Range Considerations

**Low frequencies (<5 Hz):**
- Often contain ground roll
- May contain signal (deep reflections)
- Be careful not to over-filter

**High frequencies (>Nyquist/2):**
- May be aliased
- Check fold and stacking before filtering

---

## Performance Considerations

### Computational Complexity

```
For gather with N traces, M samples:

2D FFT:        O(N·M · log(N·M))
Filter apply:  O(N·M)
2D IFFT:       O(N·M · log(N·M))

Total: O(N·M · log(N·M))

Typical timing (N=48, M=1000):
  - Forward FFT: 5-10 ms
  - Filter:      1-2 ms
  - Inverse FFT: 5-10 ms
  - Total:       ~15-25 ms per gather
```

### Memory Requirements

```
Storage per gather:

Input data:     N × M × 8 bytes (float64)
FK spectrum:    N × M × 16 bytes (complex128)
Filter weights: N × M × 8 bytes (float64)
Output:         N × M × 8 bytes (float64)

Total: ~5 × N × M × 8 bytes

Example (48 traces × 1000 samples):
  5 × 48 × 1000 × 8 = 1.92 MB per gather

For 1000 gathers: ~2 GB RAM
```

### Optimization Strategies

1. **Use FFT-optimized libraries:**
   ```
   numpy.fft  (good)
   scipy.fft  (better - uses FFTW)
   pyfftw     (best - multithreaded FFTW)
   ```

2. **Batch processing:**
   ```
   Process multiple gathers in parallel
   Use joblib or multiprocessing
   Typical speedup: 4-8x on modern CPUs
   ```

3. **Power-of-2 padding:**
   ```
   Pad to nearest power-of-2 size
   FFT is much faster: 2048 vs 2000 samples
   Typical speedup: 2-3x
   ```

4. **In-place operations:**
   ```
   Reuse arrays where possible
   Reduce memory allocation overhead
   ```

---

## Warnings and Limitations

### 1. Spatial Aliasing

**Issue:** If trace spacing is too large, high-velocity events alias to low velocity

```
Critical velocity: v_crit = f·Δx / Δt

Example:
  f = 100 Hz
  Δx = 25 m
  Δt = 0.002 s

  v_crit = 100 × 25 / 0.002 = 1,250,000 m/s → No problem

But for steep dips or irregular spacing:
  May see aliasing in FK domain
  Appears as energy at negative velocities
```

**Detection:**
- Look for "bowtie" pattern in FK spectrum
- Energy at negative k for positive f (or vice versa)

**Solutions:**
- Reduce trace spacing (acquire denser data)
- Low-pass filter before FK
- Accept limitation in filtering

### 2. Mixed-Mode Events

**Issue:** Reflections and noise may overlap in FK domain

```
Example:
  Ground roll: 500 m/s
  Shallow reflections: 600 m/s

  → Cannot separate with FK filter alone
  → Need additional processing (f-x decon, τ-p, etc.)
```

### 3. 3D Effects in 2D Processing

**Issue:** Cross-line energy appears as apparent velocity variation

```
2D FK assumes all energy in inline direction
But seismic energy is 3D:
  - Cross-line dip creates apparent velocity change
  - 2D FK cannot distinguish

Solution:
  - Use 3D FK filtering (f-kx-ky) if data available
  - Be aware of cross-line contamination in 2D
```

### 4. Edge Effects

**Issue:** First and last traces may have artifacts after filtering

```
Cause: FFT periodic boundary assumption

Mitigation:
  - Taper edge traces before FK
  - Expect ~5-10% edge traces to be compromised
  - Flag edge traces in output
```

---

## Integration with Existing Workflow

### Two-Phase Processing Workflow

**Phase 1: Design (One-time setup)**

```
1. Load data with GatherNavigator
   └─ Navigate to representative gather

2. Select "FK Filter" algorithm in Control Panel
   └─ Choose "Design" mode

3. Click [Open FK Filter Designer...]
   ├─ FK spectrum computed for current gather
   ├─ Check: Uniform trace spacing? (Warning if not)
   └─ Check: Sufficient traces? (Recommended: 24+)

4. Design filter interactively
   ├─ Select preset OR manually tune
   ├─ Drag velocity lines in FK spectrum
   ├─ Adjust v_min, v_max, taper
   └─ Enable live preview for instant feedback

5. Review side-by-side results
   ├─ Input gather (left)
   ├─ Filtered gather (middle)
   ├─ Rejected energy (right)
   └─ Check energy metrics (preserved vs rejected)

6. Name and save configuration
   ├─ Give descriptive name (e.g., "Ground_Roll_Removal_v1")
   ├─ Click [Save Configuration]
   └─ Configuration stored for Apply mode
```

**Phase 2: Apply (Batch processing)**

```
1. Select "FK Filter" algorithm in Control Panel
   └─ Choose "Apply" mode

2. Select saved configuration from list
   ├─ Review parameters summary
   └─ Enable "Auto-process on gather change"

3. Browse through gathers
   ├─ Use Previous/Next buttons
   ├─ Jump to specific gather
   ├─ Navigate by ensemble
   └─ FK filter applied automatically to each

4. Monitor results in three viewers
   ├─ Input (left) - Original gather
   ├─ Processed (middle) - FK filtered
   └─ Difference (right) - Rejected noise

5. Adjust if needed
   ├─ If results unsatisfactory on some gathers
   ├─ Click [Edit Selected] to return to Design mode
   ├─ Fine-tune parameters
   └─ Re-save configuration

6. Continue QC workflow
   └─ Browse all gathers with consistent filtering
```

### Compatibility with Other Filters

**Can combine with:**
- ✅ Bandpass filter (frequency domain)
- ✅ TF-Denoise (time-frequency adaptive)
- ✅ Trace muting
- ✅ Gain controls

**Typical order:**
1. Bandpass (remove extreme frequencies)
2. FK filter (remove coherent linear noise)
3. TF-Denoise (remove random noise)
4. Gain/display adjustments

---

## Future Enhancements

### Phase 1 (Current Design)
- Velocity fan filter
- Basic UI with presets
- FK spectrum display

### Phase 2 (Advanced)
- Interactive polygon filter
- Dip-dependent filtering
- Multiple reject zones
- Filter comparison mode

### Phase 3 (Professional)
- 3D FK filtering (f-kx-ky)
- Radial trace FK
- τ-p (tau-p) domain filtering
- Machine learning guided filter design

---

## References

### Theory
- Yilmaz, O. (2001). *Seismic Data Analysis*. SEG. Chapter 6: Velocity Analysis.
- Sheriff, R.E. & Geldart, L.P. (1995). *Exploration Seismology*. Cambridge University Press.

### Implementation
- Scipy FFT documentation: https://docs.scipy.org/doc/scipy/reference/fft.html
- NumPy FFT tutorial: https://numpy.org/doc/stable/reference/routines.fft.html

### Best Practices
- SEG Wiki: FK filtering best practices
- Processing guidelines for land data

---

**Document Version:** 1.0
**Date:** 2025-11-18
**Status:** Design Complete - Ready for Implementation Review
