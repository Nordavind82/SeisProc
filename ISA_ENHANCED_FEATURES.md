# ISA Enhanced Features

## Summary of Enhancements

The Interactive Spectral Analysis (ISA) window has been enhanced with the following features:

### 1. ✓ Log/Linear Scale Selection for Spectrum

**Location:** Control Panel → "Spectrum Display" group

**Options:**
- **Linear (dB)** - Default, shows amplitude in decibels (logarithmic amplitude scale)
- **Logarithmic** - Shows linear amplitude on log Y-axis scale

**Use Cases:**
- **Linear (dB)**: Standard for seismic QC, better for comparing relative amplitudes
- **Logarithmic**: Better for viewing wide dynamic range, seeing small peaks clearly

**Technical:**
- Conversion: `linear_amplitude = 10^(dB/20)`
- Log scale applied to Y-axis using matplotlib `set_yscale('log')`

---

### 2. ✓ Matplotlib Toolbar Removed

**Change:** The matplotlib navigation toolbar has been removed from the spectrum panel for a cleaner interface.

**Before:** Toolbar with Pan, Zoom, Home, Save buttons
**After:** Clean spectrum display with no toolbar

**Benefit:** Less cluttered interface, more space for spectrum display

**Note:** You can still zoom/pan using the seismic viewer controls and frequency range settings.

---

### 3. ✓ Colormap Selection for Seismic Data

**Location:** Control Panel → "Data Colormap" group

**Available Colormaps:**
- **seismic** (default) - Red-White-Blue, standard seismic colormap
- **gray** - Grayscale, good for printing
- **RdBu** - Red-Blue diverging
- **viridis** - Perceptually uniform, colorblind-friendly
- **plasma** - High contrast
- **coolwarm** - Blue-White-Red diverging

**Use Cases:**
- **seismic/RdBu**: Standard for amplitude displays
- **gray**: Black and white displays/printing
- **viridis/plasma**: Colorblind-accessible, modern colormaps
- **coolwarm**: Alternative diverging colormap

**Technical:**
- Changes applied through ViewportState
- Updates PyQtGraph seismic viewer colormap in real-time

---

### 4. ✓ Time Window Selection for Windowed Spectrum

**Location:** Control Panel → "Time Window" group

**Controls:**
- **Checkbox:** "Use time window for analysis"
- **Start (ms):** Start time in milliseconds
- **End (ms):** End time in milliseconds

**How It Works:**
1. Enable "Use time window for analysis" checkbox
2. Set start and end times (in milliseconds)
3. Spectrum is computed only for data within this time window
4. Window info appears in spectrum title: `[100.0-500.0 ms]`

**Use Cases:**
- **Shallow analysis:** Analyze only near-surface arrivals (e.g., 0-500ms)
- **Deep analysis:** Focus on deep reflections (e.g., 1500-3000ms)
- **Event isolation:** Isolate specific seismic events in time
- **Time-varying frequency:** Compare frequency content at different times

**Example:**
```
Data duration: 2000 ms
Time window: 500-1000 ms
Result: Spectrum computed only for 500ms-1000ms portion of each trace
```

**Technical Details:**
- Converts time (ms) to sample indices: `sample_idx = int(time_ms / sample_rate_ms)`
- Extracts windowed data: `windowed_traces = data.traces[start_idx:end_idx, :]`
- FFT computed on windowed segment
- Frequency resolution unchanged (depends on window length)

---

## Complete Control Panel Layout

```
┌─────────────────────────────────────┐
│ Trace Selection                     │
│  - Trace: [spinbox]                 │
│  - □ Show average spectrum          │
├─────────────────────────────────────┤
│ Time Window                    ← NEW│
│  - □ Use time window                │
│  - Start (ms): [spinbox]            │
│  - End (ms): [spinbox]              │
├─────────────────────────────────────┤
│ Spectrum Display               ← NEW│
│  - Y-axis scale:                    │
│    ○ Linear (dB)                    │
│    ○ Logarithmic                    │
├─────────────────────────────────────┤
│ Data Colormap                  ← NEW│
│  - Colormap: [dropdown]             │
├─────────────────────────────────────┤
│ Frequency Range                     │
│  - Min (Hz): [spinbox]              │
│  - Max (Hz): [spinbox]              │
│  - [Reset Range]                    │
├─────────────────────────────────────┤
│ Information                         │
│  - Data Info                        │
│  - Current Trace                    │
└─────────────────────────────────────┘
```

---

## Usage Examples

### Example 1: Analyze Shallow vs Deep Frequency Content

**Goal:** Compare frequency content of shallow (0-500ms) vs deep (1500-2000ms) data

**Steps:**
1. Open ISA window
2. Enable "Use time window for analysis"
3. Set window: 0-500 ms
4. Note dominant frequency (e.g., 45 Hz)
5. Change window: 1500-2000 ms
6. Compare dominant frequency (e.g., 25 Hz - lower due to attenuation)

**Result:** Quantify frequency attenuation with depth

---

### Example 2: View Wide Dynamic Range with Log Scale

**Goal:** See both strong and weak frequency components clearly

**Steps:**
1. Open ISA window
2. Click on a trace with wide amplitude range
3. In "Spectrum Display", select "Logarithmic"
4. Weak frequency peaks now visible alongside strong peaks

**Result:** Better visualization of full frequency content

---

### Example 3: Find Best Colormap for Your Monitor

**Goal:** Optimize seismic data visualization

**Steps:**
1. Open ISA window
2. Try different colormaps from "Data Colormap" dropdown
3. Compare: seismic (traditional) vs viridis (modern)
4. Select based on your preference and display

**Result:** Optimized data visualization

---

### Example 4: Isolated Event Analysis

**Goal:** Analyze frequency content of a specific reflection event at 800-900ms

**Steps:**
1. Identify event time in seismic viewer: 800-900ms
2. Enable "Use time window for analysis"
3. Set window: 800-900 ms
4. View spectrum of just this event
5. Check if filtering will preserve/remove this event

**Result:** Targeted frequency analysis of specific events

---

## Testing

All enhanced features have been tested:

```bash
# Run enhanced feature tests
python test_isa_enhanced.py
```

**Test Results:**
- ✓ Time windowing (time-to-sample conversion)
- ✓ Log/linear scale conversion (dB to linear)
- ✓ Window extraction from traces
- ✓ Colormap changes applied correctly

---

## Technical Implementation

### Files Modified

**views/isa_window.py:**
- Added time window controls and logic
- Added log/linear scale selection
- Added colormap dropdown
- Removed matplotlib toolbar
- Updated `_update_spectrum()` to handle windowing and scale

**Key Changes:**
```python
# Time windowing
if self.use_time_window:
    start_idx = int(self.time_window_start / self.data.sample_rate)
    end_idx = int(self.time_window_end / self.data.sample_rate)
    windowed_traces = self.data.traces[start_idx:end_idx, :]

# Log/linear scale
if self.spectrum_scale == 'log':
    y_data = 10 ** (amplitudes_db / 20.0)
    self.spectrum_ax.set_yscale('log')

# Colormap
self.viewport_state.set_colormap(colormap_name)
```

---

## Comparison: Before vs After

| Feature | Before | After |
|---------|--------|-------|
| Spectrum Scale | dB only | dB or Linear log scale |
| Matplotlib Toolbar | Visible | Removed (cleaner) |
| Data Colormap | Fixed (seismic) | 6 options |
| Time Window | Full trace only | Selectable window |
| Control Panel Width | 300px | 320px (more controls) |

---

## Tips

**Time Window:**
- Start with full trace, then narrow down to regions of interest
- Window size affects frequency resolution (smaller window = coarser resolution)
- Use for isolating events, not for improving noise reduction

**Log Scale:**
- Use logarithmic for wide dynamic range data
- Linear (dB) is standard for most seismic QC
- Try both to see which reveals features better

**Colormap:**
- Stick with 'seismic' for traditional workflows
- Use 'viridis' or 'plasma' for modern colorblind-friendly displays
- 'gray' is best for black & white printing

**Performance:**
- Time windowing is fast (just array slicing)
- All operations are real-time, no lag
- Colormaps change instantly

---

## Conclusion

These enhancements make the ISA window more powerful and flexible for seismic QC:

1. **Time windowing** enables targeted analysis of specific time ranges
2. **Log/linear scale** improves visualization for different data types
3. **Colormap selection** allows customization for different displays/purposes
4. **Cleaner interface** (no matplotlib toolbar) improves usability

All features tested and working ✓
