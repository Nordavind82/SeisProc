# ISA Quick Start Guide

## What is ISA?
Interactive Spectral Analysis - a tool for viewing frequency content of seismic traces in real-time.

## How to Use

### 1. Open ISA Window
```
Main Menu → View → Open ISA Window
or press Ctrl+I
```

### 2. View Spectrum
- **Click on any trace** in the seismic viewer (top panel)
- Spectrum appears instantly in bottom panel
- Red dot marks the dominant (peak) frequency

### 3. Controls (Right Panel)

**Trace Selection:**
- Use spinbox to select trace number
- Or click directly on seismic display

**Average Spectrum:**
- Check "Show average spectrum" to see ensemble average
- Useful for identifying common frequency content across all traces

**Frequency Range:**
- Set Min/Max frequency limits to zoom in
- Click "Reset Range" to restore full range (0 to Nyquist)

### 4. Understanding the Spectrum

**X-axis:** Frequency in Hz (0 to Nyquist frequency)
**Y-axis:** Amplitude in dB (decibels)

**Peak Frequency (Red Dot):**
- Shows the dominant frequency component
- Labeled with exact frequency value
- Example: "Peak: 30.5 Hz"

## Common Use Cases

### 1. Check Data Bandwidth
- Open ISA window
- Enable "Show average spectrum"
- Observe frequency range with significant energy
- Typical seismic data: 5-60 Hz

### 2. Verify Filter Effectiveness
- Apply bandpass filter in main window
- Open ISA for input data
- Check spectrum shows energy outside filter passband
- Expected: frequencies outside filter band should be present

### 3. Identify Noise Frequencies
- Click on noisy traces
- Look for narrow peaks at specific frequencies
- Common: 50/60 Hz power line noise
- Use this to design notch filters

### 4. Compare Traces
- Click through different traces
- Compare their spectra
- Identify variations in frequency content

## Tips

- **Zoom spectrum:** Use matplotlib toolbar (top of spectrum panel)
- **Pan spectrum:** Click "Pan" tool in matplotlib toolbar
- **Multiple ISA windows:** Open multiple ISA windows for different gathers
- **Frequency range:** Narrow the range to focus on specific bands

## Example Workflow

```
1. Load seismic data (File → Load SEG-Y or Generate Sample Data)
2. Open ISA (View → Open ISA Window / Ctrl+I)
3. Click on a clean trace to see signal spectrum
4. Click on a noisy trace to identify noise frequencies
5. Use this info to configure bandpass filter
6. Apply filter in main window
7. Open new ISA window to verify filter results
```

## Keyboard Shortcuts

- **Ctrl+I** - Open ISA Window
- **Arrow Keys** - Navigate through trace spinbox
- **Mouse Wheel** - Scroll through traces (when spinbox focused)

## Technical Info

**Nyquist Frequency:**
- Maximum detectable frequency
- Equals: sampling_rate_hz / 2
- Example: 2ms sample rate → 500 Hz sampling → 250 Hz Nyquist

**dB Scale:**
- Logarithmic amplitude scale
- 20 dB difference = 10x amplitude ratio
- Higher dB = stronger signal

**Dominant Frequency:**
- Frequency with highest amplitude
- Marked with red dot on plot
- Shows in legend with exact value

## Troubleshooting

**"No Data" error:**
- Load seismic data first (File menu)

**Spectrum looks noisy:**
- Normal for noisy data
- Try "Show average spectrum" for cleaner view

**Can't click on traces:**
- Make sure cursor is over the seismic image
- Try clicking on different parts of the image

**Frequency range too narrow:**
- Click "Reset Range" button
- Check Nyquist frequency in Info panel

## Related Features

- **Bandpass Filter** (main window) - Use ISA to verify filter effectiveness
- **Flip Window** - Quick view cycling for QC
- **FK Filter** - Frequency-wavenumber filtering (use ISA for frequency QC)

---

**Need help?** Check ISA_FEATURE_SUMMARY.md for detailed technical documentation.
