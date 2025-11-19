# Quick Start Guide

## Installation

### Step 1: Set Up Virtual Environment (Recommended)

**Automated Setup:**
```bash
./setup_env.sh
```

**Manual Setup:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Activate Environment

Every time you want to work on the project:
```bash
source venv/bin/activate
```

Or use the helper script:
```bash
source activate.sh
```

### Step 3: Verify Installation (Optional)
```bash
python test_components.py
```

You should see:
```
============================================================
✓ ALL TESTS PASSED
============================================================
```

## Running the Application

### Launch the GUI
```bash
# Make sure virtual environment is activated
python main.py
```

## First Steps

### 1. Generate Sample Data
- Click **File → Generate Sample Data** (or press `Ctrl+G`)
- This creates synthetic seismic data with:
  - 100 traces
  - 1000 samples (2000ms duration)
  - Multiple reflection events
  - Realistic noise

### 2. Apply Bandpass Filter
- In the left control panel, set:
  - **Low Freq**: 10.0 Hz
  - **High Freq**: 80.0 Hz
  - **Filter Order**: 4
- Click **Apply Filter**

### 3. Observe Results
- **Left panel**: Original input data
- **Middle panel**: Filtered data
- **Right panel**: Difference (noise removed by filter)

All three panels are synchronized!

### 4. Adjust Display

**Gain Control**:
- Move the **Gain** slider to adjust amplitude
- Range: 0.1x (quiet) to 10x (loud)
- All three panels update together

**Clip Percentile**:
- Move the **Clip Percentile** slider
- 99% = clip at 99th percentile (standard)
- Lower values = more clipping (better for weak signals)

### 5. Zoom and Pan

**Zoom In/Out**:
- Click **Zoom In** button (50% of current view)
- Click **Zoom Out** button (200% of current view)
- Or use matplotlib toolbar zoom tools

**Reset View**:
- Click **Reset View** to show all data

**Pan**:
- Use matplotlib toolbar pan tool
- All three panels pan together

## Understanding the Display

### Color Scale
- **Red**: Positive amplitude
- **White**: Zero amplitude
- **Blue**: Negative amplitude

This is the standard seismic colormap used in industry.

### Axes
- **Horizontal**: Trace number (spatial dimension)
- **Vertical**: Time in milliseconds (increases downward)

### Colorbar
- Shows amplitude range
- Updates with gain changes

## Working with Real Data

### SEG-Y Files (Future Feature)
Currently, SEG-Y loading is not implemented. To add your own data:

```python
# In utils/data_loader.py (create this file)
import segyio
from models import SeismicData

def load_segy(filename):
    with segyio.open(filename) as f:
        traces = segyio.tools.cube(f)[0].T  # (samples, traces)
        sample_rate = segyio.tools.dt(f) / 1000.0  # Convert to ms

    return SeismicData(traces=traces, sample_rate=sample_rate)
```

Then add to File menu in `main_window.py`.

## Common Workflows

### QC Workflow: Comparing Filters

1. Load data
2. Apply bandpass filter with settings A
3. Note the difference panel results
4. Change filter settings to B
5. Apply again
6. Compare difference panels (which removed more noise?)

### Gain Optimization

1. Load data
2. Set gain to 1.0x
3. Apply filter
4. Adjust gain on difference panel to see residuals
5. If residuals are large → filter too aggressive
6. If residuals are small → filter working well

### Frequency Analysis

1. Start with wide bandpass (5-100 Hz)
2. Note what's removed (difference panel)
3. Narrow the bandpass (10-80 Hz)
4. Compare difference panels
5. Identify frequency content of noise vs signal

## Keyboard Shortcuts

- `Ctrl+O`: Load data (not yet implemented)
- `Ctrl+G`: Generate sample data
- `Ctrl+Q`: Quit application

## Tips and Tricks

### Nyquist Frequency
- Sample rate = 2ms → Nyquist = 250 Hz
- Sample rate = 4ms → Nyquist = 125 Hz
- Filter high frequency MUST be < Nyquist
- App will warn you if you try to exceed it

### Filter Order
- Higher order = sharper cutoff
- Lower order = smoother transition
- Typical: 4th order (default)
- Range: 1-10

### Gain vs Clip Percentile
- **Gain**: Multiplies all amplitudes equally
- **Clip**: Controls display range, doesn't change data
- Use gain for overall scaling
- Use clip to enhance weak signals

### Difference Panel Interpretation
- Large amplitudes in difference = filter removed a lot
- Small amplitudes in difference = filter preserved data
- Organized patterns = filter removed signal (bad!)
- Random noise = filter removed noise (good!)

## Troubleshooting

### "No Data" Warning
**Problem**: Trying to apply filter without data
**Solution**: Generate sample data first (Ctrl+G)

### "High frequency must be < Nyquist"
**Problem**: Filter frequency too high for sample rate
**Solution**: Reduce high frequency or check sample rate

### Application Won't Start
**Problem**: Missing dependencies
**Solution**: Run `pip install -r requirements.txt`

### Plots Look Wrong
**Problem**: Matplotlib backend issues
**Solution**: Try setting `export QT_API=pyqt6` in terminal

### Slow Performance
**Problem**: Large dataset
**Solution**: Reduce data size or implement downsampling

## Next Steps

### Extend the Application

**Add AGC (Automatic Gain Control)**:
```python
# In processors/agc.py
class AGCProcessor(BaseProcessor):
    def process(self, data):
        # Implement AGC algorithm
        pass
```

**Add Trace Editing**:
- Mute bad traces
- Zero out time ranges
- Apply trace scaling

**Add Export**:
- Save processed data to SEG-Y
- Export images to PNG/PDF
- Save processing parameters to JSON

### Learn More

- Read `ARCHITECTURE.md` for design details
- Read `README.md` for feature list
- Check source code comments for implementation details

## Getting Help

If you encounter issues:
1. Check this guide
2. Read error messages carefully (they're informative!)
3. Review the architecture documentation
4. Check the source code (it's well-commented)

## Example Session

```
$ python main.py
[Application opens]

File → Generate Sample Data
[Three panels appear with synthetic data]

Set Low Freq: 10 Hz
Set High Freq: 80 Hz
Click "Apply Filter"
[Middle panel shows filtered data, right panel shows removed noise]

Adjust Gain slider
[All panels update amplitudes together]

Click "Zoom In"
[All panels zoom to center 50% of data]

Use matplotlib pan tool
[All panels pan together]

Click "Reset View"
[Back to full data view]

File → Exit
[Application closes]
```

Enjoy your seismic QC analysis!
