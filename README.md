# Seismic Data Processing QC Tool

Professional seismic data quality control application built with PyQt6.

## Features

### Three Synchronized Viewers
- **Input Data**: Original seismic data
- **Processed Data**: After applying filters/processing
- **Difference**: Residual (Input - Processed) for QC analysis

### Interactive Controls
- **Zoom In/Out**: Focus on specific time/trace ranges
- **Pan**: Navigate through the data
- **Gain Control**: Adjust amplitude scaling (0.1x to 10x)
- **Clip Percentile**: Control display clipping (90% to 100%)

### Processing Capabilities
- **Zero-Phase Bandpass Filter**: Butterworth filter without time shifts
- **Extensible Pipeline**: Easy to add new processing modules
- **Amplitude Preservation**: Non-destructive workflows

## Architecture

```
seismic_qc_app/
├── models/                 # Data structures
│   ├── seismic_data.py    # Seismic data container
│   └── viewport_state.py  # Synchronized view state
├── processors/            # Processing operations
│   ├── base_processor.py # Abstract base class
│   └── bandpass_filter.py # Bandpass implementation
├── views/                 # UI components
│   ├── seismic_viewer.py # Seismic display widget
│   └── control_panel.py  # Processing controls
├── utils/                 # Utilities
│   └── sample_data.py    # Test data generator
├── main_window.py         # Main application window
└── main.py               # Entry point
```

## Design Principles

### Architecture Best Practices
✅ **Separation of Concerns**: Data, processing, and visualization are independent
✅ **Observer Pattern**: Views react to state changes via signals
✅ **Extensibility**: Easy to add new processors
✅ **Immutability**: Processing creates new data, never modifies input

### Geophysical Best Practices
✅ **Amplitude Preservation**: User-controlled gain, no auto-normalization
✅ **Zero-Phase Filtering**: No time shifts in QC workflows
✅ **Nyquist Validation**: Filter frequencies validated against sample rate
✅ **Independent Difference Scaling**: QC residuals shown with proper gain

### What's NOT in the App (Anti-Patterns Avoided)
❌ Hard-coded parameters in UI
❌ Direct coupling between panels
❌ Blocking operations in main thread
❌ Global state variables
❌ Automatic destructive operations
❌ Mixed time/depth domains
❌ Phase shifts in filters

## Installation

### Requirements
```bash
pip install PyQt6 numpy scipy matplotlib
```

### Running the Application
```bash
cd seismic_qc_app
python main.py
```

## Usage

1. **Generate Sample Data**
   - File → Generate Sample Data (Ctrl+G)
   - Creates synthetic seismic data with reflections

2. **Apply Bandpass Filter**
   - Set Low Frequency (Hz)
   - Set High Frequency (Hz) - must be < Nyquist
   - Set Filter Order (1-10, default 4)
   - Click "Apply Filter"

3. **Adjust Display**
   - Use Gain slider for amplitude scaling
   - Use Clip Percentile to control dynamic range
   - Zoom In/Out buttons for magnification
   - Reset View to show all data

4. **Quality Control**
   - Compare Input vs Processed in synchronized views
   - Examine Difference panel for processing artifacts
   - All three panels zoom/pan together

## Extending the Application

### Adding New Processors

1. Create new processor class inheriting from `BaseProcessor`:

```python
from processors.base_processor import BaseProcessor
from models.seismic_data import SeismicData

class MyProcessor(BaseProcessor):
    def _validate_params(self):
        # Validate parameters
        pass

    def process(self, data: SeismicData) -> SeismicData:
        # Process data and return new SeismicData
        pass

    def get_description(self) -> str:
        return "My processor description"
```

2. Add to control panel UI
3. Connect to processing pipeline

### Future Enhancements

- SEG-Y file loader
- AGC (Automatic Gain Control)
- Deconvolution
- FK filtering
- Trace editing/muting
- Export to SEG-Y
- Processing history tracking
- Undo/redo functionality

## Technical Details

- **Sample Rate**: Milliseconds (typical: 2ms, 4ms)
- **Nyquist Frequency**: 1000 / (2 × sample_rate) Hz
- **Filter**: Zero-phase Butterworth (scipy.signal.sosfiltfilt)
- **Display**: Matplotlib with seismic colormap (red-white-blue)
- **Coordinate Convention**: Time increases downward (standard seismic)

## License

Educational/demonstration purposes.
