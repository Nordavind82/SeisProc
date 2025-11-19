# Architecture Documentation

## System Overview

This application follows **Model-View-Controller (MVC)** architecture with **Observer pattern** for synchronized views.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Window                               │
│  ┌──────────────┐  ┌────────────────────────────────────────┐  │
│  │              │  │  Three Synchronized Seismic Viewers    │  │
│  │   Control    │  │  ┌──────────┬──────────┬──────────┐   │  │
│  │    Panel     │  │  │  Input   │Processed │Difference│   │  │
│  │              │  │  │  Viewer  │ Viewer   │  Viewer  │   │  │
│  │  - Filter    │  │  │          │          │          │   │  │
│  │    Params    │  │  │          │          │          │   │  │
│  │  - Gain      │  │  │          │          │          │   │  │
│  │  - Clip      │  │  │          │          │          │   │  │
│  │  - Zoom/Pan  │  │  └──────────┴──────────┴──────────┘   │  │
│  └──────────────┘  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────┐
                              │                         │
                              ▼                         ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │  ViewportState   │    │   SeismicData    │
                    │  (Observable)    │    │     (Model)      │
                    │                  │    │                  │
                    │  - limits        │    │  - traces        │
                    │  - gain          │    │  - sample_rate   │
                    │  - clip %        │    │  - headers       │
                    │                  │    │  - metadata      │
                    │  Signals:        │    └──────────────────┘
                    │  - limits_changed│              │
                    │  - gain_changed  │              │
                    │  - clip_changed  │              ▼
                    └──────────────────┘    ┌──────────────────┐
                              ▲             │  BaseProcessor   │
                              │             │   (Abstract)     │
                              │             └──────────────────┘
                    ┌─────────┴──────────┐           │
                    │                    │           │
              All 3 viewers          Observers       ▼
              observe state                  ┌──────────────────┐
              changes via                    │ BandpassFilter   │
              Qt signals                     │                  │
                                             │ - low_freq       │
                                             │ - high_freq      │
                                             │ - order          │
                                             │                  │
                                             │ process()        │
                                             │ validate()       │
                                             └──────────────────┘
```

## Component Responsibilities

### Models (`models/`)

**SeismicData**
- Immutable container for seismic traces and metadata
- Validates data integrity (shape, sample rate, Nyquist)
- Provides convenience methods (get_time_axis, duration, etc.)
- **Does NOT**: Modify data, handle visualization, perform processing

**ViewportState (Observable)**
- Single source of truth for view synchronization
- Manages zoom, pan, gain, and clip settings
- Emits Qt signals when state changes
- **Does NOT**: Store seismic data, perform rendering, handle processing

### Processors (`processors/`)

**BaseProcessor (Abstract Base Class)**
- Defines contract for all processors
- Enforces parameter validation
- Ensures immutability (returns new SeismicData)
- **Does NOT**: Implement specific algorithms, handle UI

**BandpassFilter**
- Zero-phase Butterworth bandpass filtering
- Validates frequencies against Nyquist
- Applies scipy.signal.sosfiltfilt (forward-backward filtering)
- **Does NOT**: Modify input data, handle display, store state

**ProcessingPipeline**
- Chain of responsibility pattern
- Applies multiple processors sequentially
- Tracks processing history
- **Does NOT**: Implement specific algorithms, handle concurrency

### Views (`views/`)

**SeismicViewer**
- Matplotlib-based seismic display widget
- Observes ViewportState for synchronized updates
- Renders variable density (color) display
- **Does NOT**: Modify data, perform processing, manage state

**ControlPanel**
- User interface for parameters and controls
- Emits signals for user actions
- Validates input ranges
- **Does NOT**: Process data, manage global state, communicate directly with viewers

### Main Window (`main_window.py`)

**MainWindow (Controller)**
- Coordinates all components
- Manages data flow: Input → Processor → Processed → Difference
- Handles user actions and menu commands
- **Does NOT**: Implement processing logic, handle rendering details

## Data Flow

### Loading Data
```
User Action (Menu/Button)
    ↓
MainWindow._generate_sample_data()
    ↓
generate_sample_seismic_data() → SeismicData
    ↓
MainWindow.input_data = SeismicData
    ↓
input_viewer.set_data(SeismicData)
    ↓
ViewportState.reset_to_data()
    ↓
All viewers update (via signal)
```

### Processing Flow
```
User clicks "Apply Filter"
    ↓
ControlPanel.process_requested signal → BandpassFilter
    ↓
MainWindow._on_process_requested(processor)
    ↓
processed_data = processor.process(input_data)
    ├─ Validates Nyquist frequency
    ├─ Applies zero-phase filter
    └─ Returns NEW SeismicData (input unchanged)
    ↓
difference_data = input - processed
    ↓
Update viewers:
    - processed_viewer.set_data(processed_data)
    - difference_viewer.set_data(difference_data)
    ↓
Viewers render with current ViewportState
```

### View Synchronization
```
User adjusts gain slider
    ↓
ControlPanel.gain_changed signal → float
    ↓
ViewportState.set_gain(value)
    ↓
ViewportState.gain_changed signal → ALL viewers
    ↓
Each viewer._on_gain_changed()
    ↓
Each viewer._update_display()
    ↓
All three panels update simultaneously
```

## Design Patterns Used

### 1. Observer Pattern
- **ViewportState** is the subject
- **Three SeismicViewers** are observers
- Changes to viewport propagate automatically via Qt signals

### 2. Model-View-Controller (MVC)
- **Model**: SeismicData, ViewportState
- **View**: SeismicViewer, ControlPanel
- **Controller**: MainWindow

### 3. Strategy Pattern
- **BaseProcessor** defines strategy interface
- Different processors (BandpassFilter, etc.) implement different strategies
- MainWindow uses processors without knowing implementation details

### 4. Chain of Responsibility
- **ProcessingPipeline** chains multiple processors
- Each processor handles data and passes to next
- Easy to add/remove processing steps

### 5. Immutability
- SeismicData objects are never modified
- Processing creates new objects
- Prevents accidental data corruption

## Geophysical Design Principles

### Amplitude Preservation
- No automatic normalization
- User-controlled gain with explicit values
- Clip percentile visible to user
- Difference panel has independent scaling

### Zero-Phase Filtering
- Uses forward-backward filtering (scipy.signal.sosfiltfilt)
- Prevents time shifts that would invalidate QC
- Critical for comparing input vs processed

### Nyquist Frequency Validation
```python
nyquist = 1000.0 / (2.0 * sample_rate)
if high_freq >= nyquist:
    raise ValueError(...)
```
- Prevents aliasing artifacts
- Validates at processor creation AND execution

### Seismic Display Conventions
- Time increases downward (ax.invert_yaxis())
- Seismic colormap (red-white-blue)
- Time in milliseconds
- Trace numbers as spatial coordinate

## Anti-Patterns Avoided

### ❌ God Object
**Avoided by**: Separating concerns into specialized classes
- Not one massive "SeismicApp" class with 1000+ lines

### ❌ Tight Coupling
**Avoided by**: Signal/slot communication and abstract interfaces
- Viewers don't know about each other
- Processing doesn't know about UI

### ❌ Magic Numbers
**Avoided by**: Named constants and explicit parameters
- Sample rate stored in data object
- Nyquist calculated from sample rate
- All filter parameters explicit

### ❌ Mixed Responsibilities
**Avoided by**: Clear separation of data, processing, and visualization
- SeismicData doesn't render itself
- Processors don't update UI
- Views don't process data

## Extensibility Points

### Adding New Processors
1. Inherit from `BaseProcessor`
2. Implement `_validate_params()`, `process()`, `get_description()`
3. Add to ControlPanel UI
4. Connect signal to MainWindow

### Adding New Viewers
1. Inherit from `QWidget`
2. Connect to ViewportState signals
3. Implement `_on_limits_changed()`, etc.
4. Add to MainWindow layout

### Adding Data Loaders
1. Create function: `SeismicData load_segy(path: str)`
2. Add to File menu
3. Connect to MainWindow method

## Thread Safety

**Current**: All processing in main thread
**Future**: Use QThread for heavy processing
```python
class ProcessingWorker(QThread):
    finished = pyqtSignal(SeismicData)

    def run(self):
        result = self.processor.process(self.data)
        self.finished.emit(result)
```

## Testing Strategy

**Unit Tests**: Each component tested in isolation
- `test_seismic_data.py`: Data model
- `test_processors.py`: Processing logic
- `test_viewport_state.py`: State management

**Integration Tests**: Component interaction
- `test_components.py`: Current test file
- Tests data → processing → difference flow

**GUI Tests**: (Future) Qt Test framework
- User interaction scenarios
- Signal/slot connections

## Performance Considerations

**Current Optimizations**:
- NumPy for all numerical operations
- Matplotlib blitting (could be improved)
- Minimal data copying (immutability via copy-on-write)

**Future Optimizations**:
- Memory-mapped arrays for large files
- GPU acceleration (CuPy) for processing
- Level-of-detail rendering for zoom
- Caching processed results

## Deployment

**Standalone Application**: Use PyInstaller
```bash
pyinstaller --onefile --windowed main.py
```

**Docker Container**: For reproducible environment
```dockerfile
FROM python:3.11
RUN pip install PyQt6 numpy scipy matplotlib
COPY seismic_qc_app /app
CMD ["python", "/app/main.py"]
```
