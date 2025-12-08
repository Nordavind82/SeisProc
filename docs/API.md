# SeisProc API Documentation

Comprehensive API reference for the SeisProc seismic data processing library.

## Table of Contents

- [Models](#models)
  - [SeismicData](#seismicdata)
  - [LazySeismicData](#lazyseismicdata)
  - [ViewportState](#viewportstate)
  - [GatherNavigator](#gathernavigator)
- [Processors](#processors)
  - [BaseProcessor](#baseprocessor)
  - [BandpassFilter](#bandpassfilter)
  - [TFDenoise](#tfdenoise)
  - [FKFilter](#fkfilter)
- [SEG-Y Import/Export](#seg-y-importexport)
  - [SEGYReader](#segyreader)
  - [SEGYExporter](#segyexporter)
  - [HeaderMapping](#headermapping)
- [GPU Acceleration](#gpu-acceleration)
  - [DeviceManager](#devicemanager)
  - [TFDenoiseGPU](#tfdenoisegpu)
- [Utilities](#utilities)
  - [ThemeManager](#thememanager)
  - [MemoryMonitor](#memorymonitor)

---

## Models

### SeismicData

Primary container for seismic trace data.

```python
from models import SeismicData
```

#### Constructor

```python
SeismicData(traces: np.ndarray, sample_rate: float, headers: Optional[pd.DataFrame] = None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `traces` | `np.ndarray` | 2D array of shape (n_samples, n_traces) |
| `sample_rate` | `float` | Sample interval in milliseconds |
| `headers` | `pd.DataFrame` | Optional trace headers |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_samples` | `int` | Number of samples per trace |
| `n_traces` | `int` | Number of traces |
| `sample_rate` | `float` | Sample interval (ms) |
| `duration_ms` | `float` | Total record length (ms) |
| `traces` | `np.ndarray` | Trace data array |
| `headers` | `pd.DataFrame` | Trace header DataFrame |

#### Example

```python
import numpy as np
from models import SeismicData

# Create synthetic data
traces = np.random.randn(1000, 50).astype(np.float32)
data = SeismicData(traces=traces, sample_rate=2.0)

print(f"Traces: {data.n_traces}, Samples: {data.n_samples}")
print(f"Duration: {data.duration_ms} ms")
```

---

### LazySeismicData

Memory-efficient container for large datasets using Zarr chunked storage.

```python
from models import LazySeismicData
```

#### Class Methods

```python
LazySeismicData.from_storage_dir(storage_dir: str) -> LazySeismicData
```

Load lazy data from a Zarr storage directory.

#### Methods

```python
get_window(time_start: float, time_end: float,
           trace_start: int, trace_end: int) -> np.ndarray
```

Extract a windowed subset of data without loading the entire dataset.

```python
iterate_chunks(chunk_size: int = 100) -> Iterator[np.ndarray]
```

Iterate over trace chunks for memory-efficient processing.

#### Example

```python
from models import LazySeismicData

# Load from Zarr storage
lazy_data = LazySeismicData.from_storage_dir("/path/to/zarr_dir")

# Get a specific window
window = lazy_data.get_window(
    time_start=100,   # ms
    time_end=500,     # ms
    trace_start=0,
    trace_end=100
)

# Process in chunks
for chunk in lazy_data.iterate_chunks(chunk_size=50):
    process(chunk)
```

---

### ViewportState

Synchronized viewport state for multi-viewer applications.

```python
from models import ViewportState, ViewportLimits
```

#### Signals

| Signal | Parameters | Description |
|--------|------------|-------------|
| `viewport_changed` | `ViewportLimits` | Emitted when viewport boundaries change |
| `gain_changed` | `float` | Emitted when display gain changes |
| `colormap_changed` | `str` | Emitted when colormap changes |

#### Methods

```python
set_limits(x_min: float, x_max: float, y_min: float, y_max: float)
```

Set viewport boundaries (triggers `viewport_changed` signal).

```python
set_gain(gain: float)
```

Set display gain (triggers `gain_changed` signal).

#### Example

```python
from models import ViewportState

# Create shared viewport state
viewport = ViewportState()

# Connect viewers
viewer1.connect_viewport(viewport)
viewer2.connect_viewport(viewport)

# Changing one viewer updates all connected viewers
viewport.set_limits(0, 100, 0, 2000)
```

---

### GatherNavigator

Manages ensemble/gather navigation with prefetching.

```python
from models import GatherNavigator
```

#### Constructor

```python
GatherNavigator(storage_dir: str, headers: pd.DataFrame,
                sort_keys: List[str] = None)
```

#### Methods

```python
get_current_gather() -> Tuple[np.ndarray, pd.DataFrame]
```

Get traces and headers for current gather.

```python
navigate_to(index: int) -> bool
```

Navigate to a specific gather index.

```python
next_gather() -> bool
previous_gather() -> bool
```

Navigate to next/previous gather.

---

## Processors

### BaseProcessor

Abstract base class for all processors.

```python
from processors import BaseProcessor, ProgressCallback
```

#### Type Aliases

```python
ProgressCallback = Callable[[int, int, str], None]
# Parameters: (current, total, message)
```

#### Methods

```python
set_progress_callback(callback: Optional[ProgressCallback]) -> BaseProcessor
```

Set callback for progress reporting. Returns self for chaining.

```python
process(data: SeismicData) -> SeismicData
```

Process seismic data (abstract - implemented by subclasses).

```python
get_description() -> str
```

Get human-readable processor description.

---

### BandpassFilter

Zero-phase Butterworth bandpass filter.

```python
from processors import BandpassFilter
```

#### Constructor

```python
BandpassFilter(low_freq: float, high_freq: float, order: int = 4)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `low_freq` | `float` | - | Low-cut frequency (Hz) |
| `high_freq` | `float` | - | High-cut frequency (Hz) |
| `order` | `int` | 4 | Butterworth filter order |

#### Example

```python
from processors import BandpassFilter

# Create processor
bp = BandpassFilter(low_freq=10.0, high_freq=60.0, order=4)

# Apply to data
filtered_data = bp.process(seismic_data)

print(bp.get_description())
# Output: "Bandpass 10.0-60.0 Hz (order 4)"
```

---

### TFDenoise

Time-frequency domain denoising using S-Transform.

```python
from processors import TFDenoise
```

#### Constructor

```python
TFDenoise(aperture: int = 5, fmin: float = 5.0, fmax: float = 80.0,
          threshold_k: float = 3.0, threshold_type: str = 'soft')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aperture` | `int` | 5 | Number of traces in sliding window |
| `fmin` | `float` | 5.0 | Minimum frequency (Hz) |
| `fmax` | `float` | 80.0 | Maximum frequency (Hz) |
| `threshold_k` | `float` | 3.0 | MAD threshold multiplier |
| `threshold_type` | `str` | 'soft' | 'soft' or 'hard' thresholding |

#### Example

```python
from processors import TFDenoise

# Create processor with progress callback
def on_progress(current, total, message):
    print(f"Progress: {current}/{total} - {message}")

processor = TFDenoise(aperture=5, fmin=5.0, fmax=80.0, threshold_k=3.0)
processor.set_progress_callback(on_progress)

# Apply denoising
denoised_data = processor.process(seismic_data)
```

---

### FKFilter

Frequency-wavenumber (F-K) domain velocity filter.

```python
from processors import FKFilter
```

#### Constructor

```python
FKFilter(v_min: float, v_max: float, taper: float = 0.1,
         mode: str = 'reject')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `v_min` | `float` | - | Minimum velocity (m/s) |
| `v_max` | `float` | - | Maximum velocity (m/s) |
| `taper` | `float` | 0.1 | Transition zone width (fraction) |
| `mode` | `str` | 'reject' | 'pass' or 'reject' |

#### Example

```python
from processors import FKFilter

# Reject linear noise with apparent velocity 1500-2500 m/s
fk = FKFilter(v_min=1500, v_max=2500, taper=0.1, mode='reject')
filtered_data = fk.process(seismic_data)
```

---

## SEG-Y Import/Export

### SEGYReader

Read SEG-Y files with custom header mapping.

```python
from utils.segy_import import SEGYReader, HeaderMapping
```

#### Constructor

```python
SEGYReader(filename: str, mapping: HeaderMapping)
```

#### Methods

```python
read_file_info() -> Dict[str, Any]
```

Read file metadata without loading traces.

```python
read_traces() -> Tuple[np.ndarray, List[Dict]]
```

Read all traces and headers.

```python
read_traces_in_chunks(chunk_size: int = 100,
                      cancellation_token: CancellationToken = None,
                      progress_callback: Callable = None) -> Iterator
```

Read traces in memory-efficient chunks.

```python
open() -> SEGYFileHandle
```

Get a file handle for batch operations (context manager).

#### Example

```python
from utils.segy_import import SEGYReader, HeaderMapping

# Create reader
mapping = HeaderMapping()
reader = SEGYReader("/path/to/file.sgy", mapping)

# Read file info
info = reader.read_file_info()
print(f"Traces: {info['n_traces']}, Samples: {info['n_samples']}")

# Read all data
traces, headers = reader.read_traces()

# Or use file handle for batch operations
with reader.open() as handle:
    info = handle.read_file_info()
    batch1 = handle.read_traces_range(0, 100)
    batch2 = handle.read_traces_range(100, 200)
```

---

### SEGYExporter

Export processed data to SEG-Y format.

```python
from utils.segy_import import SEGYExporter, AsyncSEGYExporter
```

#### SEGYExporter

```python
exporter = SEGYExporter(output_path: str)
exporter.export(original_segy_path: str, data: SeismicData)
```

#### AsyncSEGYExporter

Double-buffered asynchronous export for large files.

```python
async_exporter = AsyncSEGYExporter(output_path: str, num_workers: int = 2)
async_exporter.export_from_zarr_async(
    original_segy_path: str,
    processed_zarr_path: str,
    chunk_size: int = 1000,
    progress_callback: Callable = None
)
```

#### Example

```python
from utils.segy_import import SEGYExporter

exporter = SEGYExporter("/path/to/output.sgy")
exporter.export("/path/to/original.sgy", processed_data)
```

---

### HeaderMapping

Configure trace header field mappings.

```python
from utils.segy_import import HeaderMapping, StandardHeaders
```

#### StandardHeaders Enum

| Constant | SEG-Y Byte | Description |
|----------|------------|-------------|
| `TRACE_SEQ_LINE` | 1-4 | Trace sequence in line |
| `TRACE_SEQ_FILE` | 5-8 | Trace sequence in file |
| `FIELD_RECORD` | 9-12 | Field record number |
| `CDP` | 21-24 | CDP number |
| `OFFSET` | 37-40 | Source-receiver offset |
| `SOURCE_X` | 73-76 | Source X coordinate |
| `SOURCE_Y` | 77-80 | Source Y coordinate |
| `RECEIVER_X` | 81-84 | Receiver X coordinate |
| `RECEIVER_Y` | 85-88 | Receiver Y coordinate |

#### Example

```python
from utils.segy_import import HeaderMapping, StandardHeaders

mapping = HeaderMapping()
mapping.add_field("CDP", StandardHeaders.CDP)
mapping.add_field("OFFSET", StandardHeaders.OFFSET)
```

---

## GPU Acceleration

### DeviceManager

Manage GPU device selection and memory.

```python
from processors.gpu import DeviceManager, get_device_manager
```

#### Functions

```python
get_device_manager() -> DeviceManager
```

Get singleton device manager instance.

#### DeviceManager Methods

```python
get_device() -> torch.device
```

Get optimal device (CUDA > MPS > CPU).

```python
is_gpu_available() -> bool
```

Check if any GPU is available.

```python
get_memory_info() -> Dict[str, float]
```

Get GPU memory usage (allocated, reserved, total).

---

### TFDenoiseGPU

GPU-accelerated time-frequency denoising.

```python
from processors.tf_denoise_gpu import TFDenoiseGPU
```

#### Constructor

```python
TFDenoiseGPU(aperture: int = 5, fmin: float = 5.0, fmax: float = 80.0,
             threshold_k: float = 3.0, use_gpu: str = 'auto')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gpu` | `str` | 'auto' | 'auto', 'force', or 'never' |

#### Automatic Fallback

The GPU processor automatically falls back to CPU when:
- No GPU is available
- GPU runs out of memory (with retry logic)
- CUDA/MPS errors occur

#### Example

```python
from processors.tf_denoise_gpu import TFDenoiseGPU

# Auto-detect GPU
processor = TFDenoiseGPU(
    aperture=5,
    fmin=5.0,
    fmax=80.0,
    threshold_k=3.0,
    use_gpu='auto'
)

result = processor.process(seismic_data)
print(processor.get_description())
# Output: "TF-Denoise [GPU-CUDA] aperture=5, k=3.0"
```

---

## Utilities

### ThemeManager

Manage application light/dark theme.

```python
from utils import ThemeManager, get_theme_manager, ThemeType
```

#### ThemeType Enum

```python
ThemeType.LIGHT
ThemeType.DARK
```

#### Methods

```python
get_theme_manager() -> ThemeManager
```

Get singleton theme manager.

```python
theme_manager.set_theme(theme: ThemeType)
```

Apply theme to application.

```python
theme_manager.get_theme() -> ThemeType
```

Get current theme.

```python
theme_manager.apply_to_app(app: QApplication)
```

Apply current theme to QApplication.

#### Example

```python
from utils import get_theme_manager, ThemeType

theme = get_theme_manager()
theme.set_theme(ThemeType.DARK)
```

---

### MemoryMonitor

Monitor application memory usage.

```python
from utils import MemoryMonitor
```

#### Signals

| Signal | Parameters | Description |
|--------|------------|-------------|
| `memory_updated` | `dict` | Emitted with memory stats |
| `memory_warning` | `float` | Emitted when usage > 80% |
| `memory_critical` | `float` | Emitted when usage > 95% |

#### Methods

```python
start_monitoring(interval_ms: int = 1000)
```

Start periodic memory monitoring.

```python
stop_monitoring()
```

Stop memory monitoring.

```python
get_memory_info() -> Dict[str, float]
```

Get current memory usage.

---

## Usage Examples

### Basic Processing Workflow

```python
from models import SeismicData
from processors import BandpassFilter, TFDenoise
from utils.segy_import import SEGYReader, SEGYExporter, HeaderMapping

# 1. Load SEG-Y file
mapping = HeaderMapping()
reader = SEGYReader("input.sgy", mapping)
traces, headers = reader.read_traces()
info = reader.read_file_info()

# 2. Create SeismicData
data = SeismicData(
    traces=traces,
    sample_rate=info['sample_interval']
)

# 3. Apply processing chain
bp = BandpassFilter(low_freq=10.0, high_freq=60.0)
tf = TFDenoise(aperture=5, threshold_k=3.0)

processed = bp.process(data)
processed = tf.process(processed)

# 4. Export result
exporter = SEGYExporter("output.sgy")
exporter.export("input.sgy", processed)
```

### GPU Processing with Progress

```python
from processors.tf_denoise_gpu import TFDenoiseGPU

def on_progress(current, total, message):
    pct = current / total * 100
    print(f"\r{message}: {pct:.1f}%", end="")

processor = TFDenoiseGPU(
    aperture=5,
    fmin=5.0,
    fmax=80.0,
    threshold_k=3.0,
    use_gpu='auto'
)
processor.set_progress_callback(on_progress)

result = processor.process(data)
print()  # Newline after progress
```

### Large File Processing with Lazy Loading

```python
from models import LazySeismicData
from processors import BandpassFilter

# Load lazily
lazy_data = LazySeismicData.from_storage_dir("/path/to/zarr")

# Process in chunks
bp = BandpassFilter(low_freq=10.0, high_freq=60.0)

for chunk in lazy_data.iterate_chunks(chunk_size=100):
    chunk_data = SeismicData(traces=chunk, sample_rate=2.0)
    processed_chunk = bp.process(chunk_data)
    # Save or accumulate results
```

---

## Error Handling

### CancellationToken

Cancel long-running operations.

```python
from utils.segy_import.segy_reader import CancellationToken, OperationCancelledError

token = CancellationToken()

# In another thread
token.cancel()

# In reader
try:
    for traces, headers in reader.read_traces_in_chunks(
        chunk_size=100,
        cancellation_token=token
    ):
        process(traces)
except OperationCancelledError:
    print("Operation was cancelled")
```

---

## Performance Tips

1. **Use GPU when available**: `TFDenoiseGPU` with `use_gpu='auto'` provides ~9x speedup
2. **Chunk large files**: Use `read_traces_in_chunks()` for memory efficiency
3. **Use connection pooling**: `reader.open()` context manager for batch operations
4. **Enable caching**: S-Transform windows are automatically cached (up to 5 configs)
5. **Use async export**: `AsyncSEGYExporter` for ~2x faster large file writes

---

*Generated for SeisProc v1.0.0 | Last updated: 2025-12-06*
