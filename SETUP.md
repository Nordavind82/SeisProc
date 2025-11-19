# Denoise App Setup Guide

## Overview

This application is a self-contained seismic data processing and QC tool with support for GPU acceleration, time-frequency denoising, and efficient handling of large SEG-Y datasets.

---

## Quick Start (Recommended)

### Automated Setup with Virtual Environment

The easiest way to set up the environment is using the provided setup script:

```bash
# Run the automated setup script
./setup_env.sh
```

This will:
1. Create a Python virtual environment in `venv/`
2. Install all required dependencies from `requirements.txt`
3. Display installation status and instructions

### Manual Setup

If you prefer manual setup or need more control:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Activating the Environment

Before running the application, always activate the virtual environment:

```bash
source venv/bin/activate
```

Your prompt will change to show `(venv)` indicating the environment is active.

### Running the Application

With the virtual environment activated:

```bash
python main.py
```

### Deactivating the Environment

When you're done:

```bash
deactivate
```

---

## Directory Structure

```
denoise_app/
├── main_window.py              # Main application entry point
├── models/                     # Data models and state management
│   ├── __init__.py
│   ├── seismic_data.py         # In-memory seismic data structure
│   ├── lazy_seismic_data.py    # Memory-efficient lazy loading
│   ├── gather_navigator.py     # Gather navigation and caching
│   └── viewport_state.py       # Viewport state management
├── views/                      # UI components
│   ├── seismic_viewer_pyqtgraph.py
│   ├── control_panel.py
│   ├── gather_navigation_panel.py
│   ├── segy_import_dialog.py
│   └── flip_window.py
├── processors/                 # Signal processing modules
│   ├── base_processor.py
│   ├── bandpass_filter.py
│   ├── gain_processor.py
│   ├── tf_denoise.py
│   ├── tf_denoise_gpu.py
│   └── chunked_processor.py
├── utils/                      # Utility functions
│   ├── sample_data.py
│   ├── memory_monitor.py
│   ├── window_cache.py
│   └── segy_import/            # SEG-Y import functionality
│       ├── __init__.py
│       ├── header_mapping.py   # Header configuration
│       ├── segy_reader.py      # SEG-Y file reader
│       ├── segy_export.py      # SEG-Y file writer
│       └── data_storage.py     # Zarr/Parquet storage
└── docs/                       # Documentation
    ├── HEADER_MAPPING_GUIDE.md
    └── example_header_mapping.json
```

---

## Running the Application

### Standard Method (Recommended):

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the application
python main.py
```

### Alternative Methods:

```bash
# Direct execution with Python 3
python3 main.py

# Or execute main_window.py directly
python3 main_window.py
```

---

## Import Structure

All imports now use **relative imports within the package**:

### Example imports in main_window.py:

```python
from models.seismic_data import SeismicData
from models.lazy_seismic_data import LazySeismicData
from models.gather_navigator import GatherNavigator
from views.seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from utils.segy_import.data_storage import DataStorage
```

### No more sys.path.insert needed!

All `sys.path.insert(0, '/Users/olegadamovich/seismic_qc_app')` lines have been **removed**.

---

## Python Path Setup

The application works because:

1. **Current directory is automatically in Python path**
   - When you run `python3 main_window.py` from denoise_app, the current directory (denoise_app) is added to sys.path

2. **Package structure with __init__.py**
   - All directories have `__init__.py` files
   - Python treats them as packages
   - Imports work as `from models.X import Y`

---

## Testing Imports

To verify all imports work correctly:

```bash
cd /Users/olegadamovich/denoise_app

python3 -c "
from models.seismic_data import SeismicData
from models.lazy_seismic_data import LazySeismicData
from models.gather_navigator import GatherNavigator
from utils.segy_import.data_storage import DataStorage
from utils.segy_import.header_mapping import HeaderMapping
print('✅ All imports successful!')
"
```

Expected output:
```
✅ All imports successful!
```

---

## Key Features

### 1. **Single-Pass SEG-Y Import**
- Reads file once, processes all data simultaneously
- 3x faster than multi-pass approach
- Memory-efficient streaming

### 2. **Lazy Loading**
- Load massive datasets without memory issues
- Only loads data on-demand
- Constant ~70-100 MB memory usage

### 3. **Header Mapping**
- Save/Load header configurations
- Format selection (2i, 4i, 4r, etc.)
- Reusable across similar files

### 4. **GPU Acceleration** (optional)
- Time-frequency denoising on GPU
- Requires CUDA-capable GPU
- Falls back to CPU if unavailable

---

## Dependencies

All dependencies are managed through `requirements.txt` and installed automatically by the setup script. The main dependencies are:

### Core GUI Framework
- **PyQt6** (>=6.4.0) - Application GUI framework

### Scientific Computing & Data Processing
- **NumPy** (>=1.24.0) - Numerical computing
- **SciPy** (>=1.10.0) - Scientific computing and signal processing
- **Pandas** (>=2.0.0) - Data manipulation and analysis

### Deep Learning & GPU Acceleration
- **PyTorch** (>=2.0.0) - GPU acceleration for time-frequency processing

### Visualization
- **pyqtgraph** (>=0.13.0) - Real-time seismic data visualization
- **matplotlib** (>=3.7.0) - Plotting and visualization

### Seismic Data Handling
- **segyio** (>=1.9.0) - SEG-Y file reading/writing
- **zarr** (>=2.14.0) - Chunked, compressed array storage
- **numcodecs** (>=0.11.0) - Data compression
- **pyarrow** (>=12.0.0) - Parquet format for header storage

### System Monitoring
- **psutil** (>=5.9.0) - Memory and system monitoring

### Installing Dependencies Manually

If you need to install dependencies manually (not recommended):

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install PyQt6 numpy scipy pandas torch pyqtgraph matplotlib segyio zarr numcodecs pyarrow psutil
```

---

## Configuration Files

### Header Mapping Files (.json)

Save custom header configurations for reuse:

**Location:** `example_header_mapping.json`

**Format:**
```json
{
  "fields": {
    "cdp": {
      "name": "cdp",
      "byte_position": 21,
      "format": "i",
      "description": "CDP ensemble number"
    }
  },
  "ensemble_keys": ["cdp"]
}
```

**Usage:**
1. Open SEG-Y Import Dialog
2. Click "Load Mapping..."
3. Select JSON file
4. All headers auto-configured!

---

## Troubleshooting

### Virtual Environment Not Activated

**Symptom:** Module import errors or wrong Python version

**Solution:** Activate the virtual environment:
```bash
source venv/bin/activate
```

Verify activation by checking your prompt for `(venv)` prefix.

### Import Error: "No module named 'PyQt6'" or other packages

**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Permission Denied when running setup_env.sh

**Solution:** Make the script executable:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Application hangs when loading large SEG-Y

**Solution:** This should be fixed with lazy loading. If it still happens:
1. Check that LazySeismicData is being used (not SeismicData)
2. Verify in main_window.py line 340: `LazySeismicData.from_storage_dir()`
3. Check memory usage - should stay under 200 MB

### "ModuleNotFoundError" for segyio, zarr, etc.

**Solution:** Install missing packages:
```bash
pip install segyio zarr pandas pyarrow numpy scipy pyqtgraph
```

---

## Development

### Adding New Processors

1. Create new file in `processors/` directory
2. Inherit from `BaseProcessor`
3. Implement `process()` method
4. Import in control_panel.py

**Example:**
```python
# processors/my_new_processor.py
from processors.base_processor import BaseProcessor

class MyNewProcessor(BaseProcessor):
    def process(self, data):
        # Your processing logic
        return processed_data
```

### Adding New View Components

1. Create new file in `views/` directory
2. Inherit from appropriate PyQt6 widget
3. Import in main_window.py

---

## File Locations

| File Type | Location | Purpose |
|-----------|----------|---------|
| **Source Code** | `/Users/olegadamovich/denoise_app/` | All application code |
| **Header Mappings** | `/Users/olegadamovich/denoise_app/*.json` | Saved header configs |
| **Documentation** | `/Users/olegadamovich/denoise_app/docs/` | User guides |
| **Zarr Data** | User-specified output directory | Imported SEG-Y data |

---

## Quick Start Checklist

- [ ] Navigate to project directory
- [ ] Run `./setup_env.sh` to set up virtual environment
- [ ] Activate environment: `source venv/bin/activate`
- [ ] Run `python main.py`
- [ ] Test import: File → Import SEG-Y
- [ ] Load header mapping (optional)
- [ ] Import small test file first
- [ ] Verify lazy loading is working (low memory usage)
- [ ] Deactivate when done: `deactivate`

---

## Performance Tips

### 1. Import Large Files
- Use streaming import (automatic for large files)
- Expect 3x speedup with single-pass import
- Memory stays constant at ~30-50 MB during import

### 2. Viewing Large Datasets
- Lazy loading automatically enabled
- Only one gather loaded at a time
- Navigate freely without memory concerns

### 3. Processing
- Use chunked processors for large gathers
- GPU acceleration for time-frequency denoising
- Monitor memory usage in Activity Monitor

---

## Architecture Highlights

### Memory Efficiency

**Old Approach (broken):**
```
Load 100,000 traces → 850 MB RAM → Desktop hangs
```

**New Approach (optimized):**
```
Load metadata → 10 MB RAM
Navigate to gather → 20 MB RAM (one gather)
Total: ~70-100 MB RAM regardless of file size
```

### Import Pipeline

**Old (3 passes):**
```
Pass 1: Read SEG-Y → Write Zarr
Pass 2: Read SEG-Y → Write Headers
Pass 3: Read SEG-Y → Detect Ensembles
Total I/O: 3x file size
```

**New (1 pass):**
```
Read SEG-Y once → {
    Write Zarr,
    Write Headers,
    Detect Ensembles
} simultaneously
Total I/O: 1x file size (3x faster!)
```

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Verify imports work (see "Testing Imports" above)
3. Review HEADER_MAPPING_GUIDE.md for SEG-Y import help
4. Check memory usage in Activity Monitor

---

## Summary

✅ **Self-contained** - No external dependencies on seismic_qc_app
✅ **Memory-efficient** - Lazy loading for large datasets
✅ **Fast imports** - Single-pass streaming (3x faster)
✅ **Reusable configs** - Save/Load header mappings
✅ **Format selection** - Proper SEG-Y data type handling

**Ready to use!** 🚀
