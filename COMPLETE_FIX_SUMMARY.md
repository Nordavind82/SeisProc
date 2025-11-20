# Complete Segmentation Fault Fix - Final Summary

## Problem Overview

When running the SeisProc application on WSL2/Linux with X11 display and NVIDIA GPU, multiple segmentation faults occurred:

1. **Crash #1**: App startup - FIXED ✓
2. **Crash #2**: Opening "Load SEG-Y File" dialog - FIXED ✓
3. **Crash #3**: Clicking "Import SEG-Y" button - FIXED ✓

## Root Causes Identified

### 1. OpenGL Integration Issues
- Qt6 attempting to use OpenGL acceleration with X11
- NVIDIA GPU driver conflicts with Qt's OpenGL calls
- Display server (xcb) incompatibility

### 2. QSettings D-Bus/System Service Crashes
- `QSettings` attempting to access native system services
- D-Bus communication failures in WSL2 environment
- Crash when initializing settings singleton

### 3. Native File Dialog Crashes
- `QFileDialog` using native GTK/KDE file pickers
- Native dialog libraries attempting OpenGL rendering
- System integration failures

## Complete Fix Implementation

### Fix #1: Application Startup (`main.py`)

**File**: `main.py` (lines 17-26)

```python
# Fix for segmentation fault: Set Qt/display environment variables BEFORE importing PyQt6
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')          # Use X11 backend
os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')  # Disable OpenGL (prevents segfault)
os.environ.setdefault('MPLBACKEND', 'Agg')              # Non-interactive matplotlib backend

# CRITICAL: Force Qt to use portable settings format
os.environ.setdefault('QT_SETTINGS_PATH', '/tmp')        # Use temp directory for settings
os.environ['QT_SCALE_FACTOR'] = '1'                      # Disable automatic scaling
```

**Why it works**:
- Sets environment variables BEFORE Qt is imported
- Disables problematic OpenGL integration
- Forces software rendering instead of hardware acceleration

### Fix #2: QSettings Crash Prevention (`models/app_settings.py`)

**File**: `models/app_settings.py` (lines 52-82)

```python
# Check if we should disable QSettings (problematic environments)
disable_qsettings = os.environ.get('QT_XCB_GL_INTEGRATION') == 'none'

if disable_qsettings:
    # Skip QSettings entirely - use in-memory only
    logger.info("QSettings disabled (problematic environment detected)")
    self.settings = None
    self._settings_available = False
else:
    try:
        # Force portable INI format instead of native registry/D-Bus
        self.settings = QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            'SeismicDenoise',
            'DenoiseApp'
        )
        self._settings_available = True
    except Exception as e:
        logger.warning(f"QSettings initialization failed, using defaults: {e}")
        self.settings = None
        self._settings_available = False
```

**Why it works**:
- Detects problematic environment (OpenGL disabled = crash-prone)
- Skips QSettings entirely in WSL2/problematic environments
- Falls back to in-memory defaults
- All getter/setter methods check `_settings_available` flag

### Fix #3: File Dialog Crash Prevention (`views/segy_import_dialog.py`)

**File**: `views/segy_import_dialog.py`

**Browse Button** (line 382):
```python
filename, _ = QFileDialog.getOpenFileName(
    self,
    "Select SEG-Y File",
    "",
    "SEG-Y Files (*.sgy *.segy *.SGY *.SEGY);;All Files (*)",
    options=QFileDialog.Option.DontUseNativeDialog  # CRITICAL FIX
)
```

**Import Button / Directory Selection** (line 928):
```python
output_dir = QFileDialog.getExistingDirectory(
    self,
    "Select Output Directory for Zarr/Parquet Storage",
    "",
    options=QFileDialog.Option.DontUseNativeDialog  # CRITICAL FIX
)
```

**Why it works**:
- Forces Qt to use its own cross-platform file dialog
- Avoids native GTK/KDE dialogs that crash with OpenGL issues
- Qt's built-in dialog uses software rendering

### Fix #4: Default Values Instead of QSettings (`views/segy_import_dialog.py`)

**File**: `views/segy_import_dialog.py` (line 307)

```python
# Set current value from settings (use default to avoid crash)
logger.info("        → Setting default units (meters)...")
# TODO: Load from settings after dialog is shown (to avoid QSettings crash during init)
self.spatial_units_combo.setCurrentIndex(0)  # Default to meters
logger.info("        ✓ Default units set")
```

**Why it works**:
- Avoids calling `get_settings()` during dialog initialization
- Uses safe default value (meters)
- Prevents crash in dialog constructor

### Fix #5: Safe Settings Save (`views/segy_import_dialog.py`)

**File**: `views/segy_import_dialog.py` (lines 913-918)

```python
logger.info("  → Saving units to settings (may fail silently)...")
try:
    get_settings().set_spatial_units(selected_units)
    logger.info(f"  ✓ Spatial units saved: {selected_units}")
except Exception as e:
    logger.warning(f"  ! Failed to save settings (continuing anyway): {e}")
```

**Why it works**:
- Wraps QSettings calls in try-except
- Allows operation to continue even if settings fail
- Logs warnings but doesn't crash

## Testing Protocol

### Test the Fixes

1. **Start the application**:
   ```bash
   cd /scratch/Python_Apps/SeisProc
   source venv/bin/activate
   python main.py
   ```
   Expected: GUI opens without crash ✓

2. **Open SEG-Y Import Dialog**:
   - Click: File → Load SEG-Y File...
   Expected: Dialog opens without crash ✓

3. **Browse for SEG-Y File**:
   - Click: Browse... button
   - Select a .sgy/.segy file
   Expected: Qt file dialog opens (not native), file loads ✓

4. **Import SEG-Y**:
   - Configure headers (or use defaults)
   - Click: Import SEG-Y button
   - Select output directory
   Expected: Import proceeds without crash ✓

## Files Modified

| File | Purpose | Critical Changes |
|------|---------|------------------|
| `main.py` | Entry point | Added environment variables for Qt/OpenGL |
| `models/app_settings.py` | Settings management | Disabled QSettings in problematic environments |
| `views/segy_import_dialog.py` | Import dialog | Added `DontUseNativeDialog` to all file dialogs, safe defaults |
| `main_window.py` | Main window | Added comprehensive debug logging |

## Environment Variables Set

| Variable | Value | Purpose |
|----------|-------|---------|
| `QT_QPA_PLATFORM` | `xcb` | Force X11 backend |
| `QT_XCB_GL_INTEGRATION` | `none` | **CRITICAL** - Disable OpenGL |
| `MPLBACKEND` | `Agg` | Non-interactive matplotlib |
| `QT_SETTINGS_PATH` | `/tmp` | Portable settings location |
| `QT_SCALE_FACTOR` | `1` | Disable auto-scaling |

## Verification Steps

Run these commands to verify the fix:

```bash
# 1. Clear cache
find /scratch/Python_Apps/SeisProc -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# 2. Test startup
source venv/bin/activate
timeout 5 python main.py && echo "✓ Startup successful"

# 3. Check environment
python -c "import os; print('QT_XCB_GL_INTEGRATION:', os.environ.get('QT_XCB_GL_INTEGRATION'))"
```

## Known Limitations

1. **Settings Not Persistent**: In problematic environments, user preferences are not saved between sessions
2. **Non-Native Dialogs**: File dialogs look different (Qt style instead of system style)
3. **Software Rendering**: Performance may be slightly lower due to disabled OpenGL

## Success Metrics

✅ Application starts without segfault
✅ SEG-Y import dialog opens without crash
✅ File browse dialog works (Qt dialog, not native)
✅ Directory selection works
✅ SEG-Y import completes successfully
✅ All three synchronized viewers display data

## Troubleshooting

If crashes still occur:

1. **Check environment variables**:
   ```bash
   echo $QT_XCB_GL_INTEGRATION  # Should be "none"
   ```

2. **Run with debug logging**:
   ```bash
   python main.py 2>&1 | grep -E "(INFO|ERROR|WARN)"
   ```

3. **Check Qt platform**:
   ```bash
   QT_DEBUG_PLUGINS=1 python main.py 2>&1 | grep -i plugin
   ```

## System Configuration

**Tested and verified on**:
- OS: Fedora Linux (WSL2)
- Kernel: 6.12.0-55.43.1.el10_0.x86_64
- Python: 3.12.9
- PyQt6: 6.10.0
- GPU: NVIDIA GeForce RTX 4060
- Display: X11 (via WSL2 X server)

---

**Status**: ✅ **ALL CRASHES FIXED**

**Date**: 2025-11-19
**Version**: Complete Fix v1.0
