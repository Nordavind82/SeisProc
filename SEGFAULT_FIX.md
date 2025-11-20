# Segmentation Fault Fix - Summary

## Issue
When running `python main.py`, the application crashed with:
```
Segmentation fault (core dumped)
```

## Root Cause Analysis

Using systematic debugging (`main_debug.py`), the crash was traced to **OpenGL integration issues** between:
- PyQt6 GUI framework
- X11 display server (xcb)
- NVIDIA GPU driver attempting OpenGL rendering

The crash occurred during Qt widget initialization when OpenGL acceleration was attempted.

## The Fix

### What Was Changed

**File: `main.py`** (lines 14-21)

Added environment variable configuration **before** importing PyQt6:

```python
import os
import sys

# Fix for segmentation fault: Set Qt/display environment variables BEFORE importing PyQt6
# This prevents OpenGL integration issues that cause crashes on some systems
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')          # Use X11 backend
os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')  # Disable OpenGL (prevents segfault)
os.environ.setdefault('MPLBACKEND', 'Agg')              # Non-interactive matplotlib backend

from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
```

### Why This Works

1. **`QT_QPA_PLATFORM=xcb`**: Forces Qt to use the X11 backend explicitly
2. **`QT_XCB_GL_INTEGRATION=none`**: **CRITICAL** - Disables OpenGL integration, preventing the segfault
3. **`MPLBACKEND=Agg`**: Uses non-interactive matplotlib backend to avoid conflicts

### Using `setdefault()` Instead of Direct Assignment

The fix uses `os.environ.setdefault()` instead of direct assignment:
- Allows users to override if needed via shell environment
- Provides defaults for systems that need them
- Non-intrusive for systems that don't have the issue

## Files Created

### 1. `main_debug.py`
Debug version with step-by-step logging to trace initialization:
```bash
python main_debug.py  # Creates debug.log
```

### 2. `run_app.sh`
Convenient launcher script:
```bash
./run_app.sh
```

### 3. `TROUBLESHOOTING.md`
Comprehensive troubleshooting guide for common issues

### 4. `SEGFAULT_FIX.md` (this file)
Technical documentation of the fix

## Verification

### Before Fix:
```bash
$ python main.py
Segmentation fault (core dumped)
```

### After Fix:
```bash
$ python main.py
# Application starts successfully ✅
# GUI displays without crash ✅
# Can load SEG-Y files ✅
```

## Technical Details

### System Configuration Where Fix Was Tested:
- **OS**: Fedora Linux (Kernel 6.12.0-55.43.1.el10_0.x86_64)
- **Python**: 3.12.9
- **PyQt6**: 6.10.0
- **GPU**: NVIDIA GeForce RTX 4060
- **Display Server**: X11 (xcb)
- **WSL**: Yes (WSL2 on Windows)

### Debug Log Output (Success):
```
Step 6: Creating QApplication instance...
  ✓ QApplication created successfully
Step 7: Setting application metadata...
  ✓ Application metadata set
Step 8: Importing MainWindow...
  ✓ MainWindow imported successfully
Step 9: Creating MainWindow instance...
GPU Device Manager initialized: NVIDIA GeForce RTX 4060
  ✓ MainWindow created successfully
Step 10: Showing MainWindow...
  ✓ MainWindow shown successfully
Step 11: Starting event loop...
Application is running. Use Ctrl+C to exit.
```

## Alternative Solutions (if issue persists)

If the fix in `main.py` doesn't work for your system:

### Option 1: Shell Script Wrapper
```bash
./run_app.sh
```

### Option 2: Manual Environment Variables
```bash
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none
export MPLBACKEND=Agg
python main.py
```

### Option 3: Permanent Shell Configuration
Add to `~/.bashrc`:
```bash
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none
```

## References

- PyQt6 Platform Plugins: https://doc.qt.io/qt-6/qpa.html
- Qt XCB Integration: https://doc.qt.io/qt-6/linux.html
- Similar issues: Qt + NVIDIA + X11 OpenGL conflicts

## Status

✅ **RESOLVED** - Application now starts successfully on the target system.

---

*Fix implemented: 2025-11-19*
*Tested on: Fedora Linux / WSL2 / Python 3.12.9 / PyQt6 6.10.0*
