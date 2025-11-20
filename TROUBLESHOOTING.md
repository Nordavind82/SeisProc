# Troubleshooting Guide

## Segmentation Fault on Startup

### Problem
When running `python main.py`, the application crashes with:
```
Segmentation fault (core dumped)
```

### Root Cause
This is caused by OpenGL integration issues between PyQt6 and the X11 display server. The NVIDIA GPU/driver may attempt to use OpenGL rendering which conflicts with Qt's default configuration.

### Solution ✅ (FIXED)

The issue has been **fixed in main.py**. The application now automatically sets the correct environment variables.

If you still experience issues, you can:

#### Option 1: Use the Launcher Script (Recommended)
```bash
./run_app.sh
```

#### Option 2: Set Environment Variables Manually
```bash
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none
export MPLBACKEND=Agg
python main.py
```

#### Option 3: Add to Your Shell Profile
Add these lines to `~/.bashrc` or `~/.bash_profile`:
```bash
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none
export MPLBACKEND=Agg
```

Then reload: `source ~/.bashrc`

---

## Environment Variable Explanations

| Variable | Value | Purpose |
|----------|-------|---------|
| `QT_QPA_PLATFORM` | `xcb` | Forces Qt to use X11 backend (prevents Wayland issues) |
| `QT_XCB_GL_INTEGRATION` | `none` | **Critical**: Disables OpenGL integration (fixes segfault) |
| `MPLBACKEND` | `Agg` | Uses non-interactive matplotlib backend (prevents conflicts) |

---

## Other Common Issues

### Display Issues on WSL2
If running on Windows Subsystem for Linux 2:

1. Install an X11 server on Windows (VcXsrv or Xming)
2. Set display variable:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
   ```

### "Cannot connect to X server"
```bash
# Check if X11 is running
echo $DISPLAY

# Test X11 connection
xeyes  # Should show a window with eyes

# If not working, set DISPLAY:
export DISPLAY=:0
```

### Missing Dependencies
```bash
# If you get "No module named 'PyQt6'":
source venv/bin/activate
pip install -r requirements.txt
```

### GPU/CUDA Warnings
If you see CUDA warnings but the app works, this is normal. The app will use CPU for processing if GPU is unavailable.

To suppress warnings:
```bash
export CUDA_VISIBLE_DEVICES=""  # Disable CUDA entirely
```

### Slow Performance
If the GUI is slow:

1. Check GPU acceleration:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Reduce data size or use lazy loading mode

3. Close other heavy applications

---

## Debug Mode

To run in debug mode with detailed logging:
```bash
python main_debug.py
```

This creates a `debug.log` file with step-by-step initialization info.

---

## System Information

To help diagnose issues, collect this info:

```bash
# Python version
python --version

# Qt platform plugin info
python -c "from PyQt6.QtCore import QLibraryInfo; print(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))"

# Display server
echo $DISPLAY
echo $XDG_SESSION_TYPE  # x11 or wayland

# GPU info
nvidia-smi  # If NVIDIA GPU

# OS info
uname -a
cat /etc/os-release
```

---

## Getting Help

If issues persist:

1. Run `python main_debug.py` and check `debug.log`
2. Collect system information (see above)
3. Check if the issue occurs with sample data (File → Generate Sample Data)
4. Try running with different environment variables:
   ```bash
   QT_DEBUG_PLUGINS=1 python main.py 2>&1 | tee qt_debug.log
   ```

---

## Verified Working Configurations

✅ **Tested and working:**
- Fedora Linux 6.12.0
- Python 3.12.9
- PyQt6 6.10.0
- NVIDIA GeForce RTX 4060
- X11 display server

The application is confirmed working with the fixes applied to `main.py`.
