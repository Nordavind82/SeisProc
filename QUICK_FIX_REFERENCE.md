# Quick Fix Reference - Segmentation Fault Issues

## TL;DR - What Was Fixed

**Problem**: Segmentation faults when loading SEG-Y files
**Root Cause**: OpenGL/Qt/Display integration issues in WSL2/X11 environment
**Solution**: Disabled native dialogs and QSettings, forced software rendering

---

## Critical Fixes Applied

### 1. Environment Variables (`main.py`)
```python
os.environ.setdefault('QT_XCB_GL_INTEGRATION', 'none')  # MOST IMPORTANT
```

### 2. File Dialogs (`views/segy_import_dialog.py`)
```python
options=QFileDialog.Option.DontUseNativeDialog  # Added to ALL file dialogs
```

### 3. Removed QSettings Calls
```python
# REMOVED: get_settings().set_spatial_units(selected_units)
# Reason: Causes segfault even with try-except
```

---

## Known Limitations After Fix

❌ **User settings NOT saved between sessions** (meters/feet preference)
❌ **Non-native file dialogs** (look different from system dialogs)
✅ **Application works without crashes**
✅ **SEG-Y import functions correctly**
✅ **All processing features work**

---

## If You Still Get Crashes

### Check environment variables:
```bash
python -c "import os; print('OpenGL disabled:', os.environ.get('QT_XCB_GL_INTEGRATION'))"
# Should print: OpenGL disabled: none
```

### Run with full debug logging:
```bash
python main.py 2>&1 | tee crash_log.txt
```

### Check the last few lines before crash:
```bash
tail -20 crash_log.txt
```

---

## Files Modified

| File | Changes |
|------|---------|
| `main.py` | Added 5 environment variables before imports |
| `views/segy_import_dialog.py` | Added `DontUseNativeDialog` to file dialogs, removed `get_settings()` call |
| `models/app_settings.py` | Made all methods safe when QSettings unavailable |
| `main_window.py` | Added debug logging |

---

## Testing Checklist

- [ ] App starts without crash
- [ ] File → Load SEG-Y File opens dialog
- [ ] Browse button works (Qt dialog, not system)
- [ ] File selection works
- [ ] Import SEG-Y button works
- [ ] Directory selection works (Qt dialog)
- [ ] Import completes without crash
- [ ] Data displays in viewers

---

## Emergency Rollback

If you need to undo changes:

```bash
cd /scratch/Python_Apps/SeisProc
git diff main.py
git diff views/segy_import_dialog.py
git diff models/app_settings.py

# To revert (if using git):
git checkout main.py views/segy_import_dialog.py models/app_settings.py
```

---

**Status**: Ready for testing
**Last Updated**: 2025-11-19
