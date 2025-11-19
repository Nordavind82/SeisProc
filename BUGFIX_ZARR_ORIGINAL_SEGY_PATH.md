# Bug Fix: Original SEG-Y Path for Zarr-Loaded Data

## Issue

When loading data from a previously imported Zarr directory, the batch processing and export features failed with the error:

```
Export requires the original SEG-Y file for headers.
This feature only works with data loaded from SEG-Y.
```

Even though the data was originally imported from a SEG-Y file, the original path wasn't being preserved when loading from Zarr storage.

## Root Cause

The workflow was:

1. **Import SEG-Y** → Dialog sets `self.original_segy_path` → Data saved to Zarr
2. **Load from Zarr** → `self.original_segy_path` NOT set → Export fails ❌

The original SEG-Y path was not being saved in the Zarr metadata during import, so when loading from Zarr later, there was no way to know where the original file was located.

## Solution

### 1. Save Original Path in Metadata During Import

Modified **segy_import_dialog.py** to save the original SEG-Y file path in the metadata:

**Non-streaming import** (line 598-599):
```python
# Add original SEG-Y path to metadata for export functionality
seismic_data.metadata['original_segy_path'] = self.segy_file
```

**Streaming import** (line 679-680):
```python
'seismic_metadata': {
    'source_file': str(self.segy_file),
    'original_segy_path': str(self.segy_file),  # For export functionality
    'file_info': file_info,
    'header_mapping': self.header_mapping.to_dict(),
},
```

### 2. Restore Path When Loading from Zarr

Modified **main_window.py:_load_from_zarr()** to extract and restore the original path (lines 363-375):

```python
# Extract original SEG-Y path from metadata if available
original_segy_path = None
if hasattr(lazy_data, 'metadata') and lazy_data.metadata:
    # Try to get it from top-level metadata first (new format)
    original_segy_path = lazy_data.metadata.get('original_segy_path')

    # Fall back to seismic_metadata section (streaming import format)
    if not original_segy_path:
        seismic_meta = lazy_data.metadata.get('seismic_metadata', {})
        original_segy_path = seismic_meta.get('original_segy_path') or seismic_meta.get('source_file')

# Load into gather navigator and display
self._on_segy_imported(lazy_data, None, ensembles_df, original_segy_path)
```

### 3. Validate Path Still Exists

Added validation in **main_window.py** for both export functions (lines 822-851 and 927-964):

```python
# Validate that the original SEG-Y file still exists
if not Path(self.original_segy_path).exists():
    reply = QMessageBox.critical(
        self,
        "Original SEG-Y File Not Found",
        f"The original SEG-Y file is required for export but cannot be found:\n\n"
        f"{self.original_segy_path}\n\n"
        f"This file may have been moved or deleted.\n\n"
        f"Options:\n"
        f"• Move the file back to its original location\n"
        f"• Re-import the SEG-Y data from its current location\n\n"
        f"Do you want to browse for the file?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        # Let user browse for the file
        new_path, _ = QFileDialog.getOpenFileName(...)
        if new_path and Path(new_path).exists():
            self.original_segy_path = new_path
```

### 4. Inform User on Load

Modified **main_window.py:_load_from_zarr()** to show path status when loading (lines 388-395):

```python
# Add note about original SEG-Y path availability
if original_segy_path:
    if Path(original_segy_path).exists():
        msg += f"\n\n✅ Original SEG-Y file found - batch processing available"
    else:
        msg += f"\n\n⚠️  Original SEG-Y file not found at:\n{original_segy_path}\n\nBatch processing will require you to locate the file."
else:
    msg += "\n\n⚠️  Original SEG-Y path not in metadata.\nTo enable batch processing, please re-import from SEG-Y."
```

## Files Modified

1. **views/segy_import_dialog.py**
   - Line 598-599: Add path to metadata in non-streaming import
   - Line 679-680: Add path to metadata in streaming import

2. **main_window.py**
   - Lines 363-375: Extract and restore original_segy_path when loading from Zarr
   - Lines 388-395: Show path status in load success message
   - Lines 810-851: Validate path for standard export with browse option
   - Lines 915-964: Validate path for memory-efficient export with browse option

## Behavior

### For New Imports (After Fix)

1. **Import SEG-Y** → Path saved in metadata → Data saved to Zarr ✅
2. **Load from Zarr** → Path restored from metadata → Export works! ✅

### For Old Zarr Data (Before Fix)

If you load a Zarr directory that was created before this fix:

**Message shown:**
```
⚠️  Original SEG-Y path not in metadata.
To enable batch processing, please re-import from SEG-Y.
```

**Solution:** Re-import the SEG-Y file, which will save the path in metadata.

### If Original File Was Moved

If the original SEG-Y file was moved or deleted:

**Message shown:**
```
Original SEG-Y File Not Found

The original SEG-Y file is required for export but cannot be found:
/old/path/to/data.sgy

Do you want to browse for the file?
```

**Options:**
- Click **Yes** to browse and locate the file at its new location
- Click **No** and manually move the file back to the original location
- Re-import the SEG-Y from its current location

## Metadata Format

### Non-Streaming Import

```json
{
  "shape": [1000, 6894476],
  "sample_rate": 2.0,
  "n_samples": 1000,
  "n_traces": 6894476,
  "seismic_metadata": {
    "original_segy_path": "/path/to/original/data.sgy",
    ...
  }
}
```

### Streaming Import

```json
{
  "shape": [1000, 6894476],
  "sample_rate": 2.0,
  "n_samples": 1000,
  "n_traces": 6894476,
  "seismic_metadata": {
    "source_file": "/path/to/original/data.sgy",
    "original_segy_path": "/path/to/original/data.sgy",
    "file_info": {...},
    "header_mapping": {...}
  }
}
```

Both formats are supported - the code checks for both locations.

## Benefits

1. **Seamless workflow**: Load from Zarr and export works immediately
2. **Path validation**: User is informed if file is missing and can browse for it
3. **Backward compatible**: Old Zarr directories get helpful message to re-import
4. **User-friendly**: Clear error messages with actionable solutions

## Testing

To test the fix:

1. **Import a SEG-Y file** → Check metadata.json contains original_segy_path
2. **Load from Zarr** → Should see "✅ Original SEG-Y file found" message
3. **Try batch processing** → Should work without "file not found" error
4. **Move the original SEG-Y** → Should get browse dialog to locate it
5. **Load old Zarr** (pre-fix) → Should get message to re-import

## Summary

The fix enables batch processing and export for data loaded from Zarr by:

✅ Saving the original SEG-Y path in metadata during import
✅ Restoring the path when loading from Zarr
✅ Validating the file still exists
✅ Allowing users to browse if the file moved
✅ Providing clear feedback about path availability

**Result**: No more "This feature only works with data loaded from SEG-Y" error! 🎉
