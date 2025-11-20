# QFileDialog Segmentation Fault Fix

## Hypothesis

The crash occurs in `QFileDialog.getOpenFileName()` when using native file dialogs on systems with OpenGL/display integration issues.

## Solution

Force Qt to use non-native file dialogs by adding the `DontUseNativeDialog` option.

## Implementation

Change all `QFileDialog.getOpenFileName()` calls from:
```python
filename, _ = QFileDialog.getOpenFileName(
    self,
    "Select SEG-Y File",
    "",
    "SEG-Y Files (*.sgy *.segy);;All Files (*)"
)
```

To:
```python
filename, _ = QFileDialog.getOpenFileName(
    self,
    "Select SEG-Y File",
    "",
    "SEG-Y Files (*.sgy *.segy);;All Files (*)",
    options=QFileDialog.Option.DontUseNativeDialog  # Prevent native dialog crash
)
```

This forces Qt to use its own cross-platform file dialog instead of the native system dialog,
preventing OpenGL-related crashes.
