# SEG-Y Header Mapping Guide

## Overview

The SEG-Y Import Dialog now supports:
1. **Save/Load Mapping Configurations** - Save your header configurations to reuse across similar files
2. **Format Selection** - Choose the correct data format for each header field

---

## Feature 1: Save/Load Header Mapping

### Saving a Mapping Configuration

1. Configure your header mapping in the table
2. Click **"Save Mapping..."** button
3. Choose a location and filename (e.g., `my_survey_headers.json`)
4. The mapping is saved including:
   - All header fields (name, byte position, format, description)
   - Ensemble keys configuration

### Loading a Mapping Configuration

1. Click **"Load Mapping..."** button
2. Select a previously saved mapping file (`.json`)
3. The table will populate with the saved configuration
4. Ensemble keys are automatically restored

### Benefits

- **Reusability**: Save once, use for all files from the same survey
- **Consistency**: Ensure same headers are extracted across datasets
- **Time-saving**: No need to reconfigure for each import
- **Sharing**: Share configurations with team members

---

## Feature 2: Format Selection

### Available Formats

The Format column now has a dropdown with these options:

| Format Code | Display Name | Description | Byte Size | SEG-Y Notation |
|-------------|--------------|-------------|-----------|----------------|
| `h` | 2i - 2-byte signed int | Short integer | 2 bytes | 2i |
| `H` | 2u - 2-byte unsigned int | Unsigned short | 2 bytes | 2u |
| `i` | 4i - 4-byte signed int | Integer (default) | 4 bytes | 4i |
| `I` | 4u - 4-byte unsigned int | Unsigned integer | 4 bytes | 4u |
| `f` | 4r - 4-byte float | Floating point | 4 bytes | 4r |
| `d` | 8r - 8-byte double | Double precision | 8 bytes | 8r |
| `b` | 1i - 1-byte signed int | Byte | 1 byte | 1i |
| `B` | 1u - 1-byte unsigned int | Unsigned byte | 1 byte | 1u |

### How to Select Format

1. Click on the Format cell for any header
2. Dropdown menu appears automatically
3. Select the appropriate format based on your SEG-Y specification
4. Format is used when reading trace headers

### Common Format Usages

**Standard SEG-Y Rev 1:**
- **CDP number** (byte 21): `i` (4-byte int)
- **Inline/Crossline** (bytes 189/193): `i` (4-byte int)
- **Offset** (byte 37): `i` (4-byte int)
- **Sample count** (byte 115): `h` (2-byte int)
- **Sample interval** (byte 117): `h` (2-byte int)
- **Scalar for coordinates** (byte 71): `h` (2-byte int)
- **Trace ID code** (byte 29): `h` (2-byte int)

**Custom/Extended Headers:**
- **Azimuth angles**: `f` (4-byte float)
- **Elevation data**: `f` or `i` depending on specification
- **Quality flags**: `h` or `b` (small integers)
- **Coordinates**: Usually `i` (4-byte int) with scalar

---

## Example Workflows

### Workflow 1: First-time Import with Custom Headers

1. **Load SEG-Y file** → Click "Browse..." and select file
2. **Load standard headers** → Click "Load Standard Headers"
3. **Add custom headers** → Click "Add Custom Header" for each custom field
   - Enter header name (e.g., "azimuth")
   - Enter byte position (e.g., "233")
   - Select format (e.g., "4r - 4-byte float")
   - Enter description (e.g., "Source azimuth angle")
4. **Configure ensembles** → Enter "cdp" or "inline,crossline"
5. **Preview** → Click "Preview Headers" to verify
6. **Save mapping** → Click "Save Mapping..." to save for future use
7. **Import** → Click "Import SEG-Y"

### Workflow 2: Repeat Import with Saved Mapping

1. **Load SEG-Y file** → Click "Browse..." and select new file
2. **Load mapping** → Click "Load Mapping..." and select saved `.json`
3. **Preview** → Click "Preview Headers" to verify (optional)
4. **Import** → Click "Import SEG-Y"

Time saved: ~5 minutes per import!

### Workflow 3: Sharing Mapping with Team

**Person A (Creates mapping):**
```bash
1. Configure headers for survey XYZ
2. Save mapping as "survey_xyz_headers.json"
3. Share file via email/network/git
```

**Person B (Uses mapping):**
```bash
1. Receive "survey_xyz_headers.json"
2. Load SEG-Y file from same survey
3. Load mapping from file
4. Import immediately - all headers configured!
```

---

## Example Mapping Files

### Example 1: Simple CDP Gather

```json
{
  "fields": {
    "cdp": {
      "name": "cdp",
      "byte_position": 21,
      "format": "i",
      "description": "CDP ensemble number"
    },
    "offset": {
      "name": "offset",
      "byte_position": 37,
      "format": "i",
      "description": "Distance from source to receiver"
    },
    "sample_count": {
      "name": "sample_count",
      "byte_position": 115,
      "format": "h",
      "description": "Number of samples"
    }
  },
  "ensemble_keys": ["cdp"]
}
```

**Usage:** Basic 2D post-stack data with CDP gathers

---

### Example 2: 3D Survey with Inline/Crossline

```json
{
  "fields": {
    "inline": {
      "name": "inline",
      "byte_position": 189,
      "format": "i",
      "description": "Inline number"
    },
    "crossline": {
      "name": "crossline",
      "byte_position": 193,
      "format": "i",
      "description": "Crossline number"
    },
    "offset": {
      "name": "offset",
      "byte_position": 37,
      "format": "i",
      "description": "Offset"
    },
    "sample_interval": {
      "name": "sample_interval",
      "byte_position": 117,
      "format": "h",
      "description": "Sample interval (microseconds)"
    }
  },
  "ensemble_keys": ["inline", "crossline"]
}
```

**Usage:** 3D pre-stack data organized by inline/crossline

---

### Example 3: Custom Headers with Mixed Formats

```json
{
  "fields": {
    "cdp": {
      "name": "cdp",
      "byte_position": 21,
      "format": "i",
      "description": "CDP number"
    },
    "azimuth": {
      "name": "azimuth",
      "byte_position": 233,
      "format": "f",
      "description": "Source azimuth (degrees)"
    },
    "quality_flag": {
      "name": "quality_flag",
      "byte_position": 237,
      "format": "h",
      "description": "Trace quality indicator"
    },
    "elevation": {
      "name": "elevation",
      "byte_position": 241,
      "format": "f",
      "description": "Elevation (meters)"
    }
  },
  "ensemble_keys": ["cdp"]
}
```

**Usage:** Dataset with custom quality control and geometry headers

---

## Format Selection Best Practices

### 1. Check Your SEG-Y Specification

Always refer to your data provider's documentation for:
- Custom header byte positions
- Data formats used
- Byte order (big-endian is standard)
- Scalar applications

### 2. Common Mistakes to Avoid

❌ **Wrong format size:**
```
Byte 115 (sample count) should be 'h' (2-byte)
Not 'i' (4-byte) - will read wrong value!
```

❌ **Signed vs Unsigned:**
```
If values are always positive (like trace count),
use unsigned ('H', 'I') for larger range
```

❌ **Float for integers:**
```
CDP numbers are integers, use 'i' not 'f'
Floats have precision issues with large integers
```

### 3. Testing Your Configuration

After configuring formats:
1. Click **"Preview Headers"** button
2. Check sample values make sense:
   - CDP numbers should be reasonable integers
   - Offsets should be realistic distances
   - Coordinates should be in expected range
3. If values look wrong, check:
   - Byte position is correct
   - Format matches specification
   - Endianness is correct (SEG-Y is big-endian)

---

## Troubleshooting

### Problem: "Invalid format" error when loading

**Solution:** Check that format codes in JSON are single characters:
```json
✅ Correct: "format": "i"
❌ Wrong:   "format": "4i"
❌ Wrong:   "format": "int32"
```

### Problem: Header values are zeros or nonsense

**Possible causes:**
1. **Wrong byte position** - Check SEG-Y specification
2. **Wrong format** - Try different format (e.g., 'h' vs 'i')
3. **Headers not populated** - Some files have empty headers

**Debugging:**
```
1. Load file with standard headers first
2. Preview to see which headers have data
3. Compare with your specification
4. Adjust byte positions/formats as needed
```

### Problem: Ensemble detection not working

**Checklist:**
1. Ensemble keys are spelled exactly as in header names
2. Selected headers actually change between ensembles
3. Headers are properly configured with correct formats
4. Preview shows varying values for ensemble keys

---

## Technical Details

### File Format

Mapping files are JSON with this structure:

```json
{
  "fields": {
    "<header_name>": {
      "name": "<header_name>",
      "byte_position": <1-240>,
      "format": "<h|H|i|I|f|d|b|B>",
      "description": "<optional description>"
    },
    ...
  },
  "ensemble_keys": ["<header1>", "<header2>", ...]
}
```

### Format Codes (Python struct module)

The format codes are from Python's `struct` module:

```python
'h' → signed short      (2 bytes, -32768 to 32767)
'H' → unsigned short    (2 bytes, 0 to 65535)
'i' → signed int        (4 bytes, -2147483648 to 2147483647)
'I' → unsigned int      (4 bytes, 0 to 4294967295)
'f' → float             (4 bytes, IEEE 754 single precision)
'd' → double            (8 bytes, IEEE 754 double precision)
'b' → signed char       (1 byte, -128 to 127)
'B' → unsigned char     (1 byte, 0 to 255)
```

All values are read as **big-endian** (SEG-Y standard).

---

## Tips for Efficient Workflows

### 1. Create Mapping Templates

Save mapping templates for common scenarios:
- `cdp_gathers.json` - Standard CDP gather configuration
- `3d_prestack.json` - 3D inline/crossline configuration
- `2d_marine.json` - Marine 2D configuration
- `land_3d.json` - Land 3D with specific headers

### 2. Version Control

Keep mapping files in version control:
```bash
git add survey_A_headers.json
git commit -m "Add header mapping for Survey A"
```

### 3. Documentation

Add comments in description fields:
```json
"description": "CDP number - verified with contractor on 2024-01-15"
```

### 4. Validation

Create a validation checklist:
- [ ] All required headers present
- [ ] Formats match specification
- [ ] Ensemble keys configured
- [ ] Preview shows realistic values
- [ ] Mapping saved with descriptive name

---

## Summary

### Key Benefits

1. **Save/Load Mappings:**
   - ✅ Reuse configurations across similar files
   - ✅ Share with team members
   - ✅ Maintain consistency
   - ✅ Save 5+ minutes per import

2. **Format Selection:**
   - ✅ Correctly interpret header values
   - ✅ Support custom header formats
   - ✅ Visual format selection (no memorization)
   - ✅ Prevent data corruption from wrong formats

### Quick Reference

| Task | Steps |
|------|-------|
| **Save mapping** | Configure → "Save Mapping..." → Choose location |
| **Load mapping** | "Load Mapping..." → Select file → Auto-populated |
| **Change format** | Click Format cell → Select from dropdown |
| **Add custom header** | "Add Custom Header" → Fill fields → Select format |

---

## Support

For issues or questions:
1. Check "Preview Headers" to verify configuration
2. Refer to your SEG-Y specification document
3. Test with known-good data first
4. Use example mapping files as templates

Happy importing! 🎉
