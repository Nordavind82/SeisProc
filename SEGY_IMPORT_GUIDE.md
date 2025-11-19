# SEG-Y Import Guide

## Overview

The SEG-Y import system provides comprehensive control over how seismic data is loaded, including:

- **Custom Header Mapping**: Define which headers to read from specific byte positions
- **Ensemble Configuration**: Specify headers that define gather boundaries
- **Efficient Storage**: Data saved to Zarr (compressed traces) and Parquet (headers)
- **Fast Access**: Indexed for quick ensemble and trace retrieval

## Storage Architecture

```
output_directory/
├── traces.zarr/              # Compressed trace data
│   ├── .zarray              # Zarr metadata
│   └── [chunk files]        # Compressed data chunks
├── headers.parquet          # All trace headers (queryable)
├── ensemble_index.parquet   # Ensemble boundaries
├── trace_index.parquet      # Trace lookup indices
└── metadata.json            # File and processing metadata
```

**Benefits**:
- **Zarr**: 3-5x compression, chunked access, fast partial reads
- **Parquet**: Fast SQL-like queries on headers, columnar storage
- **Indices**: O(1) ensemble and trace lookup

## Using the SEG-Y Import Dialog

### Step 1: Open Import Dialog

**File → Load SEG-Y File...** (Ctrl+O)

### Step 2: Select SEG-Y File

1. Click **Browse...**
2. Select your `.sgy` or `.segy` file
3. Click **File Info** to view file metadata

### Step 3: Configure Header Mapping

The header mapping table defines which trace headers to read.

#### Standard Headers (Pre-loaded)

The dialog automatically loads standard SEG-Y Rev 1 headers:

| Header Name | Byte Position | Format | Description |
|-------------|---------------|--------|-------------|
| trace_sequence_file | 5 | i (int32) | Trace sequence number |
| cdp | 21 | i (int32) | CDP ensemble number |
| offset | 37 | i (int32) | Source-receiver offset |
| sample_count | 115 | h (int16) | Samples per trace |
| sample_interval | 117 | h (int16) | Sample interval (μs) |
| inline | 189 | i (int32) | Inline number (3D) |
| crossline | 193 | i (int32) | Crossline number (3D) |

**Format Codes**:
- `i` = 4-byte signed integer (int32)
- `h` = 2-byte signed integer (int16)
- `f` = 4-byte IEEE float
- `B` = 1-byte unsigned integer

#### Adding Custom Headers

1. Click **Add Custom Header**
2. Enter:
   - **Header Name**: Descriptive name (e.g., `custom_velocity`)
   - **Byte Position**: Starting byte (1-240, SEG-Y convention)
   - **Format**: Struct format code
   - **Description**: What this header represents

**Example: Custom Header at Bytes 233-236**
```
Name: processing_code
Byte Position: 233
Format: i
Description: Custom processing identifier
```

#### Removing Headers

1. Select row(s) in table
2. Click **Remove Selected**

### Step 4: Configure Ensemble Boundaries

Ensemble keys define how traces are grouped into gathers.

**Examples**:

| Acquisition Type | Ensemble Keys | Description |
|------------------|---------------|-------------|
| 2D CDP Gathers | `cdp` | Group by CDP number |
| 3D Shots | `inline,crossline` | Group by shot location |
| Receiver Gathers | `receiver_x,receiver_y` | Group by receiver |
| Offset Gathers | `cdp,offset` | CDP sorted by offset |

**In the dialog**:
```
Ensemble Keys: cdp
```

or for 3D:
```
Ensemble Keys: inline,crossline
```

### Step 5: Preview Headers

Click **Preview Headers** to see the first 10 traces with configured headers.

**Example Output**:
```
Trace 1:
  trace_sequence_file  = 1
  cdp                  = 1001
  offset               = 100
  inline               = 100
  crossline            = 200
  sample_count         = 1000
  sample_interval      = 2000

Trace 2:
  trace_sequence_file  = 2
  cdp                  = 1001
  offset               = 200
  ...
```

This confirms your header mapping is correct.

### Step 6: Import and Save

1. Click **Import SEG-Y**
2. Select output directory for Zarr/Parquet storage
3. Wait for import (progress dialog shows status)
4. Review statistics

**Import Statistics Example**:
```
Import Successful!

Data Statistics:
- Traces: 10,000
- Samples: 1000
- Ensembles: 100

Storage:
- Zarr size: 38.2 MB
- Headers size: 0.5 MB
- Compression: 3.2x

Output: /path/to/output
```

## Loading Previously Imported Data

Once data is imported, you can reload it quickly:

**File → Load from Zarr/Parquet...**

1. Select the directory where data was saved
2. Data loads instantly (no re-import needed)
3. All headers and ensemble boundaries preserved

## Advanced Usage

### Programmatic Header Configuration

Save/load header mappings for reuse:

```python
from utils.segy_import.header_mapping import HeaderMapping, HeaderField

# Create custom mapping
mapping = HeaderMapping()

# Add standard headers
mapping.add_standard_headers()

# Add custom header
custom_field = HeaderField(
    name='inline_bin',
    byte_position=233,
    format='i',
    description='Inline bin number'
)
mapping.add_field(custom_field)

# Set ensemble keys
mapping.set_ensemble_keys(['cdp'])

# Save for reuse
mapping.save_to_file('my_mapping.json')

# Load later
mapping = HeaderMapping.load_from_file('my_mapping.json')
```

### Querying Headers

After import, use Pandas to query headers:

```python
from utils.segy_import.data_storage import DataStorage

storage = DataStorage('/path/to/data')
seismic_data, headers_df, ensembles_df = storage.load_seismic_data()

# Query headers
subset = storage.query_headers("cdp > 1000 and offset < 2000")
print(subset)

# Get specific ensemble
traces, headers = storage.get_ensemble_traces(ensemble_id=5)
```

### Working with Ensembles

Ensemble index provides fast access:

```python
# Load ensemble index
import pandas as pd
ensembles = pd.read_parquet('output/ensemble_index.parquet')

print(ensembles.head())
#    ensemble_id  start_trace  end_trace  n_traces
# 0            0            0         59        60
# 1            1           60        119        60
# 2            2          120        179        60

# Get traces for ensemble 0
start = int(ensembles.iloc[0]['start_trace'])
end = int(ensembles.iloc[0]['end_trace'])

# Load just those traces from Zarr
import zarr
z = zarr.open('output/traces.zarr', 'r')
ensemble_traces = z[:, start:end+1]
```

## Header Mapping Examples

### Example 1: 2D Marine Survey

```
Standard headers + ensemble key: cdp

Ensemble Keys: cdp
```

### Example 2: 3D Land Survey with Custom Headers

```
Standard headers plus:

Name: line_number
Byte Position: 181
Format: i
Description: 3D line number

Name: stake_number
Byte Position: 185
Format: i
Description: Stake number along line

Ensemble Keys: inline,crossline
```

### Example 3: OBC Survey with Receiver Coordinates

```
Standard headers including:
- receiver_x (byte 81)
- receiver_y (byte 85)

Ensemble Keys: receiver_x,receiver_y
```

## Troubleshooting

### "Failed to read header 'X'"

**Problem**: Invalid byte position or format
**Solution**: Check SEG-Y specification for correct byte position

### "Ensemble key 'X' not in header mapping"

**Problem**: Specified ensemble key not in configured headers
**Solution**: Add the header to mapping table first

### "High compression but data quality issue"

**Problem**: Zarr compression level too high
**Solution**: Modify compression in `data_storage.py`:
```python
compressor=zarr.Blosc(cname='zstd', clevel=1)  # Lower compression
```

### Large File Import is Slow

**Problem**: Reading entire file at once
**Solution**: In `segy_reader.py`, process in chunks or limit traces during testing

## Best Practices

1. **Preview First**: Always preview headers before full import
2. **Save Mappings**: Save header configurations for similar datasets
3. **Descriptive Names**: Use clear header names (not just h1, h2, h3)
4. **Validate Ensemble Keys**: Check preview to confirm ensemble boundaries are correct
5. **Backup Original**: Keep original SEG-Y files, Zarr/Parquet is for fast access
6. **Document Custom Headers**: Add descriptions for all custom headers

## Performance

**Import Speed**: ~10,000 traces/second on modern hardware
**Compression**: 3-5x for typical seismic data
**Query Speed**: <1ms for header queries with Parquet
**Ensemble Access**: <100ms for typical ensemble size

## Data Format Details

### Zarr Array Structure
```python
shape: (n_samples, n_traces)
chunks: (n_samples, 1000)  # 1000 traces per chunk
dtype: float32
compression: zstd level 3 with byte shuffle
```

### Parquet Schema
```
headers.parquet:
  - trace_index: int64 (indexed)
  - cdp: int32
  - offset: int32
  - inline: int32
  - crossline: int32
  - [all configured headers]

ensemble_index.parquet:
  - ensemble_id: int64
  - start_trace: int64
  - end_trace: int64
  - n_traces: int64
```

## Next Steps

After importing data, you can:
1. Apply bandpass filters
2. Compare input vs processed
3. Analyze difference (QC)
4. Export results back to SEG-Y (future feature)
5. Query and subset data by header values
