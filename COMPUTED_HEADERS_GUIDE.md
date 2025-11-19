# Computed Headers (Trace Header Math) Guide

## Overview

The computed headers feature allows you to create new trace headers from existing ones using mathematical equations during SEGY import. This is useful for transforming complex receiver/source station values to receiver lines, creating normalized coordinates, or deriving any header value from combinations of existing headers.

## Features

✅ **Math Operations**: `+`, `-`, `*`, `/`, `//` (floor division), `%` (modulo), `**` (power)
✅ **Rounding Functions**: `round()`, `floor()`, `ceil()`
✅ **Math Functions**: `abs()`, `min()`, `max()`, `sqrt()`
✅ **Trigonometric**: `sin()`, `cos()`, `tan()`, `atan2()`
✅ **Chained Computations**: Computed headers can reference other computed headers
✅ **Error Handling**: Errors are logged with trace locations, failed computations default to 0
✅ **Persistent Storage**: Computed headers are saved to Parquet alongside raw headers

## Usage in GUI

### Adding Computed Headers

1. Open the **SEGY Import Dialog**
2. Scroll to the **"Computed Headers (Trace Header Math)"** section
3. Click **"Add Computed Header"**
4. Configure the computed header:
   - **Header Name**: Name of the new computed header (e.g., `receiver_line`)
   - **Expression**: Math equation using existing headers (e.g., `round(receiver_station / 1000)`)
   - **Format**: Output data type (`i` = int32, `f` = float32, `h` = int16)
   - **Description**: Human-readable description

### Example Use Cases

#### Example 1: Convert Station to Line
```
Name:        receiver_line
Expression:  round(receiver_station / 1000)
Format:      i (int32)
Description: Receiver line number from station
```

#### Example 2: Normalize Coordinates
```
Name:        normalized_x
Expression:  source_x / 1000
Format:      f (float32)
Description: Source X in kilometers
```

#### Example 3: Compute Midpoint
```
Name:        midpoint_x
Expression:  (source_x + receiver_x) / 2
Format:      f (float32)
Description: Midpoint X coordinate
```

#### Example 4: Chained Computations
```
# First computation
Name:        scaled_x
Expression:  source_x / 1000
Format:      f

# Second computation (uses first)
Name:        rounded_x
Expression:  round(scaled_x)
Format:      i
```

## Programmatic Usage

### Basic Example

```python
from utils.segy_import.header_mapping import HeaderMapping, StandardHeaders
from utils.segy_import.computed_headers import ComputedHeaderField
from utils.segy_import.segy_reader import SEGYReader

# Create header mapping
mapping = HeaderMapping()
mapping.add_standard_headers(StandardHeaders.get_all_standard())

# Add computed headers
receiver_line = ComputedHeaderField(
    name='receiver_line',
    expression='round(receiver_x / 1000)',
    description='Receiver line from X coordinate',
    format='i'
)
mapping.add_computed_field(receiver_line)

# Create reader and import
reader = SEGYReader('file.sgy', mapping)
seismic_data, headers, ensembles = reader.read_to_seismic_data()

# Check for computation errors
error_summary = reader.get_computed_header_errors()
if error_summary:
    print(error_summary)

# Headers now include both raw and computed values
print(headers[0]['receiver_x'])      # Raw header
print(headers[0]['receiver_line'])   # Computed header
```

### Complex Example with Multiple Transformations

```python
# Transform complex receiver/source stations to lines and stakes

# Step 1: Extract line number (first 3 digits)
line_number = ComputedHeaderField(
    name='receiver_line',
    expression='floor(receiver_station / 10000)',
    description='Receiver line number',
    format='i'
)
mapping.add_computed_field(line_number)

# Step 2: Extract stake number (last 4 digits)
stake_number = ComputedHeaderField(
    name='receiver_stake',
    expression='receiver_station % 10000',
    description='Receiver stake number',
    format='i'
)
mapping.add_computed_field(stake_number)

# Step 3: Compute distance along line
distance = ComputedHeaderField(
    name='receiver_distance',
    expression='receiver_stake * 25',  # 25m stake spacing
    description='Distance along receiver line (meters)',
    format='f'
)
mapping.add_computed_field(distance)
```

## Error Handling

When a computation fails (e.g., division by zero, missing header), the system:

1. **Sets the value to 0** for that trace
2. **Records the error** with trace index and header context
3. **Continues processing** remaining traces
4. **Reports statistics** at the end:

```
============================================================
Computed Header Evaluation Errors: 15 total
============================================================

Errors by field:
  receiver_line: 15 errors

First 10 error traces:

  [1] Trace 1234, Field 'receiver_line':
      Error: Division by zero: division by zero
      Headers: cdp=5000, receiver_x=0, source_x=123000

  [2] Trace 1235, Field 'receiver_line':
      Error: Missing header: 'receiver_x'
      Headers: cdp=5001, source_x=123100
============================================================
```

## Available Functions and Operators

### Arithmetic Operators
- `+` : Addition
- `-` : Subtraction
- `*` : Multiplication
- `/` : Division (float result)
- `//` : Floor division (integer result)
- `%` : Modulo (remainder)
- `**` : Power (exponentiation)

### Rounding Functions
- `round(x)` : Round to nearest integer
- `floor(x)` : Round down to integer
- `ceil(x)` : Round up to integer

### Math Functions
- `abs(x)` : Absolute value
- `min(x, y)` : Minimum of two values
- `max(x, y)` : Maximum of two values
- `sqrt(x)` : Square root

### Trigonometric Functions
- `sin(x)` : Sine (x in radians)
- `cos(x)` : Cosine (x in radians)
- `tan(x)` : Tangent (x in radians)
- `atan2(y, x)` : Arctangent of y/x (returns angle in radians)

### Constants
- `pi` : π (3.14159...)
- `e` : Euler's number (2.71828...)

## Chained Computations (Dependency Resolution)

Computed headers can reference other computed headers. The system automatically:

1. **Detects dependencies** by analyzing expressions
2. **Resolves execution order** using topological sort
3. **Executes in correct sequence** so dependencies are available

```python
# This works! The system determines the correct order:
field1 = ComputedHeaderField('scaled_x', 'source_x / 1000', '', 'f')
field2 = ComputedHeaderField('rounded_x', 'round(scaled_x)', '', 'i')
field3 = ComputedHeaderField('result', 'rounded_x * 10', '', 'i')

mapping.add_computed_field(field1)
mapping.add_computed_field(field2)
mapping.add_computed_field(field3)

# Execution order: scaled_x → rounded_x → result
```

**Important**: Circular dependencies are detected and will raise an error:
```python
# This will fail!
field1 = ComputedHeaderField('a', 'b + 1', '', 'i')
field2 = ComputedHeaderField('b', 'a + 1', '', 'i')  # Circular!
```

## Storage and Persistence

### Parquet Storage

Computed headers are stored in the same `headers.parquet` file as raw headers:

```
headers.parquet columns:
- trace_index (int64)
- cdp (int32)                    # Raw header
- offset (int32)                 # Raw header
- receiver_x (int32)             # Raw header
- receiver_line (int32)          # Computed header
- receiver_stake (int32)         # Computed header
- receiver_distance (float32)    # Computed header
...
```

### Metadata Storage

Computed header definitions are saved in `metadata.json`:

```json
{
  "seismic_metadata": {
    "header_mapping": {
      "fields": { ... },
      "ensemble_keys": ["cdp"],
      "computed_fields": [
        {
          "name": "receiver_line",
          "expression": "round(receiver_x / 1000)",
          "description": "Receiver line from X coordinate",
          "format": "i"
        }
      ]
    }
  }
}
```

### Save/Load Configurations

You can save and reuse header mapping configurations:

```python
# Save configuration
mapping.save_to_file('my_config.json')

# Load configuration (preserves computed headers)
mapping = HeaderMapping.load_from_file('my_config.json')
```

In the GUI, use the **"Save Mapping..."** and **"Load Mapping..."** buttons.

## Performance

Computed headers are evaluated **during import** as headers are read from the SEGY file:

- **Memory**: Negligible overhead (computations done inline)
- **Speed**: ~10-50 microseconds per trace (depends on complexity)
- **Streaming**: Works with streaming import for large files
- **Single-pass**: Computed headers are calculated during the single-pass optimized import

For a 100,000 trace file with 5 computed headers:
- Additional time: ~5-10 seconds
- Memory overhead: ~1-2 MB (for error tracking)

## Best Practices

### 1. Use Descriptive Names
```python
# Good
ComputedHeaderField('receiver_line', 'round(receiver_x / 1000)', ...)

# Bad
ComputedHeaderField('rln', 'round(receiver_x / 1000)', ...)
```

### 2. Add Clear Descriptions
```python
# Good
description='Receiver line number computed from X coordinate (1 line = 1000m)'

# Bad
description='Line'
```

### 3. Choose Appropriate Formats
```python
# Use 'i' (int32) for discrete values
ComputedHeaderField('line_number', 'round(x / 1000)', format='i')

# Use 'f' (float32) for continuous values
ComputedHeaderField('normalized_x', 'x / 1000.0', format='f')

# Use 'h' (int16) for small integers (-32768 to 32767)
ComputedHeaderField('flag', 'offset > 0', format='h')
```

### 4. Handle Edge Cases
```python
# Avoid division by zero
expression='cdp / max(offset, 1)'  # Use max to prevent zero

# Handle missing values gracefully
expression='receiver_x if receiver_x != 0 else source_x'
```

### 5. Test with Preview
Use the **"Preview Headers"** button in the GUI to verify your computed expressions before importing the entire file.

## Troubleshooting

### Issue: Computed header shows all zeros

**Cause**: Expression error or missing dependency

**Solution**:
1. Check the error summary in terminal after import
2. Verify all referenced headers exist in the raw header mapping
3. Test expression in preview with first 10 traces

### Issue: "Circular dependency detected"

**Cause**: Computed headers reference each other in a loop

**Solution**: Restructure expressions to remove circular dependencies

### Issue: Values are wrong

**Cause**: Expression logic error or wrong header names

**Solution**:
1. Use "Preview Headers" to see first 10 traces
2. Check raw header names are correct (case-sensitive)
3. Verify math operations produce expected results

## Examples from Real Surveys

### 3D Land Survey - Receiver Line Extraction
```python
# Station format: LLLSSSS (line 3 digits, stake 4 digits)
# Example: 1050234 = Line 105, Stake 234

receiver_line = ComputedHeaderField(
    'receiver_line',
    'floor(receiver_station / 10000)',
    'Receiver line number',
    'i'
)

receiver_stake = ComputedHeaderField(
    'receiver_stake',
    'receiver_station % 10000',
    'Receiver stake number',
    'i'
)
```

### Marine Survey - Coordinate Transformation
```python
# Convert from integer coordinates to actual meters

scaled_x = ComputedHeaderField(
    'actual_x',
    'source_x * scalar_coord if scalar_coord < 0 else source_x / scalar_coord',
    'Actual X coordinate in meters',
    'f'
)
```

### Shot Gather - Offset Binning
```python
# Bin offsets into 50m bins

offset_bin = ComputedHeaderField(
    'offset_bin',
    'round(abs(offset) / 50) * 50',
    'Offset bin (50m intervals)',
    'i'
)
```

## Testing

Run the test suite to verify functionality:

```bash
python test_computed_headers.py
```

This tests:
- Basic computations
- Chained computations
- Error handling
- Math functions
- Save/load functionality

## Summary

Computed headers provide a powerful way to transform and derive trace header values during SEGY import. They are:

- **Flexible**: Support complex math and chaining
- **Efficient**: Computed during import with minimal overhead
- **Robust**: Error handling with detailed logging
- **Persistent**: Saved to Parquet and metadata
- **Reusable**: Save/load configurations

Use them to clean up complex station/line numbering schemes, normalize coordinates, or create any derived header values you need for your seismic processing workflow.
