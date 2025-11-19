# Memory-Efficient Batch Processing and Export

## Overview

For large datasets that don't fit in memory, use the **"Process and Export (Memory Efficient)"** feature instead of the standard batch processing.

## The Problem

When you load a large dataset in streaming mode and try to use the standard "Batch Process All Gathers" feature, you may see a warning like:

```
⚠️  Memory Note: Batch processing loads all processed
data into memory (~42081 MB for output).
```

If you don't have enough RAM (e.g., you need 42 GB but only have 16 GB), the standard batch processing won't work.

## The Solution

Use **Processing → Process and Export (Memory Efficient)** instead!

### Key Features

- ✅ **Constant memory usage**: ~100-200 MB regardless of dataset size
- ✅ **No memory limits**: Can process datasets of any size
- ✅ **One-pass operation**: Processes and exports in a single workflow
- ✅ **Progress tracking**: Shows estimated time remaining
- ✅ **Cancellable**: Can stop at any time
- ✅ **Automatic cleanup**: Removes temporary files when done

## How It Works

### Standard Batch Processing (High Memory)

```
Load Data → Process All Gathers → Store ALL in Memory (42 GB!) → Export
                                        ↑
                                   Problem here!
```

### Memory-Efficient Batch Processing (Low Memory)

```
Load Data → Process Gather 1 → Write to temp Zarr → Process Gather 2 → ...
                                       ↓
                                 (only ~100 MB)
                                       ↓
                               Export in Chunks → Final SEG-Y
```

The processed data is written to a temporary Zarr file as gathers are processed, then exported in chunks without ever loading the full dataset into memory.

## How to Use

### Step 1: Load Your Dataset

Load your large SEG-Y dataset in streaming mode as normal:

1. **File → Import SEG-Y from Zarr...**
2. Select your previously imported Zarr directory
3. Data loads in streaming mode (lazy loading)

### Step 2: Configure Processing

1. Navigate to any gather
2. Set your filter parameters (e.g., bandpass filter: 1-120 Hz)
3. Click **Apply Filter** to test and set the parameters
4. Review the results to ensure they look good

### Step 3: Process and Export

1. **Processing → Process and Export (Memory Efficient)...**
2. Choose output SEG-Y filename
3. Review the confirmation dialog:
   ```
   This will process and export all 3979 gathers:

   Zero-phase Butterworth bandpass: 1.0-120.0 Hz, order 4

   Total traces: 6,894,476
   Estimated time: ~344.7 minutes

   ✅ Memory efficient: Uses only ~100-200 MB
   ✅ Processes and exports in one pass
   ✅ No memory limits for large datasets
   ```
4. Click **Yes** to start

### Step 4: Monitor Progress

The progress dialog shows:
- Current gather being processed
- Time remaining estimate
- Export progress (after processing completes)

You can cancel at any time - temporary files are automatically cleaned up.

### Step 5: Done!

When complete, you'll see:
```
Successfully processed and exported all 3979 gathers!

Processing: Zero-phase Butterworth bandpass: 1.0-120.0 Hz, order 4
Total traces: 6,894,476
Total time: 325.4 minutes
Rate: 353 traces/second

Output file: /path/to/output.sgy

✅ Memory-efficient processing completed successfully!
```

## Memory Usage Comparison

### For a 6.9M trace dataset:

| Method | Memory Required | Can Process? |
|--------|----------------|--------------|
| Standard Batch Processing | ~42 GB | ❌ No (if RAM < 42 GB) |
| Memory-Efficient Processing | ~100-200 MB | ✅ Yes (any RAM) |

### Breakdown of Memory Usage:

**Memory-Efficient Method:**
- Input gather loading: ~20-50 MB (one at a time)
- Processing overhead: ~50-100 MB (filters, buffers)
- Temporary Zarr write: Disk only (not RAM)
- Export chunking: ~50-100 MB (5000 traces at a time)
- **Total Peak**: ~100-200 MB

**Standard Method:**
- All input data: N/A (lazy loading)
- All processed output: 42 GB (all traces × samples × 4 bytes)
- **Total Peak**: ~42 GB

## Performance

### Processing Speed

Typical performance on modern hardware:
- **Processing rate**: 300-500 traces/second
- **For 6.9M traces**: ~5-6 hours total time
- **Depends on**:
  - CPU speed
  - Filter complexity
  - Disk I/O speed (for Zarr writes)
  - Gather size

### Time Estimates

The dialog provides rough estimates based on trace count:
- Small dataset (10K traces): ~5 minutes
- Medium dataset (100K traces): ~30 minutes
- Large dataset (1M traces): ~3 hours
- Very large dataset (10M traces): ~8 hours

Actual times may vary based on your hardware and filter complexity.

## Technical Details

### Workflow Steps

1. **Initialize Temporary Zarr**
   - Creates temporary directory
   - Initializes Zarr array with correct shape
   - Chunks optimized for sequential write

2. **Process Gathers**
   - Loads one gather at a time from input Zarr
   - Applies configured processor (e.g., bandpass filter)
   - Writes processed traces to output Zarr
   - Memory is reused for each gather

3. **Chunked Export**
   - Reads processed data from Zarr in chunks (5000 traces)
   - Copies headers from original SEG-Y
   - Writes to output SEG-Y file
   - Memory is reused for each chunk

4. **Cleanup**
   - Removes temporary Zarr directory
   - Frees all resources

### Temporary Files

Location: System temporary directory (e.g., `/tmp/processed_zarr_XXXXXX/`)

Size: Same as output SEG-Y (uncompressed)
- Example: 6.9M traces × 1000 samples × 4 bytes = ~27 GB

The temporary directory is automatically deleted when processing completes or is cancelled.

## When to Use Each Method

### Use Standard Batch Processing When:
- ✅ Dataset fits in available RAM
- ✅ You want to keep processed data in memory for further analysis
- ✅ You need to process multiple times with different parameters

### Use Memory-Efficient Processing When:
- ✅ Dataset doesn't fit in RAM
- ✅ You just want the final exported SEG-Y file
- ✅ You're working on a machine with limited memory
- ✅ You don't need to keep processed data in memory

## Comparison with Standard Workflow

### Standard Workflow

```bash
# Requires: Dataset size + Processing overhead in RAM
# Example: 42 GB RAM needed

1. File → Import SEG-Y from Zarr
2. Configure filter parameters
3. Processing → Batch Process All Gathers  # Loads ALL into RAM
4. File → Export Processed SEG-Y           # Exports from RAM
```

**Pros**: Faster if you process multiple times with different parameters
**Cons**: Requires huge amounts of RAM for large datasets

### Memory-Efficient Workflow

```bash
# Requires: Only ~100-200 MB RAM

1. File → Import SEG-Y from Zarr
2. Configure filter parameters
3. Processing → Process and Export (Memory Efficient)  # Does everything
```

**Pros**: Works with any dataset size, minimal RAM usage
**Cons**: Must reprocess if you want different filter parameters

## Troubleshooting

### "Not enough disk space"

The temporary Zarr file needs disk space equal to your output file size.

**Solution**: Free up disk space or specify a different temp directory by setting the `TMPDIR` environment variable before launching the app.

### "Processing is slow"

**Possible causes**:
1. Slow disk I/O (writing to Zarr)
2. Complex filter operations
3. Large gathers (many traces per gather)
4. CPU bottleneck

**Solutions**:
- Use an SSD for faster disk I/O
- Simplify filter parameters if possible
- Monitor system resources (CPU, disk I/O)

### "Process cancelled but temp files remain"

**Solution**: The app tries to clean up automatically, but if it fails:

```bash
# Linux/Mac
rm -rf /tmp/processed_zarr_*

# Or find the specific directory
ls -lh /tmp | grep processed_zarr
```

### "Export fails with header errors"

**Cause**: Original SEG-Y file must be available for headers

**Solution**: Ensure the original SEG-Y file path is correct and the file exists

## Best Practices

1. **Test First**: Always test your filter parameters on a single gather before processing everything

2. **Save Settings**: Note your filter parameters so you can reproduce the processing

3. **Disk Space**: Ensure you have enough free disk space (at least 1.5x the output file size)

4. **Timing**: Large datasets may take hours - run during off-hours or overnight

5. **Backup**: Keep a backup of your original data

6. **Validate**: Check the output SEG-Y file with a few gathers before assuming everything is correct

## Summary

The **Memory-Efficient Batch Processing** feature enables processing of datasets of any size by:

- Processing gathers one at a time
- Writing to temporary Zarr storage
- Exporting in chunks
- Using only ~100-200 MB RAM total

This allows you to process datasets that would normally require 10s or 100s of GB of RAM on a laptop with just a few GB available.

**Perfect for your 42 GB dataset!** 🎉
