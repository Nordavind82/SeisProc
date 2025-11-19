# Bug Fix: Batch Processing with Lazy/Streaming Mode

## Issue

Batch processing failed when working with large datasets loaded in lazy/streaming mode, displaying the error:

```
Failed to process gathers: Full dataset not available
```

## Root Cause

The batch processing code in `main_window.py:_batch_process_all_gathers()` was written to only work with the legacy full data mode:

```python
# Old code (lines 643-647)
full_data = self.gather_navigator.full_data

if full_data is None:
    raise ValueError("Full dataset not available")

processed_traces = np.zeros_like(full_data.traces)
```

When data is loaded in lazy/streaming mode:
- `self.gather_navigator.full_data` is `None` (by design)
- `self.gather_navigator.lazy_data` contains the data reference
- The code would fail at the check on line 646

## Solution

Modified the batch processing to support both legacy and lazy loading modes:

### 1. Check Both Data Sources

```python
# New code (lines 660-676)
if self.gather_navigator.full_data is not None:
    # Full data mode
    full_data = self.gather_navigator.full_data
    n_samples = full_data.n_samples
    n_traces = full_data.n_traces
    sample_rate = full_data.sample_rate
    metadata = full_data.metadata.copy()
elif self.gather_navigator.lazy_data is not None:
    # Lazy loading mode
    lazy_data = self.gather_navigator.lazy_data
    n_samples = lazy_data.n_samples
    n_traces = lazy_data.n_traces
    sample_rate = lazy_data.sample_rate
    metadata = lazy_data.metadata.copy()
else:
    raise ValueError("No dataset loaded")
```

### 2. Pre-allocate with Explicit Shape

```python
# New code (line 679)
processed_traces = np.zeros((n_samples, n_traces), dtype=np.float32)
```

Instead of using `np.zeros_like(full_data.traces)`, we now explicitly create the array with the shape obtained from either data source.

### 3. Use Extracted Variables

```python
# New code (lines 724-734)
self.full_processed_data = SeismicData(
    traces=processed_traces,
    sample_rate=sample_rate,  # From extracted variable
    metadata={
        **metadata,  # From extracted metadata
        'description': f'Batch Processed - {self.last_processor.get_description()}',
        'n_gathers': n_gathers,
        'sorted': len(self.gather_navigator.sort_keys) > 0,
        'sort_keys': self.gather_navigator.sort_keys.copy() if self.gather_navigator.sort_keys else []
    }
)
```

### 4. Added Memory Warning

For large datasets in lazy mode, added a memory usage warning in the confirmation dialog:

```python
if self.gather_navigator.lazy_data is not None:
    n_samples = self.gather_navigator.lazy_data.n_samples
    estimated_mb = (n_samples * total_traces * 4) / (1024 * 1024)
    message += (
        f"⚠️  Memory Note: Batch processing loads all processed\n"
        f"data into memory (~{estimated_mb:.0f} MB for output).\n\n"
    )
```

## Files Modified

- **main_window.py** (`_batch_process_all_gathers` method, lines 642-740)
  - Added support for lazy data mode
  - Extract shape information from appropriate source
  - Added memory usage warning
  - Use extracted variables instead of assuming `full_data` exists

## How It Works

### Data Flow

1. **User loads large dataset in streaming mode**
   - `GatherNavigator.load_lazy_data()` is called
   - `self.gather_navigator.lazy_data` is set
   - `self.gather_navigator.full_data` is `None`

2. **User applies filter to one gather**
   - Sets `self.last_processor` with current parameters
   - This works fine because `get_current_gather()` supports both modes

3. **User clicks "Batch Process All Gathers"**
   - NEW: Code checks for `lazy_data` if `full_data` is None
   - NEW: Extracts `n_samples`, `n_traces`, `sample_rate` from `lazy_data`
   - Pre-allocates output array with correct shape
   - Processes gathers one-by-one (this already worked)
   - Stores results in pre-allocated array
   - Creates final `SeismicData` with processed traces

### Key Insight

The gather-by-gather processing loop already worked in both modes because `get_current_gather()` handles both:

```python
def get_current_gather(self) -> Tuple[SeismicData, pd.DataFrame, Dict]:
    # Check if using lazy loading mode
    if self.lazy_data is not None:
        return self._get_lazy_gather(self.current_gather_id)

    # Legacy full data mode
    if self.full_data is None:
        raise ValueError("No data loaded")
    ...
```

The only issue was initializing the output array, which required knowing the full dataset shape.

## Memory Considerations

### Before Fix
- Lazy mode: ❌ Batch processing not available
- User had to process gathers one by one

### After Fix
- Lazy mode: ✅ Batch processing works
- Memory usage: ~2x dataset size during batch processing
  - 1x for input (loaded gather-by-gather, reused)
  - 1x for accumulated output (kept in memory)
- User is warned about memory requirements before proceeding

### Example Memory Usage

For a 1000-trace × 2000-sample dataset:
- Per-gather processing: ~20 MB (50 traces × 2000 samples × 4 bytes)
- Output accumulation: ~8 MB (1000 traces × 2000 samples × 4 bytes)
- Total peak: ~28 MB (vs 16 MB if fully loaded upfront)

This is acceptable since batch processing is inherently a "create full output" operation.

## Testing

A test script `test_batch_processing_lazy.py` was created to validate:

1. ✅ Can load dataset in lazy mode
2. ✅ Can extract shape information from `lazy_data`
3. ✅ Can pre-allocate output array with correct dimensions
4. ✅ Can process gathers one-by-one
5. ✅ Can accumulate results in output array

## Benefits

1. **Enables batch processing on large datasets**
   - Previously impossible with lazy mode
   - Now works seamlessly

2. **Maintains memory efficiency**
   - Only loads one gather at a time during processing
   - Output accumulation is necessary for batch export

3. **Transparent to user**
   - Same workflow for both full and lazy modes
   - Warning message informs about memory usage

4. **Backward compatible**
   - Full data mode continues to work exactly as before
   - No changes to existing workflows

## Related Components

- `models/gather_navigator.py`: Manages both full and lazy data modes
- `models/lazy_seismic_data.py`: Provides `n_samples`, `n_traces`, `sample_rate` properties
- `models/seismic_data.py`: Target format for batch processed output

## Summary

The fix enables batch processing to work with datasets loaded in lazy/streaming mode by:

1. ✅ Checking for both `full_data` and `lazy_data` sources
2. ✅ Extracting shape information from the available source
3. ✅ Pre-allocating output array with explicit dimensions
4. ✅ Warning users about memory requirements
5. ✅ Maintaining backward compatibility with legacy mode

**Result**: Users can now batch process large datasets loaded via streaming mode! 🎉
