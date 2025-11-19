# Quick Start: Process Your 42 GB Dataset

## Your Situation

- Dataset: 3,979 gathers, 6,894,476 traces
- Standard batch processing needs: ~42 GB RAM
- You have: Less than 42 GB available
- **Solution**: Use the new memory-efficient feature!

## Step-by-Step Guide

### 1. You've Already Loaded the Data ✅

Your dataset is loaded in streaming mode from Zarr. Good!

### 2. You've Already Selected Filter Parameters ✅

You have:
- Zero-phase Butterworth bandpass: 1.0-120.0 Hz, order 4

Perfect!

### 3. Instead of "Batch Process All Gathers"...

**Don't click**: "Processing → Batch Process All Gathers" (needs 42 GB RAM)

**Do click**: **"Processing → Process and Export (Memory Efficient)"** (needs only ~100 MB RAM)

### 4. Choose Output File

When prompted:
1. Select where to save the processed SEG-Y file
2. Give it a meaningful name like `processed_bandpass_1-120Hz.sgy`
3. Click **Save**

### 5. Confirm and Start

You'll see:
```
This will process and export all 3979 gathers:

Zero-phase Butterworth bandpass: 1.0-120.0 Hz, order 4

Total traces: 6,894,476
Estimated time: ~344.7 minutes

✅ Memory efficient: Uses only ~100-200 MB
✅ Processes and exports in one pass
✅ No memory limits for large datasets

Output: /path/to/processed_bandpass_1-120Hz.sgy

Continue?
```

Click **Yes** to start!

### 6. Wait for Processing

- Estimated time: ~5-6 hours for your dataset
- Progress dialog shows current gather and time remaining
- You can cancel anytime if needed
- **Leave the computer running** - it's doing heavy computation

### 7. Done!

When finished:
- Your processed SEG-Y file is ready
- All original headers preserved
- Temporary files automatically cleaned up
- Memory usage never exceeded ~200 MB

## Key Differences

| Feature | Standard | Memory-Efficient |
|---------|----------|------------------|
| Menu Option | "Batch Process All Gathers" | "Process and Export (Memory Efficient)" |
| Memory Needed | 42 GB | ~100-200 MB |
| Keeps in RAM | Yes | No |
| Direct Export | No (separate step) | Yes (all in one) |
| Can Process Your Data | ❌ No | ✅ Yes! |

## What Happens Behind the Scenes

1. **Processing phase** (~90% of time):
   - Loads gather 1, processes it, writes to temp Zarr
   - Loads gather 2, processes it, writes to temp Zarr
   - ... repeats for all 3,979 gathers
   - Memory: Only one gather at a time (~50 MB)

2. **Export phase** (~10% of time):
   - Reads from temp Zarr in chunks (5,000 traces at a time)
   - Copies headers from original SEG-Y
   - Writes to output SEG-Y file
   - Memory: Only one chunk at a time (~50 MB)

3. **Cleanup**:
   - Deletes temp Zarr directory
   - Frees all resources

## Temp Disk Space Needed

You'll need about **~27 GB of free disk space** for temporary files during processing.

Check your available space:
```bash
df -h /tmp
```

If `/tmp` doesn't have enough space, you can set a different location before starting the app:
```bash
export TMPDIR=/path/to/disk/with/space
python main.py
```

## Estimated Timeline for Your Dataset

- **Total traces**: 6,894,476
- **Processing rate**: ~350 traces/second (typical)
- **Total time**: ~5.5 hours
- **Per gather**: ~5 seconds average

So you can expect:
- **Hour 1**: ~650 gathers processed (16%)
- **Hour 3**: ~1,950 gathers processed (49%)
- **Hour 5**: ~3,250 gathers processed (82%)
- **Hour 5.5**: All 3,979 gathers done! ✅

## Tips

1. **Start a test run**: If unsure, cancel after 10-20 gathers and check the output file is being created correctly

2. **Run overnight**: 5-6 hours is a long time - start it before bed or before leaving for the day

3. **Monitor memory**: Open Task Manager (Windows) or Activity Monitor (Mac) to verify memory stays low

4. **Don't close the app**: Keep the application window open - closing it will cancel processing

## You're All Set!

Just go to: **Processing → Process and Export (Memory Efficient)**

Your 42 GB dataset that wouldn't fit in RAM? No problem! 🚀
