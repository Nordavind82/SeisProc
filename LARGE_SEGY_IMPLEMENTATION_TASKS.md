# Large SEGY File Handling - Implementation Tasks

## Overview
This document defines concrete, testable tasks for implementing memory-efficient handling of large SEGY files that don't fit in RAM. Each task includes specific implementation requirements, test cases, and completion criteria.

---


## Phase 1: Chunked SEGY Import

### Task 1.1: Implement Streaming Trace Reader
**File:** `utils/segy_import/segy_reader.py`

**Implementation Requirements:**
- Add method `read_traces_in_chunks(chunk_size: int = 5000) -> Generator`
- Method yields tuples of `(traces_array, headers_list, start_index, end_index)`
- Each chunk is a numpy array of shape `(n_samples, chunk_size)` or smaller for last chunk
- Headers are extracted for each trace in the chunk
- Must work with `segyio` library's file handle
- Must not accumulate traces in memory between yields
- Must report accurate chunk boundaries for indexing

**Test Cases:**
1. **Test: Small file (1000 traces) with chunk_size=300**
   - Input: SEGY file with 1000 traces, 500 samples each
   - Expected: 4 chunks yielded (300, 300, 300, 100 traces)
   - Verify: Sum of all chunk sizes equals 1000
   - Verify: No memory growth between chunks (measure with `tracemalloc`)

2. **Test: Chunk boundary integrity**
   - Input: SEGY file with known trace values
   - Process: Read in chunks of 100
   - Verify: Trace 99 in chunk 0 matches trace 99 in original file
   - Verify: Trace 100 in chunk 1 matches trace 100 in original file
   - Verify: No traces duplicated or skipped at boundaries

3. **Test: Headers match chunk traces**
   - Input: SEGY file with varying CDP numbers
   - Process: Read in chunks
   - Verify: Length of headers_list equals traces_array.shape[1] for each chunk
   - Verify: Header CDP values match expected sequence

4. **Test: Memory efficiency**
   - Input: 100MB SEGY file
   - Process: Read with chunk_size=1000
   - Verify: Peak memory usage < 50MB during iteration
   - Verify: Memory returns to baseline after generator exhausted

**Completion Criteria:**
- All 4 tests pass
- Code includes docstring with example usage
- No memory leaks detected
- Works with both IBM float and IEEE float SEGY formats

**Expected Success Message:**
```
✓ Task 1.1 Complete: Streaming trace reader implemented
  - Successfully reads SEGY files in chunks
  - Memory usage: O(chunk_size) confirmed
  - Tested with 100MB file: peak memory 45MB
  - All boundary conditions handled correctly
```

---

### Task 1.2: Implement Direct-to-Zarr Streaming Writer
**File:** `utils/segy_import/data_storage.py`

**Implementation Requirements:**
- Add method `save_traces_streaming(trace_generator, n_samples: int, n_traces: int, chunk_size: int, progress_callback=None)`
- Creates Zarr array with full dimensions upfront: `shape=(n_samples, n_traces)`
- Writes each chunk from generator directly to Zarr at correct offset
- Uses compression: `Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)`
- Chunk configuration: `chunks=(n_samples, min(chunk_size, 1000))`
- Calls `progress_callback(current_trace, total_traces)` after each chunk if provided
- Does not load full dataset into memory at any point
- Closes Zarr array properly after streaming complete

**Test Cases:**
1. **Test: Stream 10,000 traces to Zarr**
   - Input: Generator yielding 10 chunks of 1000 traces each (500 samples)
   - Expected: Zarr file created with shape (500, 10000)
   - Verify: All data matches input (compare trace 0, 5000, 9999)
   - Verify: Zarr file size < uncompressed size (check compression ratio > 1.5x)

2. **Test: Progress callback accuracy**
   - Input: Generator yielding 5 chunks of 200 traces (total 1000)
   - Progress tracking: Capture all callback invocations
   - Expected: 5 callbacks with values: 200, 400, 600, 800, 1000
   - Verify: Final callback shows 100% (1000/1000)

3. **Test: Memory efficiency during streaming**
   - Input: Generator simulating 50MB of data in 10 chunks
   - Process: Stream to Zarr while monitoring memory
   - Verify: Peak memory < 15MB (should only hold one chunk + Zarr buffer)
   - Verify: Memory doesn't grow linearly with number of chunks

4. **Test: Chunk boundary integrity in Zarr**
   - Input: 3000 traces in chunks of 1000
   - Write: Stream to Zarr
   - Read back: Load trace ranges [999-1001], [1999-2001]
   - Verify: No discontinuities or artifacts at chunk boundaries

5. **Test: Zarr metadata correctness**
   - Input: Stream data with specific compression settings
   - Verify: Zarr .zarray metadata shows correct shape, dtype, chunks, compressor
   - Verify: Can reopen Zarr array in read mode and access data

**Completion Criteria:**
- All 5 tests pass
- Peak memory during streaming < 20MB regardless of input size
- Compression ratio > 1.5x for typical seismic data
- Progress callbacks provide accurate updates
- Zarr files are valid and readable by standard Zarr library

**Expected Success Message:**
```
✓ Task 1.2 Complete: Direct-to-Zarr streaming writer implemented
  - Successfully writes streaming data to Zarr
  - Memory usage: O(1) confirmed - peak 12MB for 50MB dataset
  - Compression ratio: 3.2x achieved
  - Progress tracking functional
  - Zarr arrays verified readable and correct
```

---

### Task 1.3: Implement Streaming Header Collection
**File:** `utils/segy_import/data_storage.py`

**Implementation Requirements:**
- Add method `save_headers_streaming(header_generator, output_path: Path, progress_callback=None)`
- Accumulates headers in batches of 10,000 before writing to Parquet
- Uses pyarrow for Parquet writing with `append` mode
- Adds `trace_index` column automatically (sequential numbering)
- Final Parquet file must be single file (not partitioned)
- Memory usage proportional to batch size only, not total headers
- Handles headers with varying fields (missing values as None/NaN)

**Test Cases:**
1. **Test: Stream 50,000 headers in batches**
   - Input: Generator yielding 50 batches of 1000 headers each
   - Expected: Single Parquet file with 50,000 rows
   - Verify: trace_index column ranges from 0 to 49,999
   - Verify: All header fields present (cdp, offset, inline, etc.)
   - Verify: Random sample of 100 headers matches input

2. **Test: Batch memory efficiency**
   - Input: Generator yielding headers (simulate 100 fields per header)
   - Process: Accumulate in batches of 10,000
   - Monitor: Memory usage after each batch write
   - Verify: Memory resets after batch write (batch buffer cleared)
   - Verify: Peak memory < 50MB for 100,000 headers

3. **Test: Missing header fields handled**
   - Input: Headers with varying fields (some have inline, some don't)
   - Process: Stream to Parquet
   - Verify: Parquet schema includes all fields seen
   - Verify: Missing values represented as null/NaN
   - Verify: Can read back with pandas and query

4. **Test: Parquet file integrity**
   - Input: Stream 25,000 headers
   - Write: To Parquet with compression
   - Verify: File size < uncompressed size
   - Verify: Can read full file with pandas
   - Verify: Can query with predicates (e.g., "cdp > 1000")

**Completion Criteria:**
- All 4 tests pass
- Memory usage proportional to batch size only (10,000 headers)
- Parquet files readable by pandas/pyarrow
- Supports querying with standard pandas operations
- No data loss or corruption

**Expected Success Message:**
```
✓ Task 1.3 Complete: Streaming header collection implemented
  - Successfully streams headers to Parquet
  - Memory usage: O(batch_size) confirmed - peak 35MB for 100k headers
  - Parquet files optimized: 5x compression achieved
  - Query operations tested and functional
  - Handles variable schema correctly
```

---

### Task 1.4: Implement Ensemble Detection from Streaming Headers
**File:** `utils/segy_import/data_storage.py`

**Implementation Requirements:**
- Add method `detect_ensembles_streaming(header_generator, ensemble_keys: List[str])`
- Detects ensemble boundaries on-the-fly as headers stream
- Yields ensemble boundary tuples: `(ensemble_id, start_trace, end_trace, key_values)`
- Memory usage: O(1) - only tracks current ensemble state
- Handles unsorted data gracefully (doesn't assume pre-sorted input)
- Works with multiple ensemble keys (e.g., ['inline', 'crossline'])
- Produces same results as batch method but with streaming input

**Test Cases:**
1. **Test: Single ensemble key (CDP)**
   - Input: 5000 headers with CDP pattern: 100 traces CDP=1, 100 traces CDP=2, etc.
   - Ensemble keys: ['cdp']
   - Expected: 50 ensembles yielded
   - Verify: First ensemble (0, 0, 99, {'cdp': 1})
   - Verify: Last ensemble (49, 4900, 4999, {'cdp': 50})

2. **Test: Multiple ensemble keys (inline + crossline)**
   - Input: 900 headers with inline=[1,1,1...2,2,2...3,3,3], crossline=[1..100, 1..100, 1..100]
   - Ensemble keys: ['inline', 'crossline']
   - Expected: 300 ensembles (3 inlines × 100 crosslines)
   - Verify: Ensemble (0, 0, 2, {'inline': 1, 'crossline': 1})
   - Verify: Correct boundaries for ensemble transitions

3. **Test: Variable ensemble sizes**
   - Input: Headers with CDP pattern: 50 traces CDP=1, 150 traces CDP=2, 30 traces CDP=3
   - Expected: 3 ensembles with correct sizes (50, 150, 30)
   - Verify: n_traces field matches actual trace count

4. **Test: Memory efficiency**
   - Input: Stream 100,000 headers
   - Process: Detect ensembles on-the-fly
   - Monitor: Peak memory usage
   - Verify: Memory < 5MB (only current ensemble state tracked)
   - Verify: No memory accumulation over time

5. **Test: Unsorted data handling**
   - Input: Headers with CDP switching: 1,1,2,2,1,1,3,3
   - Expected: Detects 5 ensembles (not 3) due to re-occurrence
   - Verify: Boundaries correct for this pattern

**Completion Criteria:**
- All 5 tests pass
- Memory usage O(1) regardless of input size
- Correctly handles single and multiple ensemble keys
- Produces identical results to batch detection method
- Performance: < 1ms per 1000 headers processed

**Expected Success Message:**
```
✓ Task 1.4 Complete: Streaming ensemble detection implemented
  - Successfully detects ensembles from streaming headers
  - Memory usage: O(1) confirmed - 2MB for 100k headers
  - Handles single and multiple ensemble keys
  - Performance: 0.5ms per 1000 headers
  - Tested with sorted and unsorted data
```

---

### Task 1.5: Integrate Streaming Import into SEGY Import Dialog
**File:** `views/segy_import_dialog.py`

**Implementation Requirements:**
- Modify `_do_import()` method to detect file size
- If file size > 500MB OR n_traces > 50,000: use streaming import
- If smaller: use existing batch import (backward compatibility)
- Progress dialog shows: "Processing chunk X of Y" and progress bar
- Cancelable import: user can stop mid-stream without corruption
- Uses new streaming methods from Tasks 1.1-1.4
- Error handling: rollback/cleanup if import fails mid-stream
- Success dialog shows: traces imported, compression ratio, time taken

**Test Cases:**
1. **Test: Small file uses batch import**
   - Input: SEGY file with 1000 traces (< 50MB)
   - Expected: Uses existing `read_all_traces()` method
   - Verify: Import completes successfully
   - Verify: Data loads correctly in viewer

2. **Test: Large file triggers streaming import**
   - Input: SEGY file with 100,000 traces (> 500MB)
   - Expected: Uses new streaming methods
   - Verify: Progress dialog shows chunk updates
   - Verify: Memory usage < 500MB during entire import
   - Verify: Final Zarr file correct size and readable

3. **Test: Progress updates during streaming**
   - Input: Medium SEGY file (50,000 traces)
   - Monitor: Progress dialog updates
   - Expected: Progress bar updates at least 10 times
   - Verify: Progress goes from 0% to 100%
   - Verify: "Processing chunk" message updates

4. **Test: Cancel during streaming import**
   - Input: Large SEGY file
   - Action: Click cancel after 30% complete
   - Expected: Import stops cleanly
   - Verify: Partial Zarr files cleaned up (output directory empty or marked incomplete)
   - Verify: No orphaned processes or file handles
   - Verify: Application remains stable

5. **Test: Error handling mid-stream**
   - Input: Corrupted SEGY file (simulate read error at trace 5000)
   - Expected: Error dialog shown with specific error
   - Verify: Partial files cleaned up
   - Verify: User can retry with different file

6. **Test: Success metrics display**
   - Input: Successfully imported file
   - Expected: Dialog shows: "50,000 traces imported in 45s, compression 3.2x, output size 180MB"
   - Verify: Metrics accurate (compare to actual file sizes)

**Completion Criteria:**
- All 6 tests pass
- Automatic selection of batch vs streaming mode
- Progress tracking functional and accurate
- Cancel operation works cleanly
- Error handling prevents corruption
- User gets clear feedback on import success

**Expected Success Message:**
```
✓ Task 1.5 Complete: Streaming import integrated into GUI
  - Auto-detection of file size working (threshold: 500MB)
  - Progress tracking functional with chunk updates
  - Cancel operation tested and working
  - Error handling prevents data corruption
  - Success metrics display correctly
  - Memory usage verified: 450MB for 100,000 trace import
```

---

## Phase 2: Lazy Data Loading

### Task 2.1: Implement LazySeismicData Class
**File:** `models/lazy_seismic_data.py` (new file)

**Implementation Requirements:**
- Create class `LazySeismicData` that wraps Zarr array
- Implements same interface as `SeismicData` where possible
- Properties: `n_samples`, `n_traces`, `sample_rate`, `duration`, `nyquist_freq`
- Method: `get_window(time_start, time_end, trace_start, trace_end) -> np.ndarray`
- Method: `get_trace_range(trace_start, trace_end) -> np.ndarray`
- Method: `get_time_range(time_start, time_end) -> np.ndarray`
- Method: `get_ensemble(ensemble_id: int) -> np.ndarray` (uses ensemble index)
- All data access returns numpy arrays (not Zarr arrays)
- Metadata loaded once at initialization (from metadata.json)
- Zarr array opened in read-only mode with memory mapping

**Test Cases:**
1. **Test: Properties match actual data**
   - Setup: Create Zarr with 500 samples, 10,000 traces, 2ms sample rate
   - Create: LazySeismicData instance
   - Verify: n_samples == 500, n_traces == 10000
   - Verify: sample_rate == 2.0, duration == 998.0ms
   - Verify: nyquist_freq == 250.0 Hz

2. **Test: Window extraction correct**
   - Setup: Zarr with known pattern (trace[i] = i, all samples)
   - Request: Window time[100:200], trace[50:100]
   - Expected: Array shape (100, 50)
   - Verify: First trace value == 50, last trace value == 99
   - Verify: Result is numpy array (not Zarr)

3. **Test: Full trace range extraction**
   - Setup: Zarr with 5000 samples, 1000 traces
   - Request: get_trace_range(100, 150) (all samples)
   - Expected: Array shape (5000, 50)
   - Verify: Correct traces returned

4. **Test: Ensemble extraction**
   - Setup: Zarr + ensemble index (ensemble 5 = traces 500-599)
   - Request: get_ensemble(5)
   - Expected: Array shape (n_samples, 100)
   - Verify: Traces 500-599 returned

5. **Test: Memory efficiency**
   - Setup: LazySeismicData wrapping 1GB Zarr array
   - Initial: Measure memory footprint
   - Expected: < 10MB (only metadata + Zarr handle)
   - Action: Extract small window (100x100)
   - Verify: Memory increase < 1MB (only window loaded)

6. **Test: Boundary handling**
   - Setup: Zarr with 1000 traces
   - Request: Window trace[900:1100] (exceeds bounds)
   - Expected: Clips to valid range [900:1000]
   - Verify: No error thrown, returns valid data

7. **Test: Read-only enforcement**
   - Setup: LazySeismicData instance
   - Action: Attempt to modify underlying Zarr array
   - Expected: Error or silently ignored (data not modified)
   - Verify: Original Zarr file unchanged

**Completion Criteria:**
- All 7 tests pass
- Interface compatible with SeismicData where applicable
- Memory footprint < 10MB regardless of underlying data size
- All data access returns numpy arrays
- Thread-safe for multiple concurrent reads
- Documentation includes usage examples

**Expected Success Message:**
```
✓ Task 2.1 Complete: LazySeismicData class implemented
  - Wraps Zarr arrays with memory-efficient interface
  - Memory footprint: 3MB for 1GB dataset
  - All access methods tested and working
  - Window extraction verified correct
  - Ensemble access integrated with index
  - Read-only protection confirmed
```

---

### Task 2.2: Implement Windowed Data Loading in Viewer
**File:** `views/seismic_viewer_pyqtgraph.py`

**Implementation Requirements:**
- Add method `set_lazy_data(lazy_data: LazySeismicData)`
- Replace direct data access with window extraction based on viewport
- Method `_load_visible_window()` that:
  - Gets current viewport limits (time_min, time_max, trace_min, trace_max)
  - Adds 10% padding on each side for smooth panning
  - Calls `lazy_data.get_window()` with padded bounds
  - Updates image_item with windowed data
- Hook into viewport change signal to reload window when needed
- Implement hysteresis: only reload if viewport moves > 25% outside current window
- Cache current window to avoid redundant loads
- Display loading indicator during window fetch (if > 100ms)

**Test Cases:**
1. **Test: Initial window load**
   - Setup: LazySeismicData with 5000 samples, 10,000 traces
   - Action: Call set_lazy_data()
   - Expected: Loads initial viewport (e.g., 0-1000ms, traces 0-100)
   - Verify: Image displayed correctly
   - Verify: Memory usage increased by ~window_size only

2. **Test: Pan within cached window**
   - Setup: Viewer with loaded window [0-1500ms, traces 0-150]
   - Action: Pan to [200-800ms, traces 20-100] (within cached range)
   - Expected: No new data load
   - Verify: Image updates instantly (< 10ms)
   - Verify: get_window() not called

3. **Test: Pan outside cached window triggers reload**
   - Setup: Cached window [0-1500ms, traces 0-150]
   - Action: Pan to [1400-2000ms, traces 0-150] (overlap < 25%)
   - Expected: New window loaded with padding
   - Verify: get_window() called once
   - Verify: New cached range includes [1200-2200ms] (with padding)

4. **Test: Zoom out loads larger window**
   - Setup: Cached window [0-1000ms, traces 0-100]
   - Action: Zoom out to view [0-2000ms, traces 0-200]
   - Expected: New larger window loaded
   - Verify: Image shows full zoomed range
   - Verify: No artifacts or gaps

5. **Test: Loading indicator for slow loads**
   - Setup: Simulate slow get_window() (add 200ms delay)
   - Action: Pan to new region
   - Expected: Loading indicator shown during fetch
   - Verify: Indicator disappears when load complete
   - Verify: UI remains responsive

6. **Test: Memory usage stays bounded**
   - Setup: Large dataset (100,000 traces)
   - Action: Pan across 10 different regions
   - Monitor: Memory usage after each pan
   - Verify: Memory stays < 200MB
   - Verify: Old windows garbage collected

7. **Test: Rapid pan operations**
   - Action: Pan quickly across 20 regions (< 100ms between pans)
   - Expected: Only final window loaded (intermediate pans cancelled)
   - Verify: No lag or queue buildup
   - Verify: Final display correct

**Completion Criteria:**
- All 7 tests pass
- Window loading transparent to user
- Memory usage bounded regardless of panning distance
- Smooth panning experience (no visible lag)
- Loading indicator shown only for slow loads (> 100ms)
- Hysteresis prevents unnecessary reloads

**Expected Success Message:**
```
✓ Task 2.2 Complete: Windowed data loading in viewer implemented
  - Lazy data loading functional in PyQtGraph viewer
  - Memory usage bounded: 150MB for 100k trace dataset
  - Panning performance excellent: < 50ms window loads
  - Hysteresis working: 75% reduction in redundant loads
  - Loading indicator functional for slow operations
  - Tested with rapid pan operations: no lag
```

---

### Task 2.3: Implement Window Caching with LRU Policy
**File:** `utils/window_cache.py` (new file)

**Implementation Requirements:**
- Create class `WindowCache` with LRU (Least Recently Used) eviction
- Configuration: max memory size (e.g., 500MB) and max windows (e.g., 5)
- Method: `get(key: tuple) -> Optional[np.ndarray]` - returns cached window or None
- Method: `put(key: tuple, data: np.ndarray)` - adds window to cache
- Method: `clear()` - empties cache
- Key format: `(time_start, time_end, trace_start, trace_end)`
- Tracks memory usage using data.nbytes
- Evicts oldest window when memory limit exceeded
- Thread-safe for concurrent access

**Test Cases:**
1. **Test: Cache hit returns data**
   - Setup: Cache with max 3 windows
   - Action: Put window A, get window A
   - Expected: get() returns same data (not None)
   - Verify: Data matches original (compare checksums)

2. **Test: Cache miss returns None**
   - Setup: Empty cache
   - Action: get() for window that wasn't cached
   - Expected: Returns None
   - Verify: No error thrown

3. **Test: LRU eviction when count exceeded**
   - Setup: Cache with max 3 windows
   - Action: Put windows A, B, C, D (in order)
   - Expected: A evicted (oldest)
   - Verify: get(A) returns None
   - Verify: get(B), get(C), get(D) return data

4. **Test: LRU eviction when memory exceeded**
   - Setup: Cache with max 100MB memory
   - Action: Put 3 windows of 40MB each
   - Expected: First window evicted when third added
   - Verify: Memory usage ≤ 100MB at all times

5. **Test: Access updates LRU order**
   - Setup: Cache with windows A, B, C (in order)
   - Action: get(A), then put(D)
   - Expected: B evicted (A was accessed recently)
   - Verify: get(A), get(C), get(D) return data; get(B) returns None

6. **Test: Clear empties cache**
   - Setup: Cache with 5 windows
   - Action: clear()
   - Verify: All get() calls return None
   - Verify: Memory usage reported as 0

7. **Test: Thread safety**
   - Setup: Cache instance
   - Action: 10 threads simultaneously putting different windows
   - Expected: No race conditions or crashes
   - Verify: All windows cached correctly
   - Verify: Cache state consistent

**Completion Criteria:**
- All 7 tests pass
- LRU eviction correct by both count and memory
- Thread-safe implementation
- Memory tracking accurate (±1% of actual)
- Performance: O(1) for get/put operations
- No memory leaks

**Expected Success Message:**
```
✓ Task 2.3 Complete: Window cache with LRU policy implemented
  - LRU eviction working correctly
  - Memory limit enforcement: ±0.5% accuracy
  - Thread-safe: tested with 10 concurrent threads
  - Performance: O(1) operations confirmed
  - Tested with 100MB cache limit
  - No memory leaks detected
```

---

### Task 2.4: Integrate Window Cache into Viewer
**File:** `views/seismic_viewer_pyqtgraph.py`

**Implementation Requirements:**
- Add WindowCache instance to viewer (max 5 windows or 500MB)
- Modify `_load_visible_window()` to check cache first
- Cache hit: use cached data (no Zarr access)
- Cache miss: load from Zarr, then cache result
- Cache key includes viewport bounds and data identity (to handle multiple datasets)
- Clear cache when new dataset loaded
- Expose cache statistics in debug mode (hits, misses, evictions)

**Test Cases:**
1. **Test: First load misses cache**
   - Setup: Viewer with new lazy data
   - Action: Load initial viewport
   - Expected: Cache miss
   - Verify: Data loaded from Zarr
   - Verify: Data added to cache

2. **Test: Return to previous view hits cache**
   - Setup: Viewer with cached window A
   - Action: Pan to window B, then back to window A
   - Expected: Cache hit on return to A
   - Verify: No Zarr access (monitor calls)
   - Verify: Display correct

3. **Test: Cache persists across zoom operations**
   - Setup: Cached window [0-1000ms, traces 0-100]
   - Action: Zoom in to [200-400ms, traces 20-50] (subset)
   - Expected: Still hits cache (subset of cached window)
   - Verify: No new Zarr load

4. **Test: Cache cleared on new dataset**
   - Setup: Viewer with cached windows for dataset A
   - Action: Load dataset B
   - Expected: Cache cleared
   - Verify: First load of dataset B misses cache

5. **Test: Memory limit prevents unbounded growth**
   - Setup: Cache max 500MB
   - Action: Pan through 10 different large windows (100MB each)
   - Monitor: Memory usage
   - Verify: Memory never exceeds 500MB
   - Verify: Old windows evicted

6. **Test: Cache statistics accurate**
   - Setup: Enable debug mode
   - Action: 5 cache hits, 3 cache misses
   - Expected: Statistics show: hits=5, misses=3, hit_rate=62.5%
   - Verify: Eviction count accurate if any occurred

**Completion Criteria:**
- All 6 tests pass
- Cache hit rate > 60% for typical navigation patterns
- Memory usage stays within configured limit
- Cache cleared appropriately on dataset change
- Noticeable performance improvement (cache hit < 10ms vs miss ~100ms)

**Expected Success Message:**
```
✓ Task 2.4 Complete: Window cache integrated into viewer
  - Cache integration functional
  - Cache hit rate: 68% in typical navigation test
  - Performance improvement: cache hits 15x faster (8ms vs 120ms)
  - Memory limit enforced: peaked at 485MB (limit 500MB)
  - Cache cleared correctly on dataset change
  - Statistics tracking accurate
```

---

## Phase 3: Ensemble/Gather Navigation

### Task 3.1: Implement Lazy Ensemble Loading in GatherNavigator
**File:** `models/gather_navigator.py`

**Implementation Requirements:**
- Modify `load_ensembles()` to accept `LazySeismicData` instead of full data
- Store reference to lazy data and ensemble index
- Method `get_current_gather()` loads only current ensemble from Zarr
- Method `prefetch_adjacent()` loads previous and next ensembles into cache
- Remove in-memory storage of all gathers (currently stores full dataset)
- Ensemble cache: max 5 ensembles in memory
- Method `get_gather_info(gather_id)` returns metadata without loading traces
- Works with both sorted and unsorted data

**Test Cases:**
1. **Test: Load single gather on demand**
   - Setup: LazySeismicData with 100 ensembles
   - Action: navigate_to_gather(50)
   - Expected: Only ensemble 50 loaded into memory
   - Verify: Memory increase ≈ ensemble size only (~10MB)
   - Verify: get_current_gather() returns correct traces

2. **Test: Navigate to next gather**
   - Setup: Current gather 10
   - Action: navigate_next()
   - Expected: Gather 11 loaded, gather 10 may be cached or evicted
   - Verify: Correct gather displayed
   - Verify: Gather number updated to 11

3. **Test: Prefetch adjacent gathers**
   - Setup: Current gather 20
   - Action: Trigger prefetch_adjacent()
   - Expected: Gathers 19 and 21 loaded into cache
   - Verify: Cache contains 3 gathers (19, 20, 21)
   - Verify: No noticeable delay when navigating to 19 or 21

4. **Test: Ensemble cache eviction**
   - Setup: Cache max 5 gathers, navigate through gathers 1-10
   - Expected: Only last 5 gathers cached
   - Verify: Memory usage ≤ 5 × gather_size
   - Verify: gathers 1-5 evicted from cache

5. **Test: Gather info without loading**
   - Setup: 100 ensembles
   - Action: Call get_gather_info(75)
   - Expected: Returns {ensemble_id: 75, n_traces: X, start_trace: Y, end_trace: Z}
   - Verify: No data loaded (memory unchanged)
   - Verify: Info matches ensemble index

6. **Test: Sort order changes**
   - Setup: Data sorted by CDP
   - Action: Change sort to offset
   - Expected: Ensemble index rebuilt
   - Verify: Gathers reorganized correctly
   - Verify: Navigation uses new sort order

**Completion Criteria:**
- All 6 tests pass
- Memory usage proportional to cache size, not dataset size
- Navigation instant for cached gathers (< 50ms)
- Navigation fast for non-cached gathers (< 300ms)
- Prefetching improves user experience (no lag on next/prev)
- Sort order changes work correctly

**Expected Success Message:**
```
✓ Task 3.1 Complete: Lazy ensemble loading in GatherNavigator
  - Gather loading on-demand functional
  - Memory usage: 45MB for 5-gather cache (100k trace dataset)
  - Navigation performance: cached=12ms, uncached=180ms
  - Prefetching working: next/prev navigation instant
  - Cache eviction correct: LRU policy
  - Sort order changes tested successfully
```

---

### Task 3.2: Implement Background Prefetching Thread
**File:** `models/gather_navigator.py`

**Implementation Requirements:**
- Create background thread that prefetches adjacent gathers
- Thread sleeps when no prefetch needed
- Wakes when user navigates to new gather
- Prefetches: previous 2 and next 2 gathers relative to current
- Uses threading.Thread with daemon=True
- Thread-safe: uses locks for cache access
- Graceful shutdown when navigator destroyed
- Does not block main thread or UI

**Test Cases:**
1. **Test: Thread starts on initialization**
   - Setup: Create GatherNavigator with lazy data
   - Expected: Prefetch thread running (check thread.is_alive())
   - Verify: Thread in daemon mode
   - Verify: No CPU usage when idle

2. **Test: Prefetch triggered on navigation**
   - Setup: Navigate to gather 20
   - Expected: Thread prefetches gathers 18, 19, 21, 22
   - Wait: 500ms for prefetch to complete
   - Verify: Cache contains these 5 gathers
   - Verify: Navigation to gather 21 instant (< 20ms)

3. **Test: Thread doesn't block UI**
   - Setup: Large gathers (slow to load)
   - Action: Navigate to new gather and immediately interact with UI
   - Expected: UI remains responsive
   - Verify: Buttons still clickable during prefetch
   - Verify: Main thread not blocked

4. **Test: Thread-safe cache access**
   - Setup: Prefetch thread running
   - Action: Manually navigate rapidly through gathers
   - Expected: No race conditions or crashes
   - Verify: Cache state remains consistent
   - Verify: No duplicate loads

5. **Test: Graceful shutdown**
   - Setup: Navigator with active prefetch thread
   - Action: Delete navigator or close application
   - Expected: Thread stops within 1 second
   - Verify: No zombie threads
   - Verify: Resources cleaned up

6. **Test: Prefetch stops at boundaries**
   - Setup: Current gather = 1 (near start)
   - Expected: Attempts to prefetch 0, 2, 3 (not -1)
   - Verify: No errors for out-of-range indices
   - Similarly test at end boundary

**Completion Criteria:**
- All 6 tests pass
- Prefetching transparent to user
- No UI blocking or lag
- Thread-safe implementation
- Graceful shutdown without leaks
- Improves navigation experience (measured subjectively)

**Expected Success Message:**
```
✓ Task 3.2 Complete: Background prefetching thread implemented
  - Prefetch thread running successfully
  - Navigation performance: next/prev now < 20ms (was 180ms)
  - UI responsiveness maintained during prefetch
  - Thread-safe: tested with rapid navigation
  - Graceful shutdown confirmed
  - Boundary conditions handled correctly
```

---

## Phase 4: Chunked Processing

### Task 4.1: Implement Chunk-based Processor Pipeline
**File:** `processors/chunked_processor.py` (new file)

**Implementation Requirements:**
- Create class `ChunkedProcessor` that processes Zarr data in chunks
- Method: `process(input_zarr_path, output_zarr_path, processor: BaseProcessor, chunk_size=5000, progress_callback=None)`
- Opens input Zarr in read mode, creates output Zarr with same dimensions
- Processes chunks sequentially: load chunk → process → write to output
- Progress callback: `callback(current_trace, total_traces, estimated_time_remaining)`
- Handles chunk boundaries correctly (no edge artifacts)
- For processors requiring overlap (e.g., filters), adds 10% overlap and crops output
- Memory usage: O(chunk_size), not O(total_size)
- Can be cancelled mid-processing (cleanup partial output)

**Test Cases:**
1. **Test: Process simple operation in chunks**
   - Setup: Input Zarr with 10,000 traces, gain processor (multiply by 2)
   - Process: chunk_size=2000
   - Expected: 5 chunks processed
   - Verify: Output[trace] = Input[trace] * 2 for all traces
   - Verify: No discontinuities at chunk boundaries (compare traces 1999-2001)

2. **Test: Chunk boundaries with filter (overlap handling)**
   - Setup: Bandpass filter processor (requires overlap)
   - Input: Known signal pattern
   - Process: chunk_size=1000 with overlap
   - Expected: No artifacts at boundaries
   - Verify: Trace 999 and 1001 show smooth filtered response (no edge effects)

3. **Test: Progress callback accuracy**
   - Setup: 5000 traces, chunk_size=1000
   - Monitor: Progress callbacks
   - Expected: 5 callbacks at 1000, 2000, 3000, 4000, 5000
   - Verify: Progress percentage accurate (20%, 40%, 60%, 80%, 100%)
   - Verify: Estimated time remaining decreases

4. **Test: Memory usage bounded**
   - Setup: 50,000 trace dataset
   - Process: chunk_size=5000
   - Monitor: Peak memory during processing
   - Verify: Memory < chunk_size × sample_size × 4 (factor of 4 for input+output+processing buffers)
   - Verify: Memory doesn't grow with more chunks

5. **Test: Cancel mid-processing**
   - Setup: Start processing 10,000 traces
   - Action: Cancel after 40% complete
   - Expected: Processing stops
   - Verify: Partial output Zarr deleted or marked incomplete
   - Verify: No orphaned file handles

6. **Test: Output Zarr matches input dimensions**
   - Setup: Input (5000 samples, 8000 traces)
   - Process: Any processor
   - Expected: Output (5000 samples, 8000 traces)
   - Verify: Sample rate preserved in metadata
   - Verify: Output is valid Zarr (can be opened and read)

**Completion Criteria:**
- All 6 tests pass
- No artifacts at chunk boundaries
- Memory usage O(chunk_size) confirmed
- Progress tracking accurate (±2%)
- Cancel operation clean
- Works with all processor types (filters, gains, etc.)

**Expected Success Message:**
```
✓ Task 4.1 Complete: Chunk-based processor pipeline implemented
  - Chunked processing functional
  - Memory usage: 85MB for 50k trace dataset (chunk_size=5k)
  - No boundary artifacts: verified with filter tests
  - Progress tracking: ±1% accuracy
  - Cancel operation tested: clean shutdown
  - Compatible with all existing processors
```

---

### Task 4.2: Integrate Chunked Processing into Main Window
**File:** `main_window.py`

**Implementation Requirements:**
- Modify `_on_process_requested()` to detect if data is lazy-loaded
- If lazy: use ChunkedProcessor with Zarr paths
- If in-memory: use existing processing pipeline (backward compatibility)
- Progress dialog shows: chunk progress, trace count, time remaining
- Processing runs in background thread (QThread)
- UI remains responsive during processing
- Cancel button stops processing immediately
- Success: automatically loads processed data into viewer
- Option: "Process all gathers" for batch processing entire dataset

**Test Cases:**
1. **Test: In-memory data uses existing pipeline**
   - Setup: Load small dataset (1000 traces) in memory
   - Action: Apply bandpass filter
   - Expected: Uses existing `processor.process()` method
   - Verify: Processing completes
   - Verify: Processed data displayed

2. **Test: Lazy data uses chunked processor**
   - Setup: Load large dataset as lazy (50,000 traces)
   - Action: Apply bandpass filter
   - Expected: Uses ChunkedProcessor
   - Verify: Progress dialog shows chunk updates
   - Verify: Memory usage < 300MB during processing

3. **Test: Progress dialog updates**
   - Setup: Process 20,000 traces in chunks of 5000
   - Monitor: Progress dialog
   - Expected: Updates every chunk (0%, 25%, 50%, 75%, 100%)
   - Verify: Time remaining updates and decreases
   - Verify: Trace count shown (e.g., "Processing traces 5000-10000")

4. **Test: UI responsive during processing**
   - Setup: Start processing large dataset
   - Action: Interact with UI (click buttons, menus)
   - Expected: UI responds immediately
   - Verify: Application not frozen
   - Verify: Processing continues in background

5. **Test: Cancel stops processing**
   - Setup: Start processing 30,000 traces
   - Action: Click cancel at 40%
   - Expected: Processing stops within 2 seconds
   - Verify: Partial results discarded
   - Verify: Original data still viewable

6. **Test: Successful processing loads result**
   - Setup: Process dataset with filter
   - Expected: Completion dialog shown
   - Action: Click OK
   - Verify: Processed data loaded into "Processed" viewer
   - Verify: Can navigate through processed data
   - Verify: Difference viewer updated

7. **Test: Process all gathers batch mode**
   - Setup: Dataset with 50 gathers
   - Action: Click "Process All Gathers"
   - Expected: All 50 gathers processed with same settings
   - Verify: Progress shows gather progress (e.g., "Gather 25/50")
   - Verify: Final result accessible via gather navigation

**Completion Criteria:**
- All 7 tests pass
- Automatic selection of processing mode
- UI remains responsive during processing
- Progress dialog informative and accurate
- Cancel works cleanly
- Processed results displayed correctly
- Batch processing functional

**Expected Success Message:**
```
✓ Task 4.2 Complete: Chunked processing integrated into main window
  - Auto-detection of lazy vs in-memory data working
  - Progress dialog functional with detailed updates
  - UI responsiveness maintained: tested with large dataset
  - Cancel operation: stops within 1.5s average
  - Processed data loading correctly
  - Batch processing: tested with 50 gathers
  - Memory usage: 280MB for 50k trace processing
```

---

## Phase 5: SEGY Export for Large Files

### Task 5.1: Implement Chunked SEGY Exporter
**File:** `utils/segy_import/segy_export.py`

**Implementation Requirements:**
- Add method `export_from_zarr_chunked(original_segy_path, processed_zarr_path, output_segy_path, chunk_size=5000, progress_callback=None)`
- Opens original SEGY for header reading
- Creates output SEGY with same spec
- Copies binary and text headers
- Processes traces in chunks: read headers → read processed traces → write to output
- Progress callback after each chunk
- Memory usage: O(chunk_size)
- Handles large files (> RAM size)
- Preserves all trace headers from original
- Output SEGY valid and readable by standard tools

**Test Cases:**
1. **Test: Export matches input dimensions**
   - Setup: Original SEGY (10,000 traces), processed Zarr (10,000 traces)
   - Export: chunk_size=2000
   - Expected: Output SEGY with 10,000 traces
   - Verify: Trace count matches
   - Verify: Sample count per trace matches

2. **Test: Headers preserved**
   - Setup: Original SEGY with known CDP, offset values
   - Export: Processed Zarr to new SEGY
   - Verify: Output trace 100 has same CDP as original trace 100
   - Verify: All standard headers preserved (offset, coordinates, etc.)

3. **Test: Trace data matches processed**
   - Setup: Processed Zarr with known values (e.g., trace[i] = i × 2)
   - Export: To SEGY
   - Read back: Output SEGY
   - Verify: Trace data matches Zarr data (compare trace 0, 5000, 9999)
   - Verify: No data corruption or truncation

4. **Test: Chunk boundaries correct**
   - Setup: 5000 traces exported in chunks of 1000
   - Focus: Traces at boundaries (999-1001, 1999-2001, etc.)
   - Verify: No discontinuities or duplicated traces
   - Verify: Sequential trace numbering preserved

5. **Test: Progress callback accuracy**
   - Setup: 10,000 traces, chunk_size=2500
   - Monitor: Progress callbacks
   - Expected: 4 callbacks at 25%, 50%, 75%, 100%
   - Verify: Accuracy ±1%

6. **Test: Memory usage bounded**
   - Setup: 50,000 trace export
   - Monitor: Peak memory during export
   - Verify: Memory < 200MB regardless of dataset size
   - Verify: Memory doesn't grow with more chunks

7. **Test: Binary and text headers preserved**
   - Setup: Original SEGY with custom text header
   - Export: Processed data
   - Read back: Output SEGY
   - Verify: Text header matches original
   - Verify: Binary header fields match (job ID, line number, etc.)

8. **Test: Output readable by external tools**
   - Setup: Export processed data to SEGY
   - Action: Open with segyio independently
   - Verify: File opens without errors
   - Verify: Trace count correct
   - Verify: Can read traces and headers

**Completion Criteria:**
- All 8 tests pass
- Headers perfectly preserved
- Trace data matches processed Zarr
- Memory usage O(chunk_size)
- Output valid SEGY format
- Compatible with industry standard tools

**Expected Success Message:**
```
✓ Task 5.1 Complete: Chunked SEGY exporter implemented
  - Chunked export functional
  - Memory usage: 125MB for 50k trace export (chunk_size=5k)
  - Headers preserved: 100% match verified
  - Trace data accurate: spot checks passed
  - No boundary artifacts detected
  - Output SEGY validated with segyio and commercial software
  - Progress tracking: ±0.8% accuracy
```

---

### Task 5.2: Integrate Chunked Export into Main Window
**File:** `main_window.py`

**Implementation Requirements:**
- Add menu item: "File → Export Processed to SEGY..."
- Detects if processed data is lazy-loaded or in-memory
- If lazy: use chunked exporter with progress dialog
- If in-memory: use existing exporter (backward compatibility)
- Dialog: select output path, show options (chunk size for advanced users)
- Progress dialog: shows export progress, traces exported, time remaining
- Runs in background thread (non-blocking UI)
- Cancel button stops export (cleanup partial file)
- Success dialog: "Exported 50,000 traces to output.sgy in 2m 15s"
- Validates original SEGY path available before export

**Test Cases:**
1. **Test: Menu item enabled when processed data available**
   - Setup: Load and process data
   - Check: "Export Processed to SEGY..." menu item
   - Expected: Enabled (not grayed out)
   - Action: Click menu item
   - Expected: Export dialog opens

2. **Test: Menu item disabled when no processed data**
   - Setup: Only input data loaded (no processing done)
   - Check: Export menu item
   - Expected: Disabled (grayed out)

3. **Test: Export dialog file selection**
   - Setup: Processed data available
   - Action: Open export dialog, select output path
   - Expected: File dialog opens
   - Verify: Default filename suggests based on input (e.g., "input_processed.sgy")

4. **Test: Lazy data uses chunked export**
   - Setup: Lazy-loaded processed data (50,000 traces)
   - Action: Export to SEGY
   - Expected: Chunked exporter used
   - Verify: Progress dialog shows chunk updates
   - Verify: Memory < 300MB during export

5. **Test: In-memory data uses standard export**
   - Setup: Small processed dataset in memory (1000 traces)
   - Action: Export to SEGY
   - Expected: Standard exporter used (faster)
   - Verify: Export completes quickly

6. **Test: Progress dialog functional**
   - Setup: Export 20,000 traces
   - Monitor: Progress dialog
   - Expected: Shows progress 0% → 100%
   - Verify: Traces exported count updates
   - Verify: Time remaining shown and decreases

7. **Test: UI responsive during export**
   - Setup: Start export of large dataset
   - Action: Interact with UI (open menus, click buttons)
   - Expected: UI remains responsive
   - Verify: No freezing or lag

8. **Test: Cancel stops export**
   - Setup: Start export
   - Action: Click cancel at 50%
   - Expected: Export stops within 2 seconds
   - Verify: Partial output file deleted
   - Verify: No error dialogs

9. **Test: Success dialog shows metrics**
   - Setup: Complete export successfully
   - Expected: Dialog shows: trace count, file size, time taken
   - Verify: Metrics accurate

10. **Test: Missing original SEGY path handled**
    - Setup: Processed data but original SEGY file moved/deleted
    - Action: Attempt export
    - Expected: Error dialog: "Original SEGY file not found for headers"
    - Verify: No crash or corrupt export

**Completion Criteria:**
- All 10 tests pass
- Menu item state correct (enabled/disabled)
- Export dialog user-friendly
- Automatic mode selection working
- Progress tracking functional
- Cancel works cleanly
- Error handling for missing files

**Expected Success Message:**
```
✓ Task 5.2 Complete: Chunked export integrated into main window
  - Export menu integration complete
  - Auto-detection of lazy vs in-memory working
  - File dialog with smart default naming
  - Progress tracking: detailed and accurate
  - UI responsiveness maintained during export
  - Cancel operation: clean with file cleanup
  - Error handling: tested missing original file
  - Successfully exported 50k trace test file
```

---

## Phase 6: UI/UX Improvements

### Task 6.1: Implement Adaptive Progress Indicators
**File:** `views/progress_dialog.py` (new file)

**Implementation Requirements:**
- Create custom QProgressDialog subclass: `AdaptiveProgressDialog`
- Shows: operation name, current item, total items, percentage, elapsed time, ETA
- Updates: every 100ms minimum (avoid UI flicker)
- ETA calculation: exponential moving average of last 10 chunks
- Visual elements: progress bar, status text, cancel button
- Auto-closes on completion (after 1 second) or user can close immediately
- Supports indeterminate mode (when total unknown)
- Cancellation: sets flag that caller can check

**Test Cases:**
1. **Test: Progress updates correctly**
   - Setup: Dialog for 100 items
   - Simulate: Update progress to 25, 50, 75, 100
   - Expected: Bar shows 25%, 50%, 75%, 100%
   - Verify: Percentage text matches bar

2. **Test: ETA calculation accuracy**
   - Setup: Simulate processing at constant rate (10 items/sec)
   - Progress: 20 items complete, 80 remaining
   - Expected: ETA shows ~8 seconds
   - Verify: ETA updates as processing continues

3. **Test: ETA handles variable rates**
   - Setup: First 50 items fast (20 items/sec), next 50 slow (5 items/sec)
   - Expected: ETA adapts (increases when rate slows)
   - Verify: Uses moving average, not linear projection

4. **Test: Status text updates**
   - Setup: Dialog with status updates
   - Update: "Processing chunk 5 of 10"
   - Expected: Status label shows text
   - Verify: Updates in real-time

5. **Test: Cancel button sets flag**
   - Setup: Dialog showing progress
   - Action: Click cancel
   - Expected: is_cancelled() returns True
   - Verify: Progress dialog remains open (for caller to clean up)

6. **Test: Auto-close on completion**
   - Setup: Progress reaches 100%
   - Expected: Dialog closes after 1 second
   - Verify: Can close immediately with close button

7. **Test: Indeterminate mode**
   - Setup: Create dialog with total=0 (unknown)
   - Expected: Progress bar in indeterminate/busy mode
   - Verify: No percentage shown
   - Verify: Cancel still works

**Completion Criteria:**
- All 7 tests pass
- ETA accuracy within 20% for constant-rate tasks
- Updates smooth (no flicker)
- Cancel mechanism reliable
- Professional appearance

**Expected Success Message:**
```
✓ Task 6.1 Complete: Adaptive progress indicators implemented
  - Custom progress dialog created
  - ETA calculation: within 15% accuracy for constant-rate tasks
  - ETA adapts to variable rates: tested and verified
  - Visual updates smooth: 100ms refresh rate
  - Cancel mechanism tested and reliable
  - Auto-close functionality working
  - Indeterminate mode functional
```

---

### Task 6.2: Implement Memory Usage Monitor
**File:** `utils/memory_monitor.py` (new file)

**Implementation Requirements:**
- Create class `MemoryMonitor` that tracks application memory usage
- Method: `get_current_usage() -> int` (returns bytes)
- Method: `get_available_memory() -> int` (system available RAM)
- Method: `get_usage_percentage() -> float` (app memory / total system memory)
- Updates every 2 seconds in background thread
- Emits signal when usage exceeds threshold (e.g., 80% of available)
- Low overhead: < 0.1% CPU usage
- Platform-independent (works on Linux, macOS, Windows)

**Test Cases:**
1. **Test: Current usage reported**
   - Setup: Create MemoryMonitor
   - Action: Allocate 100MB numpy array
   - Check: get_current_usage()
   - Expected: Usage increased by ~100MB (±20%)

2. **Test: Available memory reasonable**
   - Check: get_available_memory()
   - Expected: Returns positive value < total system RAM
   - Verify: Value makes sense for current system

3. **Test: Usage percentage calculated**
   - Setup: Known memory usage
   - Check: get_usage_percentage()
   - Expected: Percentage = (app_usage / total_ram) × 100
   - Verify: Value between 0 and 100

4. **Test: Threshold signal emitted**
   - Setup: Monitor with 50MB threshold
   - Action: Allocate 60MB
   - Expected: Threshold exceeded signal emitted
   - Verify: Signal emitted once (not repeatedly)

5. **Test: Background thread low overhead**
   - Setup: Start monitor
   - Measure: CPU usage over 30 seconds
   - Expected: Monitor thread uses < 0.5% CPU
   - Verify: Doesn't impact application performance

6. **Test: Thread cleanup**
   - Setup: Create and destroy monitor multiple times
   - Expected: No thread leaks
   - Verify: Threads terminate when monitor destroyed

**Completion Criteria:**
- All 6 tests pass
- Accurate memory reporting (±10%)
- Low CPU overhead (< 0.1%)
- Platform-independent (tested on at least 2 platforms)
- Clean thread management

**Expected Success Message:**
```
✓ Task 6.2 Complete: Memory usage monitor implemented
  - Memory tracking functional
  - Accuracy: ±8% verified with known allocations
  - CPU overhead: 0.05% average
  - Threshold signals working correctly
  - Background thread management clean
  - Tested on Linux and Windows platforms
```

---

### Task 6.3: Integrate Memory Monitor into Status Bar
**File:** `main_window.py`

**Implementation Requirements:**
- Add MemoryMonitor instance to main window
- Status bar shows: "Memory: 450 MB / 16 GB (2.8%)"
- Updates every 2 seconds
- Color codes: green (< 50%), yellow (50-80%), red (> 80%)
- Warning dialog if memory exceeds 90%: "High memory usage detected. Consider closing other applications."
- Memory display can be clicked to show detailed breakdown (cache size, window size, etc.)
- Option to clear caches from memory dialog

**Test Cases:**
1. **Test: Memory display in status bar**
   - Setup: Launch application
   - Check: Status bar
   - Expected: Shows memory usage (e.g., "Memory: 250 MB / 8 GB (3.1%)")
   - Verify: Values reasonable and update every 2 seconds

2. **Test: Color coding correct**
   - Simulate: Memory at 25%, 60%, 85%
   - Expected: Green at 25%, yellow at 60%, red at 85%
   - Verify: Color changes match thresholds

3. **Test: Warning dialog at 90%**
   - Simulate: Memory usage reaches 90%
   - Expected: Warning dialog appears
   - Verify: Dialog shows once (not repeatedly)
   - Verify: User can dismiss

4. **Test: Click for detailed view**
   - Action: Click memory display in status bar
   - Expected: Dialog shows breakdown:
     - Window cache: 120 MB
     - Ensemble cache: 45 MB
     - Current data: 180 MB
     - Other: 105 MB
   - Verify: Values sum to total shown

5. **Test: Clear cache functionality**
   - Setup: Caches populated
   - Action: Click "Clear Caches" in detail dialog
   - Expected: Memory usage decreases
   - Verify: Cache sizes reset to 0 MB in detail view

6. **Test: Performance impact minimal**
   - Setup: Application running with monitor
   - Measure: UI responsiveness (frame rate, interaction lag)
   - Expected: No noticeable impact
   - Verify: Memory updates don't cause UI stutters

**Completion Criteria:**
- All 6 tests pass
- Status bar display clear and updating
- Color coding intuitive
- Warning dialog helpful
- Detailed view informative
- Cache clearing functional

**Expected Success Message:**
```
✓ Task 6.3 Complete: Memory monitor integrated into status bar
  - Status bar display functional and updating
  - Color coding: green/yellow/red thresholds working
  - Warning dialog: triggered correctly at 90%
  - Detailed view: shows cache breakdown accurately
  - Clear cache: reduces memory by expected amount
  - Performance: no UI impact detected
  - User testing: feedback positive
```

---

### Task 6.4: Implement Large File Detection and Recommendations
**File:** `views/segy_import_dialog.py`

**Implementation Requirements:**
- On file selection, check file size before import
- Thresholds:
  - Small: < 500 MB → Standard import recommended
  - Medium: 500 MB - 2 GB → Streaming import, warning shown
  - Large: > 2 GB → Streaming import required, recommendations shown
- For large files, show dialog: "Large file detected (5.2 GB). Recommendations:
  - Streaming import will be used (memory-efficient)
  - Estimated import time: 8 minutes
  - Required disk space: ~6 GB (compressed)
  - Import first 10,000 traces for testing? [Yes] [No, import all]"
- Option to preview/test with subset before full import
- Estimates based on file size and system specs

**Test Cases:**
1. **Test: Small file no warning**
   - Select: 200 MB SEGY file
   - Expected: No special dialog, proceeds to import settings
   - Verify: Standard import used

2. **Test: Medium file warning**
   - Select: 1.2 GB SEGY file
   - Expected: Info dialog: "Medium-sized file detected. Streaming import will be used."
   - Verify: User can proceed
   - Verify: Streaming import used

3. **Test: Large file recommendations**
   - Select: 5 GB SEGY file
   - Expected: Detailed dialog with estimates
   - Verify: Shows file size, import time estimate, disk space required
   - Verify: Offers subset import option

4. **Test: Subset import option**
   - Select: Large file, choose "Import first 10,000 traces"
   - Expected: Import proceeds with limit
   - Verify: Only 10,000 traces imported
   - Verify: User notified this is a subset

5. **Test: Time estimate accuracy**
   - Import: Known file with known import speed
   - Compare: Estimated time vs actual time
   - Expected: Within 30% accuracy
   - Verify: Estimate shown to user

6. **Test: Disk space check**
   - Select: File requiring 10 GB disk space
   - Available: Only 5 GB free
   - Expected: Warning: "Insufficient disk space. Required: 10 GB, Available: 5 GB"
   - Verify: User cannot proceed without clearing space

**Completion Criteria:**
- All 6 tests pass
- File size detection accurate
- Thresholds appropriate
- Recommendations helpful
- Time estimates within 30% accuracy
- Disk space check prevents errors

**Expected Success Message:**
```
✓ Task 6.4 Complete: Large file detection and recommendations implemented
  - File size detection functional
  - Threshold-based recommendations working
  - Large file dialog: informative and helpful
  - Subset import option tested successfully
  - Time estimates: within 25% accuracy on average
  - Disk space check: prevents insufficient space errors
  - User experience improved significantly
```

---

## Integration Testing

### Task 7.1: End-to-End Test - Import Large File
**Scope:** Full workflow from file selection to viewing

**Test Scenario:**
- Input: 10 GB SEGY file with 100,000 traces, 1000 samples, 2ms sampling
- Expected behavior:
  1. File size detected: triggers streaming import
  2. Progress dialog shows: chunks processing with ETA
  3. Import completes in < 15 minutes
  4. Data saved to Zarr: ~3-4 GB compressed
  5. Headers saved to Parquet: < 50 MB
  6. Ensemble index created: 1000 ensembles
  7. Data loads in viewer as lazy
  8. Memory usage during import: < 500 MB peak
  9. Memory usage after load: < 200 MB
  10. Can navigate through ensembles smoothly

**Success Criteria:**
- All 10 steps complete successfully
- No errors or crashes
- Performance targets met
- Memory usage within limits
- Data integrity verified (spot checks)

**Expected Success Message:**
```
✓ Task 7.1 Complete: End-to-end import test passed
  - 10 GB file imported successfully
  - Import time: 12 minutes 35 seconds
  - Compression: 3.5x (2.86 GB output)
  - Peak memory: 425 MB during import
  - Viewing memory: 165 MB after load
  - Navigation tested: instant response
  - Spot checks: 20 random traces verified correct
```

---

### Task 7.2: End-to-End Test - Process Large Dataset
**Scope:** Full processing workflow on large lazy-loaded data

**Test Scenario:**
- Input: Lazy-loaded 50,000 trace dataset (from Task 7.1)
- Processor: Bandpass filter 10-80 Hz
- Expected behavior:
  1. Processing uses chunked pipeline
  2. Progress dialog shows: chunk progress, ETA
  3. Processing completes in < 10 minutes
  4. Memory usage during processing: < 400 MB
  5. Output saved to new Zarr array
  6. Can view processed data in viewer
  7. Difference view functional
  8. Can navigate processed gathers
  9. No data corruption (spot checks)
  10. Can export processed to SEGY

**Success Criteria:**
- All 10 steps complete successfully
- Processing accurate (compare subset with reference)
- Performance acceptable
- Memory bounded
- Output valid

**Expected Success Message:**
```
✓ Task 7.2 Complete: End-to-end processing test passed
  - 50k traces processed successfully
  - Processing time: 8 minutes 15 seconds
  - Peak memory: 340 MB during processing
  - Output verified: filter response correct
  - Viewing processed data: smooth navigation
  - Difference view: shows expected patterns
  - Spot checks: 15 traces verified against reference
```

---

### Task 7.3: End-to-End Test - Export Large Dataset
**Scope:** Export processed large dataset back to SEGY

**Test Scenario:**
- Input: Processed 50,000 trace Zarr from Task 7.2
- Expected behavior:
  1. Export dialog opens with options
  2. Chunked export used automatically
  3. Progress dialog shows: export progress, ETA
  4. Export completes in < 8 minutes
  5. Memory usage: < 300 MB
  6. Output SEGY file size: ~2 GB
  7. Output readable by segyio
  8. Spot check: 20 traces match processed Zarr
  9. Headers preserved from original
  10. File validates with commercial software

**Success Criteria:**
- All 10 steps complete
- Export accurate
- Performance acceptable
- Memory bounded
- Output valid SEGY

**Expected Success Message:**
```
✓ Task 7.3 Complete: End-to-end export test passed
  - 50k traces exported successfully
  - Export time: 6 minutes 48 seconds
  - Peak memory: 265 MB during export
  - Output file: 2.1 GB
  - Validated with segyio: all traces readable
  - Headers verified: 100% match original
  - Commercial software test: file opens correctly
  - Spot checks: 20 traces match Zarr exactly
```

---

### Task 7.4: Stress Test - Very Large File (100 GB+)
**Scope:** Test system limits with very large file

**Test Scenario:**
- Input: 100 GB SEGY file (or simulated) with 1,000,000 traces
- System: Standard laptop (16 GB RAM, 1 TB disk)
- Expected behavior:
  1. Import succeeds without memory errors
  2. Import time proportional to file size (< 3 hours)
  3. Peak memory < 1 GB during any operation
  4. Can navigate through data smoothly
  5. Processing completes successfully
  6. Export completes successfully
  7. System remains stable throughout
  8. Disk space managed efficiently

**Success Criteria:**
- All operations complete without errors
- Memory usage never exceeds 1 GB
- System stable (no crashes, hangs)
- Performance acceptable (< 4 hours total workflow)

**Expected Success Message:**
```
✓ Task 7.4 Complete: Stress test passed
  - 100 GB file (1M traces) processed successfully
  - Import: 2 hours 45 minutes
  - Peak memory: 850 MB (import phase)
  - Navigation: smooth with prefetching
  - Processing: 1 hour 20 minutes (full dataset)
  - Export: 1 hour 35 minutes
  - Total workflow: 5 hours 40 minutes
  - System stable throughout
  - No memory errors, crashes, or data loss
```

---

## Summary

**Total Tasks:** 30 concrete, testable tasks across 7 phases
- Phase 1: 5 tasks (Chunked SEGY import)
- Phase 2: 4 tasks (Lazy data loading)
- Phase 3: 2 tasks (Gather navigation)
- Phase 4: 2 tasks (Chunked processing)
- Phase 5: 2 tasks (SEGY export)
- Phase 6: 4 tasks (UI/UX)
- Integration: 4 tasks (End-to-end tests)

**Estimated Implementation Time:** 6-8 weeks (1 developer)
- Phase 1: 2 weeks
- Phase 2: 1.5 weeks
- Phase 3: 1 week
- Phase 4: 1 week
- Phase 5: 0.5 weeks
- Phase 6: 1 week
- Integration/Testing: 1 week

**Critical Success Metrics:**
- Memory usage: O(1) regardless of file size - target < 500 MB peak
- Import speed: ~10,000 traces/second
- Navigation: < 50ms for cached, < 300ms for uncached gathers
- Processing: < 2x overhead vs in-memory processing
- Export: Matches import speed

**Risk Mitigation:**
- Each task has clear completion criteria
- Tests defined before implementation
- Backward compatibility maintained
- Incremental deployment possible
- Can fallback to existing methods if issues arise
