# ADR-002: Zarr for Intermediate Data Storage

## Status
Accepted

## Date
2024-01-20

## Context
SeisProc processes SEG-Y files that can be 1-50+ GB. We need:
- Memory-efficient storage during processing
- Fast random access to arbitrary trace windows
- Persistence across sessions (resume interrupted work)
- Support for lazy loading (only load visible data)

Options considered:
1. **In-memory NumPy arrays** - Simple but memory-limited
2. **HDF5** - Industry standard, mature
3. **Zarr** - Modern chunked array storage
4. **Memory-mapped files** - OS-level caching
5. **Parquet** - Columnar storage for headers

## Decision
Use **Zarr** for trace data storage and **Parquet** for trace headers.

## Rationale

### Zarr for Traces

#### Chunked Storage
- Data split into chunks (e.g., 1000 traces × 2000 samples)
- Only needed chunks loaded into memory
- Optimal for windowed access patterns in seismic

#### Compression
- LZ4 compression reduces storage by 2-4x
- Blosc compressor with multithreading
- Transparent: code sees uncompressed arrays

#### Cloud-Ready
- Works with local files, S3, GCS, Azure
- Designed for distributed/parallel access
- Future-proof for cloud deployment

#### Python-Native
- Pure Python, easy to install (no HDF5 library)
- NumPy array interface (zero-copy when possible)
- Good pandas integration

### Parquet for Headers

#### Columnar Format
- Efficient for queries like "all FFID=100 traces"
- Compression per column (e.g., delta encoding for sequential headers)
- Fast aggregations and filtering

#### Pandas Integration
- Direct `pd.read_parquet()` returns DataFrame
- Preserves dtypes (int32 for headers)
- Predicate pushdown for efficient filtering

## Implementation

```python
# Trace data storage
traces = zarr.open('data.zarr', mode='w', shape=(n_samples, n_traces),
                   chunks=(n_samples, 1000), dtype='float32',
                   compressor=Blosc(cname='lz4', clevel=5))

# Header storage
headers_df.to_parquet('headers.parquet', engine='pyarrow')
```

## Alternatives Considered

### HDF5
- **Pros**: Mature, widely used, good compression
- **Cons**: Complex library dependencies, GIL issues in Python, single-writer limitation
- **Verdict**: Zarr is more Pythonic and cloud-friendly

### Memory-Mapped NumPy
- **Pros**: Simple, OS handles caching
- **Cons**: No compression, fixed dtype, platform differences
- **Verdict**: Good for small files but doesn't scale

### In-Memory Only
- **Pros**: Simplest implementation
- **Cons**: Limits dataset size to available RAM
- **Verdict**: Kept as fallback for small datasets

## Consequences

### Positive
- Process 20GB files on 16GB RAM machines
- Session persistence: reload partially processed data
- Cloud deployment possible without code changes
- ~3x storage reduction with LZ4 compression

### Negative
- Additional dependencies (zarr, pyarrow)
- Initial import time to convert SEG-Y → Zarr
- Chunk size tuning needed for optimal performance

### Performance Metrics
| Operation | Time (50K traces, 3000 samples) |
|-----------|--------------------------------|
| SEG-Y → Zarr import | ~45 seconds |
| Load 1000-trace window | ~50ms |
| Full gather iteration | ~2 seconds |

## References
- Zarr documentation: https://zarr.readthedocs.io/
- Blosc compression: https://www.blosc.org/
- `utils/segy_import/data_storage.py` implementation
