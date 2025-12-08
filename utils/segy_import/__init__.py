"""
SEG-Y import/export utilities with custom header mapping and Zarr/Parquet storage.

Performance optimizations:
- Memory mapping (mmap) for faster I/O
- Pre-allocated arrays to avoid memory fragmentation
- Batch header reading using segyio attributes
- Buffer pooling for chunked operations
- Configurable chunk sizes for memory/performance tuning
- Multiprocess parallel import for large files (bypasses GIL)
"""

# Performance configuration constants
DEFAULT_CHUNK_SIZE = 10000          # Default traces per chunk (balanced)
SMALL_MEMORY_CHUNK_SIZE = 5000      # For systems with <16GB RAM
LARGE_MEMORY_CHUNK_SIZE = 25000     # For systems with 32GB+ RAM
HEADER_BATCH_SIZE = 10000           # Headers per Parquet write batch
PARALLEL_THRESHOLD = 100000         # Use parallel import for files > this many traces

from .header_mapping import HeaderMapping, StandardHeaders
from .segy_reader import SEGYReader, SEGYFileHandle, CancellationToken, OperationCancelledError, LoadingProgress
from .segy_reader_fast import FastSEGYReader, create_segy_reader, is_fast_reader_available
from .segy_export import SEGYExporter, AsyncSEGYExporter
from .data_storage import DataStorage
from .computed_headers import ComputedHeaderField, ComputedHeaderEvaluator, ComputedHeaderProcessor

# Multiprocess import (for large files)
from .multiprocess_import import (
    ParallelImportCoordinator,
    ImportConfig,
    ImportProgress,
    ImportResult,
)
from .multiprocess_import.coordinator import get_optimal_workers

__all__ = [
    # Configuration constants
    'DEFAULT_CHUNK_SIZE',
    'SMALL_MEMORY_CHUNK_SIZE',
    'LARGE_MEMORY_CHUNK_SIZE',
    'HEADER_BATCH_SIZE',
    'PARALLEL_THRESHOLD',
    # Header mapping
    'HeaderMapping',
    'StandardHeaders',
    # Standard reader (optimized with mmap + batch headers)
    'SEGYReader',
    'SEGYFileHandle',
    # Fast reader (uses segfast if available)
    'FastSEGYReader',
    'create_segy_reader',
    'is_fast_reader_available',
    # Multiprocess import (for large files)
    'ParallelImportCoordinator',
    'ImportConfig',
    'ImportProgress',
    'ImportResult',
    'get_optimal_workers',
    # Utilities
    'CancellationToken',
    'OperationCancelledError',
    'LoadingProgress',
    # Export
    'SEGYExporter',
    'AsyncSEGYExporter',
    # Storage
    'DataStorage',
    # Computed headers
    'ComputedHeaderField',
    'ComputedHeaderEvaluator',
    'ComputedHeaderProcessor',
]
