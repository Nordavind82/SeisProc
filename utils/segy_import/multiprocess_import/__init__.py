"""
Multiprocess SEGY import for bypassing Python GIL.

Uses process-based parallelism to fully utilize multi-core CPUs.
Each worker process reads a segment and writes directly to a shared Zarr array.
Only header parquet files need merging after workers complete.
"""

from .partitioner import SmartPartitioner, Segment, PartitionConfig
from .worker import import_segment, WorkerTask, WorkerResult
from .coordinator import ParallelImportCoordinator, ImportConfig, ImportProgress, ImportResult

__all__ = [
    # Partitioner
    'SmartPartitioner',
    'Segment',
    'PartitionConfig',
    # Worker
    'import_segment',
    'WorkerTask',
    'WorkerResult',
    # Coordinator
    'ParallelImportCoordinator',
    'ImportConfig',
    'ImportProgress',
    'ImportResult',
]
