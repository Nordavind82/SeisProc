"""
Parallel SEG-Y export infrastructure.

This package provides high-performance multiprocess SEG-Y export
that bypasses the Python GIL for significant speedup on multi-core systems.

Main components:
- HeaderVectorizer: Fast vectorized header access
- ExportWorker: Worker function for parallel export
- ParallelExportCoordinator: Orchestrates parallel export
- SEGYSegmentMerger: Combines segment files
"""

from .config import (
    ExportConfig,
    ExportTask,
    ExportWorkerResult,
    ExportProgress,
    ExportResult,
    TraceSegment
)
from .header_vectorizer import HeaderVectorizer, vectorize_headers, get_trace_headers
from .worker import export_trace_range
from .coordinator import ParallelExportCoordinator, ExportStageResult
from .merger import SEGYSegmentMerger

__all__ = [
    # Config
    'ExportConfig',
    'ExportTask',
    'ExportWorkerResult',
    'ExportProgress',
    'ExportResult',
    'TraceSegment',
    # Header vectorization
    'HeaderVectorizer',
    'vectorize_headers',
    'get_trace_headers',
    # Worker
    'export_trace_range',
    # Coordinator
    'ParallelExportCoordinator',
    'ExportStageResult',
    # Merger
    'SEGYSegmentMerger',
]
