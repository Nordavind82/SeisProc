"""
Parallel processing for seismic data using multiprocessing.

Bypasses Python GIL by using ProcessPoolExecutor for true parallelism.
Each worker process handles a range of gathers, writing directly to
a shared output Zarr array.

Usage:
    from utils.parallel_processing import (
        ParallelProcessingCoordinator,
        ProcessingConfig,
        ProcessingResult
    )

    config = ProcessingConfig(
        input_storage_dir='/path/to/input',
        output_storage_dir='/path/to/output',
        processor_config=processor.to_dict()
    )

    coordinator = ParallelProcessingCoordinator(config)
    result = coordinator.run(progress_callback=update_ui)
"""

from .config import ProcessingConfig, ProcessingTask, ProcessingWorkerResult, ProcessingResult, ProcessingProgress, SortOptions
from .partitioner import GatherPartitioner, GatherSegment
from .worker import process_gather_range
from .coordinator import ParallelProcessingCoordinator, get_optimal_workers

__all__ = [
    # Configuration
    'ProcessingConfig',
    'ProcessingTask',
    'ProcessingWorkerResult',
    'ProcessingResult',
    'ProcessingProgress',
    'SortOptions',
    # Partitioner
    'GatherPartitioner',
    'GatherSegment',
    # Worker
    'process_gather_range',
    # Coordinator
    'ParallelProcessingCoordinator',
    'get_optimal_workers',
]
