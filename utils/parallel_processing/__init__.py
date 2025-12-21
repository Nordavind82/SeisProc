"""
DEPRECATED: Legacy parallel processing module using ProcessPoolExecutor.

This module is deprecated and will be removed in a future version.
Use utils.ray_orchestration instead, which provides:
- Ray-based distributed processing with proper resource management
- Metal GPU acceleration via kernel_backend
- Cancellation, pause/resume support
- Checkpointing and fault tolerance

Migration:
    # Old (deprecated):
    from utils.parallel_processing import ParallelProcessingCoordinator

    # New (recommended):
    from utils.ray_orchestration import RayProcessingCoordinator

The Ray-based system provides 2-3x better performance through:
- Proper CPU resource allocation (no over-subscription)
- Zero-copy data sharing via Ray object store
- Dynamic load balancing
"""
import warnings

# Emit deprecation warning on import
warnings.warn(
    "utils.parallel_processing is deprecated. "
    "Use utils.ray_orchestration.RayProcessingCoordinator instead. "
    "This module will be removed in version 2.0.",
    DeprecationWarning,
    stacklevel=2
)

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
