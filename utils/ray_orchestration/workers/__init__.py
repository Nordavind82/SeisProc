"""
Ray Worker Actors for Distributed Processing

Provides Ray-based worker actors that replace ProcessPoolExecutor
with proper cancellation, progress reporting, and fault tolerance.
"""

from .base_worker import BaseWorkerActor, WorkerState
from .cpu_worker import (
    create_cpu_worker_actor,
    WorkerResult,
    GatherResult,
    MuteConfig,
    SortConfig,
    apply_mute_to_gather_inplace,
    compute_gather_sort_indices,
    StreamingSortWriter,
    read_streaming_sort_file,
)
from .processor_wrapper import ProcessorWrapper, wrap_processor
from .metal_worker import (
    MetalWorkerActor,
    GPUDeviceInfo,
    GPUProcessingResult,
    create_metal_worker_actor,
)

__all__ = [
    'BaseWorkerActor',
    'WorkerState',
    'create_cpu_worker_actor',
    'WorkerResult',
    'GatherResult',
    'MuteConfig',
    'SortConfig',
    'apply_mute_to_gather_inplace',
    'compute_gather_sort_indices',
    'StreamingSortWriter',
    'read_streaming_sort_file',
    'ProcessorWrapper',
    'wrap_processor',
    'MetalWorkerActor',
    'GPUDeviceInfo',
    'GPUProcessingResult',
    'create_metal_worker_actor',
]
