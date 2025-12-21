"""
Ray Orchestration Module

Provides distributed task execution with Ray for:
- Parallel batch processing
- SEGY I/O operations
- Multi-level cancellation support
- Fault tolerance and recovery
- Resource monitoring
"""

from .config import RayConfig, get_default_config, get_minimal_config
from .cluster import (
    RayClusterManager,
    ClusterResources,
    initialize_ray,
    shutdown_ray,
    is_ray_initialized,
    get_cluster_resources,
)
from .cancellation import (
    CancellationToken,
    CancellationCoordinator,
    CancellationReason,
    CancellationRequest,
    CancellationError,
    get_cancellation_coordinator,
    create_cancellation_token,
    cancel_job,
    cancel_all_jobs,
)
from .job_manager import (
    JobManager,
    get_job_manager,
)
from .checkpoint import (
    Checkpoint,
    CheckpointManager,
    get_checkpoint_manager,
    save_checkpoint,
    load_latest_checkpoint,
)
from .resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    ResourceAlert,
    ResourceThresholds,
    get_resource_monitor,
    start_monitoring,
    stop_monitoring,
    get_current_resources,
)
from .processing_job_adapter import (
    ProcessingJobAdapter,
    RayProcessingJobAdapter,
    create_processing_job,
)
from .ray_processing_coordinator import RayProcessingCoordinator
from .segy_job_adapter import (
    SEGYImportJobAdapter,
    SEGYExportJobAdapter,
    create_import_job,
)
from .segy_workers import (
    SEGYImportWorker,
    SEGYExportWorker,
    SEGYImportResult,
    SEGYExportResult,
    ParallelExportWorker,
    ParallelExportResult,
)
from .processing_api import (
    ProcessingResult,
    ExportResult,
    run_parallel_processing,
    run_segy_export,
    cancel_processing_job,
    pause_processing_job,
    resume_processing_job,
    get_job_status,
    get_optimal_workers,
)

__all__ = [
    # Configuration
    'RayConfig',
    'get_default_config',
    'get_minimal_config',
    # Cluster management
    'RayClusterManager',
    'ClusterResources',
    'initialize_ray',
    'shutdown_ray',
    'is_ray_initialized',
    'get_cluster_resources',
    # Cancellation
    'CancellationToken',
    'CancellationCoordinator',
    'CancellationReason',
    'CancellationRequest',
    'CancellationError',
    'get_cancellation_coordinator',
    'create_cancellation_token',
    'cancel_job',
    'cancel_all_jobs',
    # Job management
    'JobManager',
    'get_job_manager',
    # Checkpointing
    'Checkpoint',
    'CheckpointManager',
    'get_checkpoint_manager',
    'save_checkpoint',
    'load_latest_checkpoint',
    # Resource monitoring
    'ResourceMonitor',
    'ResourceSnapshot',
    'ResourceAlert',
    'ResourceThresholds',
    'get_resource_monitor',
    'start_monitoring',
    'stop_monitoring',
    'get_current_resources',
    # Processing adapters
    'ProcessingJobAdapter',
    'RayProcessingJobAdapter',
    'RayProcessingCoordinator',
    'create_processing_job',
    # SEGY adapters
    'SEGYImportJobAdapter',
    'SEGYExportJobAdapter',
    'create_import_job',
    # SEGY workers
    'SEGYImportWorker',
    'SEGYExportWorker',
    'SEGYImportResult',
    'SEGYExportResult',
    'ParallelExportWorker',
    'ParallelExportResult',
    # High-level processing API
    'ProcessingResult',
    'ExportResult',
    'run_parallel_processing',
    'run_segy_export',
    'cancel_processing_job',
    'pause_processing_job',
    'resume_processing_job',
    'get_job_status',
    'get_optimal_workers',
]
