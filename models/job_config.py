"""
Job Configuration Models

Defines configuration structures for jobs and resource requirements.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = auto()
    GPU = auto()
    MEMORY = auto()
    DISK = auto()


@dataclass
class ResourceRequirements:
    """
    Resource requirements for a job.

    Attributes
    ----------
    num_cpus : float
        Number of CPU cores required
    num_gpus : float
        Number of GPUs required
    memory_mb : int
        Memory required in MB
    disk_mb : int
        Disk space required in MB
    custom_resources : dict
        Custom Ray resources
    """

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory_mb: int = 512
    disk_mb: int = 0
    custom_resources: Dict[str, float] = field(default_factory=dict)

    def to_ray_resources(self) -> Dict[str, float]:
        """Convert to Ray resource dict."""
        resources = {
            "num_cpus": self.num_cpus,
        }
        if self.num_gpus > 0:
            resources["num_gpus"] = self.num_gpus
        if self.memory_mb > 0:
            resources["memory"] = self.memory_mb * 1024 * 1024  # Convert to bytes
        resources.update(self.custom_resources)
        return resources

    @classmethod
    def for_cpu_intensive(cls, cores: int = 4) -> 'ResourceRequirements':
        """Create requirements for CPU-intensive work."""
        return cls(num_cpus=cores, memory_mb=1024 * cores)

    @classmethod
    def for_gpu_processing(cls, gpus: float = 1.0) -> 'ResourceRequirements':
        """Create requirements for GPU processing."""
        return cls(num_cpus=1.0, num_gpus=gpus, memory_mb=4096)

    @classmethod
    def for_io_bound(cls) -> 'ResourceRequirements':
        """Create requirements for I/O-bound work."""
        return cls(num_cpus=0.5, memory_mb=256)


@dataclass
class CheckpointConfig:
    """
    Configuration for job checkpointing.

    Attributes
    ----------
    enabled : bool
        Whether checkpointing is enabled
    interval_seconds : int
        Interval between checkpoints
    checkpoint_dir : str
        Directory for checkpoint files
    keep_count : int
        Number of checkpoints to keep
    """

    enabled: bool = True
    interval_seconds: int = 60
    checkpoint_dir: str = ".checkpoints"
    keep_count: int = 3
    compress: bool = True


@dataclass
class RetryConfig:
    """
    Configuration for job retry behavior.

    Attributes
    ----------
    max_retries : int
        Maximum number of retry attempts
    retry_delay_seconds : float
        Initial delay between retries
    exponential_backoff : bool
        Use exponential backoff
    max_delay_seconds : float
        Maximum delay between retries
    retry_on_exceptions : list
        Exception types to retry on
    """

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    max_delay_seconds: float = 60.0
    retry_on_exceptions: List[str] = field(
        default_factory=lambda: ["TimeoutError", "ConnectionError", "OSError"]
    )


@dataclass
class CancellationConfig:
    """
    Configuration for job cancellation behavior.

    Attributes
    ----------
    graceful_timeout_seconds : float
        Time to wait for graceful shutdown
    force_kill_after : bool
        Force kill if graceful timeout exceeded
    save_partial_results : bool
        Save partial results on cancellation
    cleanup_temp_files : bool
        Clean up temporary files
    """

    graceful_timeout_seconds: float = 10.0
    force_kill_after: bool = True
    save_partial_results: bool = True
    cleanup_temp_files: bool = True


@dataclass
class JobConfig:
    """
    Complete configuration for a job.

    Attributes
    ----------
    resources : ResourceRequirements
        Resource requirements
    checkpoint : CheckpointConfig
        Checkpointing configuration
    retry : RetryConfig
        Retry configuration
    cancellation : CancellationConfig
        Cancellation configuration
    timeout_seconds : float, optional
        Overall job timeout
    batch_size : int
        Items per batch for parallel work
    num_workers : int, optional
        Number of parallel workers (None = auto)
    custom_config : dict
        Job-type specific configuration
    """

    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    cancellation: CancellationConfig = field(default_factory=CancellationConfig)
    timeout_seconds: Optional[float] = None
    batch_size: int = 100
    num_workers: Optional[int] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "resources": {
                "num_cpus": self.resources.num_cpus,
                "num_gpus": self.resources.num_gpus,
                "memory_mb": self.resources.memory_mb,
                "disk_mb": self.resources.disk_mb,
            },
            "checkpoint": {
                "enabled": self.checkpoint.enabled,
                "interval_seconds": self.checkpoint.interval_seconds,
                "checkpoint_dir": self.checkpoint.checkpoint_dir,
                "keep_count": self.checkpoint.keep_count,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "retry_delay_seconds": self.retry.retry_delay_seconds,
                "exponential_backoff": self.retry.exponential_backoff,
            },
            "cancellation": {
                "graceful_timeout_seconds": self.cancellation.graceful_timeout_seconds,
                "save_partial_results": self.cancellation.save_partial_results,
            },
            "timeout_seconds": self.timeout_seconds,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "custom_config": self.custom_config,
        }

    @classmethod
    def for_segy_import(cls, file_size_mb: int) -> 'JobConfig':
        """Create configuration optimized for SEGY import."""
        # Scale workers based on file size
        workers = min(8, max(2, file_size_mb // 500))
        return cls(
            resources=ResourceRequirements.for_io_bound(),
            checkpoint=CheckpointConfig(enabled=True, interval_seconds=30),
            batch_size=1000,
            num_workers=workers,
        )

    @classmethod
    def for_batch_processing(cls, trace_count: int) -> 'JobConfig':
        """Create configuration optimized for batch processing."""
        workers = min(16, max(2, trace_count // 10000))
        return cls(
            resources=ResourceRequirements.for_cpu_intensive(2),
            checkpoint=CheckpointConfig(enabled=True, interval_seconds=60),
            batch_size=500,
            num_workers=workers,
        )

    @classmethod
    def for_gpu_processing(cls) -> 'JobConfig':
        """Create configuration for GPU-accelerated processing."""
        return cls(
            resources=ResourceRequirements.for_gpu_processing(),
            checkpoint=CheckpointConfig(enabled=False),  # GPU checkpoints complex
            batch_size=5000,  # Large batches for GPU
            num_workers=1,  # Single GPU worker
        )
