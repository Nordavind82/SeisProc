"""
Ray Configuration Module

Defines configuration settings for Ray cluster initialization and resource management.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import psutil


@dataclass
class RayConfig:
    """Configuration for Ray cluster initialization."""

    # Cluster settings
    num_cpus: Optional[int] = None  # None = auto-detect
    num_gpus: Optional[int] = None  # None = auto-detect
    memory: Optional[int] = None  # Bytes, None = auto
    object_store_memory: Optional[int] = None  # Bytes for plasma store

    # Dashboard settings
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8265
    include_dashboard: bool = True

    # Logging settings
    log_to_driver: bool = True
    logging_level: str = "INFO"

    # Resource settings
    resources: Dict[str, float] = field(default_factory=dict)

    # Runtime environment
    runtime_env: Optional[Dict[str, Any]] = None

    # Connection settings (for connecting to existing cluster)
    address: Optional[str] = None  # e.g., "ray://localhost:10001"

    # Namespace for job isolation
    namespace: str = "seisproc"

    # Failure handling
    max_task_retries: int = 3
    task_retry_delay_ms: int = 1000

    def to_init_kwargs(self) -> Dict[str, Any]:
        """Convert config to ray.init() keyword arguments."""
        kwargs = {
            "namespace": self.namespace,
            "log_to_driver": self.log_to_driver,
            "logging_level": self.logging_level,
            "include_dashboard": self.include_dashboard,
        }

        # Only set if explicitly configured
        if self.num_cpus is not None:
            kwargs["num_cpus"] = self.num_cpus
        if self.num_gpus is not None:
            kwargs["num_gpus"] = self.num_gpus
        if self.memory is not None:
            kwargs["_memory"] = self.memory
        if self.object_store_memory is not None:
            kwargs["object_store_memory"] = self.object_store_memory
        if self.address is not None:
            kwargs["address"] = self.address
        if self.resources:
            kwargs["resources"] = self.resources
        if self.runtime_env is not None:
            kwargs["runtime_env"] = self.runtime_env

        # Dashboard settings
        if self.include_dashboard:
            kwargs["dashboard_host"] = self.dashboard_host
            kwargs["dashboard_port"] = self.dashboard_port

        return kwargs


def get_default_config() -> RayConfig:
    """
    Get default Ray configuration based on system resources.

    Returns
    -------
    RayConfig
        Configuration optimized for the current system.
    """
    import platform

    # Get system info
    cpu_count = os.cpu_count() or 4
    memory_bytes = psutil.virtual_memory().total

    # Reserve some resources for the OS and UI
    usable_cpus = max(1, cpu_count - 2)
    usable_memory = int(memory_bytes * 0.7)  # Use 70% of RAM
    object_store = int(memory_bytes * 0.3)  # 30% for object store

    # CRITICAL: Cap object store on Mac due to Ray performance issue
    # Ray has known degradation with object_store > 2GB on macOS
    if platform.system() == "Darwin":
        max_object_store_mac = 2 * 1024 * 1024 * 1024  # 2GB
        if object_store > max_object_store_mac:
            object_store = max_object_store_mac

    return RayConfig(
        num_cpus=usable_cpus,
        memory=usable_memory,
        object_store_memory=object_store,
        include_dashboard=True,
        log_to_driver=True,
        logging_level="INFO",
    )


def get_minimal_config() -> RayConfig:
    """
    Get minimal Ray configuration for testing or low-resource systems.

    Returns
    -------
    RayConfig
        Minimal configuration with reduced resource usage.
    """
    return RayConfig(
        num_cpus=2,
        memory=1024 * 1024 * 1024,  # 1GB
        object_store_memory=512 * 1024 * 1024,  # 512MB
        include_dashboard=False,
        log_to_driver=False,
        logging_level="WARNING",
    )
