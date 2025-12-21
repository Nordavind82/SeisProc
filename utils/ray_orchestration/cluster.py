"""
Ray Cluster Management

Handles Ray cluster initialization, shutdown, and resource monitoring.
Thread-safe singleton pattern for application-wide cluster management.
"""

import logging
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .config import RayConfig, get_default_config

logger = logging.getLogger(__name__)

# Lazy import ray to avoid import errors if not installed
_ray = None


def _get_ray():
    """Lazy import Ray module."""
    global _ray
    if _ray is None:
        try:
            import ray
            _ray = ray
        except ImportError:
            raise ImportError(
                "Ray is not installed. Install with: pip install 'ray[default]>=2.9'"
            )
    return _ray


@dataclass
class ClusterResources:
    """Information about available cluster resources."""
    num_cpus: float
    num_gpus: float
    memory_bytes: int
    object_store_bytes: int
    num_nodes: int
    node_ids: list


class RayClusterManager:
    """
    Thread-safe singleton manager for Ray cluster lifecycle.

    Usage
    -----
    >>> manager = RayClusterManager()
    >>> manager.initialize()  # Start local cluster
    >>> # ... do work ...
    >>> manager.shutdown()
    """

    _instance: Optional['RayClusterManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'RayClusterManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
                cls._instance._config = None
            return cls._instance

    def initialize(self, config: Optional[RayConfig] = None) -> bool:
        """
        Initialize Ray cluster with given configuration.

        Parameters
        ----------
        config : RayConfig, optional
            Cluster configuration. Uses defaults if not provided.

        Returns
        -------
        bool
            True if initialization successful, False otherwise.
        """
        ray = _get_ray()

        with self._lock:
            if self._initialized and ray.is_initialized():
                logger.info("Ray cluster already initialized")
                return True

            try:
                self._config = config or get_default_config()
                init_kwargs = self._config.to_init_kwargs()

                logger.info(f"Initializing Ray cluster with config: {init_kwargs}")

                # Initialize Ray
                ray.init(**init_kwargs)

                self._initialized = True

                # Log cluster info
                resources = self.get_resources()
                logger.info(
                    f"Ray cluster started: "
                    f"{resources.num_cpus} CPUs, "
                    f"{resources.num_gpus} GPUs, "
                    f"{resources.memory_bytes / (1024**3):.1f} GB memory"
                )

                if self._config.include_dashboard:
                    logger.info(
                        f"Ray dashboard: http://{self._config.dashboard_host}:"
                        f"{self._config.dashboard_port}"
                    )

                return True

            except Exception as e:
                logger.error(f"Failed to initialize Ray cluster: {e}")
                self._initialized = False
                return False

    def shutdown(self) -> None:
        """Shutdown Ray cluster and release resources."""
        ray = _get_ray()

        with self._lock:
            if not self._initialized:
                return

            try:
                logger.info("Shutting down Ray cluster...")
                ray.shutdown()
                self._initialized = False
                self._config = None
                logger.info("Ray cluster shutdown complete")
            except Exception as e:
                logger.error(f"Error during Ray shutdown: {e}")

    def is_initialized(self) -> bool:
        """Check if Ray cluster is currently initialized."""
        ray = _get_ray()
        return self._initialized and ray.is_initialized()

    def get_resources(self) -> ClusterResources:
        """
        Get current cluster resource information.

        Returns
        -------
        ClusterResources
            Current cluster resources.

        Raises
        ------
        RuntimeError
            If cluster is not initialized.
        """
        ray = _get_ray()

        if not self.is_initialized():
            raise RuntimeError("Ray cluster not initialized")

        resources = ray.cluster_resources()
        nodes = ray.nodes()

        return ClusterResources(
            num_cpus=resources.get("CPU", 0),
            num_gpus=resources.get("GPU", 0),
            memory_bytes=int(resources.get("memory", 0)),
            object_store_bytes=int(resources.get("object_store_memory", 0)),
            num_nodes=len(nodes),
            node_ids=[n["NodeID"] for n in nodes],
        )

    def get_available_resources(self) -> Dict[str, float]:
        """
        Get currently available (not in use) resources.

        Returns
        -------
        dict
            Dictionary of available resources.
        """
        ray = _get_ray()

        if not self.is_initialized():
            raise RuntimeError("Ray cluster not initialized")

        return ray.available_resources()

    @property
    def config(self) -> Optional[RayConfig]:
        """Get current cluster configuration."""
        return self._config


# Module-level convenience functions

def initialize_ray(config: Optional[RayConfig] = None) -> bool:
    """
    Initialize Ray cluster (module-level convenience function).

    Parameters
    ----------
    config : RayConfig, optional
        Cluster configuration.

    Returns
    -------
    bool
        True if successful.
    """
    manager = RayClusterManager()
    return manager.initialize(config)


def shutdown_ray() -> None:
    """Shutdown Ray cluster (module-level convenience function)."""
    manager = RayClusterManager()
    manager.shutdown()


def is_ray_initialized() -> bool:
    """Check if Ray is initialized (module-level convenience function)."""
    manager = RayClusterManager()
    return manager.is_initialized()


def get_cluster_resources() -> ClusterResources:
    """Get cluster resources (module-level convenience function)."""
    manager = RayClusterManager()
    return manager.get_resources()
