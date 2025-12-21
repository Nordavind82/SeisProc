"""
Tests for Ray Cluster Management

Tests Ray initialization, shutdown, and resource management.
"""

import pytest
import sys


# Skip all tests if ray is not installed
ray = pytest.importorskip("ray")


class TestRayConfig:
    """Tests for RayConfig."""

    def test_default_config_creation(self):
        """Test creating default config based on system resources."""
        from utils.ray_orchestration import get_default_config

        config = get_default_config()

        assert config.num_cpus is not None
        assert config.num_cpus >= 1
        assert config.memory is not None
        assert config.object_store_memory is not None
        assert config.namespace == "seisproc"

    def test_minimal_config_creation(self):
        """Test creating minimal config for testing."""
        from utils.ray_orchestration import get_minimal_config

        config = get_minimal_config()

        assert config.num_cpus == 2
        assert config.include_dashboard is False
        assert config.logging_level == "WARNING"

    def test_config_to_init_kwargs(self):
        """Test converting config to ray.init() kwargs."""
        from utils.ray_orchestration import RayConfig

        config = RayConfig(
            num_cpus=4,
            num_gpus=1,
            namespace="test",
        )

        kwargs = config.to_init_kwargs()

        assert kwargs["num_cpus"] == 4
        assert kwargs["num_gpus"] == 1
        assert kwargs["namespace"] == "test"


class TestRayClusterManager:
    """Tests for RayClusterManager."""

    def setup_method(self):
        """Ensure Ray is shutdown before each test."""
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    def teardown_method(self):
        """Clean up Ray after each test."""
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
            # Reset singleton state
            from utils.ray_orchestration.cluster import RayClusterManager
            RayClusterManager._instance = None
        except Exception:
            pass

    def test_cluster_singleton(self):
        """Test that RayClusterManager is a singleton."""
        from utils.ray_orchestration import RayClusterManager

        manager1 = RayClusterManager()
        manager2 = RayClusterManager()

        assert manager1 is manager2

    def test_ray_cluster_initialization(self):
        """Ray cluster initializes successfully."""
        from utils.ray_orchestration import (
            initialize_ray,
            is_ray_initialized,
            shutdown_ray,
            get_minimal_config,
        )

        config = get_minimal_config()
        result = initialize_ray(config)

        assert result is True
        assert is_ray_initialized() is True

        shutdown_ray()

        assert is_ray_initialized() is False

    def test_ray_cluster_with_minimal_config(self):
        """Ray cluster initializes with minimal config."""
        from utils.ray_orchestration import (
            initialize_ray,
            is_ray_initialized,
            get_minimal_config,
        )

        config = get_minimal_config()
        result = initialize_ray(config)

        assert result is True
        assert is_ray_initialized() is True

    def test_get_cluster_resources(self):
        """Test getting cluster resources after initialization."""
        from utils.ray_orchestration import (
            initialize_ray,
            get_cluster_resources,
            get_minimal_config,
        )

        config = get_minimal_config()
        initialize_ray(config)

        resources = get_cluster_resources()

        assert resources.num_cpus >= 1
        assert resources.num_nodes >= 1
        assert resources.memory_bytes > 0

    def test_double_initialization_safe(self):
        """Test that double initialization is safe."""
        from utils.ray_orchestration import (
            initialize_ray,
            is_ray_initialized,
            get_minimal_config,
        )

        config = get_minimal_config()
        result1 = initialize_ray(config)
        result2 = initialize_ray(config)

        assert result1 is True
        assert result2 is True  # Returns True even if already initialized
        assert is_ray_initialized() is True

    def test_shutdown_when_not_initialized(self):
        """Test that shutdown is safe when not initialized."""
        from utils.ray_orchestration import shutdown_ray

        # Should not raise
        shutdown_ray()
        shutdown_ray()
