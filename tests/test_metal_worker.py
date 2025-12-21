"""
Tests for Metal Worker Actor

Tests the MetalWorkerActor for GPU-accelerated processing.
"""

import pytest
import numpy as np
from uuid import uuid4
from unittest.mock import Mock, patch, MagicMock


class TestGPUDeviceInfo:
    """Tests for GPUDeviceInfo dataclass."""

    def test_device_info_creation(self):
        """Test creating GPU device info."""
        from utils.ray_orchestration.workers.metal_worker import GPUDeviceInfo

        info = GPUDeviceInfo(
            available=True,
            device_name="Apple M4 Max",
            memory_total_mb=65536,
            memory_available_mb=50000,
            supports_float16=True,
            max_threads_per_group=1024,
        )

        assert info.available is True
        assert info.device_name == "Apple M4 Max"
        assert info.memory_total_mb == 65536

    def test_device_info_unavailable(self):
        """Test device info when GPU unavailable."""
        from utils.ray_orchestration.workers.metal_worker import GPUDeviceInfo

        info = GPUDeviceInfo(available=False)

        assert info.available is False
        assert info.device_name == "Unknown"


class TestGPUProcessingResult:
    """Tests for GPUProcessingResult dataclass."""

    def test_result_success(self):
        """Test successful processing result."""
        from utils.ray_orchestration.workers.metal_worker import GPUProcessingResult

        output = np.random.randn(100, 500).astype(np.float32)

        result = GPUProcessingResult(
            success=True,
            output_data=output,
            elapsed_ms=15.5,
            gpu_memory_used_mb=128.0,
        )

        assert result.success is True
        assert result.output_data is not None
        assert result.elapsed_ms == 15.5
        assert result.error is None

    def test_result_failure(self):
        """Test failed processing result."""
        from utils.ray_orchestration.workers.metal_worker import GPUProcessingResult

        result = GPUProcessingResult(
            success=False,
            error="Out of GPU memory",
        )

        assert result.success is False
        assert result.output_data is None
        assert "memory" in result.error.lower()


class TestMetalWorkerActor:
    """Tests for MetalWorkerActor."""

    def test_worker_creation(self):
        """Test creating a Metal worker actor."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        job_id = uuid4()
        worker = MetalWorkerActor(job_id, "gpu-worker-0")

        assert worker.job_id == job_id
        assert worker.worker_id == "gpu-worker-0"
        assert worker._gpu_initialized is False

    def test_worker_initial_state(self):
        """Test worker is in IDLE state initially."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor
        from utils.ray_orchestration.workers.base_worker import WorkerState

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")

        assert worker.state == WorkerState.IDLE

    @patch('processors.kernel_backend.is_metal_available')
    def test_initialize_gpu_not_available(self, mock_metal_available):
        """Test GPU initialization when Metal not available."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_metal_available.return_value = False

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        info = worker.initialize_gpu()

        assert info.available is False
        assert worker.is_gpu_available() is False

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_initialize_gpu_available(self, mock_available, mock_get_module):
        """Test GPU initialization when Metal is available."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = True

        mock_module = Mock()
        mock_module.get_device_info.return_value = {
            'device_name': 'Apple M4 Max',
            'memory_total_mb': 65536,
            'memory_available_mb': 50000,
            'supports_float16': True,
            'max_threads_per_group': 1024,
        }
        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        info = worker.initialize_gpu()

        assert info.available is True
        assert info.device_name == 'Apple M4 Max'
        assert worker.is_gpu_available() is True

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_get_device_info_initializes_if_needed(self, mock_available, mock_get_module):
        """Test get_device_info auto-initializes."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = False

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")

        # Should call initialize_gpu
        info = worker.get_device_info()

        assert info is not None
        assert info.available is False

    @patch('processors.kernel_backend.is_metal_available')
    def test_process_gather_gpu_not_available(self, mock_available):
        """Test processing when GPU not available returns error."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = False

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")

        data = np.random.randn(100, 500).astype(np.float32)
        result = worker.process_gather_gpu(
            data,
            processor_type='dwt_denoise',
            processor_config={'wavelet': 'db4'},
        )

        assert result.success is False
        assert "not available" in result.error.lower()

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_process_gather_gpu_success(self, mock_available, mock_get_module):
        """Test successful GPU processing."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = True

        # Mock the metal module
        mock_module = Mock()
        mock_module.get_device_info.return_value = {
            'device_name': 'Apple GPU',
            'memory_total_mb': 8192,
        }
        mock_module.get_memory_usage.return_value = {
            'used_mb': 100.0,
            'total_mb': 8192.0,
            'percent': 1.2,
        }

        # Mock dwt_denoise to return processed data
        input_data = np.random.randn(100, 500).astype(np.float32)
        output_data = input_data * 0.9  # Simulated processing
        mock_module.dwt_denoise.return_value = output_data

        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        worker.initialize_gpu()

        result = worker.process_gather_gpu(
            input_data,
            processor_type='dwt_denoise',
            processor_config={'wavelet': 'db4', 'level': 4},
        )

        assert result.success is True
        assert result.output_data is not None
        assert result.elapsed_ms > 0
        mock_module.dwt_denoise.assert_called_once()

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_process_unknown_processor_type(self, mock_available, mock_get_module):
        """Test processing with unknown processor type uses fallback."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = True

        mock_module = Mock()
        mock_module.get_device_info.return_value = {'device_name': 'GPU'}
        mock_module.get_memory_usage.return_value = {'used_mb': 0, 'total_mb': 8192, 'percent': 0}
        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        worker.initialize_gpu()

        data = np.random.randn(100, 500).astype(np.float32)
        result = worker.process_gather_gpu(
            data,
            processor_type='unknown_processor',
            processor_config={},
        )

        # Should succeed with passthrough
        assert result.success is True
        assert result.output_data is not None

    @patch('processors.kernel_backend.is_metal_available')
    def test_get_gpu_memory_usage_unavailable(self, mock_available):
        """Test memory usage when GPU unavailable."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = False

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        usage = worker.get_gpu_memory_usage()

        assert usage['used_mb'] == 0
        assert usage['total_mb'] == 0
        assert usage['percent'] == 0

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_cleanup(self, mock_available, mock_get_module):
        """Test GPU resource cleanup."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = True

        mock_module = Mock()
        mock_module.get_device_info.return_value = {'device_name': 'GPU'}
        mock_module.cleanup = Mock()
        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        worker.initialize_gpu()

        assert worker.is_gpu_available() is True

        worker.cleanup()

        assert worker._gpu_initialized is False
        mock_module.cleanup.assert_called_once()

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_worker_cancellation_during_processing(self, mock_available, mock_get_module):
        """Test worker handles cancellation during processing."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor
        from utils.ray_orchestration.cancellation import CancellationError

        mock_available.return_value = True

        mock_module = Mock()
        mock_module.get_device_info.return_value = {'device_name': 'GPU'}
        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        worker.initialize_gpu()

        # Cancel the worker
        worker.cancel()

        data = np.random.randn(100, 500).astype(np.float32)

        with pytest.raises(CancellationError):
            worker.process_gather_gpu(
                data,
                processor_type='dwt_denoise',
                processor_config={},
            )


class TestMetalWorkerActorProcessors:
    """Tests for specific processor GPU implementations."""

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_stft_filter_processor(self, mock_available, mock_get_module):
        """Test STFT filter GPU processing."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = True

        mock_module = Mock()
        mock_module.get_device_info.return_value = {'device_name': 'GPU'}
        mock_module.get_memory_usage.return_value = {'used_mb': 0, 'total_mb': 8192, 'percent': 0}
        mock_module.stft_filter.return_value = np.zeros((100, 500), dtype=np.float32)
        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        worker.initialize_gpu()

        data = np.random.randn(100, 500).astype(np.float32)
        result = worker.process_gather_gpu(
            data,
            processor_type='stft_filter',
            processor_config={'nperseg': 256, 'freq_low': 5.0, 'freq_high': 80.0},
        )

        assert result.success is True
        mock_module.stft_filter.assert_called_once()

    @patch('processors.kernel_backend.get_metal_module')
    @patch('processors.kernel_backend.is_metal_available')
    def test_agc_processor(self, mock_available, mock_get_module):
        """Test AGC GPU processing."""
        from utils.ray_orchestration.workers.metal_worker import MetalWorkerActor

        mock_available.return_value = True

        mock_module = Mock()
        mock_module.get_device_info.return_value = {'device_name': 'GPU'}
        mock_module.get_memory_usage.return_value = {'used_mb': 0, 'total_mb': 8192, 'percent': 0}
        mock_module.agc.return_value = np.ones((100, 500), dtype=np.float32)
        mock_get_module.return_value = mock_module

        worker = MetalWorkerActor(uuid4(), "gpu-worker-0")
        worker.initialize_gpu()

        data = np.random.randn(100, 500).astype(np.float32)
        result = worker.process_gather_gpu(
            data,
            processor_type='agc',
            processor_config={'window_ms': 500, 'target_rms': 1.0},
        )

        assert result.success is True
        mock_module.agc.assert_called_once()


class TestCreateMetalWorkerActor:
    """Tests for create_metal_worker_actor factory."""

    def test_create_without_ray(self):
        """Test creating Ray actor fails without Ray."""
        from utils.ray_orchestration.workers.metal_worker import create_metal_worker_actor
        import utils.ray_orchestration.workers.metal_worker as mw

        # Temporarily set RAY_AVAILABLE to False
        original = mw.RAY_AVAILABLE
        mw.RAY_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="Ray is not available"):
                create_metal_worker_actor()
        finally:
            mw.RAY_AVAILABLE = original
