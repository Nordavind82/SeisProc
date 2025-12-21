"""
Metal Worker Actor for GPU Processing

Provides Ray actor-based GPU processing using Metal shaders.
Integrates with the existing kernel_backend system for GPU acceleration.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID

import numpy as np

from .base_worker import BaseWorkerActor, WorkerState, WorkerProgress

logger = logging.getLogger(__name__)

# Check for Ray availability
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class GPUDeviceInfo:
    """Information about the GPU device."""
    available: bool
    device_name: str = "Unknown"
    memory_total_mb: float = 0.0
    memory_available_mb: float = 0.0
    supports_float16: bool = False
    max_threads_per_group: int = 1024


@dataclass
class GPUProcessingResult:
    """Result from GPU processing."""
    success: bool
    output_data: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0
    gpu_memory_used_mb: float = 0.0
    error: Optional[str] = None


class MetalWorkerActor(BaseWorkerActor):
    """
    Ray actor for Metal GPU processing.

    Provides GPU-accelerated processing for seismic data using
    Metal shaders through the kernel_backend system.

    Features:
    - GPU device initialization per worker (fresh Metal context)
    - Memory management and monitoring
    - Integration with existing Metal kernels
    - Cancellation checking during GPU operations
    - Mute and sort support (delegated to CPU)

    Usage
    -----
    >>> worker = MetalWorkerActor(job_id, "gpu-worker-0", ...)
    >>> worker.initialize_gpu()
    >>> result = worker.process(gather_tasks, headers_subset)
    """

    def __init__(
        self,
        job_id: UUID,
        worker_id: str,
        input_zarr_path: str,
        output_zarr_path: Optional[str],
        noise_zarr_path: Optional[str],
        processor_config: Dict[str, Any],
        sample_rate: float,
        metadata: Dict[str, Any],
        output_mode: str = 'processed',
        mute_config: Optional[Dict[str, Any]] = None,
        sort_config: Optional[Dict[str, Any]] = None,
        progress_callback=None,
    ):
        """
        Initialize Metal worker actor.

        Parameters
        ----------
        job_id : UUID
            Parent job identifier
        worker_id : str
            Unique worker identifier
        input_zarr_path : str
            Path to input Zarr array
        output_zarr_path : str, optional
            Path to output Zarr array (for processed output)
        noise_zarr_path : str, optional
            Path to noise Zarr array (for noise output)
        processor_config : dict
            Processor configuration for reconstruction
        sample_rate : float
            Sample rate in milliseconds
        metadata : dict
            Seismic metadata
        output_mode : str
            'processed', 'noise', or 'both'
        mute_config : dict, optional
            Mute configuration
        sort_config : dict, optional
            Sort configuration
        progress_callback : callable, optional
            Progress callback function
        """
        super().__init__(job_id, worker_id, progress_callback)

        # Store paths and config (same as CPUWorkerActor)
        self._input_zarr_path = input_zarr_path
        self._output_zarr_path = output_zarr_path
        self._noise_zarr_path = noise_zarr_path
        self._processor_config = processor_config
        self._sample_rate = sample_rate
        self._metadata = metadata
        self._output_mode = output_mode
        self._mute_config = mute_config
        self._sort_config = sort_config

        # GPU-specific state
        self._gpu_initialized = False
        self._device_info: Optional[GPUDeviceInfo] = None
        self._metal_module = None
        self._kernel_backend = None

        # Lazily initialized
        self._input_zarr = None
        self._output_zarr = None
        self._noise_zarr = None
        self._processor = None

        logger.info(f"MetalWorkerActor {worker_id} created for job {job_id}")

    def initialize_gpu(self) -> GPUDeviceInfo:
        """
        Initialize the GPU device for this worker.

        Returns
        -------
        GPUDeviceInfo
            Information about the initialized GPU device
        """
        from processors.kernel_backend import (
            is_metal_available,
            get_metal_module,
            get_backend_info,
        )

        self._state = WorkerState.INITIALIZING

        if not is_metal_available():
            self._device_info = GPUDeviceInfo(
                available=False,
                device_name="No GPU available",
            )
            logger.warning(f"Worker {self.worker_id}: Metal GPU not available")
            return self._device_info

        try:
            self._metal_module = get_metal_module()

            if self._metal_module is None:
                self._device_info = GPUDeviceInfo(
                    available=False,
                    device_name="Metal module not loaded",
                )
                return self._device_info

            # Get device info from Metal module
            device_info = self._metal_module.get_device_info()

            self._device_info = GPUDeviceInfo(
                available=True,
                device_name=device_info.get('device_name', 'Apple GPU'),
                memory_total_mb=device_info.get('memory_total_mb', 0),
                memory_available_mb=device_info.get('memory_available_mb', 0),
                supports_float16=device_info.get('supports_float16', False),
                max_threads_per_group=device_info.get('max_threads_per_group', 1024),
            )

            self._gpu_initialized = True
            self._state = WorkerState.IDLE

            logger.info(
                f"Worker {self.worker_id}: GPU initialized - "
                f"{self._device_info.device_name}"
            )

            return self._device_info

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: GPU initialization failed: {e}")
            self._device_info = GPUDeviceInfo(
                available=False,
                device_name=f"Initialization failed: {e}",
            )
            self._state = WorkerState.FAILED
            return self._device_info

    def get_device_info(self) -> GPUDeviceInfo:
        """
        Get GPU device information.

        Returns
        -------
        GPUDeviceInfo
            Current GPU device information
        """
        if self._device_info is None:
            return self.initialize_gpu()
        return self._device_info

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and initialized."""
        return self._gpu_initialized and self._device_info and self._device_info.available

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns
        -------
        dict
            Memory usage information with 'used_mb', 'total_mb', 'percent'
        """
        if not self.is_gpu_available() or self._metal_module is None:
            return {'used_mb': 0, 'total_mb': 0, 'percent': 0}

        try:
            if hasattr(self._metal_module, 'get_memory_usage'):
                return self._metal_module.get_memory_usage()

            # Fallback to device info
            return {
                'used_mb': 0,
                'total_mb': self._device_info.memory_total_mb,
                'percent': 0,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory usage: {e}")
            return {'used_mb': 0, 'total_mb': 0, 'percent': 0}

    def process_gather_gpu(
        self,
        gather_data: np.ndarray,
        processor_type: str,
        processor_config: Dict[str, Any],
    ) -> GPUProcessingResult:
        """
        Process a seismic gather using GPU acceleration.

        Parameters
        ----------
        gather_data : np.ndarray
            Input gather data (n_traces, n_samples)
        processor_type : str
            Type of processor to use (e.g., 'dwt_denoise', 'stft_filter')
        processor_config : dict
            Processor configuration parameters

        Returns
        -------
        GPUProcessingResult
            Processing result with output data and metrics
        """
        # Check cancellation before starting
        self._check_cancellation()

        if not self.is_gpu_available():
            return GPUProcessingResult(
                success=False,
                error="GPU not available or not initialized",
            )

        self._state = WorkerState.PROCESSING
        start_time = time.perf_counter()

        try:
            # Ensure data is contiguous float32
            if not gather_data.flags['C_CONTIGUOUS']:
                gather_data = np.ascontiguousarray(gather_data)

            if gather_data.dtype != np.float32:
                gather_data = gather_data.astype(np.float32)

            # Get initial memory
            mem_before = self.get_gpu_memory_usage()

            # Process based on processor type
            output_data = self._dispatch_gpu_processing(
                gather_data,
                processor_type,
                processor_config,
            )

            # Check cancellation after processing
            self._check_cancellation()

            # Get final memory
            mem_after = self.get_gpu_memory_usage()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self._items_processed += 1
            self._report_progress()

            return GPUProcessingResult(
                success=True,
                output_data=output_data,
                elapsed_ms=elapsed_ms,
                gpu_memory_used_mb=mem_after.get('used_mb', 0) - mem_before.get('used_mb', 0),
            )

        except Exception as e:
            # Check if it's a cancellation
            from ..cancellation import CancellationError
            if isinstance(e, CancellationError):
                self._state = WorkerState.CANCELLED
                raise

            logger.error(f"Worker {self.worker_id}: GPU processing failed: {e}")
            self._state = WorkerState.FAILED

            return GPUProcessingResult(
                success=False,
                error=str(e),
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
            )

    def _dispatch_gpu_processing(
        self,
        data: np.ndarray,
        processor_type: str,
        config: Dict[str, Any],
    ) -> np.ndarray:
        """
        Dispatch to the appropriate GPU kernel.

        Parameters
        ----------
        data : np.ndarray
            Input data
        processor_type : str
            Processor type name
        config : dict
            Processor configuration

        Returns
        -------
        np.ndarray
            Processed output data
        """
        # Map processor types to Metal kernel functions
        kernel_map = {
            'dwt_denoise': self._process_dwt_gpu,
            'stft_filter': self._process_stft_gpu,
            'fkk_filter': self._process_fkk_gpu,
            'agc': self._process_agc_gpu,
            'bandpass': self._process_bandpass_gpu,
        }

        kernel_func = kernel_map.get(processor_type)

        if kernel_func is None:
            # Fall back to generic processing (log only once per worker)
            if not hasattr(self, '_warned_no_kernel'):
                self._warned_no_kernel = set()
            if processor_type not in self._warned_no_kernel:
                self._warned_no_kernel.add(processor_type)
                logger.debug(
                    f"No direct GPU kernel for {processor_type}, using processor fallback"
                )
            return self._process_generic_gpu(data, processor_type, config)

        return kernel_func(data, config)

    def _process_dwt_gpu(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Process using DWT denoising on GPU."""
        if self._metal_module is None:
            return data

        try:
            if hasattr(self._metal_module, 'dwt_denoise'):
                return self._metal_module.dwt_denoise(
                    data,
                    wavelet=config.get('wavelet', 'db4'),
                    level=config.get('level', 4),
                    threshold_mode=config.get('threshold_mode', 'soft'),
                )
        except Exception as e:
            logger.warning(f"DWT GPU processing failed, using CPU fallback: {e}")

        # CPU fallback
        return data

    def _process_stft_gpu(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Process using STFT filtering on GPU."""
        if self._metal_module is None:
            return data

        try:
            if hasattr(self._metal_module, 'stft_filter'):
                return self._metal_module.stft_filter(
                    data,
                    nperseg=config.get('nperseg', 256),
                    noverlap=config.get('noverlap', 128),
                    freq_low=config.get('freq_low', 5.0),
                    freq_high=config.get('freq_high', 80.0),
                )
        except Exception as e:
            logger.warning(f"STFT GPU processing failed, using CPU fallback: {e}")

        return data

    def _process_fkk_gpu(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Process using FK filter on GPU."""
        if self._metal_module is None:
            return data

        try:
            if hasattr(self._metal_module, 'fkk_filter'):
                return self._metal_module.fkk_filter(
                    data,
                    velocity_low=config.get('velocity_low', 1500.0),
                    velocity_high=config.get('velocity_high', 6000.0),
                )
        except Exception as e:
            logger.warning(f"FKK GPU processing failed, using CPU fallback: {e}")

        return data

    def _process_agc_gpu(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Process using AGC on GPU."""
        if self._metal_module is None:
            return data

        try:
            if hasattr(self._metal_module, 'agc'):
                return self._metal_module.agc(
                    data,
                    window_ms=config.get('window_ms', 500),
                    target_rms=config.get('target_rms', 1.0),
                )
        except Exception as e:
            logger.warning(f"AGC GPU processing failed, using CPU fallback: {e}")

        return data

    def _process_bandpass_gpu(self, data: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Process using bandpass filter on GPU."""
        if self._metal_module is None:
            return data

        try:
            if hasattr(self._metal_module, 'bandpass_filter'):
                return self._metal_module.bandpass_filter(
                    data,
                    freq_low=config.get('freq_low', 5.0),
                    freq_high=config.get('freq_high', 80.0),
                    sample_rate=config.get('sample_rate', 1000.0),
                )
        except Exception as e:
            logger.warning(f"Bandpass GPU processing failed, using CPU fallback: {e}")

        return data

    def _process_generic_gpu(
        self,
        data: np.ndarray,
        processor_type: str,
        config: Dict[str, Any],
    ) -> np.ndarray:
        """Generic GPU processing fallback."""
        # For unsupported processor types, just return the input
        # The actual processing would happen on CPU through the normal processor
        logger.debug(f"Generic GPU processing for {processor_type} - passthrough")
        return data

    def process(self, *args, **kwargs) -> Any:
        """
        Main processing entry point.

        For MetalWorkerActor, this delegates to process_gather_gpu.
        """
        return self._run_with_lifecycle(
            self._process_impl,
            *args,
            **kwargs
        )

    def _process_impl(
        self,
        gathers: List[np.ndarray],
        processor_type: str,
        processor_config: Dict[str, Any],
    ) -> List[GPUProcessingResult]:
        """
        Process multiple gathers.

        Parameters
        ----------
        gathers : list
            List of gather data arrays
        processor_type : str
            Processor type
        processor_config : dict
            Processor configuration

        Returns
        -------
        list
            List of GPUProcessingResult for each gather
        """
        if not self.is_gpu_available():
            self.initialize_gpu()

        self._items_total = len(gathers)
        results = []

        for i, gather in enumerate(gathers):
            # Check cancellation
            self._check_cancellation()

            # Wait if paused
            if not self._wait_if_paused():
                break

            result = self.process_gather_gpu(
                gather,
                processor_type,
                processor_config,
            )
            results.append(result)

            self._items_processed = i + 1
            self._report_progress()

        return results

    def process(
        self,
        gather_tasks: List[Tuple[int, int, int]],
        headers_subset: Optional[Any] = None,
    ) -> Any:
        """
        Process a list of gathers (unified interface with CPUWorkerActor).

        This method provides the same interface as CPUWorkerActor.process()
        but uses GPU acceleration where possible.

        Parameters
        ----------
        gather_tasks : list
            List of (gather_idx, start_trace, end_trace) tuples
        headers_subset : DataFrame, optional
            Headers for this worker's gathers

        Returns
        -------
        WorkerResult
            Processing result
        """
        return self._run_with_lifecycle(
            self._process_gathers_gpu,
            gather_tasks,
            headers_subset,
        )

    def _process_gathers_gpu(
        self,
        gather_tasks: List[Tuple[int, int, int]],
        headers_subset: Optional[Any],
    ) -> Any:
        """GPU-accelerated gather processing with CPU fallback."""
        import zarr
        import gc
        from models.seismic_data import SeismicData
        from processors.base_processor import BaseProcessor

        # Import result type from cpu_worker
        from .cpu_worker import WorkerResult, GatherResult

        # Initialize GPU if not done
        if not self._gpu_initialized:
            self.initialize_gpu()

        # Initialize Zarr arrays if needed
        if self._input_zarr is None:
            self._input_zarr = zarr.open(self._input_zarr_path, mode='r')

            if self._output_zarr_path and self._output_mode in ('processed', 'both'):
                self._output_zarr = zarr.open(self._output_zarr_path, mode='r+')

            if self._noise_zarr_path and self._output_mode in ('noise', 'both'):
                self._noise_zarr = zarr.open(self._noise_zarr_path, mode='r+')

            # Reconstruct processor (for CPU fallback)
            self._processor = BaseProcessor.from_dict(self._processor_config)

        self._items_total = len(gather_tasks)
        self._items_processed = 0

        gather_results = []
        total_traces = 0

        output_processed = self._output_mode in ('processed', 'both')
        output_noise = self._output_mode in ('noise', 'both')
        noise_only = self._output_mode == 'noise'

        for gather_idx, start_trace, end_trace in gather_tasks:
            import time
            gather_start = time.time()
            n_traces = end_trace - start_trace + 1

            try:
                # Check cancellation
                self._check_cancellation()

                # Wait if paused
                if not self._wait_if_paused():
                    raise InterruptedError("Processing interrupted by cancellation")

                # Load gather traces
                gather_traces = np.array(
                    self._input_zarr[:, start_trace:end_trace + 1]
                )

                # Try GPU processing if kernel available
                processor_type = self._processor_config.get('class_name', '').lower()
                if self.is_gpu_available() and self._has_gpu_kernel(processor_type):
                    # Use GPU kernel
                    result = self.process_gather_gpu(
                        gather_traces,
                        processor_type,
                        self._processor_config,
                    )
                    if result.success:
                        processed_traces = result.output_data
                    else:
                        # Fall back to CPU
                        gather_data = SeismicData(
                            traces=gather_traces,
                            sample_rate=self._sample_rate,
                            metadata=self._metadata,
                        )
                        processed = self._processor.process(gather_data)
                        processed_traces = processed.traces
                else:
                    # CPU fallback
                    gather_data = SeismicData(
                        traces=gather_traces,
                        sample_rate=self._sample_rate,
                        metadata=self._metadata,
                    )
                    processed = self._processor.process(gather_data)
                    processed_traces = processed.traces

                # Write outputs
                if noise_only:
                    np.subtract(gather_traces, processed_traces, out=gather_traces)
                    if self._noise_zarr is not None:
                        self._noise_zarr[:, start_trace:end_trace + 1] = gather_traces
                else:
                    if output_processed and self._output_zarr is not None:
                        self._output_zarr[:, start_trace:end_trace + 1] = processed_traces

                    if output_noise and self._noise_zarr is not None:
                        noise_traces = gather_traces - processed_traces
                        self._noise_zarr[:, start_trace:end_trace + 1] = noise_traces

                elapsed = time.time() - gather_start
                gather_results.append(GatherResult(
                    gather_idx=gather_idx,
                    n_traces=n_traces,
                    elapsed_seconds=elapsed,
                    success=True,
                ))

                total_traces += n_traces
                self._items_processed += 1
                self._report_progress()

                if self._items_processed % 10 == 0:
                    gc.collect()

            except Exception as e:
                from ..cancellation import CancellationError
                if isinstance(e, CancellationError):
                    raise

                elapsed = time.time() - gather_start
                gather_results.append(GatherResult(
                    gather_idx=gather_idx,
                    n_traces=n_traces,
                    elapsed_seconds=elapsed,
                    success=False,
                    error=str(e),
                ))

                logger.warning(
                    f"Worker {self.worker_id}: Gather {gather_idx} failed: {e}"
                )

        total_elapsed = time.time() - self._start_time if self._start_time else 0

        return WorkerResult(
            worker_id=self.worker_id,
            n_gathers_processed=self._items_processed,
            n_traces_processed=total_traces,
            elapsed_seconds=total_elapsed,
            success=True,
            gather_results=gather_results,
        )

    def _has_gpu_kernel(self, processor_type: str) -> bool:
        """
        Check if a GPU kernel is available for the processor type.

        Note: Currently returns False for all types because the Metal
        kernels (stft_filter, dwt_denoise, etc.) have different interfaces
        than the processor classes. The CPU fallback uses the processor
        which internally uses Metal via the kernel_backend system.
        """
        # For now, always use CPU fallback which goes through the processor.
        # The processor itself uses Metal GPU via kernel_backend when available.
        # Direct GPU kernels would need interface matching which isn't implemented.
        return False

    def cleanup(self):
        """Clean up GPU resources."""
        if self._metal_module is not None:
            try:
                if hasattr(self._metal_module, 'cleanup'):
                    self._metal_module.cleanup()
            except Exception as e:
                logger.warning(f"GPU cleanup error: {e}")

        self._gpu_initialized = False
        self._metal_module = None
        logger.debug(f"Worker {self.worker_id}: GPU resources cleaned up")


def create_metal_worker_actor():
    """
    Create a Ray remote Metal worker actor class.

    Returns
    -------
    class
        Ray remote actor class for Metal GPU processing

    Notes
    -----
    Metal GPU access is handled directly by the Metal kernel, not through
    Ray's resource management. Ray does not detect Metal GPUs (only CUDA),
    so we don't request GPU resources. The Metal framework handles GPU
    scheduling internally.
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available")

    # NOTE: Do NOT request num_gpus - Ray only tracks CUDA GPUs, not Metal.
    # Metal GPU access is managed by the Metal framework, not Ray.
    # Each actor will use Metal GPU directly when processing.
    # max_concurrency=2 allows get_progress() to run while process() is executing,
    # enabling real-time progress monitoring. The internal state updates are atomic.
    @ray.remote(num_cpus=1, max_concurrency=2)
    class MetalWorkerActorRemote(MetalWorkerActor):
        """Ray remote version of MetalWorkerActor."""
        pass

    return MetalWorkerActorRemote
